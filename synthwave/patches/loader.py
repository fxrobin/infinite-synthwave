from __future__ import annotations

import re
import tempfile
from pathlib import Path

import yaml
from pydantic import ValidationError

try:  # libyaml : ~10x plus rapide que le parseur Python pur
    from yaml import CSafeLoader as _YamlLoader
except ImportError:  # pragma: no cover - dépend du build de PyYAML
    from yaml import SafeLoader as _YamlLoader

from .model import (
    AnyPatch,
    D50PatchModel,
    DrumPatchModel,
    Dx7PatchModel,
    PatchModel,
    SolinaPatchModel,
)

LIBRARY = Path(__file__).parent / "library"
USER_DIR = Path.home() / ".config" / "synthwave" / "patches"

# --- sécurité : bornes et allowlist ---
_MAX_PATCH_BYTES = 256 * 1024  # 256 Ko max par fichier YAML
_ALLOWED_PATCH_SUFFIXES = {".yaml", ".yml"}
_BARE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _allowed_patch_roots() -> list[Path]:
    """Allowed patch roots."""
    import contextlib

    roots: list[Path] = []
    for p in (LIBRARY, USER_DIR, Path.cwd(), Path.home(), Path(tempfile.gettempdir())):
        with contextlib.suppress(Exception):
            # USER_DIR peut ne pas exister
            if p.exists():
                roots.append(p.resolve())
            elif p == USER_DIR:
                # autorise quand même le parent résolu pour les vérifications
                roots.append(p.resolve())
    # dédup
    seen: set[Path] = set()
    uniq: list[Path] = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def _is_within_allowed_roots(resolved: Path) -> bool:
    """Is within allowed roots."""
    for root in _allowed_patch_roots():
        try:
            if resolved.is_relative_to(root):
                return True
        except ValueError:
            pass
    return False


class PatchError(Exception):
    """Patcherror."""


# Le parse YAML d'un patch coûte de 1 à 20 ms (un D-50 est volumineux) et
# `Renderer._apply_plan_mood` recharge des patches depuis le thread audio à chaque
# changement de section : sans cache, ça dépasse le budget d'un bloc et provoque un
# under-run. On mémorise le YAML décodé par (chemin, mtime, taille) — éditer un fichier
# de patch pendant la lecture reste donc pris en compte — et on reconstruit un modèle
# neuf à chaque appel (0,04 ms), pour qu'aucun appelant ne partage d'objet muté.
_YAML_CACHE: dict[Path, tuple[int, int, object]] = {}
_YAML_CACHE_MAX = 512


def _load_yaml_cached(path: Path) -> object:
    """Parsed YAML for *path*, reused while the file's mtime and size are unchanged."""
    try:
        st = path.stat()
    except OSError as e:
        raise PatchError(f"cannot read patch file {path}: {e}") from e
    hit = _YAML_CACHE.get(path)
    if hit is not None and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
        return hit[2]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise PatchError(f"cannot read patch file {path}: {e}") from e
    try:
        data = yaml.load(text, Loader=_YamlLoader)
    except yaml.YAMLError as e:
        raise PatchError(f"invalid YAML in {path.name}: {e}") from e
    if len(_YAML_CACHE) >= _YAML_CACHE_MAX:
        _YAML_CACHE.clear()
    _YAML_CACHE[path] = (st.st_mtime_ns, st.st_size, data)
    return data


def _dirs() -> list[Path]:
    """Dirs."""
    return [d for d in (USER_DIR, LIBRARY) if d.is_dir()]


def list_patches() -> list[str]:
    """List patches."""
    names = {p.stem for d in _dirs() for p in d.glob("*.yaml")}
    return sorted(names)


def patch_from_dict(data: dict) -> AnyPatch:
    """Patch from dict."""
    if not isinstance(data, dict):
        raise PatchError("patch must be a mapping")
    try:
        if data.get("kind") == "drums":
            return DrumPatchModel.model_validate(data)
        if data.get("kind") == "dx7":
            return Dx7PatchModel.model_validate(data)
        if data.get("kind") == "solina":
            return SolinaPatchModel.model_validate(data)
        if data.get("kind") == "d50":
            return D50PatchModel.model_validate(data)
        return PatchModel.model_validate(data)
    except ValidationError as e:
        raise PatchError(str(e)) from e


def _resolve_bare_patch(name: str) -> Path:
    """Resolve bare patch."""
    if not _BARE_NAME_RE.match(name):
        raise PatchError(
            f"invalid patch name {name!r}: bare names must match {_BARE_NAME_RE.pattern}"
        )
    for d in _dirs():
        cand = d / f"{name}.yaml"
        if cand.exists():
            return cand
    raise PatchError(f"patch {name!r} not found in {[str(d) for d in _dirs()]}")


def _resolve_explicit_patch(name: str, path: Path) -> Path:
    """Resolve explicit patch."""
    if path.suffix.lower() not in _ALLOWED_PATCH_SUFFIXES:
        raise PatchError(f"patch file must end with .yaml/.yml, got {path.suffix!r}")
    try:
        resolved = path.expanduser().resolve()
    except Exception as e:
        raise PatchError(f"invalid patch path {name!r}: {e}") from e
    if not _is_within_allowed_roots(resolved):
        raise PatchError(
            f"patch path {name!r} is outside allowed directories "
            f"{[str(p) for p in _allowed_patch_roots()]}"
        )
    try:
        resolved.relative_to(resolved.anchor)
    except Exception:
        raise PatchError(f"invalid patch path {name!r}") from None
    return resolved


def _verify_patch_file(path: Path) -> Path:
    """Verify patch file."""
    try:
        resolved_final = path.resolve()
    except Exception as e:
        raise PatchError(f"invalid patch path {path}: {e}") from e
    if not resolved_final.is_file():
        raise PatchError(f"patch file {path} not found")
    try:
        size = resolved_final.stat().st_size
    except OSError as e:
        raise PatchError(f"cannot stat patch file {path}: {e}") from e
    if size > _MAX_PATCH_BYTES:
        raise PatchError(f"patch file {path} too large ({size} > {_MAX_PATCH_BYTES} bytes)")
    return resolved_final


def load_patch(name_or_path: str) -> AnyPatch:
    """Load patch."""
    if not isinstance(name_or_path, str) or not name_or_path or "\x00" in name_or_path:
        raise PatchError("invalid patch name")
    path = Path(name_or_path)
    if not path.suffix:
        path = _resolve_bare_patch(name_or_path)
    else:
        path = _resolve_explicit_patch(name_or_path, path)
    resolved_final = _verify_patch_file(path)
    return patch_from_dict(_load_yaml_cached(resolved_final))


def set_param(patch: AnyPatch, path: str, value) -> AnyPatch:
    """Set param."""
    data = patch.model_dump()
    keys = path.split(".")
    node = data
    try:
        for k in keys[:-1]:
            node = node[int(k)] if isinstance(node, list) else node[k]
        last = keys[-1]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            if last not in node:
                raise KeyError(last)
            node[last] = value
    except (KeyError, IndexError, ValueError, TypeError) as e:
        raise PatchError(f"bad parameter path {path!r}: {e}") from e
    return patch_from_dict(data)


_BOUNDS = {
    "cutoff": (20.0, 20000.0),
    "resonance": (0.0, 1.0),
    "detune": (0.0, 100.0),
    "pwm": (0.05, 0.95),
    "level": (0.0, 2.0),
    "volume": (0.0, 2.0),
    "sustain": (0.0, 1.0),
    "fm_index": (0.0, 10.0),
    "fm_ratio": (0.1, 16.0),
    "spread": (0.0, 1.0),
    "mix": (0.0, 1.0),
    "feedback": (0.0, 0.9),
    "drive": (1.0, 12.0),
    "depth": (0.0, 1.0),
    "rate": (0.01, 40.0),
    "glide": (0.0, 1.0),
    "amount": (-20000.0, 20000.0),
}


def _get_nested(data: dict, path: str):
    """Get nested."""
    keys = path.split(".")
    node = data
    cur = None
    try:
        for k in keys[:-1]:
            node = node[int(k)] if isinstance(node, list) else node[k]
        last = keys[-1]
        cur = node[int(last)] if isinstance(node, list) else node[last]
    except (KeyError, IndexError, ValueError, TypeError):
        return None, None, None
    return node, last, cur


def apply_tweaks(patch: AnyPatch, tweaks: dict[str, float]) -> AnyPatch:
    """
    Multiply numeric parameters by factors: {"filter.cutoff": 0.6,
    "oscillators.0.detune": 1.5}.

    Values are clamped to sane bounds; paths that do not exist in this patch are
    ignored, so one set of gestures can be applied to any patch.
    """
    if not tweaks:
        return patch
    data = patch.model_dump()
    for path, factor in tweaks.items():
        node, last, cur = _get_nested(data, path)
        if cur is None or node is None:
            continue
        if not isinstance(cur, (int, float)) or isinstance(cur, bool):
            continue
        lo, hi = _BOUNDS.get(last, (-1e9, 1e9))
        try:
            val = float(min(hi, max(lo, cur * float(factor))))
        except (ValueError, TypeError):
            continue
        if isinstance(node, list):
            node[int(last)] = val
        else:
            node[last] = val
    try:
        return patch_from_dict(data)
    except PatchError:
        return patch
