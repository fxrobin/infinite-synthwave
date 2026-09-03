from __future__ import annotations

import re
import tempfile
from pathlib import Path

import yaml
from pydantic import ValidationError

from .model import AnyPatch, DrumPatchModel, PatchModel

LIBRARY = Path(__file__).parent / "library"
USER_DIR = Path.home() / ".config" / "synthwave" / "patches"

# --- sécurité : bornes et allowlist ---
_MAX_PATCH_BYTES = 256 * 1024  # 256 Ko max par fichier YAML
_ALLOWED_PATCH_SUFFIXES = {".yaml", ".yml"}
_BARE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _allowed_patch_roots() -> list[Path]:
    roots: list[Path] = []
    for p in (LIBRARY, USER_DIR, Path.cwd(), Path.home(), Path(tempfile.gettempdir())):
        try:
            # USER_DIR peut ne pas exister
            if p.exists():
                roots.append(p.resolve())
            elif p == USER_DIR:
                # autorise quand même le parent résolu pour les vérifications
                roots.append(p.resolve())
        except Exception:
            continue
    # dédup
    seen: set[Path] = set()
    uniq: list[Path] = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def _is_within_allowed_roots(resolved: Path) -> bool:
    for root in _allowed_patch_roots():
        try:
            if resolved.is_relative_to(root):
                return True
        except ValueError:
            continue
    return False


class PatchError(Exception):
    pass


def _dirs() -> list[Path]:
    return [d for d in (USER_DIR, LIBRARY) if d.is_dir()]


def list_patches() -> list[str]:
    names = {p.stem for d in _dirs() for p in d.glob("*.yaml")}
    return sorted(names)


def patch_from_dict(data: dict) -> AnyPatch:
    if not isinstance(data, dict):
        raise PatchError("patch must be a mapping")
    try:
        if data.get("kind") == "drums":
            return DrumPatchModel.model_validate(data)
        return PatchModel.model_validate(data)
    except ValidationError as e:
        raise PatchError(str(e)) from e


def _resolve_bare_patch(name: str) -> Path:
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
    if not isinstance(name_or_path, str) or not name_or_path or "\x00" in name_or_path:
        raise PatchError("invalid patch name")
    path = Path(name_or_path)
    if not path.suffix:
        path = _resolve_bare_patch(name_or_path)
    else:
        path = _resolve_explicit_patch(name_or_path, path)
    resolved_final = _verify_patch_file(path)
    try:
        text = resolved_final.read_text(encoding="utf-8")
    except OSError as e:
        raise PatchError(f"cannot read patch file {path}: {e}") from e
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise PatchError(f"invalid YAML in {path.name}: {e}") from e
    return patch_from_dict(data)


def set_param(patch: AnyPatch, path: str, value) -> AnyPatch:
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
    """Multiply numeric parameters by factors: {"filter.cutoff": 0.6, "oscillators.0.detune": 1.5}.

    Values are clamped to sane bounds; paths that do not exist in this patch are ignored, so
    one set of gestures can be applied to any patch."""
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
