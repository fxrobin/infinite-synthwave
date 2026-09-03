from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from .model import AnyPatch, DrumPatchModel, PatchModel

LIBRARY = Path(__file__).parent / "library"
USER_DIR = Path.home() / ".config" / "synthwave" / "patches"


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


def load_patch(name_or_path: str) -> AnyPatch:
    path = Path(name_or_path)
    if not path.suffix:
        for d in _dirs():
            cand = d / f"{name_or_path}.yaml"
            if cand.exists():
                path = cand
                break
        else:
            raise PatchError(f"patch {name_or_path!r} not found in {[str(d) for d in _dirs()]}")
    if not path.exists():
        raise PatchError(f"patch file {path} not found")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise PatchError(f"invalid YAML in {path}: {e}") from e
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


_BOUNDS = {"cutoff": (20.0, 20000.0), "resonance": (0.0, 1.0), "detune": (0.0, 100.0),
           "pwm": (0.05, 0.95), "level": (0.0, 2.0), "volume": (0.0, 2.0), "sustain": (0.0, 1.0),
           "fm_index": (0.0, 10.0), "fm_ratio": (0.1, 16.0), "spread": (0.0, 1.0),
           "mix": (0.0, 1.0), "feedback": (0.0, 0.9), "drive": (1.0, 12.0), "depth": (0.0, 1.0),
           "rate": (0.01, 40.0), "glide": (0.0, 1.0), "amount": (-20000.0, 20000.0)}


def apply_tweaks(patch: AnyPatch, tweaks: dict[str, float]) -> AnyPatch:
    """Multiply numeric parameters by factors: {"filter.cutoff": 0.6, "oscillators.0.detune": 1.5}.

    Values are clamped to sane bounds; paths that do not exist in this patch are ignored, so
    one set of gestures can be applied to any patch."""
    if not tweaks:
        return patch
    data = patch.model_dump()
    for path, factor in tweaks.items():
        keys = path.split(".")
        node = data
        try:
            for k in keys[:-1]:
                node = node[int(k)] if isinstance(node, list) else node[k]
            last = keys[-1]
            cur = node[int(last)] if isinstance(node, list) else node[last]
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        if not isinstance(cur, (int, float)) or isinstance(cur, bool):
            continue
        lo, hi = _BOUNDS.get(last, (-1e9, 1e9))
        val = float(min(hi, max(lo, cur * float(factor))))
        if isinstance(node, list):
            node[int(last)] = val
        else:
            node[last] = val
    try:
        return patch_from_dict(data)
    except PatchError:
        return patch
