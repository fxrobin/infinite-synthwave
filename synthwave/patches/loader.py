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
