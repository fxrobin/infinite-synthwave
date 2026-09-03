from __future__ import annotations

import tempfile
from pathlib import Path

import soundfile as sf

from .renderer import Renderer

_ALLOWED_EXPORT_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}
_MAX_EXPORT_SECONDS = 4 * 3600  # 4h max (DoS disque)
_MAX_BLOCKSIZE = 8192
_SENSITIVE_EXPORT_ROOTS = [
    Path("/etc"),
    Path("/usr"),
    Path("/bin"),
    Path("/sbin"),
    Path("/boot"),
    Path("/root"),
    Path("/proc"),
    Path("/sys"),
    Path("/dev"),
]


def _allowed_export_roots() -> list[Path]:
    """Allowed export roots."""
    import contextlib

    roots: list[Path] = []
    for p in (Path.cwd(), Path.home(), Path(tempfile.gettempdir())):
        with contextlib.suppress(Exception):
            if p.exists():
                roots.append(p.resolve())
    seen: set[Path] = set()
    uniq: list[Path] = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def _is_within_allowed_export_roots(resolved: Path) -> bool:
    """Is within allowed export roots."""
    for root in _allowed_export_roots():
        try:
            if resolved.is_relative_to(root):
                return True
        except ValueError:
            pass
    return False


def _check_export_extension(raw: Path, path: str) -> None:
    """Check export extension."""
    if raw.suffix.lower() not in _ALLOWED_EXPORT_SUFFIXES:
        raise ValueError(
            f"export path must end with one of {sorted(_ALLOWED_EXPORT_SUFFIXES)}, "
            f"got {raw.suffix!r}"
        )


def _resolve_export_paths(raw: Path) -> tuple[Path, Path]:
    """Resolve export paths."""
    try:
        expanded = raw.expanduser()
        parent = expanded.parent if str(expanded.parent) not in ("", ".") else Path.cwd()
        if not parent.is_absolute():
            parent = (Path.cwd() / parent).resolve()
        else:
            parent = parent.resolve()
        resolved = (parent / expanded.name).resolve()
    except Exception as e:
        raise ValueError(f"invalid export path {raw}: {e}") from e
    if not parent.is_dir():
        raise ValueError(f"export directory does not exist: {parent}")
    return parent, resolved


def _check_export_sensitive(parent: Path, resolved: Path, path: str) -> None:
    """Check export sensitive."""
    for sensitive in _SENSITIVE_EXPORT_ROOTS:
        try:
            s_res = sensitive.resolve()
            if resolved.is_relative_to(s_res) or parent.is_relative_to(s_res):
                raise ValueError(f"export path {path!r} is inside sensitive directory {sensitive}")
        except ValueError as e:
            if "sensitive directory" in str(e):
                raise
        except Exception:
            pass


def _check_export_hidden(resolved: Path, path: str) -> None:
    """Check export hidden."""
    try:
        home_res = Path.home().resolve()
        if resolved.is_relative_to(home_res):
            rel = resolved.relative_to(home_res)
            if any(part.startswith(".") for part in rel.parts):
                raise ValueError(f"export to hidden path not allowed: {path!r}")
    except ValueError as e:
        if "hidden path" in str(e):
            raise


def _validate_export_path(path: str) -> Path:
    """Validate export path."""
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError(f"invalid export path {path!r}")
    raw = Path(path)
    _check_export_extension(raw, path)
    parent, resolved = _resolve_export_paths(raw)
    _check_export_sensitive(parent, resolved, path)
    _check_export_hidden(resolved, path)
    allowed = _is_within_allowed_export_roots
    if not allowed(resolved) and not allowed(parent):
        raise ValueError(
            f"export path {path!r} is outside allowed directories "
            f"{[str(p) for p in _allowed_export_roots()]}"
        )
    return resolved


def export_wav(renderer: Renderer, seconds: float, path: str, blocksize: int = 1024) -> int:
    """Render offline until `seconds` reached (and, in duration mode, until the
    outro finishes).

    Format follows the extension: .wav (PCM 16 bit), .flac, .ogg, .mp3.
    """
    # --- validation ---
    try:
        seconds_f = float(seconds)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid seconds {seconds!r}: {e}") from e
    if not 0 < seconds_f <= _MAX_EXPORT_SECONDS:
        raise ValueError(f"seconds must be in (0, {_MAX_EXPORT_SECONDS}], got {seconds_f}")
    if not 64 <= int(blocksize) <= _MAX_BLOCKSIZE:
        raise ValueError(f"blocksize must be in [64, {_MAX_BLOCKSIZE}], got {blocksize}")
    validated_path = _validate_export_path(path)
    ext = validated_path.suffix.lower().lstrip(".")
    subtype = {"mp3": "MPEG_LAYER_III", "ogg": "VORBIS", "flac": "PCM_16"}.get(ext, "PCM_16")
    target = int(seconds_f * renderer.sr)
    limit = target + 60 * renderer.sr
    written = 0
    with sf.SoundFile(
        str(validated_path), "w", samplerate=renderer.sr, channels=2, subtype=subtype
    ) as f:
        while written < limit:
            if written >= target and (renderer.finished or not renderer.cfg.duration_s):
                break
            block = renderer.render(int(blocksize))
            f.write(block)
            written += len(block)
    return written
