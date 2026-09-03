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
    roots: list[Path] = []
    for p in (Path.cwd(), Path.home(), Path(tempfile.gettempdir())):
        try:
            if p.exists():
                roots.append(p.resolve())
        except Exception:
            continue
    seen: set[Path] = set()
    uniq: list[Path] = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def _is_within_allowed_export_roots(resolved: Path) -> bool:
    for root in _allowed_export_roots():
        try:
            if resolved.is_relative_to(root):
                return True
        except ValueError:
            continue
    return False


def _validate_export_path(path: str) -> Path:
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError(f"invalid export path {path!r}")
    raw = Path(path)
    # extension obligatoire et whitelistée
    if raw.suffix.lower() not in _ALLOWED_EXPORT_SUFFIXES:
        raise ValueError(
            f"export path must end with one of "
            f"{sorted(_ALLOWED_EXPORT_SUFFIXES)}, got {raw.suffix!r}"
        )
    try:
        expanded = raw.expanduser()
        # parent doit exister — on résout le parent, pas le fichier final (peut ne pas exister)
        parent = expanded.parent if str(expanded.parent) not in ("", ".") else Path.cwd()
        # si parent relatif, on l'ancre au cwd avant resolve
        if not parent.is_absolute():
            parent = (Path.cwd() / parent).resolve()
        else:
            parent = parent.resolve()
        resolved = (parent / expanded.name).resolve()
    except Exception as e:
        raise ValueError(f"invalid export path {path!r}: {e}") from e
    if not parent.is_dir():
        raise ValueError(f"export directory does not exist: {parent}")
    # blocklist système même si dans allowlist
    # (défense en profondeur : /home inside / mais /etc non)
    for sensitive in _SENSITIVE_EXPORT_ROOTS:
        try:
            s_res = sensitive.resolve()
            if resolved.is_relative_to(s_res) or parent.is_relative_to(s_res):
                raise ValueError(f"export path {path!r} is inside sensitive directory {sensitive}")
        except Exception:
            continue
    # bloque l'écriture dans les chemins cachés du home (ex: ~/.ssh, ~/.gnupg)
    try:
        home_res = Path.home().resolve()
        if resolved.is_relative_to(home_res):
            rel = resolved.relative_to(home_res)
            if any(part.startswith(".") for part in rel.parts):
                raise ValueError(f"export to hidden path not allowed: {path!r}")
    except ValueError as e:
        # ne pas masquer l'erreur hidden précédente
        if "hidden path" in str(e):
            raise
        pass
    allowed = _is_within_allowed_export_roots
    if not allowed(resolved) and not allowed(parent):
        raise ValueError(
            f"export path {path!r} is outside allowed directories "
            f"{[str(p) for p in _allowed_export_roots()]}"
        )
    return resolved


def export_wav(renderer: Renderer, seconds: float, path: str, blocksize: int = 1024) -> int:
    """Render offline until `seconds` reached (and, in duration mode, until the outro finishes).

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
