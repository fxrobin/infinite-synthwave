from __future__ import annotations

import soundfile as sf

from .renderer import Renderer


def export_wav(renderer: Renderer, seconds: float, path: str, blocksize: int = 1024) -> int:
    """Render offline until `seconds` reached (and, in duration mode, until the outro finishes).

    Format follows the extension: .wav (PCM 16 bit), .flac, .ogg, .mp3.
    """
    ext = path.lower().rsplit(".", 1)[-1]
    subtype = {"mp3": "MPEG_LAYER_III", "ogg": "VORBIS", "flac": "PCM_16"}.get(ext, "PCM_16")
    target = int(seconds * renderer.sr)
    limit = target + 60 * renderer.sr
    written = 0
    with sf.SoundFile(path, "w", samplerate=renderer.sr, channels=2, subtype=subtype) as f:
        while written < limit:
            if written >= target and (renderer.finished or not renderer.cfg.duration_s):
                break
            block = renderer.render(blocksize)
            f.write(block)
            written += len(block)
    return written
