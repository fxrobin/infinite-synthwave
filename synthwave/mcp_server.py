"""MCP server (stdio) piloting a live Player. Run with: synthwave mcp"""
from __future__ import annotations

import threading

from mcp.server.mcpserver import MCPServer

from .audio.renderer import RenderConfig, Renderer
from .composer.moods import MOODS
from .patches import loader

mcp = MCPServer("infinite-synthwave")
_lock = threading.Lock()
_player = None
_renderer: Renderer | None = None


def _live() -> Renderer | None:
    if _player is None or not _player.running:
        return None
    return _renderer


def _command(fn) -> dict:
    r = _live()
    if r is None:
        return {"ok": False, "error": "player not running; call start first"}
    done = threading.Event()
    box: dict = {}

    def run():
        try:
            fn(r)
        except Exception as e:
            box["error"] = str(e)
        finally:
            done.set()

    r.submit(run)
    done.wait(2.0)
    if "error" in box:
        return {"ok": False, "error": box["error"]}
    return {"ok": True, "status": r.status()}


@mcp.tool()
def start(mood: str | None = None, bpm: float | None = None, seed: int | None = None,
          duration_s: float | None = None, bpm_range: list[float] | None = None) -> dict:
    """Start synthwave on the audio output (infinite unless duration_s given).

    mood None: drawn at random and redrawn at every transition; a given mood is kept.
    bpm fixes the tempo; bpm_range=[low, high] bounds the random tempo drawn at start
    and at each transition (default: the mood's own range)."""
    global _player, _renderer
    from .audio.output import Player
    with _lock:
        if _player is not None and _player.running:
            return {"ok": False, "error": "already running; call stop first"}
        try:
            rng = (float(bpm_range[0]), float(bpm_range[1])) if bpm_range else None
            _renderer = Renderer(RenderConfig(mood=mood, bpm=bpm, seed=seed,
                                              duration_s=duration_s, bpm_range=rng))
            _player = Player(_renderer)
            _player.start()
        except Exception as e:
            _player = None
            return {"ok": False, "error": str(e)}
    return {"ok": True, "status": _renderer.status()}


@mcp.tool()
def stop() -> dict:
    """Stop playback."""
    global _player
    with _lock:
        if _player is None:
            return {"ok": False, "error": "not running"}
        _player.stop()
        _player = None
    return {"ok": True}


@mcp.tool()
def status() -> dict:
    """Current transport, key, chord, section and layer state."""
    if _live() is None:
        return {"running": False, "moods": list(MOODS)}
    return {"running": True, "underruns": _player.underruns, **_renderer.status()}


@mcp.tool()
def set_tempo(bpm: float) -> dict:
    """Change tempo (60-180 BPM)."""
    return _command(lambda r: r.set_tempo(bpm))


@mcp.tool()
def set_mood(mood: str) -> dict:
    """Change mood at the next transition and keep it: dark|noir|dreamy|outrun.
    'random' releases the lock so every transition draws a new mood."""
    return _command(lambda r: r.set_mood(mood))


@mcp.tool()
def set_layer(layer: str, mute: bool | None = None, solo: bool | None = None,
              volume: float | None = None) -> dict:
    """Mute/solo/volume (0-2) for a layer: drums|bass|arp|pad|lead|ambient."""
    return _command(lambda r: r.set_layer(layer, mute=mute, solo=solo, volume=volume))


@mcp.tool()
def list_patches() -> dict:
    """List available synth/drum patches (library + ~/.config/synthwave/patches)."""
    return {"patches": loader.list_patches()}


@mcp.tool()
def load_patch(layer: str, name: str) -> dict:
    """Load a patch (library name or YAML path) into a layer."""
    return _command(lambda r: r.load_patch(layer, name))


@mcp.tool()
def set_patch_param(layer: str, path: str, value: float | str) -> dict:
    """Set one patch parameter, e.g. path='filter.cutoff' value=800, 'oscillators.0.detune' 20."""
    return _command(lambda r: r.set_patch_param(layer, path, value))


@mcp.tool()
def set_layer_effects(layer: str, effects: list[dict] | None = None) -> dict:
    """Manual insert effects for a layer or 'master', e.g. [{"type":"gate","rate":"1/16"}],
    [{"type":"lofi","bits":8}], [{"type":"bitcrush","bits":6,"downsample":4}]. None = auto."""
    return _command(lambda r: r.set_layer_effects(layer, effects))


@mcp.tool()
def next_section() -> dict:
    """Jump to the next section at the next bar."""
    return _command(lambda r: r.next_section())


@mcp.tool()
def export_wav(path: str, seconds: float, mood: str = "dark", bpm: float | None = None,
               seed: int | None = None) -> dict:
    """Render a standalone track offline to a WAV file (does not disturb live playback)."""
    from .audio.export import export_wav as _export
    try:
        r = Renderer(RenderConfig(mood=mood, bpm=bpm, seed=seed, duration_s=seconds))
        n = _export(r, seconds, path)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "path": path, "seconds": round(n / r.sr, 2), "seed": r.seed}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
