"""One live playback session (Renderer + Player) shared by the MCP server and the web UI."""

from __future__ import annotations

import threading

from .audio.renderer import RenderConfig, Renderer
from .composer.moods import MOODS


class Session:
    """Session."""
    def __init__(self) -> None:
        """Initialize."""
        self._lock = threading.Lock()
        self.player = None
        self.renderer: Renderer | None = None

    @property
    def live(self) -> Renderer | None:
        """Live."""
        if self.player is None or not self.player.running:
            return None
        return self.renderer

    def start(self, cfg: RenderConfig, blocksize: int = 1024, device=None) -> dict:
        """Start."""
        from .audio.output import Player

        with self._lock:
            if self.player is not None and self.player.running:
                return {"ok": False, "error": "already running; call stop first"}
            try:
                self.renderer = Renderer(cfg)
                self.player = Player(self.renderer, blocksize=blocksize, device=device)
                self.player.start()
            except Exception as e:
                self.player = None
                return {"ok": False, "error": str(e)}
        return {"ok": True, "status": self.renderer.status()}

    def stop(self) -> dict:
        """Stop."""
        with self._lock:
            if self.player is None:
                return {"ok": False, "error": "not running"}
            self.player.stop()
            self.player = None
        return {"ok": True}

    def status(self) -> dict:
        """Status."""
        r = self.live
        if r is None:
            return {"running": False, "moods": list(MOODS)}
        return {"running": True, "underruns": self.player.underruns, **r.status()}

    def command(self, fn, timeout: float = 2.0) -> dict:
        """Run `fn(renderer)` on the audio thread and wait for it (thread-safe)."""
        r = self.live
        if r is None:
            return {"ok": False, "error": "player not running; call start first"}
        done = threading.Event()
        box: dict = {}

        def run():
            """Run."""
            try:
                fn(r)
            except Exception as e:
                box["error"] = str(e)
            finally:
                done.set()

        r.submit(run)
        done.wait(timeout)
        if "error" in box:
            return {"ok": False, "error": box["error"]}
        return {"ok": True, "status": r.status()}


SESSION = Session()
