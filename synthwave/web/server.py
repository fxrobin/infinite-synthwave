"""Local web UI: Starlette app serving one page, a JSON API and a WebSocket status feed.

Run with: synthwave ui  (http://127.0.0.1:8765)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from ..audio.renderer import LAYERS, RenderConfig
from ..composer.moods import MOODS
from ..engine.effects import _REGISTRY
from ..patches import loader
from ..session import SESSION

STATIC = Path(__file__).parent / "static"
FX_DEFAULTS = {
    "chorus": {"rate": 0.5, "depth": 0.003, "mix": 0.4},
    "delay": {"time": "1/8", "feedback": 0.4, "mix": 0.3, "pingpong": True},
    "reverb": {"size": 0.8, "damping": 0.5, "mix": 0.3},
    "gate": {"rate": "1/16", "depth": 0.8, "duty": 0.5},
    "bitcrush": {"bits": 8, "downsample": 4, "mix": 0.8},
    "lofi": {"bits": 10, "downsample": 3, "cutoff": 5000, "wobble": 0.002, "noise": 0.004},
    "distortion": {"drive": 4.0, "tone": 4000, "mix": 0.8},
    "autopan": {"rate": "1/2", "depth": 0.8},
    "phaser": {"rate": 0.3, "depth": 0.8, "stages": 4, "mix": 0.5},
    "flanger": {"rate": 0.25, "depth": 0.002, "base": 0.003, "feedback": 0.5, "mix": 0.5},
}


async def index(_: Request) -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


async def meta(_: Request) -> JSONResponse:
    return JSONResponse({
        "moods": {name: {"bpm_range": list(m.bpm_range), "scale": m.scale,
                         "halftime": m.halftime} for name, m in MOODS.items()},
        "patches": loader.list_patches(),
        "layers": list(LAYERS),
        "effects": {k: FX_DEFAULTS.get(k, {}) for k in _REGISTRY if k != "limiter"},
    })


async def status(_: Request) -> JSONResponse:
    return JSONResponse(SESSION.status())


async def _body(request: Request) -> dict:
    try:
        data = await request.json()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


async def start(request: Request) -> JSONResponse:
    b = await _body(request)
    try:
        cfg = RenderConfig(
            mood=b.get("mood") or None,
            bpm=float(b["bpm"]) if b.get("bpm") else None,
            seed=int(b["seed"]) if b.get("seed") not in (None, "") else None,
            track_s=float(b.get("track_s") or 210.0),
            bpm_range=tuple(float(v) for v in b["bpm_range"]) if b.get("bpm_range") else None,
        )
    except (TypeError, ValueError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    res = await asyncio.to_thread(SESSION.start, cfg, int(b.get("blocksize") or 1024),
                                  b.get("device") or None)
    return JSONResponse(res, status_code=200 if res["ok"] else 409)


async def stop(_: Request) -> JSONResponse:
    return JSONResponse(await asyncio.to_thread(SESSION.stop))


def _cmd(fn) -> JSONResponse:
    res = SESSION.command(fn)
    return JSONResponse(res, status_code=200 if res["ok"] else 409)


async def tempo(request: Request) -> JSONResponse:
    b = await _body(request)
    return _cmd(lambda r: r.set_tempo(float(b["bpm"])))


async def mood(request: Request) -> JSONResponse:
    b = await _body(request)
    return _cmd(lambda r: r.set_mood(str(b.get("mood", "random"))))


async def layer(request: Request) -> JSONResponse:
    b = await _body(request)
    return _cmd(lambda r: r.set_layer(str(b["layer"]), mute=b.get("mute"), solo=b.get("solo"),
                                      volume=b.get("volume")))


async def patch(request: Request) -> JSONResponse:
    b = await _body(request)
    return _cmd(lambda r: r.load_patch(str(b["layer"]), str(b["name"])))


async def patch_param(request: Request) -> JSONResponse:
    b = await _body(request)
    return _cmd(lambda r: r.set_patch_param(str(b["layer"]), str(b["path"]), b.get("value")))


async def patch_state(request: Request) -> JSONResponse:
    r = SESSION.live
    name = request.path_params["layer"]
    if r is None or name not in r.instruments or name == "riser":
        return JSONResponse({"ok": False, "error": "no patch"}, status_code=404)
    return JSONResponse({"ok": True, "layer": name, "name": r.patch_names[name],
                         "patch": r.instruments[name].patch.model_dump()})


async def effects(request: Request) -> JSONResponse:
    b = await _body(request)
    specs = b.get("effects")
    return _cmd(lambda r: r.set_layer_effects(str(b["layer"]),
                                              list(specs) if specs is not None else None))


async def auto_tweaks(request: Request) -> JSONResponse:
    b = await _body(request)
    return _cmd(lambda r: r.set_auto_tweaks(bool(b.get("enabled", True))))


async def next_section(_: Request) -> JSONResponse:
    return _cmd(lambda r: r.next_section())


async def feed(ws: WebSocket) -> None:
    """Push status + meters + oscilloscope ~12 times per second."""
    await ws.accept()
    try:
        while True:
            st = SESSION.status()
            r = SESSION.live
            if r is not None:
                st["scope"] = [round(float(v), 3) for v in r.scope]
            await ws.send_text(json.dumps(st))
            await asyncio.sleep(1 / 12)
    except (WebSocketDisconnect, RuntimeError, ConnectionError):
        return


app = Starlette(routes=[
    Route("/", index),
    Route("/api/meta", meta),
    Route("/api/status", status),
    Route("/api/start", start, methods=["POST"]),
    Route("/api/stop", stop, methods=["POST"]),
    Route("/api/tempo", tempo, methods=["POST"]),
    Route("/api/mood", mood, methods=["POST"]),
    Route("/api/layer", layer, methods=["POST"]),
    Route("/api/patch", patch, methods=["POST"]),
    Route("/api/patch/{layer}", patch_state),
    Route("/api/patch_param", patch_param, methods=["POST"]),
    Route("/api/effects", effects, methods=["POST"]),
    Route("/api/next_section", next_section, methods=["POST"]),
    Route("/api/auto_tweaks", auto_tweaks, methods=["POST"]),
    WebSocketRoute("/ws", feed),
])


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
