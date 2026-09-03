"""Local web UI: Starlette app serving one page, a JSON API and a WebSocket
status feed.

Run with: synthwave ui  (http://127.0.0.1:8765)
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from ..audio.renderer import LAYERS, RenderConfig
from ..composer.moods import MOODS
from ..engine.effects import _REGISTRY
from ..patches import loader
from ..session import SESSION

# --- sécurité web : constantes ---
_MAX_BODY_BYTES = 16 * 1024
_MAX_STR_LEN = 64
_ALLOWED_MOODS = set(MOODS) | {"random"}
_ALLOWED_LAYERS = set(LAYERS)
_ALLOWED_LAYERS_MASTER = set(LAYERS) | {"master"}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Securityheadersmiddleware."""

    async def dispatch(self, request, call_next):
        # limite taille requête
        """Dispatch."""
        clen = request.headers.get("content-length")
        if clen and clen.isdigit() and int(clen) > _MAX_BODY_BYTES:
            return JSONResponse({"ok": False, "error": "request too large"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; connect-src 'self' ws: wss:; "
            "img-src 'self' data:; object-src 'none'"
        )
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


def _validate_str(value, max_len=_MAX_STR_LEN, allow_empty=False) -> str:
    """Validate str."""
    if not isinstance(value, str):
        raise ValueError("expected string")
    if not allow_empty and not value:
        raise ValueError("empty string")
    if len(value) > max_len:
        raise ValueError("string too long")
    if "\x00" in value:
        raise ValueError("null byte")
    return value


def _validate_mood(name: str) -> None:
    """Validate mood."""
    if name not in _ALLOWED_MOODS:
        raise ValueError(f"unknown mood {name!r}")


def _validate_layer(name: str, allow_master: bool = False) -> None:
    """Validate layer."""
    allowed = _ALLOWED_LAYERS_MASTER if allow_master else _ALLOWED_LAYERS
    if name not in allowed:
        raise ValueError(f"unknown layer {name!r}")


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
    """Index."""
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


async def meta(_: Request) -> JSONResponse:
    """Meta."""
    return JSONResponse(
        {
            "moods": {
                name: {"bpm_range": list(m.bpm_range), "scale": m.scale, "halftime": m.halftime}
                for name, m in MOODS.items()
            },
            "patches": loader.list_patches(),
            "layers": list(LAYERS),
            "effects": {k: FX_DEFAULTS.get(k, {}) for k in _REGISTRY if k != "limiter"},
        }
    )


async def status(_: Request) -> JSONResponse:
    """Status."""
    return JSONResponse(SESSION.status())


async def _body(request: Request) -> dict:
    # limite taille brute avant JSON parsing
    """Body."""
    body = await request.body()
    if len(body) > _MAX_BODY_BYTES:
        return {}
    if not body:
        return {}
    try:
        data = json.loads(body)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


async def start(request: Request) -> JSONResponse:  # noqa: C901 - request validation branches
    """Start."""
    b = await _body(request)
    try:
        # --- validation stricte ---
        mood = b.get("mood")
        if mood not in (None, ""):
            _validate_str(str(mood))
            _validate_mood(str(mood))
            mood = str(mood)
        else:
            mood = None
        bpm = None
        if b.get("bpm") not in (None, ""):
            bpm = float(b["bpm"])
            if not 40 <= bpm <= 200:
                raise ValueError("bpm out of range 40-200")
        seed = None
        if b.get("seed") not in (None, ""):
            seed = int(b["seed"])
            if not -(2**31) <= seed <= 2**31 - 1:
                raise ValueError("seed out of range")
        track_s = float(b.get("track_s") or 210.0)
        if not 30 <= track_s <= 900:
            raise ValueError("track_s out of range 30-900")
        bpm_range = None
        if b.get("bpm_range"):
            br = b["bpm_range"]
            if not isinstance(br, (list, tuple)) or len(br) != 2:
                raise ValueError("bpm_range must be [low, high]")
            bpm_range = (float(br[0]), float(br[1]))
            if not 40 <= bpm_range[0] <= bpm_range[1] <= 200:
                raise ValueError("bpm_range out of range 40-200")
        blocksize = int(b.get("blocksize") or 1024)
        if not 64 <= blocksize <= 8192:
            raise ValueError("blocksize out of range 64-8192")
        device = b.get("device")
        if device not in (None, ""):
            _validate_str(str(device), max_len=128)
            device = str(device)
        else:
            device = None
        cfg = RenderConfig(mood=mood, bpm=bpm, seed=seed, track_s=track_s, bpm_range=bpm_range)
    except (TypeError, ValueError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    res = await asyncio.to_thread(SESSION.start, cfg, blocksize, device)
    return JSONResponse(res, status_code=200 if res["ok"] else 409)


async def stop(_: Request) -> JSONResponse:
    """Stop."""
    return JSONResponse(await asyncio.to_thread(SESSION.stop))


def _cmd(fn) -> JSONResponse:
    """Cmd."""
    res = SESSION.command(fn)
    return JSONResponse(res, status_code=200 if res["ok"] else 409)


async def tempo(request: Request) -> JSONResponse:
    """Tempo."""
    b = await _body(request)
    try:
        bpm = float(b["bpm"])
        if not 40 <= bpm <= 200:
            raise ValueError("bpm out of range")
    except (TypeError, ValueError, KeyError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return _cmd(lambda r: r.set_tempo(bpm))


async def mood(request: Request) -> JSONResponse:
    """Mood."""
    b = await _body(request)
    try:
        name = str(b.get("mood", "random"))
        _validate_str(name)
        _validate_mood(name)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return _cmd(lambda r: r.set_mood(name))


async def layer(request: Request) -> JSONResponse:
    """Layer."""
    b = await _body(request)
    try:
        name = str(b["layer"])
        _validate_str(name)
        _validate_layer(name)
        # validation volume
        vol = b.get("volume")
        if vol is not None:
            vol = float(vol)
            if not 0 <= vol <= 2:
                raise ValueError("volume out of range 0-2")
        # mute/solo bool
        for k in ("mute", "solo"):
            if b.get(k) is not None and not isinstance(b[k], bool):
                raise ValueError(f"{k} must be bool")
    except (ValueError, KeyError, TypeError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return _cmd(lambda r: r.set_layer(name, mute=b.get("mute"), solo=b.get("solo"), volume=vol))


async def patch(request: Request) -> JSONResponse:
    """Patch."""
    b = await _body(request)
    try:
        lyr = _validate_str(str(b["layer"]))
        _validate_layer(lyr)
        name = _validate_str(str(b["name"]), max_len=256)
    except (ValueError, KeyError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return _cmd(lambda r: r.load_patch(lyr, name))


async def patch_param(request: Request) -> JSONResponse:
    """Patch param."""
    b = await _body(request)
    try:
        lyr = _validate_str(str(b["layer"]))
        _validate_layer(lyr)
        path = _validate_str(str(b["path"]), max_len=128)
        val = b.get("value")
        # limite taille/valeur
        if isinstance(val, str) and len(val) > 64:
            raise ValueError("value too long")
        if isinstance(val, (int, float)) and not -1e6 <= float(val) <= 1e6:
            raise ValueError("value out of range")
    except (ValueError, KeyError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return _cmd(lambda r: r.set_patch_param(lyr, path, val))


async def patch_state(request: Request) -> JSONResponse:
    """Patch state."""
    r = SESSION.live
    name = request.path_params["layer"]
    if r is None or name not in r.instruments or name == "riser":
        return JSONResponse({"ok": False, "error": "no patch"}, status_code=404)
    return JSONResponse(
        {
            "ok": True,
            "layer": name,
            "name": r.patch_names[name],
            "patch": r.instruments[name].patch.model_dump(),
        }
    )


async def effects(request: Request) -> JSONResponse:
    """Effects."""
    b = await _body(request)
    try:
        lyr = _validate_str(str(b["layer"]))
        if lyr not in _ALLOWED_LAYERS_MASTER:
            raise ValueError(f"unknown layer {lyr!r}")
        specs = b.get("effects")
        if specs is not None:
            if not isinstance(specs, list) or len(specs) > 8:
                raise ValueError("effects must be list <=8")
            # validation rapide côté web avant build_effects
            for s in specs:
                if not isinstance(s, dict) or len(s) > 12:
                    raise ValueError("invalid effect spec")
        else:
            specs = None
    except (ValueError, KeyError, TypeError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    # copie pour éviter mutation
    safe_specs = list(specs) if specs is not None else None
    return _cmd(lambda r: r.set_layer_effects(lyr, safe_specs))


async def auto_tweaks(request: Request) -> JSONResponse:
    """Auto tweaks."""
    b = await _body(request)
    val = b.get("enabled", True)
    if not isinstance(val, bool):
        return JSONResponse({"ok": False, "error": "enabled must be bool"}, status_code=400)
    return _cmd(lambda r: r.set_auto_tweaks(bool(val)))


async def next_section(_: Request) -> JSONResponse:
    """Next section."""
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


app = Starlette(
    middleware=[Middleware(SecurityHeadersMiddleware)],
    routes=[
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
    ],
)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    # avertissement si bind non-local
    """Serve."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"[synthwave] WARNING: UI bound to {host} is exposed to network — "
            "no auth, use at your own risk. Prefer 127.0.0.1",
            file=sys.stderr,
        )
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
