from __future__ import annotations

import re
import time

import typer

from .audio.export import export_wav
from .audio.output import Player
from .audio.renderer import LAYERS, RenderConfig, Renderer
from .composer.moods import MOODS
from .engine.effects import _REGISTRY
from .patches.loader import list_patches

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None  # type: ignore[assignment]

app = typer.Typer(help="Infinite procedural synthwave generator.", no_args_is_help=True)
_DUR = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?$")
FX_OPTION = typer.Option(
    None, help="Insert manuel: layer:type[:k=v,...] ex. pad:gate:rate=1/16 master:lofi:bits=8"
)


def parse_duration(text: str) -> float:
    """Parse duration."""
    if not isinstance(text, str) or len(text) > 32 or "\x00" in text:
        raise ValueError(f"invalid duration {text!r}")
    m = _DUR.match(text.strip())
    if not m or not any(m.groups()):
        raise ValueError(f"invalid duration {text!r} (examples: 90, 90s, 5m, 1h30m)")
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    total = h * 3600 + mi * 60 + s
    if not 1 <= total <= 14400:
        raise ValueError(f"duration out of range 1-14400s, got {total}")
    return float(total)


def parse_bpm_range(text: str) -> tuple[float, float]:
    """Parse bpm range."""
    try:
        lo, hi = (float(v) for v in text.split("-", 1))
    except ValueError as e:
        raise ValueError(f"invalid --bpm-range {text!r}, expected e.g. 85-100") from e
    if not 40 <= lo <= hi <= 200:
        raise ValueError(f"bpm range {text!r} must satisfy 40 <= low <= high <= 200")
    return lo, hi


def _parse_fx_value(v: str):
    """Convertit une valeur fx : '0.8' -> 0.8, '8' -> 8, sinon str."""
    try:
        fv = float(v) if "." in v or v.lstrip("-").isdigit() else v  # type: ignore[assignment]
        if isinstance(fv, float) and fv.is_integer() and "." not in v:
            return int(fv)
        return fv
    except ValueError:
        return v


def _parse_fx_params(spec: dict, params: str) -> None:
    """Parse fx params."""
    if len(params) > 200:
        raise ValueError("fx params too long")
    for kv in params.split(","):
        if not kv or "=" not in kv:
            raise ValueError(f"invalid fx param {kv!r}, expected k=v")
        k, v = kv.split("=", 1)
        if len(k) > 32 or len(v) > 64 or "\x00" in k or "\x00" in v:
            raise ValueError(f"fx param too long: {kv!r}")
        spec[k] = _parse_fx_value(v)
        if len(spec) > 12:
            raise ValueError("too many fx params")


def parse_fx(text: str) -> tuple[str, dict]:  # noqa: C901 - fx parsing needs branches
    """
    'pad:gate:rate=1/16,depth=0.8' -> ('pad', {'type': 'gate', 'rate':

    '1/16', 'depth': 0.8}).
    """
    if not isinstance(text, str) or len(text) > 256 or "\x00" in text:
        raise ValueError(f"invalid --fx {text!r}")
    parts = text.split(":", 2)
    if len(parts) < 2:
        raise ValueError(f"invalid --fx {text!r}, expected layer:type[:k=v,...]")
    layer, typ = parts[0], parts[1]
    if len(layer) > 32 or len(typ) > 32:
        raise ValueError("fx layer/type too long")
    if layer not in set(LAYERS) | {"master"}:
        raise ValueError(f"unknown fx layer {layer!r}")
    if typ not in _REGISTRY:
        raise ValueError(f"unknown effect type {typ!r}")
    spec: dict = {"type": typ}
    if len(parts) == 3 and parts[2]:
        _parse_fx_params(spec, parts[2])
    return layer, spec


def _validate_play_head(blocksize: int | None, device: str | None, mood: str | None) -> None:
    """Validate play head."""
    if blocksize is not None and not 64 <= blocksize <= 8192:
        raise typer.BadParameter("blocksize must be 64-8192")
    if device is not None and (len(device) > 128 or "\x00" in device):
        raise typer.BadParameter("device string too long")
    if mood is not None and mood not in MOODS:
        raise typer.BadParameter(f"unknown mood {mood!r}, choose from {list(MOODS)}")


def _parse_play_timings(
    duration: str | None, bpm_range: str | None, track: str
) -> tuple[float | None, tuple[float, float] | None, float]:
    """Parse play timings."""
    try:
        seconds = parse_duration(duration) if duration else None
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    try:
        rng = parse_bpm_range(bpm_range) if bpm_range else None
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    try:
        track_s = parse_duration(track)
    except ValueError as e:
        raise typer.BadParameter(f"invalid --track {track!r}: {e}") from e
    return seconds, rng, track_s


def _build_renderer(  # noqa: PLR0913 - renderer wiring bundles CLI options
    bpm: float | None,
    mood: str | None,
    seed: int | None,
    seconds: float | None,
    rng: tuple[float, float] | None,
    track_s: float,
    fx: list[str] | None,
) -> Renderer:
    """Build renderer."""
    try:
        renderer = Renderer(
            RenderConfig(
                bpm=bpm, mood=mood, seed=seed, duration_s=seconds, bpm_range=rng, track_s=track_s
            )
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    try:
        for layer, spec in (parse_fx(f) for f in fx or []):
            renderer.set_layer_effects(layer, [spec])
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    return renderer


def _run_live_player(renderer: Renderer, blocksize: int, device: str | None) -> None:
    """Run live player."""
    dev = int(device) if device and device.isdigit() else device
    player = Player(renderer, blocksize=blocksize, device=dev)
    try:
        player.start()
    except Exception as e:
        typer.echo(
            f"audio output unavailable: {e}\nTry: synthwave play --duration 2m --export out.wav",
            err=True,
        )
        raise typer.Exit(1) from e
    typer.echo("playing... Ctrl+C to stop")
    last = None
    try:
        while player.running:
            st = renderer.status()
            line = (
                f"[{st['section']:>10}{'*' if st['drop'] else ' '}] "
                f"t{st['track']} {st['track_bar']:>3}/{st['track_bars']:<3} "
                f"{st['mood']:<7} {st['bpm']:>5} "
                f"{st['key']:<16} {st['chord']:<6} underruns={player.underruns}"
            )
            if line != last:
                typer.echo(line)
                last = line
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        player.stop()
    if player.error:
        typer.echo(f"render error: {player.error}", err=True)
        raise typer.Exit(1)


@app.command()
def play(  # noqa: PLR0913 - CLI entry point bundles user options
    duration: str | None = typer.Option(None, help="ex: 5m, 90s, 1h. Absent = infini"),
    bpm: float | None = typer.Option(None, min=60, max=180, help="Tempo fixe"),
    bpm_range: str | None = typer.Option(
        None, help="Zone de tempo, ex. 85-100 (défaut : zone du mood)"
    ),
    seed: int | None = typer.Option(None),
    track: str = typer.Option(
        "3m30", help="Durée visée d'un morceau (intro → outro) avant transition, ex. 3m30, 4m"
    ),
    mood: str | None = typer.Option(
        None, help="|".join(MOODS) + " ; absent = tiré au hasard et changé à chaque transition"
    ),
    export: str | None = typer.Option(None, help="Rendu hors-ligne vers un WAV"),
    blocksize: int = typer.Option(1024),
    device: str | None = typer.Option(None, help="Nom ou index du périphérique"),
    fx: list[str] | None = FX_OPTION,
):
    """Joue de la synthwave sur la sortie audio (ou exporte en WAV)."""
    _validate_play_head(blocksize, device, mood)
    seconds, rng, track_s = _parse_play_timings(duration, bpm_range, track)
    if export and seconds is None:
        raise typer.BadParameter("--export requires --duration")
    renderer = _build_renderer(bpm, mood, seed, seconds, rng, track_s, fx)
    typer.echo(
        f"seed={renderer.seed} bpm={renderer.bpm:g} mood={renderer.mood.name}"
        f"{'' if mood else ' (random)'} key={renderer.arranger.harmony.key_name}"
    )
    if export:
        try:
            n = export_wav(renderer, seconds, export, blocksize)
        except ValueError as e:
            raise typer.BadParameter(str(e)) from e
        typer.echo(f"wrote {export} ({n / renderer.sr:.1f}s)")
        return
    _run_live_player(renderer, blocksize, device)


@app.command()
def patches():
    """Liste les patches disponibles (bibliothèque + ~/.config/synthwave/patches)."""
    for name in list_patches():
        typer.echo(name)


@app.command()
def devices():
    """Liste les périphériques audio."""
    if sd is None:
        typer.echo("sounddevice not available", err=True)
        raise typer.Exit(1)
    typer.echo(str(sd.query_devices()))


@app.command()
def ui(
    port: int = typer.Option(8765, help="Port HTTP"),
    host: str = typer.Option("127.0.0.1"),
    no_browser: bool = typer.Option(False, help="Ne pas ouvrir le navigateur"),
):
    """Lance l'interface web (visualisation + contrôle live)."""
    from .web.server import serve

    typer.echo(f"UI: http://{host}:{port}  (Ctrl+C pour quitter)")
    serve(host=host, port=port, open_browser=not no_browser)


@app.command()
def mcp():
    """Lance le serveur MCP (stdio)."""
    from .mcp_server import main

    main()


if __name__ == "__main__":
    app()
