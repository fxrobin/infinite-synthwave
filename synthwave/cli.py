from __future__ import annotations

import re
import time

import typer

from .audio.renderer import RenderConfig, Renderer
from .composer.moods import MOODS
from .patches.loader import list_patches

app = typer.Typer(help="Infinite procedural synthwave generator.", no_args_is_help=True)
_DUR = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?$")
FX_OPTION = typer.Option(None, help="Insert manuel: layer:type[:k=v,...] "
                               "ex. pad:gate:rate=1/16 master:lofi:bits=8")


def parse_duration(text: str) -> float:
    m = _DUR.match(text.strip())
    if not m or not any(m.groups()):
        raise ValueError(f"invalid duration {text!r} (examples: 90, 90s, 5m, 1h30m)")
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + s


def parse_bpm_range(text: str) -> tuple[float, float]:
    try:
        lo, hi = (float(v) for v in text.split("-", 1))
    except ValueError as e:
        raise ValueError(f"invalid --bpm-range {text!r}, expected e.g. 85-100") from e
    if not 40 <= lo <= hi <= 200:
        raise ValueError(f"bpm range {text!r} must satisfy 40 <= low <= high <= 200")
    return lo, hi


def parse_fx(text: str) -> tuple[str, dict]:
    """'pad:gate:rate=1/16,depth=0.8' -> ('pad', {'type': 'gate', 'rate': '1/16', 'depth': 0.8})."""
    parts = text.split(":", 2)
    if len(parts) < 2:
        raise ValueError(f"invalid --fx {text!r}, expected layer:type[:k=v,...]")
    spec: dict = {"type": parts[1]}
    if len(parts) == 3 and parts[2]:
        for kv in parts[2].split(","):
            k, v = kv.split("=", 1)
            try:
                spec[k] = float(v) if "." in v or v.lstrip("-").isdigit() else v
                if isinstance(spec[k], float) and spec[k].is_integer() and "." not in v:
                    spec[k] = int(spec[k])
            except ValueError:
                spec[k] = v
    return parts[0], spec


@app.command()
def play(duration: str | None = typer.Option(None, help="ex: 5m, 90s, 1h. Absent = infini"),
         bpm: float | None = typer.Option(None, min=60, max=180, help="Tempo fixe"),
         bpm_range: str | None = typer.Option(None, help="Zone de tempo, ex. 85-100 "
                                                          "(défaut : zone du mood)"),
         seed: int | None = typer.Option(None),
         mood: str | None = typer.Option(None, help="|".join(MOODS) + " ; absent = tiré au "
                                                       "hasard et changé à chaque transition"),
         export: str | None = typer.Option(None, help="Rendu hors-ligne vers un WAV"),
         blocksize: int = typer.Option(1024),
         device: str | None = typer.Option(None, help="Nom ou index du périphérique"),
         fx: list[str] | None = FX_OPTION):
    """Joue de la synthwave sur la sortie audio (ou exporte en WAV)."""
    seconds = parse_duration(duration) if duration else None
    if export and seconds is None:
        raise typer.BadParameter("--export requires --duration")
    rng = parse_bpm_range(bpm_range) if bpm_range else None
    renderer = Renderer(RenderConfig(bpm=bpm, mood=mood, seed=seed, duration_s=seconds,
                                     bpm_range=rng))
    for layer, spec in (parse_fx(f) for f in fx or []):
        renderer.set_layer_effects(layer, [spec])
    typer.echo(f"seed={renderer.seed} bpm={renderer.bpm:g} mood={renderer.mood.name}"
               f"{'' if mood else ' (random)'} key={renderer.arranger.harmony.key_name}")
    if export:
        from .audio.export import export_wav
        n = export_wav(renderer, seconds, export, blocksize)
        typer.echo(f"wrote {export} ({n / renderer.sr:.1f}s)")
        return
    from .audio.output import Player
    dev = int(device) if device and device.isdigit() else device
    player = Player(renderer, blocksize=blocksize, device=dev)
    try:
        player.start()
    except Exception as e:
        typer.echo(f"audio output unavailable: {e}\n"
                   "Try: synthwave play --duration 2m --export out.wav", err=True)
        raise typer.Exit(1) from e
    typer.echo("playing... Ctrl+C to stop")
    last = None
    try:
        while player.running:
            st = renderer.status()
            line = (f"[{st['section']:>10}] bar {st['bar']:>4} {st['mood']:<7} {st['bpm']:>5} "
                    f"{st['key']:<16} {st['chord']:<6} underruns={player.underruns}")
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
def patches():
    """Liste les patches disponibles (bibliothèque + ~/.config/synthwave/patches)."""
    for name in list_patches():
        typer.echo(name)


@app.command()
def devices():
    """Liste les périphériques audio."""
    import sounddevice as sd
    typer.echo(str(sd.query_devices()))


@app.command()
def mcp():
    """Lance le serveur MCP (stdio)."""
    from .mcp_server import main
    main()


if __name__ == "__main__":
    app()
