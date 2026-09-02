from __future__ import annotations

import re
import time

import typer

from .audio.renderer import RenderConfig, Renderer
from .composer.moods import MOODS
from .patches.loader import list_patches

app = typer.Typer(help="Infinite procedural synthwave generator.", no_args_is_help=True)
_DUR = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?$")


def parse_duration(text: str) -> float:
    m = _DUR.match(text.strip())
    if not m or not any(m.groups()):
        raise ValueError(f"invalid duration {text!r} (examples: 90, 90s, 5m, 1h30m)")
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + s


@app.command()
def play(duration: str | None = typer.Option(None, help="ex: 5m, 90s, 1h. Absent = infini"),
         bpm: float | None = typer.Option(None, min=60, max=180),
         seed: int | None = typer.Option(None),
         mood: str = typer.Option("dark", help="|".join(MOODS)),
         export: str | None = typer.Option(None, help="Rendu hors-ligne vers un WAV"),
         blocksize: int = typer.Option(1024),
         device: str | None = typer.Option(None, help="Nom ou index du périphérique")):
    """Joue de la synthwave sur la sortie audio (ou exporte en WAV)."""
    seconds = parse_duration(duration) if duration else None
    if export and seconds is None:
        raise typer.BadParameter("--export requires --duration")
    renderer = Renderer(RenderConfig(bpm=bpm, mood=mood, seed=seed, duration_s=seconds))
    typer.echo(f"seed={renderer.seed} bpm={renderer.bpm:g} mood={mood} "
               f"key={renderer.arranger.harmony.key_name}")
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
        raise typer.Exit(1)
    typer.echo("playing... Ctrl+C to stop")
    last = None
    try:
        while player.running:
            st = renderer.status()
            line = (f"[{st['section']:>6}] bar {st['bar']:>4}  {st['key']:<9} {st['chord']:<6} "
                    f"underruns={player.underruns}")
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
