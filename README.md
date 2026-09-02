# Infinite Synthwave

Générateur procédural de synthwave / outrun qui joue en continu sur la sortie audio
(ou pendant une durée fixée), avec des synthétiseurs internes programmables en YAML,
un tracker 16 pas piloté par une harmonie markovienne, une CLI et un serveur MCP.

## Installation

```bash
uv sync
```

Dépendances : numpy, scipy, sounddevice (PortAudio → PipeWire/Pulse/ALSA), soundfile,
pyyaml, pydantic, mcp, typer.

## CLI

```bash
uv run synthwave play                          # infini, mood dark, Ctrl+C pour arrêter
uv run synthwave play --duration 5m --mood outrun --bpm 118
uv run synthwave play --duration 3m --seed 42 --export track.wav   # rendu hors-ligne
uv run synthwave patches                       # patches disponibles
uv run synthwave devices                       # périphériques audio
uv run synthwave mcp                           # serveur MCP (stdio)
```

Moods : `dark` (100 BPM, mineur, filtres fermés), `dreamy` (108 BPM, majeur possible,
pads dominants), `outrun` (118 BPM, batterie dense, arpèges permanents).

Même seed + mêmes options ⇒ même musique.

## Architecture

```
synthwave/
  engine/     DSP numpy : oscillateurs polyBLEP, ADSR, biquad, LFO, effets, voix, synthé, batterie
  patches/    modèles pydantic + bibliothèque YAML
  composer/   moods, harmonie (Markov), générateurs de patterns, arrangeur par sections
  sequencer/  transport (BPM, pas) et tracker (patterns → événements note-on/off)
  audio/      renderer (mix, sidechain, limiteur), sortie sounddevice, export WAV
  cli.py      typer
  mcp_server.py
```

Couches : `drums`, `bass`, `arp`, `pad`, `lead`, `ambient`. Sections : intro → verse →
chorus → break … avec fills en fin de section, modulation de tonalité toutes les 6 sections
et, en mode durée, un outro avec fondu.

## Patches YAML

Bibliothèque : `synthwave/patches/library/`. Les fichiers placés dans
`~/.config/synthwave/patches/` sont aussi listés et priment sur la bibliothèque.

```yaml
name: pad_juno
polyphony: 6
volume: 0.5
oscillators:
  - {wave: saw, unison: 3, detune: 14, level: 0.7, spread: 1.0}   # saw|square|triangle|sine|noise
  - {wave: square, octave: -1, level: 0.25, pwm: 0.45}
amp_env: {attack: 0.9, decay: 0.6, sustain: 0.8, release: 1.8}
filter:
  type: lp                     # lp|hp|bp
  cutoff: 1400
  resonance: 0.15
  env: {attack: 0.8, decay: 1.2, sustain: 0.4, release: 1.5, amount: 900}
lfo: {wave: sine, rate: 0.25, target: cutoff, amount: 350}       # pitch|cutoff|amp|pwm
effects:
  - {type: chorus, rate: 0.6, depth: 0.004, mix: 0.45}
  - {type: delay, time: "1/8d", feedback: 0.35, mix: 0.3, pingpong: true}
  - {type: reverb, size: 0.9, damping: 0.45, mix: 0.35, predelay: 0.03}
```

Patch batterie (`kind: drums`) : paramètres `kick`, `snare` (gated reverb), `hat`, `clap`,
`tom`. Voir `drums_808.yaml`.

## MCP

`.mcp.json` à la racine déclare le serveur pour Claude Code. Outils :

| Outil | Rôle |
|---|---|
| `start(mood, bpm, seed, duration_s)` | démarre la lecture |
| `stop()` | arrête |
| `status()` | tempo, tonalité, accord, section, couches |
| `set_tempo(bpm)` | 60–180 |
| `set_mood(mood)` | dark / dreamy / outrun |
| `set_layer(layer, mute, solo, volume)` | mixage par couche |
| `list_patches()` / `load_patch(layer, name)` | patches |
| `set_patch_param(layer, path, value)` | ex. `filter.cutoff` 800 |
| `next_section()` | passe à la section suivante |
| `export_wav(path, seconds, mood, bpm, seed)` | rendu hors-ligne |

## Tests

```bash
uv run pytest -q
uv run ruff check .
```
