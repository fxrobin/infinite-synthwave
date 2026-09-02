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
uv run synthwave play --duration 5m --mood outrun --bpm 118      # tempo fixe
uv run synthwave play --mood dark --bpm-range 80-95                # zone de tempo
uv run synthwave play --duration 3m --seed 42 --export track.wav   # rendu hors-ligne (.wav .flac .ogg .mp3)
uv run synthwave play --fx pad:gate:rate=1/16,depth=0.8 --fx master:lofi:bits=8
uv run synthwave patches                       # patches disponibles
uv run synthwave devices                       # périphériques audio
uv run synthwave mcp                           # serveur MCP (stdio)
```

Moods :

| Mood | BPM | Gamme | Accords | Rythme | Patches |
|---|---|---|---|---|---|
| `dark` | 82–100 | phrygien | triades, i‑bII, i‑iv‑bII‑i, i‑bVI‑bII‑i | half‑time | jeu `*_dark` (FM, écho percussions) |
| `noir` | 86–104 | mineur harmonique | triades, i‑VI‑V‑i, i‑iv‑V‑i | half‑time | jeu `*_dark` |
| `dreamy` | 98–114 | mineur naturel / majeur | 7e | 4/4 | jeu par défaut |
| `outrun` | 110–128 | mineur naturel | 7e | 4/4 dense | jeu par défaut |

Le tempo est tiré dans la zone du mood au démarrage et redessiné à chaque transition
(`--bpm-range 80-95` ou `bpm_range` dans l'outil MCP `start` pour imposer une zone,
`--bpm` pour un tempo fixe). Le jeu de patches suit le mood (changé lors des transitions)
sauf pour les couches chargées à la main via `load_patch`. La basse et le lead changent
d'instrument à chaque section, tirés dans un pool par mood (`bass_dark`, `bass_industrial`,
`bass_reese` / `bass_moog`, `bass_reese`, `bass_acid` ; `lead_dark`, `lead_industrial` /
`lead_saw`, `lead_industrial`). Styles de basse : `eighths`, `octaves`, `syncopated`,
`sixteenths`, `walk`, `riff` (chromatique b2 / triton), mutés une mesure sur deux.

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
chorus → break → **transition** … avec fills en fin de section et, en mode durée, un outro
avec fondu. Les transitions (4 mesures, ambient + pad seuls) portent les changements de
tonalité, de tempo et de mood ; un `set_mood` MCP est appliqué via une transition.

Effets d'insert choisis par l'arrangeur selon la section (`gate` hachure synchro tempo sur
pad/arp en chorus, `lofi` master en intro/break, `bitcrush` sur l'arp) ; réglables à la main
avec `--fx` ou l'outil MCP `set_layer_effects`.

## Patches YAML

Bibliothèque : `synthwave/patches/library/`. Les fichiers placés dans
`~/.config/synthwave/patches/` sont aussi listés et priment sur la bibliothèque.

```yaml
name: pad_juno
polyphony: 6
volume: 0.5
oscillators:
  - {wave: saw, unison: 3, detune: 14, level: 0.7, spread: 1.0}   # saw|square|triangle|sine|noise|fm
  - {wave: fm, fm_ratio: 3.5, fm_index: 1.4, level: 0.4}           # FM 2 opérateurs (cloches, growl)
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
  - {type: gate, rate: "1/16", depth: 0.8, duty: 0.5}          # hachure synchro tempo
  - {type: bitcrush, bits: 8, downsample: 4, mix: 0.5}
  - {type: lofi, bits: 10, downsample: 3, cutoff: 4000, wobble: 0.003, noise: 0.005}
  - {type: distortion, drive: 6.0, tone: 2500, mix: 0.8}      # saturation tanh + tone
```

Patch batterie (`kind: drums`) : `kick` (pitch, `sub`, `drive` saturation, `gain`), `snare`
(gated reverb, `gain`), `hat`, `clap`, `tom`, `crash_gain`, et `perc_effects` : chaîne
d'effets appliquée à toutes les percussions sauf le kick (écho ping‑pong + reverb dans
`drums_dark.yaml`). Voir `drums_808.yaml`.

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
| `set_layer_effects(layer, effects)` | inserts manuels, `layer` ou `master`, `None` = auto |
| `next_section()` | passe à la section suivante |
| `export_wav(path, seconds, mood, bpm, seed)` | rendu hors-ligne |

## Tests

```bash
uv run pytest -q
uv run ruff check .
```
