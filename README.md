# Infinite Synthwave

Générateur procédural de synthwave / outrun qui joue en continu sur la sortie audio
(ou pendant une durée fixée), avec des synthétiseurs internes programmables en YAML,
un tracker 16 pas piloté par une harmonie markovienne, une CLI et un serveur MCP.

## TL;DR

```bash
# 1. Installer
uv sync

# 2. Jouer en continu (infini, dark aléatoire) — Ctrl+C pour arrêter
uv run synthwave play

# 3. Rendre un morceau hors-ligne et l'exporter
uv run synthwave play --duration 3m --seed 42 --export track.wav

# 4. Lister les patches / périphériques
uv run synthwave patches
uv run synthwave devices
```

Prérequis : Python ≥ 3.13, PortAudio (PipeWire / Pulse / ALSA), `uv`.
Pas de DAW, pas de plugin externe — tout est synthétisé en numpy/scipy.

---

## Installation

```bash
uv sync
```

Dépendances : `numpy`, `scipy`, `sounddevice` (PortAudio → PipeWire/Pulse/ALSA), `soundfile`,
`pyyaml`, `pydantic`, `mcp`, `typer`.

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

| Option | Description |
|---|---|
| `--duration 5m` | `90`, `90s`, `5m`, `1h30m`. Sans → infini en live, obligatoire avec `--export`. |
| `--mood dark\|noir\|dreamy\|outrun` | Sans → tiré au hasard au démarrage et changé à chaque transition. Avec → figé (sauf `set_mood` MCP). |
| `--bpm 118` | Tempo fixe (60–180). Sans → tiré dans la zone du mood. |
| `--bpm-range 80-95` | Zone de tempo (40–200). Redessiné à chaque transition. Prime sur la zone du mood. |
| `--seed 42` | Seed RNG. Même seed + mêmes options ⇒ même musique bit-identique. |
| `--export track.wav` | Rendu hors-ligne. Extension → format : `.wav` PCM16, `.flac` PCM16, `.ogg` Vorbis, `.mp3` MP3. Requiert `--duration`. |
| `--device 2` / `hw:0,0` | Index ou nom sounddevice. |
| `--blocksize 1024` | Taille de bloc audio. |
| `--fx layer:type:k=v,...` | Insert manuel (répétable). `layer` ou `master`. Ex. `pad:gate:rate=1/16,depth=0.8` `master:lofi:bits=8` `lead:phaser:rate=2/1`. `None` en MCP = retour auto. |

## Moods

| Mood | BPM | Gamme | Accords | Rythme | Patches |
|---|---|---|---|---|---|
| `dark` | 82–100 | phrygien | triades, `i‑bII`, `i‑iv‑bII‑i`, `i‑bVI‑bII‑i` | half‑time | jeu `*_dark` (FM, écho percussions) |
| `noir` | 86–104 | mineur harmonique | triades, `i‑VI‑V‑i`, `i‑iv‑V‑i` | half‑time | jeu `*_dark` |
| `dreamy` | 98–114 | mineur naturel / majeur | 7e | 4/4 | jeu par défaut |
| `outrun` | 110–128 | mineur naturel | 7e | 4/4 dense | jeu par défaut |
| `cyberpunk` | 118–132 | mineur naturel | triades, i‑bII, i‑v‑VI‑iv | 4/4 dense, basse riff/16e | jeu `*_dark`, industriel |
| `horror` | 66–80 | locrien | triades, i‑bII‑bv‑i | half‑time très lent | jeu `*_dark` |
| `desert` | 88–104 | phrygien dominant | triades, I‑bII‑I‑bvii | half‑time | jeu `*_dark` |
| `chill` | 84–96 | dorien | 7e, i‑IV, i‑bVII‑IV‑i | 4/4 léger, basse walk | jeu par défaut |
| `retro` | 104–118 | mixolydien / majeur | 7e, I‑V‑vi‑IV, I‑bVII‑IV | 4/4 | jeu par défaut, lead pulse |
| `drive` | 122–136 | mineur naturel | 7e, i‑VI‑III‑VII | 4/4 très dense | jeu par défaut |

Sans `--mood`, le mood est tiré au hasard au démarrage et retiré à chaque transition ;
`--mood X` le fige (en MCP, `set_mood("random")` rend la main au hasard).
Le tempo est tiré dans la zone du mood au démarrage et redessiné à chaque transition
(`--bpm-range 80-95` ou `bpm_range` dans l'outil MCP `start` pour imposer une zone,
`--bpm` pour un tempo fixe). Le jeu de patches suit le mood (changé lors des transitions)
sauf pour les couches chargées à la main via `load_patch`. La basse et le lead changent
d'instrument à chaque section, tirés dans un pool par mood (`bass_dark`, `bass_industrial`,
`bass_sub`, `bass_growl`, `bass_reese` / `bass_moog`, `bass_sub`, `bass_pulse`, `bass_reese` ;
`lead_dark`, `lead_industrial`, `lead_scream` / `lead_saw`, `lead_pulse`, `lead_industrial`). Styles de basse : `eighths`, `octaves`, `syncopated`,
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

Couches : `drums`, `bass`, `arp`, `pad`, `lead`, `ambient`, `riser` (annonces de chorus
synthétisées : uplifter 2 mesures, cymbale reverse, cri FM, impact). Sections : intro → verse →
chorus → break → **transition** … avec fills en fin de section et, en mode durée, un outro
avec fondu. Les transitions (4 mesures, ambient + pad seuls) portent les changements de
tonalité, de tempo et de mood ; un `set_mood` MCP est appliqué via une transition.

Le chorus est annoncé sur ses deux dernières mesures d'approche (uplifter, cymbale reverse,
cri) et frappe avec un impact, une batterie 4/4 pleine (kick, snare + clap, hats) même en
mood half‑time, une basse en croches ou doubles, et un lead quasi systématique.

Effets d'insert choisis par l'arrangeur selon la section (`gate` hachure synchro tempo sur
pad/arp en chorus, `lofi` master en intro/break, `bitcrush` sur l'arp, et pour le lead en
chorus : `autopan`, gate + autopan, distortion + autopan, bitcrush, `phaser` ou flanger +
distortion) ; réglables à la main
avec `--fx` ou l'outil MCP `set_layer_effects`.

## Synthétiseur interne — tout ce qui est paramétrable

### Oscillateurs (`synthwave/engine/oscillator.py:21`)

| Paramètre | Valeurs / plage | Rôle |
|---|---|---|
| `wave` | `saw` `square` `triangle` `sine` `noise` `fm` | Forme d'onde. `saw`/`square` sont band-limited (polyBLEP) |
| `fm_ratio` / `fm_index` | `0.1–16.0` / `0.0–10.0` | FM 2 opérateurs (uniquement `wave: fm`) — cloches, growls, scream |
| `unison` | `1–8` | Voix désaccordées empilées, pan stéréo en éventail |
| `detune` | `0–100` cents | Écart max du unison |
| `spread` | `0.0–1.0` | Largeur stéréo du unison |
| `octave` / `semi` | `-3–+3` / `-12–+12` | Transposition statique |
| `level` | `0.0–2.0` | Gain de l'oscillateur |
| `pwm` | `0.05–0.95` | Rapport cyclique (uniquement `square`, modulable par LFO) |

### Filtre (`synthwave/engine/filter.py:8`)

Biquad RBJ par bloc, état `lfilter`.

| Paramètre | Valeurs | Note |
|---|---|---|
| `type` | `lp` `hp` `bp` | Passe-bas / haut / bande |
| `cutoff` | `20–20000` Hz | Modulable : enveloppe + LFO + key-track |
| `resonance` | `0.0–1.0` | Q = 0.5 + res×9.5 |
| `env` | `{attack, decay, sustain, release, amount}` | Enveloppe ADSR dédiée filtre, `amount` en Hz |
| `key_track` | `0.0–1.0` | Suit la hauteur de note (`freq - 261.6`) |

### Enveloppes (`synthwave/engine/envelope.py:9`)

ADSR par blocs : attack linéaire, decay/release exponentiels, stage `SUSTAIN` tenu. Deux instances par voix si `filter.env` est présent (`amp_env` + `filt_env`).

| Champ | Défaut patch | Plage |
|---|---|---|
| `attack` | `0.01` | `0.0–` s |
| `decay` | `0.1` | `0.0–` s |
| `sustain` | `1.0` | `0.0–1.0` |
| `release` | `0.2` | `0.0–` s |

Glide/portamento : `glide: 0.04` (secondes, `synthwave/engine/voice.py:29`) — exponentiel par bloc, legato si monophonique.

### LFO (`synthwave/engine/lfo.py:6`, `synthwave/engine/voice.py:52`)

| Paramètre | Valeurs |
|---|---|
| `wave` | `sine` `triangle` `square` `saw` |
| `rate` | Hz (`>0`) |
| `target` | `pitch` (demi-tons) · `cutoff` (Hz) · `amp` (tremolo 0..1) · `pwm` (0.5±0.45) |
| `amount` | demi-tons / Hz / 0..1 selon target |

Le vibrato `pitch` est évalué sur la moyenne du bloc, le `pwm` est passé par-voix à l'oscillateur, le `amp` multiplie l'enveloppe.

### Polyphonie & voix (`synthwave/engine/voice.py:16`, `synthwave/engine/synth.py:12`)

`polyphony: 1–16`, voice-stealing au plus ancien (`age`). Monophonique ⇒ legato + glide. Chaque voix possède ses oscillateurs, ses enveloppes, son filtre et son LFO.

---

## Effets — référence complète

12 effets bloc-vectorisés (`synthwave/engine/effects.py:392`). Paramètres temps synchronisés au BPM via `note_to_seconds` (`1/4`, `1/8`, `1/8d` dotée, `1/8t` triolet, ou secondes flottantes). Empilables dans `effects:` d'un patch, surchargeables par section par l'arrangeur, et par `--fx` / `set_layer_effects` (par couche ou `master`).

| # | `type` | Classe | Paramètres (défauts) | Ce que ça fait | Où utilisé |
|---|---|---|---|---|---|
| 1 | `chorus` | `Chorus` | `rate=0.5` Hz, `depth=0.003` s, `mix=0.4` | Double délai modulé par 2 LFO en quadrature (base 20 ms, buffer 0.2 s). Épaissit pad/ambient. | `pad_juno`, `lead_saw`, `ambient_*` |
| 2 | `delay` | `Delay` | `time="1/8"` (ou `"1/8d"/"1/4"` …), `feedback=0.4` (0–0.95), `mix=0.3`, `pingpong=true` | Delay tempo-sync, traité par tranches ≤ délai. Ping-pong mono→stéréo. | `arp_*`, `bass?`, `drums_dark` perc |
| 3 | `reverb` | `Reverb` | `size=0.8` (0–1 → feedback 0.7–0.98), `damping=0.5` (0–0.95 lowpass combs), `mix=0.3`, `predelay=0.02` s | Freeverb Schroeder : 8 combs lowpass-feedback + 4 allpasses par canal, predelay. | Quasi tous les patches + `drums_dark` |
| 4 | `gated_reverb` | `GatedReverb` | `size=0.85`, `hold=0.25` s, `threshold=0.1`, `mix=0.5` | Reverb + gate : queue coupée `hold` après chaque hit > threshold. Son snare/clap 80s. | `snare`/`clap` dans `DrumKit` uniquement |
| 5 | `limiter` | `Limiter` | `threshold=0.9` (0.95 master), `release=0.1` s | Limiteur brickwall par bloc : attaque instantanée, release exponentielle, clipping final. | Master Bus (`audio/renderer.py:76`) |
| 6 | `gate` | `Gate` | `rate="1/16"` (tempo-sync), `depth=1.0` (0–1), `duty=0.5` (0.05–0.95), `smooth=0.002` s | Trance-gate : hachure synchro tempo à enveloppe lissée (1-pole). | Arrangeur `pad`/`arp` en chorus/verse |
| 7 | `bitcrush` | `Bitcrush` | `bits=8` (2–16), `downsample=4` (1–), `mix=1.0` | Réduction bit-depth + sample-and-hold. | Arrangeur `arp`/`lead` chorus |
| 8 | `lofi` | `LoFi` | `bits=10`, `downsample=3`, `cutoff=5000` Hz, `wobble=0.002` s, `noise=0.004`, `mix=1.0` | Chaîne : bitcrush → lowpass → chorus wobble (rate 0.4 Hz) → hiss. Cassette. | Arrangeur `master` intro/break |
| 9 | `distortion` | `Distortion` | `drive=4.0` (≥1), `tone=4000` Hz, `mix=1.0` | `tanh(drive)` + comp. `1/tanh(drive*0.5)` + lowpass tone. Saturation. | `bass_*`, `lead_*`, arrangeur lead chorus |
| 10 | `autopan` | `AutoPan` | `rate="1/2"` (Hz ou note), `depth=0.8` (0–1), `wave=sine` | Auto-pan constant-power à LFO, rendu mono→stéréo en loi √2. | Arrangeur `lead` chorus |
| 11 | `phaser` | `Phaser` | `rate=0.3` (Hz ou note), `depth=0.8`, `stages=4` (2–8), `mix=0.5`, `feedback=0.3` (0–0.9) | Cascade d'allpasses 1er ordre balayés 300 Hz–2.4 kHz par LFO triangle, par chunks 128. | `arp_dark`, `lead_*` |
| 12 | `flanger` | `Flanger` | `rate=0.25` (Hz ou note), `depth=0.002` s, `base=0.003` s, `feedback=0.5` (0–0.9), `mix=0.5` | Court délai modulé sine (buffer 50 ms) avec feedback, interpolé linéaire. | `lead_pulse`, arrangeur `lead` chorus |

Notes :
- `note_to_seconds` : `"1/16"`, `"1/8d"` (= ×1.5), `"1/8t"` (= ×2/3), `"1/4"` etc. → `4 * (60/bpm) * num/den * mult`.
- Chaînes : `master` et par-couche s'additionnent ; `manual_fx` remplace `auto_fx` de l'arrangeur ; `_rebuild_inserts` reconstruit à chaque changement de BPM.
- DrumKit `perc_effects` = chaîne appliquée à toutes les percu *sauf* le kick ; RiserKit n'a pas d'inserts.

## Patches YAML

Bibliothèque : `synthwave/patches/library/` (21 patches). Les fichiers placés dans
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
  - {type: autopan, rate: "1/2", depth: 0.9}                   # pan stéréo synchro tempo
  - {type: phaser, rate: 0.3, depth: 0.8, stages: 4, mix: 0.5, feedback: 0.3}
  - {type: flanger, rate: 0.25, depth: 0.002, base: 0.003, feedback: 0.5, mix: 0.5}
```

Patch batterie (`kind: drums`) : `kick` (pitch, `sub`, `drive` saturation, `gain`), `snare`
(gated reverb, `gain`), `hat`, `clap`, `tom`, `crash_gain`, et `perc_effects` : chaîne
d'effets appliquée à toutes les percussions sauf le kick (écho ping‑pong + reverb dans
`drums_dark.yaml`). Voir `drums_808.yaml`.

### Bibliothèque livrée (21 patches)

| Catégorie | Patch | Caractère |
|---|---|---|
| **Pad** | `pad_juno` | Juno saw 3×14 cts, chorus+reverb |
| | `pad_dark` | Saw 4×10 cts + FM, cutoff 520 Hz sombre |
| **Bass** | `bass_moog` | Saw + square sub, moog ladder-style |
| | `bass_dark` | Saw + FM 0.5×0.6 + sine sub, LFO cutoff 0.5 Hz |
| | `bass_acid` | Saw résonante, env 1200 Hz, overdrive |
| | `bass_sub` | Sine pur + triangle, LP 900 Hz |
| | `bass_pulse` | Square PWM 0.35 unison, LFO cutoff |
| | `bass_reese` | Saw 3×22 cts, LFO cutoff 0.3 Hz |
| | `bass_growl` | FM 1.0×1.6 + LFO amp 3.5 Hz, disto 3.5 |
| | `bass_industrial` | Saw unison + FM 2.0×1.2, env 400 Hz, disto 6.0 |
| **Arp** | `arp_pluck` | Saw 2×8 cts env 2200 Hz, delay 1/8d |
| | `arp_dark` | Square PWM 0.25 + saw, disto+phaser+delay |
| **Lead** | `lead_saw` | Saw 4×18 cts, LFO vibrato 5.5 Hz, chorus+phaser |
| | `lead_dark` | Saw 3×16 cts + FM, vibrato 4.8 Hz, disto+phaser |
| | `lead_industrial` | FM 2.0×2.5 + saw, env 1800 Hz, disto 5.0 |
| | `lead_pulse` | Square PWM LFO, flanger+delay |
| | `lead_scream` | Saw 5×25 cts, vibrato 6 Hz, disto 7.0 + phaser 6 stages |
| **Ambient** | `ambient_drone` | Sine+triangle+noise, LFO cutoff, chorus+reverb |
| | `ambient_dark` | Noise+FM+sine, LFO triangle 0.05 Hz, reverb 1.0 |
| **Drums** | `drums_808` | 808 sec, hats 8500 Hz |
| | `drums_dark` | Kick grave 140→40 Hz, hats feutrés, delay 1/8d + reverb perc |

---

## Batterie synthétisée (`synthwave/engine/drums.py:31`)

Chaque hit est rendu une fois à l'init en buffer stéréo, puis mixé à la demande (polyphonie par superposition, `closed hat choke`).

| Pièce | Synthèse | Paramètres patch |
|---|---|---|
| `kick` | Sine pitch-drop exponentiel `pitch_start→pitch_end` + sub sine + click noise 2 ms + `tanh(drive)` | `pitch_start/end/decay`, `click`, `sub/sub_decay`, `drive`, `gain` |
| `snare` | Sine `tone` + bruit HP 1800 Hz, enveloppes séparées, `GatedReverb` | `tone/tone_decay/noise_decay`, `gate_hold`, `reverb_size/mix`, `gain` |
| `clap` | Bruit BP 1500 Hz + 4 bursts 11 ms + tail, `GatedReverb` | `decay`, `gate_hold`, `reverb_mix`, `gain` |
| `hat` closed/open | Bruit HP `cutoff`, decays exponentiels | `closed_decay/open_decay`, `cutoff`, `gain` |
| `tom` low/mid | Sine pitch-bend `1+0.6*exp(-t/0.04)` | `pitch_low/mid`, `decay`, `gain` |
| `crash` | Bruit BP 6000 Hz, decay 0.7 s | `crash_gain` |
| `perc_effects` | Chaîne post-mix hors kick | `perc_effects: [delay, reverb, …]` |

Notes MIDI : `kick 36`, `snare 38`, `clap 39`, `hat_closed 42`, `hat_open 46`, `tom_low 45`, `tom_mid 47`, `crash 49`.

## Risers & transitions (`synthwave/engine/risers.py:32`)

One-shots resynthétisés à chaque changement de BPM (durées en bars) :

| Note | Nom | Synthèse | Durée |
|---|---|---|---|
| 60 | `reverse_cymbal` | Bruit LP sweep 800→9000 Hz, enveloppe `exp((t-1)*5)` | 1 bar |
| 64 | `reverse_short` | Moitié du reverse, fade `0.3→1.0` | 0.5 bar |
| 61 | `uplifter` | Saw montante 1 octave + bruit, LP sweep 300→12000 Hz, `t^1.5` | 2 bars |
| 62 | `scream` | FM 300 Hz→1.36 kHz, `idx 1→8`, `tanh(3)` | 1 bar |
| 63 | `impact` | Sub 45→105 Hz + burst LP 6000→200 Hz, `tanh(2)` | 1.2 s |

Annonces arrangeur (`synthwave/composer/arranger.py:219`) : `uplifter` à J-2 du chorus, `reverse` + `scream` 60% à J-1, `reverse_short` 50% en fin de section non-chorus, `impact` au downbeat du chorus.

## Mix, sidechain & master (`synthwave/audio/renderer.py:25`)

- **Sidechain** (`engine/effects.py:209`) : ducking `gain = 1 - depth*exp(-t/release)` déclenché sur chaque kick, `depth 0.45 / release 0.22 s`. Atténuation par couche : `pad 100%`, `bass 60%`, `ambient 60%`, `arp 50%`.
- **Gains de section** : `intro drums 0.6/lead 0.0` → `verse` → `chorus full` → `break drums 0.0/lead 0.0` → `transition drums/bass/arp/lead 0.0` → `outro` avec `fade` linéaire sur 8 bars.
- **Trim par couche** (`audio/renderer.py:26`) : `LAYER_TRIM = {"lead": 2.5}` — make-up gain statique appliqué avant mix pour compenser le niveau perçu du lead (×2.5, les autres ×1.0).
- **Master** : `master_volume 0.7` + rampe de gain par couche (`current→target` sur un bloc) + `fade` linéaire + inserts `master` + `Limiter threshold 0.95`.

---

## Composition — toutes les variations implémentées

### Gammes (`synthwave/composer/harmony.py:21`)

`minor` (0 2 3 5 7 8 10), `major` (0 2 4 5 7 9 11), `phrygian` (0 1 3 5 7 8 10), `harmonic_minor` (0 2 3 5 7 8 11). Tonalité `tonic` 0–11 tirée au hasard, `modulate()` saute de ±3/5/7 demi-tons à chaque transition (70% de chances).

### Progressions (`harmony.py:11`)

| Nom | Degrés | Mood |
|---|---|---|
| `i-VI-III-VII` | 0 5 2 6 | dreamy/outrun |
| `i-VII-VI-VII` | 0 6 5 6 | dreamy/outrun |
| `i-iv-VI-V` | 0 3 5 4 | dreamy/outrun |
| `i-VI-VII-i` | 0 5 6 0 | dreamy/outrun |
| `VI-VII-i-i` | 5 6 0 0 | dreamy/outrun |
| `i-III-VII-VI` | 0 2 6 5 | dreamy/outrun |
| `iv-VI-i-VII` | 3 5 0 6 | dreamy/outrun |
| `i-bII-i-bII` | 0 1 0 1 | dark |
| `i-iv-bII-i` | 0 3 1 0 | dark |
| `i-bVI-bII-i` | 0 5 1 0 | dark |
| `i-bvii-bVI-bII` | 0 6 5 1 | dark |
| `i-i-bVI-bII` | 0 0 5 1 | dark |
| `i-iv-i-bII` | 0 3 0 1 | dark |
| `i-VI-V-i` | 0 5 4 0 | noir |
| `i-iv-V-i` | 0 3 4 0 | noir |
| `i-i-VI-V` | 0 0 5 4 | noir |

Qualité : sans `sevenths` → triades ; avec → 7e empilées (tierces). Nommage `m/m7/maj7/7/dim/m7b5…`.
Markov : poids par mood, anti-répétition `×0.25` si même progression que précédemment.

### Patterns 16 pas (`synthwave/composer/patterns.py`)

| Couche | Générateur | Variations |
|---|---|---|
| `drums` | `gen_drums` | `strong` (chorus 4/4 dense, kick extra, hats 1/8), `halftime` (kick syncopé, snare beat 3 seul), `fill` (4 hits 12–15, rolls toms), `density` (hats off-beat, open hat 14), `crash` downbeat verse/chorus |
| `bass` | `gen_bass` | 6 styles : `eighths` (croches), `octaves` (alterné), `sixteenths` (doubles + oct. aléat.), `walk` (root-5th-oct), `riff` (b2/triton chromatique), `syncopated` (0 3 6 8 11 14). + mutation 20% une mesure sur deux, octave pop fin de pattern 30% |
| `arp` | `gen_arp` | 3 modes : `up`, `updown`, `random` ; 2 octaves ; 16 doubles par bar |
| `pad` | `gen_pad` | Tenue `STEPS` sur notes de l'accord, voicing grave si root≥6, octave ajoutée si triade |
| `ambient` | `gen_ambient` | Tenue root + 5th (octave 3) |
| `lead` | `gen_lead` | Grille 0 2 4 6 8 10 12 14 (8 pos. paires, sans 3/11), `2+density*3` notes, marche ≤5 demi-tons, tonique d'accord favorisée sur temps fort, legato `length 2–8` ; tessiture `scale_notes(lo, lo+19)` avec `lo=55` (dark/noir) sinon `60` |
| `riser` | `_risers` | Voir ci-dessus |
| `mutate` | `mutate` | Drop 30%, nudge ±1 step 30%, substitution note 40%, insertion libre 50% ; utilisé basse/drums/ambient |

### Sections & arrangeur (`synthwave/composer/arranger.py:38`)

| Section | Bars | Gains particuliers | FX auto possibles |
|---|---|---|---|
| `intro` | 8 | drums 0.6 lead 0.0 | `master lofi` 50% |
| `verse` | 16 | arp 0.85 pad 0.9 lead 0.35 | `arp gate 1/32` 25% |
| `chorus` | 16 | full + lead 1.0 | `pad gate 1/16-1/32` 35–55%, `arp bitcrush`, `lead` 6 pools (autopan / gate+autopan / disto+autopan / bitcrush / phaser 6 stages / flanger+disto) |
| `break` | 8 | drums 0.0 lead 0.0 | `master lofi` 60% sinon `pad gate 1/8` |
| `transition` | 4 | drums/bass/arp/lead 0.0 pad 0.5 ambient 1.0 | — (porte modulation) |
| `outro` | 8 | fade linéaire 1→0 | — |

Enchaînement : `intro→verse→(chorus 2×/break pondéré)→…` ; transition tous les 6 sections ou 25% dès 3 sections, + sur `set_mood`; `outro` quand `bar ≥ total_bars-8` en mode durée. Basse/lead pool resélectionnés chaque section ; `arp_on` selon `arp_prob` (0.45 dark → 0.95 outrun) ; `lead` proba `0.2–0.55` (≥0.7 en chorus) ; batterie mutée un pattern sur ~4 par `mutate`.

## MCP

`.mcp.json` à la racine déclare le serveur pour Claude Code (lancé depuis le dossier du
projet ; pour un autre dossier, ajouter `--directory <chemin>` aux arguments). Outils :

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
