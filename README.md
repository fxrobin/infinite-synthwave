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

# 5. Interface web (visualisation + tweak live) → http://127.0.0.1:8765
uv run synthwave ui
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
uv run synthwave play --mood minimal --patch bass:dx7_bass --patch pad:dx7_epiano  # DX7 6-op fidèle
uv run synthwave play --mood retro --patch pad:solina_wywh --patch lead2:solina_horn  # Solina String Ensemble
uv run synthwave play --mood dreamy --patch pad:d50_Fantasia --patch arp:d50_Staccato_Heaven  # Roland D-50
uv run synthwave patches                       # patches disponibles (724)
uv run synthwave import-d50 PND50-00.syx       # importe un bank D-50 (64 patches) → YAML d50_*
uv run synthwave import-dx7 bank.syx           # importe 32 voix DX7 bulk → YAML dx7_*
uv run synthwave devices                       # périphériques audio
uv run synthwave mcp                           # serveur MCP (stdio)
uv run synthwave ui --port 8765 --no-browser   # interface web
```

| Option | Description |
|---|---|
| `--duration 5m` | `90`, `90s`, `5m`, `1h30m`. Sans → infini en live, obligatoire avec `--export`. |
| `--mood <nom>` | `dark\|noir\|dreamy\|outrun\|cyberpunk\|horror\|desert\|chill\|retro\|drive\|minimal\|programming`. Sans → tiré au hasard au démarrage et changé à chaque transition. Avec → figé (sauf `set_mood("random")` MCP). |
| `--bpm 118` | Tempo fixe (60–180). Sans → tiré dans la zone du mood. |
| `--bpm-range 80-95` | Zone de tempo (40–200). Redessiné à chaque transition. Prime sur la zone du mood. |
| `--track 3m30` | Durée visée d'un morceau (intro → outro) avant la transition vers le suivant. Défaut `3m30` (±8 %). |
| `--seed 42` | Seed RNG. Même seed + mêmes options ⇒ même musique bit-identique. |
| `--export track.wav` | Rendu hors-ligne. Extension → format : `.wav` PCM16, `.flac` PCM16, `.ogg` Vorbis, `.mp3` MP3. Requiert `--duration`. |
| `--device 2` / `hw:0,0` | Index ou nom sounddevice. |
| `--blocksize 1024` | Taille de bloc audio. |
| `--fx layer:type:k=v,...` | Insert manuel (répétable). `layer` ou `master`. Ex. `pad:gate:rate=1/16,depth=0.8` `master:lofi:bits=8` `lead:phaser:rate=2/1`. `None` en MCP = retour auto. |
| `--patch layer:name` | Patch par couche (répétable, `--patches` alias). Ex. `bass:dx7_bass` `pad:dx7_epiano` `lead:dx7_bell` `pad:solina_strings`. Outrepasse le mood pour la couche. |
| `--master-color clean\|tape\|vhs\|mic\|crush` | Couleur permanente du master (défaut `tape`). |

## Moods — 12 ambiances implémentées (`synthwave/composer/moods.py:43`)

| Mood | BPM | Gamme | 7e | Rythme | Densité drums/arp/lead | Brightness | Patches | Pools (rotation par section) | Styles basse |
|---|---|---|---|---|---|---|---|---|---|
| `dark` | 82–100 | phrygien | non | half‑time | 0.35 / 0.45 / 0.2 | 0.6 | `*_dark` | bass: dark/indus/sub/growl/reese · lead: dark/indus/scream | eighths 2, riff 3, sixteenths 2, sync 1, oct 1 |
| `noir` | 86–104 | mineur harmonique | non | half‑time | 0.4 / 0.5 / 0.3 | 0.7 | `*_dark` | idem dark | idem dark |
| `dreamy` | 98–114 | mineur→majeur* | oui | 4/4 droit² | 0.5 / 0.75 / 0.4 | 1.0 | bright | bass: moog/sub/pulse/reese · lead: saw/pulse/indus | eighths 3, oct 2, sync 1, walk 1, sixteenths 1 |
| `outrun` | 110–128 | mineur | oui | 4/4 droit² dense | 0.85 / 0.95 / 0.55 | 1.2 | bright | idem dreamy | idem dreamy |
| `cyberpunk` | 118–132 | mineur | non | 4/4, pad_oct 3 | 0.9 / 0.9 / 0.5 | 0.8 | `*_dark` | bass: indus/growl/reese · lead: indus/scream/dark | sixteenths 3, riff 3, eighths 2, oct 1 |
| `horror` | 66–80 | locrien | non | half‑time très lent | 0.2 / 0.3 / 0.15 | 0.5 | `*_dark` | dark pools | dark bass |
| `desert` | 88–104 | phrygien dominant | non | half‑time | 0.45 / 0.6 / 0.35 | 0.7 | `*_dark` | dark pools | dark bass |
| `chill` | 84–96 | dorien | oui | 4/4 droit² | 0.4 / 0.6 / 0.3 | 0.9 | bright | bright pools | eighths 2, walk 3, sync 2, oct 1 |
| `retro` | 104–118 | mixolydien (major 50%) | oui | 4/4 droit² | 0.6 / 0.8 / 0.45 | 1.1 | bright | bright pools | bright bass |
| `drive` | 122–136 | mineur | oui | 4/4 droit² très dense | 1.0 / 1.0 / 0.5 | 1.2 | drive† | bass: moog/sub/pulse/reese · lead: saw/pulse/scream | eighths 3, sixteenths 3, oct 2, sync 1 |
| `minimal` | 90–108 | dorien | non | 4/4 droit² mono‡ | 0.18 / 0.22 / 0.10 | 0.70 | `minimal` | bass: sub/pulse/moog/808/pluck/sh101/dub/thump/juno60/soft · lead: hollow/organ/juno | eighths 4, octaves 3, sync 1 |
| `programming` | 88–104 | mineur | non | 4/4 droit² mono‡ | 0.14 / 0.18 / 0.08 | 0.65 | `minimal` | idem minimal (hypnotique) | eighths 4, octaves 3, sync 1 |

*`major_prob` : dreamy 0.35, retro 0.5, les autres 0.0. `²4/4 droit` = `Mood.straight` : groove années 80 figé sur la grille, kits de percussion secs (voir « Batterie synthétisée »). `†drive` pools = moog/sub/pulse/reese + saw/pulse/scream. `‡minimal/programming` : `mono_drums` = une seule voix par section `kick xor snare xor hat` (pas de fills/rolls/crash, pas de drop), FX/gestures/risers réduits au minimum pour boucle hypnotique `I-IV` (Basic Channel / dub-techno). `pad_octave` 3 (dark/cyberpunk/horror/desert) vs 4 (les autres).

Sans `--mood`, le mood est tiré au hasard au démarrage et retiré à chaque transition ;
`--mood X` le fige (en MCP, `set_mood("random")` rend la main au hasard).
Le tempo est tiré dans la zone du mood au démarrage et redessiné à chaque transition
(`--bpm-range 80-95` ou `bpm_range` dans l'outil MCP `start` pour imposer une zone,
`--bpm` pour un tempo fixe). Le jeu de patches suit le mood (changé lors des transitions)
sauf pour les couches chargées à la main via `load_patch`. Toutes les couches (basse, lead, pad,
arp, ambient, batterie) changent d'instrument à chaque section, tirées dans les pools du mood
(`DARK_POOLS`, `BRIGHT_POOLS`, `CYBER_POOLS`, `DRIVE_POOLS` dans `moods.py`) ; la basse tire aussi un style pondéré par mood (voir colonnes). Styles : `eighths`, `octaves`, `syncopated`, `sixteenths`, `walk`, `riff` (chromatique b2/triton), chorus → `eighths/sixteenths/octaves` forcé.

Même seed + mêmes options ⇒ même musique.

## Architecture

```
synthwave/
  engine/     DSP numpy : oscillateurs polyBLEP, ADSR, biquad, LFO, effets, voix, synthé, batterie
  patches/    modèles pydantic + bibliothèque YAML
  composer/   moods, harmonie (Markov), générateurs de patterns, arrangeur par sections
  sequencer/  transport (BPM, pas) et tracker (patterns → événements note-on/off)
  audio/      renderer (mix, sidechain, limiteur), sortie sounddevice, export WAV
  session.py  player partagé (MCP + web)
  web/        serveur Starlette + page HTML (synthwave ui)
  cli.py      typer
  mcp_server.py
```

Couches : `drums`, `bass`, `arp`, `pad`, `lead`, `lead2`, `ambient`, `riser` (annonces de chorus
synthétisées : uplifter 2 mesures, cymbale reverse, cri FM, impact).

Le flux infini est une suite de **morceaux** d'environ 3 min 30 (`--track`, `track_s` en MCP) :
`intro → verse / chorus / break … → outro → transition → intro` du morceau suivant. L'outro
(8 mesures) retire les couches une à une (arp à 2, kick seul à 4, basse et batterie à 6) ; la
transition (4 mesures, ambient + pad seuls) porte le changement de tonalité, de tempo et de mood ;
en mode durée, le dernier outro fond jusqu'au silence. Un `set_mood` MCP déclenche outro puis
transition (directement la transition pendant l'intro).

**Build-up** : dans chaque section les couches entrent toutes les 2 mesures. Intro : pad+ambient,
arp à 2, kick à 4, basse à 6. Verse : kick/snare + hats en croches, arp à 2, groove complet et lead
à 4. Chorus : tout d'un coup (le drop), lead à 2. Break : **tout reste en place**, seule la
mélodie rentre (lead à 1, lead2 à 2).

**Breaks** : un break est un *changement mélodique*, pas un changement de groupe. Les patches de
la section précédente sont conservés (aucun `BarPlan.patches` sur sa première mesure), l'arp ne
disparaît pas, batterie et basse continuent (gains 0.85), la couleur master ne descend que d'un
cran sur l'échelle et les gestes de patch tombent à 25 % de probabilité. Le contraste vient de la
**contre-mélodie** du thème, jouée à chaque mesure (variation 0.3, octave supérieure en 2e moitié)
sur un nouveau style de basse, un nouveau mode d'arp et une batterie regénérée.

**Drops** : la dernière mesure avant un chorus est un *drop* (`BarPlan.drop`) : un trou. Batterie,
basse, arp, lead et lead2 sont coupés (gain 0), il ne reste qu'un pad lointain (0.35) et une queue
d'ambient (0.5) ; la mesure est remplie par une cymbale reverse (pas 0, 16 pas) puis un riser qui
monte sur la seconde moitié (pas 8) et un cri 35 % du temps sur le dernier temps — l'énergie
culmine pile sur le downbeat suivant. Le chorus tombe alors avec un impact, une batterie 4/4
pleine et un crash. Un chorus sur ~2,5 a aussi un drop au milieu (mesure 8, impact mesure 9).
Les moods `minimal`/`programming` (`mono_drums`) n'ont pas de drop : la boucle hypnotique ne
s'interrompt jamais.

**Transitions douces** : le mood suivant est tiré avec un poids favorisant les tempos voisins,
le même feel (half-time) et la même gamme ; le tempo reste au plus près du tempo courant dans la
zone du nouveau mood (±4 BPM) ; la tonalité est choisie parmi les tonics qui partagent le plus
de notes avec la tonalité courante (relatif, parallèle, même tonique) avec bonus si le dernier
accord du morceau est un accord pivot de la nouvelle tonalité ; la transition tient cet accord
pivot (`Harmony.pivot_chord`) avant la nouvelle progression.

Le chorus est annoncé sur ses deux dernières mesures d'approche (uplifter, cymbale reverse,
cri) et frappe avec un impact, une batterie 4/4 pleine (kick, snare + clap, hats) même en
mood half‑time, une basse en croches ou doubles, et un lead quasi systématique.

Effets d'insert choisis par l'arrangeur selon la section (`gate` hachure synchro tempo sur
pad/arp en chorus, `lofi` master en intro/break, `bitcrush` sur l'arp, et pour le lead en
chorus : `autopan`, gate + autopan, distortion + autopan, bitcrush, `phaser` ou flanger +
distortion) ; réglables à la main
avec `--fx` ou l'outil MCP `set_layer_effects`.

## Compositeur live — gestes sur les patches

L'arrangeur module les patches en direct (`BarPlan.tweaks`, multiplicateurs appliqués par
`apply_tweaks` sur le patch de base, sans couper les voix grâce à `Synth.update_patch`) :

- **Gestes de section** (`_GESTURES` dans `arranger.py`, tirés par couche avec une probabilité
  par section : chorus 0.55, intro 0.5, verse 0.45, outro 0.4, break 0.25) : cutoff ×0.6–1.5,
  résonance ×1.8–2.5, LFO rate/amount, attaque/decay d'enveloppe, détune ×1.5–1.6, glide,
  mix du premier effet, PWM…
- **Build-up** : sur les 4 mesures avant un chorus, le cutoff de pad et arp s'ouvre de ×0.55 à
  ×1.6 et la résonance du pad monte (×1 → ×2.2) ; drop : basse ×0.5.
- **Break** : pad et basse assombris (×0.7). **Intro** : pad qui s'ouvre de ×0.5 à ×1.
- Les réglages manuels (`set_patch_param`, panneau Tweak de l'UI) vivent sous les gestes et
  sont conservés. `set_auto_tweaks(false)` (MCP), `POST /api/auto_tweaks`, bouton
  « gestes auto » dans l'UI : coupe les gestes. L'UI affiche les gestes actifs par couche
  (chips ambre « cutoff ×0.7 ») et `status()["tweaks"]` les expose.

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

### DX7 6-op fidèle (`synthwave/engine/dx7.py:13`)

Moteur FM 6 opérateurs `sine` + 32 algorithmes Yamaha (flags `IN/OUT/FB/ADD`, bus 0/1/2, `isCarrier`), feedback `0..7` 1-sample, 6× EG 8-param `R1-4/L1-4 0-99` (ou ADSR fallback), `ratio` `coarse+fine` via `coarsemul`, `detune -7..8`, `velocity`/`level scaling`. Rendu vectorisé `numpy` par bloc 1024 (`phases + inbuf`, `sin(2π(phase+mod))·env`, `tanh` DAC). Import bulk `F0 43 00 09 20 00` 4104 B / raw 4096 B / single 155 B via `dx7_sysex_to_patches()` + CLI `synthwave import-dx7 bank.syx`. Patches `kind: dx7` (`algorithm 1-32`, `feedback`, `operators[6]`, `volume`, `effects`) — 550 DX7 dans `patches/library/dx7_*.yaml` (ex. `dx7_bass` `alg1 fb5` Lately Bass, `dx7_epiano` `alg5 fb4`, `dx7_bell` `alg4`). Volumes calibrés `0.70→0.574` (−1.8dB, `0.525` leads) + trim renderer `0.85` (`0.72` sur `lead/lead2`) pour équilibrer vs `LAYER_TRIM lead 2.5`.


### Solina String Ensemble fidèle (`synthwave/engine/solina.py`)

Émulation du circuit de l'Eminent/ARP Solina String Ensemble (1974), la « string machine » de
*Wish You Were Here*, *Oxygène* (Eminent 310, même circuit), *Moon Safari*, *Dream Weaver*,
*Video Killed the Radio Star*. Le Solina n'a **aucune mémoire de patch** : un patch YAML
`kind: solina` = les 6 boutons de registre + les faders.

| Élément du circuit | Émulation |
|---|---|
| Générateur | 1 horloge maître → top-octave (C8–B8) → diviseurs `2**k` (`note_hz`) : toutes les touches sont **verrouillées en phase** (zéro dérive entre voix, d'où la nécessité de l'ensemble). Onde « dent de scie en escalier » 16 niveaux (`staircase_saw`, compteur 4 bits = somme d'octaves de carrés), passe-bas 9 kHz anti-alias. |
| Keyers | `RcKeyer` : une enveloppe RC attaque/release **par touche** (49 touches C2–C6, MIDI 36–84 ; les notes hors clavier sont repliées par octave). `crescendo` = τ attaque (0.005–2 s), `sustain_length` = τ release (0.05–4 s). Ni decay ni sustain ; re-déclenchement pendant le release = reprise du niveau courant. Le crescendo n'agit pas sur trumpet/horn (attaque fixe 8 ms). |
| Registres haut | `viola` 8', `violin` = viola +1 octave (4'), `trumpet` 8' cuivré, `horn` = trumpet filtré plus sombre (**Horn prime sur Trumpet**). Filtres formants fixes appliqués sur le bus sommé de chaque registre (paraphonique, RC passifs sur l'original) : viola HP 300 Hz + bosse 1 kHz, violin HP 600 + bosse 2.5 kHz, trumpet HP 200 + bosses 1.2/3.2 kHz, horn LP 900 + bosse 500 Hz. |
| Section basse | `cello` 8' et `contrabass` 16' (cello −1 octave), **monophonique** (note la plus grave tenue), uniquement ≤ `split_note` (défaut 55 = G3, les 20 touches graves), keyer dédié, fader `bass_volume`, **hors ensemble**. Cello LP 1.2 kHz + bosse 250 Hz, contrabass LP 600 Hz. |
| Ensemble | `SolinaEnsemble` : 3 lignes BBD ≈ 5 ms **100 % wet** (aucun signal sec), modulées par deux LFO 3-phases — chorus 0.6 Hz ±1.5 ms et vibrato 6 Hz ±0.15 ms — chaque ligne décalée de 120°. Passe-bas 1 pôle 6 kHz avant/après (bande passante des TCA350) + `tanh` doux. `stereo: true` (défaut) panne les 3 lignes G/C/D — l'original est mono, `stereo: false` restitue la somme mono. Aussi disponible comme effet `ensemble` sur n'importe quel patch. |

Sources : Wikipedia « ARP String Ensemble », Sound On Sound « Eminent 310 », jhaible « Triple Chorus », KVR « solina ensemble effect », guide Zoe Blade, manuel Arturia Solina V, guide Behringer Solina.

### Roland D-50 — Linear Arithmetic (`synthwave/engine/d50.py`, `synthwave/engine/d50_pcm.py`)

Émulation de l'architecture LA du D-50 (1987) : un patch = 2 **tones** (upper / lower, key mode
WHOLE / DUAL / SPLIT), un tone = 2 **partials** + bloc commun, 7 **structures** (1 `S+S`, 2 `S×S`,
3 `P+S`, 4 `P×S`, 5 `S×P`, 6 `P+P`, 7 `P×P` ; `×` = ring modulator, sortie `P1 + P1·P2`).
Patches `kind: d50` en valeurs panneau entières, import sysex sans perte :
`synthwave import-d50 bank.syx` (bulk dump 64 patches × 448 octets, checksums vérifiés,
`volume` calibré automatiquement sur un accord).

| Élément | Émulation |
|---|---|
| Partial synthé (LA32) | Pas de filtre : carré à flancs cosinus dont la demi-largeur `w = f / (2·cutoff)` s'élargit quand le cutoff baisse (sinus quand cutoff ≤ f, atténuation en dessous), résonance = sinus amorti à la fréquence de coupure relancé à chaque cycle, **saw = rampe à 2f** (une octave au-dessus du carré, comme sur la machine ; le bank d'usine est écrit autour de ça). PW 0–100, PWM par LFO. `cutoff_hz = f_C4 · 2^((cutoff−50)/8 + KF·(note−60)/12 + bias + env·depth + LFO)`. |
| Partial PCM | Lit une des 100 ondes selon la règle du D-50 : `f × 2048` mots/s (une boucle de L mots sonne à `f·2048/L`, un one-shot joue à vitesse native pour f ≈ C0). Les ondes ROM étant copyrightées, elles sont **synthétisées** par famille (`d50_pcm.py`) avec les longueurs de la table reconstruite : 1–47 transitoires (maillets, pianos, cordes pincées, percussions, souffles, cuivres, archets), 48–76 boucles (orgues, EP, clavi, basses, cordes, sax, voix `Aah/Ooh/Male`, Spectrum 1–7, Noise), 77–100 boucles composites. |
| Enveloppes | TVF / TVA : T1–T5 `0,004·20000^(v/100)` s (4 ms → 80 s), L1–L3, sustain, end ; TVA linéaire en dB, TVF linéaire. P-ENV : T `0,009·1000^(v/50)` s, niveaux ±1 / 1,5 / 2 octaves selon `penv_velo`. Keyfollow temps, vélocité (niveau, temps 1, profondeur TVF). |
| LFO | 3 par tone, TRI / SAW / SQU / RND, rate `0,0004·67500^(v/100)` Hz, delay `10·(v/100)²` s. Vibrato ±600 cents × loi de profondeur Roland (double tous les 10 pas). Routage `+1 −1 +2 −2 +3 −3` vers pitch, PW, TVF, TVA. |
| Bloc commun | EQ low shelf (16 fréquences) + peak (22 fréquences, 9 Q) ; chorus 8 types (chorus 1/2, flanger 1/2, feedback chorus, tremolo, chorus-tremolo, dimension) ; mute et balance des partials. |
| Patch | Key mode, split, key shift / tune par tone, balance des tones, volume ; **reverb 32 types** mappés sur `Reverb` / `Delay` / `GatedReverb` avec les temps Roland (halls, rooms, delays 102–338 ms, cross delays, gate 200/480 ms, reverse gate, slap back, Twisted Space, Space). |

Polyphonie 16 (8 en dual / split), vol de la plus ancienne. Hors périmètre : aftertouch, bender, portamento, modes mono `-S`, chargement de ROM.

Sources : guide de paramètres Roland D-05, notes de service (plages temporelles), Wikipedia, modèle LA32 documenté par munt, format sysex vérifié par D50SysexBinConverter et d5_syx_to_patches.py, banks ROM Roland PN-D50-00…04 (bobbyblues.recup.ch, cultofd50).

---

## Effets — référence complète

13 effets bloc-vectorisés (`synthwave/engine/effects.py:392`). Paramètres temps synchronisés au BPM via `note_to_seconds` (`1/4`, `1/8`, `1/8d` dotée, `1/8t` triolet, ou secondes flottantes). Empilables dans `effects:` d'un patch, surchargeables par section par l'arrangeur, et par `--fx` / `set_layer_effects` (par couche ou `master`).

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
| 13 | `ensemble` | `SolinaEnsemble` | `chorus_rate=0.6` Hz, `chorus_depth=0.0015` s, `vibrato_rate=6.0` Hz, `vibrato_depth=0.00015` s, `base_delay=0.005` s, `stereo=true`, `bandwidth=6000` Hz | Triple BBD du Solina : 3 lignes 100 % wet, deux LFO 3-phases à 120°, passe-bas BBD, mono interne pannée G/C/D. | Moteur Solina (`kind: solina`), utilisable sur tout patch |

Notes :

- `note_to_seconds` : `"1/16"`, `"1/8d"` (= ×1.5), `"1/8t"` (= ×2/3), `"1/4"` etc. → `4 * (60/bpm) * num/den * mult`.
- Chaînes : `master` et par-couche s'additionnent ; `manual_fx` remplace `auto_fx` de l'arrangeur ; `_rebuild_inserts` reconstruit à chaque changement de BPM.
- DrumKit `perc_effects` = chaîne appliquée à toutes les percu *sauf* le kick ; RiserKit n'a pas d'inserts.

## Patches YAML

Bibliothèque : `synthwave/patches/library/` (646 patches — 96 synthés soustractifs + 550 DX7 6-op, volumes DX7 calibrés `0.70→0.574` + trim renderer `0.85/0.72` lead). Les fichiers placés dans
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

Patch DX7 (`kind: dx7`) — 6-op FM fidèle, import `.syx` via `synthwave import-dx7 bank.syx` :
```yaml
name: dx7_epiano
kind: dx7
algorithm: 5        # 1..32 (flags IN/OUT/FB)
feedback: 4         # 0..7
polyphony: 8
volume: 0.5         # calibré 0.574 bulk, 0.525 leads
operators:
  - {ratio: 1.0, level: 0.88, attack: 0.005, decay: 0.25, sustain: 0.7, release: 0.4} # EG ADSR ou dx7 R/L
  - {ratio: 14.0, level: 0.42, eg_type: dx7, eg_rate1: 99, eg_level1: 99, eg_rate2: 70, eg_level2: 60}
effects:
  - {type: reverb, size: 0.75, mix: 0.25}
```

Patch Solina (`kind: solina`) — boutons + faders, voir « Solina String Ensemble fidèle » :
```yaml
name: solina_wywh
kind: solina
registers: {violin: true, viola: true, trumpet: false, horn: false, cello: true, contrabass: false}
crescendo: 1.2        # τ attaque cordes (s), 0.005–2 ; sans effet sur trumpet/horn
sustain_length: 3.0   # τ release (s), 0.05–4
ensemble: true        # triple BBD 100 % wet (section basse hors ensemble)
stereo: true          # false = somme mono fidèle
bass_volume: 0.8      # cello/contrabass (0–1.5)
split_note: 55        # dernière touche de la section basse mono (G3)
tune: 0.0             # cents
volume: 0.42          # calibré RMS ≈ pad_strings
effects:
  - {type: reverb, size: 0.9, damping: 0.5, mix: 0.3, predelay: 0.03}
```

Patch D-50 (`kind: d50`) — valeurs panneau entières, voir « Roland D-50 » ; import `synthwave import-d50 bank.syx` :
```yaml
name: d50_Fantasia
kind: d50
key_mode: 1              # 0 WHOLE, 1 DUAL, 2 SPLIT (+ variantes)
split: 24                # C2 + n
reverb_type: 2           # 1..32 (Medium Hall)
reverb_balance: 38
tone_balance: 50
patch_volume: 100
upper:
  common: {structure: 1, chorus_type: 1, chorus_rate: 40, chorus_depth: 60, chorus_balance: 50,
           penv_t: [0, 0, 0, 0], penv_l: [0, 0, 0, 0, 0], lfos: [{wave: 0, rate: 74, delay: 0, sync: 0}, {}, {}],
           partial_mute: 3, partial_balance: 50}
  partials:
    - {wave: saw, coarse: 24, fine: -5, keyfollow: 11, cutoff: 52, resonance: 0, cutoff_kf: 9,
       tvf_env_depth: 52, tvf_env: {t: [17, 55, 44, 34, 88], l: [90, 62, 70], sustain: 44, end: 0},
       tva_level: 100, tva_env: {t: [37, 58, 45, 78, 56], l: [64, 90, 100], sustain: 88, end: 0}}
    - {wave: saw, coarse: 24, cutoff: 48, tva_level: 100}
lower:
  common: {structure: 6}
  partials:
    - {pcm: 13, coarse: 54}   # Bells (one-shot)
    - {pcm: 69, coarse: 19}   # Spectrum 2 (boucle)
volume: 0.45
```

Patch batterie (`kind: drums`) : `kick` (pitch, `sub`, `drive` saturation, `gain`), `snare`
(gated reverb, `gain`), `hat`, `clap`, `tom`, `crash_gain`, et `perc_effects` : chaîne
d'effets appliquée à toutes les percussions sauf le kick (écho ping‑pong + reverb dans
`drums_dark.yaml`). Voir `drums_808.yaml`.

### Bibliothèque livrée (724 patches — 96 soustractifs + 550 DX7 + 14 Solina + 64 D-50)

| Catégorie | Patch | Caractère |
|---|---|---|
| **Pad** | `pad_juno` | Juno saw 3×14 cts, chorus+reverb |
| | `pad_dark` | Saw 4×10 cts + FM, cutoff 520 Hz sombre |
| | `pad_strings` | Saw 5×12 cts + octave, attaque lente, chorus |
| | `pad_glass` | FM 2.0×0.6 + triangle, tremolo, delay 1/4 |
| | `pad_choir` | Square PWM LFO + triangle, chorus large |
| | `pad_warm` | Triangle unison + sine sub, LP 900 Hz, LFO lent |
| | `pad_shimmer` | Saw + octave sup., HP 400 Hz, phaser 6 stages, reverb 1.0 |
| | `pad_prophet` | Prophet-5 : saw + pulse PWM, filtre résonant enveloppé, sans chorus |
| | `pad_jupiter` | Jupiter-8 : saw unison + pulse, chorus, sine sub |
| | `pad_obx` | OB-Xa : 2 saws désaccordées larges, attaque courte, brass |
| | `pad_cs80` | CS-80 : saw + square, attaque lente, env filtre 2.5 s, delay |
| | `pad_polysix` | Polysix : saw seule, double chorus ensemble |
| | `pad_piano_atmos` | Piano « atmos » feutré : sine + FM 2.0 index 0.18 + triangle octave, LP 1300 Hz key-track, dérive 0.015 st, delay 1/4d, reverb 0.96 |
| **Bass** | `bass_moog` | Saw + square sub, moog ladder-style |
| | `bass_dark` | Saw + FM 0.5×0.6 + sine sub, LFO cutoff 0.5 Hz |
| | `bass_acid` | Saw résonante, env 1200 Hz, overdrive |
| | `bass_sub` | Sine pur + triangle, LP 900 Hz |
| | `bass_pulse` | Square PWM 0.35 unison, LFO cutoff |
| | `bass_reese` | Saw 3×22 cts, LFO cutoff 0.3 Hz |
| | `bass_growl` | FM 1.0×1.6 + LFO amp 3.5 Hz, disto 3.5 |
| | `bass_industrial` | Saw unison + FM 2.0×1.2, env 400 Hz, disto 6.0 |
| | `bass_fm` | FM 1.0×2.2 percussif + sine sub |
| | `bass_square` | Square + square sub, LP 520 Hz, classique |
| | `bass_pluck` | Saw decay court, env 900 Hz, delay 1/8 |
| | `bass_wobble` | Saw + square, LFO cutoff 2 Hz triangle |
| | `bass_808` | Sine longue, glide 0.06, saturation douce |
| | `bass_prophet` | Prophet : saw + pulse sub, LP 380 Hz |
| | `bass_sh101` | SH-101 : saw + sub square, résonance 0.3, decay court |
| | `bass_ms20` | MS-20 : résonance 0.45, disto 3.5 |
| | `bass_dx7` | DX7 « Lately Bass » : FM 1.0×3.0 + 0.5×1.0, sans glide |
| | `bass_dub` | Dub sub Basic Channel : sine + triangle, LP 180 Hz, attaque douce — hypnotique |
| | `bass_thump` | Club minimal dry : square PWM 0.25 + sine sub, env 900 Hz, punchy |
| | `bass_juno60` | Juno-60 PWM doux : square PWM 0.4 unison + saw, LFO cutoff 0.35 Hz |
| | `bass_soft` | Triangle soft non-fatiguant : triangle + sine sub, LP 320 Hz |
| **Arp** | `arp_pluck` | Saw 2×8 cts env 2200 Hz, delay 1/8d |
| | `arp_dark` | Square PWM 0.25 + saw, disto+phaser+delay |
| | `arp_saw` | Saw 3×12 cts, env 1800 Hz, delay 1/8d |
| | `arp_square` | Square PWM LFO + octave, delay 1/16 |
| | `arp_fm` | FM 2.0×1.2 court, phaser, delay 1/8 |
| | `arp_glass` | Triangle + sine octave, HP 500 Hz, chorus |
| | `arp_stab` | Saw 4×15 cts, disto + flanger, stab court |
| | `arp_jupiter` | Jupiter : pulse + saw, chorus, delay 1/8d |
| | `arp_sh101` | SH-101 séquencé : saw + sub, résonance 0.4 |
| | `arp_dx7` | DX7 : FM 1.0×1.8 + 3.0×0.5, chorus |
| | `arp_piano_atmos` | Piano « atmos » arpégé, delay 1/8d |
| **Lead** | `lead_saw` | Saw 4×18 cts, LFO vibrato 5.5 Hz, chorus+phaser |
| | `lead_dark` | Saw 3×16 cts + FM, vibrato 4.8 Hz, disto+phaser |
| | `lead_industrial` | FM 2.0×2.5 + saw, env 1800 Hz, disto 5.0 |
| | `lead_pulse` | Square PWM LFO, flanger+delay |
| | `lead_scream` | Saw 5×25 cts, vibrato 6 Hz, disto 7.0 + phaser 6 stages |
| | `lead_brass` | Saw 3×10 cts + sub, attaque 80 ms, env 1600 Hz |
| | `lead_hollow` | Square + triangle, chorus + phaser |
| | `lead_growl` | FM 1.5×3.0 + saw, LFO cutoff 3.5 Hz, disto 6.0 + flanger |
| | `lead_glide` | Glide 0.14, saw + square, autopan 1/1, delay 1/4 |
| | `lead_organ` | Sines harmoniques (8', 4', 2'2/3), tremolo 6 Hz, phaser |
| | `lead_prophet` | Prophet-5 : saw + pulse + sub, résonance 0.35, env 2400 Hz |
| | `lead_obx` | OB-Xa : saws larges + square, chorus, delay 1/4 |
| | `lead_cs80` | CS-80 « Blade Runner » : glide 0.1, vibrato 0.25, delay + reverb |
| | `lead_ms20` | MS-20 : résonance 0.55, disto 4.5 |
| | `lead_minimoog` | Minimoog : 3 osc (2 saw + square -1), glide 0.07, disto légère |
| | `lead_piano_atmos` | Piano « atmos » mélodique, polyphonie 4, longue queue |
| | `lead_jupiter` | Jupiter-8 : saw 3×12 cts + pulse, chorus stéréo, env 1700 Hz |
| | `lead_juno` | Juno-106 : DCO saw + sub square, PWM lente (LFO 0.5 Hz), chorus 0.5 |
| | `lead_dx7` | DX7 « solid lead » : 2 opérateurs FM (1.0×2.2, 2.0×2.5), key-track, pas de résonance |
| | `lead_sh101` | SH-101 : mono saw + pulse + sub, glide 0.09, résonance 0.45, env 2200 Hz |
| | `lead_odyssey` | ARP Odyssey : duophonique, pulse fine +7 st, FM sync, résonance 0.5, disto |
| | `lead_polysix` | Korg Polysix : saw 2×8 cts + sub, ensemble (chorus 0.6), rond |
| | `lead_arp2600` | ARP 2600 : pulse + saw, LFO carré 3.7 Hz sur cutoff (S&H), résonance 0.5, flanger |
| | `lead_d50` | Roland D-50 : attaque FM « cloche » + corps saw, polyphonie 2, réverbe 0.45 |
| | `lead_m1` | Korg M1 : square + sine octave, key-track 0.5, très net, delay 1/8d |
| | `lead_jx8p` | JX-8P : saws croisées, attaque 40 ms, chorus, cutoff 1200 Hz |
| | `lead_synthex` | Elka Synthex « Laser Harp » : pulse 12 % + saw, résonance 0.4, delay 1/4d fb 0.55 |
| | `lead_pro_one` | Pro-One : saw + pulse 25 % + FM, attaque 5 ms, résonance 0.4, disto |
| | `lead_memorymoog` | Memorymoog : 3 saw unisson 15 cts + sub, glide 0.08, ladder 900 Hz |
| | `lead_sem` | Oberheim SEM : filtre passe-bande 1100 Hz, nasillard, phaser |
| **Ambient** | `ambient_drone` | Sine+triangle+noise, LFO cutoff, chorus+reverb |
| | `ambient_dark` | Noise+FM+sine, LFO triangle 0.05 Hz, reverb 1.0 |
| | `ambient_wind` | Bruit BP 600 Hz résonant, LFO cutoff, autopan lent |
| | `ambient_shimmer` | Saw octave sup. unison, delay 1/2, reverb 1.0 |
| | `ambient_deep` | Sine -2 oct. + FM 0.25, LP 300 Hz |
| | `ambient_rain` | Bruit HP 2500 Hz, tremolo, bitcrush + lofi |
| | `ambient_choir` | Triangle 4×10 cts + square, flanger + chorus |
| | `ambient_tape` | Cassette : bourdon triangle + lofi 10 bits, wobble de bande 6 ms |
| | `ambient_glass` | Verre : FM 3.0×0.8 une octave au-dessus, HP 900 Hz, delay 1/2 |
| | `ambient_ocean` | Océan : bruit passe-bande balayé à 0.045 Hz, autopan 0.04, ressac |
| | `ambient_neon` | Néon : saw 4×14 cts, phaser 6 étages à 0.06 Hz, halo brillant |
| | `ambient_vhs` | VHS : souffle + FM 0.5×0.6, bitcrush 7 bits, flanger fatigué |
| | `ambient_bells` | Cloches lointaines : FM 3.5 + 7.0 inharmoniques, reverb 0.75 |
| **Drums** | `drums_808` | 808 sec, hats 8500 Hz |
| | `drums_dark` | Kick grave 140→40 Hz, hats feutrés, delay 1/8d + reverb perc |
| | `drums_linn` | LinnDrum : kick court, snare 210 Hz sèche, hats 9 kHz |
| | `drums_lofi` | Hats 5.5 kHz, `lofi` 8 bits ×4 sur les percussions |
| | `drums_industrial` | Kick drive 5.0, disto + bitcrush + delay 1/16 perc |
| | `drums_tight` | Decays courts, reverb quasi nulle, punchy |
| | `drums_hall` | Gated reverb longue (hold 0.5), delay 1/4 perc |
| | `drums_acoustic` | Kick propre (drive 1.0, battant 0.8, 120→55 Hz), snare tonique, reverb légère |
| | `drums_soft80` | Kick doux 80s (drive 1.1, battant 0.5), gated reverb moyenne |
| | `drums_breaks` | Kick vintage court (drive 1.0, battant 0.9), lofi 10 bits sur les percussions |
| | `drums_dmx` | Oberheim DMX / LinnDrum : sons purs et secs, snare peu réverbérée, aucun delay |
| **DX7** | `dx7_bass` | Lately Bass `alg1 fb5` 6-op — punchy FM |
| | `dx7_epiano` | EP FM `alg5 fb4` chorus+reverb — Rhodes FM |
| | `dx7_bell` | Cloche `alg4` inharmonique 3.5/7.0 — Bell FM |
| | `dx7_*` (547) | Banks Dexed `/homepages.abdn.ac.uk/d.j.benson` importés `import-dx7` (32 algs fidèles, bulk 4104) |
| **Solina** | `solina_strings` | Violin + viola, crescendo 0.35 / sustain 1.2 — le son « classique » |
| | `solina_full` | Les 6 boutons (horn prime sur trumpet), « all buttons in » |
| | `solina_wywh` | Violin + viola + cello, crescendo 1.2 / sustain 3.0 + reverb — nappes lentes *Wish You Were Here* |
| | `solina_oxygene` | Violin + viola + contrabass, crescendo 0.6 / sustain 2.0 + reverb — Jarre, Eminent 310 |
| | `solina_moon_safari` | Viola + cello, crescendo 0.2 / sustain 1.0 — Air, cordes douces médium |
| | `solina_dream_weaver` | Violin + viola + horn, crescendo 0.5 / sustain 1.5 — Gary Wright, mur de cordes + cuivre |
| | `solina_radio_star` | Viola + trumpet, crescendo 0.05 / sustain 0.3 — Buggles, stabs courts |
| | `solina_cello` | Cello (mono grave) + viola, `bass_volume` 1.0 |
| | `solina_contrabass` | Contrabass + cello, section basse seule, `bass_volume` 1.0 |
| | `solina_violin` / `solina_viola` / `solina_trumpet` / `solina_horn` | Registres seuls (0.15 / 0.6 cordes, 0.05 / 0.4–0.5 cuivres) |
| | `solina_dry` | Violin + viola sans ensemble, mono — son brut des diviseurs, organ-like |
| **D-50** | `d50_Fantasia` | Le preset 1-1 : saws LA32 (upper) + Bells / Spectrum 2 PCM (lower), reverb Medium Hall |
| | `d50_DigitalNativeDance` | Structure PCM rythmique, LFO carrés, chorus |
| | `d50_Staccato_Heaven` | Pizz + saws courtes, gate reverb — arp |
| | `d50_Pizzagogo` | Pizzicato PCM + synthé, delay — arp (Enya) |
| | `d50_Soundtrack` / `d50_Spacious_Sweep` / `d50_Glass_Voices` / `d50_Nightmare` / `d50_Future_Pad` | Nappes et textures — pools pad / ambient |
| | `d50_*` (64) | Bank ROM Roland PN-D50-00 importé par `import-d50` (volumes calibrés RMS ≈ 0.1) |

---

## Batterie synthétisée (`synthwave/engine/drums.py:31`)

Chaque hit est rendu une fois à l'init en buffer stéréo, puis mixé à la demande (polyphonie par superposition, `closed hat choke`).

| Pièce | Synthèse | Paramètres patch |
|---|---|---|
| `kick` | Sine pitch-drop exponentiel `pitch_start→pitch_end` + sub sine + click noise 2 ms + battant feutré (`beater`, bruit LP 900 Hz 20 ms) + `tanh(drive)` (`drive: 1.0` = propre, acoustique ; 2+ = techno) | `pitch_start/end/decay`, `click`, `sub/sub_decay`, `drive`, `beater`, `gain` |
| `snare` | Sine `tone` + bruit HP 1800 Hz, enveloppes séparées, `GatedReverb` | `tone/tone_decay/noise_decay`, `gate_hold`, `reverb_size/mix`, `gain` |
| `clap` | Bruit BP 1500 Hz + 4 bursts 11 ms + tail, `GatedReverb` | `decay`, `gate_hold`, `reverb_mix`, `gain` |
| `hat` closed/open | Bruit HP `cutoff`, decays exponentiels | `closed_decay/open_decay`, `cutoff`, `gain` |
| `tom` low/mid | Sine pitch-bend `1+0.6*exp(-t/0.04)` | `pitch_low/mid`, `decay`, `gain` |
| `crash` | Bruit BP 6000 Hz, decay 0.7 s | `crash_gain` |
| `snap` | Claquement de doigts : 2 bursts BP `tone` à 6 ms + thump 320 Hz, gated reverb courte | `tone`, `decay`, `body`, `reverb_mix`, `gain` |
| `ride` | Bruit HP + 4 partiels inharmoniques (3.15/4.72/6.39/7.81 kHz), decay long | `decay`, `cutoff`, `ping`, `gain` |
| `shaker` | Bruit BP attaque 4 ms, decay court | `decay`, `cutoff`, `gain` |
| `tick` | Charley fermé sec « tic tic » : bruit HP `cutoff`, decay 12 ms | `decay`, `cutoff`, `gain` |
| `crash_roll` | Roulement de cymbale aux mailloches : bruit BP 5.5 kHz, coups ~14/s, crescendo `t^2.2` sur **une mesure** (resynthétisé à chaque BPM) puis queue de crash | `crash_roll_gain` |
| `perc_effects` | Chaîne post-mix hors kick | `perc_effects: [delay, reverb, …]` |

Notes MIDI : `kick 36`, `snare 38`, `clap 39`, `snap 40`, `hat_closed 42`, `hat_open 46`, `tom_low 45`, `tom_mid 47`, `tick 44`, `crash 49`, `ride 51`, `crash_roll 57`, `shaker 70`.

**Groove droit années 80** (`Mood.straight`, moods `dreamy`, `outrun`, `chill`, `retro`, `drive`, `minimal`, `programming`) : la
grille ne bouge jamais d'une mesure à l'autre — kick sur les quatre temps (sans kick syncopé
aléatoire), snare **et** clap superposés sur 2 et 4, charleys fermés sur les croches (doubles en
chorus), open hat sur 14, aucun roll en milieu de section, fill de fin de phrase = quatre doubles
de snare crescendo sous un kick qui ne s'interrompt pas. Les pools de ces moods n'utilisent que des
kits secs (pas de `delay` sur le bus percussions : `drums_808`, `drums_linn`, `drums_dmx`,
`drums_tight`, `drums_acoustic`, `drums_soft80`). `minimal`/`programming` ajoutent `mono_drums` : une seule voix par section (`kick xor snare xor hat`, pas de rolls/fills/crash, pas de drop, risers réduits à l'impact). Les moods sombres (`dark`, `noir`, `cyberpunk`,
`horror`, `desert`) gardent le groove génératif avec kicks syncopés, rolls et kits humides.

Le flag `straight` ne touche pas que la batterie : la basse joue des notes détachées en ostinato, l'arpège garde la même figure sur un accord donné, et la mélodie est jouée staccato (notes courtes séparées par un silence) au lieu d'être tenue jusqu'à la note suivante. Mesuré sur 400 mesures : durée moyenne d'une note de lead 2.0 pas et 98 % des notes détachées en `outrun`, contre 4.9 pas et 19 % en `dark`.

Couleurs de percussion par mood (tirées par section) : `snap_prob` (claquements de doigts sur le backbeat, remplacent la snare hors chorus, s'y superposent en chorus), `ride_prob` (ride à la place des charleys sur les croches), `shaker_prob` (shaker en doubles). chill 0.8/0.4/0.6, dreamy 0.4/0.3/0.3, retro 0.35/0.5/0.2, noir 0.5/0.3/0, desert 0.2/0/0.5, outrun ride 0.3, drive ride 0.35. `tick_prob` (défaut 0.5, chill/retro 0.8) : charley fermé sec sur toutes les croches à la place des hats bruités. Charleys : croches accentuées + off-beats selon la densité, open hat sur 6/10/14, gains ×1.45 sur tous les kits. Cymbales : crash au début de verse/chorus/break, 60 % aux mesures 9 et 13 du chorus ; **roulement de cymbale** 40 % sur les fills de fin de section (la mesure de drop, elle, est vide de batterie : seuls les risers la remplissent).

## Risers & transitions (`synthwave/engine/risers.py:32`)

One-shots resynthétisés à chaque changement de BPM (durées en bars) :

| Note | Nom | Synthèse | Durée |
|---|---|---|---|
| 60 | `reverse_cymbal` | Bruit LP sweep 800→9000 Hz, enveloppe `exp((t-1)*5)` | 1 bar |
| 64 | `reverse_short` | Moitié du reverse, fade `0.3→1.0` | 0.5 bar |
| 61 | `uplifter` | Saw montante 1 octave + bruit, LP sweep 300→12000 Hz, `t^1.5` | 2 bars |
| 62 | `scream` | Sirène additive band-limited : 6 harmoniques 1/k, 220→880 Hz + vibrato 6 Hz, LP 900→6000 Hz, `tanh(1.4)` | 1 bar |
| 63 | `impact` | Sub 45→105 Hz + burst LP 6000→200 Hz, `tanh(2)` | 1.2 s |

Annonces arrangeur (`synthwave/composer/arranger.py:219`) : `uplifter` à J-2 du chorus, `reverse` + `scream` 35% à J-1, `reverse_short` 50% en fin de section non-chorus, `impact` au downbeat du chorus.

## Mix, sidechain & master (`synthwave/audio/renderer.py:25`)

- **Sidechain** (`engine/effects.py:209`) : ducking `gain = 1 - depth*exp(-t/release)` déclenché sur chaque kick, `depth 0.45 / release 0.22 s`. Atténuation par couche : `pad 100%`, `bass 60%`, `ambient 60%`, `arp 50%`.
- **Gains de section** : `intro drums 0.6/lead 0.0` → `verse` → `chorus full` → `break drums 0.85/bass 0.85/lead 0.6` → `drop tout à 0 sauf pad 0.35, ambient 0.5, riser` → `transition drums/bass/arp/lead 0.0` → `outro` avec `fade` linéaire sur 8 bars.
- **Trim par couche** (`audio/renderer.py:26`) : `LAYER_TRIM = {"lead": 2.5}` + `DX7 ×0.85` (`×0.72` sur `lead/lead2`) — DX7 6-op plus présent, volumes calibrés `0.574` (`0.525` leads) pour équilibrer.
- **Master** : rampe de gain par couche (`current→target` sur un bloc, appliquée **avant** les effets de la couche pour que les queues de delay/reverb continuent de sonner) → inserts `master` (auto/manuels) → `master_volume 0.7` × `fade` linéaire → **couleur master** → `Limiter threshold 0.95`.
- **Couleur master** (`MASTER_COLORS`, `--master-color`, MCP `set_master_color`, `POST /api/master_color`) : étage de saturation permanent placé après le volume, donc toujours attaqué au même niveau. **Choisie par la composition** par défaut (`auto`) : une couleur tirée par morceau selon la luminosité du mood (moods clairs `tape`/`clean`, moods sombres `vhs`/`mic`), puis déplacée d'un cran sur l'échelle `clean < tape < vhs < mic < crush` selon la section — intro plus sale (50 %), chorus plus propre (40 %), break sur bande usée. Une valeur donnée à la main (`--master-color vhs`, MCP, UI) verrouille le choix ; `auto` rend la main. `clean` (aucun), `tape` (défaut : saturation douce + bitcrush 13 bits), `vhs` (bande usée : disto + lofi 10 bits, wobble, souffle), `mic` (micro saturé : disto drive 3.2, bande 4.5 kHz, lofi 11 bits), `crush` (bitcrush 8 bits + disto). Le `lofi` n'y apparaît qu'en mix 1.0 : son chemin humide est retardé par le wobble, un mix partiel filtrerait le master en peigne. Mesuré sur un rendu `outrun` de 15 s :

| Couleur | Crête | RMS | Énergie > 4 kHz | Distorsion (sinus 220 Hz) |
|---|---|---|---|---|
| `clean` | 0.791 | 0.149 | 41 % | 0 % |
| `tape` | 0.731 | 0.156 | 34 % | 1.8 % |
| `vhs` | 0.668 | 0.165 | 23 % | 2.6 % |
| `mic` | 0.620 | 0.184 | 18 % | 4.1 % |
| `crush` | 0.615 | 0.189 | 17 % | 9.1 % |

---

## Composition — toutes les variations implémentées

### Gammes — 8 échelles (`synthwave/composer/harmony.py:35`)

| Échelle | Degrés (demi-tons) | Moods |
|---|---|---|
| `minor` (naturel) | 0 2 3 5 7 8 10 | dreamy (65%), outrun, cyberpunk, drive, programming |
| `major` | 0 2 4 5 7 9 11 | dreamy/retro via `major_prob` |
| `phrygian` | 0 1 3 5 7 8 10 | dark |
| `harmonic_minor` | 0 2 3 5 7 8 11 | noir |
| `dorian` | 0 2 3 5 7 9 10 | chill, minimal |
| `mixolydian` | 0 2 4 5 7 9 10 | retro (50% majeur) |
| `locrian` | 0 1 3 5 6 8 10 | horror |
| `phrygian_dominant` | 0 1 4 5 7 8 10 | desert |

Tonalité `tonic` 0–11 tirée au hasard au démarrage. À chaque transition `change_key()` choisit le nouveau tonic parmi ceux qui partagent le plus de notes avec la gamme courante (+3 si le dernier accord est un pivot, +1 même tonique) puis `pivot_chord()` fournit l'accord tenu pendant la transition. `set_mood` applique `SCALES[mood.scale]` (ou `MAJOR` si `rng < major_prob`).

### Progressions — 39 progressions (`harmony.py:11`)

| Famille | Progression | Degrés | Usages (poids) |
|---|---|---|---|
| outrun/dreamy/drive | `i-VI-III-VII` | 0 5 2 6 | dreamy 3, outrun 3, drive 3 |
| | `i-VII-VI-VII` | 0 6 5 6 | dreamy 2, outrun 3, drive 3 |
| | `i-iv-VI-V` | 0 3 5 4 | dreamy 1, outrun 2 |
| | `i-VI-VII-i` | 0 5 6 0 | dreamy 2, outrun 2, drive 2 (+cyberpunk 2) |
| | `VI-VII-i-i` | 5 6 0 0 | dreamy 2, outrun 2, drive 2 |
| | `i-III-VII-VI` | 0 2 6 5 | dreamy 2 |
| | `iv-VI-i-VII` | 3 5 0 6 | dreamy 1, outrun 1 |
| dark | `i-bII-i-bII` | 0 1 0 1 | dark 3, horror 2, desert 1, cyberpunk 2 |
| | `i-iv-bII-i` | 0 3 1 0 | dark 3, cyberpunk 2 |
| | `i-bVI-bII-i` | 0 5 1 0 | dark 2 |
| | `i-bvii-bVI-bII` | 0 6 5 1 | dark 2 |
| | `i-i-bVI-bII` | 0 0 5 1 | dark 2 |
| | `i-iv-i-bII` | 0 3 0 1 | dark 2 |
| noir | `i-VI-V-i` | 0 5 4 0 | noir 3 |
| | `i-iv-V-i` | 0 3 4 0 | noir 3 |
| | `i-i-VI-V` | 0 0 5 4 | noir 2 |
| + harmonic | `i-V-VI-V` | 0 4 5 4 | (réservé, pondération future) |
| | `iv-V-i-i` | 3 4 0 0 | — |
| mineur nat. étendu | `i-VII-III-VI` | 0 6 2 5 | drive 2 |
| | `i-v-VI-iv` | 0 4 5 3 | cyberpunk 2 |
| | `VI-i-VII-III` | 5 0 6 2 | drive 1 |
| | `i-III-iv-VI` | 0 2 3 5 | drive 1 |
| | `i-iv-i-VI` | 0 3 0 5 | cyberpunk 1 |
| | `i-VI-iv-VII` | 0 5 3 6 | cyberpunk 2 |
| majeur/mixo | `I-V-vi-IV` | 0 4 5 3 | retro 3 |
| | `vi-IV-I-V` | 5 3 0 4 | retro 3 |
| | `I-vi-IV-V` | 0 5 3 4 | retro 2 |
| | `IV-I-V-vi` | 3 0 4 5 | retro 1 |
| | `I-bVII-IV-I` | 0 6 3 0 | retro 2 |
| | `I-IV-bVII-IV` | 0 3 6 3 | retro 2 |
| | `I-v-bVII-IV` | 0 4 6 3 | retro 1 |
| dorien / minimal | `i-IV-i-IV` | 0 3 0 3 | chill 3, minimal 4, programming 4 |
| | `I-IV-I-IV` | 0 3 0 3 | minimal 3, programming 3 (alias majeur) |
| | `i-bVII-IV-i` | 0 6 3 0 | chill 2, programming 1 |
| | `I-IV-bVII-I` | 0 3 6 0 | minimal 1 (alias) |
| | `i-ii-bIII-ii` | 0 1 2 1 | chill 2 |
| | `i-IV-bVII-i` | 0 3 6 0 | chill 2, minimal 2, programming 2 |
| | `i-ii-IV-i` | 0 1 3 0 | chill 1 |
| tension | `i-bII-bv-i` | 0 1 4 0 | horror 3 |
| | `i-biii-bII-i` | 0 2 1 0 | horror 2 |
| | `i-bII-bVI-bv` | 0 1 5 4 | horror 2 |
| | `I-bII-I-bvii` | 0 1 0 6 | desert 3 |
| | `I-iv-bII-I` | 0 3 1 0 | desert 3 |
| | `I-bVI-bvii-I` | 0 5 6 0 | desert 2 |

Qualité : sans `sevenths` → triades ; avec → 7e empilées (tierces). Nommage `m/m7/maj7/7/dim/m7b5…`.
Markov : poids par mood, anti-répétition `×0.25` si même progression que précédemment.

### Patterns 16 pas (`synthwave/composer/patterns.py`)

| Couche | Générateur | Variations |
|---|---|---|
| `drums` | `gen_drums` + `add_roll` / `add_straight_fill` | Groove tiré **une fois par section** puis figé (`strong` chorus 4/4 dense, `halftime` kick syncopé / snare beat 3, `density` hats off-beat, open hat 14, `crash` downbeat verse/chorus). Ornements superposés sans toucher le kick : **roll sur le 3** (4e mesure de chaque phrase, 30 %+densité×50 % : 4 doubles snare/toms crescendo sur 8–11, ou pickup 6–7 en half-time), **fill** en fin de section = groove + roll 12–15 (voix `snare`, `toms_down`, `toms_up`, `alternate`) |
| `bass` | `gen_bass` | 6 styles : `eighths` (croches), `octaves` (alterné), `sixteenths` (doubles + oct. aléat.), `walk` (root-5th-oct), `riff` (b2/triton chromatique), `syncopated` (0 3 6 8 11 14). + mutation 20% une mesure sur deux, octave pop fin de pattern 30%. **Moods droits** (`straight`) : notes détachées (durée 1 au lieu de 2), aucun ornement aléatoire, pas de mutation → ostinato répété à l'identique |
| `arp` | `gen_arp` | 3 modes : `up`, `updown`, `random` ; 2 octaves ; 16 doubles par bar. **Moods droits** : mode `random` exclu, la figure d'un accord revient identique à chaque mesure |
| `pad` | `gen_pad` | Tenue `STEPS` sur notes de l'accord, voicing grave si root≥6, octave ajoutée si triade |
| `ambient` | `gen_ambient` | Tenue root + 5th (octave 3) |
| `lead` | `gen_theme` + `render_motif` | **Thème par morceau** : motif *question* (2–5 notes sur la grille des croches, note d'accord sur les temps forts, pas ≤ tierce) + *réponse* (même rythme, contour retouché, résout sur la tonique) + *contre-mélodie* (contour inversé, notes longues). Rendu par accord en **séquence diatonique** (offsets en degrés depuis la fondamentale) avec ornements (`vary`) : question/réponse alternées par mesure en verse/chorus, octave sup. en 2e moitié de chorus, contre-mélodie systématique dans les breaks (variation 0.3, octave sup. en 2e moitié) et en 2e moitié de verse (30 %). Tessiture `lo=55` (dark/noir) sinon `60`. **Moods droits** : phrasé staccato — chaque note dure au plus 3 pas et laisse un silence avant la suivante (durée `min(3, écart-1)` au lieu de tenir jusqu'à la note suivante), ornements divisés par 2.5 : le thème revient tel quel |
| `lead2` | `harmonize` | Voix d'harmonie : le motif du lead transposé de −2 (tierce), −5 (sixte) ou −7 degrés diatoniques, tirage figé par section ; même rythme, vélocité ×0.75, notes bornées à la gamme. Muette dès que le lead se tait et pendant un drop. Patch propre (`pools["lead2"]` : voix douces), trim ×1.6 contre ×2.5 pour le lead |
| `riser` | `_risers` | Voir ci-dessus |
| `mutate` | `mutate` | Drop 30%, nudge ±1 step 30%, substitution note 40%, insertion libre 50% ; utilisé basse/drums/ambient |

### Sections & arrangeur (`synthwave/composer/arranger.py:38`)

| Section | Bars | Gains particuliers | FX auto possibles |
|---|---|---|---|
| `intro` | 8 | build-up : arp 2, kick 4, basse 6 | `master lofi` 50% ; couleur master d'un cran plus sale 50 % |
| `verse` | 16 | arp 0.85 pad 0.9 lead 0.35 lead2 0.25 ; arp à 2, lead à 4, lead2 à 8 ; drop en dernière mesure si chorus suit | `arp gate 1/32` 25% |
| `chorus` | 16 | full + lead 1.0 (à 2) + lead2 0.7 (à 4) ; drop au milieu 40 % | `pad gate 1/16-1/32` 35–55%, `arp bitcrush`, `lead` 6 pools (autopan / gate+autopan / disto+autopan / bitcrush / phaser 6 stages / flanger+disto) ; couleur master d'un cran plus propre 40 % |
| `break` | 8 | mêmes instruments que la section précédente (aucun changement de patch), drums 0.85 / bass 0.85 / arp 0.8, contre-mélodie lead 0.6 (à 1) + lead2 0.45 (à 2), drop si chorus suit | `master lofi` 60% sinon `pad gate 1/8` ; couleur master d'un cran plus sale |
| `transition` | 4 | drums/bass/arp/lead 0.0 pad 0.5 ambient 1.0, accord pivot | — (porte tonalité/tempo/mood) |
| `outro` | 8 | arp off à 2, kick seul à 4, basse+batterie off à 6 ; fade 1→0 en mode durée seulement | — |

Enchaînement : `intro→verse→(chorus 2×/break pondéré)→…→outro→transition→intro` ; l'outro arrive quand le morceau atteint sa durée (`--track`, sections raccourcies à un multiple de 4 pour tenir), après 8 sections, ou sur `set_mood`. Mode durée : outro final avec fondu quand `bar ≥ total_bars-8`. Basse/lead pool resélectionnés chaque section **sauf en break** (les instruments y sont conservés) ; `arp_on` selon `arp_prob` (0.45 dark → 0.95 outrun) ; `lead` proba `0.2–0.55` (≥0.7 en chorus) ; batterie figée par section, rolls en fin de phrase (voir `drums`).

## Interface web

`uv run synthwave ui` sert une page unique (Starlette + WebSocket, aucun build) sur
<http://127.0.0.1:8765> et ouvre le navigateur. Elle affiche en direct : section (avec DROP),
morceau et progression (mesures de la section / du morceau), tonalité, accord, tempo, mood et
mood en attente, oscilloscope du master, VU-mètre par couche et gain courant de l'arrangeur.
Contrôles live : Start (mood/BPM/seed/durée de morceau), Stop, section suivante, tempo,
changement de mood (via outro + transition), mute / solo / volume par couche, patch par couche
(liste filtrée par famille + 550 DX7 `dx7_*`, 14 Solina `solina_*` et 64 D-50 `d50_*` sur chaque couche hors `drums`, pastille jaune `DX7` / orange `SOLINA` / bleue `D-50` quand actif), panneau **Tweak** (sliders générés depuis le patch : filtre,
enveloppes, LFO, oscillateurs, kick/snare/hats…) et chaîne d'effets par couche ou master
(chips : clic = retirer, `+ effet` = ajouter avec des valeurs par défaut, `↺ auto` = rendre la
main à l'arrangeur).

API JSON (`synthwave/web/server.py`) : `GET /api/meta`, `GET /api/status`,
`GET /api/patch/{layer}`, `POST /api/start|stop|tempo|mood|layer|patch|patch_param|effects|next_section`,
flux `WS /ws` (statut + niveaux + oscilloscope, 12 Hz). Le player est partagé avec le serveur MCP
(`synthwave/session.py`).

## MCP

`.mcp.json` à la racine déclare le serveur pour Claude Code (lancé depuis le dossier du
projet ; pour un autre dossier, ajouter `--directory <chemin>` aux arguments). Outils :

| Outil | Rôle |
|---|---|
| `start(mood, bpm, seed, duration_s, bpm_range, track_s)` | démarre la lecture (`track_s` ≈ durée d'un morceau, 210 s) |
| `stop()` | arrête |
| `status()` | tempo, tonalité, accord, section, morceau (`track`, `track_bar`/`track_bars`), `drop`, couches |
| `set_tempo(bpm)` | 60–180 |
| `set_mood(mood)` | dark / dreamy / outrun |
| `set_master_color(color)` | clean / tape / vhs / mic / crush : couleur du master |
| `set_layer(layer, mute, solo, volume)` | mixage par couche |
| `list_patches()` / `load_patch(layer, name)` | patches |
| `set_patch_param(layer, path, value)` | ex. `filter.cutoff` 800 |
| `set_layer_effects(layer, effects)` | inserts manuels, `layer` ou `master`, `None` = auto |
| `set_auto_tweaks(enabled)` | active / coupe les gestes du compositeur live |
| `next_section()` | passe à la section suivante |
| `export_wav(path, seconds, mood, bpm, seed)` | rendu hors-ligne |

## Tests

```bash
uv run pytest -q
uv run ruff check .
```
