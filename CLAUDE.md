# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projet

Générateur procédural de synthwave infini (ou à durée fixe) : synthèse 100 % numpy/scipy, sortie
audio via sounddevice (PortAudio), CLI typer et serveur MCP (Model Context Protocol). Python ≥ 3.13,
géré par `uv`.

**Glossaire** : BPM (Beats Per Minute), FX (Effects), MCP (Model Context Protocol), FM (Frequency
Modulation), ADSR (Attack Decay Sustain Release), LFO (Low-Frequency Oscillator), RMS (Root Mean
Square), WAV (Waveform Audio File Format), UP (Update).
Le README (en français) est la référence exhaustive des moods, effets, patches et paramètres :
le tenir à jour à chaque nouveauté (nouvel effet, mood, gamme, progression, patch).

## Commandes

```bash
uv sync                                  # installer (dev inclus : pytest, ruff)
uv run pytest -q                         # suite complète (~110 tests, ~90 s : rendus audio réels)
uv run pytest -q tests/test_effects.py   # un fichier
uv run pytest -q -k "phaser"             # un test par mot-clé
uv run ruff check .                      # lint (E, F, I, UP, B, W ; line-length 100)
uv run ruff check . --fix                # tri des imports / corrections auto

uv run synthwave play                                        # lecture infinie, mood + tempo aléatoires
uv run synthwave play --duration 30s --mood dark --seed 1 --export out.wav   # rendu hors-ligne (.wav .flac .ogg .mp3)
uv run synthwave play --fx lead:phaser:rate=2/1 --fx master:lofi:bits=8      # inserts manuels
uv run synthwave patches | devices | mcp
uv run synthwave ui --no-browser                             # web UI http://127.0.0.1:8765
```

- `--export` exige `--duration`. Même seed + mêmes options ⇒ rendu bit-identique : utiliser
  `--seed` pour reproduire un problème d'écoute.
- Rendu hors-ligne ≈ 5× temps réel. Pour mesurer un niveau ou un timbre, exporter en WAV puis
  analyser avec numpy (crêtes, RMS, centroïde spectral) plutôt qu'écouter.
- Le serveur MCP est déclaré dans `.mcp.json` (`uv run synthwave mcp`) et activé pour Claude Code
  dans `.claude/settings.local.json` ; outils `start/stop/status/set_tempo/set_mood/...`.
- Jamais de chemin absolu machine (du type `/home/<user>/…`) dans les fichiers versionnés : le dépôt est
  public (github.com/fxrobin/infinite-synthwave) et l'historique a déjà été réécrit pour ça.

## Architecture

Chaîne par bloc audio (1024 échantillons par défaut) :

```
Transport (BPM → ticks de pas, 16 pas/mesure)
  └─ Tracker.advance(n) : au pas 0 de chaque mesure appelle Arranger.next_bar() → BarPlan
       └─ transforme les Pattern (Note step/note/vel/length) en NoteEvent on/off par couche
Renderer.render(n) : draine la file de commandes, applique le BarPlan (gains, mood, patches,
  bpm, fx auto), rend chaque couche (Synth | DrumKit | RiserKit), inserts, sidechain, rampe de
  gain, master fx, fade, Limiter
Player (audio/output.py) : thread producteur → queue → callback sounddevice (copie seule)
```

Couches fixes : `LAYERS = drums, bass, arp, pad, lead, ambient, riser` (`composer/arranger.py`).
`riser` est built-in (`RiserKit`, pas de patch YAML, one-shots resynthétisés à chaque BPM).

### Composition (`synthwave/composer/`)

- `moods.py` : `Mood` frozen dataclass = gamme, sevenths, halftime, bpm_range, densités,
  poids de progressions, jeu de patches, pools basse/lead (rotation par section), poids de styles
  de basse. `MOODS` dict = source de vérité des noms de moods (CLI/MCP valident dessus).
- `harmony.py` : `SCALES`, `PROGRESSIONS` (degrés 0–6), chaîne de Markov pondérée par mood,
  anti-répétition, `modulate()` à chaque transition.
- `patterns.py` : générateurs 16 pas par couche (`gen_drums/bass/arp/pad/ambient`), `mutate`,
  rolls/pre-drop, et la mélodie : `Theme` (question/réponse/contre-mélodie en offsets diatoniques)
  créé par `gen_theme` une fois par morceau, joué sur chaque accord par `render_motif`.
- `arranger.py` : machine à états `Section` par **morceau** (~`track_s` = 210 s) :
  intro→verse/chorus/break…→outro→transition→intro suivant. Build-up par tables `_ENTRY` /
  `_EXIT` / `_DRUM_LEVELS` (une couche entre toutes les 2 mesures) ; pre-drop (`BarPlan.drop`)
  sur la dernière mesure avant un chorus et au milieu d'un chorus ; la transition tient un accord
  pivot et porte tonalité/tempo/mood (`Harmony.change_key`, `pivot_chord`, `draw_bpm` proche du
  tempo courant). Choisit aussi les FX auto (`BarPlan.fx`), les patches de section
  (`BarPlan.patches`) et les risers (notes 60–64). `mood_locked` fige le mood (`--mood`).

### Moteur (`synthwave/engine/`)

- `Synth` (voix polyphoniques `Voice` : oscillateurs polyBLEP/FM, ADSR amp + filtre, biquad, LFO,
  glide) et `DrumKit` (hits pré-rendus à l'init, `perc_effects` sur tout sauf le kick) exposent
  la même interface : `render(n, events)`, `set_patch`, `set_bpm`.
- `effects.py` : `Effect.process(x)` sur blocs stéréo float32 ; les lignes à retard sont
  traitées par tranches ≤ délai. Un nouvel effet = classe `__init__(sr, bpm, **params)` +
  entrée dans `_REGISTRY` ; il devient alors utilisable dans les patches YAML, `--fx`,
  `set_layer_effects` et les pools FX de l'arrangeur. Paramètres temporels acceptent Hz/secondes
  ou notes tempo-sync (`"1/8"`, `"1/8d"`, `"1/8t"`) via `note_to_seconds`.

### Patches (`synthwave/patches/`)

- `model.py` : pydantic (`PatchModel`, `DrumPatchModel` avec `kind: drums`), bornes validées.
  Ajouter un paramètre = champ du modèle + prise en compte dans `Voice`/`DrumKit`.
- `loader.py` : YAML de `patches/library/` ; `~/.config/synthwave/patches/` prime sur la
  bibliothèque. `set_param(patch, "filter.cutoff", 800)` reconstruit un modèle validé.
- Un patch référencé par un `Mood` (`patches` ou `pools`) doit exister dans la bibliothèque,
  sinon `Renderer.__init__` lève `PatchError`. Tout patch de la bibliothèque doit être dans un
  pool (test `test_every_library_patch_is_in_a_pool`). Un nouveau patch : mesurer peak/RMS contre
  ses voisins de famille (pads/ambients sur 6 s à cause des attaques lentes) avant de fixer `volume`.

### Contrôle temps réel

`session.py` (`SESSION`) détient le Player live et sert de façade commune au serveur MCP et à
l'UI web (`web/server.py`, Starlette + WebSocket ; page unique `web/static/index.html`, JS
vanilla, pas de build). Le renderer expose `levels` (crête par couche + master) et `scope`
(256 points du dernier bloc) pour les vumètres et l'oscilloscope. Tester l'API avec
`starlette.testclient` (voir `tests/test_web.py`) : ça ne démarre pas d'audio.

Toutes les mutations depuis CLI/MCP passent par `Renderer.submit(fn)` : la file est drainée en
tête de `render()` sur le thread audio, les exceptions y sont attrapées (jamais de crash audio).
`set_mood` est différé : il programme une transition, `status()["pending_mood"]` l'indique.
`manual_fx` / `manual_patch` priment sur les choix auto de l'arrangeur ; `None` rend la main.
Tout changement de BPM reconstruit les inserts et les effets des patches (`_rebuild_inserts`,
`Synth.set_bpm`).
Deux niveaux de patch par couche : `base_patch` (chargé + éditions manuelles) et le patch
effectif = base × `auto_tweaks` (gestes de l'arrangeur, `BarPlan.tweaks`, appliqués par
`apply_tweaks` avec clamp). Un changement de paramètres passe par `Synth.update_patch` (voix
conservées, pas de clic) ; `set_patch` réinitialise les voix et n'est utilisé que pour un
changement de structure (oscillateurs, filtre, chaîne d'effets).

## Conventions de mix et tests

- Équilibre de mix : mesurer crêtes/RMS par couche en solo (`set_layer(solo=True)` ou rendu
  d'un `Synth` isolé) avant d'ajuster `volume` d'un patch ou `LAYER_TRIM` (`audio/renderer.py`).
- Un morceau ≈ 100 mesures : les tests de structure itèrent 200–700 `next_bar()` (rapide, pas de
  rendu audio) et ignorent les mesures `fill` / `drop` quand ils comparent des grooves.
- Les tests de rendu et d'arrangeur fixent toujours un `mood` et un `seed` : sans mood fixé,
  le mood est tiré au hasard à chaque transition et les assertions deviennent flaky.
- `mcp` est en 2.x : `from mcp.server.mcpserver import MCPServer` (pas `FastMCP`) ;
  `call_tool` renvoie `.structured_content` ou `.content[0].text` (voir `tests/test_mcp.py`).
- Docs de conception : `docs/superpowers/specs/` et `docs/superpowers/plans/`.
