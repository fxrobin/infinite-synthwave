# Émulation fidèle du Solina String Ensemble

Date : 2026-09-04. Statut : validé (design approuvé en discussion).

## Objectif

Ajouter un moteur `SolinaSynth` reproduisant le circuit de l'Eminent/ARP Solina String Ensemble
(1974) : générateur divide-down, keyers RC par touche, bus de registres filtrés, section basse
monophonique, ensemble triple BBD. Même interface que `Synth` / `Dx7Synth`, patches YAML
`kind: solina`, intégration renderer / loader / UI / moods / README, plus un bank de patches
encodant les réglages célèbres documentés (le Solina n'a pas de mémoire de patch : un « patch »
est une combinaison de boutons de registre + faders).

## Sources

- Wikipedia « ARP String Ensemble » ; Sound On Sound « Eminent 310 » (ancêtre, même circuit,
  bande passante BBD ≈ 6 kHz) ; jhaible « Triple Chorus » (3 BBD, deux générateurs 3-phases
  chorus lent / vibrato rapide) ; KVR « solina ensemble effect » (BBD TCA350 185 étages,
  LFO ≈ 0,6 Hz et ≈ 6 Hz, phases 120°, faible profondeur, différence relative entre lignes
  plus importante que le délai absolu) ; guide Zoe Blade (plages MIDI, caractère des
  registres, crescendo sans effet sur les cuivres) ; manuel Arturia Solina V (Horn prime sur
  Trumpet, Violin = +1 octave, Contrabass = Cello −1 octave, sustain max 4 s, basse toujours
  mono) ; guide Behringer Solina (contrôles, 49 voix, cello/contrabass mono).

## Faits retenus du circuit

| Élément | Comportement |
| --- | --- |
| Clavier | 49 touches C2–C6 (MIDI 36–84). Section haute polyphonique 36–84, section basse **mono** 36–55 (20 touches graves). |
| Générateur | 1 horloge maître → top-octave (12 notes) → diviseurs par 2. Toutes les notes verrouillées en phase, zéro dérive entre voix. Onde « dent de scie en escalier » : somme d'octaves de carrés (1, ½, ¼, ⅛). |
| Keyer | 1 enveloppe RC Attack/Release par touche. `crescendo` = attaque (≈ 5 ms → 2 s), `sustain_length` = release (≈ 50 ms → 4 s). Pas de decay ni sustain. Re-déclenchement pendant le release : reprise depuis le niveau courant. Le crescendo n'agit pas sur trumpet/horn (attaque fixe rapide). |
| Registres haut | Violin = viola +1 octave. Viola = 8', brillant. Trumpet = 8' formant cuivré. Horn = trumpet filtré (plus sombre) ; Horn **prime** sur Trumpet (jamais les deux). Filtres formants fixes appliqués sur le **bus sommé** du registre (paraphonique), pas par voix. |
| Registres basse | Cello 8', Contrabass = cello −1 octave. Mono, priorité à la note la plus grave tenue. Fader `bass_volume` séparé. Ne passent pas dans l'ensemble. |
| Ensemble | 3 lignes BBD parallèles, **100 % wet**, délai nominal ≈ 5 ms. Modulation = LFO chorus 0,6 Hz (±1,5 ms) + LFO vibrato 6 Hz (±0,15 ms), chaque ligne déphasée de 120° sur les deux LFO. Passe-bas ≈ 6 kHz avant/après (bande BBD) + légère saturation douce. Sortie mono d'origine. |

## Architecture

### `synthwave/engine/solina.py`

- `TOP_OCTAVE_HZ` : 12 fréquences de C8 à B8 (tempérament égal sur A = 440), notes
  obtenues par division exacte `2**k` → `note_hz(midi)`.
- `staircase_saw(phase, steps=4)` : somme des 4 octaves de carrés pondérées 1, ½, ¼, ⅛,
  normalisée. Polynôme band-limité inutile : les carrés naïfs alias au-dessus de 6 kHz mais
  le passe-bas BBD et le filtre de registre les masquent ; on applique un passe-bas 4 pôles
  à 9 kHz sur le bus pour éviter l'aliasing au-delà.
- `RcKeyer` : 49 enveloppes vectorisées (tableau numpy `level[49]`, `gate[49]`). Par bloc,
  charge exponentielle `level += (1 - level) * (1 - exp(-n/(τa·sr)))` sur les touches gate
  on, décharge `level *= exp(-n/(τr·sr))` sur les autres (rendu échantillon par
  échantillon par segment linéaire du niveau sur le bloc, suffisant à 1024 échantillons).
  Une touche est « active » tant que `level > 1e-4`.
- `SolinaSynth(patch, sr, rng, bpm)` : interface `render(n, events, gain=None)`,
  `set_patch`, `update_patch`, `set_bpm`, `note_on`, `note_off`, attribut `voices`
  (liste vide, compat renderer) ; `patch.volume` appliqué avant `gain`, `effects` après.
  Rendu :
  1. Dispatch des `NoteEvent` en tranches comme `Synth.render`.
  2. Section haute : pour chaque touche active (36–84), phase continue par touche
     (`phase[49]`, pour violin l'octave supérieure est un second accumulateur) ; onde
     escalier × enveloppe. Somme sur deux bus : `strings` (viola + violin) et `brass`
     (trumpet ou horn, enveloppe attaque fixe 8 ms + même release).
  3. Filtres de bus (biquads RBJ `Filter` existants, coefficients fixes) :
     viola : passe-haut 300 Hz + bosse 1 kHz (bp mix) ; violin : idem +1 octave, bosse
     2,5 kHz ; trumpet : passe-bande 1,2 kHz Q modéré mélangé au brut, brillant ;
     horn : passe-bas 900 Hz + bosse 500 Hz.
  4. Section basse : note la plus grave tenue ≤ 55, keyer dédié, cello (passe-bas 1,2 kHz
     + bosse 250 Hz) et contrabass (−1 octave, passe-bas 600 Hz), × `bass_volume`.
  5. Ensemble (`SolinaEnsemble`, réutilisable comme effet `ensemble` dans `_REGISTRY`) sur
     `strings + brass` uniquement, puis somme avec la basse sèche.
  6. `stereo: true` (défaut) : les 3 lignes d'ensemble pannées G / C / D (infidélité
     documentée) ; `false` : somme mono dupliquée.
- `SolinaEnsemble(sr, bpm, chorus_rate=0.6, chorus_depth=0.0015, vibrato_rate=6.0,
  vibrato_depth=0.00015, base_delay=0.005, stereo=True, bandwidth=6000)` : buffer
  circulaire mono, lecture interpolée linéaire pour 3 lignes, passe-bas 1 pôle × 2 avant
  et après, `tanh` doux à l'entrée (compression BBD). Entrée stéréo sommée en mono (fidèle :
  le Solina est mono en interne).

### `synthwave/patches/model.py`

```python
class SolinaRegisters(BaseModel):
    violin: bool = True
    viola: bool = True
    trumpet: bool = False
    horn: bool = False
    cello: bool = False
    contrabass: bool = False

class SolinaPatchModel(BaseModel):
    name: str
    kind: Literal["solina"] = "solina"
    registers: SolinaRegisters = SolinaRegisters()
    crescendo: float = Field(0.3, ge=0.005, le=2.0)      # attaque cordes (s)
    sustain_length: float = Field(0.8, ge=0.05, le=4.0)  # release (s)
    ensemble: bool = True
    stereo: bool = True
    bass_volume: float = Field(0.8, ge=0.0, le=1.5)
    split_note: int = Field(55, ge=36, le=84)             # dernière touche de la section basse
    tune: float = Field(0.0, ge=-100.0, le=100.0)         # cents
    volume: float = Field(0.5, ge=0.0, le=2.0)
    effects: list[EffectSpec] = []
```

`AnyPatch` étendu ; `loader.patch_from_dict` route `kind == "solina"`.

### Intégration

- `renderer.py` : branche `isinstance(patch, SolinaPatchModel)` → `SolinaSynth` (comme
  DX7) ; trim par couche mesuré après calibration (cible : peak/RMS alignés sur
  `pad_strings`). `apply_tweaks` doit ignorer les patches Solina (pas de `filter.cutoff`).
- `web/static/index.html` : filtre de liste `solina_*` sur toutes les couches hors drums,
  pastille `SOLINA` (orange). Panneau Tweak : sliders `crescendo`, `sustain_length`,
  `bass_volume`, `volume` si le patch est Solina (via `set_patch_param`, chemins plats).
- `moods.py` : `solina_strings` et `solina_full` dans les pools `pad` (DARK, BRIGHT,
  MINIMAL), `solina_cello` dans `pad` DARK, `solina_brass` dans `lead2` BRIGHT,
  `solina_oxygene` dans `ambient` BRIGHT/MINIMAL. Test du pool satisfait pour tout le bank.
- README : section « Solina String Ensemble fidèle », tableau du bank, paramètres YAML,
  faits du circuit et sources. CLAUDE.md : une ligne dans Moteur + patch kinds.

### Bank de patches (`patches/library/solina_*.yaml`)

Tous avec `ensemble: true` sauf mention. Réglages tirés des combinaisons documentées.

| Nom | Registres | Crescendo / Sustain | Origine |
| --- | --- | --- | --- |
| `solina_violin` | violin | 0.15 / 0.6 | registre seul |
| `solina_viola` | viola | 0.15 / 0.6 | registre seul |
| `solina_trumpet` | trumpet | — / 0.4 | registre seul |
| `solina_horn` | horn | — / 0.5 | registre seul |
| `solina_cello` | cello + viola | 0.25 / 0.9 | registre basse + tenue |
| `solina_contrabass` | contrabass + cello | 0.1 / 0.7 | registre basse |
| `solina_strings` | violin + viola | 0.35 / 1.2 | son « classique » du Solina |
| `solina_full` | les 6 (horn prime) | 0.3 / 1.0 | « all buttons in » |
| `solina_wywh` | violin + viola + cello | 1.2 / 3.0 | Pink Floyd, Wish You Were Here : nappes lentes |
| `solina_oxygene` | viola + violin + contrabass | 0.6 / 2.0 | Jarre, Oxygène/Équinoxe (Eminent 310) |
| `solina_moon_safari` | viola + cello | 0.2 / 1.0 | Air, Moon Safari : cordes douces, médium |
| `solina_dream_weaver` | violin + viola + horn | 0.5 / 1.5 | Gary Wright : mur de cordes + cuivre |
| `solina_radio_star` | viola + trumpet | 0.05 / 0.3 | Buggles : stabs courts |
| `solina_dry` | violin + viola, `ensemble: false` | 0.3 / 1.0 | son brut avant BBD (organ-like) |

Les `effects` YAML restent disponibles (reverb typique 70s ajoutée sur `wywh`, `oxygene`).

## Tests (`tests/test_solina.py`)

- Fréquences divide-down : `note_hz(69) == 440` et ratios exacts d'octave.
- Keyer RC : montée exponentielle, `crescendo` respecté à ±10 %, re-trigger sans reset.
- Section basse mono : 2 notes graves → une seule fondamentale dans le spectre (la plus grave).
- Horn prime sur trumpet : rendu identique horn seul vs horn + trumpet.
- Ensemble : spectre d'un la tenu s'élargit (énergie hors raie > seuil) vs `ensemble: false` ;
  sortie mono identique G/D quand `stereo: false`.
- Patch : validation pydantic, chargement du bank, chaque `solina_*` dans un pool
  (test existant), rendu fini et non nul pour chaque patch du bank.
- Renderer : `--patch pad:solina_strings` rend 2 s sans erreur ; `set_patch_param` sur
  `crescendo` passe.

## Hors périmètre

Phaser Behringer, arpégiateur / résonateur Arturia, modes paraphoniques, MIDI.
