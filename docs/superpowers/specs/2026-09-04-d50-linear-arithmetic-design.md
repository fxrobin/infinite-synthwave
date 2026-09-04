# Émulation du Roland D-50 (Linear Arithmetic synthesis)

Date : 2026-09-04. Statut : validé en discussion.

## Objectif

Moteur `D50Synth` (`kind: d50`) reproduisant l'architecture LA du D-50 (1987) : 2 tones × 2
partials, 7 structures avec ring modulator, partials synthé LA32 (onde « déjà filtrée », pas de
filtre), partials PCM, enveloppes 5 temps TVF/TVA, P-ENV, 3 LFO par tone, EQ + chorus par tone,
reverb par patch. Import des banks sysex (`synthwave import-d50 bank.syx`), bank ROM Roland
PN-D50-00 (Fantasia, Digital Native Dance, Staccato Heaven, Pizzagogo…) livré dans la bibliothèque.

## Sources

Guide de paramètres Roland D-05 (valeurs et plages, structures, liste des 100 PCM, tables reverb /
chorus / EQ), Wikipedia, notes de service (plages temporelles : TVF/TVA 4 ms–80 s, P-ENV 9 ms–9 s,
LFO 0,0004–27 Hz, delay 0–10 s, pitch mod ±600 cents LFO / ±2400 env), modèle LA32 documenté par
munt (carré à flancs cosinus, résonance = sinus amorti relancé par cycle, saw = carré × cosinus
⇒ une octave au-dessus), format sysex vérifié par deux parsers open source (D50SysexBinConverter,
d5_syx_to_patches.py) : DT1 `F0 41 dev 14 12 aa bb cc … cs F7`, adresse 02-00-00 = 0x8000,
64 patches × 448 octets, 7 blocs de 64 : U-P1, U-P2, U-common, L-P1, L-P2, L-common, patch.

## Contrainte : les PCM

Les 100 ondes PCM sont dans des ROM Roland copyrightées, absentes des sysex. Elles sont
**synthétisées procéduralement** par famille (`d50_pcm.py`) à 32 kHz avec les longueurs de la
table reconstruite (one-shots 2048–8192 mots, boucles de cycle 128–2048 mots). Règle de
transposition du D-50 : un partial PCM avance de `f × 2048` mots par seconde (f = hauteur du
partial), donc une boucle de L mots sonne à `f × 2048 / L` et un one-shot joue à vitesse native
pour f ≈ 15,6 Hz (C0) — c'est pourquoi les patches d'usine placent les partials PCM très bas.
Hors périmètre : chargement de dumps ROM utilisateur.

## Offsets sysex (par bloc de 64 octets)

Partial `p[]` : 0 coarse (C1–C7, 36 = C4) · 1 fine (−50..50) · 2 keyfollow (17 ratios) · 3 LFO
mode · 4 P-ENV mode · 5 bend mode · 6 wave (0 SQU / 1 SAW) · 7 PCM (0–99) · 8 PW · 9 PW velo ·
10 PW LFO sel · 11 PW LFO depth · 12 PW AT · 13 cutoff · 14 reso (0–30) · 15 cutoff KF (15
ratios) · 16 bias point · 17 bias level · 18 TVF env depth · 19 TVF velo · 20 TVF depth KF ·
21 TVF time KF · 22–26 TVF T1–T5 · 27–29 TVF L1–L3 · 30 TVF sustain · 31 TVF end · 32 TVF LFO
sel · 33 TVF LFO depth · 34 TVF AT · 35 TVA level · 36 TVA velo (−50..50) · 37 TVA bias point ·
38 TVA bias level · 39–43 TVA T1–T5 · 44–46 TVA L1–L3 · 47 TVA sustain · 48 TVA end · 49 TVA velo
time · 50 TVA time KF · 51 TVA LFO sel · 52 TVA LFO depth · 53 TVA AT.

Common `c[]` : 0–9 nom · 10 structure (0–6) · 11 P-ENV velo · 12 P-ENV time KF · 13–16 P-ENV
T1–T4 · 17–21 P-ENV L0, L1, L2, sustain, end (−50..50) · 22 P-Mod LFO depth · 23 lever · 24 AT ·
25–36 LFO1–3 (wave, rate, delay, sync) · 37 EQ Lf · 38 EQ Lg · 39 EQ Hf · 40 EQ HQ · 41 EQ Hg ·
42 chorus type (0–7) · 43 rate · 44 depth · 45 balance · 46 partial mute (bits P1, P2) ·
47 partial balance.

Patch `pb[]` : 0–17 nom · 18 key mode (0–8) · 19 split (C2 + n) · 20 porta mode · 22/23 key
shift U/L (−24) · 24/25 tune U/L (−50) · 26 bend range · 27 AT bend · 28 porta time ·
29 output mode · 30 reverb type (0–31) · 31 reverb balance · 32 volume · 33 tone balance ·
41 porta switch.

## Moteur (`synthwave/engine/d50.py`)

- `La32Partial` : phase φ ; carré à rapport cyclique `0,5 + 0,47·PW/100` à flancs cosinus de
  demi-largeur `w = f / (2·cutoff_hz)` (plafonnée ⇒ sinus quand cutoff ≤ f, atténuation
  `cutoff/f` en dessous) ; saw = rampe à `2f` à reset cosinus ; résonance = `(res/30)^1.5 ·
  sin(2π·cutoff·t_cycle) · exp(−cutoff·t_cycle·(1,2 − res/30))`. `cutoff_hz = f_C4(coarse, fine)
  · 2^((cutoff − 50)/8) · 2^(KF_cut·(note − 60)/12) · 2^(env·depth + LFO + bias)`.
- `PcmPartial` : lecture interpolée de la table `d50_pcm.py`, incrément `f·2048/sr` mots.
- `Env5` : T1–T5 → `0,004·20000^(v/100)` s, niveaux 0–100 ; TVA en dB (plancher −60 dB), TVF
  linéaire ; keyfollow temps et vélocité. `PitchEnv` : T → `0,009·1000^(v/50)` s, niveaux
  −50..50 → ±1/1,5/2 octaves selon P-ENV velo. `Lfo` : rate → `0,0004·67500^(v/100)` Hz, delay
  → `10·(v/100)²` s, profondeur pitch `600·2^((d − 100)/10)` cents.
- Structure : sortie = `P1 + P2` (1, 3, 6) ou `P1 + P1·P2` (2, 4, 5, 7). Balance partials,
  mute. Tone : key mode WHOLE (upper), DUAL, SPLIT ; key shift, tune ; balance des tones.
- Bus par tone → EQ (low shelf + peak) → chorus (8 types mappés : chorus, flanger, feedback,
  tremolo, chorus+tremolo, dimension) → somme → reverb (32 types mappés sur `Reverb` /
  `Delay` / `GatedReverb` avec les temps du tableau Roland) → volume → `effects` YAML.
- Polyphonie 16 (8 en dual/split), vol d'age.

## Modèle (`D50PatchModel kind: d50`)

`upper` / `lower` : `D50Tone(name, partials[2]: D50Partial, common: D50Common)` ; `key_mode`,
`split`, `key_shift_upper/lower`, `tune_upper/lower`, `reverb_type`, `reverb_balance`,
`patch_volume`, `tone_balance`, `polyphony`, `volume`, `effects`. Tous les champs sont les
valeurs panneau (entiers) pour que l'import sysex soit sans perte.

## Intégration

`loader` route `kind: d50` ; renderer `isinstance` ; UI pastille `D-50` + liste `d50_*` ;
CLI `import-d50` ; bank PN-D50-00 (64 patches `d50_*`) importé dans la bibliothèque ; pools :
`d50_Fantasia`, `d50_Soundtrack`… en pad/ambient, `d50_Digital_Native_Dance`, `d50_Staccato_Heaven`
en arp/lead ; le test des pools exempte le préfixe `d50_` comme `dx7_`. Tests
`tests/test_d50.py` : parse sysex (noms, plages), saw à l'octave du carré, ring mod, structures,
Env5 temps, PCM one-shot s'arrête / boucle continue, rendu fini de tout le bank, renderer.
README + CLAUDE.md.

## Hors périmètre

Aftertouch, bender, portamento, output modes, mode mono `-S`, ROM utilisateur.
