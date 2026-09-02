# Infinite Synthwave — Design

Date : 2026-09-02

## Objectif

Programme qui génère de la synthwave (cyberpunk / outrun) en continu sur la sortie audio,
ou pendant une durée fixée, avec des synthétiseurs internes programmables et des couches
ambient. Pilotable en CLI et via un serveur MCP.

## Décisions

| Sujet | Choix |
|---|---|
| Stack | Python 3.13, numpy, sounddevice (PortAudio → PipeWire), soundfile, pyyaml, pydantic, mcp, typer ; gestion uv |
| Interfaces | CLI `synthwave` + serveur MCP stdio |
| Synthés | Patches YAML déclaratifs, rechargeables à chaud |
| Composition | Tracker procédural : patterns 16 pas par canal, harmonie par chaîne de Markov, arrangement par sections |

## Références de style

- Tempo 100–118 BPM (défaut 110), 4/4.
- Tonalité mineure dominante. Progressions pondérées : i‑VI‑III‑VII, i‑VII‑VI‑VII, i‑iv‑VI‑V, i‑VI‑VII‑i, VI‑VII‑i‑i.
- Accords avec 7e (min7, maj7) sur les pads.
- Basse : croches sur la fondamentale, octaves, saw + filtre LP résonant.
- Arpège : 16e, motif up / up‑down / random sur les notes de l'accord, pluck court.
- Pads : saw unison détuné + chorus + reverb hall longue, attaque lente.
- Lead : saw brillant, delay ping‑pong, phrases rares (sections chorus).
- Ambient : nappes bruit filtré + drone sinus lent, présence continue mais faible.
- Batterie : kick 4/4, snare 2 et 4 avec gated reverb, hats 8e/16e, fills fin de section.
- Sidechain : le kick compresse pads et basse (« pump »).

## Architecture

```
synthwave/
  engine/       DSP numpy : oscillator, envelope, filter, lfo, voice, synth, effects, drums
  patches/      YAML : bass_moog, pad_juno, arp_pluck, lead_saw, ambient_drone, drums_808
  composer/     harmony (Markov), patterns (par canal), arranger (sections)
  sequencer/    tracker (patterns → événements), transport (bpm, position, seed)
  audio/        renderer (mix blocs), output (sounddevice), export (WAV)
  cli.py        typer
  mcp_server.py FastMCP
```

### engine

- `Oscillator` : saw, square (pwm), triangle, sine, noise. Unison N voix, detune en cents, spread stéréo. Génération vectorisée par bloc à partir de la phase.
- `ADSR` : attaque, decay, sustain, release, en secondes ; forme exponentielle.
- `Filter` : SVF 2 pôles, modes LP/HP/BP, cutoff Hz, résonance 0–1, enveloppe et LFO sur cutoff.
- `LFO` : sine/tri/square/saw, fréquence Hz ou synchro tempo, cibles : pitch, cutoff, amp, pwm.
- `Voice` : osc → filtre → amp ADSR. `Synth` : polyphonie N, allocation voix (vol de la plus ancienne), glide optionnel (mono).
- Effets : `Chorus`, `Delay` (temps synchro tempo, feedback, ping‑pong), `Reverb` (Schroeder : combs + allpass, taille, damping), `GatedReverb` (reverb + gate à seuil/hold), `Sidechain` (enveloppe déclenchée par le kick, profondeur, release), `Limiter` (bus master).
- `Drums` : synthèse (kick sine pitch‑drop, snare bruit + tonal, hat bruit HP court, tom, clap), pas d'échantillons.

### patches (YAML)

```yaml
name: pad_juno
polyphony: 8
oscillators:
  - wave: saw
    unison: 5
    detune: 12      # cents
    level: 0.8
  - wave: square
    octave: -1
    level: 0.3
amp_env: {attack: 0.8, decay: 0.5, sustain: 0.8, release: 1.5}
filter:
  type: lp
  cutoff: 1800
  resonance: 0.2
  env: {attack: 0.5, decay: 1.0, sustain: 0.3, release: 1.0, amount: 800}
lfo: {wave: sine, rate: 0.3, target: cutoff, amount: 300}
effects:
  - {type: chorus, rate: 0.5, depth: 0.004, mix: 0.4}
  - {type: reverb, size: 0.9, damping: 0.4, mix: 0.35}
```

Validation pydantic ; erreur lisible ; patch précédent conservé si invalide.

### composer

- `Harmony` : tonalité (tonique, mode mineur / majeur), chaîne de Markov entre progressions
  de 4 accords, degrés → notes MIDI avec 7e. Modulation de tonalité (±quinte, relatif)
  toutes 4–8 sections.
- `PatternGen` : par canal, entrée = accord courant, section, densité, RNG ; sortie = pattern
  16 pas (liste d'événements : pas, note, vélocité, durée). Mutation contrôlée d'un pattern
  précédent (10–30 % des pas) pour la continuité.
- `Arranger` : machine à états intro → verse → chorus → break → verse … ; densité par section
  (quels canaux actifs, volume) ; fills sur la dernière mesure ; en mode durée fixe, outro +
  fade pour finir à l'heure ; jamais deux fois exactement le même pattern d'affilée.
- Moods : `dark` (mineur, tempo bas, filtre fermé), `dreamy` (majeur/relatif, pads dominants),
  `outrun` (tempo 118+, drums denses, arp permanent). Le mood pondère Markov et densités.

### sequencer

- `Transport` : bpm, sample rate, position (mesure, pas, sample), seed. Conversion pas ↔ samples.
- `Tracker` : patterns par canal ; à chaque bloc audio, émet note‑on/off aux samples exacts ;
  demande à l'Arranger la section suivante une mesure avant la fin.

### audio

- `Renderer.render(n_frames) -> float32 (n, 2)` : fait avancer transport, tracker, synthés,
  effets par canal, bus master (sidechain, reverb send, limiter). Déterministe par seed.
- `Output` : callback sounddevice, blocksize 512, 44 100 Hz ; file de commandes thread‑safe
  depuis CLI/MCP (set_tempo, mute, load_patch…) appliquées entre blocs.
- `Export` : rendu offline vers WAV via soundfile.

### CLI

```
synthwave play [--duration 5m] [--bpm 110] [--seed 42] [--mood dark|dreamy|outrun] [--export out.wav]
synthwave patches list
synthwave devices
synthwave mcp        # lance le serveur MCP stdio
```

### MCP (FastMCP, stdio)

Outils : `start(mood, bpm, seed)`, `stop()`, `status()`, `set_tempo(bpm)`, `set_mood(mood)`,
`set_layer(layer, mute|solo|volume)`, `list_patches()`, `load_patch(layer, name)`,
`set_patch_param(layer, path, value)`, `next_section()`, `export_wav(path, seconds)`.

## Erreurs

- Aucun périphérique audio : message clair + suggestion `--export`.
- Underrun : compteur exposé dans `status`, warning log.
- Patch invalide : erreur pydantic, état inchangé.

## Tests (pytest)

- Oscillateurs : fréquence fondamentale via FFT, niveau.
- ADSR : montée/descente, sustain, release à zéro.
- Filtre : atténuation au‑dessus du cutoff.
- Harmony : accords dans la tonalité, transitions Markov valides, reproductibilité par seed.
- PatternGen : pas dans [0,16), notes dans l'accord, mutation borne.
- Arranger : enchaînement des sections, outro en mode durée.
- Tracker : événements aux samples attendus.
- Renderer : rendu 2 s déterministe (hash), pas de clipping, pas de NaN.
- Patches : chargement des YAML livrés, rejet d'un patch invalide.
- MCP : appel des outils via client in‑process.

## Hors périmètre (v1)

Interface graphique, import de MOD/XM, échantillons externes, MIDI out.
