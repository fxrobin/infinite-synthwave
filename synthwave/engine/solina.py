"""Solina String Ensemble (Eminent/ARP, 1974) — émulation du circuit.

Chaîne fidèle à l'original :

* une horloge maître → top-octave (12 notes) → diviseurs par 2 : toutes les touches sont
  verrouillées en phase (aucune dérive entre voix, contrairement à un polysynthé) ;
* onde « dent de scie en escalier » (somme d'octaves de carrés = compteur 4 bits) ;
* un keyer RC attaque/release par touche (`crescendo` / `sustain_length`), 49 touches C2–C6 ;
* registres haut (viola 8', violin 4', trumpet, horn — Horn prime sur Trumpet) filtrés sur
  le bus sommé (paraphonique), section basse (cello 8', contrabass 16') **monophonique**
  sur les 20 touches graves, hors ensemble ;
* ensemble triple BBD 100 % wet : 3 lignes ≈ 5 ms modulées par deux LFO 3-phases
  (chorus ≈ 0,6 Hz, vibrato ≈ 6 Hz), bande passante ≈ 6 kHz.
"""

from __future__ import annotations

import numpy as np

from ..patches.model import SolinaPatchModel
from .blocks import arange as _arange
from .blocks import arange1 as _arange1
from .blocks import lfilter
from .blocks import segments as _segments
from .effects import Effect, build_effects
from .events import NoteEvent
from .filter import biquad_coeffs

KEY_LOW, KEY_HIGH = 36, 84  # clavier 49 touches C2–C6
N_KEYS = KEY_HIGH - KEY_LOW + 1
BRASS_ATTACK_S = 0.008  # le crescendo n'agit pas sur trumpet/horn (attaque fixe)
_STEPS = 16  # compteur 4 bits : 4 octaves de carrés sommées

# Top-octave : C8..B8 (MIDI 108..119) en tempérament égal sur A4 = 440 Hz.
TOP_OCTAVE_HZ = tuple(440.0 * 2.0 ** ((m - 69) / 12.0) for m in range(108, 120))


def note_hz(midi: int, tune_cents: float = 0.0) -> float:
    """Fréquence d'une touche par division exacte (2**k) de la top-octave."""
    k = 9 - midi // 12
    hz = TOP_OCTAVE_HZ[midi % 12] / float(2**k) if k >= 0 else TOP_OCTAVE_HZ[midi % 12] * 2 ** (-k)
    return hz * 2.0 ** (tune_cents / 1200.0) if tune_cents else hz


def staircase_saw(phase: np.ndarray) -> np.ndarray:
    """Dent de scie en escalier 16 niveaux (sortie d'un compteur binaire 4 bits), ±1."""
    return np.floor(phase * _STEPS) * (2.0 / (_STEPS - 1)) - 1.0


class RcKeyer:
    """Enveloppes RC attaque/release vectorisées : une par touche, gate indépendant."""

    def __init__(
        self, sr: int, n_keys: int = N_KEYS, attack_s: float = 0.3, release_s: float = 0.8
    ):
        self.sr, self.n = sr, n_keys
        self.level = np.zeros(n_keys)
        self.gate = np.zeros(n_keys, dtype=bool)
        self.set_times(attack_s, release_s)

    def set_times(self, attack_s: float, release_s: float) -> None:
        """Constantes de temps (s) — RC : 63 % du chemin à τ."""
        self.tau_a = max(1e-4, float(attack_s)) * self.sr
        self.tau_r = max(1e-4, float(release_s)) * self.sr

    def gate_on(self, i: int) -> None:
        self.gate[i] = True

    def gate_off(self, i: int) -> None:
        self.gate[i] = False

    @property
    def active(self) -> np.ndarray:
        return self.gate | (self.level > 1e-4)

    def render(self, n: int) -> np.ndarray:
        """(n, n_keys) niveaux ; met à jour l'état.

        Seules les touches ouvertes ou encore en release sont calculées : une touche au
        repos a ``gate=False`` et ``level`` exactement nul, donc sa colonne vaut zéro et
        le résultat est identique au calcul sur les 49 touches.
        """
        act = np.flatnonzero(self.gate | (self.level > 0.0))
        out = np.zeros((n, self.n))
        if act.size == 0:
            return out
        t = _arange1(n)[:, None]
        lvl = self.level[act][None, :]
        up = 1.0 - (1.0 - lvl) * np.exp(-t / self.tau_a)
        down = lvl * np.exp(-t / self.tau_r)
        col = np.where(self.gate[act][None, :], up, down)
        out[:, act] = col
        last = col[-1].copy()
        last[last < 1e-6] = 0.0
        self.level[act] = last
        return out


class _Biquad:
    """Biquad RBJ mono à coefficients fixes (filtres formants des bus de registre)."""

    def __init__(self, kind: str, fc: float, res: float, sr: int):
        self.b, self.a = biquad_coeffs(kind, fc, res, sr)
        self.zi = np.zeros(2)

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        return y


class _OnePole:
    """Passe-bas 1 pôle mono (bande passante BBD)."""

    def __init__(self, fc: float, sr: int):
        a = 1.0 - float(np.exp(-2.0 * np.pi * fc / sr))
        self.b, self.a = np.array([a]), np.array([1.0, a - 1.0])
        self.zi = np.zeros(1)

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        return y


class _Register:
    """Filtre formant fixe d'un registre : chaîne série + bosses parallèles."""

    def __init__(
        self,
        sr: int,
        series: list[tuple[str, float, float]],
        bumps: list[tuple[float, float, float]],
        dry: float = 1.0,
    ):
        self.series = [_Biquad(k, f, r, sr) for k, f, r in series]
        self.bumps = [(_Biquad("bp", f, r, sr), g) for f, r, g in bumps]
        self.dry = dry

    def process(self, x: np.ndarray) -> np.ndarray:
        for f in self.series:
            x = f.process(x)
        y = x * self.dry
        for f, g in self.bumps:
            y = y + f.process(x) * g
        return y


def _make_registers(sr: int) -> dict[str, _Register]:
    """Formants fixes (RC passifs sur l'original), bus par registre."""
    return {
        "viola": _Register(sr, [("hp", 300.0, 0.1)], [(1000.0, 0.45, 0.8)]),
        "violin": _Register(sr, [("hp", 600.0, 0.1)], [(2500.0, 0.45, 0.7)]),
        "trumpet": _Register(
            sr, [("hp", 200.0, 0.1)], [(1200.0, 0.55, 1.2), (3200.0, 0.5, 0.5)], dry=0.5
        ),
        "horn": _Register(sr, [("lp", 900.0, 0.15)], [(500.0, 0.5, 0.9)]),
        "cello": _Register(sr, [("lp", 1200.0, 0.15)], [(250.0, 0.45, 0.8)]),
        "contrabass": _Register(sr, [("lp", 600.0, 0.15)], [(120.0, 0.4, 0.6)]),
    }


class SolinaEnsemble(Effect):
    """Triple BBD du Solina : 3 lignes ≈ 5 ms, deux LFO 3-phases, 100 % wet, mono interne.

    `stereo=True` panne les 3 lignes G/C/D (infidélité assumée, l'original est mono).
    """

    def __init__(
        self,
        sr: int,
        bpm: float = 120.0,
        chorus_rate: float = 0.6,
        chorus_depth: float = 0.0015,
        vibrato_rate: float = 6.0,
        vibrato_depth: float = 0.00015,
        base_delay: float = 0.005,
        stereo: bool = True,
        bandwidth: float = 6000.0,
    ):
        self.sr = sr
        self.chunk = 4096  # le buffer doit couvrir un bloc + le délai max
        self.size = self.chunk + int(sr * 0.02)
        self.buf = np.zeros(self.size)
        self.pos = 0
        self.base = float(base_delay) * sr
        self.cd, self.vd = float(chorus_depth) * sr, float(vibrato_depth) * sr
        self.cr, self.vr = float(chorus_rate), float(vibrato_rate)
        self.ph_c, self.ph_v = 0.0, 0.0
        self.stereo = bool(stereo)
        self.pre = [_OnePole(bandwidth, sr), _OnePole(bandwidth, sr)]
        self.post = [[_OnePole(bandwidth, sr), _OnePole(bandwidth * 1.3, sr)] for _ in range(3)]
        # panoramique constant-power G / C / D
        angles = np.array([0.15, 0.5, 0.85]) * np.pi / 2.0
        self.gain_l, self.gain_r = np.cos(angles), np.sin(angles)

    def process(self, x: np.ndarray) -> np.ndarray:
        if len(x) > self.chunk:
            return np.concatenate(
                [self._process(x[i : i + self.chunk]) for i in range(0, len(x), self.chunk)]
            )
        return self._process(x)

    def _process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        mono = x.mean(axis=1).astype(np.float64)
        for f in self.pre:
            mono = f.process(mono)
        mono = np.tanh(mono * 1.3) / 1.3  # compression douce des BBD
        for b0, b1, o0, o1 in _segments(self.pos, n, self.size):
            self.buf[b0:b1] = mono[o0:o1]
        t = _arange1(n)
        pc = (self.ph_c + self.cr / self.sr * t) % 1.0
        pv = (self.ph_v + self.vr / self.sr * t) % 1.0
        self.ph_c, self.ph_v = float(pc[-1]), float(pv[-1])
        out = np.zeros((n, 2))
        for i in range(3):
            off = i / 3.0
            d = (
                self.base
                + self.cd * np.sin(2 * np.pi * (pc + off))
                + self.vd * np.sin(2 * np.pi * (pv + off))
            )
            p = self.pos + _arange(n) - d
            i0 = np.floor(p).astype(int)
            fr = p - i0
            i0 %= self.size
            i1 = i0 + 1
            i1[i1 == self.size] = 0
            line = self.buf[i0] * (1 - fr) + self.buf[i1] * fr
            for f in self.post[i]:
                line = f.process(line)
            if self.stereo:
                out[:, 0] += line * self.gain_l[i]
                out[:, 1] += line * self.gain_r[i]
            else:
                out[:, 0] += line
                out[:, 1] += line
        self.pos = (self.pos + n) % self.size
        return (out * (0.62 if self.stereo else 1.0 / 3.0)).astype(np.float32)


class SolinaSynth:
    """Solina String Ensemble — même interface que `Synth` / `Dx7Synth`."""

    voices: list = []

    def __init__(self, patch: SolinaPatchModel, sr: int, rng: np.random.Generator, bpm: float):
        self.sr, self.rng, self.bpm = sr, rng, bpm
        self.set_patch(patch)

    # ----- patch -----
    def set_patch(self, patch: SolinaPatchModel) -> None:
        self.patch = patch
        sr = self.sr
        self.keyer = RcKeyer(sr, N_KEYS, patch.crescendo, patch.sustain_length)
        self.brass_keyer = RcKeyer(sr, N_KEYS, BRASS_ATTACK_S, patch.sustain_length)
        self.bass_keyer = RcKeyer(sr, 1, patch.crescendo, patch.sustain_length)
        self.registers = _make_registers(sr)
        self.anti_alias = [_Biquad("lp", 9000.0, 0.2, sr), _Biquad("lp", 9000.0, 0.2, sr)]
        self.ensemble = (
            SolinaEnsemble(sr, self.bpm, stereo=patch.stereo) if patch.ensemble else None
        )
        self.effects = build_effects([e.model_dump() for e in patch.effects], sr, self.bpm)
        # accumulateur de phase à l'octave grave (16') : 8' = 2φ, 4' = 4φ — verrouillage de phase
        self.phase_lo = np.zeros(N_KEYS)
        self.freq_lo = np.array([note_hz(KEY_LOW + i, patch.tune) / 2.0 for i in range(N_KEYS)])
        self.held_bass: list[int] = []
        self.bass_phase = 0.0
        self.bass_note: int | None = None

    def update_patch(self, patch: SolinaPatchModel) -> None:
        """Changement de paramètres sans reset des keyers (pas de clic)."""
        old = self.patch
        self.patch = patch
        self.keyer.set_times(patch.crescendo, patch.sustain_length)
        self.brass_keyer.set_times(BRASS_ATTACK_S, patch.sustain_length)
        self.bass_keyer.set_times(patch.crescendo, patch.sustain_length)
        if patch.tune != old.tune:
            self.freq_lo = np.array([note_hz(KEY_LOW + i, patch.tune) / 2.0 for i in range(N_KEYS)])
        if patch.ensemble != old.ensemble or patch.stereo != old.stereo:
            self.ensemble = (
                SolinaEnsemble(self.sr, self.bpm, stereo=patch.stereo) if patch.ensemble else None
            )
        if [e.model_dump() for e in patch.effects] != [e.model_dump() for e in old.effects]:
            self.effects = build_effects([e.model_dump() for e in patch.effects], self.sr, self.bpm)

    def set_bpm(self, bpm: float) -> None:
        self.bpm = bpm
        self.effects = build_effects([e.model_dump() for e in self.patch.effects], self.sr, bpm)

    # ----- clavier -----
    @staticmethod
    def _fold(note: int) -> int:
        """Ramène une note hors clavier dans C2–C6 par octaves (le Solina n'a que 49 touches)."""
        while note < KEY_LOW:
            note += 12
        while note > KEY_HIGH:
            note -= 12
        return note

    def note_on(self, note: int, velocity: float) -> None:
        note = self._fold(int(note))
        i = note - KEY_LOW
        self.keyer.gate_on(i)
        self.brass_keyer.gate_on(i)
        if note <= self.patch.split_note:
            if note not in self.held_bass:
                self.held_bass.append(note)
            self.bass_keyer.gate_on(0)

    def note_off(self, note: int) -> None:
        note = self._fold(int(note))
        i = note - KEY_LOW
        self.keyer.gate_off(i)
        self.brass_keyer.gate_off(i)
        if note in self.held_bass:
            self.held_bass.remove(note)
            if not self.held_bass:
                self.bass_keyer.gate_off(0)

    # ----- rendu -----
    def _upper(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Bus cordes et bus cuivres (mono) de la section haute."""
        r = self.patch.registers
        strings = np.zeros(n)
        brass = np.zeros(n)
        want_strings = r.viola or r.violin
        want_brass = r.horn or r.trumpet
        active = np.flatnonzero(self.keyer.active | self.brass_keyer.active)
        env_s = self.keyer.render(n)
        env_b = self.brass_keyer.render(n)
        if not len(active) or not (want_strings or want_brass):
            self.phase_lo = (self.phase_lo + self.freq_lo / self.sr * n) % 1.0
            return strings, brass
        t = np.arange(1, n + 1)[:, None]
        ph = (self.phase_lo[None, active] + self.freq_lo[None, active] / self.sr * t) % 1.0
        self.phase_lo = (self.phase_lo + self.freq_lo / self.sr * n) % 1.0
        w8 = staircase_saw((2.0 * ph) % 1.0)
        if want_strings:
            e = env_s[:, active]
            if r.viola:
                strings += self.registers["viola"].process((w8 * e).sum(axis=1))
            if r.violin:
                w4 = staircase_saw((4.0 * ph) % 1.0)
                strings += self.registers["violin"].process((w4 * e).sum(axis=1))
        if want_brass:
            raw = (w8 * env_b[:, active]).sum(axis=1)
            reg = "horn" if r.horn else "trumpet"  # Horn prime sur Trumpet
            brass = self.registers[reg].process(raw)
        return strings, brass

    def _bass(self, n: int) -> np.ndarray:
        """Section basse monophonique : note la plus grave tenue, keyer dédié, hors ensemble."""
        r = self.patch.registers
        env = self.bass_keyer.render(n)[:, 0]
        if not (r.cello or r.contrabass):
            return np.zeros(n)
        if self.held_bass:
            self.bass_note = min(self.held_bass)
        if self.bass_note is None or env.max() < 1e-4:
            return np.zeros(n)
        f_lo = note_hz(self.bass_note, self.patch.tune) / 2.0
        ph = (self.bass_phase + f_lo / self.sr * np.arange(1, n + 1)) % 1.0
        self.bass_phase = float(ph[-1])
        out = np.zeros(n)
        if r.cello:
            out += self.registers["cello"].process(staircase_saw((2.0 * ph) % 1.0) * env)
        if r.contrabass:
            out += self.registers["contrabass"].process(staircase_saw(ph) * env)
        return out * self.patch.bass_volume

    def _render_block(self, n: int) -> np.ndarray:
        strings, brass = self._upper(n)
        upper = strings + brass
        for f in self.anti_alias:
            upper = f.process(upper)
        upper = upper * 0.22
        st = np.stack([upper, upper], axis=1).astype(np.float32)
        if self.ensemble is not None:
            st = self.ensemble.process(st)
        else:
            st *= 0.33  # même niveau que la somme mono des 3 lignes BBD
        bass = (self._bass(n) * 0.3).astype(np.float32)
        st[:, 0] += bass
        st[:, 1] += bass
        return st

    def render(self, n: int, events: list[NoteEvent], gain: np.ndarray | None = None) -> np.ndarray:
        out = np.zeros((n, 2), dtype=np.float32)
        pos = 0
        for ev in sorted(events):
            off = min(max(ev.offset, pos), n)
            if off > pos:
                out[pos:off] = self._render_block(off - pos)
                pos = off
            if ev.on:
                self.note_on(ev.note, ev.velocity)
            else:
                self.note_off(ev.note)
        if pos < n:
            out[pos:] = self._render_block(n - pos)
        out *= self.patch.volume
        if gain is not None:
            out *= gain[:, None]
        for fx in self.effects:
            out = fx.process(out)
        return out
