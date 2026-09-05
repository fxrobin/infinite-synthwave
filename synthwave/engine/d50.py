"""Roland D-50 — Linear Arithmetic synthesis (1987).

Architecture reproduite : un patch = 2 tones (upper / lower, key mode WHOLE / DUAL / SPLIT),
un tone = 2 partials + bloc commun, 7 structures (mélange ou ring modulation, partial synthé
ou PCM). Le partial synthé imite le LA32 : pas de filtre, mais une onde carrée « déjà filtrée »
dont les flancs cosinus s'élargissent quand le cutoff baisse, une résonance = sinus amorti
relancé à chaque cycle, et une dent de scie qui joue une octave au-dessus du carré (le bank
d'usine est écrit autour de cette particularité). Le partial PCM lit une des 100 ondes
(``d50_pcm``) selon la règle du D-50 : ``f × 2048`` mots par seconde.

Enveloppes 5 temps / 3 niveaux (TVF linéaire, TVA en dB), P-ENV 4 temps / 5 niveaux, 3 LFO par
tone, EQ + chorus par tone, reverb (32 types) par patch. Import sysex : ``d50_sysex_to_patches``.
"""

from __future__ import annotations

import numpy as np

from ..patches.model import D50Common, D50Env, D50Lfo, D50Partial, D50PatchModel, D50Tone
from .blocks import lfilter
from .d50_pcm import PCM_SR, WORDS_PER_HZ, pcm_wave
from .effects import Chorus, Delay, Effect, Flanger, GatedReverb, Reverb, build_effects
from .events import NoteEvent
from .lfo import LFO

# ----- tables Roland -----
KEYFOLLOW = (
    -1,
    -0.5,
    -0.25,
    0,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1,
    1.25,
    1.5,
    2,
    1.0,
    1.0,
)
KEYFOLLOW_CENTS = (
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    5,
)  # s1, s2 : octave + 1 / 5 cents
CUTOFF_KF = KEYFOLLOW[:15]
EQ_LOW_HZ = (63, 75, 88, 105, 125, 150, 175, 210, 250, 300, 350, 420, 500, 600, 700, 840)
EQ_HIGH_HZ = (
    250,
    300,
    350,
    420,
    500,
    600,
    700,
    840,
    1000,
    1200,
    1400,
    1700,
    2000,
    2400,
    2800,
    3400,
    4000,
    4800,
    5700,
    6700,
    8000,
    9500,
)
EQ_HIGH_Q = (0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.2, 6.0)
# 32 reverbs D-50 → (effet, paramètres) sur les effets existants
REVERB_TYPES: dict[int, tuple[str, dict]] = {
    1: ("reverb", {"size": 0.7, "damping": 0.5}),
    2: ("reverb", {"size": 0.8, "damping": 0.45}),
    3: ("reverb", {"size": 0.9, "damping": 0.4}),
    4: ("reverb", {"size": 0.93, "damping": 0.3}),
    5: ("reverb", {"size": 0.4, "damping": 0.7}),
    6: ("reverb", {"size": 0.45, "damping": 0.2}),
    7: ("reverb", {"size": 0.5, "damping": 0.6}),
    8: ("reverb", {"size": 0.6, "damping": 0.55}),
    9: ("reverb", {"size": 0.68, "damping": 0.5}),
    10: ("reverb", {"size": 0.78, "damping": 0.5}),
    11: ("delay", {"time": 0.102, "feedback": 0.0, "pingpong": False}),
    12: ("delay", {"time": 0.180, "feedback": 0.35, "pingpong": True}),
    13: ("delay", {"time": 0.224, "feedback": 0.35, "pingpong": True}),
    14: ("delay", {"time": 0.148, "feedback": 0.4, "pingpong": True}),
    15: ("gated_reverb", {"size": 0.8, "hold": 0.2}),
    16: ("gated_reverb", {"size": 0.85, "hold": 0.48}),
    17: ("reverb", {"size": 0.85, "damping": 0.15}),
    18: ("reverb", {"size": 0.97, "damping": 0.35}),
    19: ("reverb", {"size": 0.6, "damping": 0.1}),
    20: ("delay", {"time": 0.248, "feedback": 0.3, "pingpong": False}),
    21: ("delay", {"time": 0.338, "feedback": 0.3, "pingpong": False}),
    22: ("delay", {"time": 0.157, "feedback": 0.35, "pingpong": True}),
    23: ("delay", {"time": 0.252, "feedback": 0.35, "pingpong": True}),
    24: ("delay", {"time": 0.274, "feedback": 0.4, "pingpong": True}),
    25: ("gated_reverb", {"size": 0.85, "hold": 0.3}),
    26: ("gated_reverb", {"size": 0.9, "hold": 0.36}),
    27: ("gated_reverb", {"size": 0.9, "hold": 0.48}),
    28: ("delay", {"time": 0.08, "feedback": 0.1, "pingpong": False}),
    29: ("delay", {"time": 0.11, "feedback": 0.1, "pingpong": False}),
    30: ("delay", {"time": 0.14, "feedback": 0.15, "pingpong": False}),
    31: ("reverb", {"size": 0.96, "damping": 0.05}),
    32: ("reverb", {"size": 0.98, "damping": 0.2}),
}
_REVERB_CLS = {"reverb": Reverb, "delay": Delay, "gated_reverb": GatedReverb}
LA_LEVEL = 0.35  # gain interne d'un partial à TVA level 100


def env_time(v: int) -> float:
    """TVF/TVA 0..100 → 4 ms .. 80 s (exponentiel, notes de service)."""
    return 0.004 * 20000.0 ** (v / 100.0)


def penv_time(v: int) -> float:
    """P-ENV 0..50 → 9 ms .. 9 s."""
    return 0.009 * 1000.0 ** (v / 50.0)


def lfo_rate_hz(v: int) -> float:
    """LFO 0..100 → 0,0004 .. 27 Hz."""
    return 0.0004 * 67500.0 ** (v / 100.0)


def depth_curve(v: int) -> float:
    """Loi de profondeur Roland : double tous les 10 pas (0 → 0, 100 → 1)."""
    return 0.0 if v <= 0 else 2.0 ** ((v - 100) / 10.0)


def partial_c4_hz(p: D50Partial, key_shift: int = 0, tune: int = 0) -> float:
    return 261.6256 * 2.0 ** ((p.coarse - 36 + key_shift) / 12.0 + (p.fine + tune) / 1200.0)


def partial_hz(p: D50Partial, note: int, key_shift: int = 0, tune: int = 0) -> float:
    """Hauteur d'un partial : coarse/fine à C4, puis keyfollow (octaves par octave)."""
    kf = KEYFOLLOW[p.keyfollow]
    cents = KEYFOLLOW_CENTS[p.keyfollow] * (note - 60) / 12.0
    return partial_c4_hz(p, key_shift, tune) * 2.0 ** (kf * (note - 60) / 12.0 + cents / 1200.0)


def _lfo_route(sel: int, depth: int) -> tuple[int, float]:
    """Sélecteur +1 -1 +2 -2 +3 -3 → (index LFO, profondeur signée 0..1)."""
    return sel // 2, (1.0 if sel % 2 == 0 else -1.0) * depth / 100.0


def _bias(point: int, level: int, note: int, magnitudes) -> float:
    """Bias TVF/TVA : décalage (octaves) appliqué au-delà (ou en deçà) du bias point."""
    above = (point >> 6) & 1
    ref = (point & 0x3F) + 33  # A1 = MIDI 33
    dist = (note - ref) if above else (ref - note)
    if dist <= 0:
        return 0.0
    return magnitudes[level] * dist / 12.0


# ----- enveloppes -----
class Env5:
    """Enveloppe D-50 : T1..T5 / L1..L3, sustain, end. ``log`` = segments linéaires en dB (TVA)."""

    def __init__(self, spec: D50Env, sr: int, log: bool = False, time_scale: float = 1.0):
        self.sr, self.log = sr, log
        self.times = [max(1, int(env_time(t) * time_scale * sr)) for t in spec.t]
        self.levels = [x / 100.0 for x in spec.l] + [spec.sustain / 100.0]
        self.end = spec.end / 100.0
        self.stage = 5  # 0..3 attaque/decay, 4 sustain, 5 idle, 6 release
        self.t = 0
        self.level = 0.0
        self.start = 0.0

    @property
    def finished(self) -> bool:
        return self.stage == 5

    def gate_on(self) -> None:
        self.stage, self.t, self.start = 0, 0, self.level

    def gate_off(self) -> None:
        if self.stage != 5:
            self.stage, self.t, self.start = 6, 0, self.level

    def _interp(self, a: float, b: float, frac: np.ndarray) -> np.ndarray:
        if not self.log:
            return a + (b - a) * frac
        da, db = 20 * np.log10(max(a, 1e-3)), 20 * np.log10(max(b, 1e-3))
        y = 10 ** ((da + (db - da) * frac) / 20.0)
        return np.where(y <= 1.05e-3, 0.0, y)

    def render(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.float32)
        filled = 0
        while filled < n:
            k = n - filled
            if self.stage == 5:
                out[filled:] = 0.0
                self.level = 0.0
                break
            if self.stage == 4:
                out[filled:] = self.levels[3]
                self.level = self.levels[3]
                break
            length = self.times[4] if self.stage == 6 else self.times[self.stage]
            target = self.end if self.stage == 6 else self.levels[self.stage]
            k = min(k, length - self.t)
            frac = (self.t + np.arange(1, k + 1)) / length
            seg = self._interp(self.start, target, frac)
            out[filled : filled + k] = seg
            self.level = float(seg[-1])
            self.t += k
            filled += k
            if self.t >= length:
                self.t, self.start = 0, target
                if self.stage == 6:
                    self.stage = 5 if target <= 1e-3 else 4
                    if self.stage == 4:
                        self.levels[3] = target
                else:
                    self.stage += 1
        return out


class PitchEnv:
    """P-ENV : L0 → L1 → L2 → sustain (tenue), release → end ; en demi-tons."""

    def __init__(self, c: D50Common, sr: int):
        span = 12.0 * (1.0 + 0.5 * c.penv_velo)  # ±1 / 1.5 / 2 octaves
        self.levels = [x / 50.0 * span for x in c.penv_l]  # L0 L1 L2 sus end
        self.times = [max(1, int(penv_time(t) * sr)) for t in c.penv_t]
        self.stage, self.t, self.level, self.start = 4, 0, 0.0, 0.0

    def gate_on(self) -> None:
        self.stage, self.t, self.level, self.start = 0, 0, self.levels[0], self.levels[0]

    def gate_off(self) -> None:
        self.stage, self.t, self.start = 3, 0, self.level

    def render(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.float32)
        filled = 0
        while filled < n:
            k = n - filled
            if self.stage == 4:
                out[filled:] = self.level
                break
            length = self.times[self.stage]
            target = self.levels[self.stage + 1]
            k = min(k, length - self.t)
            frac = (self.t + np.arange(1, k + 1)) / length
            seg = self.start + (target - self.start) * frac
            out[filled : filled + k] = seg
            self.level = float(seg[-1])
            self.t += k
            filled += k
            if self.t >= length:
                self.t, self.start = 0, target
                self.stage = 4 if self.stage in (2, 3) else self.stage + 1
                if self.stage == 4 and self.start == self.levels[3]:
                    self.level = self.levels[3]
        return out


class ToneLfo:
    """LFO du D-50 : TRI SAW SQU RND, delay avant apparition."""

    _WAVES = ("triangle", "saw", "square", "random")

    def __init__(self, wave: int, rate: int, delay: int, sr: int, rng: np.random.Generator):
        self.sr, self.rng = sr, rng
        self.wave = self._WAVES[wave]
        self.lfo = LFO(self.wave if self.wave != "random" else "square", lfo_rate_hz(rate), sr)
        self.delay = int(10.0 * (delay / 100.0) ** 2 * sr)
        self.elapsed = 0
        self.hold = 0.0

    def restart(self) -> None:
        self.lfo.phase, self.elapsed = 0.0, 0

    def render(self, n: int) -> np.ndarray:
        if self.wave == "random":
            ph0 = self.lfo.phase
            y = self.lfo.render(n)
            steps = int(self.lfo.phase + (n * self.lfo.rate / self.sr) + 1)  # nb de cycles
            if ph0 + n * self.lfo.rate / self.sr >= 1.0 or steps > 1:
                self.hold = float(self.rng.uniform(-1, 1))
            y = np.full(n, self.hold)
        else:
            y = self.lfo.render(n)
        if self.delay:
            t = self.elapsed + np.arange(1, n + 1)
            y = y * np.clip((t - self.delay) / max(1, self.delay), 0.0, 1.0)
        self.elapsed += n
        return y


# ----- partials -----
class _Partial:
    def __init__(
        self, p: D50Partial, sr: int, note: int, velocity: float, key_shift: int, tune: int
    ):
        self.p, self.sr, self.note = p, sr, note
        self.vel = velocity
        tkf = 2.0 ** (-p.tva_time_kf * (note - 60) / 48.0)
        self.tva = Env5(p.tva_env, sr, log=True, time_scale=tkf)
        self.tva.times[0] = max(
            1, int(self.tva.times[0] * (1.0 - 0.2 * p.tva_velo_time * velocity))
        )
        self.base_hz = partial_hz(p, note, key_shift, tune)
        # niveau : TVA level, vélocité (±50 → ±100 %), bias
        vel_gain = 1.0 + (p.tva_velo / 50.0) * (velocity - 0.5) * 2.0 * 0.5
        bias_db = -_bias(
            p.tva_bias_point, 12 - p.tva_bias_level, note, [0] + [k for k in range(1, 13)]
        )
        self.gain = (p.tva_level / 100.0) * max(0.0, vel_gain) * 10 ** (bias_db / 20.0) * LA_LEVEL
        self.phase = 0.0

    @property
    def active(self) -> bool:
        return not self.tva.finished

    def gate_off(self) -> None:
        self.tva.gate_off()

    def _amp_lfo(self, lfos: list[np.ndarray], n: int) -> np.ndarray:
        idx, depth = _lfo_route(self.p.tva_lfo, self.p.tva_lfo_depth)
        if depth == 0.0:
            return np.ones(n)
        return np.clip(1.0 + 0.5 * depth * lfos[idx], 0.0, 1.5)


_TVF_BIAS_MAG = (170, 120, 85, 54, 34, 21, 11, 0, -11, -21, -34, -54, -85, -120, -170)


class SynthPartial(_Partial):
    """Partial LA32 : carré / saw à flancs cosinus, résonance, TVF (enveloppe + LFO + keyfollow)."""

    def __init__(self, p, sr, note, velocity, key_shift, tune):
        super().__init__(p, sr, note, velocity, key_shift, tune)
        tkf = 2.0 ** (-p.tvf_time_kf * (note - 60) / 48.0)
        self.tvf = Env5(p.tvf_env, sr, log=False, time_scale=tkf)
        kf = CUTOFF_KF[min(p.cutoff_kf, 14)]
        bias_oct = _bias(p.bias_point, p.bias_level, note, _TVF_BIAS_MAG) / 100.0
        self.cut_base = partial_c4_hz(p, key_shift, tune) * 2.0 ** (
            (p.cutoff - 50) / 8.0 + kf * (note - 60) / 12.0 + bias_oct
        )
        vel_depth = 1.0 + (p.tvf_velo / 100.0) * (velocity - 0.5) * 2.0
        dkf = 2.0 ** (p.tvf_depth_kf * (note - 60) / 48.0)
        self.env_oct = (p.tvf_env_depth / 100.0) * 6.0 * vel_depth * dkf  # profondeur max ≈ 6 oct
        self.duty = 0.5 + 0.47 * p.pw / 100.0

    def gate_on(self) -> None:
        self.tva.gate_on()
        self.tvf.gate_on()

    def render(self, n: int, pitch_semis: np.ndarray, lfos: list[np.ndarray]) -> np.ndarray:
        p = self.p
        f = self.base_hz * 2.0 ** (pitch_semis / 12.0)
        ph = (self.phase + np.cumsum(f) / self.sr) % 1.0
        self.phase = float(ph[-1])
        # cutoff : base × enveloppe × LFO
        env = self.tvf.render(n)
        idx, depth = _lfo_route(p.tvf_lfo, p.tvf_lfo_depth)
        oct_mod = self.env_oct * (env - 0.0) + 3.0 * depth * lfos[idx]
        cutoff = np.clip(self.cut_base * 2.0**oct_mod, 20.0, self.sr * 0.45)
        duty = self.duty
        pidx, pdepth = _lfo_route(p.pw_lfo, p.pw_lfo_depth)
        if pdepth:
            duty = np.clip(duty + 0.47 * pdepth * lfos[pidx], 0.05, 0.97)
        if p.wave == "saw":
            wave = _soft_saw((2.0 * ph) % 1.0, 2.0 * f, cutoff)
        else:
            wave = _soft_square(ph, duty, f, cutoff)
        if p.resonance:
            wave = wave + _resonance(ph, f, cutoff, p.resonance / 30.0)
        amp = self.tva.render(n) * self.gain * self._amp_lfo(lfos, n)
        return wave * amp


def _soft_square(ph: np.ndarray, duty, f: np.ndarray, cutoff: np.ndarray) -> np.ndarray:
    """Carré à flancs cosinus de demi-largeur w = f / (2·cutoff) ; sinus quand cutoff ≤ f."""
    lim = np.minimum(duty, 1.0 - duty) * 0.5
    w = np.minimum(f / (2.0 * cutoff), lim)
    atten = np.minimum(1.0, cutoff / f)
    # flanc montant autour de 0 (et 1), descendant autour de duty
    d_up = np.minimum(ph, 1.0 - ph)  # distance au flanc montant
    d_dn = np.abs(ph - duty)
    y = np.where(ph < duty, 1.0, -1.0)
    up = np.sin(np.pi / 2.0 * np.clip(np.where(ph < 0.5, ph, ph - 1.0) / w, -1.0, 1.0))
    dn = -np.sin(np.pi / 2.0 * np.clip((ph - duty) / w, -1.0, 1.0))
    y = np.where(d_up < w, up, y)
    y = np.where(d_dn < w, dn, y)
    return y * atten


def _soft_saw(ph: np.ndarray, f: np.ndarray, cutoff: np.ndarray) -> np.ndarray:
    """Rampe descendante avec reset cosinus de demi-largeur w = f / (2·cutoff)."""
    w = np.minimum(f / (2.0 * cutoff), 0.25)
    atten = np.minimum(1.0, cutoff / f)
    y = 1.0 - 2.0 * ph
    d = np.minimum(ph, 1.0 - ph)
    x = np.where(ph < 0.5, ph, ph - 1.0) / w  # -1..1 autour du reset
    edge = -np.sin(np.pi / 2.0 * np.clip(x, -1.0, 1.0)) * (1.0 - w)
    return np.where(d < w, edge, y) * atten


def _resonance(ph: np.ndarray, f: np.ndarray, cutoff: np.ndarray, res: float) -> np.ndarray:
    """Sinus amorti à la fréquence de coupure, relancé à chaque cycle."""
    tc = ph / f
    return (
        res**1.5
        * np.sin(2.0 * np.pi * cutoff * tc)
        * np.exp(-cutoff * tc * (1.2 - res))
        * np.sin(np.pi * ph)
    )


class PcmPartial(_Partial):
    """Partial PCM : lecture d'une onde ``d50_pcm`` à ``f × 2048`` mots par seconde."""

    def __init__(self, p, sr, note, velocity, key_shift, tune):
        super().__init__(p, sr, note, velocity, key_shift, tune)
        self.wave, self.looped = pcm_wave(p.pcm)
        self.pos = 0.0
        self.done = False

    @property
    def active(self) -> bool:
        return not self.tva.finished and not self.done

    def gate_on(self) -> None:
        self.tva.gate_on()
        self.pos, self.done = 0.0, False

    def render(self, n: int, pitch_semis: np.ndarray, lfos: list[np.ndarray]) -> np.ndarray:
        f = self.base_hz * 2.0 ** (pitch_semis / 12.0)
        pos = self.pos + np.cumsum(f * WORDS_PER_HZ / self.sr)
        self.pos = float(pos[-1])
        length = len(self.wave)
        if self.looped:
            i0 = np.floor(pos).astype(np.int64) % length
            fr = pos - np.floor(pos)
            y = self.wave[i0] * (1 - fr) + self.wave[(i0 + 1) % length] * fr
        else:
            i0 = np.floor(pos).astype(np.int64)
            fr = pos - i0
            valid = i0 < length - 1
            i0c = np.clip(i0, 0, length - 2)
            y = np.where(valid, self.wave[i0c] * (1 - fr) + self.wave[i0c + 1] * fr, 0.0)
            if not valid.any():
                self.done = True
        amp = self.tva.render(n) * self.gain * self._amp_lfo(lfos, n)
        return y * amp


# ----- tone voice -----
_STRUCT_PCM = {
    1: (False, False),
    2: (False, False),
    3: (True, False),
    4: (True, False),
    5: (False, True),
    6: (True, True),
    7: (True, True),
}
_STRUCT_RING = {1: False, 2: True, 3: False, 4: True, 5: True, 6: False, 7: True}


class ToneVoice:
    """Une note sur un tone : 2 partials, P-ENV, 3 LFO, structure."""

    def __init__(
        self, tone: D50Tone, sr: int, rng, note: int, velocity: float, key_shift: int, tune: int
    ):
        c = tone.common
        self.c, self.sr, self.note = c, sr, note
        is_pcm = _STRUCT_PCM[c.structure]
        self.partials: list[_Partial] = []
        for i, p in enumerate(tone.partials):
            cls = PcmPartial if is_pcm[i] else SynthPartial
            self.partials.append(cls(p, sr, note, velocity, key_shift, tune))
        self.ring = _STRUCT_RING[c.structure]
        self.mute = [(c.partial_mute >> i) & 1 for i in range(2)]
        bal = c.partial_balance / 100.0
        self.bal = (min(1.0, 2.0 * (1.0 - bal)), min(1.0, 2.0 * bal))
        self.penv = PitchEnv(c, sr)
        self.lfos = [ToneLfo(lf.wave, lf.rate, lf.delay, sr, rng) for lf in c.lfos]
        self.vib_cents = 600.0 * depth_curve(c.pmod_lfo_depth)
        self.age = 0
        self.gate_on()

    def gate_on(self) -> None:
        for p in self.partials:
            p.gate_on()
        self.penv.gate_on()
        for lf in self.lfos:
            lf.restart()

    def gate_off(self) -> None:
        for p in self.partials:
            p.gate_off()
        self.penv.gate_off()

    @property
    def active(self) -> bool:
        return any(p.active and m for p, m in zip(self.partials, self.mute, strict=True))

    def render(self, n: int) -> np.ndarray:
        lfos = [lf.render(n) for lf in self.lfos]
        penv = self.penv.render(n)
        outs = []
        for i, p in enumerate(self.partials):
            if not self.mute[i] or not p.active:
                outs.append(np.zeros(n))
                continue
            semis = np.zeros(n)
            if p.p.penv_mode == 1:
                semis = semis + penv
            elif p.p.penv_mode == 2:
                semis = semis - penv
            if p.p.lfo_mode in (1, 2):
                sign = 1.0 if p.p.lfo_mode == 1 else -1.0
                semis = semis + sign * self.vib_cents / 100.0 * lfos[0]
            outs.append(p.render(n, semis, lfos))
        a, b = outs[0] * self.bal[0], outs[1] * self.bal[1]
        if self.ring:
            return a + a * b / LA_LEVEL
        return a + b


# ----- effets de tone / patch -----
def _shelf_low(fc: float, gain_db: float, sr: int):
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2.0 * np.sqrt(2.0)
    b = np.array(
        [
            a * ((a + 1) - (a - 1) * cw + 2 * np.sqrt(a) * alpha),
            2 * a * ((a - 1) - (a + 1) * cw),
            a * ((a + 1) - (a - 1) * cw - 2 * np.sqrt(a) * alpha),
        ]
    )
    aa = np.array(
        [
            (a + 1) + (a - 1) * cw + 2 * np.sqrt(a) * alpha,
            -2 * ((a - 1) + (a + 1) * cw),
            (a + 1) + (a - 1) * cw - 2 * np.sqrt(a) * alpha,
        ]
    )
    return b / aa[0], aa / aa[0]


def _peak(fc: float, q: float, gain_db: float, sr: int):
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    b = np.array([1 + alpha * a, -2 * cw, 1 - alpha * a])
    aa = np.array([1 + alpha / a, -2 * cw, 1 - alpha / a])
    return b / aa[0], aa / aa[0]


class ToneEq(Effect):
    """EQ 2 bandes du tone : low shelf + peak haut."""

    def __init__(self, c: D50Common, sr: int):
        self.filters = []
        if c.eq_low_gain:
            self.filters.append(_shelf_low(EQ_LOW_HZ[c.eq_low_freq], c.eq_low_gain, sr))
        if c.eq_high_gain:
            self.filters.append(
                _peak(EQ_HIGH_HZ[c.eq_high_freq], EQ_HIGH_Q[c.eq_high_q], c.eq_high_gain, sr)
            )
        self.zi = [[np.zeros(2), np.zeros(2)] for _ in self.filters]

    def process(self, x: np.ndarray) -> np.ndarray:
        for i, (b, a) in enumerate(self.filters):
            y = np.empty_like(x)
            for ch in (0, 1):
                y[:, ch], self.zi[i][ch] = lfilter(b, a, x[:, ch], zi=self.zi[i][ch])
            x = y
        return x


class ToneChorus(Effect):
    """8 types de chorus D-50 sur les effets existants (chorus, flanger, tremolo, dimension)."""

    def __init__(self, c: D50Common, sr: int, bpm: float):
        rate = 0.05 * 200.0 ** (c.chorus_rate / 100.0)  # 0,05 .. 10 Hz
        depth = c.chorus_depth / 100.0
        mix = c.chorus_balance / 100.0
        t = c.chorus_type
        self.fx: list[Effect] = []
        self.trem = None
        if t in (1, 2, 5, 7, 8):
            d = 0.002 + 0.004 * depth * (1.5 if t == 5 else 1.0)
            self.fx.append(Chorus(sr, bpm, rate=rate, depth=d, mix=mix * (1.0 if t != 8 else 0.7)))
            if t == 8:
                self.fx.append(Chorus(sr, bpm, rate=rate * 0.7, depth=d * 0.6, mix=0.5 * mix))
        if t in (3, 4):
            self.fx.append(
                Flanger(
                    sr,
                    bpm,
                    rate=rate,
                    depth=0.0015 + 0.003 * depth,
                    base=0.002 if t == 3 else 0.004,
                    feedback=0.4 + 0.4 * depth,
                    mix=mix,
                )
            )
        if t in (6, 7):
            self.trem = (LFO("sine", rate, sr), depth * mix)

    def process(self, x: np.ndarray) -> np.ndarray:
        for f in self.fx:
            x = f.process(x)
        if self.trem is not None:
            lfo, d = self.trem
            x = x * (1.0 - 0.5 * d * (1.0 - lfo.render(len(x))))[:, None]
        return x.astype(np.float32)


def build_reverb(rtype: int, balance: int, sr: int, bpm: float) -> Effect | None:
    if balance <= 0:
        return None
    kind, prm = REVERB_TYPES[rtype]
    return _REVERB_CLS[kind](sr, bpm, mix=balance / 100.0, **prm)


# ----- synth -----
class D50Synth:
    """Roland D-50 polyphonique — même interface que ``Synth`` / ``Dx7Synth``."""

    voices: list = []

    def __init__(self, patch: D50PatchModel, sr: int, rng: np.random.Generator, bpm: float):
        self.sr, self.rng, self.bpm = sr, rng, bpm
        self.counter = 0
        self.set_patch(patch)

    def set_patch(self, patch: D50PatchModel) -> None:
        self.patch = patch
        self.notes: dict[str, list[ToneVoice]] = {"upper": [], "lower": []}
        self._build_fx()

    def _build_fx(self) -> None:
        p, sr, bpm = self.patch, self.sr, self.bpm
        self.eq = {t: ToneEq(getattr(p, t).common, sr) for t in ("upper", "lower")}
        self.chorus = {t: ToneChorus(getattr(p, t).common, sr, bpm) for t in ("upper", "lower")}
        self.reverb = build_reverb(p.reverb_type, p.reverb_balance, sr, bpm)
        self.effects = build_effects([e.model_dump() for e in p.effects], sr, bpm)

    def update_patch(self, patch: D50PatchModel) -> None:
        old = self.patch
        self.patch = patch
        structural = (
            any(
                getattr(old, t).common.structure != getattr(patch, t).common.structure
                for t in ("upper", "lower")
            )
            or old.key_mode != patch.key_mode
        )
        if structural:
            self.notes = {"upper": [], "lower": []}
        self._build_fx()

    def set_bpm(self, bpm: float) -> None:
        self.bpm = bpm
        self._build_fx()

    # ----- clavier -----
    def _tones_for(self, note: int) -> list[str]:
        km = self.patch.key_mode % 4 if self.patch.key_mode != 8 else 3
        if km == 0:
            return ["upper"]
        if km in (1, 3):
            return ["upper", "lower"]
        split = 36 + self.patch.split
        return ["lower"] if note < split else ["upper"]

    def note_on(self, note: int, velocity: float) -> None:
        self.counter += 1
        p = self.patch
        limit = p.polyphony if self.patch.key_mode % 4 == 0 else max(1, p.polyphony // 2)
        for t in self._tones_for(note):
            voices = self.notes[t]
            shift = p.key_shift_upper if t == "upper" else p.key_shift_lower
            tune = p.tune_upper if t == "upper" else p.tune_lower
            v = ToneVoice(getattr(p, t), self.sr, self.rng, note, velocity, shift, tune)
            v.age = self.counter
            voices.append(v)
            if len(voices) > limit:
                voices.sort(key=lambda x: x.age)
                voices.pop(0)

    def note_off(self, note: int) -> None:
        for voices in self.notes.values():
            for v in voices:
                if v.note == note:
                    v.gate_off()

    # ----- rendu -----
    def _render_tones(self, n: int) -> np.ndarray:
        p = self.patch
        out = np.zeros((n, 2), dtype=np.float32)
        bal = p.tone_balance / 100.0
        gains = {"upper": min(1.0, 2.0 * (1.0 - bal)), "lower": min(1.0, 2.0 * bal)}
        if p.key_mode % 4 == 0 and p.key_mode != 8:
            gains["upper"] = 1.0
        for t, voices in self.notes.items():
            alive = [v for v in voices if v.active]
            self.notes[t] = alive
            if not alive:
                continue
            mono = np.zeros(n)
            for v in alive:
                mono += v.render(n)
            st = np.stack([mono, mono], axis=1).astype(np.float32) * gains[t]
            st = self.eq[t].process(st)
            st = self.chorus[t].process(st)
            out += st
        if self.reverb is not None:
            out = self.reverb.process(out)
        return out * (p.patch_volume / 100.0)

    def render(self, n: int, events: list[NoteEvent], gain: np.ndarray | None = None) -> np.ndarray:
        out = np.zeros((n, 2), dtype=np.float32)
        pos = 0
        for ev in sorted(events):
            off = min(max(ev.offset, pos), n)
            if off > pos:
                out[pos:off] = self._render_tones(off - pos)
                pos = off
            if ev.on:
                self.note_on(ev.note, ev.velocity)
            else:
                self.note_off(ev.note)
        if pos < n:
            out[pos:] = self._render_tones(n - pos)
        out *= self.patch.volume
        if gain is not None:
            out *= gain[:, None]
        for fx in self.effects:
            out = fx.process(out)
        return out


# ----- sysex -----
PATCH_BASE = 0x8000  # adresse 02-00-00
PATCH_SIZE = 448
_CHARS = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890-"


def _parse_dt1(raw: bytes) -> dict[int, int]:
    """Reconstitue l'espace d'adresses des messages DT1 (F0 41 dev 14 12 aa bb cc … cs F7)."""
    mem: dict[int, int] = {}
    i = 0
    while True:
        a = raw.find(b"\xf0", i)
        if a < 0:
            break
        b = raw.find(b"\xf7", a)
        if b < 0:
            break
        m = raw[a : b + 1]
        i = b + 1
        if len(m) < 10 or m[1] != 0x41 or m[3] != 0x14 or m[4] != 0x12:
            continue
        addr = m[5] * 16384 + m[6] * 128 + m[7]
        data = m[8:-2]
        if (128 - (sum(m[5:8]) + sum(data)) % 128) % 128 != m[-2]:
            raise ValueError(
                f"checksum sysex invalide à l'adresse {m[5]:02X}-{m[6]:02X}-{m[7]:02X}"
            )
        for k, v in enumerate(data):
            mem[addr + k] = v
    if not mem:
        raise ValueError("aucun message DT1 Roland D-50")
    return mem


def _name(block: bytes, n: int) -> str:
    return "".join(_CHARS[v & 0x3F] for v in block[:n]).strip()


def _env(b: bytes, t0: int, l0: int, s: int, e: int) -> D50Env:
    return D50Env(
        t=[min(100, b[t0 + k]) for k in range(5)],
        l=[min(100, b[l0 + k]) for k in range(3)],
        sustain=min(100, b[s]),
        end=100 if b[e] else 0,
    )


def _partial(b: bytes) -> D50Partial:
    return D50Partial(
        coarse=min(72, b[0]),
        fine=int(b[1]) - 50,
        keyfollow=min(16, b[2]),
        lfo_mode=min(3, b[3]),
        penv_mode=min(2, b[4]),
        bend_mode=min(2, b[5]),
        wave="saw" if b[6] else "square",
        pcm=min(100, b[7] + 1),
        pw=min(100, b[8]),
        pw_velo=int(min(14, b[9])) - 7,
        pw_lfo=min(5, b[10]),
        pw_lfo_depth=min(100, b[11]),
        cutoff=min(100, b[13]),
        resonance=min(30, b[14]),
        cutoff_kf=min(14, b[15]),
        bias_point=b[16] & 0x7F,
        bias_level=min(14, b[17]),
        tvf_env_depth=min(100, b[18]),
        tvf_velo=min(100, b[19]),
        tvf_depth_kf=min(4, b[20]),
        tvf_time_kf=min(4, b[21]),
        tvf_env=_env(b, 22, 27, 30, 31),
        tvf_lfo=min(5, b[32]),
        tvf_lfo_depth=min(100, b[33]),
        tva_level=min(100, b[35]),
        tva_velo=int(min(100, b[36])) - 50,
        tva_bias_point=b[37] & 0x7F,
        tva_bias_level=min(12, b[38]),
        tva_env=_env(b, 39, 44, 47, 48),
        tva_velo_time=min(4, b[49]),
        tva_time_kf=min(4, b[50]),
        tva_lfo=min(5, b[51]),
        tva_lfo_depth=min(100, b[52]),
    )


def _common(b: bytes) -> D50Common:
    return D50Common(
        structure=min(6, b[10]) + 1,
        penv_velo=min(2, b[11]),
        penv_time_kf=min(4, b[12]),
        penv_t=[min(50, b[13 + k]) for k in range(4)],
        penv_l=[int(np.clip(int(b[17 + k]) - 50, -50, 50)) for k in range(5)],
        pmod_lfo_depth=min(100, b[22]),
        pmod_lever=min(100, b[23]),
        pmod_at=min(100, b[24]),
        lfos=[
            D50Lfo(
                wave=min(3, b[25 + 4 * i]),
                rate=min(100, b[26 + 4 * i]),
                delay=min(100, b[27 + 4 * i]),
                sync=min(2, b[28 + 4 * i]),
            )
            for i in range(3)
        ],
        eq_low_freq=min(15, b[37]),
        eq_low_gain=int(np.clip(int(b[38]) - 12, -12, 12)),
        eq_high_freq=min(21, b[39]),
        eq_high_q=min(8, b[40]),
        eq_high_gain=int(np.clip(int(b[41]) - 12, -12, 12)),
        chorus_type=min(7, b[42]) + 1,
        chorus_rate=min(100, b[43]),
        chorus_depth=min(100, b[44]),
        chorus_balance=min(100, b[45]),
        partial_mute=b[46] & 0x3,
        partial_balance=min(100, b[47]),
    )


def d50_patch_from_bytes(body: bytes) -> D50PatchModel:
    """Un patch D-50 de 448 octets (7 blocs de 64) → ``D50PatchModel``."""
    if len(body) != PATCH_SIZE:
        raise ValueError(f"patch D-50 : {len(body)} octets au lieu de {PATCH_SIZE}")
    blk = [bytes(body[i * 64 : (i + 1) * 64]) for i in range(7)]
    pb = blk[6]
    upper = D50Tone(
        name=_name(blk[2], 10),
        partials=[_partial(blk[0]), _partial(blk[1])],
        common=_common(blk[2]),
    )
    lower = D50Tone(
        name=_name(blk[5], 10),
        partials=[_partial(blk[3]), _partial(blk[4])],
        common=_common(blk[5]),
    )
    return D50PatchModel(
        name=_name(pb, 18) or "D50",
        upper=upper,
        lower=lower,
        key_mode=min(8, pb[18]),
        split=min(60, pb[19]),
        key_shift_upper=int(np.clip(int(pb[22]) - 24, -24, 24)),
        key_shift_lower=int(np.clip(int(pb[23]) - 24, -24, 24)),
        tune_upper=int(np.clip(int(pb[24]) - 50, -50, 50)),
        tune_lower=int(np.clip(int(pb[25]) - 50, -50, 50)),
        reverb_type=min(31, pb[30]) + 1,
        reverb_balance=min(100, pb[31]),
        patch_volume=min(100, pb[32]),
        tone_balance=min(100, pb[33]),
    )


def d50_sysex_to_patches(raw: bytes) -> list[D50PatchModel]:
    """Bank sysex D-50 (bulk dump 64 patches à 02-00-00) → liste de ``D50PatchModel``."""
    mem = _parse_dt1(raw)
    patches = []
    for i in range(64):
        base = PATCH_BASE + i * PATCH_SIZE
        if base + PATCH_SIZE - 1 not in mem:
            break
        body = bytes(mem.get(base + k, 0) for k in range(PATCH_SIZE))
        patches.append(d50_patch_from_bytes(body))
    if not patches:
        raise ValueError("aucun patch à l'adresse 02-00-00")
    return patches


__all__ = ["D50Synth", "d50_sysex_to_patches", "d50_patch_from_bytes", "PCM_SR"]
