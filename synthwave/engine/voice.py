from __future__ import annotations

import numpy as np

from ..patches.model import PatchModel
from .envelope import ADSR
from .filter import Filter
from .lfo import LFO
from .oscillator import Oscillator


def midi_to_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)


class Voice:
    def __init__(self, patch: PatchModel, sr: int, rng: np.random.Generator):
        self.patch, self.sr = patch, sr
        self.oscs = [Oscillator(o.wave, sr, rng, o.unison, o.detune, o.octave, o.semi, o.level,
                                o.pwm, o.spread) for o in patch.oscillators]
        e = patch.amp_env
        self.amp_env = ADSR(e.attack, e.decay, e.sustain, e.release, sr)
        self.filter = Filter(patch.filter.type, sr) if patch.filter else None
        fe = patch.filter.env if patch.filter else None
        self.filt_env = ADSR(fe.attack, fe.decay, fe.sustain, fe.release, sr) if fe else None
        self.lfo = LFO(patch.lfo.wave, patch.lfo.rate, sr) if patch.lfo else None
        self.note, self.velocity, self.freq, self.target_freq = 60, 0.0, 261.6, 261.6
        self.glide_coef = float(np.exp(-1.0 / (patch.glide * sr))) if patch.glide > 0 else 0.0
        self.age = 0

    @property
    def active(self) -> bool:
        return not self.amp_env.finished

    def note_on(self, note: int, velocity: float, legato: bool = False) -> None:
        self.note, self.velocity = note, float(velocity)
        self.target_freq = midi_to_hz(note)
        if not (legato and self.glide_coef):
            self.freq = self.target_freq
        if not legato or not self.active:
            self.amp_env.gate_on()
            if self.filt_env:
                self.filt_env.gate_on()

    def note_off(self) -> None:
        self.amp_env.gate_off()
        if self.filt_env:
            self.filt_env.gate_off()

    def render(self, n: int) -> np.ndarray:
        p = self.patch
        lfo = self.lfo.render(n) if self.lfo else None
        lfo_mean = float(lfo.mean()) if lfo is not None else 0.0
        if self.glide_coef:
            self.freq = self.target_freq + (self.freq - self.target_freq) * self.glide_coef ** n
        freq = self.freq
        pwm = None
        if p.lfo and p.lfo.target == "pitch":
            freq *= 2.0 ** (p.lfo.amount * lfo_mean / 12.0)
        if p.lfo and p.lfo.target == "pwm":
            pwm = 0.5 + 0.45 * p.lfo.amount * lfo_mean
        sig = np.zeros((n, 2), dtype=np.float32)
        for osc in self.oscs:
            sig += osc.render(freq, n, pwm)
        if self.filter:
            f = p.filter
            cutoff = f.cutoff + f.key_track * (freq - 261.6)
            if self.filt_env:
                cutoff += f.env.amount * float(self.filt_env.render(n).mean())
            if p.lfo and p.lfo.target == "cutoff":
                cutoff += p.lfo.amount * lfo_mean
            sig = self.filter.process(sig, cutoff, f.resonance)
        amp = self.amp_env.render(n) * self.velocity
        if p.lfo and p.lfo.target == "amp":
            amp = amp * (1.0 - p.lfo.amount * 0.5 * (1.0 - lfo))
        return sig * amp[:, None]
