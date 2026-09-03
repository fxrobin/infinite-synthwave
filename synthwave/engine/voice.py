from __future__ import annotations

import numpy as np

from ..patches.model import PatchModel
from .envelope import ADSR
from .filter import Filter
from .lfo import LFO
from .oscillator import Oscillator


def midi_to_hz(note: float) -> float:
    """Midi to hz."""
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)


class Voice:
    """Voice."""

    def __init__(self, patch: PatchModel, sr: int, rng: np.random.Generator):
        """Initialize."""
        self.patch, self.sr = patch, sr
        self.oscs = [
            Oscillator(
                o.wave,
                sr,
                rng,
                o.unison,
                o.detune,
                o.octave,
                o.semi,
                o.level,
                o.pwm,
                o.spread,
                o.fm_ratio,
                o.fm_index,
            )
            for o in patch.oscillators
        ]
        e = patch.amp_env
        self.amp_env = ADSR(e.attack, e.decay, e.sustain, e.release, sr)
        self.filter = Filter(patch.filter.type, sr) if patch.filter else None
        fe = patch.filter.env if patch.filter else None
        self.filt_env = ADSR(fe.attack, fe.decay, fe.sustain, fe.release, sr) if fe else None
        self.lfo = LFO(patch.lfo.wave, patch.lfo.rate, sr) if patch.lfo else None
        self.note, self.velocity, self.freq, self.target_freq = 60, 0.0, 261.6, 261.6
        self.glide_coef = float(np.exp(-1.0 / (patch.glide * sr))) if patch.glide > 0 else 0.0
        self.age = 0

    def retune(self, patch: PatchModel) -> None:
        """Update continuous parameters in place (same structure as the current
        patch)."""
        self.patch = patch
        sr = self.sr
        for osc, o in zip(self.oscs, patch.oscillators, strict=True):
            osc.level, osc.pwm = float(o.level), float(o.pwm)
            osc.fm_ratio, osc.fm_index = float(o.fm_ratio), float(o.fm_index)
            osc.norm = osc.level / np.sqrt(osc.unison)
            if osc.unison > 1:
                cents = np.linspace(-1.0, 1.0, osc.unison) * o.detune
                pans = np.linspace(-1.0, 1.0, osc.unison) * o.spread
                osc.ratios = 2.0 ** (cents / 1200.0)
                angle = (pans + 1.0) * np.pi / 4.0
                osc.gain_l, osc.gain_r = np.cos(angle), np.sin(angle)
        e = patch.amp_env
        self._set_env(self.amp_env, e.attack, e.decay, e.sustain, e.release)
        if self.filt_env and patch.filter and patch.filter.env:
            fe = patch.filter.env
            self._set_env(self.filt_env, fe.attack, fe.decay, fe.sustain, fe.release)
        if self.lfo and patch.lfo:
            self.lfo.rate, self.lfo.wave = float(patch.lfo.rate), patch.lfo.wave
        self.glide_coef = float(np.exp(-1.0 / (patch.glide * sr))) if patch.glide > 0 else 0.0

    def _set_env(self, env: ADSR, a: float, d: float, sus: float, r: float) -> None:
        """Set env."""
        env.a, env.d = max(1, int(a * self.sr)), max(1, int(d * self.sr))
        env.s, env.r = float(np.clip(sus, 0.0, 1.0)), max(1, int(r * self.sr))

    @property
    def active(self) -> bool:
        """Active."""
        return not self.amp_env.finished

    def note_on(self, note: int, velocity: float, legato: bool = False) -> None:
        """Note on."""
        self.note, self.velocity = note, float(velocity)
        self.target_freq = midi_to_hz(note)
        if not (legato and self.glide_coef):
            self.freq = self.target_freq
        if not legato or not self.active:
            self.amp_env.gate_on()
            if self.filt_env:
                self.filt_env.gate_on()

    def note_off(self) -> None:
        """Note off."""
        self.amp_env.gate_off()
        if self.filt_env:
            self.filt_env.gate_off()

    def render(self, n: int) -> np.ndarray:
        """Render."""
        p = self.patch
        lfo = self.lfo.render(n) if self.lfo else None
        lfo_mean = float(lfo.mean()) if lfo is not None else 0.0
        if self.glide_coef:
            self.freq = self.target_freq + (self.freq - self.target_freq) * self.glide_coef**n
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
