"""Band-limited oscillators (polyBLEP) with unison, detune and stereo
spread.
"""

from __future__ import annotations

import numpy as np

WAVES = ("saw", "square", "triangle", "sine", "noise", "fm")


def _polyblep(t: np.ndarray, dt: float) -> np.ndarray:
    """Polyblep."""
    out = np.zeros_like(t)
    m = t < dt
    x = t[m] / dt
    out[m] = x + x - x * x - 1.0
    m2 = t > 1.0 - dt
    x = (t[m2] - 1.0) / dt
    out[m2] = x * x + x + x + 1.0
    return out


def render_wave(
    wave: str,
    phase: np.ndarray,
    dt: float,
    pwm: float = 0.5,
    rng=None,
    fm_ratio: float = 2.0,
    fm_index: float = 0.0,
) -> np.ndarray:
    """Render one waveform from phase in [0,1). dt = phase increment per sample."""
    if wave == "fm":  # 2-operator phase modulation
        mod = np.sin(2.0 * np.pi * phase * fm_ratio)
        return np.sin(2.0 * np.pi * phase + fm_index * mod)
    if wave == "sine":
        return np.sin(2.0 * np.pi * phase)
    if wave == "saw":
        return (2.0 * phase - 1.0) - _polyblep(phase, dt)
    if wave == "square":
        pwm = float(np.clip(pwm, 0.05, 0.95))
        y = np.where(phase < pwm, 1.0, -1.0)
        y = y + _polyblep(phase, dt)
        y = y - _polyblep((phase + 1.0 - pwm) % 1.0, dt)
        return y
    if wave == "triangle":
        return 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
    if wave == "noise":
        rng = rng or np.random.default_rng()
        return rng.uniform(-1.0, 1.0, size=phase.shape)
    raise ValueError(f"unknown wave {wave!r}")


class Oscillator:
    """Oscillator."""

    def __init__(  # noqa: PLR0913 - DSP voice needs many params (bundled in OscSpec in caller)
        self,
        wave: str,
        sr: int,
        rng: np.random.Generator,
        unison: int = 1,
        detune: float = 0.0,
        octave: int = 0,
        semi: int = 0,
        level: float = 1.0,
        pwm: float = 0.5,
        spread: float = 1.0,
        fm_ratio: float = 2.0,
        fm_index: float = 0.0,
    ):
        """Initialize oscillator voice."""
        if wave not in WAVES:
            raise ValueError(f"unknown wave {wave!r}")
        self.wave, self.sr, self.rng = wave, sr, rng
        self.unison = max(1, int(unison))
        self.level, self.pwm = float(level), float(pwm)
        self.fm_ratio, self.fm_index = float(fm_ratio), float(fm_index)
        self.transpose = 2.0 ** ((octave * 12 + semi) / 12.0)
        self.phases = rng.uniform(0.0, 1.0, self.unison)
        if self.unison > 1:
            cents = np.linspace(-1.0, 1.0, self.unison) * detune
            pans = np.linspace(-1.0, 1.0, self.unison) * spread
        else:
            cents, pans = np.zeros(1), np.zeros(1)
        self.ratios = 2.0 ** (cents / 1200.0)
        angle = (pans + 1.0) * np.pi / 4.0
        self.gain_l, self.gain_r = np.cos(angle), np.sin(angle)
        self.norm = self.level / np.sqrt(self.unison)

    def render(self, freq_hz: float, n: int, pwm: float | None = None) -> np.ndarray:
        """Render."""
        out = np.zeros((n, 2), dtype=np.float64)
        pwm = self.pwm if pwm is None else pwm
        idx = np.arange(1, n + 1)
        for i in range(self.unison):
            dt = freq_hz * self.transpose * self.ratios[i] / self.sr
            phase = self.phases[i] + dt * idx
            self.phases[i] = phase[-1] % (1.0 if self.wave != "fm" else 64.0)
            if self.wave != "fm":
                phase = phase % 1.0
            y = render_wave(self.wave, phase, dt, pwm, self.rng, self.fm_ratio, self.fm_index)
            out[:, 0] += y * self.gain_l[i]
            out[:, 1] += y * self.gain_r[i]
        return (out * self.norm).astype(np.float32)
