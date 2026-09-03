from __future__ import annotations

import numpy as np


class LFO:
    """Lfo."""

    def __init__(self, wave: str, rate_hz: float, sr: int, phase: float = 0.0):
        """Initialize."""
        self.wave, self.rate, self.sr, self.phase = wave, float(rate_hz), sr, float(phase)

    def render(self, n: int) -> np.ndarray:
        """Render."""
        dt = self.rate / self.sr
        ph = (self.phase + dt * np.arange(1, n + 1)) % 1.0
        self.phase = float(ph[-1])
        if self.wave == "sine":
            return np.sin(2 * np.pi * ph)
        if self.wave == "triangle":
            return 2.0 * np.abs(2.0 * ph - 1.0) - 1.0
        if self.wave == "square":
            return np.where(ph < 0.5, 1.0, -1.0)
        if self.wave == "saw":
            return 2.0 * ph - 1.0
        raise ValueError(f"unknown lfo wave {self.wave!r}")
