"""RBJ biquad filter, coefficients per block, state carried by scipy lfilter."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def biquad_coeffs(kind: str, cutoff_hz: float, resonance: float, sr: int):
    """Biquad coeffs."""
    fc = float(np.clip(cutoff_hz, 20.0, sr * 0.45))
    q = 0.5 + float(np.clip(resonance, 0.0, 1.0)) * 9.5
    w0 = 2.0 * np.pi * fc / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    if kind == "lp":
        b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
    elif kind == "hp":
        b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
    elif kind == "bp":
        b = np.array([alpha, 0.0, -alpha])
    else:
        raise ValueError(f"unknown filter kind {kind!r}")
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


class Filter:
    """Filter."""
    def __init__(self, kind: str, sr: int):
        """Initialize."""
        self.kind, self.sr = kind, sr
        self.zi = [np.zeros(2), np.zeros(2)]

    def process(self, x: np.ndarray, cutoff_hz: float, resonance: float) -> np.ndarray:
        """Process."""
        b, a = biquad_coeffs(self.kind, cutoff_hz, resonance, self.sr)
        y = np.empty_like(x, dtype=np.float32)
        for ch in (0, 1):
            yc, self.zi[ch] = lfilter(b, a, x[:, ch].astype(np.float64), zi=self.zi[ch])
            y[:, ch] = yc
        return y
