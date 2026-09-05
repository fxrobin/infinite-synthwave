"""RBJ biquad filter, coefficients per block, state carried by scipy lfilter."""

from __future__ import annotations

import numpy as np

from .blocks import lfilter


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
        # state for both channels at once: lfilter runs along axis 0 in a single call,
        # which halves the (dominant) scipy call overhead per block.
        self.zi = np.zeros((2, 2))
        self._key: tuple[float, float] | None = None
        self._coeffs: tuple[np.ndarray, np.ndarray] | None = None

    def _coeffs_for(self, cutoff_hz: float, resonance: float):
        """Biquad coefficients, memoised while cutoff/resonance are unchanged."""
        key = (cutoff_hz, resonance)
        if key != self._key:
            self._coeffs = biquad_coeffs(self.kind, cutoff_hz, resonance, self.sr)
            self._key = key
        return self._coeffs

    def process(self, x: np.ndarray, cutoff_hz: float, resonance: float) -> np.ndarray:
        """Process."""
        b, a = self._coeffs_for(cutoff_hz, resonance)
        y, self.zi = lfilter(b, a, x.astype(np.float64), axis=0, zi=self.zi)
        return y.astype(np.float32)
