"""Transition one-shots (reverse cymbal, uplifter, scream, impact) synthesised per tempo."""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .events import NoteEvent
from .filter import biquad_coeffs

RISER_NOTES = {"reverse_cymbal": 60, "uplifter": 61, "scream": 62, "impact": 63,
               "reverse_short": 64}
NOTE_TO_RISER = {v: k for k, v in RISER_NOTES.items()}


def _stereo(x: np.ndarray, gain: float) -> np.ndarray:
    x = x / max(1e-9, float(np.abs(x).max())) * gain
    return np.stack([x, x], axis=1).astype(np.float32)


def _sweep_lp(x: np.ndarray, f_start: float, f_end: float, sr: int, chunks: int = 32):
    """Lowpass whose cutoff glides geometrically from f_start to f_end over the buffer."""
    out = np.empty_like(x)
    zi = np.zeros(2)
    edges = np.linspace(0, len(x), chunks + 1).astype(int)
    for i in range(chunks):
        fc = f_start * (f_end / f_start) ** ((i + 0.5) / chunks)
        b, a = biquad_coeffs("lp", fc, 0.3, sr)
        out[edges[i]:edges[i + 1]], zi = lfilter(b, a, x[edges[i]:edges[i + 1]], zi=zi)
    return out


class RiserKit:
    def __init__(self, sr: int, rng: np.random.Generator, bpm: float, volume: float = 0.6):
        self.sr, self.rng, self.volume = sr, rng, volume
        self.active: list[tuple[np.ndarray, int, float]] = []
        self.set_bpm(bpm)

    def set_bpm(self, bpm: float) -> None:
        self.bpm = bpm
        sr, rng = self.sr, self.rng
        bar = int(sr * 16 * 60.0 / bpm / 4.0)
        # reverse cymbal: noise swelling exponentially, opening filter, hard stop at the bar
        t = np.arange(bar) / bar
        noise = rng.uniform(-1, 1, bar)
        rev = _sweep_lp(noise, 800, 9000, sr) * np.exp((t - 1.0) * 5.0)
        # short version: half a bar
        short = rev[bar // 2:] if bar // 2 > 0 else rev
        short = short * np.linspace(0.3, 1.0, len(short))
        # uplifter: two bars, saw rising one octave + noise, filter opening
        n = 2 * bar
        f = 110.0 * 2.0 ** (np.arange(n) / n)
        saw = 2.0 * ((np.cumsum(f) / sr) % 1.0) - 1.0
        up = _sweep_lp(saw * 0.6 + rng.uniform(-1, 1, n) * 0.4, 300, 12000, sr)
        up = up * (np.arange(n) / n) ** 1.5
        # scream: FM with rising pitch and index, saturated, cut at the bar end
        t3 = np.arange(bar) / bar
        car = 300.0 * 2.0 ** (2.2 * t3)
        idx = 1.0 + 7.0 * t3
        ph = np.cumsum(car) / sr
        mod = np.sin(2 * np.pi * ph * 1.5)
        scr = np.tanh(3.0 * np.sin(2 * np.pi * ph + idx * mod))
        scr = scr * np.minimum(1.0, t3 * 4.0) * np.exp((t3 - 1.0) * 2.0)
        # impact: sub boom + noise burst, decaying
        n4 = int(sr * 1.2)
        t4 = np.arange(n4) / sr
        boom = np.sin(2 * np.pi * (45.0 + 60.0 * np.exp(-t4 / 0.05)) * t4) * np.exp(-t4 / 0.5)
        burst = _sweep_lp(rng.uniform(-1, 1, n4), 6000, 200, sr) * np.exp(-t4 / 0.25)
        imp = np.tanh(2.0 * (boom + 0.5 * burst))
        self.samples = {
            "reverse_cymbal": _stereo(rev, 0.7), "reverse_short": _stereo(short, 0.6),
            "uplifter": _stereo(up, 0.55), "scream": _stereo(scr, 0.6),
            "impact": _stereo(imp, 0.9),
        }

    def render(self, n: int, events: list[NoteEvent]) -> np.ndarray:
        out = np.zeros((n, 2), dtype=np.float32)
        for ev in events:
            name = NOTE_TO_RISER.get(ev.note)
            if ev.on and name:
                self.active.append((self.samples[name], -int(ev.offset), float(ev.velocity)))
        still = []
        for sample, pos, gain in self.active:
            start, src = max(0, -pos), max(0, pos)
            k = min(n - start, len(sample) - src)
            if k > 0:
                out[start:start + k] += sample[src:src + k] * gain
            pos += n
            if pos < len(sample):
                still.append((sample, pos, gain))
        self.active = still
        return out * self.volume
