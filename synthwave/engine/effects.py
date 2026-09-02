"""Block-vectorised effects. Delay lines are processed in chunks no longer than their delay."""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .lfo import LFO


def note_to_seconds(value: float | str, bpm: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = value.strip()
    mult = 1.0
    if s.endswith("d"):
        mult, s = 1.5, s[:-1]
    elif s.endswith("t"):
        mult, s = 2.0 / 3.0, s[:-1]
    num, den = s.split("/")
    return 4.0 * (60.0 / bpm) * int(num) / int(den) * mult


class Effect:
    def process(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class Chorus(Effect):
    def __init__(self, sr: int, bpm: float, rate: float = 0.5, depth: float = 0.003,
                 mix: float = 0.4):
        self.size = int(sr * 0.2)
        self.buf = np.zeros((self.size, 2))
        self.pos = 0
        self.lfo = [LFO("sine", rate, sr, 0.0), LFO("sine", rate, sr, 0.25)]
        self.depth, self.base, self.mix = depth * sr, 0.02 * sr, mix

    def process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        idx = (self.pos + np.arange(n)) % self.size
        self.buf[idx] = x
        out = np.empty((n, 2))
        for ch in (0, 1):
            d = self.base + self.depth * (1.0 + self.lfo[ch].render(n))
            p = self.pos + np.arange(n) - d
            i0 = np.floor(p).astype(int)
            fr = p - i0
            wet = self.buf[i0 % self.size, ch] * (1 - fr) + self.buf[(i0 + 1) % self.size, ch] * fr
            out[:, ch] = x[:, ch] * (1 - self.mix) + wet * self.mix
        self.pos = (self.pos + n) % self.size
        return out.astype(np.float32)


class Delay(Effect):
    def __init__(self, sr: int, bpm: float, time: float | str = "1/8", feedback: float = 0.4,
                 mix: float = 0.3, pingpong: bool = True):
        self.d = max(1, int(note_to_seconds(time, bpm) * sr))
        self.size = self.d * 2 + 8192
        self.buf = np.zeros((self.size, 2))
        self.pos = 0
        self.feedback = float(np.clip(feedback, 0, 0.95))
        self.mix, self.pingpong = mix, pingpong

    def process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        out = np.empty((n, 2))
        start = 0
        while start < n:
            k = min(self.d, n - start)
            xs = x[start:start + k]
            ridx = (self.pos + np.arange(k) - self.d) % self.size
            y = self.buf[ridx]
            widx = (self.pos + np.arange(k)) % self.size
            if self.pingpong:
                mono = xs.mean(axis=1)
                self.buf[widx, 0] = mono + y[:, 1] * self.feedback
                self.buf[widx, 1] = y[:, 0] * self.feedback
            else:
                self.buf[widx] = xs + y * self.feedback
            out[start:start + k] = xs * (1 - self.mix) + y * self.mix
            self.pos = (self.pos + k) % self.size
            start += k
        return out.astype(np.float32)


_COMBS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
_ALLPASSES = (556, 441, 341, 225)
_SPREAD = 23


class _Line:
    def __init__(self, length: int):
        self.len, self.buf, self.pos, self.zi = length, np.zeros(length), 0, np.zeros(1)


class Reverb(Effect):
    """Freeverb-style Schroeder reverb: 8 lowpass-feedback combs + 4 allpasses per channel."""

    def __init__(self, sr: int, bpm: float, size: float = 0.8, damping: float = 0.5,
                 mix: float = 0.3, predelay: float = 0.02):
        scale = sr / 44100.0
        self.fb = 0.7 + 0.28 * float(np.clip(size, 0, 1))
        self.damp = float(np.clip(damping, 0, 0.95))
        self.mix = mix
        self.combs = [[_Line(int(c * scale) + ch * _SPREAD) for c in _COMBS] for ch in (0, 1)]
        self.allp = [[_Line(int(a * scale) + ch * _SPREAD) for a in _ALLPASSES] for ch in (0, 1)]
        self.pre = max(0, int(predelay * sr))
        self.pre_line = _Line(self.pre + 8192) if self.pre else None

    def _comb(self, c: _Line, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        start = 0
        while start < len(x):
            k = min(c.len, len(x) - start)
            idx = (c.pos + np.arange(k)) % c.len
            y = c.buf[idx]
            lp, c.zi = lfilter([1 - self.damp], [1, -self.damp], y, zi=c.zi)
            c.buf[idx] = x[start:start + k] + lp * self.fb
            out[start:start + k] = y
            c.pos = (c.pos + k) % c.len
            start += k
        return out

    @staticmethod
    def _allpass(a: _Line, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        start = 0
        while start < len(x):
            k = min(a.len, len(x) - start)
            idx = (a.pos + np.arange(k)) % a.len
            z = a.buf[idx]
            xs = x[start:start + k]
            a.buf[idx] = xs + z * 0.5
            out[start:start + k] = z - xs
            a.pos = (a.pos + k) % a.len
            start += k
        return out

    def _predelay(self, x: np.ndarray) -> np.ndarray:
        if not self.pre_line:
            return x
        p = self.pre_line
        n = len(x)
        out = np.empty_like(x)
        start = 0
        while start < n:
            k = min(self.pre, n - start)
            widx = (p.pos + np.arange(k)) % p.len
            ridx = (p.pos + np.arange(k) - self.pre) % p.len
            out[start:start + k] = p.buf[ridx]
            p.buf[widx] = x[start:start + k]
            p.pos = (p.pos + k) % p.len
            start += k
        return out

    def process(self, x: np.ndarray) -> np.ndarray:
        mono = self._predelay(x.astype(np.float64).mean(axis=1)) * 0.03
        out = np.empty((len(x), 2))
        for ch in (0, 1):
            wet = sum(self._comb(c, mono) for c in self.combs[ch])
            for a in self.allp[ch]:
                wet = self._allpass(a, wet)
            out[:, ch] = x[:, ch] * (1 - self.mix) + wet * self.mix
        return out.astype(np.float32)


class GatedReverb(Effect):
    def __init__(self, sr: int, bpm: float, size: float = 0.85, hold: float = 0.25,
                 threshold: float = 0.1, mix: float = 0.5):
        self.rev = Reverb(sr, bpm, size=size, damping=0.3, mix=1.0, predelay=0.0)
        self.hold, self.thr, self.mix = int(hold * sr), threshold, mix
        self.open_left = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        wet = self.rev.process(x)
        gate = np.zeros(n)
        if self.open_left > 0:
            gate[:min(n, self.open_left)] = 1.0
        hits = np.flatnonzero(np.abs(x).max(axis=1) > self.thr)
        end = -1
        for h in hits:
            if h < end:
                continue
            end = h + self.hold
            gate[h:min(n, end)] = 1.0
        self.open_left = max(self.open_left - n, end - n, 0)
        return (x * (1 - self.mix) + wet * gate[:, None] * self.mix).astype(np.float32)


class Limiter(Effect):
    def __init__(self, sr: int, bpm: float, threshold: float = 0.9, release: float = 0.1):
        self.thr, self.coef, self.gain = threshold, float(np.exp(-1.0 / (release * sr))), 1.0

    def process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        peak = float(np.abs(x).max()) if n else 0.0
        target = min(1.0, self.thr / peak) if peak > 0 else 1.0
        if target < self.gain:  # instant attack: the whole block gets the safe gain
            self.gain = target
            ramp = np.full(n, target)
        else:
            new = target + (self.gain - target) * self.coef ** n
            ramp = np.linspace(self.gain, new, n, endpoint=False)
            self.gain = new
        return np.clip(x * ramp[:, None], -1.0, 1.0).astype(np.float32)


class Sidechain:
    """Ducking envelope triggered at sample offsets; gain = 1 - depth * exp(-t/release)."""

    def __init__(self, sr: int, depth: float = 0.5, release: float = 0.25):
        self.depth, self.rel, self.t = depth, max(1.0, release * sr), 10 ** 9

    def gain(self, n: int, triggers: list[int]) -> np.ndarray:
        g = np.empty(n)
        pos = 0
        for tr in sorted(triggers) + [n]:
            k = tr - pos
            if k > 0:
                ts = self.t + np.arange(1, k + 1)
                g[pos:tr] = 1.0 - self.depth * np.exp(-ts / self.rel)
                self.t += k
                pos = tr
            if tr < n:
                self.t = 0
        return g


_REGISTRY = {"chorus": Chorus, "delay": Delay, "reverb": Reverb,
             "gated_reverb": GatedReverb, "limiter": Limiter}


def build_effects(specs: list[dict], sr: int, bpm: float) -> list[Effect]:
    out = []
    for spec in specs:
        kw = dict(spec)
        kind = kw.pop("type")
        out.append(_REGISTRY[kind](sr, bpm, **kw))
    return out
