"""
Block-vectorised effects.

Delay lines are processed in chunks no longer than their delay.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .filter import Filter
from .lfo import LFO

_MAX_DELAY_SEC = 2.0
_MAX_NOTE_SEC = 8.0
_MAX_EFFECTS_PER_LAYER = 8
_MAX_SPEC_KEYS = 12
_MAX_STR_LEN = 64


def note_to_seconds(value: float | str, bpm: float) -> float:  # noqa: C901 - note/BPM parsing branches
    """Note to seconds."""
    if isinstance(value, (int, float)):
        v = float(value)
        if not 0.001 <= v <= _MAX_NOTE_SEC:
            raise ValueError(f"time value out of range: {v}")
        return v
    s = str(value).strip()
    if len(s) > _MAX_STR_LEN:
        raise ValueError("time value too long")
    mult = 1.0
    if s.endswith("d"):
        mult, s = 1.5, s[:-1]
    elif s.endswith("t"):
        mult, s = 2.0 / 3.0, s[:-1]
    if "/" not in s:
        raise ValueError(f"invalid note value {value!r}")
    a, b = s.split("/", 1)
    try:
        num, den = int(a), int(b)
    except ValueError as e:
        raise ValueError(f"invalid note value {value!r}: {e}") from e
    if not 1 <= num <= 16 or not 1 <= den <= 32:
        raise ValueError(f"note fraction out of range: {value!r}")
    if not 40 <= bpm <= 200:
        raise ValueError(f"bpm out of range: {bpm}")
    sec = 4.0 * (60.0 / bpm) * num / den * mult
    if not 0.001 <= sec <= _MAX_NOTE_SEC:
        raise ValueError(f"computed delay out of range: {sec}")
    return sec


class Effect:
    """Effect."""

    def process(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Process."""
        raise NotImplementedError


class Chorus(Effect):
    """Chorus."""

    def __init__(
        self, sr: int, bpm: float, rate: float = 0.5, depth: float = 0.003, mix: float = 0.4
    ):
        """Initialize chorus."""
        self.size = int(sr * 0.2)
        self.buf = np.zeros((self.size, 2))
        self.pos = 0
        self.lfo = [LFO("sine", rate, sr, 0.0), LFO("sine", rate, sr, 0.25)]
        self.depth, self.base, self.mix = depth * sr, 0.02 * sr, mix

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process."""
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
    """Delay."""

    def __init__(
        self,
        sr: int,
        bpm: float,
        time: float | str = "1/8",
        feedback: float = 0.4,
        mix: float = 0.3,
        pingpong: bool = True,
    ):
        """Initialize."""
        sec = note_to_seconds(time, bpm)
        # clamp final pour rester < _MAX_DELAY_SEC (défense en profondeur)
        sec = float(np.clip(sec, 0.001, _MAX_DELAY_SEC))
        self.d = max(1, min(int(sec * sr), int(_MAX_DELAY_SEC * sr)))
        self.size = self.d * 2 + 8192
        # garde supplémentaire : bloque allocation > 4s stéréo
        if self.size > int(_MAX_DELAY_SEC * sr * 2 + 8192):
            raise ValueError("delay buffer too large")
        self.buf = np.zeros((self.size, 2))
        self.pos = 0
        self.feedback = float(np.clip(feedback, 0, 0.95))
        self.mix = float(np.clip(mix, 0, 1))
        self.pingpong = bool(pingpong)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process."""
        n = len(x)
        out = np.empty((n, 2))
        start = 0
        while start < n:
            k = min(self.d, n - start)
            xs = x[start : start + k]
            ridx = (self.pos + np.arange(k) - self.d) % self.size
            y = self.buf[ridx]
            widx = (self.pos + np.arange(k)) % self.size
            if self.pingpong:
                mono = xs.mean(axis=1)
                self.buf[widx, 0] = mono + y[:, 1] * self.feedback
                self.buf[widx, 1] = y[:, 0] * self.feedback
            else:
                self.buf[widx] = xs + y * self.feedback
            out[start : start + k] = xs * (1 - self.mix) + y * self.mix
            self.pos = (self.pos + k) % self.size
            start += k
        return out.astype(np.float32)


_COMBS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
_ALLPASSES = (556, 441, 341, 225)
_SPREAD = 23


class _Line:
    """Line."""

    def __init__(self, length: int):
        """Initialize."""
        self.len, self.buf, self.pos, self.zi = length, np.zeros(length), 0, np.zeros(1)


class Reverb(Effect):
    """Freeverb-style Schroeder reverb: 8 lowpass-feedback combs + 4 allpasses per
    channel.
    """

    def __init__(
        self,
        sr: int,
        bpm: float,
        size: float = 0.8,
        damping: float = 0.5,
        mix: float = 0.3,
        predelay: float = 0.02,
    ):
        """Initialize."""
        scale = sr / 44100.0
        self.fb = 0.7 + 0.28 * float(np.clip(size, 0, 1))
        self.damp = float(np.clip(damping, 0, 0.95))
        self.mix = float(np.clip(mix, 0, 1))
        self.combs = [[_Line(int(c * scale) + ch * _SPREAD) for c in _COMBS] for ch in (0, 1)]
        self.allp = [[_Line(int(a * scale) + ch * _SPREAD) for a in _ALLPASSES] for ch in (0, 1)]
        predelay = float(np.clip(predelay, 0, _MAX_DELAY_SEC))
        self.pre = max(0, int(predelay * sr))
        self.pre_line = _Line(self.pre + 8192) if self.pre else None

    def _comb(self, c: _Line, x: np.ndarray) -> np.ndarray:
        """Comb."""
        out = np.empty_like(x)
        start = 0
        while start < len(x):
            k = min(c.len, len(x) - start)
            idx = (c.pos + np.arange(k)) % c.len
            y = c.buf[idx]
            lp, c.zi = lfilter([1 - self.damp], [1, -self.damp], y, zi=c.zi)
            c.buf[idx] = x[start : start + k] + lp * self.fb
            out[start : start + k] = y
            c.pos = (c.pos + k) % c.len
            start += k
        return out

    @staticmethod
    def _allpass(a: _Line, x: np.ndarray) -> np.ndarray:
        """Allpass."""
        out = np.empty_like(x)
        start = 0
        while start < len(x):
            k = min(a.len, len(x) - start)
            idx = (a.pos + np.arange(k)) % a.len
            z = a.buf[idx]
            xs = x[start : start + k]
            a.buf[idx] = xs + z * 0.5
            out[start : start + k] = z - xs
            a.pos = (a.pos + k) % a.len
            start += k
        return out

    def _predelay(self, x: np.ndarray) -> np.ndarray:
        """Predelay."""
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
            out[start : start + k] = p.buf[ridx]
            p.buf[widx] = x[start : start + k]
            p.pos = (p.pos + k) % p.len
            start += k
        return out

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process."""
        mono = self._predelay(x.astype(np.float64).mean(axis=1)) * 0.03
        out = np.empty((len(x), 2))
        for ch in (0, 1):
            wet = sum(self._comb(c, mono) for c in self.combs[ch])
            for a in self.allp[ch]:
                wet = self._allpass(a, wet)
            out[:, ch] = x[:, ch] * (1 - self.mix) + wet * self.mix
        return out.astype(np.float32)


class GatedReverb(Effect):
    """Gatedreverb."""

    def __init__(
        self,
        sr: int,
        bpm: float,
        size: float = 0.85,
        hold: float = 0.25,
        threshold: float = 0.1,
        mix: float = 0.5,
    ):
        """Initialize."""
        self.rev = Reverb(sr, bpm, size=size, damping=0.3, mix=1.0, predelay=0.0)
        # hold borné à 2s max pour éviter buffer/état géant
        hold = float(np.clip(hold, 0.01, _MAX_DELAY_SEC))
        self.hold, self.thr = int(hold * sr), float(np.clip(threshold, 0, 1))
        self.mix = float(np.clip(mix, 0, 1))
        self.open_left = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process."""
        n = len(x)
        wet = self.rev.process(x)
        gate = np.zeros(n)
        if self.open_left > 0:
            gate[: min(n, self.open_left)] = 1.0
        hits = np.flatnonzero(np.abs(x).max(axis=1) > self.thr)
        end = -1
        for h in hits:
            if h < end:
                continue
            end = h + self.hold
            gate[h : min(n, end)] = 1.0
        self.open_left = max(self.open_left - n, end - n, 0)
        return (x * (1 - self.mix) + wet * gate[:, None] * self.mix).astype(np.float32)


class Limiter(Effect):
    """Limiter."""

    def __init__(self, sr: int, bpm: float, threshold: float = 0.9, release: float = 0.1):
        """Initialize."""
        self.thr, self.coef, self.gain = threshold, float(np.exp(-1.0 / (release * sr))), 1.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process."""
        n = len(x)
        peak = float(np.abs(x).max()) if n else 0.0
        target = min(1.0, self.thr / peak) if peak > 0 else 1.0
        if target < self.gain:  # instant attack: the whole block gets the safe gain
            self.gain = target
            ramp = np.full(n, target)
        else:
            new = target + (self.gain - target) * self.coef**n
            ramp = np.linspace(self.gain, new, n, endpoint=False)
            self.gain = new
        return np.clip(x * ramp[:, None], -1.0, 1.0).astype(np.float32)


class Sidechain:
    """Ducking envelope triggered at sample offsets; gain = 1 - depth * exp(-t/release)."""

    def __init__(self, sr: int, depth: float = 0.5, release: float = 0.25):
        """Initialize."""
        self.depth, self.rel, self.t = depth, max(1.0, release * sr), 10**9

    def gain(self, n: int, triggers: list[int]) -> np.ndarray:
        """Gain."""
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


class Gate(Effect):
    """Tempo-synced trance gate / stutter: rate '1/16', duty cycle, depth, smoothed
    edges.
    """

    def __init__(
        self,
        sr: int,
        bpm: float,
        rate: float | str = "1/16",
        depth: float = 1.0,
        duty: float = 0.5,
        smooth: float = 0.002,
    ):
        """Initialize."""
        self.period = max(2.0, note_to_seconds(rate, bpm) * sr)
        self.depth, self.duty = float(np.clip(depth, 0, 1)), float(np.clip(duty, 0.05, 0.95))
        self.coef = float(np.exp(-1.0 / max(1.0, smooth * sr)))
        self.pos, self.prev = 0.0, 1.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process."""
        n = len(x)
        t = (self.pos + np.arange(n)) % self.period
        g = np.where(t < self.duty * self.period, 1.0, 1.0 - self.depth)
        g, _ = lfilter([1 - self.coef], [1, -self.coef], g, zi=[self.coef * self.prev])
        self.prev = float(g[-1])
        self.pos = (self.pos + n) % self.period
        return (x * g[:, None]).astype(np.float32)


class Bitcrush(Effect):
    """Bit-depth reduction + sample-and-hold downsampling."""

    def __init__(self, sr: int, bpm: float, bits: float = 8, downsample: int = 4, mix: float = 1.0):
        """Initialize bitcrush."""
        self.levels = 2.0 ** (float(np.clip(bits, 2, 16)) - 1)
        self.k = max(1, int(downsample))
        self.mix = mix
        self.phase = 0
        self.hold = np.zeros(2)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process."""
        n = len(x)
        idx = np.arange(n) + self.phase
        src = (idx // self.k) * self.k - self.phase
        y = x[np.maximum(src, 0)].astype(np.float64)
        y[src < 0] = self.hold
        self.hold = y[-1].copy()
        self.phase = (self.phase + n) % self.k
        y = np.round(y * self.levels) / self.levels
        return (x * (1 - self.mix) + y * self.mix).astype(np.float32)


class LoFi(Effect):
    """Lo-fi chain: bitcrush -> lowpass -> tape wobble -> hiss."""

    def __init__(
        self,
        sr: int,
        bpm: float,
        bits: float = 10,
        downsample: int = 3,
        cutoff: float = 5000,
        wobble: float = 0.002,
        noise: float = 0.004,
        mix: float = 1.0,
    ):
        """Initialize."""
        self.crush = Bitcrush(sr, bpm, bits, downsample)
        self.lp, self.cutoff = Filter("lp", sr), cutoff
        self.wob = Chorus(sr, bpm, rate=0.4, depth=wobble, mix=1.0)
        self.noise, self.mix = noise, mix
        self.rng = np.random.default_rng(1234)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process."""
        y = self.crush.process(x)
        y = self.lp.process(y, self.cutoff, 0.1)
        y = self.wob.process(y)
        if self.noise:
            y = y + self.noise * self.rng.normal(size=y.shape).astype(np.float32)
        return (x * (1 - self.mix) + y * self.mix).astype(np.float32)


class Distortion(Effect):
    """Tanh drive with a post lowpass 'tone' and level compensation."""

    def __init__(
        self, sr: int, bpm: float, drive: float = 4.0, tone: float = 4000, mix: float = 1.0
    ):
        """Initialize."""
        self.drive = max(1.0, float(drive))
        self.lp, self.tone, self.mix = Filter("lp", sr), tone, mix
        self.comp = 1.0 / np.tanh(self.drive * 0.5)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process."""
        y = np.tanh(x * self.drive) * self.comp * 0.5
        y = self.lp.process(y.astype(np.float32), self.tone, 0.0)
        return (x * (1 - self.mix) + y * self.mix).astype(np.float32)


class AutoPan(Effect):
    """Constant-power stereo auto-pan; rate in Hz or tempo-synced ('1/2', '1/4'...)."""

    def __init__(
        self, sr: int, bpm: float, rate: float | str = "1/2", depth: float = 0.8, wave: str = "sine"
    ):
        """Initialize."""
        hz = 1.0 / note_to_seconds(rate, bpm) if isinstance(rate, str) else float(rate)
        self.lfo, self.depth = LFO(wave, hz, sr), float(np.clip(depth, 0, 1))

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process."""
        pan = self.depth * self.lfo.render(len(x))
        angle = (pan + 1.0) * np.pi / 4.0
        mono = x.mean(axis=1)
        out = np.stack(
            [mono * np.cos(angle) * np.sqrt(2), mono * np.sin(angle) * np.sqrt(2)], axis=1
        )
        return out.astype(np.float32)


class Phaser(Effect):
    """Cascade of first-order allpass stages swept by an LFO; coefficients updated per
    chunk.
    """

    def __init__(
        self,
        sr: int,
        bpm: float,
        rate: float | str = 0.3,
        depth: float = 0.8,
        stages: int = 4,
        mix: float = 0.5,
        feedback: float = 0.3,
    ):
        """Initialize."""
        hz = 1.0 / note_to_seconds(rate, bpm) if isinstance(rate, str) else float(rate)
        self.sr, self.lfo = sr, LFO("triangle", hz, sr)
        self.depth, self.mix = float(np.clip(depth, 0, 1)), mix
        self.fb = float(np.clip(feedback, 0, 0.9))
        self.stages = int(np.clip(stages, 2, 8))
        self.zi = [[np.zeros(1) for _ in range(self.stages)] for _ in (0, 1)]
        self.last = np.zeros(2)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process."""
        n = len(x)
        lfo = self.lfo.render(n)
        out = np.empty((n, 2))
        chunk = 128
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            sweep = 0.5 + 0.5 * float(lfo[start:end].mean())
            fc = 300.0 * (8.0 ** (sweep * self.depth))  # 300 Hz .. 2.4 kHz
            g = (1.0 - np.tan(np.pi * fc / self.sr)) / (1.0 + np.tan(np.pi * fc / self.sr))
            for ch in (0, 1):
                seg = x[start:end, ch].astype(np.float64)
                seg[0] += self.fb * self.last[ch]
                for s in range(self.stages):
                    seg, self.zi[ch][s] = lfilter([g, -1.0], [1.0, -g], seg, zi=self.zi[ch][s])
                self.last[ch] = seg[-1]
                out[start:end, ch] = seg
        return (x * (1 - self.mix) + out * self.mix).astype(np.float32)


class Flanger(Effect):
    """Short modulated delay with feedback, processed in chunks no longer than the
    delay.
    """

    def __init__(
        self,
        sr: int,
        bpm: float,
        rate: float | str = 0.25,
        depth: float = 0.002,
        base: float = 0.003,
        feedback: float = 0.5,
        mix: float = 0.5,
    ):
        """Initialize."""
        hz = 1.0 / note_to_seconds(rate, bpm) if isinstance(rate, str) else float(rate)
        if not 0.01 <= hz <= 20:
            raise ValueError(f"flanger rate out of range: {hz}")
        self.sr, self.lfo = sr, LFO("sine", hz, sr)
        # base/depth en secondes, bornés pour éviter OOM
        base = float(np.clip(base, 0.001, 0.02))
        depth = float(np.clip(depth, 0, 0.02))
        self.base, self.depth = max(2.0, base * sr), depth * sr
        self.fb = float(np.clip(feedback, 0, 0.9))
        self.mix = float(np.clip(mix, 0, 1))
        self.size = int(sr * 0.05)
        self.buf = np.zeros((self.size, 2))
        self.pos = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process."""
        n = len(x)
        lfo = self.lfo.render(n)
        out = np.empty((n, 2))
        chunk = max(1, int(self.base) - 1)
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            k = end - start
            d = self.base + self.depth * (1.0 + lfo[start:end])
            p = self.pos + np.arange(k) - d
            i0 = np.floor(p).astype(int)
            fr = (p - i0)[:, None]
            wet = self.buf[i0 % self.size] * (1 - fr) + self.buf[(i0 + 1) % self.size] * fr
            widx = (self.pos + np.arange(k)) % self.size
            self.buf[widx] = x[start:end] + wet * self.fb
            out[start:end] = wet
            self.pos = (self.pos + k) % self.size
        return (x * (1 - self.mix) + out * self.mix).astype(np.float32)


_REGISTRY = {
    "chorus": Chorus,
    "delay": Delay,
    "reverb": Reverb,
    "gated_reverb": GatedReverb,
    "limiter": Limiter,
    "gate": Gate,
    "bitcrush": Bitcrush,
    "lofi": LoFi,
    "distortion": Distortion,
    "autopan": AutoPan,
    "phaser": Phaser,
    "flanger": Flanger,
}


def build_effects(specs: list[dict], sr: int, bpm: float) -> list[Effect]:  # noqa: C901 - effect spec validation branches
    """Build effects."""
    if not isinstance(specs, list):
        raise ValueError("effects must be a list")
    if len(specs) > _MAX_EFFECTS_PER_LAYER:
        raise ValueError(f"too many effects: {len(specs)} > {_MAX_EFFECTS_PER_LAYER}")
    out = []
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("effect spec must be a mapping")
        if len(spec) > _MAX_SPEC_KEYS:
            raise ValueError("effect spec too large")
        kw = dict(spec)
        kind = kw.pop("type", None)
        if not isinstance(kind, str) or len(kind) > _MAX_STR_LEN:
            raise ValueError(f"invalid effect type {kind!r}")
        if kind not in _REGISTRY:
            raise KeyError(kind)
        # borne taille des clés/valeurs string
        for k, v in kw.items():
            if not isinstance(k, str) or len(k) > _MAX_STR_LEN:
                raise ValueError(f"invalid effect param {k!r}")
            if isinstance(v, str) and len(v) > _MAX_STR_LEN:
                raise ValueError(f"effect param {k!r} too long")
        out.append(_REGISTRY[kind](sr, bpm, **kw))
    return out
