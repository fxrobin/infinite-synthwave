"""Drum kit: every hit is synthesised once at init into a stereo buffer, then mixed on
demand.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from ..patches.model import DrumPatchModel
from .effects import GatedReverb, build_effects
from .events import NoteEvent
from .filter import biquad_coeffs

DRUM_NOTES = {
    "kick": 36,
    "snare": 38,
    "clap": 39,
    "snap": 40,
    "hat_closed": 42,
    "tick": 44,
    "tom_low": 45,
    "hat_open": 46,
    "tom_mid": 47,
    "crash": 49,
    "ride": 51,
    "crash_roll": 57,
    "shaker": 70,
}
NOTE_TO_DRUM = {v: k for k, v in DRUM_NOTES.items()}


def _t(sr: int, seconds: float) -> np.ndarray:
    """T."""
    return np.arange(int(sr * seconds)) / sr


def _stereo(x: np.ndarray) -> np.ndarray:
    """Stereo."""
    x = x / max(1e-9, float(np.abs(x).max()))
    return np.stack([x, x], axis=1).astype(np.float32)


def _filt(kind: str, x: np.ndarray, cutoff: float, res: float, sr: int) -> np.ndarray:
    """Filt."""
    b, a = biquad_coeffs(kind, cutoff, res, sr)
    return lfilter(b, a, x)


class DrumKit:
    """Drumkit."""

    def __init__(
        self, patch: DrumPatchModel, sr: int, rng: np.random.Generator, bpm: float = 120.0
    ):
        """Initialize drum kit."""
        self.sr, self.rng, self.bpm = sr, rng, bpm
        self.active: list[tuple[np.ndarray, int, float, bool]] = []  # sample, pos, gain, kick
        self.set_patch(patch)

    def set_bpm(self, bpm: float) -> None:
        """Set bpm."""
        self.bpm = bpm
        self.perc_fx = build_effects(
            [e.model_dump() for e in self.patch.perc_effects], self.sr, bpm
        )
        self.samples["crash_roll"] = self._crash_roll(bpm)

    def _crash_roll(self, bpm: float) -> np.ndarray:
        """One bar of mallet roll on the crash: dense strokes (~14/s) whose level swells
        towards the downbeat, then a final crash-length tail.
        """
        sr = self.sr
        bar = 4.0 * 60.0 / bpm
        t = _t(sr, bar + 0.8)
        noise = _filt("bp", self.rng.uniform(-1, 1, len(t)), 5500, 0.12, sr)
        strokes = 0.55 + 0.45 * np.abs(np.sin(np.pi * t * 14.0)) ** 0.3
        swell = np.where(t < bar, (t / bar) ** 2.2, np.exp(-(t - bar) / 0.6))
        return _stereo(noise * strokes * swell) * self.patch.crash_roll_gain

    def set_patch(self, patch: DrumPatchModel) -> None:
        """Set patch."""
        self.patch = patch
        sr, rng = self.sr, self.rng
        k, s, h, c, tm = patch.kick, patch.snare, patch.hat, patch.clap, patch.tom
        sn, rd, sh, tk = patch.snap, patch.ride, patch.shaker, patch.tick
        # kick: sine with exponential pitch drop + short click
        t = _t(sr, k.decay * 3)
        f = k.pitch_end + (k.pitch_start - k.pitch_end) * np.exp(-t / k.pitch_decay)
        body = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / k.decay)
        sub = k.sub * np.sin(2 * np.pi * k.pitch_end * t) * np.exp(-t / k.sub_decay)
        if k.beater > 0:  # felt beater: soft lowpassed thump instead of a sharp click
            nb = int(0.02 * sr)
            thump = _filt("lp", rng.uniform(-1, 1, nb), 900, 0.1, sr)
            thump *= np.exp(-_t(sr, 0.02)[:nb] / 0.006)
            body[:nb] += k.beater * 0.8 * thump
        kick = np.tanh(k.drive * (body + sub)) / np.tanh(k.drive)
        nclick = int(0.002 * sr)
        kick[:nclick] += k.click * rng.uniform(-1, 1, nclick)
        # snare: tonal body + highpassed noise, through gated reverb
        t = _t(sr, 1.2)
        tone = np.sin(2 * np.pi * s.tone * t) * np.exp(-t / s.tone_decay)
        noise = _filt("hp", rng.uniform(-1, 1, len(t)), 1800, 0.2, sr) * np.exp(-t / s.noise_decay)
        snare = GatedReverb(
            sr, 120, size=s.reverb_size, hold=s.gate_hold, threshold=0.2, mix=s.reverb_mix
        ).process(_stereo(0.6 * tone + noise))[:, 0]
        # clap: four noise bursts then a tail, through gated reverb
        t = _t(sr, 1.0)
        noise = _filt("bp", rng.uniform(-1, 1, len(t)), 1500, 0.3, sr)
        env = np.zeros_like(t)
        for i in range(4):
            start = int(i * 0.011 * sr)
            env[start:] = np.maximum(env[start:], np.exp(-t[: len(t) - start] / 0.012))
        env = np.maximum(env, 0.7 * np.exp(-t / c.decay) * (t > 0.03))
        clap = GatedReverb(
            sr, 120, size=0.85, hold=c.gate_hold, threshold=0.2, mix=c.reverb_mix
        ).process(_stereo(noise * env))[:, 0]
        # hats: highpassed noise, short / long decay
        t = _t(sr, h.open_decay * 3)
        base = _filt("hp", rng.uniform(-1, 1, len(t)), h.cutoff, 0.3, sr)
        hat_c = base * np.exp(-t / h.closed_decay)
        hat_o = base * np.exp(-t / h.open_decay)
        # toms: sine with pitch bend
        toms = {}
        for name, pitch in (("tom_low", tm.pitch_low), ("tom_mid", tm.pitch_mid)):
            t = _t(sr, tm.decay * 3)
            f = pitch * (1 + 0.6 * np.exp(-t / 0.04))
            toms[name] = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / tm.decay)
        # finger snap: two very close bandpassed bursts + a small low knuckle thump
        t = _t(sr, 0.6)
        burst = _filt("bp", rng.uniform(-1, 1, len(t)), sn.tone, 0.45, sr)
        env = np.exp(-t / sn.decay)
        second = int(0.006 * sr)
        env[second:] = np.maximum(env[second:], 0.8 * np.exp(-t[: len(t) - second] / sn.decay))
        thump = sn.body * np.sin(2 * np.pi * 320.0 * t) * np.exp(-t / 0.025)
        snap = GatedReverb(sr, 120, size=0.7, hold=0.12, threshold=0.2, mix=sn.reverb_mix).process(
            _stereo(burst * env + thump)
        )[:, 0]
        # ride: highpassed noise wash + inharmonic metallic partials, long decay
        t = _t(sr, rd.decay * 3)
        wash = _filt("hp", rng.uniform(-1, 1, len(t)), rd.cutoff, 0.2, sr) * np.exp(-t / rd.decay)
        ping = sum(
            np.sin(2 * np.pi * f * t) * np.exp(-t / (rd.decay * 0.6)) / (i + 1)
            for i, f in enumerate((3150.0, 4720.0, 6390.0, 7810.0))
        )
        ride = wash * 0.7 + rd.ping * ping * 0.3
        # tick: dry, very short highpassed click (drum-machine closed hat on the 8ths)
        t = _t(sr, 0.08)
        tick = _filt("hp", rng.uniform(-1, 1, len(t)), tk.cutoff, 0.4, sr) * np.exp(-t / tk.decay)
        # shaker: bandpassed noise with a soft attack
        t = _t(sr, sh.decay * 4)
        shaker = (
            _filt("bp", rng.uniform(-1, 1, len(t)), sh.cutoff, 0.35, sr)
            * np.minimum(1.0, t / 0.004)
            * np.exp(-t / sh.decay)
        )
        # crash: bandpassed noise, long decay
        t = _t(sr, 2.0)
        crash = _filt("bp", rng.uniform(-1, 1, len(t)), 6000, 0.1, sr) * np.exp(-t / 0.7)
        self.samples = {
            "kick": _stereo(kick) * k.gain,
            "snare": _stereo(snare) * s.gain,
            "clap": _stereo(clap) * c.gain,
            "hat_closed": _stereo(hat_c) * h.gain,
            "hat_open": _stereo(hat_o) * h.gain * 0.85,
            "tom_low": _stereo(toms["tom_low"]) * tm.gain,
            "tom_mid": _stereo(toms["tom_mid"]) * tm.gain,
            "crash": _stereo(crash) * patch.crash_gain,
            "snap": _stereo(snap) * sn.gain,
            "ride": _stereo(ride) * rd.gain,
            "shaker": _stereo(shaker) * sh.gain,
            "tick": _stereo(tick) * tk.gain,
        }
        self.set_bpm(self.bpm)

    def render(self, n: int, events: list[NoteEvent]) -> np.ndarray:  # noqa: C901 - drum routing branches
        """Render."""
        kick = np.zeros((n, 2), dtype=np.float32)
        perc = np.zeros((n, 2), dtype=np.float32)
        for ev in events:
            name = NOTE_TO_DRUM.get(ev.note)
            if ev.on and name:
                if name == "hat_closed":  # closed hat chokes the open hat
                    self.active = [a for a in self.active if a[0] is not self.samples["hat_open"]]
                self.active.append(
                    (self.samples[name], -int(ev.offset), float(ev.velocity), name == "kick")
                )
        still = []
        for sample, pos, gain, is_kick in self.active:
            start = max(0, -pos)  # first output index for this hit inside the block
            src = max(0, pos)  # first sample index
            k = min(n - start, len(sample) - src)
            if k > 0:
                (kick if is_kick else perc)[start : start + k] += sample[src : src + k] * gain
            pos += n
            if pos < len(sample):
                still.append((sample, pos, gain, is_kick))
        self.active = still
        for fx in self.perc_fx:
            perc = fx.process(perc)
        return (kick + perc) * self.patch.volume
