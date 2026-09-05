"""DX7 6-op FM engine — 32 algorithms, feedback, vectorised numpy render."""

from __future__ import annotations

import math

import numpy as np

from ..patches.model import Dx7PatchModel
from .blocks import arange1
from .envelope import ADSR
from .voice import midi_to_hz

# Dexed algorithm table — 6 ops per algo, flags: IN_BUS_ONE=16, IN_BUS_TWO=32,
# OUT_BUS_ONE=1, OUT_BUS_TWO=2, OUT_BUS_ADD=4, FB_IN=64, FB_OUT=128
ALGORITHMS = [
    [0xC1, 0x11, 0x11, 0x14, 0x01, 0x14],  # 1
    [0x01, 0x11, 0x11, 0x14, 0xC1, 0x14],  # 2
    [0xC1, 0x11, 0x14, 0x01, 0x11, 0x14],  # 3
    [0xC1, 0x11, 0x94, 0x01, 0x11, 0x14],  # 4
    [0xC1, 0x14, 0x01, 0x14, 0x01, 0x14],  # 5
    [0xC1, 0x94, 0x01, 0x14, 0x01, 0x14],  # 6
    [0xC1, 0x11, 0x05, 0x14, 0x01, 0x14],  # 7
    [0x01, 0x11, 0xC5, 0x14, 0x01, 0x14],  # 8
    [0x01, 0x11, 0x05, 0x14, 0xC1, 0x14],  # 9
    [0x01, 0x05, 0x14, 0xC1, 0x11, 0x14],  # 10
    [0xC1, 0x05, 0x14, 0x01, 0x11, 0x14],  # 11
    [0x01, 0x05, 0x05, 0x14, 0xC1, 0x14],  # 12
    [0xC1, 0x05, 0x05, 0x14, 0x01, 0x14],  # 13
    [0xC1, 0x05, 0x11, 0x14, 0x01, 0x14],  # 14
    [0x01, 0x05, 0x11, 0x14, 0xC1, 0x14],  # 15
    [0xC1, 0x11, 0x02, 0x25, 0x05, 0x14],  # 16
    [0x01, 0x11, 0x02, 0x25, 0xC5, 0x14],  # 17
    [0x01, 0x11, 0x11, 0xC5, 0x05, 0x14],  # 18
    [0xC1, 0x14, 0x14, 0x01, 0x11, 0x14],  # 19
    [0x01, 0x05, 0x14, 0xC1, 0x14, 0x14],  # 20
    [0x01, 0x14, 0x14, 0xC1, 0x14, 0x14],  # 21
    [0xC1, 0x14, 0x14, 0x14, 0x01, 0x14],  # 22
    [0xC1, 0x14, 0x14, 0x01, 0x14, 0x04],  # 23
    [0xC1, 0x14, 0x14, 0x14, 0x04, 0x04],  # 24
    [0xC1, 0x14, 0x14, 0x04, 0x04, 0x04],  # 25
    [0xC1, 0x05, 0x14, 0x01, 0x14, 0x04],  # 26
    [0x01, 0x05, 0x14, 0xC1, 0x14, 0x04],  # 27
    [0x04, 0xC1, 0x11, 0x14, 0x01, 0x14],  # 28
    [0xC1, 0x14, 0x01, 0x14, 0x04, 0x04],  # 29
    [0x04, 0xC1, 0x11, 0x14, 0x04, 0x04],  # 30
    [0xC1, 0x14, 0x04, 0x04, 0x04, 0x04],  # 31
    [0xC4, 0x04, 0x04, 0x04, 0x04, 0x04],  # 32
]

# flag helpers
OUT_ONE = 1
OUT_TWO = 2
OUT_ADD = 4
IN_ONE = 16
IN_TWO = 32
FB_IN = 64
FB_OUT = 128


def is_carrier(algorithm: int, op: int) -> bool:
    """True if op is a carrier in algorithm (0-based)."""
    return bool(ALGORITHMS[algorithm][op] & OUT_ADD)


def dx7_sysex_to_patches(data: bytes) -> list:
    """Parse Yamaha DX7 bulk dump (4104 bytes packed or 163 bytes single) to Dx7PatchModels.

    Supports both 32-voice bulk (F0 43 00 09 20 00 ... F7) and 1-voice (F0 43 00 00 01 1B ...).
    Returns list of Dx7PatchModel. For packed bulk, unpacks 128-byte voices to 155 params.
    """
    # Lazy import to avoid circular
    from ..patches.model import Dx7OpSpec, Dx7PatchModel

    # strip sysex framing if present
    raw = data
    if raw[:1] == b"\xf0":
        # find bulk payload between header 6 and checksum
        # header 6 bytes: F0 43 00 09 20 00 for bulk, or F0 43 00 00 01 1B for single
        # payload 4096 for bulk, 155 for single
        if len(raw) >= 6 and raw[3] == 0x09:
            payload = raw[6:-2]  # 4096
            n_voices = len(payload) // 128
            patches = []
            for v in range(n_voices):
                chunk = payload[v * 128 : (v + 1) * 128]
                if len(chunk) < 128:
                    continue
                patch = _unpack_packed_voice(chunk, Dx7OpSpec, Dx7PatchModel, v)
                patches.append(patch)
            return patches
        elif len(raw) >= 6 and raw[3] == 0x00:
            payload = raw[6:-2]  # 155
            return [_unpack_single_voice(payload, Dx7OpSpec, Dx7PatchModel)]
    # raw unpacked: 128 single packed, 155 single unpacked, 4096 bulk packed (no sysex header)
    if len(raw) == 128:
        return [_unpack_packed_voice(raw, Dx7OpSpec, Dx7PatchModel, 0)]
    if len(raw) == 155:
        return [_unpack_single_voice(raw, Dx7OpSpec, Dx7PatchModel)]
    if len(raw) == 4096:
        # 32 voices packed, no sysex header — common in Dexed_cart
        patches = []
        for v in range(32):
            chunk = raw[v * 128 : (v + 1) * 128]
            patches.append(_unpack_packed_voice(chunk, Dx7OpSpec, Dx7PatchModel, v))
        return patches
    if len(raw) == 16384:
        # 4 banks concatenated
        patches = []
        for v in range(len(raw) // 128):
            chunk = raw[v * 128 : (v + 1) * 128]
            patches.append(_unpack_packed_voice(chunk, Dx7OpSpec, Dx7PatchModel, v))
        return patches
    raise ValueError(f"unsupported DX7 sysex length {len(raw)}")


def _unpack_packed_voice(chunk: bytes, OpSpec, PatchModel, idx: int):
    """Unpack 128-byte packed bulk voice to Dx7Patch."""
    # build 155-byte single-voice param array from packed chunk per spec F
    # We only map fields needed for our engine; scaling/AMS etc are ignored for now
    ops = []
    for op in range(6):
        base = op * 17
        # bytes per packed voice: 0-101 for 6 ops
        r1, r2, r3, r4 = chunk[base + 0], chunk[base + 1], chunk[base + 2], chunk[base + 3]
        l1, l2, l3, l4 = chunk[base + 4], chunk[base + 5], chunk[base + 6], chunk[base + 7]
        # bp, ld, rd, lc/rc, det/rs, kvs/ams, ol, fc/m, ff
        ol = chunk[base + 14]
        fc_m = chunk[base + 15]
        coarse = fc_m >> 1  # 0..31
        mode = fc_m & 1
        fine = chunk[base + 16]
        detune = (chunk[base + 12] >> 3) & 0x0F  # 0..14
        # map to our simplified ratio/detune/level + keep DX7 EG
        # coarse+fine -> ratio approx (see dx7note.cc coarsemul)
        ratio = (
            0.5 if coarse == 0 else float(coarse if coarse < 16 else coarse - 14)
        )  # rough but preserves FM character
        if mode == 0 and fine:
            ratio *= 1.0 + fine * 0.01 * 0.5
        ops.append(
            OpSpec(
                ratio=max(0.25, min(32.0, ratio)),
                detune=int(detune - 7),
                level=float(ol) / 99.0,
                eg_type="dx7",
                eg_rate1=int(r1),
                eg_level1=int(l1),
                eg_rate2=int(r2),
                eg_level2=int(l2),
                eg_rate3=int(r3),
                eg_level3=int(l3),
                eg_rate4=int(r4),
                eg_level4=int(l4),
            )
        )
    # reverse ops: chunk order OP6..OP1 — reverse to OP1..OP6
    # (OP1 is carrier in most algs)
    ops = list(reversed(ops))
    alg = (chunk[110] & 0x1F) + 1
    fb = (chunk[111] >> 1) & 0x07
    raw_name = bytes(chunk[118:128]).decode("ascii", errors="ignore")
    name = raw_name.replace("\x00", "").strip() or f"DX7_{idx:02d}"
    return PatchModel(name=name, algorithm=alg, feedback=fb, operators=ops)


def _unpack_single_voice(payload: bytes, OpSpec, PatchModel):
    """Unpack 155-byte single voice dump (param # order D)."""
    ops = []
    for op in range(6):
        off = op * 21
        r1, r2, r3, r4 = payload[off + 0], payload[off + 1], payload[off + 2], payload[off + 3]
        l1, l2, l3, l4 = payload[off + 4], payload[off + 5], payload[off + 6], payload[off + 7]
        ol = payload[off + 16]
        _mode = payload[off + 17]
        coarse = payload[off + 18]
        _fine = payload[off + 19]
        detune = payload[off + 20]
        ops.append(
            OpSpec(
                ratio=float(coarse if coarse else 0.5),
                detune=int(detune - 7),
                level=float(ol) / 99.0,
                eg_type="dx7",
                eg_rate1=int(r1),
                eg_level1=int(l1),
                eg_rate2=int(r2),
                eg_level2=int(l2),
                eg_rate3=int(r3),
                eg_level3=int(l3),
                eg_rate4=int(r4),
                eg_level4=int(l4),
            )
        )
    ops = list(reversed(ops))
    alg = int(payload[134]) + 1
    fb = int(payload[135])
    raw_name = bytes(payload[145:155]).decode("ascii", errors="ignore")
    name = raw_name.replace("\x00", "").strip() or "DX7_voice"
    return PatchModel(name=name, algorithm=alg, feedback=fb, operators=ops)


def _dx7_eg_to_adsr(op) -> tuple[float, float, float, float]:
    """Map DX7 8-param R/L to ADSR approximation."""
    if op.eg_type != "dx7":
        return op.attack, op.decay, op.sustain, op.release

    # very rough: R1->attack, R2->decay, L3->sustain, R4->release
    # rates 0..99 -> time 3s .. 0.002s ; clamp 127 -> 99
    def rate_to_time(r: int) -> float:
        rc = int(min(99, max(0, r)))
        return float(0.002 + (99 - rc) * 0.03)  # 0.002..2.97s

    a = rate_to_time(op.eg_rate1)
    d = rate_to_time(op.eg_rate2)
    s = float(min(99, max(0, op.eg_level3))) / 99.0
    r = rate_to_time(op.eg_rate4)
    return a, d, s, r


class Dx7Operator:
    """Single DX7 operator state."""

    def __init__(self, spec, sr: int, rng: np.random.Generator | None = None):
        self.spec = spec
        self.sr = sr
        # phase de départ tirée du RNG de la session : le RNG global de numpy cassait
        # la reproductibilité « même seed => rendu bit-identique » dès qu'un patch dx7
        # était joué.
        self.phase = float((rng or np.random.default_rng()).uniform(0, 1))
        a, d, s, r = _dx7_eg_to_adsr(spec)
        self.env = ADSR(a, d, s, r, sr)
        # freq ratio + detune
        self.base_ratio = float(spec.ratio)
        self.detune_cents = float(spec.detune) * 3.0  # approx DX7 detune table
        self.level = float(spec.level)

    def retune(self, spec) -> None:
        self.spec = spec
        a, d, s, r = _dx7_eg_to_adsr(spec)
        self.env.a = max(1, int(a * self.sr))
        self.env.d = max(1, int(d * self.sr))
        self.env.s = float(np.clip(s, 0, 1))
        self.env.r = max(1, int(r * self.sr))
        self.base_ratio = float(spec.ratio)
        self.detune_cents = float(spec.detune) * 3.0
        self.level = float(spec.level)


class Dx7Voice:
    """One DX7 voice — 6 ops, 32 algs, feedback, mono legato."""

    def __init__(self, patch: Dx7PatchModel, sr: int, rng: np.random.Generator):
        self.patch = patch
        self.sr = sr
        self.rng = rng
        self.ops = [Dx7Operator(o, sr, rng) for o in patch.operators]
        self.note = 60
        self.velocity = 0.0
        self.freq = 261.6
        self.target_freq = 261.6
        self.glide_coef = float(np.exp(-1.0 / (patch.glide * sr))) if patch.glide > 0 else 0.0
        self.age = 0
        # feedback state per voice (single value, as DX7 only one fb loop per algo)
        self.fb_buf = 0.0
        self.fb_shift = patch.feedback  # 0..7
        # precompute algorithm flags for speed
        self.alg = patch.algorithm - 1  # 0-based

    @property
    def active(self) -> bool:
        return any(
            not op.env.finished for idx, op in enumerate(self.ops) if is_carrier(self.alg, idx)
        )

    def retune(self, patch: Dx7PatchModel) -> None:
        self.patch = patch
        self.alg = patch.algorithm - 1
        self.fb_shift = patch.feedback
        self.glide_coef = float(np.exp(-1.0 / (patch.glide * self.sr))) if patch.glide > 0 else 0.0
        for op, spec in zip(self.ops, patch.operators, strict=True):
            op.retune(spec)

    def note_on(self, note: int, velocity: float, legato: bool = False) -> None:
        self.note = note
        self.velocity = float(velocity)
        self.target_freq = midi_to_hz(note)
        if not (legato and self.glide_coef):
            self.freq = self.target_freq
        if not legato or not self.active:
            for op in self.ops:
                op.env.gate_on()

    def note_off(self) -> None:
        for op in self.ops:
            op.env.gate_off()

    def render(self, n: int) -> np.ndarray:
        if self.glide_coef:
            self.freq = self.target_freq + (self.freq - self.target_freq) * (self.glide_coef**n)
        freq = self.freq
        alg_flags = ALGORITHMS[self.alg]
        # envelope arrays per op (n,)
        envs = [op.env.render(n) * op.level * self.velocity for op in self.ops]
        # quick silence check: if all carriers env ==0 skip
        # buffers for buses 1 & 2 and output
        buf1 = np.zeros(n, dtype=np.float64)
        buf2 = np.zeros(n, dtype=np.float64)
        out = np.zeros(n, dtype=np.float64)
        # has_contents as in fm_core: [output, bus1, bus2]
        has = [True, False, False]
        # phase increments per op
        for idx in range(6):
            flags = alg_flags[idx]
            op = self.ops[idx]
            env = envs[idx]
            # skip if both gains below thresh (~ 1120 in fixed point ~ very low)
            # use float thresh 0.002
            if np.max(env) < 0.002:
                # still need to mark has_contents for non-add? replicate C logic
                outbus = flags & 3
                is_add = bool(flags & OUT_ADD)
                if not is_add:
                    # clearing bus when silent and not add
                    if outbus == 1:
                        has[1] = False
                    elif outbus == 2:
                        has[2] = False
                continue
            inbus = (flags >> 4) & 3
            outbus = flags & 3
            is_add = bool(flags & OUT_ADD)
            is_fb = (flags & 0xC0) == 0xC0
            # frequency for this op
            ratio = op.base_ratio * (2.0 ** (op.detune_cents / 1200.0))
            f = freq * ratio
            dt = f / self.sr
            # phase ramp
            phases = (op.phase + dt * arange1(n)) % 1.0
            op.phase = float(phases[-1] % 1.0)
            # input buffer
            if is_fb and self.fb_shift < 16:
                # feedback — per-sample loop (single op per algo)
                fb_scale = (1 << self.fb_shift) / 512.0 if self.fb_shift else 0.0
                # actual DX7 feedback is op output feeding back, scaled
                # boucle par échantillon obligatoire (récursion), mais sur des float
                # Python : np.sin scalaire et l'indexation ndarray coûtaient ~10x plus.
                ph_l, env_l = phases.tolist(), env.tolist()
                sin, two_pi = math.sin, 2 * np.pi
                y_l = [0.0] * n
                fb = self.fb_buf
                for i in range(n):
                    # feedback uses previous output
                    mod = fb * fb_scale
                    fb = sin(two_pi * (ph_l[i] + mod)) * env_l[i]
                    y_l[i] = fb
                self.fb_buf = float(fb)
                y = np.array(y_l, dtype=np.float64)
                # output routing
                if outbus == 0:
                    if is_add and has[0]:
                        out += y
                    else:
                        out = y.copy() if not has[0] else out + y
                        has[0] = True
                elif outbus == 1:
                    if is_add and has[1]:
                        buf1 += y
                    else:
                        buf1 = y.copy()
                        has[1] = True
                else:
                    if is_add and has[2]:
                        buf2 += y
                    else:
                        buf2 = y.copy()
                        has[2] = True
                continue
            if inbus == 0 or not has[inbus]:
                # pure (no modulation)
                y = np.sin(2 * np.pi * phases) * env
            else:
                inbuf = buf1 if inbus == 1 else buf2
                # inbuf already contains summed modulator outputs (range about -1..1)
                # DX7 mod index is directly inbuf added to phase
                y = np.sin(2 * np.pi * (phases + inbuf)) * env
            # route to output bus
            if outbus == 0:
                if is_add and has[0]:
                    out += y
                else:
                    # first carrier to output — has[0] already True, but we add
                    if has[0] and not is_add:
                        # shouldn't happen for carriers (they are all ADD)
                        out = y.copy()
                    else:
                        out += y
                    has[0] = True
            elif outbus == 1:
                if is_add and has[1]:
                    buf1 += y
                else:
                    buf1 = y.copy() if not has[1] else buf1 + y
                    # actually pure case without has should set
                    if not has[1]:
                        buf1 = y.copy()
                    has[1] = True
                # for non-add we already set
                if not is_add:
                    # has already set above
                    pass
            else:  # outbus 2
                if is_add and has[2]:
                    buf2 += y
                else:
                    buf2 = y.copy()
                    has[2] = True
        # stereo: duplicate mono to both channels, soft clip
        stereo = np.stack([out, out], axis=1).astype(np.float32)
        # apply volume and soft tanh like DX7 DAC
        stereo = np.tanh(stereo * 0.9)
        return stereo


class Dx7Synth:
    """Polyphonic DX7 — mirrors Synth API (note_on/off, render, effects)."""

    def __init__(self, patch: Dx7PatchModel, sr: int, rng: np.random.Generator, bpm: float):
        self.patch = patch
        self.sr, self.rng, self.bpm = sr, rng, bpm
        self.counter = 0
        from .effects import build_effects

        self.voices = [Dx7Voice(patch, sr, rng) for _ in range(patch.polyphony)]
        self.effects = build_effects([e.model_dump() for e in patch.effects], sr, bpm)

    def set_patch(self, patch: Dx7PatchModel) -> None:
        self.patch = patch
        from .effects import build_effects

        self.voices = [Dx7Voice(patch, self.sr, self.rng) for _ in range(patch.polyphony)]
        self.effects = build_effects([e.model_dump() for e in patch.effects], self.sr, self.bpm)

    def update_patch(self, patch: Dx7PatchModel) -> None:
        # DX7 structure change = algorithm/feedback/operator count
        old = self.patch
        if old.algorithm != patch.algorithm or old.feedback != patch.feedback:
            self.set_patch(patch)
            return
        self.patch = patch
        for v in self.voices:
            v.retune(patch)
        # effects may have changed
        from .effects import build_effects

        if [e.model_dump() for e in patch.effects] != [e.model_dump() for e in old.effects]:
            self.effects = build_effects([e.model_dump() for e in patch.effects], self.sr, self.bpm)

    def set_bpm(self, bpm: float) -> None:
        self.bpm = bpm
        from .effects import build_effects

        self.effects = build_effects([e.model_dump() for e in self.patch.effects], self.sr, bpm)

    def note_on(self, note: int, velocity: float) -> None:
        self.counter += 1
        if len(self.voices) == 1:
            v = self.voices[0]
            v.note_on(note, velocity, legato=v.active)
            v.age = self.counter
            return
        free = [v for v in self.voices if not v.active]
        v = free[0] if free else min(self.voices, key=lambda v: v.age)
        v.note_on(note, velocity)
        v.age = self.counter

    def note_off(self, note: int) -> None:
        for v in self.voices:
            if v.active and v.note == note:
                v.note_off()

    def _render_voices(self, n: int) -> np.ndarray:
        out = np.zeros((n, 2), dtype=np.float32)
        for v in self.voices:
            if v.active:
                out += v.render(n)
        return out

    def render(self, n: int, events, gain=None) -> np.ndarray:
        out = np.zeros((n, 2), dtype=np.float32)
        pos = 0
        for ev in sorted(events):
            off = min(max(ev.offset, pos), n)
            if off > pos:
                out[pos:off] = self._render_voices(off - pos)
                pos = off
            if ev.on:
                self.note_on(ev.note, ev.velocity)
            else:
                self.note_off(ev.note)
        if pos < n:
            out[pos:] = self._render_voices(n - pos)
        out *= self.patch.volume
        if gain is not None:
            out *= gain[:, None]
        for fx in self.effects:
            out = fx.process(out)
        return out
