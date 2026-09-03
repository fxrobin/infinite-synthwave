from __future__ import annotations

import numpy as np

from ..patches.model import PatchModel
from .effects import build_effects
from .envelope import RELEASE
from .events import NoteEvent
from .voice import Voice


class Synth:
    def __init__(self, patch: PatchModel, sr: int, rng: np.random.Generator, bpm: float):
        self.sr, self.rng, self.bpm = sr, rng, bpm
        self.counter = 0
        self.set_patch(patch)

    def set_patch(self, patch: PatchModel) -> None:
        self.patch = patch
        self.voices = [Voice(patch, self.sr, self.rng) for _ in range(patch.polyphony)]
        self.effects = build_effects([e.model_dump() for e in patch.effects], self.sr, self.bpm)

    def update_patch(self, patch: PatchModel) -> None:
        """Apply parameter changes without resetting voices (no click, held notes survive).

        Falls back to a full reset when the structure changed (oscillator count, waves,
        unison, polyphony, filter type or effect chain layout)."""
        old = self.patch
        structural = (
            patch.polyphony != old.polyphony
            or len(patch.oscillators) != len(old.oscillators)
            or any(a.wave != b.wave or a.unison != b.unison or a.octave != b.octave
                   or a.semi != b.semi for a, b in zip(patch.oscillators, old.oscillators,
                                                        strict=True))
            or (patch.filter is None) != (old.filter is None)
            or (patch.filter and old.filter and (patch.filter.type != old.filter.type
                                                 or (patch.filter.env is None)
                                                 != (old.filter.env is None)))
            or (patch.lfo is None) != (old.lfo is None)
            or [e.type for e in patch.effects] != [e.type for e in old.effects]
        )
        if structural:
            self.set_patch(patch)
            return
        self.patch = patch
        for v in self.voices:
            v.retune(patch)
        if [e.model_dump() for e in patch.effects] != [e.model_dump() for e in old.effects]:
            self.effects = build_effects([e.model_dump() for e in patch.effects], self.sr,
                                         self.bpm)

    def set_bpm(self, bpm: float) -> None:
        self.bpm = bpm
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
            if v.active and v.note == note and v.amp_env.stage != RELEASE:
                v.note_off()

    def _render_voices(self, n: int) -> np.ndarray:
        out = np.zeros((n, 2), dtype=np.float32)
        for v in self.voices:
            if v.active:
                out += v.render(n)
        return out

    def render(self, n: int, events: list[NoteEvent]) -> np.ndarray:
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
        for fx in self.effects:
            out = fx.process(out)
        return out
