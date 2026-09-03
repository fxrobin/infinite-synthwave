from __future__ import annotations

import numpy as np

from ..patches.model import PatchModel
from .effects import build_effects
from .envelope import RELEASE
from .events import NoteEvent
from .voice import Voice


def _oscillators_changed(new: PatchModel, old: PatchModel) -> bool:
    """Oscillators changed."""
    if len(new.oscillators) != len(old.oscillators):
        return True
    return any(
        a.wave != b.wave or a.unison != b.unison or a.octave != b.octave or a.semi != b.semi
        for a, b in zip(new.oscillators, old.oscillators, strict=True)
    )


def _filter_changed(new: PatchModel, old: PatchModel) -> bool:
    """Filter changed."""
    if (new.filter is None) != (old.filter is None):
        return True
    if new.filter and old.filter:
        if new.filter.type != old.filter.type:
            return True
        return (new.filter.env is None) != (old.filter.env is None)
    return False


def _has_structural_change(new: PatchModel, old: PatchModel) -> bool:
    """Has structural change."""
    if new.polyphony != old.polyphony:
        return True
    if _oscillators_changed(new, old):
        return True
    if _filter_changed(new, old):
        return True
    if (new.lfo is None) != (old.lfo is None):
        return True
    return [e.type for e in new.effects] != [e.type for e in old.effects]


def _effects_differ(new: PatchModel, old: PatchModel) -> bool:
    """Effects differ."""
    return [e.model_dump() for e in new.effects] != [e.model_dump() for e in old.effects]


class Synth:
    """Synth."""

    def __init__(self, patch: PatchModel, sr: int, rng: np.random.Generator, bpm: float):
        """Initialize."""
        self.sr, self.rng, self.bpm = sr, rng, bpm
        self.counter = 0
        self.set_patch(patch)

    def set_patch(self, patch: PatchModel) -> None:  # noqa: C901 - trivial but flagged by lizard
        """Set patch."""
        self.patch = patch
        self.voices = [Voice(patch, self.sr, self.rng) for _ in range(patch.polyphony)]
        self.effects = build_effects([e.model_dump() for e in patch.effects], self.sr, self.bpm)

    def update_patch(self, patch: PatchModel) -> None:
        """
        Apply parameter changes without resetting voices (no click, held notes survive).

        Falls back to a full reset when the structure changed (oscillator count, waves,
        unison, polyphony, filter type or effect chain layout).
        """
        old = self.patch
        if _has_structural_change(patch, old):
            self.set_patch(patch)
            return
        self.patch = patch
        for v in self.voices:
            v.retune(patch)
        if _effects_differ(patch, old):
            self.effects = build_effects([e.model_dump() for e in patch.effects], self.sr, self.bpm)

    def set_bpm(self, bpm: float) -> None:
        """Set bpm."""
        self.bpm = bpm
        self.effects = build_effects([e.model_dump() for e in self.patch.effects], self.sr, bpm)

    def note_on(self, note: int, velocity: float) -> None:
        """Note on."""
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
        """Note off."""
        for v in self.voices:
            if v.active and v.note == note and v.amp_env.stage != RELEASE:
                v.note_off()

    def _render_voices(self, n: int) -> np.ndarray:
        """Render voices."""
        out = np.zeros((n, 2), dtype=np.float32)
        for v in self.voices:
            if v.active:
                out += v.render(n)
        return out

    def render(self, n: int, events: list[NoteEvent]) -> np.ndarray:  # noqa: C901 - event slicing branches
        """Render."""
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
