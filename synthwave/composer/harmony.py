"""Key, chords with sevenths, and a Markov chain over synthwave progressions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .moods import Mood

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
PROGRESSIONS: dict[str, tuple[int, ...]] = {
    "i-VI-III-VII": (0, 5, 2, 6), "i-VII-VI-VII": (0, 6, 5, 6), "i-iv-VI-V": (0, 3, 5, 4),
    "i-VI-VII-i": (0, 5, 6, 0), "VI-VII-i-i": (5, 6, 0, 0), "i-III-VII-VI": (0, 2, 6, 5),
    "iv-VI-i-VII": (3, 5, 0, 6),
}
_QUALITY = {(0, 3, 7, 10): "m7", (0, 4, 7, 11): "maj7", (0, 4, 7, 10): "7",
            (0, 3, 6, 10): "m7b5", (0, 4, 8, 11): "maj7#5"}


@dataclass(frozen=True)
class Chord:
    root_pc: int
    degree: int
    intervals: tuple[int, ...]

    def notes(self, octave: int = 4) -> list[int]:
        root = 12 * octave + self.root_pc
        return [root + i for i in self.intervals]

    def bass_note(self) -> int:
        return 12 * 3 + self.root_pc  # 36..47

    @property
    def name(self) -> str:
        return NOTE_NAMES[self.root_pc % 12] + _QUALITY.get(self.intervals, "")


class Harmony:
    MINOR = (0, 2, 3, 5, 7, 8, 10)
    MAJOR = (0, 2, 4, 5, 7, 9, 11)

    def __init__(self, rng: np.random.Generator, mood: Mood):
        self.rng, self.mood = rng, mood
        self.tonic = int(rng.integers(0, 12))
        self.mode = self.MAJOR if rng.random() < mood.major_prob else self.MINOR
        self.current: str | None = None

    def set_mood(self, mood: Mood) -> None:
        self.mood = mood

    @property
    def key_name(self) -> str:
        return NOTE_NAMES[self.tonic] + (" minor" if self.mode == self.MINOR else " major")

    def chord_for_degree(self, deg: int) -> Chord:
        s = self.mode
        root = s[deg % 7]
        intervals = tuple((s[(deg + k) % 7] - root) % 12 for k in (0, 2, 4, 6))
        return Chord((self.tonic + root) % 12, deg % 7, intervals)

    def next_progression(self) -> list[Chord]:
        names = [n for n in PROGRESSIONS if self.mood.progressions.get(n, 0) > 0]
        w = np.array([self.mood.progressions[n] * (0.25 if n == self.current else 1.0)
                      for n in names], dtype=float)
        self.current = names[int(self.rng.choice(len(names), p=w / w.sum()))]
        return [self.chord_for_degree(d) for d in PROGRESSIONS[self.current]]

    def modulate(self) -> None:
        self.tonic = (self.tonic + int(self.rng.choice([5, 7, -3, 3]))) % 12

    def scale_notes(self, low: int, high: int) -> list[int]:
        pcs = {(self.tonic + i) % 12 for i in self.mode}
        return [n for n in range(low, high + 1) if n % 12 in pcs]
