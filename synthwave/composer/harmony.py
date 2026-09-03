"""Key, chords with sevenths, and a Markov chain over synthwave progressions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .moods import Mood

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
PROGRESSIONS: dict[str, tuple[int, ...]] = {
    "i-VI-III-VII": (0, 5, 2, 6),
    "i-VII-VI-VII": (0, 6, 5, 6),
    "i-iv-VI-V": (0, 3, 5, 4),
    "i-VI-VII-i": (0, 5, 6, 0),
    "VI-VII-i-i": (5, 6, 0, 0),
    "i-III-VII-VI": (0, 2, 6, 5),
    "iv-VI-i-VII": (3, 5, 0, 6),
    # dark (phrygian degrees: 0=i 1=bII 3=iv 5=bVI 6=bvii)
    "i-bII-i-bII": (0, 1, 0, 1),
    "i-iv-bII-i": (0, 3, 1, 0),
    "i-bVI-bII-i": (0, 5, 1, 0),
    "i-bvii-bVI-bII": (0, 6, 5, 1),
    "i-i-bVI-bII": (0, 0, 5, 1),
    "i-iv-i-bII": (0, 3, 0, 1),
    # harmonic minor
    "i-VI-V-i": (0, 5, 4, 0),
    "i-iv-V-i": (0, 3, 4, 0),
    "i-i-VI-V": (0, 0, 5, 4),
    "i-V-VI-V": (0, 4, 5, 4),
    "iv-V-i-i": (3, 4, 0, 0),
    # more natural minor
    "i-VII-III-VI": (0, 6, 2, 5),
    "i-v-VI-iv": (0, 4, 5, 3),
    "VI-i-VII-III": (5, 0, 6, 2),
    "i-III-iv-VI": (0, 2, 3, 5),
    "i-iv-i-VI": (0, 3, 0, 5),
    "i-VI-iv-VII": (0, 5, 3, 6),
    # major / mixolydian (degrees are scale-relative)
    "I-V-vi-IV": (0, 4, 5, 3),
    "vi-IV-I-V": (5, 3, 0, 4),
    "I-vi-IV-V": (0, 5, 3, 4),
    "IV-I-V-vi": (3, 0, 4, 5),
    "I-bVII-IV-I": (0, 6, 3, 0),
    "I-IV-bVII-IV": (0, 3, 6, 3),
    "I-v-bVII-IV": (0, 4, 6, 3),
    # dorian
    "i-IV-i-IV": (0, 3, 0, 3),
    "i-bVII-IV-i": (0, 6, 3, 0),
    "i-ii-bIII-ii": (0, 1, 2, 1),
    "i-IV-bVII-i": (0, 3, 6, 0),
    "i-ii-IV-i": (0, 1, 3, 0),
    # locrian / phrygian dominant (tension)
    "i-bII-bv-i": (0, 1, 4, 0),
    "i-biii-bII-i": (0, 2, 1, 0),
    "i-bII-bVI-bv": (0, 1, 5, 4),
    "I-bII-I-bvii": (0, 1, 0, 6),
    "I-iv-bII-I": (0, 3, 1, 0),
    "I-bVI-bvii-I": (0, 5, 6, 0),
}
SCALES = {
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "phrygian_dominant": (0, 1, 4, 5, 7, 8, 10),
}
_QUALITY = {
    (0, 3, 7, 10): "m7",
    (0, 4, 7, 11): "maj7",
    (0, 4, 7, 10): "7",
    (0, 3, 6, 10): "m7b5",
    (0, 4, 8, 11): "maj7#5",
    (0, 3, 7, 11): "mMaj7",
    (0, 3, 6, 9): "dim7",
    (0, 3, 7): "m",
    (0, 4, 7): "",
    (0, 3, 6): "dim",
    (0, 4, 8): "aug",
}


@dataclass(frozen=True)
class Chord:
    """Chord."""

    root_pc: int
    degree: int
    intervals: tuple[int, ...]

    def notes(self, octave: int = 4) -> list[int]:
        """Notes."""
        root = 12 * octave + self.root_pc
        return [root + i for i in self.intervals]

    def bass_note(self) -> int:
        """Bass note."""
        return 12 * 3 + self.root_pc  # 36..47

    @property
    def name(self) -> str:
        """Name."""
        return NOTE_NAMES[self.root_pc % 12] + _QUALITY.get(self.intervals, "")


class Harmony:
    """Harmony."""

    MINOR = (0, 2, 3, 5, 7, 8, 10)
    MAJOR = (0, 2, 4, 5, 7, 9, 11)

    def __init__(self, rng: np.random.Generator, mood: Mood):
        """Initialize."""
        self.rng, self.mood = rng, mood
        self.tonic = int(rng.integers(0, 12))
        self.current: str | None = None
        self.set_mood(mood)

    def set_mood(self, mood: Mood) -> None:
        """Set mood."""
        self.mood = mood
        self.mode = (
            self.MAJOR
            if self.rng.random() < mood.major_prob
            else SCALES.get(mood.scale, self.MINOR)
        )

    @property
    def key_name(self) -> str:
        """Key name."""
        names = {v: k for k, v in SCALES.items()}
        return NOTE_NAMES[self.tonic] + " " + names.get(self.mode, "minor").replace("_", " ")

    def chord_for_degree(self, deg: int) -> Chord:
        """Chord for degree."""
        s = self.mode
        root = s[deg % 7]
        stack = (0, 2, 4, 6) if self.mood.sevenths else (0, 2, 4)
        intervals = tuple((s[(deg + k) % 7] - root) % 12 for k in stack)
        return Chord((self.tonic + root) % 12, deg % 7, intervals)

    def next_progression(self) -> list[Chord]:
        """Next progression."""
        names = [n for n in PROGRESSIONS if self.mood.progressions.get(n, 0) > 0]
        w = np.array(
            [self.mood.progressions[n] * (0.25 if n == self.current else 1.0) for n in names],
            dtype=float,
        )
        self.current = names[int(self.rng.choice(len(names), p=w / w.sum()))]
        return [self.chord_for_degree(d) for d in PROGRESSIONS[self.current]]

    def modulate(self) -> None:
        """Modulate."""
        self.tonic = (self.tonic + int(self.rng.choice([5, 7, -3, 3]))) % 12

    def pitch_classes(self, tonic: int | None = None, mode: tuple[int, ...] | None = None) -> set:
        """Pitch classes."""
        t = self.tonic if tonic is None else tonic
        return {(t + i) % 12 for i in (mode or self.mode)}

    def change_key(self, mood: Mood, last_chord: Chord | None = None) -> None:
        """Move to `mood`'s mode on the tonic that keeps the most notes in common.

        Score = shared pitch classes + 3 if `last_chord` fits the new key (pivot chord)
        + 1 for staying on the same tonic; the new tonic is drawn among the best candidates.
        """
        self.mood = mood
        new_mode = (
            self.MAJOR
            if self.rng.random() < mood.major_prob
            else SCALES.get(mood.scale, self.MINOR)
        )
        old = self.pitch_classes()
        chord_pcs = (
            {(last_chord.root_pc + i) % 12 for i in last_chord.intervals} if last_chord else set()
        )
        scored = []
        for t in range(12):
            pcs = self.pitch_classes(t, new_mode)
            score = len(old & pcs) + (3 if chord_pcs and chord_pcs <= pcs else 0)
            score += 1 if t == self.tonic else 0
            scored.append((score, t))
        best = max(sc for sc, _ in scored)
        cands = [t for sc, t in scored if sc >= best - 1]
        self.tonic = int(self.rng.choice(cands))
        self.mode = new_mode
        self.current = None

    def pivot_chord(self, prev: Chord) -> Chord:
        """Chord of the current key closest to `prev`: same root if it exists, else most
        common tones; used to bridge two keys during a transition.
        """
        prev_pcs = {(prev.root_pc + i) % 12 for i in prev.intervals}

        def score(deg: int) -> tuple[int, int]:
            """Score."""
            c = self.chord_for_degree(deg)
            pcs = {(c.root_pc + i) % 12 for i in c.intervals}
            return (int(c.root_pc == prev.root_pc), len(pcs & prev_pcs))

        return self.chord_for_degree(max(range(7), key=score))

    def scale_notes(self, low: int, high: int) -> list[int]:
        """Scale notes."""
        pcs = {(self.tonic + i) % 12 for i in self.mode}
        return [n for n in range(low, high + 1) if n % 12 in pcs]
