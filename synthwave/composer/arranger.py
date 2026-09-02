"""Section state machine turning harmony + generators into one BarPlan per bar."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .harmony import Chord, Harmony
from .moods import Mood
from .patterns import (
    CRASH,
    HAT_C,
    Note,
    Pattern,
    gen_ambient,
    gen_arp,
    gen_bass,
    gen_drums,
    gen_lead,
    gen_pad,
    mutate,
)

LAYERS = ("drums", "bass", "arp", "pad", "lead", "ambient")


class Section(StrEnum):
    INTRO = "intro"
    VERSE = "verse"
    CHORUS = "chorus"
    BREAK = "break"
    OUTRO = "outro"


SECTION_BARS = {Section.INTRO: 8, Section.VERSE: 16, Section.CHORUS: 16, Section.BREAK: 8,
                Section.OUTRO: 8}
_NEXT = {Section.INTRO: [Section.VERSE],
         Section.VERSE: [Section.CHORUS, Section.CHORUS, Section.BREAK],
         Section.CHORUS: [Section.VERSE, Section.BREAK, Section.CHORUS],
         Section.BREAK: [Section.CHORUS, Section.CHORUS, Section.VERSE]}
_GAINS = {
    Section.INTRO: dict(drums=0.6, bass=0.8, arp=0.6, pad=1.0, lead=0.0, ambient=1.0),
    Section.VERSE: dict(drums=1.0, bass=1.0, arp=0.85, pad=0.9, lead=0.35, ambient=0.7),
    Section.CHORUS: dict(drums=1.0, bass=1.0, arp=1.0, pad=1.0, lead=1.0, ambient=0.5),
    Section.BREAK: dict(drums=0.0, bass=0.6, arp=0.7, pad=1.0, lead=0.0, ambient=1.0),
    Section.OUTRO: dict(drums=0.7, bass=0.8, arp=0.5, pad=1.0, lead=0.0, ambient=1.0),
}


@dataclass(frozen=True)
class BarPlan:
    bar: int
    section: Section
    section_bar: int
    chord: Chord
    patterns: dict[str, Pattern]
    gains: dict[str, float]
    fill: bool = False
    fade: float = 1.0
    finished: bool = False
    key: str = ""


class Arranger:
    def __init__(self, rng: np.random.Generator, harmony: Harmony, mood: Mood,
                 total_bars: int | None = None):
        self.rng, self.harmony, self.mood, self.total_bars = rng, harmony, mood, total_bars
        self.bar, self.sections_done = 0, 0
        self.section = Section.INTRO
        self.section_bar, self.section_len = 0, SECTION_BARS[Section.INTRO]
        self.progression = harmony.next_progression()
        self.finished = False
        self.prev_patterns: dict[str, Pattern] | None = None
        self._new_styles()

    def set_mood(self, mood: Mood) -> None:
        self.mood = mood
        self.harmony.set_mood(mood)

    def force_next_section(self) -> None:
        self.section_bar = self.section_len

    def _density(self) -> float:
        return self.mood.drum_density * (0.3 if self.section == Section.INTRO else 1.0)

    def _new_styles(self) -> None:
        r = self.rng
        self.bass_style = str(r.choice(["eighths", "octaves", "syncopated"], p=[0.5, 0.3, 0.2]))
        self.arp_mode = str(r.choice(["up", "updown", "random"], p=[0.45, 0.4, 0.15]))
        self.arp_on = self.section == Section.CHORUS or r.random() < self.mood.arp_prob
        self.drums_base = gen_drums(r, self._density(), snare=self.section != Section.INTRO)

    def _outro_due(self) -> bool:
        return (self.total_bars is not None
                and self.bar >= self.total_bars - SECTION_BARS[Section.OUTRO])

    def _start_section(self) -> None:
        self.sections_done += 1
        if self._outro_due():
            self.section = Section.OUTRO
        else:
            self.section = Section(self.rng.choice([s.value for s in _NEXT[self.section]]))
        self.section_bar, self.section_len = 0, SECTION_BARS[self.section]
        if self.sections_done % 6 == 0:
            self.harmony.modulate()
        if self.rng.random() < 0.6 or self.sections_done % 6 == 0:
            self.progression = self.harmony.next_progression()
        self._new_styles()

    def _silence(self) -> BarPlan:
        plan = BarPlan(self.bar, Section.OUTRO, 0, self.progression[0],
                       {layer: [] for layer in LAYERS}, {layer: 0.0 for layer in LAYERS},
                       fade=0.0, finished=True, key=self.harmony.key_name)
        self.bar += 1
        return plan

    def next_bar(self) -> BarPlan:
        if self.finished or (self.total_bars is not None and self.bar >= self.total_bars):
            self.finished = True
            return self._silence()
        if self.section_bar >= self.section_len or (
                self._outro_due() and self.section != Section.OUTRO):
            self._start_section()
        r = self.rng
        chord = self.progression[self.section_bar % len(self.progression)]
        last = self.section_bar == self.section_len - 1
        first = self.section_bar == 0
        density = self._density()
        if last and self.section != Section.OUTRO:
            drums = gen_drums(r, density, fill=True, snare=self.section != Section.INTRO)
        else:
            if self.section_bar % 4 == 0 and not first:
                self.drums_base = mutate(r, self.drums_base, 0.15, [HAT_C])
            drums = list(self.drums_base)
            if first and self.section in (Section.VERSE, Section.CHORUS):
                drums.append(Note(0, CRASH, 0.8, 1))
        gains = dict(_GAINS[self.section])
        if self.section == Section.BREAK and last:
            gains["drums"] = 1.0
        if not self.arp_on:
            gains["arp"] = 0.0
        lead_p = self.mood.lead_prob * (1.0 if self.section == Section.CHORUS else 0.5)
        scale = self.harmony.scale_notes(72, 84)
        lead = gen_lead(r, chord, scale, density) if r.random() < lead_p else []
        patterns = {
            "drums": drums,
            "bass": gen_bass(r, chord, self.bass_style),
            "arp": gen_arp(r, chord, self.arp_mode) if self.arp_on else [],
            "pad": gen_pad(chord),
            "lead": lead,
            "ambient": gen_ambient(chord),
        }
        tries = 0
        while self.prev_patterns is not None and patterns == self.prev_patterns and tries < 4:
            root = chord.bass_note()
            patterns["bass"] = mutate(r, patterns["bass"], 0.5, [root, root + 12, root + 7])
            patterns["drums"] = mutate(r, patterns["drums"], 0.3, [HAT_C])
            tries += 1
        self.prev_patterns = patterns
        fade = 1.0 - self.section_bar / self.section_len if self.section == Section.OUTRO else 1.0
        plan = BarPlan(self.bar, self.section, self.section_bar, chord, patterns, gains,
                       fill=last, fade=fade, key=self.harmony.key_name)
        self.bar += 1
        self.section_bar += 1
        return plan
