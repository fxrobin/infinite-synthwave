"""Section state machine turning harmony + generators into one BarPlan per bar."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .harmony import Chord, Harmony
from .moods import MOODS, Mood
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

LAYERS = ("drums", "bass", "arp", "pad", "lead", "ambient", "riser")
RISER_REV, RISER_UP, RISER_SCREAM, RISER_IMPACT, RISER_SHORT = 60, 61, 62, 63, 64


class Section(StrEnum):
    INTRO = "intro"
    VERSE = "verse"
    CHORUS = "chorus"
    BREAK = "break"
    TRANSITION = "transition"
    OUTRO = "outro"


SECTION_BARS = {Section.INTRO: 8, Section.VERSE: 16, Section.CHORUS: 16, Section.BREAK: 8,
                Section.TRANSITION: 4, Section.OUTRO: 8}
_NEXT = {Section.INTRO: [Section.VERSE],
         Section.VERSE: [Section.CHORUS, Section.CHORUS, Section.BREAK],
         Section.CHORUS: [Section.VERSE, Section.BREAK, Section.CHORUS],
         Section.BREAK: [Section.CHORUS, Section.CHORUS, Section.VERSE],
         Section.TRANSITION: [Section.VERSE, Section.CHORUS]}
_GAINS = {
    Section.INTRO: dict(drums=0.6, bass=0.8, arp=0.6, pad=1.0, lead=0.0, ambient=1.0, riser=1.0),
    Section.VERSE: dict(drums=1.0, bass=1.0, arp=0.85, pad=0.9, lead=0.35, ambient=0.7, riser=1.0),
    Section.CHORUS: dict(drums=1.0, bass=1.0, arp=1.0, pad=1.0, lead=1.0, ambient=0.5, riser=1.0),
    Section.BREAK: dict(drums=0.0, bass=0.6, arp=0.7, pad=1.0, lead=0.0, ambient=1.0, riser=1.0),
    Section.TRANSITION: dict(drums=0.0, bass=0.0, arp=0.0, pad=0.5, lead=0.0, ambient=1.0,
                             riser=1.0),
    Section.OUTRO: dict(drums=0.7, bass=0.8, arp=0.5, pad=1.0, lead=0.0, ambient=1.0, riser=1.0),
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
    fx: dict[str, list[dict]] | None = None   # layer (or "master") -> effect specs
    bpm: float | None = None                  # tempo change requested at this bar
    mood: str | None = None                   # mood in force from this bar
    patches: dict[str, str] | None = None     # per-section instrument choice (layer -> patch)


class Arranger:
    def __init__(self, rng: np.random.Generator, harmony: Harmony, mood: Mood,
                 total_bars: int | None = None,
                 bpm_range: tuple[float, float] | None = None):
        self.rng, self.harmony, self.mood, self.total_bars = rng, harmony, mood, total_bars
        self.bpm_range = bpm_range   # user override; None = follow the mood
        self.mood_locked = False     # True: keep the mood across transitions
        self.bar, self.sections_done = 0, 0
        self.section = Section.INTRO
        self.section_bar, self.section_len = 0, SECTION_BARS[Section.INTRO]
        self.next_section = Section.VERSE
        self.progression = harmony.next_progression()
        self.finished = False
        self.prev_patterns: dict[str, Pattern] | None = None
        self.pending_mood: Mood | None = None
        self.transition_requested = False
        self.bar_bpm: float | None = None
        self.mood_changed = False
        self._new_styles()

    def set_mood(self, mood: Mood | None) -> None:
        """Schedule a mood change (applied at the next transition) and lock it.

        None unlocks: every following transition draws a new random mood."""
        if mood is None:
            self.mood_locked = False
            self.transition_requested = True
            return
        self.mood_locked = True
        if mood != self.mood:
            self.pending_mood = mood
            self.transition_requested = True

    def force_next_section(self) -> None:
        self.section_bar = self.section_len

    def _density(self) -> float:
        return self.mood.drum_density * (0.3 if self.section == Section.INTRO else 1.0)

    def _new_styles(self) -> None:
        r = self.rng
        styles = self.mood.bass_styles or {"eighths": 3, "octaves": 2, "syncopated": 1}
        w = np.array(list(styles.values()), dtype=float)
        self.bass_style = str(list(styles)[int(r.choice(len(styles), p=w / w.sum()))])
        self.section_patches = {layer: str(pool[int(r.integers(len(pool)))])
                                for layer, pool in self.mood.pools.items()}
        self.bass_base: Pattern | None = None
        self.arp_mode = str(r.choice(["up", "updown", "random"], p=[0.45, 0.4, 0.15]))
        self.arp_on = self.section == Section.CHORUS or r.random() < self.mood.arp_prob
        self.drums_base = gen_drums(r, self._density(), snare=self.section != Section.INTRO,
                                    halftime=self.mood.halftime,
                                    strong=self.section == Section.CHORUS)
        if self.section == Section.CHORUS:
            self.bass_style = str(r.choice(["eighths", "sixteenths", "octaves"],
                                           p=[0.5, 0.35, 0.15]))
            self.arp_on = True
        self.fx = self._section_fx()

    def _section_fx(self) -> dict[str, list[dict]]:
        r, fx = self.rng, {}
        energy = self.mood.drum_density
        if self.section == Section.CHORUS:
            if r.random() < 0.35 + 0.4 * energy:
                fx["pad"] = [{"type": "gate", "rate": str(r.choice(["1/16", "1/8", "1/32"])),
                              "depth": 0.85, "duty": 0.5}]
            if r.random() < 0.3:
                fx["arp"] = [{"type": "bitcrush", "bits": 8, "downsample": 2, "mix": 0.5}]
            lead_pool = [
                [{"type": "autopan", "rate": str(r.choice(["1/2", "1/4", "1/1"])), "depth": 0.9}],
                [{"type": "gate", "rate": "1/16", "depth": 0.7, "duty": 0.5},
                 {"type": "autopan", "rate": "1/2", "depth": 0.6}],
                [{"type": "distortion", "drive": 5.0, "tone": 3000, "mix": 0.7},
                 {"type": "autopan", "rate": "1/4", "depth": 0.5}],
                [{"type": "bitcrush", "bits": 7, "downsample": 3, "mix": 0.45}],
                [{"type": "phaser", "rate": "2/1", "depth": 0.9, "stages": 6, "mix": 0.6}],
                [{"type": "flanger", "rate": 0.2, "feedback": 0.6, "mix": 0.5},
                 {"type": "distortion", "drive": 3.0, "tone": 3500, "mix": 0.5}],
            ]
            fx["lead"] = lead_pool[int(r.integers(len(lead_pool)))]
        elif self.section == Section.BREAK:
            if r.random() < 0.6:
                fx["master"] = [{"type": "lofi", "bits": 10, "downsample": 3, "cutoff": 3500,
                                 "wobble": 0.003, "noise": 0.006}]
            elif r.random() < 0.5:
                fx["pad"] = [{"type": "gate", "rate": "1/8", "depth": 0.6, "duty": 0.5}]
        elif self.section == Section.INTRO:
            if r.random() < 0.5:
                fx["master"] = [{"type": "lofi", "bits": 11, "downsample": 2, "cutoff": 3000,
                                 "wobble": 0.002, "noise": 0.004, "mix": 0.85}]
        elif self.section == Section.VERSE and r.random() < 0.25:
            fx["arp"] = [{"type": "gate", "rate": "1/32", "depth": 0.5, "duty": 0.5}]
        return fx

    def _outro_due(self) -> bool:
        return (self.total_bars is not None
                and self.bar >= self.total_bars - SECTION_BARS[Section.OUTRO])

    def _transition_due(self) -> bool:
        if self.section == Section.TRANSITION:
            return False
        if self.transition_requested:
            return True
        if self.section == Section.INTRO:
            return False
        return (self.sections_done % 6 == 0
                or (self.sections_done >= 3 and self.rng.random() < 0.25))

    def draw_bpm(self) -> float:
        lo, hi = self.bpm_range or self.mood.bpm_range
        return round(float(self.rng.uniform(lo, hi)), 1)

    def _enter_transition(self) -> None:
        """Ambient-only bars carrying the key / tempo / mood changes."""
        self.transition_requested = False
        new_mood = self.pending_mood
        self.pending_mood = None
        if new_mood is None and not self.mood_locked:
            others = [m for m in MOODS.values() if m is not self.mood]
            new_mood = others[int(self.rng.integers(len(others)))]
        if new_mood is not None:
            self.mood = new_mood
            self.harmony.set_mood(new_mood)
            self.mood_changed = True
        if new_mood is not None or self.rng.random() < 0.7:
            self.bar_bpm = self.draw_bpm()
        if self.rng.random() < 0.7:
            self.harmony.modulate()
        self.progression = self.harmony.next_progression()

    def _start_section(self) -> None:
        self.sections_done += 1
        if self._outro_due():
            self.section = Section.OUTRO
        elif self._transition_due():
            self.section = Section.TRANSITION
            self._enter_transition()
        else:
            self.section = self.next_section
            if self.rng.random() < 0.5:
                self.progression = self.harmony.next_progression()
        self.section_bar, self.section_len = 0, SECTION_BARS[self.section]
        self.next_section = Section(self.rng.choice(
            [s.value for s in _NEXT.get(self.section, [Section.VERSE])]))
        self._new_styles()

    def _risers(self) -> Pattern:
        """Announce the coming chorus on the two last bars; drop an impact on its first beat."""
        remaining = self.section_len - self.section_bar   # bars left including this one
        going_to_chorus = self.next_section == Section.CHORUS and self.section != Section.OUTRO
        p: Pattern = []
        if going_to_chorus and remaining == 2:
            p.append(Note(0, RISER_UP, 0.9, 32))
        if going_to_chorus and remaining == 1:
            p.append(Note(0, RISER_REV, 1.0, 16))
            if self.rng.random() < 0.6:
                p.append(Note(8, RISER_SCREAM, 0.8, 8))
        if remaining == 1 and not going_to_chorus and self.section not in (
                Section.TRANSITION, Section.OUTRO) and self.rng.random() < 0.5:
            p.append(Note(8, RISER_SHORT, 0.8, 8))
        if self.section == Section.CHORUS and self.section_bar == 0:
            p.append(Note(0, RISER_IMPACT, 1.0, 8))
        return p

    def _bass(self, r: np.random.Generator, chord: Chord) -> Pattern:
        """Regenerate on the chord each bar; every second bar apply a light mutation."""
        p = gen_bass(r, chord, self.bass_style)
        if self.section_bar % 2 == 1:
            root = chord.bass_note()
            p = mutate(r, p, 0.2, [root, root + 12, root + 7, root + 1, root + 6])
        return p

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
        if self.section == Section.TRANSITION:
            drums = []
        elif last and self.section != Section.OUTRO:
            drums = gen_drums(r, density, fill=True, snare=self.section != Section.INTRO,
                              halftime=self.mood.halftime, strong=self.section == Section.CHORUS)
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
        lead_p = (max(self.mood.lead_prob, 0.7) if self.section == Section.CHORUS
                  else self.mood.lead_prob * 0.5)
        lo = 55 if self.mood.pad_octave <= 3 else 60
        scale = self.harmony.scale_notes(lo, lo + 19)
        lead = gen_lead(r, chord, scale, density) if r.random() < lead_p else []
        patterns = {
            "drums": drums,
            "bass": self._bass(r, chord),
            "arp": gen_arp(r, chord, self.arp_mode) if self.arp_on else [],
            "pad": gen_pad(chord, self.mood.pad_octave),
            "lead": lead,
            "ambient": gen_ambient(chord),
            "riser": self._risers() if self.section != Section.TRANSITION else [],
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
                       fill=last, fade=fade, key=self.harmony.key_name, fx=self.fx,
                       bpm=self.bar_bpm, mood=self.mood.name if self.mood_changed else None,
                       patches=self.section_patches if first else None)
        self.bar_bpm, self.mood_changed = None, False
        self.bar += 1
        self.section_bar += 1
        return plan
