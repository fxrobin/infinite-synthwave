"""
Section state machine turning harmony + generators into one BarPlan per bar.

The stream is a sequence of *tracks* of about `track_s` seconds each: intro -> verse /
chorus / break ... -> outro -> transition -> next track. Inside a section, layers enter
one after the other every two bars (build-up); the bar before a chorus is a pre-drop
(percussion cut, snare roll) so the chorus lands as a drop.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .harmony import Chord, Harmony
from .moods import MOODS, Mood
from .patterns import (
    CRASH,
    CROLL,
    Note,
    Pattern,
    add_roll,
    add_straight_fill,
    cut_after,
    drum_layer,
    gen_ambient,
    gen_arp,
    gen_bass,
    gen_drums,
    gen_pad,
    gen_predrop,
    gen_theme,
    harmonize,
    mutate,
    render_motif,
)

LAYERS = ("drums", "bass", "arp", "pad", "lead", "lead2", "ambient", "riser")
RISER_REV, RISER_UP, RISER_SCREAM, RISER_IMPACT, RISER_SHORT = 60, 61, 62, 63, 64
TRACK_SECONDS = 210.0  # ~3'30 per track
MAX_TRACK_SECTIONS = 8  # safety net when sections are forced by hand


class Section(StrEnum):
    """Section."""

    INTRO = "intro"
    VERSE = "verse"
    CHORUS = "chorus"
    BREAK = "break"
    TRANSITION = "transition"
    OUTRO = "outro"


SECTION_BARS = {
    Section.INTRO: 8,
    Section.VERSE: 16,
    Section.CHORUS: 16,
    Section.BREAK: 8,
    Section.TRANSITION: 4,
    Section.OUTRO: 8,
}
_NEXT = {
    Section.INTRO: [Section.VERSE],
    Section.VERSE: [Section.CHORUS, Section.CHORUS, Section.BREAK],
    Section.CHORUS: [Section.VERSE, Section.BREAK, Section.CHORUS],
    Section.BREAK: [Section.CHORUS, Section.CHORUS, Section.VERSE],
    Section.OUTRO: [Section.TRANSITION],
    Section.TRANSITION: [Section.INTRO],
}
_ROLL_SECTIONS = (Section.VERSE, Section.CHORUS, Section.OUTRO)
# Master colour picked by the composition (names from audio.renderer.MASTER_COLORS).
# Bright moods stay close to clean tape, dark ones wear the tape out.
_MASTER_COLORS_BRIGHT = (("tape", 5), ("clean", 3), ("vhs", 2), ("mic", 1))
_MASTER_COLORS_DARK = (("vhs", 4), ("mic", 3), ("tape", 2), ("crush", 1))
_BREAK_COLORS = ("vhs", "mic", "crush")  # a break drops the mix onto worn tape
# Ladder from cleanest to dirtiest: sections move along it around the track's colour.
_COLOR_LADDER = ("clean", "tape", "vhs", "mic", "crush")

_GAINS = {
    Section.INTRO: dict(
        drums=0.6, bass=0.8, arp=0.6, pad=1.0, lead=0.0, lead2=0.0, ambient=1.0, riser=1.0
    ),
    Section.VERSE: dict(
        drums=1.0, bass=1.0, arp=0.85, pad=0.9, lead=0.35, lead2=0.25, ambient=0.7, riser=1.0
    ),
    Section.CHORUS: dict(
        drums=1.0, bass=1.0, arp=1.0, pad=1.0, lead=1.0, lead2=0.7, ambient=0.5, riser=1.0
    ),
    Section.BREAK: dict(
        drums=1.0, bass=0.6, arp=0.7, pad=1.0, lead=0.6, lead2=0.45, ambient=1.0, riser=1.0
    ),
    Section.TRANSITION: dict(
        drums=0.0, bass=0.0, arp=0.0, pad=0.5, lead=0.0, lead2=0.0, ambient=1.0, riser=1.0
    ),
    Section.OUTRO: dict(
        drums=0.7, bass=0.8, arp=0.5, pad=1.0, lead=0.0, lead2=0.0, ambient=1.0, riser=1.0
    ),
}
# Build-up: bar of the section at which a layer enters (absent = from the first bar).
_ENTRY = {
    Section.INTRO: {"arp": 2, "drums": 4, "bass": 6},
    Section.VERSE: {"arp": 2, "lead": 4, "lead2": 8},
    Section.CHORUS: {"lead": 2, "lead2": 4},
    Section.BREAK: {"lead": 2, "lead2": 4, "arp": 2, "bass": 4, "drums": 6},
}
# Tear-down (outro): bar from which a layer is gone.
_EXIT = {Section.OUTRO: {"arp": 2, "bass": 6, "drums": 6}}
# Live composer: per-section patch gestures, drawn per layer (path -> multiplier).
_GESTURES = {
    "pad": [
        {"filter.cutoff": 0.65},
        {"filter.cutoff": 1.4},
        {"filter.resonance": 2.5},
        {"lfo.rate": 2.0, "lfo.amount": 1.5},
        {"amp_env.attack": 2.0},
        {"oscillators.0.detune": 1.6},
        {"effects.0.mix": 1.5},
    ],
    "arp": [
        {"filter.cutoff": 0.6},
        {"filter.env.amount": 1.6},
        {"filter.resonance": 1.8},
        {"amp_env.decay": 0.6},
        {"amp_env.decay": 1.6},
        {"oscillators.0.pwm": 0.5},
    ],
    "bass": [
        {"filter.cutoff": 0.7},
        {"filter.cutoff": 1.5},
        {"filter.env.amount": 1.8},
        {"oscillators.0.detune": 1.5},
        {"glide": 2.5},
    ],
    "lead": [
        {"oscillators.0.detune": 1.6},
        {"lfo.amount": 2.0},
        {"lfo.rate": 0.6},
        {"glide": 2.0},
        {"filter.cutoff": 0.7},
        {"filter.resonance": 1.8},
    ],
    "ambient": [{"filter.cutoff": 0.6}, {"lfo.rate": 2.5}, {"amp_env.attack": 0.5}],
}
_GESTURE_PROB = {
    Section.INTRO: 0.5,
    Section.VERSE: 0.45,
    Section.CHORUS: 0.55,
    Section.BREAK: 0.7,
    Section.OUTRO: 0.4,
    Section.TRANSITION: 0.0,
}

# Drum build-up levels (see patterns.drum_layer): (from bar, level), last match wins.
_DRUM_LEVELS = {
    Section.INTRO: ((0, 0), (6, 1)),
    Section.VERSE: ((0, 1), (4, 2)),
    Section.CHORUS: ((0, 2),),
    Section.BREAK: ((0, 0),),
    Section.OUTRO: ((0, 1), (4, 0)),
}


@dataclass(frozen=True)
class BarPlan:
    """Barplan."""

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
    fx: dict[str, list[dict]] | None = None  # layer (or "master") -> effect specs
    bpm: float | None = None  # tempo change requested at this bar
    mood: str | None = None  # mood in force from this bar
    patches: dict[str, str] | None = None  # per-section instrument choice (layer -> patch)
    drop: bool = False  # pre-drop bar: percussion cut before a hit
    track: int = 0
    track_bar: int = 0
    track_bars: int = 0
    tweaks: dict[str, dict[str, float]] | None = None  # live patch gestures (layer -> path -> ×)
    master_color: str | None = None  # master colour for this section (None = unchanged)


class Arranger:
    """Arranger."""

    def __init__(  # noqa: PLR0913 - arranger needs RNG, harmony, mood, bars, BPM and track length
        self,
        rng: np.random.Generator,
        harmony: Harmony,
        mood: Mood,
        total_bars: int | None = None,
        bpm_range: tuple[float, float] | None = None,
        bpm: float | None = None,
        track_s: float = TRACK_SECONDS,
    ):
        """Initialize arranger."""
        self.rng, self.harmony, self.mood, self.total_bars = rng, harmony, mood, total_bars
        self.bpm_range = bpm_range  # user override; None = follow the mood
        self.bpm = float(bpm) if bpm else float(mood.bpm)
        self.track_s = float(track_s)
        self.mood_locked = False  # True: keep the mood across transitions
        self.bar, self.sections_done = 0, 0
        self.track, self.track_bar, self.track_bars, self.track_sections = 0, 0, 0, 0
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
        self.mid_drop = False
        self._new_track()
        self._new_styles()

    # ----- external control -----
    def set_mood(self, mood: Mood | None) -> None:
        """
        Schedule a mood change (applied after an outro, at the next transition) and lock
        it.

        None unlocks: every following transition draws a new random mood.
        """
        if mood is None:
            self.mood_locked = False
            self.transition_requested = True
            return
        self.mood_locked = True
        if mood != self.mood:
            self.pending_mood = mood
            self.transition_requested = True

    def force_next_section(self) -> None:
        """Force next section."""
        self.section_bar = self.section_len

    def draw_bpm(self) -> float:
        """Tempo for the next track: user range -> uniform; else the mood's range,
        staying as close as possible to the current tempo (smooth transitions).
        """
        if self.bpm_range is not None:
            lo, hi = self.bpm_range
            return round(float(self.rng.uniform(lo, hi)), 1)
        lo, hi = self.mood.bpm_range
        target = float(np.clip(self.bpm, lo, hi)) + float(self.rng.uniform(-4.0, 4.0))
        return round(float(np.clip(target, lo, hi)), 1)

    # ----- track / section bookkeeping -----
    def _bar_seconds(self) -> float:
        """Bar seconds."""
        return 4.0 * 60.0 / self.bpm

    def _new_track(self) -> None:
        """New track."""
        self.track += 1
        self.track_bar, self.track_sections = 0, 0
        # one theme per track; the eighties moods play it short and detached
        self.theme = gen_theme(self.rng, self.mood.drum_density, staccato=self.mood.straight)
        self.track_color = self._draw_master_color()
        secs = self.track_s * float(self.rng.uniform(0.92, 1.08))
        bars = int(round(secs / self._bar_seconds() / 4.0)) * 4
        self.track_bars = max(bars, SECTION_BARS[Section.INTRO] + 8 + SECTION_BARS[Section.OUTRO])

    def _density(self) -> float:
        """Density."""
        return self.mood.drum_density * (0.3 if self.section == Section.INTRO else 1.0)

    def _new_styles(self) -> None:
        """New styles."""
        r = self.rng
        styles = self.mood.bass_styles or {"eighths": 3, "octaves": 2, "syncopated": 1}
        w = np.array(list(styles.values()), dtype=float)
        self.bass_style = str(list(styles)[int(r.choice(len(styles), p=w / w.sum()))])
        self.section_patches = {
            layer: str(pool[int(r.integers(len(pool)))]) for layer, pool in self.mood.pools.items()
        }
        self.bass_base: Pattern | None = None
        modes, weights = ["up", "updown", "random"], [0.45, 0.4, 0.15]
        if self.mood.straight:  # eighties: the arp is a fixed ostinato, never a random walk
            modes, weights = ["up", "updown"], [0.5, 0.5]
        self.arp_mode = str(r.choice(modes, p=weights))
        self.arp_on = self.section == Section.CHORUS or r.random() < self.mood.arp_prob
        # second lead: a diatonic third or sixth under the theme, fixed for the section
        self.harmony_degrees = int(r.choice([-2, -2, -2, -5, -7]))
        m = self.mood
        self.drums_base = gen_drums(
            r,
            self._density(),
            snare=self.section != Section.INTRO,
            halftime=m.halftime,
            strong=self.section == Section.CHORUS,
            snap=r.random() < m.snap_prob,
            ride=r.random() < m.ride_prob,
            shaker=r.random() < m.shaker_prob,
            tick=r.random() < m.tick_prob,
            straight=m.straight,
        )
        self.mid_drop = False
        self.gestures: dict[str, dict[str, float]] = {}
        prob = _GESTURE_PROB.get(self.section, 0.0)
        for layer, pool in _GESTURES.items():
            if r.random() < prob:
                self.gestures[layer] = dict(pool[int(r.integers(len(pool)))])
        if self.section == Section.CHORUS:
            self.bass_style = str(
                r.choice(["eighths", "sixteenths", "octaves"], p=[0.5, 0.35, 0.15])
            )
            self.arp_on = True
            self.mid_drop = self.section_len >= 16 and r.random() < 0.4
        self.fx = self._section_fx()
        self.section_color = self._section_color()

    def _draw_master_color(self) -> str:
        """Master colour of a track, weighted by how bright the mood is."""
        table = _MASTER_COLORS_BRIGHT if self.mood.brightness >= 0.9 else _MASTER_COLORS_DARK
        names = [n for n, _ in table]
        w = np.array([x for _, x in table], dtype=float)
        return str(self.rng.choice(names, p=w / w.sum()))

    def _section_color(self) -> str:
        """Colour for the section starting here, around the track's own colour: a break
        drops onto worn tape, an intro often comes in one notch dirtier, a chorus often
        opens up one notch cleaner, everything else plays the track colour.
        """
        if self.section == Section.BREAK:
            return str(self.rng.choice(_BREAK_COLORS))
        step = 0
        if self.section == Section.INTRO and self.rng.random() < 0.5:
            step = 1
        elif self.section == Section.CHORUS and self.rng.random() < 0.4:
            step = -1
        if not step:
            return self.track_color
        i = _COLOR_LADDER.index(self.track_color) if self.track_color in _COLOR_LADDER else 1
        return _COLOR_LADDER[int(np.clip(i + step, 0, len(_COLOR_LADDER) - 1))]

    def _section_fx(self) -> dict[str, list[dict]]:
        """Section fx."""
        r, fx = self.rng, {}
        energy = self.mood.drum_density
        if self.section == Section.CHORUS:
            if r.random() < 0.35 + 0.4 * energy:
                fx["pad"] = [
                    {
                        "type": "gate",
                        "rate": str(r.choice(["1/16", "1/8", "1/32"])),
                        "depth": 0.85,
                        "duty": 0.5,
                    }
                ]
            if r.random() < 0.3:
                fx["arp"] = [{"type": "bitcrush", "bits": 8, "downsample": 2, "mix": 0.5}]
            lead_pool = [
                [{"type": "autopan", "rate": str(r.choice(["1/2", "1/4", "1/1"])), "depth": 0.9}],
                [
                    {"type": "gate", "rate": "1/16", "depth": 0.7, "duty": 0.5},
                    {"type": "autopan", "rate": "1/2", "depth": 0.6},
                ],
                [
                    {"type": "distortion", "drive": 5.0, "tone": 3000, "mix": 0.7},
                    {"type": "autopan", "rate": "1/4", "depth": 0.5},
                ],
                [{"type": "bitcrush", "bits": 7, "downsample": 3, "mix": 0.45}],
                [{"type": "phaser", "rate": "2/1", "depth": 0.9, "stages": 6, "mix": 0.6}],
                [
                    {"type": "flanger", "rate": 0.2, "feedback": 0.6, "mix": 0.5},
                    {"type": "distortion", "drive": 3.0, "tone": 3500, "mix": 0.5},
                ],
            ]
            fx["lead"] = lead_pool[int(r.integers(len(lead_pool)))]
        elif self.section == Section.BREAK:
            if r.random() < 0.6:
                fx["master"] = [
                    {
                        "type": "lofi",
                        "bits": 10,
                        "downsample": 3,
                        "cutoff": 3500,
                        "wobble": 0.003,
                        "noise": 0.006,
                    }
                ]
            elif r.random() < 0.5:
                fx["pad"] = [{"type": "gate", "rate": "1/8", "depth": 0.6, "duty": 0.5}]
        elif self.section == Section.INTRO:
            if r.random() < 0.5:
                fx["master"] = [
                    {
                        "type": "lofi",
                        "bits": 11,
                        "downsample": 2,
                        "cutoff": 3000,
                        "wobble": 0.002,
                        "noise": 0.004,
                        "mix": 0.85,
                    }
                ]
        elif self.section == Section.VERSE and r.random() < 0.25:
            fx["arp"] = [{"type": "gate", "rate": "1/32", "depth": 0.5, "duty": 0.5}]
        return fx

    def _final_outro_due(self) -> bool:
        """Duration mode: the whole stream ends with a fading outro."""
        return (
            self.total_bars is not None
            and self.bar >= self.total_bars - SECTION_BARS[Section.OUTRO]
        )

    def _track_outro_due(self) -> bool:
        """Track outro due."""
        room = self.track_bars - SECTION_BARS[Section.OUTRO] - self.track_bar
        return room < 8 or self.track_sections >= MAX_TRACK_SECTIONS

    def _pick_mood(self) -> Mood:
        """Next track's mood, weighted towards moods close to the current one (tempo,
        feel).
        """
        others = [m for m in MOODS.values() if m is not self.mood]
        w = []
        for m in others:
            lo, hi = m.bpm_range
            weight = 1.0
            if lo * 0.88 <= self.bpm <= hi * 1.12:
                weight += 2.0
            if m.halftime == self.mood.halftime:
                weight += 1.0
            if m.scale == self.mood.scale:
                weight += 1.0
            w.append(weight)
        w = np.array(w) / sum(w)
        return others[int(self.rng.choice(len(others), p=w))]

    def _enter_transition(self) -> None:
        """Ambient-only bars bridging two tracks: pivot chord, related key, nearby
        tempo.
        """
        self.transition_requested = False
        last_chord = self.progression[(self.section_bar - 1) % len(self.progression)]
        new_mood = self.pending_mood
        self.pending_mood = None
        if new_mood is None and not self.mood_locked:
            new_mood = self._pick_mood()
        if new_mood is not None:
            self.mood = new_mood
            self.mood_changed = True
        self.harmony.change_key(self.mood, last_chord)
        self.bar_bpm = self.draw_bpm()
        self.bpm = self.bar_bpm
        pivot = self.harmony.pivot_chord(last_chord)
        self.progression = [pivot] * SECTION_BARS[Section.TRANSITION]

    def _start_section(self) -> None:
        """Start section."""
        self.sections_done += 1
        self.track_sections += 1
        if self._final_outro_due():
            self.section = Section.OUTRO
        elif self.section == Section.OUTRO:
            self.section = Section.TRANSITION
            self._enter_transition()
        elif self.section == Section.TRANSITION:
            self.section = Section.INTRO
            self._new_track()
            self.progression = self.harmony.next_progression()
        elif self.transition_requested and self.section == Section.INTRO:
            self.section = Section.TRANSITION  # nothing to wind down yet
            self._enter_transition()
        elif self.transition_requested or self._track_outro_due():
            self.section = Section.OUTRO
        else:
            self.section = self.next_section
            if self.rng.random() < 0.5:
                self.progression = self.harmony.next_progression()
        self.section_bar, self.section_len = 0, SECTION_BARS[self.section]
        if self.section not in (Section.OUTRO, Section.TRANSITION, Section.INTRO):
            room = self.track_bars - SECTION_BARS[Section.OUTRO] - self.track_bar
            self.section_len = max(4, min(self.section_len, room - room % 4))
        self.next_section = Section(
            self.rng.choice([s.value for s in _NEXT.get(self.section, [Section.VERSE])])
        )
        self._new_styles()

    # ----- per-bar helpers -----
    def _going_to_chorus(self) -> bool:
        """Going to chorus."""
        return self.next_section == Section.CHORUS and self.section not in (
            Section.OUTRO,
            Section.TRANSITION,
        )

    def _is_predrop(self) -> bool:
        """Percussion cut before a hit: last bar before a chorus, or mid-chorus (bar
        7).
        """
        last = self.section_bar == self.section_len - 1
        if last and self._going_to_chorus():
            return True
        return self.section == Section.CHORUS and self.mid_drop and self.section_bar == 7

    def _drum_level(self) -> int:
        """Drum level."""
        level = 2
        for at, lvl in _DRUM_LEVELS.get(self.section, ()):
            if self.section_bar >= at:
                level = lvl
        return level

    def _gains(self, predrop: bool) -> dict[str, float]:
        """Gains."""
        gains = dict(_GAINS[self.section])
        for layer, at in _ENTRY.get(self.section, {}).items():
            if self.section_bar < at:
                gains[layer] = 0.0
        for layer, at in _EXIT.get(self.section, {}).items():
            if self.section_bar >= at:
                gains[layer] = 0.0
        if predrop:  # the melody clears out before the drop, harmony included
            gains["lead"] = gains["lead2"] = 0.0
            gains["drums"] = 1.0
        if not self.arp_on:
            gains["arp"] = 0.0
        return gains

    def _drums(self, r: np.random.Generator, predrop: bool, density: float) -> Pattern:
        """Drums."""
        if self.section == Section.TRANSITION:
            return []
        first, last = self.section_bar == 0, self.section_bar == self.section_len - 1
        if predrop:
            return gen_predrop(r, self.drums_base)
        base = drum_layer(self.drums_base, self._drum_level())
        if last and self.section != Section.OUTRO:
            # fill on the 4: a plain snare crescendo for a straight groove, a tom roll otherwise
            drums = add_straight_fill(base) if self.mood.straight else add_roll(r, base, 12, 4)
            if r.random() < 0.4:
                drums.append(Note(0, CROLL, 0.8, 16))  # cymbal roll into the next section
            return drums
        drums = list(base)  # groove stays fixed per section
        if (
            not self.mood.straight  # a straight groove never breaks mid-section
            and self.section in _ROLL_SECTIONS
            and self.section_bar % 4 == 3
            and r.random() < 0.3 + density * 0.5
        ):
            if self.mood.halftime:
                drums = add_roll(r, drums, 6, 2)  # pickup into the snare on 3
            else:
                drums = add_roll(r, drums, 8, 4)  # roll on the 3
        hit = first or (
            self.section == Section.CHORUS and self.section_bar in (8, 12) and r.random() < 0.6
        )
        if hit and self.section in (Section.VERSE, Section.CHORUS, Section.BREAK):
            drums.append(Note(0, CRASH, 0.8 if first else 0.6, 1))
        return drums

    def _risers(self, predrop: bool) -> Pattern:
        """Announce the coming chorus on the two last bars; drop an impact on its first
        beat.
        """
        remaining = self.section_len - self.section_bar  # bars left including this one
        going_to_chorus = self._going_to_chorus()
        p: Pattern = []
        if going_to_chorus and remaining == 2:
            p.append(Note(0, RISER_UP, 0.9, 32))
        if predrop:
            p.append(Note(0, RISER_REV, 1.0, 16))
            if self.rng.random() < 0.35:
                p.append(Note(8, RISER_SCREAM, 0.7, 8))
        if (
            remaining == 1
            and not going_to_chorus
            and self.section not in (Section.TRANSITION, Section.OUTRO)
            and self.rng.random() < 0.5
        ):
            p.append(Note(8, RISER_SHORT, 0.8, 8))
        if self.section == Section.CHORUS and (
            self.section_bar == 0 or (self.mid_drop and self.section_bar == 8)
        ):
            p.append(Note(0, RISER_IMPACT, 1.0, 8))
        return p

    def _bass(self, r: np.random.Generator, chord: Chord, predrop: bool) -> Pattern:
        """Regenerate on the chord each bar; every second bar apply a light mutation
        (eighties moods keep the line as a strict ostinato instead).
        """
        p = gen_bass(r, chord, self.bass_style, straight=self.mood.straight)
        if predrop:
            return cut_after(p, 12)
        if not self.mood.straight and self.section_bar % 2 == 1:
            root = chord.bass_note()
            p = mutate(r, p, 0.2, [root, root + 12, root + 7, root + 1, root + 6])
        return p

    def _lead(self, r: np.random.Generator, chord: Chord, active: bool) -> Pattern:
        """The track's theme: question / answer motifs alternate bar by bar in verse and
        chorus (octave up in the second half of a chorus); the counter-melody plays in
        breaks and as a secondary line late in a verse.
        """
        if not active:
            return []
        lo = 55 if self.mood.pad_octave <= 3 else 60
        scale = self.harmony.scale_notes(lo, lo + 19)
        sec, bar = self.section, self.section_bar
        if sec == Section.BREAK:
            if r.random() < 0.8:
                return render_motif(r, self.theme.counter, chord, scale, vary=0.1, vel=0.7)
            return []
        p = max(self.mood.lead_prob, 0.7) if sec == Section.CHORUS else self.mood.lead_prob * 0.5
        if sec == Section.VERSE and bar >= self.section_len // 2 and r.random() < 0.3:
            return render_motif(r, self.theme.counter, chord, scale, vary=0.15, vel=0.7)
        if r.random() >= p:
            return []
        motif = self.theme.answer if bar % 2 else self.theme.question
        octave = (
            12
            if (sec == Section.CHORUS and bar >= self.section_len // 2 and self.section_len >= 16)
            else 0
        )
        vary = 0.1 if sec == Section.CHORUS else 0.25
        if self.mood.straight:  # eighties: the hook comes back the same, bar after bar
            vary *= 0.4
        return render_motif(r, motif, chord, scale, octave=octave, vary=vary)

    def _lead2(self, chord: Chord, lead: Pattern) -> Pattern:
        """Second lead: the same phrase harmonised under the theme (a diatonic third or
        sixth below, an octave below when the lead is up high), so it doubles the hook
        without competing with it. Silent whenever the lead is silent.
        """
        if not lead:
            return []
        lo = 55 if self.mood.pad_octave <= 3 else 60
        scale = self.harmony.scale_notes(lo - 12, lo + 19)
        return harmonize(lead, scale, self.harmony_degrees)

    def _tweaks(self, predrop: bool) -> dict[str, dict[str, float]]:
        """Section gestures + per-bar sweeps: the filter opens over the 4 bars before a
        chorus (build-up), closes down in a break, and rises through the intro.
        """
        out = {k: dict(v) for k, v in self.gestures.items()}

        def mul(layer: str, path: str, factor: float) -> None:
            """Mul."""
            out.setdefault(layer, {})
            out[layer][path] = out[layer].get(path, 1.0) * factor

        remaining = self.section_len - self.section_bar
        if self._going_to_chorus() and remaining <= 4:
            k = (4 - remaining) / 3.0  # 0 .. 1 over the last 4 bars
            for layer in ("pad", "arp"):
                mul(layer, "filter.cutoff", 0.55 + 1.05 * k)
            mul("pad", "filter.resonance", 1.0 + 1.2 * k)
        if self.section == Section.BREAK:
            mul("pad", "filter.cutoff", 0.7)
            mul("bass", "filter.cutoff", 0.7)
        if self.section == Section.INTRO:
            k = self.section_bar / max(1, self.section_len - 1)
            mul("pad", "filter.cutoff", 0.5 + 0.5 * k)
        if predrop:
            mul("bass", "filter.cutoff", 0.5)
        return {k: {p: round(f, 4) for p, f in v.items()} for k, v in out.items() if v}

    def _silence(self) -> BarPlan:
        """Silence."""
        plan = BarPlan(
            self.bar,
            Section.OUTRO,
            0,
            self.progression[0],
            {layer: [] for layer in LAYERS},
            {layer: 0.0 for layer in LAYERS},
            fade=0.0,
            finished=True,
            key=self.harmony.key_name,
            track=self.track,
        )
        self.bar += 1
        return plan

    def next_bar(self) -> BarPlan:
        """Next bar."""
        if self.finished or (self.total_bars is not None and self.bar >= self.total_bars):
            self.finished = True
            return self._silence()
        if self.section_bar >= self.section_len or (
            self._final_outro_due() and self.section != Section.OUTRO
        ):
            self._start_section()
        r = self.rng
        chord = self.progression[self.section_bar % len(self.progression)]
        last = self.section_bar == self.section_len - 1
        first = self.section_bar == 0
        density = self._density()
        predrop = self._is_predrop()
        gains = self._gains(predrop)
        lead = self._lead(r, chord, gains["lead"] > 0)
        patterns = {
            "drums": self._drums(r, predrop, density),
            "bass": self._bass(r, chord, predrop),
            "arp": gen_arp(r, chord, self.arp_mode) if self.arp_on else [],
            "pad": gen_pad(chord, self.mood.pad_octave),
            "lead": lead,
            "lead2": self._lead2(chord, lead),
            "ambient": gen_ambient(chord),
            "riser": self._risers(predrop) if self.section != Section.TRANSITION else [],
        }
        tries = 0
        while (
            not self.mood.straight  # a repeated bar is the point of an eighties ostinato
            and self.prev_patterns is not None
            and patterns == self.prev_patterns
            and tries < 4
        ):
            root = chord.bass_note()
            patterns["bass"] = mutate(r, patterns["bass"], 0.5, [root, root + 12, root + 7])
            tries += 1
        self.prev_patterns = patterns
        final = self.total_bars is not None and self.bar >= self.total_bars - self.section_len
        fade = (
            1.0 - self.section_bar / self.section_len
            if self.section == Section.OUTRO and final
            else 1.0
        )
        plan = BarPlan(
            self.bar,
            self.section,
            self.section_bar,
            chord,
            patterns,
            gains,
            fill=last,
            fade=fade,
            key=self.harmony.key_name,
            fx=self.fx,
            bpm=self.bar_bpm,
            mood=self.mood.name if self.mood_changed else None,
            patches=self.section_patches if first else None,
            drop=predrop,
            track=self.track,
            track_bar=self.track_bar,
            track_bars=self.track_bars,
            tweaks=self._tweaks(predrop) or None,
            master_color=self.section_color if first else None,
        )
        self.bar_bpm, self.mood_changed = None, False
        self.bar += 1
        self.section_bar += 1
        self.track_bar += 1
        return plan
