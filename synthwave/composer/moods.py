from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mood:
    name: str
    bpm: float
    drum_density: float     # 0..1 hats/kick extras
    arp_prob: float         # probability the arp layer plays in a section
    lead_prob: float        # probability of a lead phrase per chorus bar
    brightness: float       # multiplier applied to filter cutoffs
    major_prob: float       # probability of a major key
    progressions: dict[str, float] = field(default_factory=dict)


MOODS: dict[str, Mood] = {
    "dark": Mood("dark", 100, 0.5, 0.55, 0.25, 0.7, 0.0,
                 {"i-VI-III-VII": 2, "i-VII-VI-VII": 3, "i-iv-VI-V": 2, "i-VI-VII-i": 2,
                  "VI-VII-i-i": 1, "i-III-VII-VI": 1, "iv-VI-i-VII": 2}),
    "dreamy": Mood("dreamy", 108, 0.5, 0.75, 0.4, 1.0, 0.35,
                   {"i-VI-III-VII": 3, "i-VII-VI-VII": 2, "i-iv-VI-V": 1, "i-VI-VII-i": 2,
                    "VI-VII-i-i": 2, "i-III-VII-VI": 2, "iv-VI-i-VII": 1}),
    "outrun": Mood("outrun", 118, 0.85, 0.95, 0.55, 1.2, 0.1,
                   {"i-VI-III-VII": 3, "i-VII-VI-VII": 3, "i-iv-VI-V": 2, "i-VI-VII-i": 2,
                    "VI-VII-i-i": 2, "i-III-VII-VI": 1, "iv-VI-i-VII": 1}),
}
