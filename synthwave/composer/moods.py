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
    scale: str = "minor"    # minor | major | phrygian | harmonic_minor
    sevenths: bool = True   # stack a 7th on every chord
    halftime: bool = False  # snare on beat 3 only
    pad_octave: int = 4
    patches: dict[str, str] = field(default_factory=dict)   # layer -> patch name
    pools: dict[str, list[str]] = field(default_factory=dict)  # layer -> patches per section
    bass_styles: dict[str, float] = field(default_factory=dict)  # style -> weight
    bpm_range: tuple[float, float] = (100.0, 118.0)  # tempo drawn here at start / transitions


DARK_PATCHES = {"drums": "drums_dark", "bass": "bass_dark", "arp": "arp_dark",
                "pad": "pad_dark", "lead": "lead_dark", "ambient": "ambient_dark"}
DARK_POOLS = {"bass": ["bass_dark", "bass_industrial", "bass_reese", "bass_industrial"],
              "lead": ["lead_dark", "lead_industrial"]}
BRIGHT_POOLS = {"bass": ["bass_moog", "bass_reese", "bass_acid", "bass_moog"],
                "lead": ["lead_saw", "lead_saw", "lead_industrial"]}
DARK_BASS = {"eighths": 2, "riff": 3, "sixteenths": 2, "syncopated": 1, "octaves": 1}
BRIGHT_BASS = {"eighths": 3, "octaves": 2, "syncopated": 1, "walk": 1, "sixteenths": 1}

MOODS: dict[str, Mood] = {
    "dark": Mood("dark", 92, 0.35, 0.45, 0.2, 0.6, 0.0,
                 {"i-bII-i-bII": 3, "i-iv-bII-i": 3, "i-bVI-bII-i": 2, "i-bvii-bVI-bII": 2,
                  "i-i-bVI-bII": 2, "i-iv-i-bII": 2},
                 scale="phrygian", sevenths=False, halftime=True, pad_octave=3,
                 patches=DARK_PATCHES, pools=DARK_POOLS, bass_styles=DARK_BASS,
                 bpm_range=(82.0, 100.0)),
    "noir": Mood("noir", 96, 0.4, 0.5, 0.3, 0.7, 0.0,
                 {"i-VI-V-i": 3, "i-iv-V-i": 3, "i-iv-VI-V": 2, "i-i-VI-V": 2, "i-VI-III-VII": 1},
                 scale="harmonic_minor", sevenths=False, halftime=True, pad_octave=3,
                 patches=DARK_PATCHES, pools=DARK_POOLS, bass_styles=DARK_BASS,
                 bpm_range=(86.0, 104.0)),
    "dreamy": Mood("dreamy", 108, 0.5, 0.75, 0.4, 1.0, 0.35,
                   {"i-VI-III-VII": 3, "i-VII-VI-VII": 2, "i-iv-VI-V": 1, "i-VI-VII-i": 2,
                    "VI-VII-i-i": 2, "i-III-VII-VI": 2, "iv-VI-i-VII": 1},
                   pools=BRIGHT_POOLS, bass_styles=BRIGHT_BASS, bpm_range=(98.0, 114.0)),
    "outrun": Mood("outrun", 118, 0.85, 0.95, 0.55, 1.2, 0.1,
                   {"i-VI-III-VII": 3, "i-VII-VI-VII": 3, "i-iv-VI-V": 2, "i-VI-VII-i": 2,
                    "VI-VII-i-i": 2, "i-III-VII-VI": 1, "iv-VI-i-VII": 1},
                   pools=BRIGHT_POOLS, bass_styles=BRIGHT_BASS, bpm_range=(110.0, 128.0)),
}
