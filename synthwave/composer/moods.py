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
DARK_POOLS = {"bass": ["bass_dark", "bass_industrial", "bass_sub", "bass_growl", "bass_reese"],
              "lead": ["lead_dark", "lead_industrial", "lead_scream"]}
BRIGHT_POOLS = {"bass": ["bass_moog", "bass_sub", "bass_pulse", "bass_reese"],
                "lead": ["lead_saw", "lead_pulse", "lead_industrial"]}
DARK_BASS = {"eighths": 2, "riff": 3, "sixteenths": 2, "syncopated": 1, "octaves": 1}
BRIGHT_BASS = {"eighths": 3, "octaves": 2, "syncopated": 1, "walk": 1, "sixteenths": 1}

CYBER_POOLS = {"bass": ["bass_industrial", "bass_growl", "bass_reese", "bass_industrial"],
               "lead": ["lead_industrial", "lead_scream", "lead_dark"]}
CYBER_BASS = {"sixteenths": 3, "riff": 3, "eighths": 2, "octaves": 1}
DRIVE_POOLS = {"bass": ["bass_moog", "bass_sub", "bass_pulse", "bass_reese"],
               "lead": ["lead_saw", "lead_pulse", "lead_scream"]}
DRIVE_BASS = {"eighths": 3, "sixteenths": 3, "octaves": 2, "syncopated": 1}
CHILL_BASS = {"eighths": 2, "walk": 3, "syncopated": 2, "octaves": 1}

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
    "cyberpunk": Mood("cyberpunk", 124, 0.9, 0.9, 0.5, 0.8, 0.0,
                      {"i-bII-i-bII": 2, "i-iv-bII-i": 2, "i-VI-VII-i": 2, "i-VII-VI-VII": 2,
                       "i-v-VI-iv": 2, "i-iv-i-VI": 1, "i-VI-iv-VII": 2},
                      scale="minor", sevenths=False, halftime=False, pad_octave=3,
                      patches=DARK_PATCHES, pools=CYBER_POOLS, bass_styles=CYBER_BASS,
                      bpm_range=(118.0, 132.0)),
    "horror": Mood("horror", 72, 0.2, 0.3, 0.15, 0.5, 0.0,
                   {"i-bII-bv-i": 3, "i-biii-bII-i": 2, "i-bII-bVI-bv": 2, "i-bII-i-bII": 2},
                   scale="locrian", sevenths=False, halftime=True, pad_octave=3,
                   patches=DARK_PATCHES, pools=DARK_POOLS, bass_styles=DARK_BASS,
                   bpm_range=(66.0, 80.0)),
    "desert": Mood("desert", 96, 0.45, 0.6, 0.35, 0.7, 0.0,
                   {"I-bII-I-bvii": 3, "I-iv-bII-I": 3, "I-bVI-bvii-I": 2, "i-bII-i-bII": 1},
                   scale="phrygian_dominant", sevenths=False, halftime=True, pad_octave=3,
                   patches=DARK_PATCHES, pools=DARK_POOLS, bass_styles=DARK_BASS,
                   bpm_range=(88.0, 104.0)),
    "chill": Mood("chill", 90, 0.4, 0.6, 0.3, 0.9, 0.0,
                  {"i-IV-i-IV": 3, "i-bVII-IV-i": 2, "i-ii-bIII-ii": 2, "i-IV-bVII-i": 2,
                   "i-ii-IV-i": 1},
                  scale="dorian", sevenths=True, halftime=False, pad_octave=4,
                  pools=BRIGHT_POOLS, bass_styles=CHILL_BASS, bpm_range=(84.0, 96.0)),
    "retro": Mood("retro", 110, 0.6, 0.8, 0.45, 1.1, 0.5,
                  {"I-V-vi-IV": 3, "vi-IV-I-V": 3, "I-vi-IV-V": 2, "IV-I-V-vi": 1,
                   "I-bVII-IV-I": 2, "I-IV-bVII-IV": 2, "I-v-bVII-IV": 1},
                  scale="mixolydian", sevenths=True, halftime=False, pad_octave=4,
                  pools=BRIGHT_POOLS, bass_styles=BRIGHT_BASS, bpm_range=(104.0, 118.0)),
    "drive": Mood("drive", 128, 1.0, 1.0, 0.5, 1.2, 0.0,
                  {"i-VI-III-VII": 3, "i-VII-VI-VII": 3, "i-VI-VII-i": 2, "VI-VII-i-i": 2,
                   "i-VII-III-VI": 2, "VI-i-VII-III": 1, "i-III-iv-VI": 1},
                  scale="minor", sevenths=True, halftime=False, pad_octave=4,
                  pools=DRIVE_POOLS, bass_styles=DRIVE_BASS, bpm_range=(122.0, 136.0)),
}
