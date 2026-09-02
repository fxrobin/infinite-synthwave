"""Per-layer 16-step pattern generators and a bounded mutation operator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .harmony import Chord

STEPS = 16
KICK, SNARE, CLAP, HAT_C, TOM_L, HAT_O, TOM_M, CRASH = 36, 38, 39, 42, 45, 46, 47, 49


@dataclass(frozen=True)
class Note:
    step: int
    note: int
    vel: float
    length: int = 1


Pattern = list[Note]


def _sorted(p: Pattern) -> Pattern:
    return sorted(p, key=lambda n: (n.step, n.note))


def gen_drums(rng: np.random.Generator, density: float, fill: bool = False,
              snare: bool = True, crash: bool = False) -> Pattern:
    p: Pattern = [Note(s, KICK, 1.0) for s in (0, 4, 8, 12)]
    if rng.random() < density * 0.5:
        p.append(Note(int(rng.choice([10, 14, 7])), KICK, 0.8))
    if snare:
        p += [Note(4, SNARE, 1.0), Note(12, SNARE, 1.0)]
        if rng.random() < density * 0.4:
            p += [Note(4, CLAP, 0.7), Note(12, CLAP, 0.7)]
    for s in range(0, STEPS, 2):
        p.append(Note(s, HAT_C, 0.8 if s % 4 == 0 else 0.55))
    for s in range(1, STEPS, 2):
        if rng.random() < density * 0.7:
            p.append(Note(s, HAT_C, 0.4))
    if rng.random() < density * 0.6:
        p.append(Note(14, HAT_O, 0.6))
    if fill:
        p = [n for n in p if n.step < 12]
        roll = rng.choice([SNARE, TOM_M, TOM_L], size=4)
        p += [Note(12 + i, int(roll[i]), 0.6 + 0.1 * i) for i in range(4)]
        if rng.random() < 0.5:
            p += [Note(13, SNARE, 0.5), Note(15, SNARE, 0.9)]
    if crash:
        p.append(Note(0, CRASH, 0.8))
    return _sorted(p)


def gen_bass(rng: np.random.Generator, chord: Chord, style: str) -> Pattern:
    root = chord.bass_note()
    fifth = root + 7
    if style == "eighths":
        p = [Note(s, root, 1.0 if s % 4 == 0 else 0.8, 2) for s in range(0, STEPS, 2)]
    elif style == "octaves":
        p = [Note(s, root + (12 if (s // 2) % 2 else 0), 0.9, 1) for s in range(0, STEPS, 2)]
    else:  # syncopated
        steps = (0, 3, 6, 8, 11, 14)
        p = [Note(s, root, 0.9, 2) for s in steps]
        if rng.random() < 0.5:
            p[-1] = Note(14, fifth, 0.8, 2)
    if rng.random() < 0.3:
        p.append(Note(15, root + 12, 0.6, 1))
    return _sorted(p)


def gen_arp(rng: np.random.Generator, chord: Chord, mode: str, octaves: int = 2) -> Pattern:
    tones = [n for o in range(octaves) for n in chord.notes(4 + o)]
    if mode == "up":
        seq = [tones[i % len(tones)] for i in range(STEPS)]
    elif mode == "updown":
        cyc = tones + tones[-2:0:-1]
        seq = [cyc[i % len(cyc)] for i in range(STEPS)]
    else:
        seq = [int(rng.choice(tones)) for _ in range(STEPS)]
    return [Note(s, seq[s], 0.85 if s % 4 == 0 else 0.7, 1) for s in range(STEPS)]


def gen_pad(chord: Chord) -> Pattern:
    notes = chord.notes(4)
    if chord.root_pc >= 6:  # keep voicing low: drop the root an octave
        notes = [notes[0] - 12] + notes[1:]
    return [Note(0, n, 0.7, STEPS) for n in notes]


def gen_ambient(chord: Chord) -> Pattern:
    root = chord.notes(3)[0]
    return [Note(0, root, 0.6, STEPS), Note(0, root + 7, 0.4, STEPS)]


def gen_lead(rng: np.random.Generator, chord: Chord, scale_notes: list[int],
             density: float) -> Pattern:
    if not scale_notes:
        return []
    chord_pcs = {(chord.root_pc + i) % 12 for i in chord.intervals}
    grid = [0, 2, 3, 4, 6, 8, 10, 11, 12, 14]
    count = max(1, min(len(grid), int(round(2 + density * 4))))
    steps = sorted(int(s) for s in rng.choice(grid, size=count, replace=False))
    p: Pattern = []
    prev = int(rng.choice(scale_notes))
    for i, s in enumerate(steps):
        near = [n for n in scale_notes if abs(n - prev) <= 5] or scale_notes
        strong = [n for n in near if n % 12 in chord_pcs]
        note = int(rng.choice(strong if (s % 4 == 0 and strong) else near))
        nxt = steps[i + 1] if i + 1 < len(steps) else STEPS
        p.append(Note(s, note, 0.8, max(1, min(4, nxt - s))))
        prev = note
    return p


def mutate(rng: np.random.Generator, pattern: Pattern, rate: float,
           allowed_notes: list[int]) -> Pattern:
    out: Pattern = []
    taken = {n.step for n in pattern}
    for n in pattern:
        r = rng.random()
        if r < rate * 0.3:
            continue  # drop
        if r < rate * 0.6:
            s = int(np.clip(n.step + rng.choice([-1, 1]), 0, STEPS - 1))
            if s not in taken:
                taken.add(s)
                out.append(Note(s, n.note, n.vel, n.length))
                continue
        if r < rate and allowed_notes:
            out.append(Note(n.step, int(rng.choice(allowed_notes)), n.vel, n.length))
            continue
        out.append(n)
    if allowed_notes and rng.random() < rate * 0.5:
        free = [s for s in range(STEPS) if s not in taken]
        if free:
            out.append(Note(int(rng.choice(free)), int(rng.choice(allowed_notes)), 0.6, 1))
    return _sorted(out)
