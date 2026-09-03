"""Per-layer 16-step pattern generators and a bounded mutation operator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .harmony import Chord

STEPS = 16
KICK, SNARE, CLAP, HAT_C, TOM_L, HAT_O, TOM_M, CRASH = 36, 38, 39, 42, 45, 46, 47, 49
SNAP, RIDE, SHAKER, TICK, CROLL = 40, 51, 70, 44, 57


@dataclass(frozen=True)
class Note:
    """Note."""
    step: int
    note: int
    vel: float
    length: int = 1


Pattern = list[Note]


def _sorted(p: Pattern) -> Pattern:
    """Sorted."""
    return sorted(p, key=lambda n: (n.step, n.note))


def gen_drums(  # noqa: PLR0913 - drum pattern needs many flags (bundled from arranger)
    rng: np.random.Generator,
    density: float,
    fill: bool = False,
    snare: bool = True,
    crash: bool = False,
    halftime: bool = False,
    strong: bool = False,
    snap: bool = False,
    ride: bool = False,
    shaker: bool = False,
    tick: bool = False,
) -> Pattern:
    """One bar of drums. `snap`: finger snaps take the backbeat (layered with the snare in a
    strong chorus); `ride`: the ride cymbal replaces the closed hats on the 8ths; `shaker`:
    16th-note shaker underneath."""
    p = _gen_drums(rng, density, fill, snare, crash, halftime, strong)
    if snap and snare:
        keep = strong
        p = [n for n in p if n.note != SNARE or keep]
        p += [Note(s, SNAP, 0.9 if not strong else 0.7) for s in ((8,) if halftime else (4, 12))]
    if ride:
        p = [Note(n.step, RIDE, n.vel) if n.note == HAT_C and n.step % 2 == 0 else n for n in p]
    if tick:  # dry "tic tic" closed hat on every 8th, replacing the washy closed hats there
        p = [n for n in p if not (n.note == HAT_C and n.step % 2 == 0)]
        p += [Note(s, TICK, 0.8 if s % 4 == 0 else 0.55) for s in range(0, STEPS, 2)]
    if shaker:
        p += [Note(s, SHAKER, 0.55 if s % 4 == 2 else 0.3) for s in range(STEPS)]
    return _sorted(p)


def _gen_drums(
    rng: np.random.Generator,
    density: float,
    fill: bool,
    snare: bool,
    crash: bool,
    halftime: bool,
    strong: bool,
) -> Pattern:
    """Gen drums."""
    if strong:  # chorus: four on the floor, snare + clap on 2 and 4, driving hats
        p: Pattern = [Note(s, KICK, 1.0) for s in (0, 4, 8, 12)]
        p.append(Note(int(rng.choice([10, 14])), KICK, 0.75))
        p += [Note(4, SNARE, 1.0), Note(12, SNARE, 1.0), Note(4, CLAP, 0.8), Note(12, CLAP, 0.8)]
        p += [Note(s, HAT_C, 0.9 if s % 4 == 0 else 0.6) for s in range(0, STEPS, 2)]
        p += [Note(s, HAT_C, 0.4) for s in range(1, STEPS, 2) if rng.random() < 0.5 + density * 0.5]
        for s in (6, 14) if rng.random() < 0.4 else (14,):
            p.append(Note(s, HAT_O, 0.65))
        if fill:
            p = [n for n in p if n.step < 12]
            p += [
                Note(12 + i, int(rng.choice([SNARE, TOM_M, TOM_L])), 0.6 + 0.1 * i)
                for i in range(4)
            ]
        if crash:
            p.append(Note(0, CRASH, 0.8))
        return _sorted(p)
    if halftime:  # slow, heavy: kick on 1 (+ a syncopated one), snare on 3 only
        p: Pattern = [Note(0, KICK, 1.0)]
        p.append(Note(int(rng.choice([6, 10, 7, 11])), KICK, 0.85))
        if rng.random() < density:
            p.append(Note(int(rng.choice([13, 14])), KICK, 0.7))
        if snare:
            p.append(Note(8, SNARE, 1.0))
            if rng.random() < 0.5:
                p.append(Note(8, CLAP, 0.6))
            if rng.random() < density * 0.6:
                p.append(Note(int(rng.choice([14, 15, 11])), int(rng.choice([TOM_L, TOM_M])), 0.6))
    else:
        p = [Note(s, KICK, 1.0) for s in (0, 4, 8, 12)]
        if rng.random() < density * 0.5:
            p.append(Note(int(rng.choice([10, 14, 7])), KICK, 0.8))
        if snare:
            p += [Note(4, SNARE, 1.0), Note(12, SNARE, 1.0)]
            if rng.random() < density * 0.4:
                p += [Note(4, CLAP, 0.7), Note(12, CLAP, 0.7)]
    for s in range(0, STEPS, 2):
        p.append(Note(s, HAT_C, 0.85 if s % 4 == 0 else 0.6))
    for s in range(1, STEPS, 2):
        if rng.random() < density * 0.7:
            p.append(Note(s, HAT_C, 0.45))
    if rng.random() < density * 0.6:
        p.append(Note(int(rng.choice([14, 6, 10])), HAT_O, 0.6))
    if fill:
        p = [n for n in p if n.step < 12]
        roll = rng.choice([SNARE, TOM_M, TOM_L], size=4)
        p += [Note(12 + i, int(roll[i]), 0.6 + 0.1 * i) for i in range(4)]
        if rng.random() < 0.5:
            p += [Note(13, SNARE, 0.5), Note(15, SNARE, 0.9)]
    if crash:
        p.append(Note(0, CRASH, 0.8))
    return _sorted(p)


_ROLL_VOICES = {
    "snare": (SNARE, SNARE, SNARE, SNARE),
    "toms_down": (SNARE, TOM_M, TOM_L, TOM_L),
    "toms_up": (TOM_L, TOM_M, SNARE, SNARE),
    "alternate": (SNARE, TOM_M, SNARE, TOM_L),
}


def gen_roll(
    rng: np.random.Generator, start: int, length: int, vel: tuple[float, float] = (0.5, 0.9)
) -> Pattern:
    """16th-note drum roll of `length` steps from `start`, crescendo, on snare and/or toms."""
    voices = _ROLL_VOICES[str(rng.choice(list(_ROLL_VOICES)))]
    vels = np.linspace(vel[0], vel[1], length)
    return [
        Note(start + i, voices[(i * 4) // length], float(round(vels[i], 3))) for i in range(length)
    ]


def add_roll(rng: np.random.Generator, pattern: Pattern, start: int, length: int) -> Pattern:
    """Overlay a roll on [start, start+length): strips snare/hats/toms there, keeps kick + crash."""
    window = set(range(start, min(STEPS, start + length)))
    kept = [n for n in pattern if n.step not in window or n.note in (KICK, CRASH)]
    return _sorted(kept + gen_roll(rng, start, len(window)))


def drum_layer(pattern: Pattern, level: int) -> Pattern:
    """Build-up layers of a groove: 0 = kick/snare/crash, 1 = + 8th hats and clap, 2 = all."""
    if level >= 2:
        return list(pattern)
    core = (KICK, SNARE, CRASH, SNAP, CROLL)
    if level <= 0:
        return [n for n in pattern if n.note in core]
    return [
        n
        for n in pattern
        if n.note in core or n.note == CLAP or (n.note in (HAT_C, RIDE, TICK) and n.step % 2 == 0)
    ]


def gen_predrop(rng: np.random.Generator, pattern: Pattern, cut: int = 12) -> Pattern:
    """Bar before a drop: kick on the 1, snare roll crescendo up to `cut`, then silence."""
    head = [n for n in drum_layer(pattern, 0) if n.step < 4 and n.note == KICK]
    return _sorted(head + gen_roll(rng, 4, cut - 4, (0.35, 1.0)) + [Note(0, CROLL, 0.9, 16)])


def cut_after(pattern: Pattern, step: int) -> Pattern:
    """Drop every note starting at or after `step` (the silence before a drop)."""
    return [n for n in pattern if n.step < step]


def gen_bass(rng: np.random.Generator, chord: Chord, style: str) -> Pattern:
    """Gen bass."""
    root = chord.bass_note()
    fifth = root + 7
    if style == "eighths":
        p = [Note(s, root, 1.0 if s % 4 == 0 else 0.8, 2) for s in range(0, STEPS, 2)]
    elif style == "octaves":
        p = [Note(s, root + (12 if (s // 2) % 2 else 0), 0.9, 1) for s in range(0, STEPS, 2)]
    elif style == "sixteenths":  # pumping 16ths, octave pops on the off-beats
        p = [
            Note(
                s,
                root + (12 if s % 4 == 3 and rng.random() < 0.5 else 0),
                0.95 if s % 4 == 0 else 0.6,
                1,
            )
            for s in range(STEPS)
        ]
    elif style == "walk":  # root, fifth, octave, seventh-ish
        walk = [root, root, fifth, root + 12, root, root + 10, fifth, root]
        p = [Note(s, walk[s // 2], 0.9 if s % 4 == 0 else 0.75, 2) for s in range(0, STEPS, 2)]
    elif style == "riff":  # chromatic menace: b2 and tritone passing tones
        b2, tritone = root + 1, root + 6
        p = [
            Note(0, root, 1.0, 2),
            Note(2, root, 0.8, 1),
            Note(3, root, 0.7, 1),
            Note(6, int(rng.choice([b2, tritone])), 0.85, 2),
            Note(8, root, 1.0, 2),
            Note(10, root, 0.8, 1),
            Note(11, root, 0.7, 1),
            Note(14, int(rng.choice([b2, tritone, fifth])), 0.85, 1),
            Note(15, root, 0.7, 1),
        ]
    else:  # syncopated
        steps = (0, 3, 6, 8, 11, 14)
        p = [Note(s, root, 0.9, 2) for s in steps]
        if rng.random() < 0.5:
            p[-1] = Note(14, fifth, 0.8, 2)
    if style not in ("sixteenths", "riff") and rng.random() < 0.3:
        p.append(Note(15, root + 12, 0.6, 1))
    return _sorted(p)


def gen_arp(rng: np.random.Generator, chord: Chord, mode: str, octaves: int = 2) -> Pattern:
    """Gen arp."""
    tones = [n for o in range(octaves) for n in chord.notes(4 + o)]
    if mode == "up":
        seq = [tones[i % len(tones)] for i in range(STEPS)]
    elif mode == "updown":
        cyc = tones + tones[-2:0:-1]
        seq = [cyc[i % len(cyc)] for i in range(STEPS)]
    else:
        seq = [int(rng.choice(tones)) for _ in range(STEPS)]
    return [Note(s, seq[s], 0.85 if s % 4 == 0 else 0.7, 1) for s in range(STEPS)]


def gen_pad(chord: Chord, octave: int = 4) -> Pattern:
    """Gen pad."""
    notes = chord.notes(octave)
    if chord.root_pc >= 6:  # keep voicing low: drop the root an octave
        notes = [notes[0] - 12] + notes[1:]
    if len(notes) == 3:  # triads: add the octave for width
        notes.append(notes[0] + 12)
    return [Note(0, n, 0.7, STEPS) for n in notes]


def gen_ambient(chord: Chord) -> Pattern:
    """Gen ambient."""
    root = chord.notes(3)[0]
    return [Note(0, root, 0.6, STEPS), Note(0, root + 7, 0.4, STEPS)]


def gen_lead(
    rng: np.random.Generator, chord: Chord, scale_notes: list[int], density: float
) -> Pattern:
    """Gen lead."""
    if not scale_notes:
        return []
    chord_pcs = {(chord.root_pc + i) % 12 for i in chord.intervals}
    grid = [0, 2, 4, 6, 8, 10, 12, 14]
    count = max(1, min(len(grid), int(round(2 + density * 3))))
    steps = sorted(int(s) for s in rng.choice(grid, size=count, replace=False))
    p: Pattern = []
    prev = int(rng.choice(scale_notes))
    for i, s in enumerate(steps):
        near = [n for n in scale_notes if abs(n - prev) <= 5] or scale_notes
        strong = [n for n in near if n % 12 in chord_pcs]
        note = int(rng.choice(strong if (s % 4 == 0 and strong) else near))
        nxt = steps[i + 1] if i + 1 < len(steps) else STEPS
        p.append(Note(s, note, 0.8, max(2, min(8, nxt - s))))  # long, legato phrases
        prev = note
    return p


@dataclass(frozen=True)
class Motif:
    """A melodic idea as rhythm + contour: (step, scale-step offset from the chord root, length).

    Offsets are diatonic (in scale steps), so rendering the same motif on another chord gives a
    real sequence: same rhythm, same shape, transposed within the key."""

    notes: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class Theme:
    """The melodic material of one track: a question motif, its answer (same rhythm, contour
    resolving to the root) and a counter-melody (inverted contour, longer notes)."""

    question: Motif
    answer: Motif
    counter: Motif


_CHORD_OFFSETS = (0, 2, 4)  # root, third, fifth as scale-step offsets


def gen_theme(rng: np.random.Generator, density: float) -> Theme:
    """Gen theme."""
    grid = [0, 2, 4, 6, 8, 10, 12, 14]
    count = max(2, min(len(grid), int(round(2 + density * 3))))
    steps = sorted(int(s) for s in rng.choice(grid, size=count, replace=False))
    if steps[0] != 0:
        steps[0] = 0  # motifs start on the downbeat
    offsets: list[int] = []
    cur = int(rng.choice(_CHORD_OFFSETS))
    for s in steps:
        if s % 8 == 0:  # strong beats: chord tones
            cur = int(rng.choice([o for o in _CHORD_OFFSETS if abs(o - cur) <= 4] or [0]))
        else:  # steps of at most a third
            cur = int(np.clip(cur + int(rng.choice([-2, -1, -1, 1, 1, 2])), -4, 7))
        offsets.append(cur)

    def lengths(st: list[int]) -> list[int]:
        """Lengths."""
        return [
            max(2, min(8, (st[i + 1] if i + 1 < len(st) else STEPS) - st[i]))
            for i in range(len(st))
        ]

    q = Motif(tuple(zip(steps, offsets, lengths(steps), strict=True)))
    ans = list(offsets)
    ans[-1] = 0  # the answer resolves to the root
    for i in range(1, len(ans) - 1):
        if rng.random() < 0.35:
            ans[i] += int(rng.choice([-1, 1]))
    a = Motif(tuple(zip(steps, ans, lengths(steps), strict=True)))
    c_steps = [s for s in steps if s % 4 == 0] or [0]
    c_offs = [int(np.clip(-o, -5, 5)) for o, s in zip(offsets, steps, strict=True) if s in c_steps]
    c = Motif(tuple(zip(c_steps, c_offs, lengths(c_steps), strict=True)))
    return Theme(q, a, c)


def render_motif(
    rng: np.random.Generator,
    motif: Motif,
    chord: Chord,
    scale_notes: list[int],
    octave: int = 0,
    vary: float = 0.0,
    vel: float = 0.8,
) -> Pattern:
    """Play a motif on a chord: offsets are counted in scale steps from the chord root's
    position in `scale_notes`. `vary` = chance of an ornament (neighbour, dropped, added note)."""
    if not scale_notes:
        return []
    root_pc = chord.root_pc
    anchors = [i for i, n in enumerate(scale_notes) if n % 12 == root_pc]
    mid = len(scale_notes) // 2
    anchor = min(anchors, key=lambda i: abs(i - mid)) if anchors else mid
    p: Pattern = []
    for i, (step, off, length) in enumerate(motif.notes):
        if vary and i > 0 and rng.random() < vary * 0.3:
            continue  # dropped note
        if vary and rng.random() < vary:
            off += int(rng.choice([-1, 1]))  # neighbour tone
        idx = int(np.clip(anchor + off, 0, len(scale_notes) - 1))
        p.append(Note(step, scale_notes[idx] + octave, vel, length))
    if vary and rng.random() < vary * 0.5:
        taken = {n.step for n in p}
        free = [s for s in (1, 3, 5, 7, 9, 11, 13, 15) if s not in taken]
        if free and p:
            s = int(rng.choice(free))
            prev = max((n for n in p if n.step < s), key=lambda n: n.step, default=p[0])
            near = [n for n in scale_notes if abs(n + octave - prev.note) <= 2] or scale_notes
            p.append(Note(s, int(rng.choice(near)) + octave, vel * 0.8, 1))
    return _sorted(p)


def mutate(
    rng: np.random.Generator, pattern: Pattern, rate: float, allowed_notes: list[int]
) -> Pattern:
    """Mutate."""
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
