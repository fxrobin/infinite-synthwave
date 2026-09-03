import numpy as np

from synthwave.composer.harmony import Chord
from synthwave.composer.patterns import (
    RIDE,
    SHAKER,
    SNAP,
    STEPS,
    TICK,
    Note,
    add_roll,
    drum_layer,
    gen_ambient,
    gen_arp,
    gen_bass,
    gen_drums,
    gen_lead,
    gen_pad,
    gen_theme,
    harmonize,
    mutate,
    render_motif,
)

AM7 = Chord(9, 0, (0, 3, 7, 10))


def in_range(p):
    return all(0 <= n.step < STEPS and 0 < n.length <= STEPS and 0 < n.vel <= 1 for n in p)


def test_drums_basic_grid():
    p = gen_drums(np.random.default_rng(0), 0.5)
    kicks = {n.step for n in p if n.note == 36}
    snares = {n.step for n in p if n.note in (38, 39)}
    assert {0, 4, 8, 12} <= kicks and snares == {4, 12} and in_range(p)


def test_drums_fill_adds_hits_at_end():
    p = gen_drums(np.random.default_rng(0), 0.5, fill=True)
    assert any(n.step >= 12 and n.note in (38, 45, 47) for n in p)
    c = gen_drums(np.random.default_rng(0), 0.5, crash=True)
    assert any(n.note == 49 and n.step == 0 for n in c)


def test_bass_styles_use_chord_root():
    for style in ("eighths", "octaves", "syncopated"):
        p = gen_bass(np.random.default_rng(1), AM7, style)
        assert p and in_range(p) and all(n.note % 12 in (9, 4) for n in p)


def test_arp_uses_chord_tones_every_step():
    p = gen_arp(np.random.default_rng(2), AM7, "updown")
    assert len(p) == STEPS and all(n.note % 12 in {9, 0, 4, 7} for n in p)
    assert [n.note for n in gen_arp(np.random.default_rng(0), AM7, "up")][:4] == [57, 60, 64, 67]


def test_pad_and_ambient_hold_whole_bar():
    pad = gen_pad(AM7)
    assert all(n.step == 0 and n.length == STEPS for n in pad) and len(pad) == 4
    amb = gen_ambient(AM7)
    assert len(amb) == 2 and all(n.note < 60 for n in amb)


def test_lead_notes_in_scale_and_sorted():
    scale = [n for n in range(60, 80) if n % 12 in {9, 11, 0, 2, 4, 5, 7}]
    p = gen_lead(np.random.default_rng(4), AM7, scale, 0.8)
    assert p and in_range(p) and all(n.note in scale for n in p)
    assert [n.step for n in p] == sorted(n.step for n in p)


def test_mutate_is_bounded_and_keeps_allowed_notes():
    base = gen_arp(np.random.default_rng(0), AM7, "up")
    allowed = [57, 60, 64, 67, 69, 72, 76, 79]
    m = mutate(np.random.default_rng(5), base, 0.2, allowed)
    assert in_range(m) and all(n.note in allowed for n in m)
    changed = len(set(base) ^ set(m))
    assert 0 < changed <= 12


def test_halftime_drums_put_snare_on_three():
    p = gen_drums(np.random.default_rng(0), 0.5, halftime=True)
    assert {n.step for n in p if n.note == 38} == {8}
    assert 0 in {n.step for n in p if n.note == 36} and 4 not in {n.step for n in p if n.note == 36}


def test_pad_octave_and_triad_widening():
    tri = Chord(4, 0, (0, 3, 7))
    p = gen_pad(tri, octave=3)
    assert [n.note for n in p] == [40, 43, 47, 52]


def test_new_bass_styles():
    for style in ("sixteenths", "walk", "riff"):
        p = gen_bass(np.random.default_rng(3), AM7, style)
        assert p and in_range(p)
    riff = gen_bass(np.random.default_rng(3), AM7, "riff")
    assert any(n.note % 12 in (10, 3) for n in riff)  # b2 or tritone of A
    six = gen_bass(np.random.default_rng(3), AM7, "sixteenths")
    assert len(six) == STEPS


def test_roll_on_beat_three_replaces_snare_and_hats_but_keeps_kick():
    base = gen_drums(np.random.default_rng(0), 0.5)
    p = add_roll(np.random.default_rng(1), base, 8, 4)
    window = [n for n in p if 8 <= n.step < 12]
    assert 8 in {n.step for n in window if n.note == 36}  # kick on the 3 kept
    assert not any(n.note == 42 for n in window)  # hats stripped
    roll = [n for n in window if n.note in (38, 45, 47)]
    assert [n.step for n in roll] == [8, 9, 10, 11]
    assert roll[0].vel < roll[-1].vel  # crescendo
    assert [n for n in p if n.step < 8] == [n for n in base if n.step < 8]
    assert [n for n in p if n.step >= 12] == [n for n in base if n.step >= 12]


def test_fill_roll_keeps_groove_before_beat_four():
    base = gen_drums(np.random.default_rng(3), 0.9)
    p = add_roll(np.random.default_rng(4), base, 12, 4)
    assert [n for n in p if n.step < 12] == [n for n in base if n.step < 12]
    assert {n.step for n in p if n.note in (38, 45, 47) and n.step >= 12} == {12, 13, 14, 15}
    assert in_range(p)


def _scale():
    return [n for n in range(60, 80) if n % 12 in {0, 2, 3, 5, 7, 8, 10}]  # C minor


def test_theme_motifs_share_rhythm_and_answer_resolves():
    t = gen_theme(np.random.default_rng(3), 0.8)
    q_steps = [n[0] for n in t.question.notes]
    assert q_steps == [n[0] for n in t.answer.notes] and q_steps[0] == 0
    assert t.answer.notes[-1][1] == 0  # resolves to the root
    assert t.question.notes[0][1] in (0, 2, 4)  # chord tone on the 1
    assert all(s % 4 == 0 for s, _, _ in t.counter.notes)  # counter: long notes
    assert len(t.counter.notes) <= len(t.question.notes)


def test_render_motif_is_a_diatonic_sequence():
    t = gen_theme(np.random.default_rng(1), 0.6)
    cm = Chord(0, 0, (0, 3, 7))
    fm = Chord(5, 3, (0, 3, 7))
    a = render_motif(np.random.default_rng(0), t.question, cm, _scale())
    b = render_motif(np.random.default_rng(0), t.question, fm, _scale())
    assert [n.step for n in a] == [n.step for n in b]  # same rhythm
    assert a[0].note % 12 == 0 and b[0].note % 12 == 5  # starts on each root
    assert all(n.note in _scale() for n in a + b) and in_range(a)
    # the contour (up / down between notes) is kept across chords
    sign = lambda xs: [np.sign(y.note - x.note) for x, y in zip(xs, xs[1:], strict=False)]  # noqa: E731
    assert sum(1 for x, y in zip(sign(a), sign(b), strict=True) if x == y) >= len(a) - 2
    up = render_motif(np.random.default_rng(0), t.question, cm, _scale(), octave=12)
    assert [n.note - 12 for n in up] == [n.note for n in a]


def test_snap_ride_shaker_flags():
    r = np.random.default_rng(0)
    p = gen_drums(r, 0.5, snap=True)
    assert {n.step for n in p if n.note == SNAP} == {4, 12} and not any(n.note == 38 for n in p)
    strong = gen_drums(r, 0.9, strong=True, snap=True)
    assert any(n.note == 38 for n in strong) and any(n.note == SNAP for n in strong)
    ride = gen_drums(r, 0.5, ride=True)
    assert {n.step for n in ride if n.note == RIDE} == set(range(0, STEPS, 2))
    assert not any(n.note == 42 and n.step % 2 == 0 for n in ride)
    sh = gen_drums(r, 0.5, shaker=True)
    assert len([n for n in sh if n.note == SHAKER]) == STEPS
    assert all(n.note == SNAP or n.note in (36, 38, 49) for n in drum_layer(p, 0))
    assert any(n.note == RIDE for n in drum_layer(ride, 1))


def test_tick_replaces_closed_hats_on_eighths():
    p = gen_drums(np.random.default_rng(0), 0.5, tick=True)
    assert {n.step for n in p if n.note == TICK} == set(range(0, STEPS, 2))
    assert not any(n.note == 42 and n.step % 2 == 0 for n in p)
    assert any(n.note == TICK for n in drum_layer(p, 1))


def test_straight_groove_is_deterministic_and_on_the_grid():
    """80s straight groove: no random kick or hat, identical whatever the rng draw."""
    a = gen_drums(np.random.default_rng(0), 0.9, straight=True)
    b = gen_drums(np.random.default_rng(7), 0.2, straight=True)
    assert a == b
    kicks = sorted(n.step for n in a if n.note == 36)
    assert kicks == [0, 4, 8, 12]  # four on the floor, nothing else
    snares = sorted(n.step for n in a if n.note == 38)
    claps = sorted(n.step for n in a if n.note == 39)
    assert snares == [4, 12] and claps == [4, 12]  # snare + clap layered on 2 and 4
    hats = sorted(n.step for n in a if n.note == 42)
    assert hats == list(range(0, STEPS, 2))  # plain 8ths, no 16th noise
    assert in_range(a)


def test_straight_chorus_adds_sixteenth_hats_not_extra_kicks():
    p = gen_drums(np.random.default_rng(1), 0.8, straight=True, strong=True)
    assert sorted(n.step for n in p if n.note == 36) == [0, 4, 8, 12]
    assert sorted(n.step for n in p if n.note == 42) == list(range(STEPS))


def test_straight_fill_is_a_plain_snare_crescendo():
    p = gen_drums(np.random.default_rng(2), 0.6, straight=True, fill=True)
    tail = sorted((n.step, n.note) for n in p if n.step >= 12)
    # snare 16ths only, and the four on the floor keeps its kick on the 4
    assert tail == [(12, 36), (12, 38), (13, 38), (14, 38), (15, 38)]


def _gaps(p):
    """(step, end) pairs sorted, to check notes do not run into each other."""
    s = sorted(p, key=lambda n: n.step)
    return [(a.step + a.length, b.step) for a, b in zip(s, s[1:], strict=False)]


def test_staccato_theme_detaches_every_note():
    t = gen_theme(np.random.default_rng(4), 0.8, staccato=True)
    for motif in (t.question, t.answer, t.counter):
        assert all(length <= 3 for _, _, length in motif.notes)
        steps = [s for s, _, _ in motif.notes]
        for (s, _, length), nxt in zip(motif.notes, steps[1:], strict=False):
            assert s + length <= nxt  # a rest before the next note, never legato


def test_legato_theme_is_unchanged_by_default():
    t = gen_theme(np.random.default_rng(4), 0.8)
    assert any(length > 3 for _, _, length in t.question.notes)


def test_straight_bass_is_short_and_deterministic():
    a = gen_bass(np.random.default_rng(0), AM7, "eighths", straight=True)
    b = gen_bass(np.random.default_rng(9), AM7, "eighths", straight=True)
    assert a == b
    assert [n.step for n in a] == list(range(0, STEPS, 2))
    assert all(n.length == 1 for n in a)  # punchy eighths, not tied


def test_harmonize_moves_every_note_by_diatonic_degrees():
    scale = [57, 59, 60, 62, 64, 65, 67, 69, 71, 72]  # A minor over two octaves
    theme = [Note(0, 69, 0.8, 2), Note(4, 72, 0.8, 2)]
    third_below = harmonize(theme, scale, -2)
    assert [n.note for n in third_below] == [65, 69]  # a diatonic third under each note
    assert [(n.step, n.length) for n in third_below] == [(0, 2), (4, 2)]
    assert all(n.vel < 0.8 for n in third_below)  # the harmony sits under the theme


def test_harmonize_keeps_notes_inside_the_scale_range():
    scale = [60, 62, 64, 65, 67]
    out = harmonize([Note(0, 60, 0.9, 1)], scale, -3)
    assert out[0].note == 60  # clipped to the lowest scale note, never off-scale
    assert harmonize([], scale, -2) == []
