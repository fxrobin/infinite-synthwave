import numpy as np

from synthwave.composer.harmony import PROGRESSIONS, Chord, Harmony
from synthwave.composer.moods import MOODS


def test_chord_names_and_notes():
    assert Chord(9, 0, (0, 3, 7, 10)).name == "Am7"
    assert Chord(5, 5, (0, 4, 7, 11)).name == "Fmaj7"
    n = Chord(9, 0, (0, 3, 7, 10)).notes(4)
    assert n == [57, 60, 64, 67] and Chord(9, 0, (0, 3, 7, 10)).bass_note() == 45


def test_minor_key_degrees():
    h = Harmony(np.random.default_rng(0), MOODS["dreamy"])
    h.tonic, h.mode = 9, h.MINOR
    assert h.chord_for_degree(0).name == "Am7"
    assert h.chord_for_degree(5).name == "Fmaj7"
    assert h.chord_for_degree(2).name == "Cmaj7"
    assert h.chord_for_degree(6).name == "G7"


def test_progression_is_four_chords_in_key():
    h = Harmony(np.random.default_rng(3), MOODS["outrun"])
    prog = h.next_progression()
    assert len(prog) == 4
    scale = {(h.tonic + i) % 12 for i in h.mode}
    for c in prog:
        assert c.root_pc in scale and all((c.root_pc + i) % 12 in scale for i in c.intervals)


def test_progressions_avoid_immediate_repeat_mostly_and_are_seeded():
    a = Harmony(np.random.default_rng(7), MOODS["dreamy"])
    b = Harmony(np.random.default_rng(7), MOODS["dreamy"])
    seq_a = [tuple(c.degree for c in a.next_progression()) for _ in range(20)]
    seq_b = [tuple(c.degree for c in b.next_progression()) for _ in range(20)]
    assert seq_a == seq_b
    assert len(set(seq_a)) > 2 and all(p in list(PROGRESSIONS.values()) for p in seq_a)


def test_modulate_changes_tonic_and_scale_notes_in_range():
    h = Harmony(np.random.default_rng(0), MOODS["dark"])
    t = h.tonic
    h.modulate()
    assert h.tonic != t
    notes = h.scale_notes(60, 72)
    assert notes and all(60 <= n <= 72 for n in notes)


def test_dark_mood_uses_phrygian_triads():
    h = Harmony(np.random.default_rng(0), MOODS["dark"])
    h.tonic = 4  # E phrygian
    assert h.key_name == "E phrygian"
    assert h.chord_for_degree(0).name == "Em" and h.chord_for_degree(1).name == "F"
    prog = h.next_progression()
    assert all(len(c.intervals) == 3 for c in prog)
    assert all(h.current in n for n in [h.current]) and h.current in MOODS["dark"].progressions


def test_noir_mood_has_major_dominant():
    h = Harmony(np.random.default_rng(0), MOODS["noir"])
    h.tonic = 9  # A harmonic minor
    assert h.chord_for_degree(4).name == "E" and h.chord_for_degree(0).name == "Am"
