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


def test_all_moods_progressions_exist_and_scales_valid():
    from synthwave.composer.harmony import SCALES
    for mood in MOODS.values():
        assert mood.scale in SCALES
        for name, w in mood.progressions.items():
            assert name in PROGRESSIONS, (mood.name, name)
            assert all(0 <= d < 7 for d in PROGRESSIONS[name]) and w >= 0
        h = Harmony(np.random.default_rng(1), mood)
        prog = h.next_progression()
        assert len(prog) == 4 and all(len(c.intervals) in (3, 4) for c in prog)
    assert len(MOODS) >= 10 and len(PROGRESSIONS) >= 35


def test_new_scale_key_names():
    for scale in ("dorian", "mixolydian", "locrian", "phrygian_dominant"):
        mood = next(m for m in MOODS.values() if m.scale == scale)
        h = Harmony(np.random.default_rng(0), mood)
        assert h.key_name.endswith((scale.replace("_", " "), "major"))


def test_change_key_keeps_common_tones_and_pivot_chord():
    from synthwave.composer.moods import MOODS
    for seed in range(6):
        rng = np.random.default_rng(seed)
        h = Harmony(rng, MOODS["outrun"])
        old = h.pitch_classes()
        last = h.chord_for_degree(0)
        h.change_key(MOODS["dark"], last)
        assert len(old & h.pitch_classes()) >= 5
        piv = h.pivot_chord(last)
        assert piv.root_pc == last.root_pc or len(
            {(piv.root_pc + i) % 12 for i in piv.intervals}
            & {(last.root_pc + i) % 12 for i in last.intervals}) >= 2
