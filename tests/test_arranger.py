import numpy as np

from synthwave.composer.arranger import LAYERS, Arranger, Section
from synthwave.composer.harmony import Harmony
from synthwave.composer.moods import MOODS


def make(seed=0, total_bars=None, mood="outrun"):
    rng = np.random.default_rng(seed)
    m = MOODS[mood]
    return Arranger(rng, Harmony(rng, m), m, total_bars)


def test_starts_with_intro_then_verse():
    a = make()
    plans = [a.next_bar() for _ in range(9)]
    assert plans[0].section == Section.INTRO and plans[7].section == Section.INTRO
    assert plans[8].section == Section.VERSE and plans[8].section_bar == 0


def test_plan_has_all_layers_and_gains():
    p = make().next_bar()
    assert set(p.patterns) == set(LAYERS) and set(p.gains) == set(LAYERS)
    assert p.gains["pad"] > 0 and p.gains["lead"] == 0


def test_fill_on_last_bar_and_bars_differ():
    a = make(seed=2)
    plans = [a.next_bar() for _ in range(40)]
    assert plans[7].fill and not plans[6].fill
    for x, y in zip(plans, plans[1:], strict=False):
        assert x.patterns != y.patterns or x.chord != y.chord


def test_duration_mode_ends_with_outro_and_finished():
    a = make(seed=1, total_bars=20)
    plans = [a.next_bar() for _ in range(22)]
    assert plans[12].section == Section.OUTRO and plans[19].section == Section.OUTRO
    assert plans[19].fade < plans[12].fade
    assert plans[20].finished and plans[21].finished
    assert all(g == 0 for g in plans[20].gains.values())


def test_force_next_section():
    a = make()
    a.next_bar()
    a.force_next_section()
    assert a.next_bar().section == Section.VERSE


def test_deterministic_by_seed():
    a, b = make(seed=9), make(seed=9)
    for _ in range(30):
        assert a.next_bar() == b.next_bar()
