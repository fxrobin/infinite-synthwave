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


def test_sections_carry_effect_specs():
    a = make(seed=4)
    plans = [a.next_bar() for _ in range(160)]
    used = {k for p in plans if p.fx for k in p.fx}
    kinds = {e["type"] for p in plans if p.fx for specs in p.fx.values() for e in specs}
    assert used & {"pad", "master", "arp"} and kinds & {"gate", "lofi", "bitcrush"}


def test_transition_is_ambient_only_and_applies_mood():
    a = make(seed=1, mood="dark")
    a.next_bar()
    a.set_mood(MOODS["outrun"])
    plans = [a.next_bar() for _ in range(12)]
    t = [p for p in plans if p.section == Section.TRANSITION]
    assert t and t[0].section_bar == 0 and t[0].mood == "outrun" and t[0].bpm
    assert 110 <= t[0].bpm <= 128 and len(t) == 4
    for p in t:
        assert p.gains["drums"] == 0 and p.gains["bass"] == 0 and p.gains["ambient"] == 1.0
        assert p.patterns["drums"] == []
    after = plans[plans.index(t[-1]) + 1]
    assert after.section in (Section.VERSE, Section.CHORUS) and a.mood.name == "outrun"


def test_transitions_occur_naturally_and_change_key():
    a = make(seed=11)
    plans = [a.next_bar() for _ in range(400)]
    trans = [p for p in plans if p.section == Section.TRANSITION]
    assert len(trans) >= 3
    assert len({p.key for p in plans}) >= 2


def test_bass_instrument_rotates_between_sections():
    a = make(seed=3, mood="dark")
    plans = [a.next_bar() for _ in range(200)]
    firsts = [p for p in plans if p.patches]
    basses = {p.patches["bass"] for p in firsts}
    assert len(firsts) >= 5 and len(basses) >= 2
    assert basses <= {"bass_dark", "bass_industrial", "bass_reese", "bass_moog", "bass_sub",
                      "bass_growl", "bass_pulse"}
    b0, b1 = plans[0].patterns["bass"], plans[1].patterns["bass"]
    assert b0 != b1 or plans[0].chord != plans[1].chord


def test_risers_announce_chorus_and_chorus_is_strong():
    a = make(seed=5, mood="dark")
    plans = [a.next_bar() for _ in range(120)]
    chorus_starts = [i for i, p in enumerate(plans)
                     if p.section == Section.CHORUS and p.section_bar == 0]
    assert chorus_starts
    i = chorus_starts[0]
    assert any(n.note == 63 for n in plans[i].patterns["riser"])          # impact
    assert any(n.note == 60 for n in plans[i - 1].patterns["riser"])      # reverse cymbal
    assert any(n.note == 61 for n in plans[i - 2].patterns["riser"])      # uplifter
    drums = plans[i].patterns["drums"]
    assert {n.step for n in drums if n.note == 36} >= {0, 4, 8, 12}
    assert {n.step for n in drums if n.note == 38} == {4, 12}
    assert plans[i].fx and "lead" in plans[i].fx
