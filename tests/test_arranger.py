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
    assert after.section == Section.INTRO and a.mood.name == "outrun"


def test_transitions_occur_naturally_and_change_key():
    a = make(seed=11)
    plans = [a.next_bar() for _ in range(400)]
    trans = [p for p in plans if p.section == Section.TRANSITION]
    assert len(trans) >= 3 * 4
    assert len({p.key for p in plans}) >= 2


def test_bass_instrument_rotates_between_sections():
    a = make(seed=3, mood="dark")
    plans = [a.next_bar() for _ in range(200)]
    firsts = [p for p in plans if p.patches]
    basses = {p.patches["bass"] for p in firsts}
    assert len(firsts) >= 5 and len(basses) >= 2
    assert all(b.startswith("bass_") for b in basses)
    first_track = [p for p in firsts if p.track == 1 and p.section != Section.TRANSITION]
    assert {p.patches["drums"] for p in first_track} <= {
        "drums_dark",
        "drums_industrial",
        "drums_hall",
        "drums_lofi",
    }
    assert len({p.patches["pad"] for p in firsts} | {p.patches["arp"] for p in firsts}) >= 3
    b0, b1 = plans[0].patterns["bass"], plans[1].patterns["bass"]
    assert b0 != b1 or plans[0].chord != plans[1].chord


def test_risers_announce_chorus_and_chorus_is_strong():
    a = make(seed=5, mood="dark")
    plans = [a.next_bar() for _ in range(120)]
    chorus_starts = [
        i for i, p in enumerate(plans) if p.section == Section.CHORUS and p.section_bar == 0
    ]
    assert chorus_starts
    i = chorus_starts[0]
    assert any(n.note == 63 for n in plans[i].patterns["riser"])  # impact
    assert any(n.note == 60 for n in plans[i - 1].patterns["riser"])  # reverse cymbal
    assert any(n.note == 61 for n in plans[i - 2].patterns["riser"])  # uplifter
    drums = plans[i].patterns["drums"]
    assert {n.step for n in drums if n.note == 36} >= {0, 4, 8, 12}
    assert {n.step for n in drums if n.note == 38} == {4, 12}
    assert plans[i].fx and "lead" in plans[i].fx


def _kicks(p):
    return {n.step for n in p.patterns["drums"] if n.note == 36}


def _rolls_on_three(p):
    return [n for n in p.patterns["drums"] if 8 <= n.step < 12 and n.note in (38, 45, 47)]


def test_groove_is_stable_within_verse_and_chorus():
    """Kick grid never changes inside one section: rolls and fills leave the kick alone."""
    for mood in ("outrun", "dark"):
        a = make(seed=7, mood=mood)
        plans = [a.next_bar() for _ in range(160)]
        sec_id, ref = None, None
        for p in plans:
            if p.section not in (Section.VERSE, Section.CHORUS) or p.fill or p.drop:
                continue
            key = (p.section, p.bar - p.section_bar)
            if key != sec_id:
                sec_id, ref = key, _kicks(p)
            assert _kicks(p) == ref


def test_rolls_on_beat_three_appear_at_phrase_ends_only():
    a = make(seed=11, mood="outrun")
    plans = [a.next_bar() for _ in range(200)]
    rolled = [p for p in plans if len(_rolls_on_three(p)) >= 4]
    assert rolled
    for p in rolled:
        assert (p.section_bar % 4 == 3 and (not p.fill or p.section == Section.OUTRO)) or p.drop


def test_fill_bar_keeps_section_groove():
    a = make(seed=3, mood="outrun")
    plans = [a.next_bar() for _ in range(60)]
    for prev, p in zip(plans, plans[1:], strict=False):
        if p.fill and not p.drop and p.section == prev.section == Section.VERSE:

            def head(plan):
                return {
                    (n.step, n.note) for n in plan.patterns["drums"] if n.step < 12 and n.note != 49
                }

            assert head(prev) == head(p)
            assert {n.step for n in p.patterns["drums"] if n.step >= 12 and n.note in (38, 45, 47)}


def _active(p):
    return {
        layer
        for layer in LAYERS
        if p.gains[layer] > 0 and p.patterns[layer] and layer not in ("riser", "lead")
    }  # lead phrases are drawn per bar


def test_layers_build_up_every_two_bars_in_verse():
    a = make(seed=5, mood="outrun")
    plans = [a.next_bar() for _ in range(60)]
    verse = [p for p in plans if p.section == Section.VERSE][:8]
    assert verse and verse[0].section_bar == 0
    counts = [len(_active(p)) for p in verse]
    assert counts == sorted(counts) and counts[-1] > counts[0]
    assert "arp" not in _active(verse[0]) and "arp" in _active(verse[2])  # arp at bar 2
    assert verse[0].gains["lead"] == 0 and verse[4].gains["lead"] > 0  # lead at bar 4

    def offbeat_hats(p):
        return [
            n for n in p.patterns["drums"] if n.note in (42, 46) and n.step % 2 == 1 or n.note == 46
        ]

    assert not any(offbeat_hats(p) for p in verse[:4])  # 8th hats only
    assert any(offbeat_hats(p) for p in verse[4:])  # full groove


def test_predrop_cuts_percussion_before_chorus_and_chorus_hits():
    a = make(seed=5, mood="outrun")
    plans = [a.next_bar() for _ in range(200)]
    starts = [i for i, p in enumerate(plans) if p.section == Section.CHORUS and p.section_bar == 0]
    assert starts
    for i in starts:
        pre = plans[i - 1]
        assert pre.drop and pre.fill
        drums = pre.patterns["drums"]
        assert not any(n.step >= 12 for n in drums)  # silence on beat 4
        assert not any(n.note == 42 for n in drums)  # no hats
        assert len([n for n in drums if n.note in (38, 45, 47)]) == 8  # snare roll 4..11
        assert not any(n.step >= 12 for n in pre.patterns["bass"])
        assert pre.gains["lead"] == 0
        assert any(n.note == 63 for n in plans[i].patterns["riser"])  # impact = drop
        assert plans[i].gains["drums"] == 1.0 and plans[i].gains["bass"] == 1.0


def test_tracks_last_about_three_and_a_half_minutes_and_end_with_outro():
    for mood, seed in (("outrun", 1), ("horror", 2), ("drive", 3)):
        rng = np.random.default_rng(seed)
        m = MOODS[mood]
        a = Arranger(rng, Harmony(rng, m), m, bpm=m.bpm)
        plans = [a.next_bar() for _ in range(600)]
        bpm = m.bpm
        for p in plans:
            if p.section == Section.TRANSITION and p.section_bar == 0:
                prev = plans[p.bar - 1]
                assert prev.section == Section.OUTRO
                seconds = prev.track_bars * 240.0 / bpm
                assert 180 <= seconds <= 240, (mood, seconds)
            if p.bpm:
                bpm = p.bpm
        assert {p.track for p in plans} >= {1, 2, 3}
        assert not any(p.fade < 1.0 for p in plans)  # infinite mode: no fade-out


def test_transition_is_harmonically_smooth():
    a = make(seed=8)
    plans = [a.next_bar() for _ in range(700)]
    for p in plans:
        if p.section == Section.TRANSITION and p.section_bar == 0:
            before = plans[p.bar - 1]
            prev_pcs = {(before.chord.root_pc + i) % 12 for i in before.chord.intervals}
            pcs = {(p.chord.root_pc + i) % 12 for i in p.chord.intervals}
            assert len(prev_pcs & pcs) >= 2 or p.chord.root_pc == before.chord.root_pc
            lo, hi = MOODS[p.mood or before.mood or a.mood.name].bpm_range
            assert lo <= p.bpm <= hi
    assert len({p.key for p in plans}) >= 2


def test_theme_is_kept_for_the_whole_track_and_breaks_get_a_counter_melody():
    a = make(seed=6, mood="outrun")
    themes, breaks = {}, []
    for _ in range(400):
        p = a.next_bar()
        themes.setdefault(p.track, set()).add(a.theme)
        if p.section == Section.BREAK and p.gains["lead"] > 0 and p.patterns["lead"]:
            breaks.append(p)
    assert all(len(v) == 1 for v in themes.values()) and len(themes) >= 2
    assert len({next(iter(v)) for v in themes.values()}) == len(themes)  # new theme per track
    notes = [n for p in breaks for n in p.patterns["lead"]]
    assert breaks and sum(1 for n in notes if n.step % 4 == 0) >= 0.8 * len(notes)
    # chorus bars replay the theme: same rhythm on even bars
    a = make(seed=6, mood="outrun")
    plans = [a.next_bar() for _ in range(80)]
    q = [n[0] for n in a.theme.question.notes]
    chorus = [
        p
        for p in plans
        if p.section == Section.CHORUS
        and p.patterns["lead"]
        and p.section_bar % 2 == 0
        and p.section_bar < 8
    ]
    assert chorus
    same = sum(1 for p in chorus if [n.step for n in p.patterns["lead"]] == q)
    assert same >= len(chorus) // 2


def test_live_composer_gestures_and_build_up_sweep():
    a = make(seed=5, mood="outrun")
    plans = [a.next_bar() for _ in range(300)]
    assert any(p.tweaks for p in plans)
    layers = {layer for p in plans if p.tweaks for layer in p.tweaks}
    assert {"pad", "arp"} <= layers
    starts = [
        i
        for i, p in enumerate(plans)
        if p.section == Section.CHORUS and p.section_bar == 0 and i >= 4
    ]
    assert starts
    i = starts[0]
    cut = [plans[j].tweaks["pad"]["filter.cutoff"] for j in range(i - 4, i)]
    assert cut == sorted(cut) and cut[-1] > cut[0] * 1.8  # filter opens towards the drop
    for p in plans:
        if p.section == Section.TRANSITION:
            assert not p.tweaks


def test_predrop_carries_a_cymbal_roll():
    a = make(seed=5, mood="outrun")
    plans = [a.next_bar() for _ in range(200)]
    pre = [p for p in plans if p.drop]
    assert pre and all(any(n.note == 57 and n.step == 0 for n in p.patterns["drums"]) for p in pre)
