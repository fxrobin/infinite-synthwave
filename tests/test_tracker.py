from synthwave.composer.arranger import LAYERS, BarPlan, Section
from synthwave.composer.harmony import Chord
from synthwave.composer.patterns import Note
from synthwave.sequencer.tracker import Tracker
from synthwave.sequencer.transport import Transport

SR = 44100


def test_transport_ticks_at_expected_offsets():
    t = Transport(SR, 120)  # step = 0.125 s = 5512.5 samples
    ticks = t.advance(12000)
    assert [(k.step, k.offset) for k in ticks] == [(0, 0), (1, 5512), (2, 11025)]
    assert t.advance(5000)[0].step == 3 and t.clock == 17000


def test_transport_bpm_change_keeps_phase():
    t = Transport(SR, 120)
    t.advance(1000)
    t.set_bpm(60)
    ticks = t.advance(20000)
    assert [k.offset for k in ticks] == [4512, 15537]


class FakeArranger:
    def __init__(self):
        self.calls = 0

    def next_bar(self):
        self.calls += 1
        pats = {layer: [] for layer in LAYERS}
        pats["bass"] = [Note(0, 45, 1.0, 2), Note(2, 45, 0.8, 1)]
        pats["pad"] = [Note(0, 57, 0.7, 16)]
        return BarPlan(
            self.calls - 1,
            Section.VERSE,
            0,
            Chord(9, 0, (0, 3, 7, 10)),
            pats,
            {layer: 1.0 for layer in LAYERS},
        )


def test_tracker_emits_ons_and_offs():
    t = Transport(SR, 120)
    tr = Tracker(t, FakeArranger())
    ev, plan = tr.advance(6000)
    assert plan is not None and plan.bar == 0
    assert [e.offset for e in ev["bass"] if e.on] == [0]
    ev2, plan2 = tr.advance(6000)
    assert plan2 is None
    offs = [e for e in ev2["bass"] if not e.on]
    assert [e.offset for e in offs] == [11025 - 6000]
    assert any(e.on and e.offset == 11025 - 6000 for e in ev2["bass"])


def test_tracker_pad_off_before_next_on_same_offset():
    t = Transport(SR, 120)
    tr = Tracker(t, FakeArranger())
    tr.advance(int(16 * t.samples_per_step))
    ev, plan = tr.advance(100)
    pads = sorted(ev["pad"])
    assert plan.bar == 1 and [e.on for e in pads] == [False, True] and pads[0].offset == 0
