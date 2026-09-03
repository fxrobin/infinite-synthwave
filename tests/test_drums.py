import numpy as np

from synthwave.engine.drums import DRUM_NOTES, DrumKit
from synthwave.engine.events import NoteEvent
from synthwave.patches.loader import load_patch

SR = 44100


def kit():
    return DrumKit(load_patch("drums_808"), SR, np.random.default_rng(0))


def test_all_samples_exist_finite_and_normalised():
    k = kit()
    for name in DRUM_NOTES:
        s = k.samples[name]
        assert s.ndim == 2 and s.shape[1] == 2 and np.isfinite(s).all()
        assert 0.25 < np.abs(s).max() <= 1.5


def test_kick_is_low_frequency():
    s = kit().samples["kick"][:, 0]
    spec = np.abs(np.fft.rfft(s))
    f = np.fft.rfftfreq(len(s), 1 / SR)[np.argmax(spec)]
    assert 30 < f < 120


def test_render_places_hit_at_offset_and_spans_blocks():
    k = kit()
    a = k.render(1024, [NoteEvent(1000, 36, 1.0, True)])
    b = k.render(1024, [])
    assert np.abs(a[:1000]).max() == 0 and np.abs(a[1000:]).max() > 0 and np.abs(b).max() > 0


def test_note_off_is_ignored():
    k = kit()
    out = k.render(512, [NoteEvent(0, 38, 1.0, False)])
    assert np.abs(out).max() == 0


def test_kick_has_strong_sub_energy():
    s = kit().samples["kick"][:, 0]
    spec = np.abs(np.fft.rfft(s)) ** 2
    f = np.fft.rfftfreq(len(s), 1 / SR)
    low, rest = spec[f < 100].sum(), spec[f >= 100].sum()
    assert low > rest * 3
    assert np.abs(s).max() > 1.2  # kick gain above unity, louder than the snare
    assert np.abs(kit().samples["snare"]).max() < 0.7


def test_perc_bus_echo_but_kick_dry():
    k = DrumKit(load_patch("drums_dark"), SR, np.random.default_rng(0), bpm=120)
    d = int(0.375 * SR)  # 1/8d at 120 BPM
    snare = k.render(d + 2000, [NoteEvent(0, 38, 1.0, True)])
    assert np.abs(snare[d + 200 : d + 1500]).max() > 0.05
    dry = load_patch("drums_dark").model_copy(update={"perc_effects": []})
    k2 = DrumKit(load_patch("drums_dark"), SR, np.random.default_rng(0), bpm=120)
    k3 = DrumKit(dry, SR, np.random.default_rng(0), bpm=120)
    kick = k2.render(d + 2000, [NoteEvent(0, 36, 1.0, True)])
    kick_dry = k3.render(d + 2000, [NoteEvent(0, 36, 1.0, True)])
    assert np.allclose(kick, kick_dry)


def test_snap_ride_shaker_render_and_are_distinct():
    import numpy as np

    from synthwave.engine.drums import DrumKit
    from synthwave.engine.events import NoteEvent
    from synthwave.patches.loader import load_patch

    kit = DrumKit(load_patch("drums_808"), 44100, np.random.default_rng(0), 110.0)
    outs = {}
    for name, note in (("snap", 40), ("ride", 51), ("shaker", 70)):
        kit.active = []
        outs[name] = kit.render(44100, [NoteEvent(0, note, 1.0, True)])
        assert np.isfinite(outs[name]).all() and np.abs(outs[name]).max() > 0.01
    dur = {k: int(np.flatnonzero(np.abs(v[:, 0]) > 0.005).max()) / 44100 for k, v in outs.items()}
    assert dur["snap"] < 0.4 < dur["ride"]  # snap short, ride long
    assert dur["shaker"] < 0.3


def test_tick_crash_roll_and_soft_kick():
    import numpy as np

    from synthwave.engine.drums import DrumKit
    from synthwave.engine.events import NoteEvent
    from synthwave.patches.loader import load_patch

    rng = np.random.default_rng(0)
    kit = DrumKit(load_patch("drums_acoustic"), 44100, rng, 120.0)
    tick = kit.render(22050, [NoteEvent(0, 44, 1.0, True)])
    assert np.abs(tick).max() > 0.01
    assert int(np.flatnonzero(np.abs(tick[:, 0]) > 0.01).max()) < 0.06 * 44100  # very short
    roll = kit.samples["crash_roll"]
    bar = int(44100 * 4 * 60 / 120)
    assert len(roll) > bar
    first, last = np.abs(roll[: bar // 4]).max(), np.abs(roll[bar * 3 // 4 : bar]).max()
    assert last > first * 3  # swells to the bar
    kit.set_bpm(60)
    assert len(kit.samples["crash_roll"]) > 2 * bar  # follows the tempo
    soft = DrumKit(load_patch("drums_acoustic"), 44100, rng).samples["kick"][:, 0]
    hard = DrumKit(load_patch("drums_industrial"), 44100, rng).samples["kick"][:, 0]

    def hf(x):
        sp = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / 44100)
        return sp[f > 1500].sum() / sp.sum()

    assert hf(soft) < hf(hard)  # less clicky / driven


def test_gain_ramp_keeps_perc_effect_tail():
    k = DrumKit(load_patch("drums_hall"), SR, np.random.default_rng(0))
    k.render(4096, [NoteEvent(0, 38, 1.0, True)])
    tail = k.render(4096, [], gain=np.zeros(4096, dtype=np.float32))
    assert np.abs(tail).max() > 1e-3
