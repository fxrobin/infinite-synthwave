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
    assert np.abs(snare[d + 200:d + 1500]).max() > 0.05
    dry = load_patch("drums_dark").model_copy(update={"perc_effects": []})
    k2 = DrumKit(load_patch("drums_dark"), SR, np.random.default_rng(0), bpm=120)
    k3 = DrumKit(dry, SR, np.random.default_rng(0), bpm=120)
    kick = k2.render(d + 2000, [NoteEvent(0, 36, 1.0, True)])
    kick_dry = k3.render(d + 2000, [NoteEvent(0, 36, 1.0, True)])
    assert np.allclose(kick, kick_dry)
