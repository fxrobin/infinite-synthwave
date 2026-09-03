import numpy as np

from synthwave.engine.events import NoteEvent
from synthwave.engine.risers import RISER_NOTES, RiserKit

SR = 44100


def test_samples_exist_and_reverse_cymbal_swells_to_bar_end():
    k = RiserKit(SR, np.random.default_rng(0), bpm=120)
    bar = int(SR * 2.0)
    for name in RISER_NOTES:
        s = k.samples[name]
        assert np.isfinite(s).all() and np.abs(s).max() <= 1.0
    rev = np.abs(k.samples["reverse_cymbal"][:, 0])
    assert len(rev) == bar and rev[-bar // 8 :].mean() > rev[: bar // 8].mean() * 10
    assert len(k.samples["uplifter"]) == 2 * bar


def test_render_and_bpm_rebuild():
    k = RiserKit(SR, np.random.default_rng(0), bpm=120)
    out = k.render(2048, [NoteEvent(100, 63, 1.0, True)])
    assert np.abs(out[:100]).max() == 0 and np.abs(out[100:]).max() > 0.1
    k.set_bpm(60)
    assert len(k.samples["reverse_cymbal"]) == SR * 4
