import numpy as np
from synthwave.engine.effects import (Chorus, Delay, GatedReverb, Limiter, Reverb, Sidechain,
                                      build_effects, note_to_seconds)

SR = 44100


def impulse(n=SR, at=0, amp=1.0):
    x = np.zeros((n, 2), np.float32)
    x[at] = amp
    return x


def test_note_to_seconds():
    assert np.isclose(note_to_seconds("1/8", 120), 0.25)
    assert np.isclose(note_to_seconds("1/8d", 120), 0.375)
    assert np.isclose(note_to_seconds(0.3, 120), 0.3)


def test_delay_echo_position():
    d = Delay(SR, 120, time="1/8", feedback=0.0, mix=1.0, pingpong=False)
    y = d.process(impulse())
    assert np.argmax(y[:, 0]) == int(0.25 * SR)


def test_delay_handles_blocks_longer_than_delay_time():
    d = Delay(SR, 120, time=0.005, feedback=0.5, mix=1.0, pingpong=False)
    y = d.process(impulse(2048))
    peaks = np.flatnonzero(y[:, 0] > 0.1)
    assert peaks[0] == int(0.005 * SR) and len(peaks) >= 2


def test_reverb_has_decaying_tail():
    r = Reverb(SR, 120, size=0.8, damping=0.5, mix=1.0, predelay=0.0)
    y = r.process(impulse(2 * SR))
    e1 = np.sum(y[SR // 10:SR // 2] ** 2)
    e2 = np.sum(y[SR:SR + SR // 2] ** 2)
    assert e1 > 1e-6 and e2 < e1 and np.isfinite(y).all()


def test_reverb_block_and_whole_equivalent():
    x = np.random.default_rng(0).normal(size=(4096, 2)).astype(np.float32) * 0.1
    a = Reverb(SR, 120, mix=1.0, predelay=0.0).process(x)
    r = Reverb(SR, 120, mix=1.0, predelay=0.0)
    b = np.concatenate([r.process(x[:1000]), r.process(x[1000:])])
    assert np.allclose(a, b, atol=1e-5)


def test_gated_reverb_cuts_after_hold():
    g = GatedReverb(SR, 120, size=0.9, hold=0.1, threshold=0.5, mix=1.0)
    y = g.process(impulse(SR))
    assert np.abs(y[int(0.02 * SR):int(0.09 * SR)]).max() > 0
    assert np.abs(y[int(0.15 * SR):]).max() == 0


def test_chorus_shape_and_stereo():
    c = Chorus(SR, 120)
    x = np.stack([np.sin(np.arange(4096) * 0.05)] * 2, axis=1).astype(np.float32)
    y = c.process(x)
    assert y.shape == x.shape and y.dtype == np.float32 and not np.allclose(y[:, 0], y[:, 1])


def test_sidechain_dips_on_trigger():
    sc = Sidechain(SR, depth=0.6, release=0.1)
    g = sc.gain(1000, [500])
    assert np.isclose(g[0], 1.0, atol=1e-3) and g[500] < 0.45 and g[999] > g[500]


def test_limiter_bounds_output():
    lim = Limiter(SR, 120, threshold=0.9)
    y = lim.process(impulse(2048, at=100, amp=3.0))
    assert np.abs(y).max() <= 1.0


def test_build_effects_registry():
    fx = build_effects([{"type": "chorus"}, {"type": "delay", "time": "1/4"}], SR, 110)
    assert len(fx) == 2 and isinstance(fx[0], Chorus)
