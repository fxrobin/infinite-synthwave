import numpy as np
from synthwave.engine.filter import Filter

SR = 44100


def tone(f, n=SR):
    t = np.arange(n) / SR
    return np.stack([np.sin(2 * np.pi * f * t)] * 2, axis=1).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(x[SR // 2:, 0] ** 2)))


def test_lowpass_attenuates_above_cutoff():
    f = Filter("lp", SR)
    low = rms(f.process(tone(200), 1000, 0.0))
    f = Filter("lp", SR)
    high = rms(f.process(tone(8000), 1000, 0.0))
    assert high < low * 0.05


def test_highpass_attenuates_below_cutoff():
    f = Filter("hp", SR)
    low = rms(f.process(tone(100), 2000, 0.0))
    f = Filter("hp", SR)
    high = rms(f.process(tone(8000), 2000, 0.0))
    assert low < high * 0.05


def test_state_persists_across_blocks():
    f1, f2 = Filter("lp", SR), Filter("lp", SR)
    x = tone(300, 2048)
    whole = f1.process(x, 500, 0.5)
    parts = np.concatenate([f2.process(x[:1024], 500, 0.5), f2.process(x[1024:], 500, 0.5)])
    assert np.allclose(whole, parts, atol=1e-5)


def test_extreme_cutoff_is_clamped_and_stable():
    f = Filter("lp", SR)
    y = f.process(tone(1000, 4096), 100000, 1.0)
    assert np.isfinite(y).all()
