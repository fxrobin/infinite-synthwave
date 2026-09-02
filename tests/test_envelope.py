import numpy as np
from synthwave.engine.envelope import ADSR
from synthwave.engine.lfo import LFO

SR = 1000


def test_attack_reaches_one_then_decays_to_sustain():
    env = ADSR(0.1, 0.2, 0.5, 0.1, SR)
    env.gate_on()
    a = env.render(100)
    assert np.isclose(a[-1], 1.0, atol=0.02) and np.all(np.diff(a) >= 0)
    d = env.render(400)
    assert np.isclose(d[-1], 0.5, atol=0.02)


def test_release_goes_to_zero_and_finishes():
    env = ADSR(0.0, 0.0, 1.0, 0.1, SR)
    env.gate_on()
    env.render(10)
    env.gate_off()
    r = env.render(200)
    assert r[0] < 1.0 and r[-1] < 0.01 and env.finished


def test_idle_renders_zeros():
    env = ADSR(0.1, 0.1, 0.5, 0.1, SR)
    assert np.all(env.render(50) == 0) and env.finished


def test_segment_spans_blocks():
    env = ADSR(0.05, 0.0, 1.0, 0.1, SR)
    env.gate_on()
    a = np.concatenate([env.render(20), env.render(20), env.render(20)])
    assert np.isclose(a[49], 1.0, atol=0.03) and a[10] < a[30]


def test_lfo_range_and_rate():
    lfo = LFO("sine", 2.0, SR)
    y = lfo.render(SR)
    assert -1.0 <= y.min() < -0.99 and 0.99 < y.max() <= 1.0
    assert 3 <= np.sum(np.diff(y > 0) != 0) <= 4
