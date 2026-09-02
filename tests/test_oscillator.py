import numpy as np

from synthwave.engine.oscillator import Oscillator, render_wave

SR = 44100


def dominant_freq(sig, sr):
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    return np.fft.rfftfreq(len(sig), 1 / sr)[np.argmax(spec)]


def test_saw_fundamental_frequency():
    osc = Oscillator("saw", SR, np.random.default_rng(0))
    sig = osc.render(440.0, SR)[:, 0]
    assert abs(dominant_freq(sig, SR) - 440.0) < 2.0


def test_square_has_no_even_harmonics():
    osc = Oscillator("square", SR, np.random.default_rng(0))
    sig = osc.render(200.0, SR)[:, 0]
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(SR, 1 / SR)
    h1 = spec[np.argmin(abs(freqs - 200))]
    h2 = spec[np.argmin(abs(freqs - 400))]
    assert h2 < h1 * 0.05


def test_unison_is_stereo_and_bounded():
    osc = Oscillator("saw", SR, np.random.default_rng(0), unison=5, detune=15, spread=1.0)
    out = osc.render(110.0, 4096)
    assert out.shape == (4096, 2) and out.dtype == np.float32
    assert np.abs(out).max() <= 1.5
    assert not np.allclose(out[:, 0], out[:, 1])


def test_phase_continuity_between_blocks():
    osc = Oscillator("sine", SR, np.random.default_rng(0))
    a = osc.render(1000.0, 512)[:, 0]
    b = osc.render(1000.0, 512)[:, 0]
    joined = np.concatenate([a, b])
    assert np.abs(np.diff(joined)).max() < 0.2


def test_noise_uses_rng():
    phase = np.zeros(100)
    n1 = render_wave("noise", phase, 0.01, rng=np.random.default_rng(1))
    n2 = render_wave("noise", phase, 0.01, rng=np.random.default_rng(1))
    assert np.array_equal(n1, n2) and np.abs(n1).max() <= 1.0
