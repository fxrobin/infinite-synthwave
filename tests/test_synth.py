import numpy as np

from synthwave.engine.events import NoteEvent
from synthwave.engine.synth import Synth
from synthwave.engine.voice import Voice
from synthwave.patches.loader import load_patch, patch_from_dict

SR = 44100


def simple_patch(**kw):
    d = {
        "name": "t",
        "polyphony": 2,
        "oscillators": [{"wave": "sine"}],
        "amp_env": {"attack": 0.001, "decay": 0.01, "sustain": 1.0, "release": 0.01},
    }
    d.update(kw)
    return patch_from_dict(d)


def test_voice_pitch():
    v = Voice(simple_patch(), SR, np.random.default_rng(0))
    v.note_on(69, 1.0)
    sig = v.render(SR)[:, 0]
    spec = np.abs(np.fft.rfft(sig * np.hanning(SR)))
    assert abs(np.fft.rfftfreq(SR, 1 / SR)[np.argmax(spec)] - 440) < 2


def test_voice_release_then_inactive():
    v = Voice(simple_patch(), SR, np.random.default_rng(0))
    v.note_on(60, 1.0)
    v.render(100)
    v.note_off()
    v.render(SR // 10)
    assert not v.active


def test_synth_events_timing():
    s = Synth(simple_patch(), SR, np.random.default_rng(0), 110)
    out = s.render(2000, [NoteEvent(1000, 60, 1.0, True)])
    assert np.abs(out[:1000]).max() == 0 and np.abs(out[1000:]).max() > 0.1


def test_synth_voice_stealing_keeps_polyphony():
    s = Synth(simple_patch(polyphony=2), SR, np.random.default_rng(0), 110)
    evs = [NoteEvent(0, 60, 1.0, True), NoteEvent(0, 64, 1.0, True), NoteEvent(0, 67, 1.0, True)]
    s.render(512, evs)
    assert sum(v.active for v in s.voices) == 2


def test_synth_with_library_patch_is_finite_and_stereo():
    s = Synth(load_patch("pad_juno"), SR, np.random.default_rng(1), 110)
    out = s.render(4096, [NoteEvent(0, 57, 0.8, True), NoteEvent(0, 60, 0.8, True)])
    assert out.shape == (4096, 2) and np.isfinite(out).all() and np.abs(out).max() > 0.01


def test_set_patch_replaces_voices():
    s = Synth(simple_patch(), SR, np.random.default_rng(0), 110)
    s.set_patch(load_patch("bass_moog"))
    assert s.patch.name == "bass_moog" and len(s.voices) == 1


def test_update_patch_keeps_voices_playing():
    import numpy as np

    from synthwave.engine.events import NoteEvent
    from synthwave.engine.synth import Synth
    from synthwave.patches.loader import apply_tweaks, load_patch

    patch = load_patch("pad_juno")
    s = Synth(patch, 44100, np.random.default_rng(0), 120.0)
    s.render(4096, [NoteEvent(0, 60, 0.9, True)])
    voices = list(s.voices)
    s.update_patch(apply_tweaks(patch, {"filter.cutoff": 0.5, "oscillators.0.detune": 1.5}))
    assert s.voices == voices and any(v.active for v in s.voices)
    assert s.patch.filter.cutoff == patch.filter.cutoff * 0.5
    out = s.render(4096, [])
    assert np.abs(out).max() > 0.001  # note still sounding
    s.update_patch(load_patch("pad_dark"))  # structural change -> reset
    assert s.voices != voices


def test_gain_ramp_applied_before_effects_keeps_delay_tail():
    """Muting a layer (gain 0) must not cut the delay/reverb tail already in the effect."""
    patch = simple_patch(effects=[{"type": "delay", "time": 0.05, "feedback": 0.5, "mix": 0.5}])
    s = Synth(patch, SR, np.random.default_rng(0), 110)
    s.render(4096, [NoteEvent(0, 60, 1.0, True), NoteEvent(2048, 60, 1.0, False)])
    zero = np.zeros(4096, dtype=np.float32)
    tail = s.render(4096, [], gain=zero)
    assert np.abs(tail).max() > 1e-3  # echo still audible although the dry signal is gated
    later = s.render(4096, [NoteEvent(0, 60, 1.0, True)], gain=zero)
    assert np.abs(later).max() < np.abs(tail).max()  # dry note gated, tail decaying
