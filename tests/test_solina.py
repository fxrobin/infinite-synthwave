import numpy as np

from synthwave.engine.events import NoteEvent
from synthwave.patches.loader import patch_from_dict
from synthwave.patches.model import SolinaPatchModel

SR = 44100


def test_solina_patch_model_roundtrip():
    p = patch_from_dict({"name": "s", "kind": "solina", "registers": {"cello": True}})
    assert isinstance(p, SolinaPatchModel)
    assert p.registers.cello and p.registers.viola and not p.registers.horn
    assert p.crescendo == 0.3 and p.sustain_length == 0.8 and p.ensemble


# ----- générateur & keyer -----
from synthwave.engine.effects import _REGISTRY  # noqa: E402
from synthwave.engine.solina import (  # noqa: E402
    RcKeyer,
    SolinaEnsemble,
    SolinaSynth,
    note_hz,
    staircase_saw,
)


def test_note_hz_is_exact_octave_division():
    assert abs(note_hz(69) - 440.0) < 1e-9
    assert note_hz(57) == note_hz(69) / 2 and note_hz(45) == note_hz(69) / 4
    assert abs(note_hz(69, 100.0) / note_hz(70) - 1.0) < 1e-9


def test_staircase_saw_has_sixteen_levels():
    ph = np.linspace(0, 1, 64, endpoint=False)
    w = staircase_saw(ph)
    assert len(np.unique(np.round(w, 6))) == 16 and w.min() == -1.0 and w.max() == 1.0


def test_rc_keyer_attack_time():
    k = RcKeyer(SR, attack_s=0.5, release_s=0.2)
    k.gate_on(3)
    lvl = k.render(SR)[:, 3]
    assert 0.55 < lvl[22050] < 0.72 and lvl[-1] > 0.85
    assert k.render(10)[:, 0].max() == 0.0


def test_rc_keyer_retrigger_keeps_level():
    k = RcKeyer(SR, attack_s=0.01, release_s=1.0)
    k.gate_on(0)
    k.render(4410)
    k.gate_off(0)
    k.render(4410)
    before = k.render(1)[0, 0]
    k.gate_on(0)
    after = k.render(1)[0, 0]
    assert 0.5 < before < after


# ----- ensemble -----
def _spread(sig):
    spec = np.abs(np.fft.rfft(sig[SR:] * np.hanning(SR)))
    f = np.fft.rfftfreq(SR, 1 / SR)
    band = (f > 430) & (f < 450)
    core = (f > 436) & (f < 444)
    return spec[band & ~core].sum() / spec[band].sum()


def test_ensemble_widens_spectrum_and_mono_option():
    t = np.arange(SR * 2) / SR
    x = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, 1).astype(np.float32)
    ens = SolinaEnsemble(SR, 110, stereo=False)
    y = np.concatenate([ens.process(x[i : i + 1024]) for i in range(0, len(x), 1024)])
    assert np.allclose(y[:, 0], y[:, 1])
    assert _spread(x[:, 0]) < 0.05 < _spread(y[:, 0])
    ens = SolinaEnsemble(SR, 110, stereo=True)
    y = np.concatenate([ens.process(x[i : i + 1024]) for i in range(0, len(x), 1024)])
    assert not np.allclose(y[:, 0], y[:, 1])
    assert "ensemble" in _REGISTRY


# ----- synth -----
def _synth(**kw):
    p = patch_from_dict({"name": "t", "kind": "solina", **kw})
    return SolinaSynth(p, SR, np.random.default_rng(0), 110)


def test_solina_renders_finite_stereo():
    out = _synth().render(4096, [NoteEvent(0, 60, 1.0, True), NoteEvent(0, 64, 1.0, True)])
    assert out.shape == (4096, 2) and np.isfinite(out).all() and np.abs(out).max() > 0.01


def test_bass_section_is_mono_lowest_note():
    s = _synth(
        registers={"cello": True, "viola": False, "violin": False}, ensemble=False, crescendo=0.005
    )
    out = s.render(SR, [NoteEvent(0, 48, 1.0, True), NoteEvent(0, 52, 1.0, True)])[:, 0]
    seg = out[22050:]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    assert spec[(f > 125) & (f < 137)].max() > 5 * spec[(f > 160) & (f < 170)].max()


def test_bass_note_above_split_is_silent():
    s = _synth(
        registers={"cello": True, "viola": False, "violin": False}, ensemble=False, crescendo=0.005
    )
    out = s.render(4096, [NoteEvent(0, 60, 1.0, True)])
    assert np.abs(out).max() == 0.0


def test_horn_overrides_trumpet():
    kw = dict(ensemble=False, crescendo=0.005)
    ev = [NoteEvent(0, 60, 1.0, True)]
    a = _synth(registers={"horn": True, "viola": False, "violin": False}, **kw).render(2048, ev)
    b = _synth(
        registers={"horn": True, "trumpet": True, "viola": False, "violin": False}, **kw
    ).render(2048, ev)
    assert np.allclose(a, b)


def test_notes_outside_keyboard_are_folded():
    s = _synth()
    out = s.render(1024, [NoteEvent(0, 20, 1.0, True), NoteEvent(0, 100, 1.0, True)])
    assert np.isfinite(out).all() and np.abs(out).max() > 0.0


def test_release_then_silence_and_update_patch_keeps_state():
    s = _synth(crescendo=0.005, sustain_length=0.05, ensemble=False)
    s.render(2048, [NoteEvent(0, 60, 1.0, True)])
    s.update_patch(
        patch_from_dict({"name": "t", "kind": "solina", "sustain_length": 0.05, "ensemble": True})
    )
    s.render(2048, [NoteEvent(0, 60, 1.0, False)])
    tail = s.render(SR // 2, [])
    assert np.abs(tail[-1024:]).max() < 1e-3


# ----- intégration -----
from synthwave.audio.renderer import RenderConfig, Renderer  # noqa: E402
from synthwave.patches.loader import list_patches, load_patch, set_param  # noqa: E402


def test_renderer_plays_solina_on_pad_and_live_param():
    r = Renderer(RenderConfig(mood="dark", seed=1, patches={"pad": "solina_strings"}))
    assert isinstance(r.instruments["pad"], SolinaSynth)
    out = r.render(2048)
    assert np.isfinite(out).all()
    p = set_param(r.base_patch["pad"], "crescendo", 0.9)
    assert p.crescendo == 0.9


def test_every_solina_patch_renders():
    names = [n for n in list_patches() if n.startswith("solina_")]
    assert len(names) >= 14
    for name in names:
        s = SolinaSynth(load_patch(name), SR, np.random.default_rng(0), 110)
        out = s.render(8192, [NoteEvent(0, 48, 1.0, True), NoteEvent(0, 64, 1.0, True)])
        assert np.isfinite(out).all() and np.abs(out).max() > 0.005, name
