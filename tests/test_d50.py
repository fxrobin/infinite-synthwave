import numpy as np
import pytest

from synthwave.engine.d50 import (
    D50Synth,
    Env5,
    d50_patch_from_bytes,
    d50_sysex_to_patches,
    env_time,
    partial_hz,
)
from synthwave.engine.d50_pcm import PCM_TABLE, pcm_wave
from synthwave.engine.events import NoteEvent
from synthwave.patches.loader import list_patches, load_patch, patch_from_dict
from synthwave.patches.model import D50Env, D50PatchModel

SR = 44100


def _patch(**kw) -> D50PatchModel:
    return patch_from_dict({"name": "t", "kind": "d50", **kw})


def _synth(patch: D50PatchModel) -> D50Synth:
    return D50Synth(patch, SR, np.random.default_rng(0), 110)


def _peak_hz(sig: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    return float(np.fft.rfftfreq(len(sig), 1 / SR)[np.argmax(spec)])


def _tone(structure=1, **partial):
    base = {"coarse": 36, "keyfollow": 11, "cutoff": 90, "tva_env": {"t": [0, 50, 50, 50, 30]}}
    base.update(partial)
    return {"common": {"structure": structure}, "partials": [dict(base), dict(base)]}


def test_partial_pitch_and_keyfollow():
    p = _patch().upper.partials[0]
    assert abs(partial_hz(p, 69) - 440.0) < 0.01
    p2 = p.model_copy(update={"keyfollow": 7})  # 1/2 octave par octave
    assert abs(partial_hz(p2, 72) / partial_hz(p2, 60) - 2**0.5) < 1e-6


def test_env_times_span_service_notes_range():
    assert abs(env_time(0) - 0.004) < 1e-9 and abs(env_time(100) - 80.0) < 1e-6
    e = Env5(D50Env(t=[50, 0, 0, 0, 0], l=[100, 100, 100], sustain=100), SR)
    e.gate_on()
    y = e.render(int(env_time(50) * SR))
    assert 0.98 < y[-1] <= 1.0 and y[len(y) // 2] < 0.6


def test_saw_plays_one_octave_above_square():
    sq = _synth(_patch(upper=_tone(wave="square", pw=0)))
    sw = _synth(_patch(upper=_tone(wave="saw")))
    ev = [NoteEvent(0, 60, 1.0, True)]
    a = sq.render(SR // 2, ev)[SR // 4 :, 0]
    b = sw.render(SR // 2, ev)[SR // 4 :, 0]
    assert abs(_peak_hz(a) - 261.6) < 4
    assert abs(_peak_hz(b) - 523.3) < 6


def test_low_cutoff_removes_harmonics():
    bright = _synth(_patch(upper=_tone(wave="square", cutoff=100)))
    dark = _synth(_patch(upper=_tone(wave="square", cutoff=40)))
    ev = [NoteEvent(0, 48, 1.0, True)]

    def hf_ratio(s):
        sig = s.render(SR // 2, ev)[SR // 4 :, 0]
        spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        f = np.fft.rfftfreq(len(sig), 1 / SR)
        return spec[f > 1000].sum() / spec.sum()

    assert hf_ratio(dark) < 0.2 * hf_ratio(bright)


def test_ring_modulation_creates_sum_and_difference():
    t = _tone(structure=2, wave="square", cutoff=60, pw=0)
    t["partials"][1]["coarse"] = 43  # 7 demi-tons au-dessus
    s = _synth(_patch(upper=t))
    sig = s.render(SR, [NoteEvent(0, 60, 1.0, True)])[SR // 2 :, 0]
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    f = np.fft.rfftfreq(len(sig), 1 / SR)
    f1, f2 = 261.6, 261.6 * 2 ** (7 / 12)
    band = lambda x: spec[(f > x - 6) & (f < x + 6)].max()  # noqa: E731
    assert band(f1 + f2) > 0.05 * band(f1) and band(f2 - f1) > 0.02 * band(f1)


def test_pcm_oneshot_stops_and_loop_sustains():
    assert len(PCM_TABLE) == 100
    assert not pcm_wave(17)[1] and pcm_wave(48)[1]
    one = _tone(structure=6, pcm=17, coarse=12, keyfollow=11)
    loop = _tone(structure=6, pcm=48, coarse=12, keyfollow=11)
    ev = [NoteEvent(0, 60, 1.0, True)]
    a = _synth(_patch(upper=one)).render(SR * 2, ev)[:, 0]
    b = _synth(_patch(upper=loop)).render(SR * 2, ev)[:, 0]
    assert np.abs(a[: SR // 4]).max() > 0.01 and np.abs(a[-SR // 4 :]).max() < 1e-4
    assert np.abs(b[-SR // 4 :]).max() > 0.01


def test_key_modes_route_tones():
    lower = _tone(wave="square")
    lower["partials"][0]["coarse"] = 24
    lower["partials"][1]["coarse"] = 24
    p = _patch(upper=_tone(wave="square"), lower=lower, key_mode=2, split=24)  # split à C4
    s = _synth(p)
    s.render(64, [NoteEvent(0, 48, 1.0, True), NoteEvent(0, 72, 1.0, True)])
    assert [v.note for v in s.notes["lower"]] == [48] and [v.note for v in s.notes["upper"]] == [72]
    s = _synth(_patch(upper=_tone(), lower=lower, key_mode=1))
    s.render(64, [NoteEvent(0, 60, 1.0, True)])
    assert len(s.notes["upper"]) == 1 and len(s.notes["lower"]) == 1


def test_polyphony_limit_steals_oldest():
    s = _synth(_patch(upper=_tone(), polyphony=2))
    s.render(
        64, [NoteEvent(0, 60, 1.0, True), NoteEvent(0, 64, 1.0, True), NoteEvent(0, 67, 1.0, True)]
    )
    assert [v.note for v in s.notes["upper"]] == [64, 67]


def test_sysex_roundtrip_of_synthetic_patch():
    body = bytearray(448)
    body[6 * 64 : 6 * 64 + 8] = bytes([6, 27, 40, 46, 27, 45, 35, 27])  # "Fantasia"
    body[6 * 64 + 18] = 1  # DUAL
    body[6 * 64 + 30] = 24  # reverb 25
    body[2 * 64 + 10] = 3  # structure 4
    body[7] = 12  # PCM 13
    body[6] = 1  # saw
    p = d50_patch_from_bytes(bytes(body))
    assert p.name == "Fantasia" and p.key_mode == 1 and p.reverb_type == 25
    assert p.upper.common.structure == 4 and p.upper.partials[0].pcm == 13
    assert p.upper.partials[0].wave == "saw"
    # message DT1 avec checksum valide
    addr = (0x02, 0x00, 0x00)
    data = bytes(body[:256])
    cs = (128 - (sum(addr) + sum(data)) % 128) % 128
    msg = bytes([0xF0, 0x41, 0x00, 0x14, 0x12, *addr, *data, cs, 0xF7])
    with pytest.raises(ValueError):
        d50_sysex_to_patches(msg[:-2] + bytes([(cs + 1) % 128, 0xF7]))


def test_library_d50_patches_render():
    names = [n for n in list_patches() if n.startswith("d50_")]
    assert len(names) >= 64
    for name in names[:: max(1, len(names) // 16)]:
        s = D50Synth(load_patch(name), SR, np.random.default_rng(0), 110)
        out = s.render(SR, [NoteEvent(0, 48, 0.8, True), NoteEvent(0, 64, 0.8, True)])
        assert np.isfinite(out).all() and np.abs(out).max() > 1e-4, name


def test_renderer_plays_d50():
    from synthwave.audio.renderer import RenderConfig, Renderer

    r = Renderer(RenderConfig(mood="dark", seed=1, patches={"pad": "d50_Fantasia"}))
    assert isinstance(r.instruments["pad"], D50Synth)
    assert np.isfinite(r.render(2048)).all()
