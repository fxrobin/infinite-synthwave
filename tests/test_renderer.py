import hashlib

import numpy as np
import pytest
import soundfile as sf

from synthwave.audio.export import export_wav
from synthwave.audio.renderer import RenderConfig, Renderer
from synthwave.patches.loader import PatchError


def render_seconds(r, s, block=1024):
    n = int(s * r.sr)
    return np.concatenate([r.render(block) for _ in range(n // block)])


def test_render_shape_finite_and_bounded():
    r = Renderer(RenderConfig(seed=1, mood="outrun"))
    out = render_seconds(r, 3)
    assert out.dtype == np.float32 and out.shape[1] == 2
    assert np.isfinite(out).all() and np.abs(out).max() <= 1.0 and np.abs(out).max() > 0.05


def test_deterministic_by_seed():
    a = render_seconds(Renderer(RenderConfig(seed=42)), 2)
    b = render_seconds(Renderer(RenderConfig(seed=42)), 2)
    assert hashlib.sha1(a.tobytes()).hexdigest() == hashlib.sha1(b.tobytes()).hexdigest()
    c = render_seconds(Renderer(RenderConfig(seed=43)), 2)
    assert not np.array_equal(a, c)


def test_duration_mode_finishes():
    r = Renderer(RenderConfig(seed=3, bpm=140, duration_s=12))
    total = 0
    while not r.finished and total < 60 * r.sr:
        r.render(4096)
        total += 4096
    assert r.finished and 10 * r.sr <= total <= 30 * r.sr


def test_commands_and_status():
    r = Renderer(RenderConfig(seed=5, mood="dark"))
    r.submit(lambda: r.set_tempo(125))
    r.submit(lambda: r.set_layer("lead", mute=True))
    r.submit(lambda: r.set_layer("pad", volume=0.5))
    r.submit(lambda: r.load_patch("bass", "lead_saw"))
    r.render(1024)
    st = r.status()
    assert st["bpm"] == 125 and st["layers"]["lead"]["muted"]
    assert st["layers"]["pad"]["volume"] == 0.5
    assert st["layers"]["bass"]["patch"] == "lead_saw" and st["section"] == "intro"
    assert st["seed"] == 5
    r.set_mood("dreamy")
    assert r.status()["pending_mood"] == "dreamy"
    bars = int(16 * r.transport.samples_per_step)
    for _ in range(9 * bars // 4096 + 2):      # intro (8 bars) then the transition
        r.render(4096)
    assert r.status()["mood"] == "dreamy" and r.status()["section"] == "transition"


def test_bad_patch_keeps_state():
    r = Renderer(RenderConfig(seed=5, mood="dark"))
    with pytest.raises(PatchError):
        r.load_patch("drums", "pad_juno")
    with pytest.raises(PatchError):
        r.set_patch_param("pad", "filter.cutoff", "nope")
    assert r.status()["layers"]["drums"]["patch"] == "drums_dark"
    r.set_patch_param("pad", "filter.cutoff", 700)
    assert r.instruments["pad"].patch.filter.cutoff == 700


def test_export_wav(tmp_path):
    r = Renderer(RenderConfig(seed=2, duration_s=4))
    out = tmp_path / "x.wav"
    n = export_wav(r, 4, str(out))
    data, sr = sf.read(str(out))
    assert sr == 44100 and len(data) == n and data.shape[1] == 2 and n >= 4 * 44100


def test_layer_effects_manual_and_auto():
    r = Renderer(RenderConfig(seed=5))
    r.set_layer_effects("pad", [{"type": "gate", "rate": "1/16"}])
    r.set_layer_effects("master", [{"type": "lofi", "bits": 8}])
    r.render(2048)
    st = r.status()["effects"]
    assert st["pad"][0]["type"] == "gate" and st["master"][0]["type"] == "lofi"
    assert "pad" in r.inserts and "master" in r.inserts
    with pytest.raises(KeyError):
        r.set_layer_effects("pad", [{"type": "nope"}])
    assert r.status()["effects"]["pad"][0]["type"] == "gate"
    r.set_layer_effects("pad", None)
    assert r.status()["effects"]["pad"] == []


def test_mood_patches_swap_on_transition():
    r = Renderer(RenderConfig(seed=5, mood="outrun"))
    assert r.status()["layers"]["pad"]["patch"] == "pad_juno"
    r.load_patch("lead", "lead_saw")
    r.set_mood("dark")
    bars = int(16 * r.transport.samples_per_step)
    for _ in range(9 * bars // 4096 + 2):
        r.render(4096)
    st = r.status()
    assert st["mood"] == "dark" and st["layers"]["pad"]["patch"] == "pad_dark"
    assert st["layers"]["drums"]["patch"] == "drums_dark"
    assert st["layers"]["lead"]["patch"] == "lead_saw"   # manual choice kept


def test_dark_mood_renders():
    r = Renderer(RenderConfig(seed=8, mood="dark"))
    out = render_seconds(r, 3)
    assert np.isfinite(out).all() and np.abs(out).max() > 0.05


def test_export_mp3(tmp_path):
    r = Renderer(RenderConfig(seed=2, duration_s=2))
    out = tmp_path / "x.mp3"
    export_wav(r, 2, str(out))
    data, sr = sf.read(str(out))
    assert sr == 44100 and len(data) >= 2 * 44100 and out.stat().st_size < 200_000


def test_section_patches_applied_by_renderer():
    r = Renderer(RenderConfig(seed=3, mood="dark"))
    seen = set()
    bars = int(16 * r.transport.samples_per_step)
    for _ in range(8):                      # force eight section changes
        r.next_section()
        for _ in range(2 * bars // 4096 + 1):
            r.render(4096)
        seen.add(r.status()["layers"]["bass"]["patch"])
    assert len(seen) >= 2


def test_bpm_drawn_from_range_at_start_and_transitions():
    bpms = {Renderer(RenderConfig(seed=s, mood="dark")).bpm for s in range(6)}
    assert len(bpms) >= 3 and all(82 <= b <= 100 for b in bpms)
    r = Renderer(RenderConfig(seed=4, mood="dark", bpm_range=(70, 75)))
    assert 70 <= r.bpm <= 75 and r.status()["bpm_range"] == [70, 75]
    seen = {r.bpm}
    r.set_mood("noir")                       # forces a transition, which redraws the tempo
    bars = int(16 * r.transport.samples_per_step)
    for _ in range(14 * bars // 4096):
        r.render(4096)
        seen.add(r.bpm)
    assert len(seen) >= 2 and all(70 <= b <= 75 for b in seen)
    assert Renderer(RenderConfig(seed=1, bpm=123)).bpm == 123


def test_riser_layer_is_builtin():
    r = Renderer(RenderConfig(seed=1))
    assert r.status()["layers"]["riser"]["patch"] == "builtin"
    with pytest.raises(PatchError):
        r.load_patch("riser", "pad_juno")


def test_random_mood_at_start_and_at_transitions():
    moods = {Renderer(RenderConfig(seed=s)).mood.name for s in range(8)}
    assert len(moods) >= 3
    r = Renderer(RenderConfig(seed=2))
    assert not r.status()["mood_locked"]
    seen = {r.mood.name}
    bars = int(16 * r.transport.samples_per_step)
    for _ in range(6):
        r.next_section()
        for _ in range(3 * bars // 4096 + 1):
            r.render(4096)
        seen.add(r.status()["mood"])
    assert len(seen) >= 2
    locked = Renderer(RenderConfig(seed=2, mood="outrun"))
    assert locked.status()["mood_locked"] and locked.mood.name == "outrun"
    locked.set_mood("random")
    assert not locked.status()["mood_locked"]
