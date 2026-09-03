import pytest

from synthwave.patches.loader import (
    PatchError,
    list_patches,
    load_patch,
    patch_from_dict,
    set_param,
)
from synthwave.patches.model import DrumPatchModel, PatchModel


def test_library_lists_and_loads_all():
    names = list_patches()
    expected = {"bass_moog", "pad_juno", "arp_pluck", "lead_saw", "ambient_drone", "drums_808"}
    assert expected <= set(names)
    for n in names:
        p = load_patch(n)
        assert isinstance(p, (PatchModel, DrumPatchModel))


def test_drum_patch_discriminated():
    assert isinstance(load_patch("drums_808"), DrumPatchModel)
    assert isinstance(load_patch("pad_juno"), PatchModel)


def test_invalid_patch_raises_patch_error():
    with pytest.raises(PatchError):
        patch_from_dict({"name": "bad", "oscillators": [{"wave": "laser"}]})
    with pytest.raises(PatchError):
        load_patch("does_not_exist")


def test_set_param_returns_new_validated_patch():
    p = load_patch("pad_juno")
    q = set_param(p, "filter.cutoff", 500)
    assert q.filter.cutoff == 500 and p.filter.cutoff != 500
    q = set_param(p, "oscillators.0.detune", 3)
    assert q.oscillators[0].detune == 3
    with pytest.raises(PatchError):
        set_param(p, "oscillators.0.wave", "laser")


def test_load_from_path(tmp_path):
    f = tmp_path / "x.yaml"
    f.write_text(
        "name: x\noscillators:\n  - wave: sine\n"
        "amp_env: {attack: 0.01, decay: 0.1, sustain: 0.5, release: 0.2}\n"
    )
    assert load_patch(str(f)).name == "x"


def test_every_library_patch_is_in_a_pool_and_every_pool_patch_exists():
    from synthwave.composer.moods import MOODS
    from synthwave.patches.loader import list_patches

    pooled = {name for m in MOODS.values() for names in m.pools.values() for name in names}
    library = set(list_patches())
    assert pooled <= library
    assert library <= pooled | {n for m in MOODS.values() for n in m.patches.values()}
    assert len(library) >= 51


def test_apply_tweaks_multiplies_and_clamps():
    from synthwave.patches.loader import apply_tweaks

    pad = load_patch("pad_juno")
    t = apply_tweaks(
        pad,
        {
            "filter.cutoff": 0.5,
            "filter.resonance": 50.0,
            "oscillators.0.detune": 2.0,
            "nope.path": 3.0,
            "name": 2.0,
        },
    )
    assert t.filter.cutoff == pad.filter.cutoff * 0.5
    assert t.filter.resonance == 1.0  # clamped
    assert t.oscillators[0].detune == pad.oscillators[0].detune * 2
    assert t.name == pad.name and apply_tweaks(pad, {}) is pad


def test_straight_moods_use_dry_kits_without_delay():
    """Eighties moods: pure percussion, no echo on the drum bus."""
    from synthwave.composer.moods import MOODS

    for mood in (m for m in MOODS.values() if m.straight):
        for name in mood.pools.get("drums", []):
            kit = load_patch(name)
            types = [e.type for e in kit.perc_effects]
            assert "delay" not in types, f"{mood.name}: {name} has a delay on the percussion"
