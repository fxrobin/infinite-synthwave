import numpy as np

from synthwave.composer.harmony import Chord
from synthwave.composer.patterns import (
    STEPS,
    gen_ambient,
    gen_arp,
    gen_bass,
    gen_drums,
    gen_lead,
    gen_pad,
    mutate,
)

AM7 = Chord(9, 0, (0, 3, 7, 10))


def in_range(p):
    return all(0 <= n.step < STEPS and 0 < n.length <= STEPS and 0 < n.vel <= 1 for n in p)


def test_drums_basic_grid():
    p = gen_drums(np.random.default_rng(0), 0.5)
    kicks = {n.step for n in p if n.note == 36}
    snares = {n.step for n in p if n.note in (38, 39)}
    assert {0, 4, 8, 12} <= kicks and snares == {4, 12} and in_range(p)


def test_drums_fill_adds_hits_at_end():
    p = gen_drums(np.random.default_rng(0), 0.5, fill=True)
    assert any(n.step >= 12 and n.note in (38, 45, 47) for n in p)
    c = gen_drums(np.random.default_rng(0), 0.5, crash=True)
    assert any(n.note == 49 and n.step == 0 for n in c)


def test_bass_styles_use_chord_root():
    for style in ("eighths", "octaves", "syncopated"):
        p = gen_bass(np.random.default_rng(1), AM7, style)
        assert p and in_range(p) and all(n.note % 12 in (9, 4) for n in p)


def test_arp_uses_chord_tones_every_step():
    p = gen_arp(np.random.default_rng(2), AM7, "updown")
    assert len(p) == STEPS and all(n.note % 12 in {9, 0, 4, 7} for n in p)
    assert [n.note for n in gen_arp(np.random.default_rng(0), AM7, "up")][:4] == [57, 60, 64, 67]


def test_pad_and_ambient_hold_whole_bar():
    pad = gen_pad(AM7)
    assert all(n.step == 0 and n.length == STEPS for n in pad) and len(pad) == 4
    amb = gen_ambient(AM7)
    assert len(amb) == 2 and all(n.note < 60 for n in amb)


def test_lead_notes_in_scale_and_sorted():
    scale = [n for n in range(72, 85) if n % 12 in {9, 11, 0, 2, 4, 5, 7}]
    p = gen_lead(np.random.default_rng(4), AM7, scale, 0.8)
    assert p and in_range(p) and all(n.note in scale for n in p)
    assert [n.step for n in p] == sorted(n.step for n in p)


def test_mutate_is_bounded_and_keeps_allowed_notes():
    base = gen_arp(np.random.default_rng(0), AM7, "up")
    allowed = [57, 60, 64, 67, 69, 72, 76, 79]
    m = mutate(np.random.default_rng(5), base, 0.2, allowed)
    assert in_range(m) and all(n.note in allowed for n in m)
    changed = len(set(base) ^ set(m))
    assert 0 < changed <= 12
