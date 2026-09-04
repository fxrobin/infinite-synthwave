# Solina String Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un moteur `SolinaSynth` fidèle au circuit du Solina (divide-down, keyers RC, bus de registres, basse mono, ensemble triple BBD), patches `kind: solina`, bank de 14 patches, intégration renderer/UI/moods/README.

**Architecture:** Nouveau module `synthwave/engine/solina.py` exposant la même interface que `Dx7Synth`. Modèle pydantic `SolinaPatchModel` routé par `loader.patch_from_dict`. Le renderer instancie `SolinaSynth` par `isinstance`. L'ensemble est aussi enregistré comme effet `ensemble` dans `_REGISTRY`.

**Tech Stack:** Python ≥ 3.13, numpy/scipy, pydantic, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-04-solina-string-ensemble-design.md`

## Global Constraints

- Pas de chemin absolu machine dans les fichiers versionnés.
- `uv run ruff check .` propre (line-length 100), `uv run pytest -q` vert.
- Tout patch de la bibliothèque doit être dans un pool de mood (`test_every_library_patch_is_in_a_pool`).
- Nouveau patch : mesurer peak/RMS contre `pad_strings` (6 s) avant de fixer `volume`.

---

### Task 1: Modèle de patch `SolinaPatchModel` + loader

**Files:**
- Modify: `synthwave/patches/model.py` (avant `AnyPatch`)
- Modify: `synthwave/patches/loader.py:74-80`
- Test: `tests/test_solina.py`

**Interfaces:**
- Produces: `SolinaRegisters(violin, viola, trumpet, horn, cello, contrabass: bool)`,
  `SolinaPatchModel(name, kind="solina", registers, crescendo, sustain_length, ensemble,
  stereo, bass_volume, split_note, tune, volume, effects)`.

- [ ] **Step 1: test**

```python
from synthwave.patches.loader import patch_from_dict
from synthwave.patches.model import SolinaPatchModel

def test_solina_patch_model_roundtrip():
    p = patch_from_dict({"name": "s", "kind": "solina", "registers": {"cello": True}})
    assert isinstance(p, SolinaPatchModel)
    assert p.registers.cello and p.registers.viola and not p.registers.horn
    assert p.crescendo == 0.3 and p.sustain_length == 0.8 and p.ensemble
```

- [ ] **Step 2: run, expect ImportError**
- [ ] **Step 3: implémenter les deux classes (voir spec) + `if data.get("kind") == "solina"` dans `patch_from_dict` + `AnyPatch |= SolinaPatchModel`**
- [ ] **Step 4: run, PASS ; commit `feat(solina): modèle de patch`**

### Task 2: Générateur divide-down + keyer RC

**Files:**
- Create: `synthwave/engine/solina.py`
- Test: `tests/test_solina.py`

**Interfaces:**
- Produces: `note_hz(midi: int, tune_cents=0.0) -> float`, `staircase_saw(phase: np.ndarray) -> np.ndarray`,
  `RcKeyer(sr, n_keys=49, attack_s, release_s)` avec `gate_on(i)`, `gate_off(i)`,
  `render(n) -> np.ndarray[(n, n_keys)]` (niveau par touche), `active -> np.ndarray[bool]`,
  `set_times(attack_s, release_s)`.

- [ ] **Step 1: tests**

```python
def test_note_hz_is_exact_octave_division():
    assert abs(note_hz(69) - 440.0) < 1e-9
    assert note_hz(57) == note_hz(69) / 2 and note_hz(45) == note_hz(69) / 4

def test_staircase_saw_has_four_levels_per_period():
    ph = np.linspace(0, 1, 64, endpoint=False)
    assert len(np.unique(np.round(staircase_saw(ph), 6))) == 16  # 4 bits

def test_rc_keyer_attack_time():
    k = RcKeyer(44100, attack_s=0.5, release_s=0.2)
    k.gate_on(3)
    lvl = k.render(44100)[:, 3]
    # RC : 63 % à τ
    assert 0.55 < lvl[22050] < 0.72 and lvl[-1] > 0.85

def test_rc_keyer_retrigger_keeps_level():
    k = RcKeyer(44100, attack_s=0.01, release_s=1.0)
    k.gate_on(0); k.render(4410); k.gate_off(0); k.render(4410)
    before = k.render(1)[0, 0]
    k.gate_on(0)
    after = k.render(1)[0, 0]
    assert after >= before
```

- [ ] **Step 2: run, FAIL** — [ ] **Step 3: implémenter** — [ ] **Step 4: PASS, commit**

### Task 3: `SolinaEnsemble` (triple BBD) + registre effet `ensemble`

**Files:**
- Modify: `synthwave/engine/solina.py`
- Modify: `synthwave/engine/effects.py:545` (`_REGISTRY["ensemble"]`), `synthwave/patches/model.py` (`EffectSpec.type` Literal)
- Test: `tests/test_solina.py`

**Interfaces:**
- Produces: `SolinaEnsemble(sr, bpm, chorus_rate=0.6, chorus_depth=0.0015, vibrato_rate=6.0,
  vibrato_depth=0.00015, base_delay=0.005, stereo=True, bandwidth=6000.0)`, `process(x[(n,2)]) -> (n,2) float32`.

- [ ] **Step 1: tests**

```python
def test_ensemble_widens_spectrum_and_mono_option():
    sr = 44100; t = np.arange(sr * 2) / sr
    x = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, 1).astype(np.float32)
    ens = SolinaEnsemble(sr, 110, stereo=False)
    y = np.concatenate([ens.process(x[i:i + 1024]) for i in range(0, len(x), 1024)])
    assert np.allclose(y[:, 0], y[:, 1])
    def spread(sig):
        spec = np.abs(np.fft.rfft(sig[sr:] * np.hanning(sr)))
        f = np.fft.rfftfreq(sr, 1 / sr); band = (f > 430) & (f < 450)
        return spec[band & ~((f > 439) & (f < 441))].sum() / spec[band].sum()
    assert spread(y[:, 0]) > 3 * spread(x[:, 0])
    assert "ensemble" in _REGISTRY
```

- [ ] **Step 2–4: FAIL → implémenter → PASS, commit**

### Task 4: `SolinaSynth` (bus de registres, basse mono, rendu)

**Files:**
- Modify: `synthwave/engine/solina.py`
- Test: `tests/test_solina.py`

**Interfaces:**
- Produces: `SolinaSynth(patch, sr, rng, bpm)` : `render(n, events, gain=None)`, `set_patch`,
  `update_patch`, `set_bpm`, `note_on(note, vel)`, `note_off(note)`, `voices = []`.

- [ ] **Step 1: tests**

```python
def _synth(**kw):
    return SolinaSynth(patch_from_dict({"name": "t", "kind": "solina", **kw}), 44100, np.random.default_rng(0), 110)

def test_solina_renders_finite_stereo():
    out = _synth().render(4096, [NoteEvent(0, 60, 1.0, True), NoteEvent(0, 64, 1.0, True)])
    assert out.shape == (4096, 2) and np.isfinite(out).all() and np.abs(out).max() > 0.01

def test_bass_section_is_mono_lowest_note():
    s = _synth(registers={"cello": True, "viola": False, "violin": False}, ensemble=False, crescendo=0.005)
    out = s.render(44100, [NoteEvent(0, 48, 1.0, True), NoteEvent(0, 52, 1.0, True)])[:, 0]
    spec = np.abs(np.fft.rfft(out[22050:] * np.hanning(22050))); f = np.fft.rfftfreq(22050, 1 / 44100)
    assert spec[(f > 125) & (f < 137)].max() > 5 * spec[(f > 160) & (f < 170)].max()

def test_horn_overrides_trumpet():
    kw = dict(ensemble=False, crescendo=0.005)
    a = _synth(registers={"horn": True, "viola": False, "violin": False}, **kw).render(2048, [NoteEvent(0, 60, 1.0, True)])
    b = _synth(registers={"horn": True, "trumpet": True, "viola": False, "violin": False}, **kw).render(2048, [NoteEvent(0, 60, 1.0, True)])
    assert np.allclose(a, b)

def test_notes_outside_keyboard_are_clamped_silently():
    out = _synth().render(1024, [NoteEvent(0, 20, 1.0, True), NoteEvent(0, 100, 1.0, True)])
    assert np.isfinite(out).all()
```

- [ ] **Step 2–4: FAIL → implémenter → PASS, commit**

### Task 5: Intégration renderer + web + moods + bank

**Files:**
- Modify: `synthwave/audio/renderer.py:17,128,188-200,445-447`
- Modify: `synthwave/web/static/index.html:226-230,366-367,293-300`
- Modify: `synthwave/composer/moods.py` (pools pad/ambient/lead2)
- Create: `synthwave/patches/library/solina_*.yaml` (14, voir spec)
- Test: `tests/test_solina.py`, `tests/test_patches.py` (existant)

- [ ] **Step 1: tests**

```python
def test_renderer_plays_solina_on_pad():
    r = Renderer(RenderConfig(mood="dark", seed=1, patches={"pad": "solina_strings"}))
    out = r.render(2048)
    assert np.isfinite(out).all()

def test_every_solina_patch_renders():
    for name in list_patches():
        if name.startswith("solina_"):
            s = SolinaSynth(load_patch(name), 44100, np.random.default_rng(0), 110)
            out = s.render(8192, [NoteEvent(0, 48, 1.0, True), NoteEvent(0, 64, 1.0, True)])
            assert np.isfinite(out).all() and np.abs(out).max() > 0.005, name
```

- [ ] **Step 2: implémenter ; calibrer `volume` : rendre 6 s de `solina_strings` et `pad_strings` sur un accord, aligner RMS à ±1 dB**
- [ ] **Step 3: suite complète + ruff, commit**

### Task 6: README + CLAUDE.md

- [ ] Section « Solina String Ensemble fidèle » dans README (moteur, tableau du bank, YAML, sources) ; ligne Moteur dans CLAUDE.md ; commit.
