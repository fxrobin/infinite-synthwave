# Infinite Synthwave Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Générateur synthwave infini (ou à durée fixe) sur la sortie audio, synthés YAML programmables, tracker procédural, CLI + serveur MCP.

**Architecture:** Moteur DSP numpy par blocs (oscillateurs polyBLEP, ADSR, biquad scipy, effets à lignes de retard vectorisées). Composer (harmonie Markov + générateurs de patterns + arrangeur par sections) alimente un tracker 16 pas qui émet des événements aux samples exacts. Un renderer mixe les couches, applique sidechain et limiteur, et sert un thread producteur consommé par le callback sounddevice ou l'export WAV.

**Tech Stack:** Python 3.13, uv, numpy, scipy, sounddevice, soundfile, pyyaml, pydantic v2, mcp (FastMCP), typer, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-02-infinite-synthwave-design.md`

## Global Constraints

- Python ≥ 3.13, sample rate 44 100 Hz, stéréo float32, blocs de 1024 samples par défaut.
- Tout le DSP est vectorisé numpy/scipy : aucune boucle Python par sample.
- Rendu déterministe pour une seed donnée (un seul `numpy.random.Generator` propagé).
- 16 pas par mesure, 4/4 ; couches : `drums, bass, arp, pad, lead, ambient`.
- Patches YAML validés par pydantic ; un patch invalide ne modifie jamais l'état courant.
- Commits fréquents, messages Conventional Commits, `pytest` vert avant chaque commit.

## File Structure

```
pyproject.toml
synthwave/__init__.py
synthwave/engine/__init__.py
synthwave/engine/oscillator.py     polyBLEP saw/square, tri, sine, noise, unison/detune/spread
synthwave/engine/envelope.py       ADSR par segments vectorisés
synthwave/engine/filter.py         biquad RBJ lp/hp/bp via scipy.signal.lfilter avec état
synthwave/engine/lfo.py            LFO sine/tri/square/saw
synthwave/engine/effects.py        Chorus, Delay, Reverb, GatedReverb, Sidechain, Limiter, build_effects
synthwave/engine/voice.py          Voice = oscs → filtre → amp
synthwave/engine/synth.py          Synth polyphonique + chaîne d'effets, rendu segmenté par événements
synthwave/engine/drums.py          DrumKit : one-shots synthétisés à l'init
synthwave/patches/__init__.py
synthwave/patches/model.py         modèles pydantic PatchModel / DrumPatchModel
synthwave/patches/loader.py        list_patches, load_patch, PatchError
synthwave/patches/library/*.yaml   bass_moog, pad_juno, arp_pluck, lead_saw, ambient_drone, drums_808
synthwave/composer/__init__.py
synthwave/composer/moods.py        Mood dataclass + MOODS
synthwave/composer/harmony.py      Chord, Harmony (Markov)
synthwave/composer/patterns.py     Note, gen_* , mutate
synthwave/composer/arranger.py     Section, BarPlan, Arranger
synthwave/sequencer/__init__.py
synthwave/sequencer/transport.py   Transport, StepTick
synthwave/sequencer/tracker.py     NoteEvent, Tracker
synthwave/audio/__init__.py
synthwave/audio/renderer.py        Renderer (mix, sidechain, limiter, commandes)
synthwave/audio/output.py          Player (thread producteur + sounddevice)
synthwave/audio/export.py          export_wav
synthwave/cli.py                   typer
synthwave/mcp_server.py            FastMCP
tests/test_*.py
```

---

### Task 1: Scaffolding + Oscillator

**Files:**
- Create: `pyproject.toml`, `synthwave/__init__.py`, `synthwave/engine/__init__.py`, `synthwave/engine/oscillator.py`
- Test: `tests/test_oscillator.py`

**Interfaces:**
- Produces: `render_wave(wave: str, phase: np.ndarray, dt: float, pwm: float = 0.5, rng=None) -> np.ndarray` ; `Oscillator(wave, sr, rng, unison=1, detune=0.0, octave=0, semi=0, level=1.0, pwm=0.5, spread=1.0)` avec `render(freq_hz: float, n: int, pwm: float | None = None) -> np.ndarray (n, 2) float32`.

- [ ] **Step 1: pyproject**

```toml
[project]
name = "infinite-synthwave"
version = "0.1.0"
description = "Infinite procedural synthwave generator with programmable synths, CLI and MCP server"
requires-python = ">=3.13"
dependencies = [
  "numpy>=2.0", "scipy>=1.14", "sounddevice>=0.5", "soundfile>=0.12",
  "pyyaml>=6.0", "pydantic>=2.7", "mcp>=1.2", "typer>=0.12",
]
[project.scripts]
synthwave = "synthwave.cli:app"
[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
packages = ["synthwave"]
[tool.ruff]
line-length = 100
[tool.pytest.ini_options]
testpaths = ["tests"]
```

Run: `uv sync` — Expected: `.venv` créé, dépendances installées.

- [ ] **Step 2: Test oscillateur**

```python
# tests/test_oscillator.py
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
```

- [ ] **Step 3: Run** `uv run pytest tests/test_oscillator.py -v` — Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 4: Implémentation**

```python
# synthwave/engine/oscillator.py
"""Band-limited oscillators (polyBLEP) with unison, detune and stereo spread."""
from __future__ import annotations
import numpy as np

WAVES = ("saw", "square", "triangle", "sine", "noise")


def _polyblep(t: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(t)
    m = t < dt
    x = t[m] / dt
    out[m] = x + x - x * x - 1.0
    m2 = t > 1.0 - dt
    x = (t[m2] - 1.0) / dt
    out[m2] = x * x + x + x + 1.0
    return out


def render_wave(wave: str, phase: np.ndarray, dt: float, pwm: float = 0.5, rng=None) -> np.ndarray:
    """Render one cycle-normalised waveform from phase in [0,1). dt = phase increment/sample."""
    if wave == "sine":
        return np.sin(2.0 * np.pi * phase)
    if wave == "saw":
        return (2.0 * phase - 1.0) - _polyblep(phase, dt)
    if wave == "square":
        pwm = float(np.clip(pwm, 0.05, 0.95))
        y = np.where(phase < pwm, 1.0, -1.0)
        y = y + _polyblep(phase, dt)
        y = y - _polyblep((phase + 1.0 - pwm) % 1.0, dt)
        return y
    if wave == "triangle":
        return 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
    if wave == "noise":
        rng = rng or np.random.default_rng()
        return rng.uniform(-1.0, 1.0, size=phase.shape)
    raise ValueError(f"unknown wave {wave!r}")


class Oscillator:
    def __init__(self, wave: str, sr: int, rng: np.random.Generator, unison: int = 1,
                 detune: float = 0.0, octave: int = 0, semi: int = 0, level: float = 1.0,
                 pwm: float = 0.5, spread: float = 1.0):
        if wave not in WAVES:
            raise ValueError(f"unknown wave {wave!r}")
        self.wave, self.sr, self.rng = wave, sr, rng
        self.unison = max(1, int(unison))
        self.level, self.pwm = float(level), float(pwm)
        self.transpose = 2.0 ** ((octave * 12 + semi) / 12.0)
        self.phases = rng.uniform(0.0, 1.0, self.unison)
        if self.unison > 1:
            cents = np.linspace(-1.0, 1.0, self.unison) * detune
            pans = np.linspace(-1.0, 1.0, self.unison) * spread
        else:
            cents, pans = np.zeros(1), np.zeros(1)
        self.ratios = 2.0 ** (cents / 1200.0)
        angle = (pans + 1.0) * np.pi / 4.0
        self.gain_l, self.gain_r = np.cos(angle), np.sin(angle)
        self.norm = self.level / np.sqrt(self.unison)

    def render(self, freq_hz: float, n: int, pwm: float | None = None) -> np.ndarray:
        out = np.zeros((n, 2), dtype=np.float64)
        pwm = self.pwm if pwm is None else pwm
        idx = np.arange(1, n + 1)
        for i in range(self.unison):
            dt = freq_hz * self.transpose * self.ratios[i] / self.sr
            phase = (self.phases[i] + dt * idx) % 1.0
            self.phases[i] = phase[-1]
            y = render_wave(self.wave, phase, dt, pwm, self.rng)
            out[:, 0] += y * self.gain_l[i]
            out[:, 1] += y * self.gain_r[i]
        return (out * self.norm).astype(np.float32)
```

- [ ] **Step 5: Run** `uv run pytest tests/test_oscillator.py -v` — Expected: PASS.
- [ ] **Step 6: Commit** `git add -A && git commit -m "feat(engine): polyBLEP oscillator with unison"`

### Task 2: ADSR + LFO

**Files:**
- Create: `synthwave/engine/envelope.py`, `synthwave/engine/lfo.py`
- Test: `tests/test_envelope.py`

**Interfaces:**
- Produces: `ADSR(attack, decay, sustain, release, sr)` : `gate_on()`, `gate_off()`, `render(n) -> np.ndarray (n,) float32`, `finished: bool`, `level: float`. `LFO(wave, rate_hz, sr, phase=0.0)` : `render(n) -> np.ndarray (n,) in [-1,1]`.

- [ ] **Step 1: Tests**

```python
# tests/test_envelope.py
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
    env.gate_on(); env.render(10)
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
    assert np.sum(np.diff(np.sign(y)) != 0) == 4
```

- [ ] **Step 2: Run** `uv run pytest tests/test_envelope.py -v` — Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# synthwave/engine/envelope.py
"""ADSR envelope rendered per block, segment by segment (no per-sample Python loop)."""
from __future__ import annotations
import numpy as np

IDLE, ATTACK, DECAY, SUSTAIN, RELEASE = range(5)


class ADSR:
    def __init__(self, attack: float, decay: float, sustain: float, release: float, sr: int):
        self.a = max(1, int(attack * sr))
        self.d = max(1, int(decay * sr))
        self.s = float(np.clip(sustain, 0.0, 1.0))
        self.r = max(1, int(release * sr))
        self.stage, self.t, self.start, self.level = IDLE, 0, 0.0, 0.0

    @property
    def finished(self) -> bool:
        return self.stage == IDLE

    def gate_on(self) -> None:
        self.stage, self.t, self.start = ATTACK, 0, self.level

    def gate_off(self) -> None:
        if self.stage != IDLE:
            self.stage, self.t, self.start = RELEASE, 0, self.level

    def render(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.float32)
        filled = 0
        while filled < n:
            k = n - filled
            if self.stage == IDLE:
                out[filled:] = 0.0
                self.level = 0.0
                break
            if self.stage == SUSTAIN:
                out[filled:] = self.s
                self.level = self.s
                break
            length = {ATTACK: self.a, DECAY: self.d, RELEASE: self.r}[self.stage]
            k = min(k, length - self.t)
            ts = self.t + np.arange(1, k + 1)
            if self.stage == ATTACK:
                seg = self.start + (1.0 - self.start) * ts / length
            elif self.stage == DECAY:
                seg = self.s + (1.0 - self.s) * np.exp(-5.0 * ts / length)
            else:
                seg = self.start * np.exp(-6.0 * ts / length)
            out[filled:filled + k] = seg
            self.level = float(seg[-1])
            self.t += k
            filled += k
            if self.t >= length:
                self.t = 0
                if self.stage == ATTACK:
                    self.stage, self.level = DECAY, 1.0
                elif self.stage == DECAY:
                    self.stage, self.level = SUSTAIN, self.s
                else:
                    self.stage, self.level = IDLE, 0.0
        return out
```

```python
# synthwave/engine/lfo.py
from __future__ import annotations
import numpy as np


class LFO:
    def __init__(self, wave: str, rate_hz: float, sr: int, phase: float = 0.0):
        self.wave, self.rate, self.sr, self.phase = wave, float(rate_hz), sr, float(phase)

    def render(self, n: int) -> np.ndarray:
        dt = self.rate / self.sr
        ph = (self.phase + dt * np.arange(1, n + 1)) % 1.0
        self.phase = float(ph[-1])
        if self.wave == "sine":
            return np.sin(2 * np.pi * ph)
        if self.wave == "triangle":
            return 2.0 * np.abs(2.0 * ph - 1.0) - 1.0
        if self.wave == "square":
            return np.where(ph < 0.5, 1.0, -1.0)
        if self.wave == "saw":
            return 2.0 * ph - 1.0
        raise ValueError(f"unknown lfo wave {self.wave!r}")
```

- [ ] **Step 4: Run** `uv run pytest tests/test_envelope.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -am "feat(engine): ADSR envelope and LFO"` (après `git add -A`).

### Task 3: Filtre biquad

**Files:**
- Create: `synthwave/engine/filter.py`
- Test: `tests/test_filter.py`

**Interfaces:**
- Produces: `biquad_coeffs(kind, cutoff_hz, resonance, sr) -> (b, a)` ; `Filter(kind, sr)` : `process(x: (n,2), cutoff_hz: float, resonance: float) -> (n,2) float32`.

- [ ] **Step 1: Tests**

```python
# tests/test_filter.py
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
```

- [ ] **Step 2: Run** `uv run pytest tests/test_filter.py -v` — Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# synthwave/engine/filter.py
"""RBJ biquad filter, coefficients per block, state carried by scipy lfilter."""
from __future__ import annotations
import numpy as np
from scipy.signal import lfilter


def biquad_coeffs(kind: str, cutoff_hz: float, resonance: float, sr: int):
    fc = float(np.clip(cutoff_hz, 20.0, sr * 0.45))
    q = 0.5 + float(np.clip(resonance, 0.0, 1.0)) * 9.5
    w0 = 2.0 * np.pi * fc / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    if kind == "lp":
        b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
    elif kind == "hp":
        b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
    elif kind == "bp":
        b = np.array([alpha, 0.0, -alpha])
    else:
        raise ValueError(f"unknown filter kind {kind!r}")
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


class Filter:
    def __init__(self, kind: str, sr: int):
        self.kind, self.sr = kind, sr
        self.zi = [np.zeros(2), np.zeros(2)]

    def process(self, x: np.ndarray, cutoff_hz: float, resonance: float) -> np.ndarray:
        b, a = biquad_coeffs(self.kind, cutoff_hz, resonance, self.sr)
        y = np.empty_like(x, dtype=np.float32)
        for ch in (0, 1):
            yc, self.zi[ch] = lfilter(b, a, x[:, ch].astype(np.float64), zi=self.zi[ch])
            y[:, ch] = yc
        return y
```

- [ ] **Step 4: Run** `uv run pytest tests/test_filter.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat(engine): biquad filter"`

### Task 4: Effets (Chorus, Delay, Reverb, GatedReverb, Sidechain, Limiter)

**Files:**
- Create: `synthwave/engine/effects.py`
- Test: `tests/test_effects.py`

**Interfaces:**
- Consumes: `LFO` (Task 2).
- Produces: `note_to_seconds(value: float | str, bpm: float) -> float` ; classe de base `Effect.process(x: (n,2)) -> (n,2) float32` ; `Chorus(sr, bpm, rate=0.5, depth=0.003, mix=0.4)`, `Delay(sr, bpm, time="1/8", feedback=0.4, mix=0.3, pingpong=True)`, `Reverb(sr, bpm, size=0.8, damping=0.5, mix=0.3, predelay=0.02)`, `GatedReverb(sr, bpm, size=0.85, hold=0.25, threshold=0.1, mix=0.5)`, `Limiter(sr, bpm, threshold=0.9, release=0.1)` ; `Sidechain(sr, depth=0.5, release=0.25)` avec `gain(n, triggers: list[int]) -> (n,)` ; `build_effects(specs: list[dict], sr, bpm) -> list[Effect]`.

- [ ] **Step 1: Tests**

```python
# tests/test_effects.py
import numpy as np
from synthwave.engine.effects import (Chorus, Delay, GatedReverb, Limiter, Reverb, Sidechain,
                                      build_effects, note_to_seconds)

SR = 44100

def impulse(n=SR, at=0, amp=1.0):
    x = np.zeros((n, 2), np.float32); x[at] = amp; return x

def test_note_to_seconds():
    assert np.isclose(note_to_seconds("1/8", 120), 0.25)
    assert np.isclose(note_to_seconds("1/8d", 120), 0.375)
    assert np.isclose(note_to_seconds(0.3, 120), 0.3)

def test_delay_echo_position():
    d = Delay(SR, 120, time="1/8", feedback=0.0, mix=1.0, pingpong=False)
    y = d.process(impulse())
    assert np.argmax(y[:, 0]) == int(0.25 * SR)

def test_delay_handles_blocks_longer_than_delay_time():
    d = Delay(SR, 120, time=0.005, feedback=0.5, mix=1.0, pingpong=False)
    y = d.process(impulse(2048))
    peaks = np.flatnonzero(y[:, 0] > 0.1)
    assert peaks[0] == int(0.005 * SR) and len(peaks) >= 2

def test_reverb_has_decaying_tail():
    r = Reverb(SR, 120, size=0.8, damping=0.5, mix=1.0, predelay=0.0)
    y = r.process(impulse(2 * SR))
    e1 = np.sum(y[SR // 10:SR // 2] ** 2); e2 = np.sum(y[SR:SR + SR // 2] ** 2)
    assert e1 > 1e-6 and e2 < e1 and np.isfinite(y).all()

def test_reverb_block_and_whole_equivalent():
    x = np.random.default_rng(0).normal(size=(4096, 2)).astype(np.float32) * 0.1
    a = Reverb(SR, 120, mix=1.0, predelay=0.0).process(x)
    r = Reverb(SR, 120, mix=1.0, predelay=0.0)
    b = np.concatenate([r.process(x[:1000]), r.process(x[1000:])])
    assert np.allclose(a, b, atol=1e-5)

def test_gated_reverb_cuts_after_hold():
    g = GatedReverb(SR, 120, size=0.9, hold=0.1, threshold=0.5, mix=1.0)
    y = g.process(impulse(SR))
    assert np.abs(y[int(0.02 * SR):int(0.09 * SR)]).max() > 0
    assert np.abs(y[int(0.15 * SR):]).max() == 0

def test_chorus_shape_and_stereo():
    c = Chorus(SR, 120)
    x = np.stack([np.sin(np.arange(4096) * 0.05)] * 2, axis=1).astype(np.float32)
    y = c.process(x)
    assert y.shape == x.shape and y.dtype == np.float32 and not np.allclose(y[:, 0], y[:, 1])

def test_sidechain_dips_on_trigger():
    sc = Sidechain(SR, depth=0.6, release=0.1)
    g = sc.gain(1000, [500])
    assert np.isclose(g[0], 1.0, atol=1e-3) and g[500] < 0.45 and g[999] > g[500]

def test_limiter_bounds_output():
    lim = Limiter(SR, 120, threshold=0.9)
    y = lim.process(impulse(2048, at=100, amp=3.0))
    assert np.abs(y).max() <= 1.0

def test_build_effects_registry():
    fx = build_effects([{"type": "chorus"}, {"type": "delay", "time": "1/4"}], SR, 110)
    assert len(fx) == 2 and isinstance(fx[0], Chorus)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_effects.py -v` — Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# synthwave/engine/effects.py
"""Block-vectorised effects. Delay lines are processed in chunks no longer than their delay."""
from __future__ import annotations
import numpy as np
from scipy.signal import lfilter
from .lfo import LFO


def note_to_seconds(value: float | str, bpm: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = value.strip()
    mult = 1.0
    if s.endswith("d"):
        mult, s = 1.5, s[:-1]
    elif s.endswith("t"):
        mult, s = 2.0 / 3.0, s[:-1]
    num, den = s.split("/")
    return 4.0 * (60.0 / bpm) * int(num) / int(den) * mult


class Effect:
    def process(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class Chorus(Effect):
    def __init__(self, sr: int, bpm: float, rate: float = 0.5, depth: float = 0.003,
                 mix: float = 0.4):
        self.size = int(sr * 0.2)
        self.buf = np.zeros((self.size, 2))
        self.pos = 0
        self.lfo = [LFO("sine", rate, sr, 0.0), LFO("sine", rate, sr, 0.25)]
        self.depth, self.base, self.mix = depth * sr, 0.02 * sr, mix

    def process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        idx = (self.pos + np.arange(n)) % self.size
        self.buf[idx] = x
        out = np.empty((n, 2))
        for ch in (0, 1):
            d = self.base + self.depth * (1.0 + self.lfo[ch].render(n))
            p = self.pos + np.arange(n) - d
            i0 = np.floor(p).astype(int)
            fr = p - i0
            wet = self.buf[i0 % self.size, ch] * (1 - fr) + self.buf[(i0 + 1) % self.size, ch] * fr
            out[:, ch] = x[:, ch] * (1 - self.mix) + wet * self.mix
        self.pos = (self.pos + n) % self.size
        return out.astype(np.float32)


class Delay(Effect):
    def __init__(self, sr: int, bpm: float, time: float | str = "1/8", feedback: float = 0.4,
                 mix: float = 0.3, pingpong: bool = True):
        self.d = max(1, int(note_to_seconds(time, bpm) * sr))
        self.size = self.d * 2 + 8192
        self.buf = np.zeros((self.size, 2))
        self.pos = 0
        self.feedback, self.mix, self.pingpong = float(np.clip(feedback, 0, 0.95)), mix, pingpong

    def process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        out = np.empty((n, 2))
        start = 0
        while start < n:
            k = min(self.d, n - start)
            xs = x[start:start + k]
            ridx = (self.pos + np.arange(k) - self.d) % self.size
            y = self.buf[ridx]
            widx = (self.pos + np.arange(k)) % self.size
            if self.pingpong:
                mono = xs.mean(axis=1)
                self.buf[widx, 0] = mono + y[:, 1] * self.feedback
                self.buf[widx, 1] = y[:, 0] * self.feedback
            else:
                self.buf[widx] = xs + y * self.feedback
            out[start:start + k] = xs * (1 - self.mix) + y * self.mix
            self.pos = (self.pos + k) % self.size
            start += k
        return out.astype(np.float32)


_COMBS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
_ALLPASSES = (556, 441, 341, 225)
_SPREAD = 23


class _Line:
    def __init__(self, length: int):
        self.len, self.buf, self.pos, self.zi = length, np.zeros(length), 0, np.zeros(1)


class Reverb(Effect):
    """Freeverb-style Schroeder reverb: 8 lowpass-feedback combs + 4 allpasses per channel."""

    def __init__(self, sr: int, bpm: float, size: float = 0.8, damping: float = 0.5,
                 mix: float = 0.3, predelay: float = 0.02):
        scale = sr / 44100.0
        self.fb = 0.7 + 0.28 * float(np.clip(size, 0, 1))
        self.damp = float(np.clip(damping, 0, 0.95))
        self.mix = mix
        self.combs = [[_Line(int(c * scale) + ch * _SPREAD) for c in _COMBS] for ch in (0, 1)]
        self.allp = [[_Line(int(a * scale) + ch * _SPREAD) for a in _ALLPASSES] for ch in (0, 1)]
        self.pre = max(0, int(predelay * sr))
        self.pre_line = _Line(self.pre + 8192) if self.pre else None

    def _comb(self, c: _Line, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        start = 0
        while start < len(x):
            k = min(c.len, len(x) - start)
            idx = (c.pos + np.arange(k)) % c.len
            y = c.buf[idx]
            lp, c.zi = lfilter([1 - self.damp], [1, -self.damp], y, zi=c.zi)
            c.buf[idx] = x[start:start + k] + lp * self.fb
            out[start:start + k] = y
            c.pos = (c.pos + k) % c.len
            start += k
        return out

    @staticmethod
    def _allpass(a: _Line, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        start = 0
        while start < len(x):
            k = min(a.len, len(x) - start)
            idx = (a.pos + np.arange(k)) % a.len
            z = a.buf[idx]
            xs = x[start:start + k]
            a.buf[idx] = xs + z * 0.5
            out[start:start + k] = z - xs
            a.pos = (a.pos + k) % a.len
            start += k
        return out

    def _predelay(self, x: np.ndarray) -> np.ndarray:
        if not self.pre_line:
            return x
        p = self.pre_line
        n = len(x)
        out = np.empty_like(x)
        start = 0
        while start < n:
            k = min(self.pre, n - start)
            widx = (p.pos + np.arange(k)) % p.len
            ridx = (p.pos + np.arange(k) - self.pre) % p.len
            out[start:start + k] = p.buf[ridx]
            p.buf[widx] = x[start:start + k]
            p.pos = (p.pos + k) % p.len
            start += k
        return out

    def process(self, x: np.ndarray) -> np.ndarray:
        mono = self._predelay(x.astype(np.float64).mean(axis=1)) * 0.03
        out = np.empty((len(x), 2))
        for ch in (0, 1):
            wet = sum(self._comb(c, mono) for c in self.combs[ch])
            for a in self.allp[ch]:
                wet = self._allpass(a, wet)
            out[:, ch] = x[:, ch] * (1 - self.mix) + wet * self.mix
        return out.astype(np.float32)


class GatedReverb(Effect):
    def __init__(self, sr: int, bpm: float, size: float = 0.85, hold: float = 0.25,
                 threshold: float = 0.1, mix: float = 0.5):
        self.rev = Reverb(sr, bpm, size=size, damping=0.3, mix=1.0, predelay=0.0)
        self.hold, self.thr, self.mix = int(hold * sr), threshold, mix
        self.open_left = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        wet = self.rev.process(x)
        gate = np.zeros(n)
        if self.open_left > 0:
            gate[:min(n, self.open_left)] = 1.0
        hits = np.flatnonzero(np.abs(x).max(axis=1) > self.thr)
        end = -1
        for h in hits:
            if h < end:
                continue
            end = h + self.hold
            gate[h:min(n, end)] = 1.0
        self.open_left = max(self.open_left - n, end - n, 0)
        return (x * (1 - self.mix) + wet * gate[:, None] * self.mix).astype(np.float32)


class Limiter(Effect):
    def __init__(self, sr: int, bpm: float, threshold: float = 0.9, release: float = 0.1):
        self.thr, self.coef, self.gain = threshold, float(np.exp(-1.0 / (release * sr))), 1.0

    def process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        peak = float(np.abs(x).max()) if n else 0.0
        target = min(1.0, self.thr / peak) if peak > 0 else 1.0
        new = target if target < self.gain else target + (self.gain - target) * self.coef ** n
        ramp = np.linspace(self.gain, new, n, endpoint=False)
        self.gain = new
        return np.clip(x * ramp[:, None], -1.0, 1.0).astype(np.float32)


class Sidechain:
    """Ducking envelope triggered at sample offsets; gain = 1 - depth * exp(-t/release)."""

    def __init__(self, sr: int, depth: float = 0.5, release: float = 0.25):
        self.depth, self.rel, self.t = depth, max(1.0, release * sr), 10 ** 9

    def gain(self, n: int, triggers: list[int]) -> np.ndarray:
        g = np.empty(n)
        pos = 0
        for tr in sorted(triggers) + [n]:
            k = tr - pos
            if k > 0:
                ts = self.t + np.arange(1, k + 1)
                g[pos:tr] = 1.0 - self.depth * np.exp(-ts / self.rel)
                self.t += k
                pos = tr
            if tr < n:
                self.t = 0
        return g


_REGISTRY = {"chorus": Chorus, "delay": Delay, "reverb": Reverb,
             "gated_reverb": GatedReverb, "limiter": Limiter}


def build_effects(specs: list[dict], sr: int, bpm: float) -> list[Effect]:
    out = []
    for spec in specs:
        kw = dict(spec)
        kind = kw.pop("type")
        out.append(_REGISTRY[kind](sr, bpm, **kw))
    return out
```

- [ ] **Step 4: Run** `uv run pytest tests/test_effects.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat(engine): chorus, delay, reverb, gated reverb, sidechain, limiter"`

### Task 5: Modèles de patch, loader et bibliothèque YAML

**Files:**
- Create: `synthwave/patches/__init__.py`, `synthwave/patches/model.py`, `synthwave/patches/loader.py`, `synthwave/patches/library/{bass_moog,pad_juno,arp_pluck,lead_saw,ambient_drone,drums_808}.yaml`
- Test: `tests/test_patches.py`

**Interfaces:**
- Produces: pydantic `OscSpec, EnvSpec, FilterSpec, LfoSpec, EffectSpec, PatchModel, KickSpec, SnareSpec, HatSpec, ClapSpec, TomSpec, DrumPatchModel` ; `AnyPatch = PatchModel | DrumPatchModel` ; `PatchError(Exception)` ; `list_patches() -> list[str]` ; `load_patch(name_or_path: str) -> AnyPatch` ; `patch_from_dict(data: dict) -> AnyPatch` ; `set_param(patch: AnyPatch, path: str, value) -> AnyPatch` (retourne un nouveau patch validé, chemin type `filter.cutoff` ou `oscillators.0.detune`).

- [ ] **Step 1: Tests**

```python
# tests/test_patches.py
import pytest
from synthwave.patches.loader import PatchError, list_patches, load_patch, patch_from_dict, set_param
from synthwave.patches.model import DrumPatchModel, PatchModel

def test_library_lists_and_loads_all():
    names = list_patches()
    assert {"bass_moog", "pad_juno", "arp_pluck", "lead_saw", "ambient_drone", "drums_808"} <= set(names)
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
    f.write_text("name: x\noscillators:\n  - wave: sine\namp_env: {attack: 0.01, decay: 0.1, sustain: 0.5, release: 0.2}\n")
    assert load_patch(str(f)).name == "x"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_patches.py -v` — Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# synthwave/patches/model.py
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

Wave = Literal["saw", "square", "triangle", "sine", "noise"]


class OscSpec(BaseModel):
    wave: Wave
    unison: int = Field(1, ge=1, le=8)
    detune: float = Field(0.0, ge=0.0, le=100.0)
    octave: int = Field(0, ge=-3, le=3)
    semi: int = Field(0, ge=-12, le=12)
    level: float = Field(1.0, ge=0.0, le=2.0)
    pwm: float = Field(0.5, ge=0.05, le=0.95)
    spread: float = Field(1.0, ge=0.0, le=1.0)


class EnvSpec(BaseModel):
    attack: float = Field(0.01, ge=0.0)
    decay: float = Field(0.1, ge=0.0)
    sustain: float = Field(1.0, ge=0.0, le=1.0)
    release: float = Field(0.2, ge=0.0)
    amount: float = 0.0  # used for filter env (Hz)


class FilterSpec(BaseModel):
    type: Literal["lp", "hp", "bp"] = "lp"
    cutoff: float = Field(2000.0, ge=20.0, le=20000.0)
    resonance: float = Field(0.0, ge=0.0, le=1.0)
    env: EnvSpec | None = None
    key_track: float = Field(0.0, ge=0.0, le=1.0)


class LfoSpec(BaseModel):
    wave: Literal["sine", "triangle", "square", "saw"] = "sine"
    rate: float = Field(1.0, gt=0.0)
    target: Literal["pitch", "cutoff", "amp", "pwm"] = "cutoff"
    amount: float = 0.0  # semitones for pitch, Hz for cutoff, 0..1 for amp/pwm


class EffectSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["chorus", "delay", "reverb", "gated_reverb", "limiter"]


class PatchModel(BaseModel):
    name: str
    kind: Literal["synth"] = "synth"
    polyphony: int = Field(8, ge=1, le=16)
    glide: float = Field(0.0, ge=0.0)
    volume: float = Field(0.8, ge=0.0, le=2.0)
    oscillators: list[OscSpec] = Field(min_length=1)
    amp_env: EnvSpec = EnvSpec()
    filter: FilterSpec | None = None
    lfo: LfoSpec | None = None
    effects: list[EffectSpec] = []


class KickSpec(BaseModel):
    pitch_start: float = 160.0
    pitch_end: float = 45.0
    pitch_decay: float = 0.05
    decay: float = 0.4
    click: float = 0.3


class SnareSpec(BaseModel):
    tone: float = 180.0
    tone_decay: float = 0.08
    noise_decay: float = 0.18
    gate_hold: float = 0.25
    reverb_size: float = 0.85
    reverb_mix: float = 0.5


class HatSpec(BaseModel):
    closed_decay: float = 0.05
    open_decay: float = 0.35
    cutoff: float = 8000.0


class ClapSpec(BaseModel):
    decay: float = 0.25
    gate_hold: float = 0.2
    reverb_mix: float = 0.5


class TomSpec(BaseModel):
    pitch_low: float = 110.0
    pitch_mid: float = 160.0
    decay: float = 0.3


class DrumPatchModel(BaseModel):
    name: str
    kind: Literal["drums"]
    volume: float = Field(0.9, ge=0.0, le=2.0)
    kick: KickSpec = KickSpec()
    snare: SnareSpec = SnareSpec()
    hat: HatSpec = HatSpec()
    clap: ClapSpec = ClapSpec()
    tom: TomSpec = TomSpec()


AnyPatch = PatchModel | DrumPatchModel
```

```python
# synthwave/patches/loader.py
from __future__ import annotations
from pathlib import Path
import yaml
from pydantic import ValidationError
from .model import AnyPatch, DrumPatchModel, PatchModel

LIBRARY = Path(__file__).parent / "library"
USER_DIR = Path.home() / ".config" / "synthwave" / "patches"


class PatchError(Exception):
    pass


def _dirs() -> list[Path]:
    return [d for d in (USER_DIR, LIBRARY) if d.is_dir()]


def list_patches() -> list[str]:
    names = {p.stem for d in _dirs() for p in d.glob("*.yaml")}
    return sorted(names)


def patch_from_dict(data: dict) -> AnyPatch:
    try:
        if not isinstance(data, dict):
            raise PatchError("patch must be a mapping")
        if data.get("kind") == "drums":
            return DrumPatchModel.model_validate(data)
        return PatchModel.model_validate(data)
    except ValidationError as e:
        raise PatchError(str(e)) from e


def load_patch(name_or_path: str) -> AnyPatch:
    path = Path(name_or_path)
    if not path.suffix:
        for d in _dirs():
            cand = d / f"{name_or_path}.yaml"
            if cand.exists():
                path = cand
                break
        else:
            raise PatchError(f"patch {name_or_path!r} not found in {[str(d) for d in _dirs()]}")
    if not path.exists():
        raise PatchError(f"patch file {path} not found")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise PatchError(f"invalid YAML in {path}: {e}") from e
    return patch_from_dict(data)


def set_param(patch: AnyPatch, path: str, value) -> AnyPatch:
    data = patch.model_dump()
    keys = path.split(".")
    node = data
    try:
        for k in keys[:-1]:
            node = node[int(k)] if isinstance(node, list) else node[k]
        last = keys[-1]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value
    except (KeyError, IndexError, ValueError, TypeError) as e:
        raise PatchError(f"bad parameter path {path!r}: {e}") from e
    return patch_from_dict(data)
```

Bibliothèque YAML :

```yaml
# synthwave/patches/library/bass_moog.yaml
name: bass_moog
polyphony: 1
glide: 0.03
volume: 0.9
oscillators:
  - {wave: saw, level: 0.9}
  - {wave: square, octave: -1, level: 0.5, pwm: 0.5}
amp_env: {attack: 0.005, decay: 0.15, sustain: 0.7, release: 0.12}
filter:
  type: lp
  cutoff: 380
  resonance: 0.35
  env: {attack: 0.002, decay: 0.18, sustain: 0.1, release: 0.1, amount: 1400}
effects: []
```

```yaml
# synthwave/patches/library/pad_juno.yaml
name: pad_juno
polyphony: 6
volume: 0.5
oscillators:
  - {wave: saw, unison: 3, detune: 14, level: 0.7, spread: 1.0}
  - {wave: square, octave: -1, level: 0.25, pwm: 0.45}
amp_env: {attack: 0.9, decay: 0.6, sustain: 0.8, release: 1.8}
filter:
  type: lp
  cutoff: 1400
  resonance: 0.15
  env: {attack: 0.8, decay: 1.2, sustain: 0.4, release: 1.5, amount: 900}
lfo: {wave: sine, rate: 0.25, target: cutoff, amount: 350}
effects:
  - {type: chorus, rate: 0.6, depth: 0.004, mix: 0.45}
  - {type: reverb, size: 0.9, damping: 0.45, mix: 0.35, predelay: 0.03}
```

```yaml
# synthwave/patches/library/arp_pluck.yaml
name: arp_pluck
polyphony: 6
volume: 0.55
oscillators:
  - {wave: saw, unison: 2, detune: 8, level: 0.8}
  - {wave: square, octave: 1, level: 0.2, pwm: 0.3}
amp_env: {attack: 0.002, decay: 0.22, sustain: 0.0, release: 0.15}
filter:
  type: lp
  cutoff: 900
  resonance: 0.3
  env: {attack: 0.001, decay: 0.16, sustain: 0.0, release: 0.1, amount: 3200}
effects:
  - {type: delay, time: "1/8d", feedback: 0.35, mix: 0.3, pingpong: true}
  - {type: reverb, size: 0.7, damping: 0.5, mix: 0.2}
```

```yaml
# synthwave/patches/library/lead_saw.yaml
name: lead_saw
polyphony: 2
glide: 0.05
volume: 0.5
oscillators:
  - {wave: saw, unison: 3, detune: 18, level: 0.8}
  - {wave: saw, octave: 1, level: 0.3}
amp_env: {attack: 0.02, decay: 0.3, sustain: 0.7, release: 0.4}
filter:
  type: lp
  cutoff: 2600
  resonance: 0.2
  env: {attack: 0.01, decay: 0.4, sustain: 0.5, release: 0.3, amount: 2000}
lfo: {wave: sine, rate: 5.5, target: pitch, amount: 0.12}
effects:
  - {type: delay, time: "1/4", feedback: 0.45, mix: 0.35, pingpong: true}
  - {type: reverb, size: 0.85, damping: 0.4, mix: 0.3}
```

```yaml
# synthwave/patches/library/ambient_drone.yaml
name: ambient_drone
polyphony: 3
volume: 0.35
oscillators:
  - {wave: sine, level: 0.6}
  - {wave: triangle, octave: 1, level: 0.25}
  - {wave: noise, level: 0.12}
amp_env: {attack: 2.5, decay: 1.0, sustain: 0.9, release: 3.0}
filter:
  type: lp
  cutoff: 600
  resonance: 0.1
lfo: {wave: triangle, rate: 0.08, target: cutoff, amount: 300}
effects:
  - {type: chorus, rate: 0.2, depth: 0.006, mix: 0.5}
  - {type: reverb, size: 1.0, damping: 0.6, mix: 0.6, predelay: 0.04}
```

```yaml
# synthwave/patches/library/drums_808.yaml
name: drums_808
kind: drums
volume: 0.9
kick: {pitch_start: 170, pitch_end: 48, pitch_decay: 0.045, decay: 0.42, click: 0.35}
snare: {tone: 185, tone_decay: 0.09, noise_decay: 0.2, gate_hold: 0.28, reverb_size: 0.9, reverb_mix: 0.55}
hat: {closed_decay: 0.045, open_decay: 0.3, cutoff: 8500}
clap: {decay: 0.22, gate_hold: 0.22, reverb_mix: 0.5}
tom: {pitch_low: 105, pitch_mid: 150, decay: 0.32}
```

- [ ] **Step 4: Run** `uv run pytest tests/test_patches.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat(patches): pydantic models, loader and YAML library"`

### Task 6: Voice + Synth polyphonique

**Files:**
- Create: `synthwave/engine/voice.py`, `synthwave/engine/synth.py`
- Test: `tests/test_synth.py`

**Interfaces:**
- Consumes: `Oscillator`, `ADSR`, `LFO`, `Filter`, `build_effects`, `PatchModel`.
- Produces: `NoteEvent(offset: int, note: int, velocity: float, on: bool)` (dataclass dans `synthwave/engine/events.py`) ; `Voice(patch, sr, rng)` : `note_on(note, velocity)`, `note_off()`, `active: bool`, `render(n) -> (n,2)` ; `Synth(patch, sr, rng, bpm)` : `set_patch(patch)`, `set_bpm(bpm)`, `note_on(note, velocity)`, `note_off(note)`, `render(n, events: list[NoteEvent]) -> (n,2) float32`, `patch: PatchModel`.

- [ ] **Step 1: Tests**

```python
# tests/test_synth.py
import numpy as np
from synthwave.engine.events import NoteEvent
from synthwave.engine.synth import Synth
from synthwave.engine.voice import Voice
from synthwave.patches.loader import load_patch, patch_from_dict

SR = 44100

def simple_patch(**kw):
    d = {"name": "t", "polyphony": 2, "oscillators": [{"wave": "sine"}],
         "amp_env": {"attack": 0.001, "decay": 0.01, "sustain": 1.0, "release": 0.01}}
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
    v.note_on(60, 1.0); v.render(100); v.note_off(); v.render(SR // 10)
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
```

- [ ] **Step 2: Run** `uv run pytest tests/test_synth.py -v` — Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# synthwave/engine/events.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class NoteEvent:
    offset: int          # sample offset inside the block
    note: int            # MIDI note
    velocity: float
    on: bool             # False = note-off
```

```python
# synthwave/engine/voice.py
from __future__ import annotations
import numpy as np
from ..patches.model import PatchModel
from .envelope import ADSR
from .filter import Filter
from .lfo import LFO
from .oscillator import Oscillator


def midi_to_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)


class Voice:
    def __init__(self, patch: PatchModel, sr: int, rng: np.random.Generator):
        self.patch, self.sr = patch, sr
        self.oscs = [Oscillator(o.wave, sr, rng, o.unison, o.detune, o.octave, o.semi, o.level,
                                o.pwm, o.spread) for o in patch.oscillators]
        e = patch.amp_env
        self.amp_env = ADSR(e.attack, e.decay, e.sustain, e.release, sr)
        self.filter = Filter(patch.filter.type, sr) if patch.filter else None
        fe = patch.filter.env if patch.filter else None
        self.filt_env = ADSR(fe.attack, fe.decay, fe.sustain, fe.release, sr) if fe else None
        self.lfo = LFO(patch.lfo.wave, patch.lfo.rate, sr) if patch.lfo else None
        self.note, self.velocity, self.freq, self.target_freq = 60, 0.0, 261.6, 261.6
        self.glide_coef = float(np.exp(-1.0 / (patch.glide * sr))) if patch.glide > 0 else 0.0
        self.age = 0

    @property
    def active(self) -> bool:
        return not self.amp_env.finished

    def note_on(self, note: int, velocity: float, legato: bool = False) -> None:
        self.note, self.velocity = note, float(velocity)
        self.target_freq = midi_to_hz(note)
        if not (legato and self.glide_coef):
            self.freq = self.target_freq
        if not legato or not self.active:
            self.amp_env.gate_on()
            if self.filt_env:
                self.filt_env.gate_on()

    def note_off(self) -> None:
        self.amp_env.gate_off()
        if self.filt_env:
            self.filt_env.gate_off()

    def render(self, n: int) -> np.ndarray:
        p = self.patch
        lfo = self.lfo.render(n) if self.lfo else None
        lfo_mean = float(lfo.mean()) if lfo is not None else 0.0
        if self.glide_coef:
            self.freq = self.target_freq + (self.freq - self.target_freq) * self.glide_coef ** n
        freq = self.freq
        pwm = None
        if p.lfo and p.lfo.target == "pitch":
            freq *= 2.0 ** (p.lfo.amount * lfo_mean / 12.0)
        if p.lfo and p.lfo.target == "pwm":
            pwm = 0.5 + 0.45 * p.lfo.amount * lfo_mean
        sig = np.zeros((n, 2), dtype=np.float32)
        for osc in self.oscs:
            sig += osc.render(freq, n, pwm)
        if self.filter:
            f = p.filter
            cutoff = f.cutoff + f.key_track * (freq - 261.6)
            if self.filt_env:
                cutoff += f.env.amount * float(self.filt_env.render(n).mean())
            if p.lfo and p.lfo.target == "cutoff":
                cutoff += p.lfo.amount * lfo_mean
            sig = self.filter.process(sig, cutoff, f.resonance)
        amp = self.amp_env.render(n) * self.velocity
        if p.lfo and p.lfo.target == "amp":
            amp = amp * (1.0 - p.lfo.amount * 0.5 * (1.0 - lfo))
        return sig * amp[:, None]
```

```python
# synthwave/engine/synth.py
from __future__ import annotations
import numpy as np
from ..patches.model import PatchModel
from .effects import build_effects
from .events import NoteEvent
from .voice import Voice


class Synth:
    def __init__(self, patch: PatchModel, sr: int, rng: np.random.Generator, bpm: float):
        self.sr, self.rng, self.bpm = sr, rng, bpm
        self.counter = 0
        self.set_patch(patch)

    def set_patch(self, patch: PatchModel) -> None:
        self.patch = patch
        self.voices = [Voice(patch, self.sr, self.rng) for _ in range(patch.polyphony)]
        self.effects = build_effects([e.model_dump() for e in patch.effects], self.sr, self.bpm)

    def set_bpm(self, bpm: float) -> None:
        self.bpm = bpm
        self.effects = build_effects([e.model_dump() for e in self.patch.effects], self.sr, bpm)

    def note_on(self, note: int, velocity: float) -> None:
        self.counter += 1
        if len(self.voices) == 1:
            v = self.voices[0]
            v.note_on(note, velocity, legato=v.active)
            v.age = self.counter
            return
        free = [v for v in self.voices if not v.active]
        v = free[0] if free else min(self.voices, key=lambda v: v.age)
        v.note_on(note, velocity)
        v.age = self.counter

    def note_off(self, note: int) -> None:
        for v in self.voices:
            if v.active and v.note == note and v.amp_env.stage != 4:
                v.note_off()

    def _render_voices(self, n: int) -> np.ndarray:
        out = np.zeros((n, 2), dtype=np.float32)
        for v in self.voices:
            if v.active:
                out += v.render(n)
        return out

    def render(self, n: int, events: list[NoteEvent]) -> np.ndarray:
        out = np.zeros((n, 2), dtype=np.float32)
        pos = 0
        for ev in sorted(events):
            off = min(max(ev.offset, pos), n)
            if off > pos:
                out[pos:off] = self._render_voices(off - pos)
                pos = off
            if ev.on:
                self.note_on(ev.note, ev.velocity)
            else:
                self.note_off(ev.note)
        if pos < n:
            out[pos:] = self._render_voices(n - pos)
        out *= self.patch.volume
        for fx in self.effects:
            out = fx.process(out)
        return out
```

Note : `NoteEvent` est `order=True` ; le tri place les note-off (`on=False`) avant les note-on au même offset car `False < True`.

- [ ] **Step 4: Run** `uv run pytest tests/test_synth.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat(engine): voice and polyphonic synth with event-accurate rendering"`

### Task 7: DrumKit synthétisé

**Files:**
- Create: `synthwave/engine/drums.py`
- Test: `tests/test_drums.py`

**Interfaces:**
- Consumes: `DrumPatchModel`, `GatedReverb`, `Filter`, `NoteEvent`.
- Produces: `DRUM_NOTES = {"kick": 36, "snare": 38, "clap": 39, "hat_closed": 42, "tom_low": 45, "hat_open": 46, "tom_mid": 47, "crash": 49}` ; `DrumKit(patch, sr, rng, bpm=None)` : `set_patch(patch)`, `set_bpm(bpm)` (no-op), `render(n, events) -> (n,2) float32`, `samples: dict[str, np.ndarray]`, `patch`.

- [ ] **Step 1: Tests**

```python
# tests/test_drums.py
import numpy as np
from synthwave.engine.drums import DRUM_NOTES, DrumKit
from synthwave.engine.events import NoteEvent
from synthwave.patches.loader import load_patch

SR = 44100

def kit():
    return DrumKit(load_patch("drums_808"), SR, np.random.default_rng(0))

def test_all_samples_exist_finite_and_normalised():
    k = kit()
    for name in DRUM_NOTES:
        s = k.samples[name]
        assert s.ndim == 2 and s.shape[1] == 2 and np.isfinite(s).all()
        assert 0.5 < np.abs(s).max() <= 1.0

def test_kick_is_low_frequency():
    s = kit().samples["kick"][:, 0]
    spec = np.abs(np.fft.rfft(s))
    f = np.fft.rfftfreq(len(s), 1 / SR)[np.argmax(spec)]
    assert 30 < f < 120

def test_render_places_hit_at_offset_and_spans_blocks():
    k = kit()
    a = k.render(1024, [NoteEvent(1000, 36, 1.0, True)])
    b = k.render(1024, [])
    assert np.abs(a[:1000]).max() == 0 and np.abs(a[1000:]).max() > 0 and np.abs(b).max() > 0

def test_note_off_is_ignored():
    k = kit()
    out = k.render(512, [NoteEvent(0, 38, 1.0, False)])
    assert np.abs(out).max() == 0
```

- [ ] **Step 2: Run** `uv run pytest tests/test_drums.py -v` — Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# synthwave/engine/drums.py
"""Drum kit: every hit is synthesised once at init into a stereo buffer, then mixed on demand."""
from __future__ import annotations
import numpy as np
from scipy.signal import lfilter
from ..patches.model import DrumPatchModel
from .effects import GatedReverb
from .events import NoteEvent
from .filter import biquad_coeffs

DRUM_NOTES = {"kick": 36, "snare": 38, "clap": 39, "hat_closed": 42, "tom_low": 45,
              "hat_open": 46, "tom_mid": 47, "crash": 49}
NOTE_TO_DRUM = {v: k for k, v in DRUM_NOTES.items()}


def _t(sr: int, seconds: float) -> np.ndarray:
    return np.arange(int(sr * seconds)) / sr


def _stereo(x: np.ndarray) -> np.ndarray:
    x = x / max(1e-9, float(np.abs(x).max()))
    return np.stack([x, x], axis=1).astype(np.float32)


def _filt(kind: str, x: np.ndarray, cutoff: float, res: float, sr: int) -> np.ndarray:
    b, a = biquad_coeffs(kind, cutoff, res, sr)
    return lfilter(b, a, x)


class DrumKit:
    def __init__(self, patch: DrumPatchModel, sr: int, rng: np.random.Generator, bpm=None):
        self.sr, self.rng = sr, rng
        self.active: list[tuple[np.ndarray, int, float]] = []  # (sample, position, gain)
        self.set_patch(patch)

    def set_bpm(self, bpm: float) -> None:
        pass

    def set_patch(self, patch: DrumPatchModel) -> None:
        self.patch = patch
        sr, rng = self.sr, self.rng
        k, s, h, c, tm = patch.kick, patch.snare, patch.hat, patch.clap, patch.tom
        # kick
        t = _t(sr, k.decay * 3)
        f = k.pitch_end + (k.pitch_start - k.pitch_end) * np.exp(-t / k.pitch_decay)
        kick = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / k.decay)
        kick[: int(0.002 * sr)] += k.click * rng.uniform(-1, 1, int(0.002 * sr))
        # snare
        t = _t(sr, 1.2)
        tone = np.sin(2 * np.pi * s.tone * t) * np.exp(-t / s.tone_decay)
        noise = _filt("hp", rng.uniform(-1, 1, len(t)), 1800, 0.2, sr) * np.exp(-t / s.noise_decay)
        snare = 0.6 * tone + noise
        snare = GatedReverb(sr, 120, size=s.reverb_size, hold=s.gate_hold, threshold=0.2,
                            mix=s.reverb_mix).process(_stereo(snare))[:, 0]
        # clap
        t = _t(sr, 1.0)
        noise = _filt("bp", rng.uniform(-1, 1, len(t)), 1500, 0.3, sr)
        env = np.zeros_like(t)
        for i in range(4):
            start = int(i * 0.011 * sr)
            env[start:] = np.maximum(env[start:], np.exp(-(t[: len(t) - start]) / 0.012))
        env = np.maximum(env, 0.7 * np.exp(-t / c.decay) * (t > 0.03))
        clap = GatedReverb(sr, 120, size=0.85, hold=c.gate_hold, threshold=0.2,
                           mix=c.reverb_mix).process(_stereo(noise * env))[:, 0]
        # hats
        t = _t(sr, h.open_decay * 3)
        base = _filt("hp", rng.uniform(-1, 1, len(t)), h.cutoff, 0.3, sr)
        hat_c = base * np.exp(-t / h.closed_decay)
        hat_o = base * np.exp(-t / h.open_decay)
        # toms
        toms = {}
        for name, pitch in (("tom_low", tm.pitch_low), ("tom_mid", tm.pitch_mid)):
            t = _t(sr, tm.decay * 3)
            f = pitch * (1 + 0.6 * np.exp(-t / 0.04))
            toms[name] = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / tm.decay)
        # crash
        t = _t(sr, 2.0)
        crash = _filt("bp", rng.uniform(-1, 1, len(t)), 6000, 0.1, sr) * np.exp(-t / 0.7)
        self.samples = {
            "kick": _stereo(kick), "snare": _stereo(snare), "clap": _stereo(clap),
            "hat_closed": _stereo(hat_c), "hat_open": _stereo(hat_o),
            "tom_low": _stereo(toms["tom_low"]), "tom_mid": _stereo(toms["tom_mid"]),
            "crash": _stereo(crash),
        }
        self.samples["hat_closed"] *= 0.6
        self.samples["hat_open"] *= 0.5
        self.samples["crash"] *= 0.6

    def render(self, n: int, events: list[NoteEvent]) -> np.ndarray:
        out = np.zeros((n, 2), dtype=np.float32)
        for ev in events:
            name = NOTE_TO_DRUM.get(ev.note)
            if ev.on and name:
                if name == "hat_closed":  # closed hat chokes the open hat
                    self.active = [a for a in self.active if a[0] is not self.samples["hat_open"]]
                self.active.append((self.samples[name], -int(ev.offset), float(ev.velocity)))
        still = []
        for sample, pos, gain in self.active:
            start = max(0, -pos)          # first output index for this hit inside the block
            src = max(0, pos)             # first sample index
            k = min(n - start, len(sample) - src)
            if k > 0:
                out[start:start + k] += sample[src:src + k] * gain
            pos += n
            if pos < len(sample):
                still.append((sample, pos, gain))
        self.active = still
        return out * self.patch.volume
```

- [ ] **Step 4: Run** `uv run pytest tests/test_drums.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat(engine): synthesised drum kit with gated reverb snare"`

### Task 8: Moods + Harmonie (Markov)

**Files:**
- Create: `synthwave/composer/__init__.py`, `synthwave/composer/moods.py`, `synthwave/composer/harmony.py`
- Test: `tests/test_harmony.py`

**Interfaces:**
- Produces: `Mood(name, bpm, drum_density, arp_prob, lead_prob, brightness, major_prob, progressions: dict[str, float])`, `MOODS: dict[str, Mood]` ; `Chord(root_pc: int, degree: int, intervals: tuple[int, ...])` avec `notes(octave=4) -> list[int]`, `bass_note() -> int`, `name: str` ; `Harmony(rng, mood)` : `tonic: int`, `mode: tuple`, `chord_for_degree(deg) -> Chord`, `next_progression() -> list[Chord]`, `modulate()`, `scale_notes(low, high) -> list[int]`, `key_name: str`.

- [ ] **Step 1: Tests**

```python
# tests/test_harmony.py
import numpy as np
from synthwave.composer.harmony import Chord, Harmony, PROGRESSIONS
from synthwave.composer.moods import MOODS

def test_chord_names_and_notes():
    assert Chord(9, 0, (0, 3, 7, 10)).name == "Am7"
    assert Chord(5, 5, (0, 4, 7, 11)).name == "Fmaj7"
    n = Chord(9, 0, (0, 3, 7, 10)).notes(4)
    assert n == [57, 60, 64, 67] and Chord(9, 0, (0, 3, 7, 10)).bass_note() == 33

def test_minor_key_degrees():
    h = Harmony(np.random.default_rng(0), MOODS["dark"])
    h.tonic, h.mode = 9, h.MINOR
    assert h.chord_for_degree(0).name == "Am7"
    assert h.chord_for_degree(5).name == "Fmaj7"
    assert h.chord_for_degree(2).name == "Cmaj7"
    assert h.chord_for_degree(6).name == "G7"

def test_progression_is_four_chords_in_key():
    h = Harmony(np.random.default_rng(3), MOODS["outrun"])
    prog = h.next_progression()
    assert len(prog) == 4
    scale = {(h.tonic + i) % 12 for i in h.mode}
    for c in prog:
        assert c.root_pc in scale and all((c.root_pc + i) % 12 in scale for i in c.intervals)

def test_progressions_avoid_immediate_repeat_mostly_and_are_seeded():
    a = Harmony(np.random.default_rng(7), MOODS["dreamy"])
    b = Harmony(np.random.default_rng(7), MOODS["dreamy"])
    seq_a = [tuple(c.degree for c in a.next_progression()) for _ in range(20)]
    seq_b = [tuple(c.degree for c in b.next_progression()) for _ in range(20)]
    assert seq_a == seq_b
    assert len(set(seq_a)) > 2 and all(p in [tuple(v) for v in PROGRESSIONS.values()] for p in seq_a)

def test_modulate_changes_tonic_and_scale_notes_in_range():
    h = Harmony(np.random.default_rng(0), MOODS["dark"])
    t = h.tonic
    h.modulate()
    assert h.tonic != t
    notes = h.scale_notes(60, 72)
    assert notes and all(60 <= n <= 72 for n in notes)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_harmony.py -v` — Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# synthwave/composer/moods.py
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mood:
    name: str
    bpm: float
    drum_density: float     # 0..1 hats/kick extras
    arp_prob: float         # probability the arp layer plays in verse
    lead_prob: float        # probability of a lead phrase per chorus bar
    brightness: float       # multiplier applied to filter cutoffs
    major_prob: float       # probability of a major key
    progressions: dict[str, float] = field(default_factory=dict)


MOODS: dict[str, Mood] = {
    "dark": Mood("dark", 100, 0.5, 0.55, 0.25, 0.7, 0.0,
                 {"i-VI-III-VII": 2, "i-VII-VI-VII": 3, "i-iv-VI-V": 2, "i-VI-VII-i": 2,
                  "VI-VII-i-i": 1, "i-III-VII-VI": 1, "iv-VI-i-VII": 2}),
    "dreamy": Mood("dreamy", 108, 0.5, 0.75, 0.4, 1.0, 0.35,
                   {"i-VI-III-VII": 3, "i-VII-VI-VII": 2, "i-iv-VI-V": 1, "i-VI-VII-i": 2,
                    "VI-VII-i-i": 2, "i-III-VII-VI": 2, "iv-VI-i-VII": 1}),
    "outrun": Mood("outrun", 118, 0.85, 0.95, 0.55, 1.2, 0.1,
                   {"i-VI-III-VII": 3, "i-VII-VI-VII": 3, "i-iv-VI-V": 2, "i-VI-VII-i": 2,
                    "VI-VII-i-i": 2, "i-III-VII-VI": 1, "iv-VI-i-VII": 1}),
}
```

```python
# synthwave/composer/harmony.py
"""Key, chords with sevenths, and a Markov chain over synthwave progressions."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .moods import Mood

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
PROGRESSIONS: dict[str, tuple[int, ...]] = {
    "i-VI-III-VII": (0, 5, 2, 6), "i-VII-VI-VII": (0, 6, 5, 6), "i-iv-VI-V": (0, 3, 5, 4),
    "i-VI-VII-i": (0, 5, 6, 0), "VI-VII-i-i": (5, 6, 0, 0), "i-III-VII-VI": (0, 2, 6, 5),
    "iv-VI-i-VII": (3, 5, 0, 6),
}
_QUALITY = {(0, 3, 7, 10): "m7", (0, 4, 7, 11): "maj7", (0, 4, 7, 10): "7",
            (0, 3, 6, 10): "m7b5", (0, 4, 8, 11): "maj7#5"}


@dataclass(frozen=True)
class Chord:
    root_pc: int
    degree: int
    intervals: tuple[int, ...]

    def notes(self, octave: int = 4) -> list[int]:
        root = 12 * (octave + 1) + self.root_pc
        return [root + i for i in self.intervals]

    def bass_note(self) -> int:
        return 12 * 3 + self.root_pc  # octave 2 (36..47)

    @property
    def name(self) -> str:
        return NOTE_NAMES[self.root_pc % 12] + _QUALITY.get(self.intervals, "")


class Harmony:
    MINOR = (0, 2, 3, 5, 7, 8, 10)
    MAJOR = (0, 2, 4, 5, 7, 9, 11)

    def __init__(self, rng: np.random.Generator, mood: Mood):
        self.rng, self.mood = rng, mood
        self.tonic = int(rng.integers(0, 12))
        self.mode = self.MAJOR if rng.random() < mood.major_prob else self.MINOR
        self.current: str | None = None

    def set_mood(self, mood: Mood) -> None:
        self.mood = mood

    @property
    def key_name(self) -> str:
        return NOTE_NAMES[self.tonic] + (" minor" if self.mode == self.MINOR else " major")

    def chord_for_degree(self, deg: int) -> Chord:
        s = self.mode
        root = s[deg % 7]
        intervals = tuple((s[(deg + k) % 7] - root) % 12 for k in (0, 2, 4, 6))
        return Chord((self.tonic + root) % 12, deg % 7, intervals)

    def next_progression(self) -> list[Chord]:
        names = [n for n in PROGRESSIONS if self.mood.progressions.get(n, 0) > 0]
        w = np.array([self.mood.progressions[n] * (0.25 if n == self.current else 1.0)
                      for n in names], dtype=float)
        self.current = names[int(self.rng.choice(len(names), p=w / w.sum()))]
        return [self.chord_for_degree(d) for d in PROGRESSIONS[self.current]]

    def modulate(self) -> None:
        self.tonic = (self.tonic + int(self.rng.choice([5, 7, -3, 3]))) % 12

    def scale_notes(self, low: int, high: int) -> list[int]:
        pcs = {(self.tonic + i) % 12 for i in self.mode}
        return [n for n in range(low, high + 1) if n % 12 in pcs]
```

- [ ] **Step 4: Run** `uv run pytest tests/test_harmony.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat(composer): moods and Markov harmony"`

### Task 9: Générateurs de patterns

**Files:**
- Create: `synthwave/composer/patterns.py`
- Test: `tests/test_patterns.py`

**Interfaces:**
- Consumes: `Chord`.
- Produces: `Note(step: int, note: int, vel: float, length: int)` ; `Pattern = list[Note]` ; `STEPS = 16` ; `gen_drums(rng, density, fill=False, snare=True, crash=False) -> Pattern` ; `gen_bass(rng, chord, style: str) -> Pattern` (styles `eighths|octaves|syncopated`) ; `gen_arp(rng, chord, mode: str, octaves=2) -> Pattern` (modes `up|updown|random`) ; `gen_pad(chord) -> Pattern` ; `gen_lead(rng, chord, scale_notes, density) -> Pattern` ; `gen_ambient(chord) -> Pattern` ; `mutate(rng, pattern, rate, allowed_notes) -> Pattern`.

- [ ] **Step 1: Tests**

```python
# tests/test_patterns.py
import numpy as np
from synthwave.composer.harmony import Chord
from synthwave.composer.patterns import (STEPS, gen_ambient, gen_arp, gen_bass, gen_drums, gen_lead,
                                         gen_pad, mutate)

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
    assert any(n.note == 49 and n.step == 0 for n in gen_drums(np.random.default_rng(0), 0.5, crash=True))

def test_bass_styles_use_chord_root():
    for style in ("eighths", "octaves", "syncopated"):
        p = gen_bass(np.random.default_rng(1), AM7, style)
        assert p and in_range(p) and all(n.note % 12 == 9 or n.note % 12 == 4 for n in p)

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
    allowed = [57, 60, 64, 67]
    m = mutate(np.random.default_rng(5), base, 0.2, allowed)
    assert in_range(m) and all(n.note in allowed for n in m)
    changed = sum(1 for a, b in zip(base, m) if a != b) + abs(len(base) - len(m))
    assert 0 < changed <= 8
```

- [ ] **Step 2: Run** `uv run pytest tests/test_patterns.py -v` — Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# synthwave/composer/patterns.py
"""Per-layer 16-step pattern generators and a bounded mutation operator."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .harmony import Chord

STEPS = 16
KICK, SNARE, CLAP, HAT_C, TOM_L, HAT_O, TOM_M, CRASH = 36, 38, 39, 42, 45, 46, 47, 49


@dataclass(frozen=True)
class Note:
    step: int
    note: int
    vel: float
    length: int = 1


Pattern = list[Note]


def _sorted(p: Pattern) -> Pattern:
    return sorted(p, key=lambda n: (n.step, n.note))


def gen_drums(rng: np.random.Generator, density: float, fill: bool = False,
              snare: bool = True, crash: bool = False) -> Pattern:
    p: Pattern = [Note(s, KICK, 1.0) for s in (0, 4, 8, 12)]
    if rng.random() < density * 0.5:
        p.append(Note(int(rng.choice([10, 14, 7])), KICK, 0.8))
    if snare:
        p += [Note(4, SNARE, 1.0), Note(12, SNARE, 1.0)]
        if rng.random() < density * 0.4:
            p += [Note(4, CLAP, 0.7), Note(12, CLAP, 0.7)]
    for s in range(0, STEPS, 2):
        p.append(Note(s, HAT_C, 0.8 if s % 4 == 0 else 0.55))
    for s in range(1, STEPS, 2):
        if rng.random() < density * 0.7:
            p.append(Note(s, HAT_C, 0.4))
    if rng.random() < density * 0.6:
        p.append(Note(14, HAT_O, 0.6))
    if fill:
        p = [n for n in p if n.step < 12]
        roll = rng.choice([SNARE, TOM_M, TOM_L], size=4)
        p += [Note(12 + i, int(roll[i]), 0.6 + 0.1 * i) for i in range(4)]
        if rng.random() < 0.5:
            p += [Note(13, SNARE, 0.5), Note(15, SNARE, 0.9)]
    if crash:
        p.append(Note(0, CRASH, 0.8))
    return _sorted(p)


def gen_bass(rng: np.random.Generator, chord: Chord, style: str) -> Pattern:
    root = chord.bass_note()
    fifth = root + 7
    if style == "eighths":
        p = [Note(s, root, 1.0 if s % 4 == 0 else 0.8, 2) for s in range(0, STEPS, 2)]
    elif style == "octaves":
        p = [Note(s, root + (12 if (s // 2) % 2 else 0), 0.9, 1) for s in range(0, STEPS, 2)]
    else:  # syncopated
        steps = (0, 3, 6, 8, 11, 14)
        p = [Note(s, root, 0.9, 2) for s in steps]
        if rng.random() < 0.5:
            p[-1] = Note(14, fifth, 0.8, 2)
    if rng.random() < 0.3:
        p.append(Note(15, root + 12, 0.6, 1))
    return _sorted(p)


def gen_arp(rng: np.random.Generator, chord: Chord, mode: str, octaves: int = 2) -> Pattern:
    tones = [n for o in range(octaves) for n in chord.notes(4 + o)]
    if mode == "up":
        seq = [tones[i % len(tones)] for i in range(STEPS)]
    elif mode == "updown":
        cyc = tones + tones[-2:0:-1]
        seq = [cyc[i % len(cyc)] for i in range(STEPS)]
    else:
        seq = [int(rng.choice(tones)) for _ in range(STEPS)]
    return [Note(s, seq[s], 0.85 if s % 4 == 0 else 0.7, 1) for s in range(STEPS)]


def gen_pad(chord: Chord) -> Pattern:
    notes = chord.notes(4)
    if chord.root_pc >= 6:  # keep voicing low: drop the root an octave
        notes = [notes[0] - 12] + notes[1:]
    return [Note(0, n, 0.7, STEPS) for n in notes]


def gen_ambient(chord: Chord) -> Pattern:
    root = chord.notes(3)[0]
    return [Note(0, root, 0.6, STEPS), Note(0, root + 7, 0.4, STEPS)]


def gen_lead(rng: np.random.Generator, chord: Chord, scale_notes: list[int],
             density: float) -> Pattern:
    if not scale_notes:
        return []
    chord_pcs = {(chord.root_pc + i) % 12 for i in chord.intervals}
    grid = [0, 2, 3, 4, 6, 8, 10, 11, 12, 14]
    count = max(1, min(len(grid), int(round(2 + density * 4))))
    steps = sorted(int(s) for s in rng.choice(grid, size=count, replace=False))
    p: Pattern = []
    prev = int(rng.choice(scale_notes))
    for i, s in enumerate(steps):
        near = [n for n in scale_notes if abs(n - prev) <= 5] or scale_notes
        strong = [n for n in near if n % 12 in chord_pcs]
        note = int(rng.choice(strong if (s % 4 == 0 and strong) else near))
        nxt = steps[i + 1] if i + 1 < len(steps) else STEPS
        p.append(Note(s, note, 0.8, max(1, min(4, nxt - s))))
        prev = note
    return p


def mutate(rng: np.random.Generator, pattern: Pattern, rate: float,
           allowed_notes: list[int]) -> Pattern:
    out: Pattern = []
    taken = {n.step for n in pattern}
    for n in pattern:
        r = rng.random()
        if r < rate * 0.3:
            continue  # drop
        if r < rate * 0.6:
            s = int(np.clip(n.step + rng.choice([-1, 1]), 0, STEPS - 1))
            if s not in taken:
                taken.add(s)
                out.append(Note(s, n.note, n.vel, n.length))
                continue
        if r < rate and allowed_notes:
            out.append(Note(n.step, int(rng.choice(allowed_notes)), n.vel, n.length))
            continue
        out.append(n)
    if allowed_notes and rng.random() < rate * 0.5:
        free = [s for s in range(STEPS) if s not in taken]
        if free:
            out.append(Note(int(rng.choice(free)), int(rng.choice(allowed_notes)), 0.6, 1))
    return _sorted(out)
```

- [ ] **Step 4: Run** `uv run pytest tests/test_patterns.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat(composer): pattern generators and mutation"`

### Task 10: Arrangeur par sections

**Files:**
- Create: `synthwave/composer/arranger.py`
- Test: `tests/test_arranger.py`

**Interfaces:**
- Consumes: `Harmony`, `Mood`, `gen_*`, `mutate`.
- Produces: `LAYERS = ("drums", "bass", "arp", "pad", "lead", "ambient")` ; `Section(str, Enum)` INTRO/VERSE/CHORUS/BREAK/OUTRO ; `SECTION_BARS` ; `BarPlan(bar, section, section_bar, chord, patterns: dict[str, Pattern], gains: dict[str, float], fill: bool, fade: float, finished: bool)` ; `Arranger(rng, harmony, mood, total_bars=None)` : `next_bar() -> BarPlan`, `force_next_section()`, `set_mood(mood)`, `section`, `bar`.

- [ ] **Step 1: Tests**

```python
# tests/test_arranger.py
import numpy as np
from synthwave.composer.arranger import LAYERS, Arranger, Section
from synthwave.composer.harmony import Harmony
from synthwave.composer.moods import MOODS

def make(seed=0, total_bars=None, mood="outrun"):
    rng = np.random.default_rng(seed)
    m = MOODS[mood]
    return Arranger(rng, Harmony(rng, m), m, total_bars)

def test_starts_with_intro_then_verse():
    a = make()
    plans = [a.next_bar() for _ in range(9)]
    assert plans[0].section == Section.INTRO and plans[7].section == Section.INTRO
    assert plans[8].section == Section.VERSE and plans[8].section_bar == 0

def test_plan_has_all_layers_and_gains():
    p = make().next_bar()
    assert set(p.patterns) == set(LAYERS) and set(p.gains) == set(LAYERS)
    assert p.gains["pad"] > 0 and p.gains["lead"] == 0

def test_fill_on_last_bar_and_no_identical_consecutive_drums():
    a = make(seed=2)
    plans = [a.next_bar() for _ in range(40)]
    assert plans[7].fill and not plans[6].fill
    for x, y in zip(plans, plans[1:]):
        if x.section == y.section and x.gains["drums"] > 0 and y.gains["drums"] > 0:
            assert x.patterns["drums"] != y.patterns["drums"] or x.patterns["bass"] != y.patterns["bass"] or x.chord != y.chord

def test_duration_mode_ends_with_outro_and_finished():
    a = make(seed=1, total_bars=20)
    plans = [a.next_bar() for _ in range(22)]
    assert plans[12].section == Section.OUTRO and plans[19].section == Section.OUTRO
    assert plans[19].fade < plans[12].fade
    assert plans[20].finished and plans[21].finished and all(g == 0 for g in plans[20].gains.values())

def test_force_next_section():
    a = make()
    a.next_bar(); a.force_next_section()
    assert a.next_bar().section == Section.VERSE

def test_deterministic_by_seed():
    a, b = make(seed=9), make(seed=9)
    for _ in range(30):
        pa, pb = a.next_bar(), b.next_bar()
        assert pa == pb
```

- [ ] **Step 2: Run** `uv run pytest tests/test_arranger.py -v` — Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# synthwave/composer/arranger.py
"""Section state machine turning harmony + generators into one BarPlan per bar."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from .harmony import Chord, Harmony
from .moods import Mood
from .patterns import (Pattern, gen_ambient, gen_arp, gen_bass, gen_drums, gen_lead, gen_pad,
                       mutate)

LAYERS = ("drums", "bass", "arp", "pad", "lead", "ambient")


class Section(str, Enum):
    INTRO = "intro"
    VERSE = "verse"
    CHORUS = "chorus"
    BREAK = "break"
    OUTRO = "outro"


SECTION_BARS = {Section.INTRO: 8, Section.VERSE: 16, Section.CHORUS: 16, Section.BREAK: 8,
                Section.OUTRO: 8}
_NEXT = {Section.INTRO: [Section.VERSE], Section.VERSE: [Section.CHORUS, Section.CHORUS, Section.BREAK],
         Section.CHORUS: [Section.VERSE, Section.BREAK, Section.CHORUS],
         Section.BREAK: [Section.CHORUS, Section.CHORUS, Section.VERSE]}
_GAINS = {
    Section.INTRO: dict(drums=0.6, bass=0.8, arp=0.6, pad=1.0, lead=0.0, ambient=1.0),
    Section.VERSE: dict(drums=1.0, bass=1.0, arp=0.85, pad=0.9, lead=0.35, ambient=0.7),
    Section.CHORUS: dict(drums=1.0, bass=1.0, arp=1.0, pad=1.0, lead=1.0, ambient=0.5),
    Section.BREAK: dict(drums=0.0, bass=0.6, arp=0.7, pad=1.0, lead=0.0, ambient=1.0),
    Section.OUTRO: dict(drums=0.7, bass=0.8, arp=0.5, pad=1.0, lead=0.0, ambient=1.0),
}


@dataclass(frozen=True)
class BarPlan:
    bar: int
    section: Section
    section_bar: int
    chord: Chord
    patterns: dict[str, Pattern]
    gains: dict[str, float]
    fill: bool = False
    fade: float = 1.0
    finished: bool = False
    key: str = ""


class Arranger:
    def __init__(self, rng: np.random.Generator, harmony: Harmony, mood: Mood,
                 total_bars: int | None = None):
        self.rng, self.harmony, self.mood, self.total_bars = rng, harmony, mood, total_bars
        self.bar, self.sections_done = 0, 0
        self.section = Section.INTRO
        self.section_bar, self.section_len = 0, SECTION_BARS[Section.INTRO]
        self.progression = harmony.next_progression()
        self.finished = False
        self._new_styles()

    def set_mood(self, mood: Mood) -> None:
        self.mood = mood
        self.harmony.set_mood(mood)

    def force_next_section(self) -> None:
        self.section_bar = self.section_len

    def _new_styles(self) -> None:
        r = self.rng
        self.bass_style = str(r.choice(["eighths", "octaves", "syncopated"], p=[0.5, 0.3, 0.2]))
        self.arp_mode = str(r.choice(["up", "updown", "random"], p=[0.45, 0.4, 0.15]))
        self.arp_on = r.random() < self.mood.arp_prob or self.section == Section.CHORUS
        density = self.mood.drum_density * (1.0 if self.section != Section.INTRO else 0.3)
        self.drums_base = gen_drums(r, density, snare=self.section != Section.INTRO)

    def _start_section(self) -> None:
        self.sections_done += 1
        if self.total_bars is not None and self.bar >= self.total_bars - SECTION_BARS[Section.OUTRO]:
            self.section = Section.OUTRO
        else:
            self.section = Section(self.rng.choice([s.value for s in _NEXT[self.section]]))
        self.section_bar, self.section_len = 0, SECTION_BARS[self.section]
        if self.sections_done % 6 == 0:
            self.harmony.modulate()
        if self.rng.random() < 0.6 or self.sections_done % 6 == 0:
            self.progression = self.harmony.next_progression()
        self._new_styles()

    def next_bar(self) -> BarPlan:
        if self.finished or (self.total_bars is not None and self.bar >= self.total_bars):
            self.finished = True
            chord = self.progression[0]
            plan = BarPlan(self.bar, Section.OUTRO, 0, chord, {l: [] for l in LAYERS},
                           {l: 0.0 for l in LAYERS}, fade=0.0, finished=True, key=self.harmony.key_name)
            self.bar += 1
            return plan
        if self.section_bar >= self.section_len:
            self._start_section()
        elif (self.total_bars is not None and self.section != Section.OUTRO
              and self.bar >= self.total_bars - SECTION_BARS[Section.OUTRO]):
            self.section_bar = self.section_len
            self._start_section()
        r = self.rng
        chord = self.progression[self.section_bar % len(self.progression)]
        last = self.section_bar == self.section_len - 1
        first = self.section_bar == 0
        density = self.mood.drum_density * (0.3 if self.section == Section.INTRO else 1.0)
        if last and self.section != Section.OUTRO:
            drums = gen_drums(r, density, fill=True, snare=self.section != Section.INTRO)
        else:
            if self.section_bar % 4 == 0 and not first:
                self.drums_base = mutate(r, self.drums_base, 0.15, [42])
            drums = self.drums_base
            if first and self.section in (Section.VERSE, Section.CHORUS):
                drums = drums + [type(drums[0])(0, 49, 0.8, 1)] if drums else drums
        gains = dict(_GAINS[self.section])
        if self.section == Section.BREAK and last:
            gains["drums"] = 1.0
        if not self.arp_on:
            gains["arp"] = 0.0
        lead_p = self.mood.lead_prob * (1.0 if self.section == Section.CHORUS else 0.5)
        scale = self.harmony.scale_notes(72, 84)
        lead = gen_lead(r, chord, scale, density) if r.random() < lead_p else []
        patterns = {
            "drums": drums,
            "bass": gen_bass(r, chord, self.bass_style),
            "arp": gen_arp(r, chord, self.arp_mode) if self.arp_on else [],
            "pad": gen_pad(chord),
            "lead": lead,
            "ambient": gen_ambient(chord),
        }
        fade = 1.0 - self.section_bar / self.section_len if self.section == Section.OUTRO else 1.0
        plan = BarPlan(self.bar, self.section, self.section_bar, chord, patterns, gains,
                       fill=last, fade=fade, key=self.harmony.key_name)
        self.bar += 1
        self.section_bar += 1
        return plan
```

- [ ] **Step 4: Run** `uv run pytest tests/test_arranger.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat(composer): section arranger with duration mode"`

### Task 11: Transport + Tracker

**Files:**
- Create: `synthwave/sequencer/__init__.py`, `synthwave/sequencer/transport.py`, `synthwave/sequencer/tracker.py`
- Test: `tests/test_tracker.py`

**Interfaces:**
- Consumes: `Arranger`, `BarPlan`, `NoteEvent`, `LAYERS`.
- Produces: `StepTick(bar, step, offset)` ; `Transport(sr, bpm)` : `bpm`, `clock: int`, `samples_per_step: float`, `bar_seconds: float`, `set_bpm(bpm)`, `advance(n) -> list[StepTick]`, `step_samples(k) -> int` ; `Tracker(transport, arranger)` : `advance(n) -> tuple[dict[str, list[NoteEvent]], BarPlan | None]`, `plan: BarPlan | None`.

- [ ] **Step 1: Tests**

```python
# tests/test_tracker.py
import numpy as np
from synthwave.composer.arranger import Arranger, BarPlan, Section, LAYERS
from synthwave.composer.harmony import Chord
from synthwave.composer.patterns import Note
from synthwave.sequencer.tracker import Tracker
from synthwave.sequencer.transport import Transport

SR = 44100

def test_transport_ticks_at_expected_offsets():
    t = Transport(SR, 120)  # step = 0.125 s = 5512.5 samples
    ticks = t.advance(12000)
    assert [(k.step, k.offset) for k in ticks] == [(0, 0), (1, 5512), (2, 11025)]
    assert t.advance(5000)[0].step == 3 and t.clock == 17000

def test_transport_bpm_change_keeps_phase():
    t = Transport(SR, 120)
    t.advance(1000); t.set_bpm(60)
    ticks = t.advance(20000)
    assert [k.offset for k in ticks] == [4512, 15537]

class FakeArranger:
    def __init__(self):
        self.calls = 0
    def next_bar(self):
        self.calls += 1
        pats = {l: [] for l in LAYERS}
        pats["bass"] = [Note(0, 45, 1.0, 2), Note(2, 45, 0.8, 1)]
        pats["pad"] = [Note(0, 57, 0.7, 16)]
        return BarPlan(self.calls - 1, Section.VERSE, 0, Chord(9, 0, (0, 3, 7, 10)), pats,
                       {l: 1.0 for l in LAYERS})

def test_tracker_emits_ons_and_offs():
    t = Transport(SR, 120)
    tr = Tracker(t, FakeArranger())
    ev, plan = tr.advance(6000)
    assert plan is not None and plan.bar == 0
    ons = [e for e in ev["bass"] if e.on]
    assert [e.offset for e in ons] == [0]
    ev2, plan2 = tr.advance(6000)
    assert plan2 is None
    offs = [e for e in ev2["bass"] if not e.on]
    assert [e.offset for e in offs] == [11025 - 6000] and any(e.on and e.offset == 11025 - 6000 for e in ev2["bass"])

def test_tracker_pad_off_before_next_on_same_offset():
    t = Transport(SR, 120)
    tr = Tracker(t, FakeArranger())
    bar = int(16 * t.samples_per_step)
    tr.advance(bar)
    ev, plan = tr.advance(100)
    pads = sorted(ev["pad"])
    assert plan.bar == 1 and [e.on for e in pads] == [False, True] and pads[0].offset == 0
```

- [ ] **Step 2: Run** `uv run pytest tests/test_tracker.py -v` — Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# synthwave/sequencer/transport.py
from __future__ import annotations
from dataclasses import dataclass

STEPS_PER_BAR = 16


@dataclass(frozen=True)
class StepTick:
    bar: int
    step: int
    offset: int


class Transport:
    def __init__(self, sr: int, bpm: float):
        self.sr, self.bpm = sr, float(bpm)
        self.clock = 0
        self.next_step_time = 0.0
        self.step_index = 0

    @property
    def samples_per_step(self) -> float:
        return self.sr * 60.0 / self.bpm / 4.0

    @property
    def bar_seconds(self) -> float:
        return STEPS_PER_BAR * 60.0 / self.bpm / 4.0

    def set_bpm(self, bpm: float) -> None:
        self.bpm = float(bpm)

    def step_samples(self, k: float) -> int:
        return int(k * self.samples_per_step)

    def advance(self, n: int) -> list[StepTick]:
        ticks = []
        end = self.clock + n
        while self.next_step_time < end:
            ticks.append(StepTick(self.step_index // STEPS_PER_BAR, self.step_index % STEPS_PER_BAR,
                                  int(self.next_step_time - self.clock)))
            self.step_index += 1
            self.next_step_time += self.samples_per_step
        self.clock = end
        return ticks
```

```python
# synthwave/sequencer/tracker.py
from __future__ import annotations
from ..composer.arranger import LAYERS, BarPlan
from ..engine.events import NoteEvent
from .transport import Transport


class Tracker:
    def __init__(self, transport: Transport, arranger):
        self.transport, self.arranger = transport, arranger
        self.plan: BarPlan | None = None
        self.pending: list[tuple[int, str, int]] = []  # (absolute off time, layer, note)

    def advance(self, n: int) -> tuple[dict[str, list[NoteEvent]], BarPlan | None]:
        events: dict[str, list[NoteEvent]] = {layer: [] for layer in LAYERS}
        base = self.transport.clock
        new_plan = None
        for tick in self.transport.advance(n):
            if tick.step == 0:
                self.plan = self.arranger.next_bar()
                new_plan = self.plan
            if self.plan is None:
                continue
            for layer, pattern in self.plan.patterns.items():
                for note in pattern:
                    if note.step == tick.step:
                        events[layer].append(NoteEvent(tick.offset, note.note, note.vel, True))
                        off = base + tick.offset + self.transport.step_samples(note.length)
                        self.pending.append((off, layer, note.note))
        end = base + n
        due = [p for p in self.pending if p[0] < end]
        self.pending = [p for p in self.pending if p[0] >= end]
        for t, layer, note in due:
            events[layer].append(NoteEvent(max(0, t - base), note, 0.0, False))
        return events, new_plan
```

- [ ] **Step 4: Run** `uv run pytest tests/test_tracker.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat(sequencer): transport and tracker"`

### Task 12: Renderer + export WAV

**Files:**
- Create: `synthwave/audio/__init__.py`, `synthwave/audio/renderer.py`, `synthwave/audio/export.py`
- Test: `tests/test_renderer.py`

**Interfaces:**
- Consumes: tout ce qui précède.
- Produces: `RenderConfig(sr=44100, bpm=None, mood="dark", seed=None, duration_s=None, patches={})` ; `Renderer(cfg)` : `render(n) -> (n,2) float32`, `finished: bool`, `submit(fn)`, `set_tempo(bpm)`, `set_mood(name)`, `set_layer(layer, mute=None, solo=None, volume=None)`, `load_patch(layer, name)`, `set_patch_param(layer, path, value)`, `next_section()`, `status() -> dict`, `seed`, `bpm`, `sr` ; `export_wav(renderer, seconds, path, blocksize=1024) -> int` (samples écrits).

- [ ] **Step 1: Tests**

```python
# tests/test_renderer.py
import hashlib
import numpy as np
import soundfile as sf
from synthwave.audio.export import export_wav
from synthwave.audio.renderer import RenderConfig, Renderer
from synthwave.patches.loader import PatchError
import pytest

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
        r.render(4096); total += 4096
    assert r.finished and 10 * r.sr <= total <= 30 * r.sr

def test_commands_and_status():
    r = Renderer(RenderConfig(seed=5))
    r.submit(lambda: r.set_tempo(125))
    r.submit(lambda: r.set_layer("lead", mute=True))
    r.submit(lambda: r.set_layer("pad", volume=0.5))
    r.submit(lambda: r.load_patch("bass", "lead_saw"))
    r.render(1024)
    st = r.status()
    assert st["bpm"] == 125 and st["layers"]["lead"]["muted"] and st["layers"]["pad"]["volume"] == 0.5
    assert st["layers"]["bass"]["patch"] == "lead_saw" and st["section"] == "intro" and st["seed"] == 5
    r.set_mood("dreamy"); r.next_section(); r.render(1024)
    assert r.status()["mood"] == "dreamy"

def test_bad_patch_keeps_state():
    r = Renderer(RenderConfig(seed=5))
    with pytest.raises(PatchError):
        r.load_patch("drums", "pad_juno")
    with pytest.raises(PatchError):
        r.set_patch_param("pad", "filter.cutoff", "nope")
    assert r.status()["layers"]["drums"]["patch"] == "drums_808"
    r.set_patch_param("pad", "filter.cutoff", 700)
    assert r.instruments["pad"].patch.filter.cutoff == 700

def test_export_wav(tmp_path):
    r = Renderer(RenderConfig(seed=2, duration_s=4))
    out = tmp_path / "x.wav"
    n = export_wav(r, 4, str(out))
    data, sr = sf.read(str(out))
    assert sr == 44100 and len(data) == n and data.shape[1] == 2 and n >= 4 * 44100
```

- [ ] **Step 2: Run** `uv run pytest tests/test_renderer.py -v` — Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# synthwave/audio/renderer.py
"""Mixes all layers into a stereo block; owns the command queue used by CLI/MCP threads."""
from __future__ import annotations
import math
import queue
from dataclasses import dataclass, field
import numpy as np
from ..composer.arranger import LAYERS, Arranger, BarPlan
from ..composer.harmony import Harmony
from ..composer.moods import MOODS
from ..engine.drums import DrumKit
from ..engine.effects import Limiter, Sidechain
from ..engine.synth import Synth
from ..patches.loader import PatchError, load_patch, set_param
from ..patches.model import DrumPatchModel, PatchModel
from ..sequencer.tracker import Tracker
from ..sequencer.transport import Transport

DEFAULT_PATCHES = {"drums": "drums_808", "bass": "bass_moog", "arp": "arp_pluck",
                   "pad": "pad_juno", "lead": "lead_saw", "ambient": "ambient_drone"}
DUCKED = {"pad": 1.0, "bass": 1.0, "ambient": 0.6, "arp": 0.5}


@dataclass
class RenderConfig:
    sr: int = 44100
    bpm: float | None = None
    mood: str = "dark"
    seed: int | None = None
    duration_s: float | None = None
    patches: dict[str, str] = field(default_factory=dict)


class Renderer:
    def __init__(self, cfg: RenderConfig):
        if cfg.mood not in MOODS:
            raise ValueError(f"unknown mood {cfg.mood!r}, choose from {list(MOODS)}")
        self.cfg, self.sr = cfg, cfg.sr
        self.seed = int(cfg.seed) if cfg.seed is not None else int(np.random.SeedSequence().entropy % 2**31)
        self.rng = np.random.default_rng(self.seed)
        self.mood = MOODS[cfg.mood]
        self.bpm = float(cfg.bpm or self.mood.bpm)
        self.transport = Transport(self.sr, self.bpm)
        total_bars = (math.ceil(cfg.duration_s / self.transport.bar_seconds)
                      if cfg.duration_s else None)
        self.arranger = Arranger(self.rng, Harmony(self.rng, self.mood), self.mood, total_bars)
        self.tracker = Tracker(self.transport, self.arranger)
        self.instruments: dict[str, Synth | DrumKit] = {}
        self.patch_names: dict[str, str] = {}
        for layer in LAYERS:
            name = cfg.patches.get(layer, DEFAULT_PATCHES[layer])
            self._install(layer, name, load_patch(name))
        self.layer_volume = {l: 1.0 for l in LAYERS}
        self.muted: set[str] = set()
        self.solo: set[str] = set()
        self.plan_gain = {l: 0.0 for l in LAYERS}
        self.current_gain = {l: 0.0 for l in LAYERS}
        self.sidechain = Sidechain(self.sr, depth=0.45, release=0.22)
        self.limiter = Limiter(self.sr, self.bpm, threshold=0.95)
        self.master_volume = 0.9
        self.fade_target, self.fade = 1.0, 1.0
        self.commands: queue.SimpleQueue = queue.SimpleQueue()
        self.plan: BarPlan | None = None
        self.finished = False
        self.rendered = 0

    # ----- instruments -----
    def _install(self, layer: str, name: str, patch) -> None:
        if layer == "drums":
            if not isinstance(patch, DrumPatchModel):
                raise PatchError(f"layer 'drums' needs a drum patch, got {patch.name!r}")
            inst = self.instruments.get(layer)
            if inst is None:
                inst = DrumKit(patch, self.sr, self.rng)
            else:
                inst.set_patch(patch)
        else:
            if not isinstance(patch, PatchModel):
                raise PatchError(f"layer {layer!r} needs a synth patch, got {patch.name!r}")
            inst = self.instruments.get(layer)
            if inst is None:
                inst = Synth(patch, self.sr, self.rng, self.bpm)
            else:
                inst.set_patch(patch)
        self.instruments[layer] = inst
        self.patch_names[layer] = name

    # ----- commands (thread-safe through submit) -----
    def submit(self, fn) -> None:
        self.commands.put(fn)

    def _drain(self) -> None:
        while True:
            try:
                fn = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception as e:  # never kill the audio thread
                print(f"[synthwave] command failed: {e}")

    def set_tempo(self, bpm: float) -> None:
        self.bpm = float(np.clip(bpm, 60, 180))
        self.transport.set_bpm(self.bpm)
        for inst in self.instruments.values():
            inst.set_bpm(self.bpm)

    def set_mood(self, name: str) -> None:
        if name not in MOODS:
            raise ValueError(f"unknown mood {name!r}")
        self.mood = MOODS[name]
        self.arranger.set_mood(self.mood)

    def set_layer(self, layer: str, mute: bool | None = None, solo: bool | None = None,
                  volume: float | None = None) -> None:
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}, choose from {LAYERS}")
        if mute is not None:
            (self.muted.add if mute else self.muted.discard)(layer)
        if solo is not None:
            (self.solo.add if solo else self.solo.discard)(layer)
        if volume is not None:
            self.layer_volume[layer] = float(np.clip(volume, 0.0, 2.0))

    def load_patch(self, layer: str, name: str) -> None:
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}")
        self._install(layer, name, load_patch(name))

    def set_patch_param(self, layer: str, path: str, value) -> None:
        inst = self.instruments[layer]
        self._install(layer, self.patch_names[layer], set_param(inst.patch, path, value))

    def next_section(self) -> None:
        self.arranger.force_next_section()

    def status(self) -> dict:
        p = self.plan
        return {
            "bpm": self.bpm, "mood": self.mood.name, "seed": self.seed,
            "bar": p.bar if p else 0, "section": p.section.value if p else "intro",
            "chord": p.chord.name if p else "", "key": p.key if p else self.arranger.harmony.key_name,
            "elapsed_s": round(self.rendered / self.sr, 1), "finished": self.finished,
            "layers": {l: {"gain": round(self._effective_gain(l), 3), "muted": l in self.muted,
                           "solo": l in self.solo, "volume": self.layer_volume[l],
                           "patch": self.patch_names[l]} for l in LAYERS},
        }

    # ----- rendering -----
    def _effective_gain(self, layer: str) -> float:
        if layer in self.muted or (self.solo and layer not in self.solo):
            return 0.0
        return self.plan_gain[layer] * self.layer_volume[layer]

    def render(self, n: int) -> np.ndarray:
        self._drain()
        events, plan = self.tracker.advance(n)
        if plan is not None:
            self.plan = plan
            self.plan_gain = dict(plan.gains)
            self.fade_target = plan.fade
            if plan.finished:
                self.finished = True
        kicks = [e.offset for e in events["drums"] if e.on and e.note == 36]
        duck = self.sidechain.gain(n, kicks)
        mix = np.zeros((n, 2), dtype=np.float32)
        for layer in LAYERS:
            target = self._effective_gain(layer)
            g0 = self.current_gain[layer]
            sig = self.instruments[layer].render(n, events[layer])
            if layer in DUCKED:
                sig = sig * (1.0 - DUCKED[layer] * (1.0 - duck))[:, None]
            mix += sig * np.linspace(g0, target, n, endpoint=False)[:, None].astype(np.float32)
            self.current_gain[layer] = target
        fade = np.linspace(self.fade, self.fade_target, n, endpoint=False)[:, None]
        self.fade = self.fade_target
        mix = mix * (self.master_volume * fade).astype(np.float32)
        self.rendered += n
        return self.limiter.process(mix)
```

```python
# synthwave/audio/export.py
from __future__ import annotations
import numpy as np
import soundfile as sf
from .renderer import Renderer


def export_wav(renderer: Renderer, seconds: float, path: str, blocksize: int = 1024) -> int:
    """Render offline until `seconds` reached (and, in duration mode, until the outro finishes)."""
    target = int(seconds * renderer.sr)
    written = 0
    with sf.SoundFile(path, "w", samplerate=renderer.sr, channels=2, subtype="PCM_16") as f:
        while written < target or (renderer.cfg.duration_s and not renderer.finished):
            block = renderer.render(blocksize)
            f.write(block)
            written += len(block)
            if renderer.finished and written >= target:
                break
            if written > target + 60 * renderer.sr:
                break
    return written
```

- [ ] **Step 4: Run** `uv run pytest tests/test_renderer.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat(audio): renderer with sidechain, limiter, commands and WAV export"`

### Task 13: Player sounddevice + CLI

**Files:**
- Create: `synthwave/audio/output.py`, `synthwave/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `Player(renderer, blocksize=1024, prefill=6, device=None)` : `start()`, `stop()`, `wait(timeout=None)`, `running: bool`, `underruns: int`, `error: Exception | None` ; `parse_duration(text: str) -> float` ; typer `app` avec commandes `play`, `patches`, `devices`, `mcp`.

- [ ] **Step 1: Tests**

```python
# tests/test_cli.py
import pytest
from typer.testing import CliRunner
from synthwave.cli import app, parse_duration

def test_parse_duration():
    assert parse_duration("90") == 90 and parse_duration("90s") == 90
    assert parse_duration("5m") == 300 and parse_duration("1h30m") == 5400
    with pytest.raises(ValueError):
        parse_duration("abc")

def test_patches_command_lists_library():
    r = CliRunner().invoke(app, ["patches"])
    assert r.exit_code == 0 and "pad_juno" in r.output and "drums_808" in r.output

def test_play_export_offline(tmp_path):
    out = tmp_path / "o.wav"
    r = CliRunner().invoke(app, ["play", "--duration", "3s", "--seed", "1", "--export", str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists() and out.stat().st_size > 100000
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py -v` — Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# synthwave/audio/output.py
"""Producer thread renders ahead into a queue; the sounddevice callback only copies blocks."""
from __future__ import annotations
import queue
import threading
import time
import numpy as np
from .renderer import Renderer


class Player:
    def __init__(self, renderer: Renderer, blocksize: int = 1024, prefill: int = 6, device=None):
        self.renderer, self.blocksize, self.prefill, self.device = renderer, blocksize, prefill, device
        self.queue: queue.Queue = queue.Queue(maxsize=prefill)
        self.stop_event = threading.Event()
        self.done_event = threading.Event()
        self.underruns = 0
        self.error: Exception | None = None
        self.thread: threading.Thread | None = None
        self.stream = None

    def _produce(self) -> None:
        try:
            while not self.stop_event.is_set():
                if self.renderer.finished:
                    break
                block = self.renderer.render(self.blocksize)
                while not self.stop_event.is_set():
                    try:
                        self.queue.put(block, timeout=0.2)
                        break
                    except queue.Full:
                        continue
        except Exception as e:  # surface, then let the callback stop
            self.error = e
            self.stop_event.set()

    def _callback(self, outdata, frames, time_info, status) -> None:
        import sounddevice as sd
        if status and status.output_underflow:
            self.underruns += 1
        try:
            outdata[:] = self.queue.get_nowait()
        except queue.Empty:
            outdata.fill(0)
            if self.stop_event.is_set() or (self.renderer.finished and self.queue.empty()):
                raise sd.CallbackStop
            self.underruns += 1

    def start(self) -> None:
        import sounddevice as sd
        self.thread = threading.Thread(target=self._produce, daemon=True, name="synthwave-render")
        self.thread.start()
        deadline = time.time() + 5
        while self.queue.qsize() < min(self.prefill, 2) and time.time() < deadline and not self.error:
            time.sleep(0.01)
        if self.error:
            raise self.error
        self.stream = sd.OutputStream(samplerate=self.renderer.sr, channels=2, dtype="float32",
                                      blocksize=self.blocksize, device=self.device,
                                      callback=self._callback, finished_callback=self.done_event.set)
        self.stream.start()

    @property
    def running(self) -> bool:
        return self.stream is not None and self.stream.active and not self.done_event.is_set()

    def wait(self, timeout: float | None = None) -> None:
        self.done_event.wait(timeout)

    def stop(self) -> None:
        self.stop_event.set()
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.done_event.set()
```

```python
# synthwave/cli.py
from __future__ import annotations
import re
import time
import typer
from .audio.renderer import RenderConfig, Renderer
from .composer.moods import MOODS
from .patches.loader import list_patches

app = typer.Typer(help="Infinite procedural synthwave generator.", no_args_is_help=True)
_DUR = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?$")


def parse_duration(text: str) -> float:
    m = _DUR.match(text.strip())
    if not m or not any(m.groups()):
        raise ValueError(f"invalid duration {text!r} (examples: 90, 90s, 5m, 1h30m)")
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + s


@app.command()
def play(duration: str | None = typer.Option(None, help="ex: 5m, 90s, 1h. Absent = infini"),
         bpm: float | None = typer.Option(None, min=60, max=180),
         seed: int | None = typer.Option(None),
         mood: str = typer.Option("dark", help=f"{'|'.join(MOODS)}"),
         export: str | None = typer.Option(None, help="Rendu hors-ligne vers un WAV"),
         blocksize: int = typer.Option(1024),
         device: str | None = typer.Option(None, help="Nom ou index du périphérique")):
    """Joue de la synthwave sur la sortie audio (ou exporte en WAV)."""
    seconds = parse_duration(duration) if duration else None
    if export and seconds is None:
        raise typer.BadParameter("--export requires --duration")
    renderer = Renderer(RenderConfig(bpm=bpm, mood=mood, seed=seed, duration_s=seconds))
    typer.echo(f"seed={renderer.seed} bpm={renderer.bpm:g} mood={mood} key={renderer.arranger.harmony.key_name}")
    if export:
        from .audio.export import export_wav
        n = export_wav(renderer, seconds, export, blocksize)
        typer.echo(f"wrote {export} ({n / renderer.sr:.1f}s)")
        return
    from .audio.output import Player
    dev = int(device) if device and device.isdigit() else device
    player = Player(renderer, blocksize=blocksize, device=dev)
    try:
        player.start()
    except Exception as e:
        typer.echo(f"audio output unavailable: {e}\nTry: synthwave play --duration 2m --export out.wav", err=True)
        raise typer.Exit(1)
    typer.echo("playing... Ctrl+C to stop")
    last = None
    try:
        while player.running:
            st = renderer.status()
            line = f"[{st['section']:>6}] bar {st['bar']:>4}  {st['key']:<9} {st['chord']:<6} underruns={player.underruns}"
            if line != last:
                typer.echo(line)
                last = line
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        player.stop()
    if player.error:
        typer.echo(f"render error: {player.error}", err=True)
        raise typer.Exit(1)


@app.command()
def patches():
    """Liste les patches disponibles (bibliothèque + ~/.config/synthwave/patches)."""
    for name in list_patches():
        typer.echo(name)


@app.command()
def devices():
    """Liste les périphériques audio."""
    import sounddevice as sd
    typer.echo(str(sd.query_devices()))


@app.command()
def mcp():
    """Lance le serveur MCP (stdio)."""
    from .mcp_server import main
    main()


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run** `uv run pytest tests/test_cli.py -v` — Expected: PASS.
- [ ] **Step 5: Vérification manuelle** `uv run synthwave play --duration 20s --seed 1 --mood outrun` — son audible, aucun underrun.
- [ ] **Step 6: Commit** `git add -A && git commit -m "feat(audio): sounddevice player and typer CLI"`

### Task 14: Serveur MCP

**Files:**
- Create: `synthwave/mcp_server.py`
- Test: `tests/test_mcp.py`

**Interfaces:**
- Produces: `mcp = FastMCP("infinite-synthwave")` ; outils `start, stop, status, set_tempo, set_mood, set_layer, list_patches, load_patch, set_patch_param, next_section, export_wav` ; `main()`.

- [ ] **Step 1: Tests**

```python
# tests/test_mcp.py
import asyncio
import json
from synthwave import mcp_server

def call(name, **args):
    res = asyncio.run(mcp_server.mcp.call_tool(name, args))
    content = res[0] if isinstance(res, tuple) else res
    return json.loads(content[0].text)

def test_list_patches_tool():
    assert "pad_juno" in call("list_patches")["patches"]

def test_status_when_stopped():
    assert call("status")["running"] is False

def test_export_tool(tmp_path):
    out = tmp_path / "e.wav"
    r = call("export_wav", path=str(out), seconds=2, seed=1, mood="outrun")
    assert r["ok"] and out.exists() and r["seconds"] >= 2

def test_commands_without_player_return_error():
    r = call("set_tempo", bpm=120)
    assert r["ok"] is False and "not running" in r["error"]
```

- [ ] **Step 2: Run** `uv run pytest tests/test_mcp.py -v` — Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# synthwave/mcp_server.py
"""MCP server (stdio) piloting a live Player. Run with: synthwave mcp"""
from __future__ import annotations
import threading
from mcp.server.fastmcp import FastMCP
from .audio.renderer import RenderConfig, Renderer
from .composer.moods import MOODS
from .patches import loader

mcp = FastMCP("infinite-synthwave")
_lock = threading.Lock()
_player = None
_renderer: Renderer | None = None


def _live():
    if _player is None or not _player.running:
        return None
    return _renderer


def _command(fn) -> dict:
    r = _live()
    if r is None:
        return {"ok": False, "error": "player not running; call start first"}
    done = threading.Event()
    box: dict = {}

    def run():
        try:
            fn(r)
        except Exception as e:
            box["error"] = str(e)
        finally:
            done.set()

    r.submit(run)
    done.wait(2.0)
    if "error" in box:
        return {"ok": False, "error": box["error"]}
    return {"ok": True, "status": r.status()}


@mcp.tool()
def start(mood: str = "dark", bpm: float | None = None, seed: int | None = None,
          duration_s: float | None = None) -> dict:
    """Start infinite synthwave on the audio output. mood: dark|dreamy|outrun."""
    global _player, _renderer
    from .audio.output import Player
    with _lock:
        if _player is not None and _player.running:
            return {"ok": False, "error": "already running; call stop first"}
        try:
            _renderer = Renderer(RenderConfig(mood=mood, bpm=bpm, seed=seed, duration_s=duration_s))
            _player = Player(_renderer)
            _player.start()
        except Exception as e:
            _player = None
            return {"ok": False, "error": str(e)}
    return {"ok": True, "status": _renderer.status()}


@mcp.tool()
def stop() -> dict:
    """Stop playback."""
    global _player
    with _lock:
        if _player is None:
            return {"ok": False, "error": "not running"}
        _player.stop()
        _player = None
    return {"ok": True}


@mcp.tool()
def status() -> dict:
    """Current transport, key, chord, section and layer state."""
    if _live() is None:
        return {"running": False, "moods": list(MOODS)}
    return {"running": True, "underruns": _player.underruns, **_renderer.status()}


@mcp.tool()
def set_tempo(bpm: float) -> dict:
    """Change tempo (60-180 BPM)."""
    return _command(lambda r: r.set_tempo(bpm))


@mcp.tool()
def set_mood(mood: str) -> dict:
    """Change mood for upcoming sections: dark|dreamy|outrun."""
    return _command(lambda r: r.set_mood(mood))


@mcp.tool()
def set_layer(layer: str, mute: bool | None = None, solo: bool | None = None,
              volume: float | None = None) -> dict:
    """Mute/solo/volume (0-2) for a layer: drums|bass|arp|pad|lead|ambient."""
    return _command(lambda r: r.set_layer(layer, mute=mute, solo=solo, volume=volume))


@mcp.tool()
def list_patches() -> dict:
    """List available synth/drum patches."""
    return {"patches": loader.list_patches()}


@mcp.tool()
def load_patch(layer: str, name: str) -> dict:
    """Load a patch (library name or YAML path) into a layer."""
    return _command(lambda r: r.load_patch(layer, name))


@mcp.tool()
def set_patch_param(layer: str, path: str, value: float | str) -> dict:
    """Set one patch parameter, e.g. path='filter.cutoff' value=800, 'oscillators.0.detune' 20."""
    return _command(lambda r: r.set_patch_param(layer, path, value))


@mcp.tool()
def next_section() -> dict:
    """Jump to the next section at the next bar."""
    return _command(lambda r: r.next_section())


@mcp.tool()
def export_wav(path: str, seconds: float, mood: str = "dark", bpm: float | None = None,
               seed: int | None = None) -> dict:
    """Render a standalone track offline to a WAV file (does not disturb live playback)."""
    from .audio.export import export_wav as _export
    try:
        r = Renderer(RenderConfig(mood=mood, bpm=bpm, seed=seed, duration_s=seconds))
        n = _export(r, seconds, path)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "path": path, "seconds": round(n / r.sr, 2), "seed": r.seed}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run** `uv run pytest tests/test_mcp.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat(mcp): FastMCP server controlling live playback"`

### Task 15: README + configuration MCP exemple

**Files:**
- Create: `README.md`, `.mcp.json`

- [ ] **Step 1: README** — installation (`uv sync`), usage CLI (infini, durée, export, moods, seed), format des patches YAML avec exemple, dossier utilisateur `~/.config/synthwave/patches`, outils MCP et configuration Claude Code :

```json
{
  "mcpServers": {
    "synthwave": {"command": "uv", "args": ["run", "synthwave", "mcp"]}
  }
}
```

- [ ] **Step 2: Run** `uv run pytest -q && uv run ruff check .` — Expected: tout vert.
- [ ] **Step 3: Commit** `git add -A && git commit -m "docs: README and MCP config"`
