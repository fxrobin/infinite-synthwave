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
from ..engine.effects import Effect, Limiter, Sidechain, build_effects
from ..engine.synth import Synth
from ..patches.loader import PatchError, load_patch, set_param
from ..patches.model import DrumPatchModel, PatchModel
from ..sequencer.tracker import Tracker
from ..sequencer.transport import Transport

DEFAULT_PATCHES = {"drums": "drums_808", "bass": "bass_moog", "arp": "arp_pluck",
                   "pad": "pad_juno", "lead": "lead_saw", "ambient": "ambient_drone"}
DUCKED = {"pad": 1.0, "bass": 0.6, "ambient": 0.6, "arp": 0.5}


@dataclass
class RenderConfig:
    sr: int = 44100
    bpm: float | None = None
    mood: str = "dark"
    seed: int | None = None
    duration_s: float | None = None
    patches: dict[str, str] = field(default_factory=dict)
    bpm_range: tuple[float, float] | None = None   # overrides the mood's range when set


class Renderer:
    def __init__(self, cfg: RenderConfig):
        if cfg.mood not in MOODS:
            raise ValueError(f"unknown mood {cfg.mood!r}, choose from {list(MOODS)}")
        self.cfg, self.sr = cfg, cfg.sr
        self.seed = (int(cfg.seed) if cfg.seed is not None
                     else int(np.random.SeedSequence().entropy % 2**31))
        self.rng = np.random.default_rng(self.seed)
        self.mood = MOODS[cfg.mood]
        if cfg.bpm_range is not None and cfg.bpm_range[0] > cfg.bpm_range[1]:
            raise ValueError("bpm_range must be (low, high)")
        lo, hi = cfg.bpm_range or self.mood.bpm_range
        self.bpm = float(cfg.bpm) if cfg.bpm else round(float(self.rng.uniform(lo, hi)), 1)
        self.transport = Transport(self.sr, self.bpm)
        total_bars = (math.ceil(cfg.duration_s / self.transport.bar_seconds)
                      if cfg.duration_s else None)
        self.arranger = Arranger(self.rng, Harmony(self.rng, self.mood), self.mood, total_bars,
                                 bpm_range=cfg.bpm_range)
        self.tracker = Tracker(self.transport, self.arranger)
        self.instruments: dict[str, Synth | DrumKit] = {}
        self.patch_names: dict[str, str] = {}
        self.manual_patch: set[str] = set(cfg.patches)
        for layer in LAYERS:
            name = cfg.patches.get(layer, self.mood.patches.get(layer, DEFAULT_PATCHES[layer]))
            self._install(layer, name, load_patch(name))
        self.layer_volume = {layer: 1.0 for layer in LAYERS}
        self.muted: set[str] = set()
        self.solo: set[str] = set()
        self.plan_gain = {layer: 0.0 for layer in LAYERS}
        self.current_gain = {layer: 0.0 for layer in LAYERS}
        self.sidechain = Sidechain(self.sr, depth=0.45, release=0.22)
        self.limiter = Limiter(self.sr, self.bpm, threshold=0.95)
        self.master_volume = 0.7
        self.fade_target, self.fade, self.fade_rate = 1.0, 1.0, 0.0
        self.commands: queue.SimpleQueue = queue.SimpleQueue()
        self.plan: BarPlan | None = None
        self.finished = False
        self.rendered = 0
        self.auto_fx: dict[str, list[dict]] = {}
        self.manual_fx: dict[str, list[dict]] = {}
        self.inserts: dict[str, list[Effect]] = {}

    # ----- instruments -----
    def _install(self, layer: str, name: str, patch) -> None:
        inst = self.instruments.get(layer)
        if layer == "drums":
            if not isinstance(patch, DrumPatchModel):
                raise PatchError(f"layer 'drums' needs a drum patch, got {patch.name!r}")
            if inst is None:
                inst = DrumKit(patch, self.sr, self.rng, self.bpm)
            else:
                inst.set_patch(patch)
        else:
            if not isinstance(patch, PatchModel):
                raise PatchError(f"layer {layer!r} needs a synth patch, got {patch.name!r}")
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
        self._rebuild_inserts()

    def set_mood(self, name: str) -> None:
        """Request a mood change; the arranger applies it through an ambient transition."""
        if name not in MOODS:
            raise ValueError(f"unknown mood {name!r}, choose from {list(MOODS)}")
        self.arranger.set_mood(MOODS[name])

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
            raise ValueError(f"unknown layer {layer!r}, choose from {LAYERS}")
        self._install(layer, name, load_patch(name))
        self.manual_patch.add(layer)

    def _apply_mood_patches(self) -> None:
        for layer in LAYERS:
            if layer in self.manual_patch:
                continue
            name = self.mood.patches.get(layer, DEFAULT_PATCHES[layer])
            if name != self.patch_names[layer]:
                self._install(layer, name, load_patch(name))

    def set_patch_param(self, layer: str, path: str, value) -> None:
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}, choose from {LAYERS}")
        inst = self.instruments[layer]
        self._install(layer, self.patch_names[layer], set_param(inst.patch, path, value))

    def next_section(self) -> None:
        self.arranger.force_next_section()

    def set_layer_effects(self, layer: str, specs: list[dict] | None) -> None:
        """Manual insert chain for a layer or 'master'; None restores the arranger's choice."""
        if layer not in LAYERS and layer != "master":
            raise ValueError(f"unknown layer {layer!r}, choose from {LAYERS + ('master',)}")
        if specs is None:
            self.manual_fx.pop(layer, None)
        else:
            build_effects(specs, self.sr, self.bpm)  # validate before committing
            self.manual_fx[layer] = list(specs)
        self._rebuild_inserts()

    def _rebuild_inserts(self) -> None:
        self.inserts = {}
        for layer in LAYERS + ("master",):
            specs = self.manual_fx.get(layer, self.auto_fx.get(layer, []))
            if specs:
                self.inserts[layer] = build_effects(specs, self.sr, self.bpm)

    def status(self) -> dict:
        p = self.plan
        return {
            "bpm": self.bpm, "mood": self.arranger.mood.name, "seed": self.seed,
            "bpm_range": list(self.arranger.bpm_range or self.arranger.mood.bpm_range),
            "pending_mood": (self.arranger.pending_mood.name
                             if self.arranger.pending_mood else None),
            "bar": p.bar if p else 0, "section": p.section.value if p else "intro",
            "chord": p.chord.name if p else "",
            "key": p.key if p else self.arranger.harmony.key_name,
            "elapsed_s": round(self.rendered / self.sr, 1), "finished": self.finished,
            "effects": {layer: self.manual_fx.get(layer, self.auto_fx.get(layer, []))
                        for layer in LAYERS + ("master",)},
            "layers": {layer: {"gain": round(self._effective_gain(layer), 3),
                               "muted": layer in self.muted, "solo": layer in self.solo,
                               "volume": self.layer_volume[layer],
                               "patch": self.patch_names[layer]} for layer in LAYERS},
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
            bar = max(1, self.transport.step_samples(16))
            self.fade_rate = (plan.fade - self.fade) / bar
            if plan.finished:
                self.finished = True
            if plan.mood:
                self.mood = MOODS[plan.mood]
                self._apply_mood_patches()
            for layer, name in (plan.patches or {}).items():
                if layer not in self.manual_patch and name != self.patch_names[layer]:
                    self._install(layer, name, load_patch(name))
            if plan.bpm and abs(plan.bpm - self.bpm) > 0.05:
                self.set_tempo(plan.bpm)
            new_fx = plan.fx or {}
            if new_fx != self.auto_fx:
                self.auto_fx = dict(new_fx)
                self._rebuild_inserts()
        kicks = [e.offset for e in events["drums"] if e.on and e.note == 36]
        duck = self.sidechain.gain(n, kicks)
        mix = np.zeros((n, 2), dtype=np.float32)
        for layer in LAYERS:
            target = self._effective_gain(layer)
            g0 = self.current_gain[layer]
            sig = self.instruments[layer].render(n, events[layer])
            for fx in self.inserts.get(layer, ()):
                sig = fx.process(sig)
            if layer in DUCKED:
                sig = sig * (1.0 - DUCKED[layer] * (1.0 - duck))[:, None]
            ramp = np.linspace(g0, target, n, endpoint=False, dtype=np.float32)
            mix += sig * ramp[:, None]
            self.current_gain[layer] = target
        fade = self.fade + self.fade_rate * np.arange(1, n + 1, dtype=np.float32)
        fade = np.maximum(fade, self.fade_target) if self.fade_rate < 0 else np.minimum(
            fade, self.fade_target)
        self.fade = float(fade[-1])
        for fx in self.inserts.get("master", ()):
            mix = fx.process(mix)
        mix *= (self.master_volume * fade)[:, None]
        self.rendered += n
        return self.limiter.process(mix)
