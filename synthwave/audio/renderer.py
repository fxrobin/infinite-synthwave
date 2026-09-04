"""Mixes all layers into a stereo block; owns the command queue used by CLI/MCP
threads.
"""

from __future__ import annotations

import math
import queue
from dataclasses import dataclass, field

import numpy as np

from ..composer.arranger import LAYERS, TRACK_SECONDS, Arranger, BarPlan
from ..composer.harmony import Harmony
from ..composer.moods import MOODS
from ..engine.d50 import D50Synth
from ..engine.drums import DrumKit
from ..engine.dx7 import Dx7Synth
from ..engine.effects import Effect, Limiter, Sidechain, build_effects
from ..engine.risers import RiserKit
from ..engine.solina import SolinaSynth
from ..engine.synth import Synth
from ..patches.loader import PatchError, apply_tweaks, load_patch, set_param
from ..patches.model import (
    AnyPatch,
    D50PatchModel,
    DrumPatchModel,
    Dx7PatchModel,
    PatchModel,
    SolinaPatchModel,
)
from ..sequencer.tracker import Tracker
from ..sequencer.transport import Transport

DEFAULT_PATCHES = {
    "drums": "drums_808",
    "bass": "bass_moog",
    "arp": "arp_pluck",
    "pad": "pad_juno",
    "lead": "lead_saw",
    "lead2": "lead_hollow",
    "ambient": "ambient_drone",
    "riser": "builtin",
}
# Master "colour": always-on console/tape stage between the master inserts and the limiter.
# Applied after the master volume, so the saturation always sees the same level.
MASTER_COLORS: dict[str, list[dict]] = {
    "clean": [],
    # `lofi` only ever appears fully wet: its wobble delays the whole wet path, so a
    # partial mix would comb-filter the master instead of colouring it.
    "tape": [  # default: gentle tape saturation plus a little quantisation grit
        {"type": "distortion", "drive": 1.7, "tone": 14000, "mix": 0.35},
        {"type": "bitcrush", "bits": 13, "downsample": 1, "mix": 0.25},
    ],
    "vhs": [  # worn tape: darker, noisier, audibly crushed
        {"type": "distortion", "drive": 2.2, "tone": 7000, "mix": 0.5},
        {
            "type": "lofi",
            "bits": 10,
            "downsample": 2,
            "cutoff": 7000,
            "wobble": 0.003,
            "noise": 0.0025,
            "mix": 1.0,
        },
    ],
    "mic": [  # saturated microphone: narrow band, driven hard, room hiss
        {"type": "distortion", "drive": 3.2, "tone": 4500, "mix": 0.6},
        {
            "type": "lofi",
            "bits": 11,
            "downsample": 2,
            "cutoff": 5500,
            "wobble": 0.0015,
            "noise": 0.0035,
            "mix": 1.0,
        },
    ],
    "crush": [  # heaviest: bitcrushed and distorted, for a full lo-fi mix
        {"type": "bitcrush", "bits": 8, "downsample": 2, "mix": 0.6},
        {"type": "distortion", "drive": 3.5, "tone": 5000, "mix": 0.6},
    ],
}

DUCKED = {"pad": 1.0, "bass": 0.6, "ambient": 0.6, "arp": 0.5}
LAYER_TRIM = {"lead": 2.5, "lead2": 1.6}  # static make-up gain per layer, before the mix


@dataclass
class RenderConfig:
    """Renderconfig."""

    sr: int = 44100
    bpm: float | None = None
    mood: str | None = None  # None: random at start, redrawn at every transition
    seed: int | None = None
    duration_s: float | None = None
    patches: dict[str, str] = field(default_factory=dict)
    bpm_range: tuple[float, float] | None = None  # overrides the mood's range when set
    track_s: float = TRACK_SECONDS  # target length of one track (intro -> outro)
    master_color: str = "auto"  # always-on master stage: "auto" = composed, else MASTER_COLORS


class Renderer:
    """Renderer."""

    def __init__(self, cfg: RenderConfig):  # noqa: C901 - renderer wiring has many branches
        """Initialize."""
        if cfg.mood is not None and cfg.mood not in MOODS:
            raise ValueError(f"unknown mood {cfg.mood!r}, choose from {list(MOODS)}")
        self.cfg, self.sr = cfg, cfg.sr
        self.seed = (
            int(cfg.seed) if cfg.seed is not None else int(np.random.SeedSequence().entropy % 2**31)
        )
        self.rng = np.random.default_rng(self.seed)
        names = list(MOODS)
        self.mood = MOODS[cfg.mood or names[int(self.rng.integers(len(names)))]]
        if cfg.bpm_range is not None and cfg.bpm_range[0] > cfg.bpm_range[1]:
            raise ValueError("bpm_range must be (low, high)")
        lo, hi = cfg.bpm_range or self.mood.bpm_range
        self.bpm = float(cfg.bpm) if cfg.bpm else round(float(self.rng.uniform(lo, hi)), 1)
        self.transport = Transport(self.sr, self.bpm)
        total_bars = (
            math.ceil(cfg.duration_s / self.transport.bar_seconds) if cfg.duration_s else None
        )
        self.arranger = Arranger(
            self.rng,
            Harmony(self.rng, self.mood),
            self.mood,
            total_bars,
            bpm_range=cfg.bpm_range,
            bpm=self.bpm,
            track_s=cfg.track_s,
        )
        self.arranger.mood_locked = cfg.mood is not None
        self.tracker = Tracker(self.transport, self.arranger)
        self.instruments: dict[str, Synth | Dx7Synth | SolinaSynth | D50Synth | DrumKit] = {}
        self.patch_names: dict[str, str] = {}
        # loaded + manual edits
        self.base_patch: dict[str, AnyPatch] = {}
        self.auto_tweaks: dict[str, dict[str, float]] = {}  # arranger gestures
        self.auto_tweaks_enabled = True
        self.manual_patch: set[str] = set(cfg.patches)
        for layer in LAYERS:
            if layer == "riser":
                self.instruments[layer] = RiserKit(self.sr, self.rng, self.bpm)
                self.patch_names[layer] = "builtin"
                continue
            name = cfg.patches.get(layer, self.mood.patches.get(layer, DEFAULT_PATCHES[layer]))
            self._install(layer, name, load_patch(name))
        self.layer_volume = {layer: 1.0 for layer in LAYERS}
        self.muted: set[str] = set()
        self.solo: set[str] = set()
        self.plan_gain = {layer: 0.0 for layer in LAYERS}
        self.current_gain = {layer: 0.0 for layer in LAYERS}
        self.sidechain = Sidechain(self.sr, depth=0.45, release=0.22)
        self.limiter = Limiter(self.sr, self.bpm, threshold=0.95)
        if cfg.master_color != "auto" and cfg.master_color not in MASTER_COLORS:
            raise ValueError(f"unknown master colour: {cfg.master_color}")
        self.master_color_locked = cfg.master_color != "auto"
        self.master_color = cfg.master_color if self.master_color_locked else "tape"
        self.master_color_fx: list[Effect] = build_effects(
            MASTER_COLORS[self.master_color], self.sr, self.bpm
        )
        self.master_volume = 0.7
        self.fade_target, self.fade, self.fade_rate = 1.0, 1.0, 0.0
        self.commands: queue.SimpleQueue = queue.SimpleQueue()
        self.plan: BarPlan | None = None
        self.finished = False
        self.rendered = 0
        self.auto_fx: dict[str, list[dict]] = {}
        self.manual_fx: dict[str, list[dict]] = {}
        self.inserts: dict[str, list[Effect]] = {}
        self.levels: dict[str, float] = {layer: 0.0 for layer in LAYERS + ("master",)}
        self.scope = np.zeros(256, dtype=np.float32)  # last block, downsampled, for UIs

    # ----- instruments -----
    def _install(self, layer: str, name: str, patch, live: bool = False) -> None:
        """
        Load `patch` into a layer.

        `live`: parameter-only change, keep voices running.
        """
        if layer == "riser":
            raise PatchError("layer 'riser' is built-in and has no patch")
        self.base_patch[layer] = patch
        if self.auto_tweaks_enabled and layer in self.auto_tweaks:
            patch = apply_tweaks(patch, self.auto_tweaks[layer])
        inst = self.instruments.get(layer)
        if layer == "drums":
            if not isinstance(patch, DrumPatchModel):
                raise PatchError(f"layer 'drums' needs a drum patch, got {patch.name!r}")
            if inst is None:
                inst = DrumKit(patch, self.sr, self.rng, self.bpm)
            else:
                inst.set_patch(patch)
        else:
            if isinstance(patch, Dx7PatchModel):
                if inst is None or not isinstance(inst, Dx7Synth):
                    inst = Dx7Synth(patch, self.sr, self.rng, self.bpm)
                elif live:
                    inst.update_patch(patch)
                else:
                    inst.set_patch(patch)
            elif isinstance(patch, SolinaPatchModel):
                if inst is None or not isinstance(inst, SolinaSynth):
                    inst = SolinaSynth(patch, self.sr, self.rng, self.bpm)
                elif live:
                    inst.update_patch(patch)
                else:
                    inst.set_patch(patch)
            elif isinstance(patch, D50PatchModel):
                if inst is None or not isinstance(inst, D50Synth):
                    inst = D50Synth(patch, self.sr, self.rng, self.bpm)
                elif live:
                    inst.update_patch(patch)
                else:
                    inst.set_patch(patch)
            elif not isinstance(patch, PatchModel):
                raise PatchError(f"layer {layer!r} needs a synth patch, got {patch.name!r}")
            elif inst is None or not isinstance(inst, Synth):
                inst = Synth(patch, self.sr, self.rng, self.bpm)
            elif live:
                inst.update_patch(patch)
            else:
                inst.set_patch(patch)
        self.instruments[layer] = inst
        self.patch_names[layer] = name

    # ----- commands (thread-safe through submit) -----
    def submit(self, fn) -> None:
        """Submit."""
        self.commands.put(fn)

    def _drain(self) -> None:
        """Drain."""
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
        """Set tempo."""
        self.bpm = float(np.clip(bpm, 60, 180))
        self.transport.set_bpm(self.bpm)
        self.arranger.bpm = self.bpm
        for inst in self.instruments.values():
            inst.set_bpm(self.bpm)
        self._rebuild_inserts()

    def set_mood(self, name: str) -> None:
        """Request a mood change through an ambient transition; 'random' unlocks the
        draw.
        """
        if name == "random":
            self.arranger.set_mood(None)
            return
        if name not in MOODS:
            raise ValueError(f"unknown mood {name!r}, choose from {list(MOODS) + ['random']}")
        self.arranger.set_mood(MOODS[name])

    def set_layer(
        self,
        layer: str,
        mute: bool | None = None,
        solo: bool | None = None,
        volume: float | None = None,
    ) -> None:
        """Set layer."""
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}, choose from {LAYERS}")
        if mute is not None:
            (self.muted.add if mute else self.muted.discard)(layer)
        if solo is not None:
            (self.solo.add if solo else self.solo.discard)(layer)
        if volume is not None:
            self.layer_volume[layer] = float(np.clip(volume, 0.0, 2.0))

    def load_patch(self, layer: str, name: str) -> None:
        """Load patch."""
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}, choose from {LAYERS}")
        self._install(layer, name, load_patch(name))
        self.manual_patch.add(layer)

    def _apply_mood_patches(self) -> None:
        """Apply mood patches."""
        for layer in LAYERS:
            if layer in self.manual_patch or layer == "riser":
                continue
            name = self.mood.patches.get(layer, DEFAULT_PATCHES[layer])
            if name != self.patch_names[layer]:
                self._install(layer, name, load_patch(name))

    def set_patch_param(self, layer: str, path: str, value) -> None:
        """Set patch param."""
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}, choose from {LAYERS}")
        base = self.base_patch[layer]
        self._install(layer, self.patch_names[layer], set_param(base, path, value), live=True)
        self.manual_patch.add(layer)  # a hand-tweaked patch is kept across sections

    def set_auto_tweaks(self, enabled: bool) -> None:
        """Enable / disable the arranger's live patch gestures (filter sweeps,
        detune...).
        """
        self.auto_tweaks_enabled = bool(enabled)
        for layer in LAYERS:
            if layer in self.base_patch:
                self._install(layer, self.patch_names[layer], self.base_patch[layer], live=True)

    def _apply_auto_tweaks(self, tweaks: dict[str, dict[str, float]]) -> None:
        """Apply auto tweaks."""
        changed = {
            layer
            for layer in set(tweaks) | set(self.auto_tweaks)
            if tweaks.get(layer) != self.auto_tweaks.get(layer)
        }
        self.auto_tweaks = {k: dict(v) for k, v in tweaks.items()}
        if not self.auto_tweaks_enabled:
            return
        for layer in changed:
            if layer in self.base_patch:
                self._install(layer, self.patch_names[layer], self.base_patch[layer], live=True)

    def next_section(self) -> None:
        """Next section."""
        self.arranger.force_next_section()

    def set_layer_effects(self, layer: str, specs: list[dict] | None) -> None:
        """Manual insert chain for a layer or 'master'; None restores the arranger's
        choice.
        """
        if layer not in LAYERS and layer != "master":
            raise ValueError(f"unknown layer {layer!r}, choose from {LAYERS + ('master',)}")
        if specs is None:
            self.manual_fx.pop(layer, None)
        else:
            build_effects(specs, self.sr, self.bpm)  # validate before committing
            self.manual_fx[layer] = list(specs)
        self._rebuild_inserts()

    def _rebuild_inserts(self) -> None:
        """Rebuild inserts."""
        self.inserts = {}
        for layer in LAYERS + ("master",):
            specs = self.manual_fx.get(layer, self.auto_fx.get(layer, []))
            if specs:
                self.inserts[layer] = build_effects(specs, self.sr, self.bpm)
        self.master_color_fx = build_effects(MASTER_COLORS[self.master_color], self.sr, self.bpm)

    def set_master_color(self, name: str) -> None:
        """Pick the always-on master stage (see `MASTER_COLORS`); 'auto' hands it back to
        the arranger, which colours each section.
        """
        if name != "auto" and name not in MASTER_COLORS:
            raise ValueError(f"unknown master colour: {name}")
        self.master_color_locked = name != "auto"
        if self.master_color_locked:
            self.master_color = name
            self._rebuild_inserts()

    def status(self) -> dict:  # noqa: C901 - status aggregates many fields
        """Status."""
        p = self.plan
        return {
            "bpm": self.bpm,
            "mood": self.arranger.mood.name,
            "seed": self.seed,
            "mood_locked": self.arranger.mood_locked,
            "master_color": self.master_color,
            "master_color_locked": self.master_color_locked,
            "bpm_range": list(self.arranger.bpm_range or self.arranger.mood.bpm_range),
            "pending_mood": (
                self.arranger.pending_mood.name if self.arranger.pending_mood else None
            ),
            "bar": p.bar if p else 0,
            "section": p.section.value if p else "intro",
            "section_bar": p.section_bar if p else 0,
            "chord": p.chord.name if p else "",
            "drop": p.drop if p else False,
            "track": p.track if p else self.arranger.track,
            "track_bar": p.track_bar if p else 0,
            "track_bars": self.arranger.track_bars,
            "key": p.key if p else self.arranger.harmony.key_name,
            "elapsed_s": round(self.rendered / self.sr, 1),
            "finished": self.finished,
            "effects": {
                layer: self.manual_fx.get(layer, self.auto_fx.get(layer, []))
                for layer in LAYERS + ("master",)
            },
            "manual_fx": sorted(self.manual_fx),
            "auto_tweaks": self.auto_tweaks_enabled,
            "tweaks": {
                k: {p: round(f, 3) for p, f in v.items()} for k, v in self.auto_tweaks.items()
            },
            "levels": {k: round(v, 3) for k, v in self.levels.items()},
            "layers": {
                layer: {
                    "gain": round(self._effective_gain(layer), 3),
                    "muted": layer in self.muted,
                    "solo": layer in self.solo,
                    "volume": self.layer_volume[layer],
                    "patch": self.patch_names[layer],
                    "manual_patch": layer in self.manual_patch,
                }
                for layer in LAYERS
            },
        }

    # ----- rendering -----
    def _effective_gain(self, layer: str) -> float:
        """Effective gain."""
        if layer in self.muted or (self.solo and layer not in self.solo):
            return 0.0
        return self.plan_gain[layer] * self.layer_volume[layer]

    def _apply_plan_mood(self, plan) -> None:
        """Apply plan mood."""
        if plan.mood:
            self.mood = MOODS[plan.mood]
            self._apply_mood_patches()
        for layer, name in (plan.patches or {}).items():
            if layer not in self.manual_patch and name != self.patch_names[layer]:
                self._install(layer, name, load_patch(name))

    def _apply_plan_tempo(self, plan) -> None:
        """Apply plan tempo."""
        if plan.bpm and abs(plan.bpm - self.bpm) > 0.05:
            self.set_tempo(plan.bpm)
        self._apply_auto_tweaks(plan.tweaks or {})
        new_fx = plan.fx or {}
        if new_fx != self.auto_fx:
            self.auto_fx = dict(new_fx)
            self._rebuild_inserts()

    def _apply_plan(self, plan) -> None:
        """Apply plan."""
        self.plan = plan
        self.plan_gain = dict(plan.gains)
        self.fade_target = plan.fade
        bar = max(1, self.transport.step_samples(16))
        self.fade_rate = (plan.fade - self.fade) / bar
        if plan.finished:
            self.finished = True
        self._apply_plan_mood(plan)
        self._apply_plan_tempo(plan)
        if plan.master_color and not self.master_color_locked:
            if plan.master_color != self.master_color:
                self.master_color = plan.master_color
                self._rebuild_inserts()

    def _mix_layers(self, n: int, events: dict, duck) -> np.ndarray:
        """Mix layers."""
        mix = np.zeros((n, 2), dtype=np.float32)
        for layer in LAYERS:
            target = self._effective_gain(layer)
            g0 = self.current_gain[layer]
            # The gain ramp is applied to the dry signal, before the patch effects and the
            # inserts: a layer leaving the mix keeps its delay/reverb tail ringing out.
            ramp = np.linspace(g0, target, n, endpoint=False, dtype=np.float32)
            ramp *= LAYER_TRIM.get(layer, 1.0)
            # DX7 6-op : un peu moins fort, surtout les leads très présents
            if isinstance(self.instruments[layer], Dx7Synth):
                ramp = ramp * (0.72 if layer in ("lead", "lead2") else 0.85)
            sig = self.instruments[layer].render(n, events[layer], gain=ramp)
            for fx in self.inserts.get(layer, ()):
                sig = fx.process(sig)
            if layer in DUCKED:
                sig = sig * (1.0 - DUCKED[layer] * (1.0 - duck))[:, None]
            self.levels[layer] = float(np.abs(sig).max()) if n else 0.0
            mix += sig
            self.current_gain[layer] = target
        return mix

    def _apply_master(self, mix: np.ndarray, n: int) -> np.ndarray:
        """Apply master."""
        fade = self.fade + self.fade_rate * np.arange(1, n + 1, dtype=np.float32)
        fade = (
            np.maximum(fade, self.fade_target)
            if self.fade_rate < 0
            else np.minimum(fade, self.fade_target)
        )
        self.fade = float(fade[-1])
        for fx in self.inserts.get("master", ()):
            mix = fx.process(mix)
        mix *= (self.master_volume * fade)[:, None]
        for fx in self.master_color_fx:
            mix = fx.process(mix)
        return mix

    def render(self, n: int) -> np.ndarray:
        """Render."""
        self._drain()
        events, plan = self.tracker.advance(n)
        if plan is not None:
            self._apply_plan(plan)
        kicks = [e.offset for e in events["drums"] if e.on and e.note == 36]
        duck = self.sidechain.gain(n, kicks)
        mix = self._mix_layers(n, events, duck)
        mix = self._apply_master(mix, n)
        self.rendered += n
        out = self.limiter.process(mix)
        self.levels["master"] = float(np.abs(out).max()) if n else 0.0
        if n >= 256:
            self.scope = out[:: max(1, n // 256), 0][:256].astype(np.float32)
        return out
