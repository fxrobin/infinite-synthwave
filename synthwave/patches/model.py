from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Wave = Literal["saw", "square", "triangle", "sine", "noise", "fm"]


class OscSpec(BaseModel):
    wave: Wave
    unison: int = Field(1, ge=1, le=8)
    detune: float = Field(0.0, ge=0.0, le=100.0)
    octave: int = Field(0, ge=-3, le=3)
    semi: int = Field(0, ge=-12, le=12)
    level: float = Field(1.0, ge=0.0, le=2.0)
    pwm: float = Field(0.5, ge=0.05, le=0.95)
    spread: float = Field(1.0, ge=0.0, le=1.0)
    fm_ratio: float = Field(2.0, gt=0.0, le=16.0)
    fm_index: float = Field(0.0, ge=0.0, le=10.0)


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
    type: Literal[
        "chorus",
        "delay",
        "reverb",
        "gated_reverb",
        "limiter",
        "gate",
        "bitcrush",
        "lofi",
        "distortion",
        "autopan",
        "phaser",
        "flanger",
    ]


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
    sub: float = 0.6  # sub-sine layer level at pitch_end
    sub_decay: float = 0.45
    drive: float = 2.0  # tanh saturation (1.0 = clean, acoustic-like)
    beater: float = 0.0  # felt beater thump (lowpassed noise burst), 0..1
    gain: float = 1.4


class SnareSpec(BaseModel):
    tone: float = 180.0
    tone_decay: float = 0.08
    noise_decay: float = 0.18
    gate_hold: float = 0.25
    reverb_size: float = 0.85
    reverb_mix: float = 0.45
    gain: float = 0.6


class HatSpec(BaseModel):
    closed_decay: float = 0.05
    open_decay: float = 0.35
    cutoff: float = 8000.0
    gain: float = 0.35


class ClapSpec(BaseModel):
    decay: float = 0.25
    gate_hold: float = 0.2
    reverb_mix: float = 0.5
    gain: float = 0.6


class TomSpec(BaseModel):
    pitch_low: float = 110.0
    pitch_mid: float = 160.0
    decay: float = 0.3
    gain: float = 0.8


class SnapSpec(BaseModel):
    tone: float = 1900.0  # bandpass centre of the snap
    decay: float = 0.07
    body: float = 0.35  # low "knuckle" thump level
    reverb_mix: float = 0.25
    gain: float = 0.7


class RideSpec(BaseModel):
    decay: float = 0.9
    cutoff: float = 5000.0
    ping: float = 0.5  # metallic partials level
    gain: float = 0.3


class TickSpec(BaseModel):
    """Dry, very short closed hat ("tic tic") on the 8ths."""

    decay: float = 0.012
    cutoff: float = 9000.0
    gain: float = 0.3


class ShakerSpec(BaseModel):
    decay: float = 0.06
    cutoff: float = 6500.0
    gain: float = 0.3


class DrumPatchModel(BaseModel):
    name: str
    kind: Literal["drums"]
    volume: float = Field(0.9, ge=0.0, le=2.0)
    kick: KickSpec = KickSpec()
    snare: SnareSpec = SnareSpec()
    hat: HatSpec = HatSpec()
    clap: ClapSpec = ClapSpec()
    tom: TomSpec = TomSpec()
    snap: SnapSpec = SnapSpec()
    ride: RideSpec = RideSpec()
    shaker: ShakerSpec = ShakerSpec()
    tick: TickSpec = TickSpec()
    crash_gain: float = 0.4
    crash_roll_gain: float = 0.5  # one-bar mallet roll on the crash, swelling to the downbeat
    perc_effects: list[EffectSpec] = []  # applied to everything except the kick


AnyPatch = PatchModel | DrumPatchModel
