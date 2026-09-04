"""Les 100 ondes PCM du D-50, synthétisées procéduralement.

Les vraies ondes sont dans des ROM Roland copyrightées : on reproduit ici leur *rôle* dans
l'architecture LA (transitoire one-shot 1–47, boucle statique 48–76, boucles composites
77–100) avec les longueurs de la table reconstruite (mots à 32 kHz) et une synthèse par
famille (maillets, cordes pincées, percussions bruitées, souffles, archets, cuivres, spectres
additifs, bruit). Règle de lecture du D-50 : un partial PCM avance de ``f × 2048`` mots par
seconde ; une boucle de L mots sonne donc à ``f × 2048 / L``.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from scipy.signal import lfilter

from .filter import biquad_coeffs

PCM_SR = 32000
WORDS_PER_HZ = 2048  # mots lus par seconde pour f = 1 Hz

# (nom, famille, longueur en mots, paramètres) — index 1..100
PCM_TABLE: dict[int, tuple[str, str, int, dict]] = {
    1: ("Marmba", "mallet", 4096, {"f0": 31.25 * 8, "ratios": (1, 4.0, 9.9), "decay": 0.06}),
    2: ("Vibes", "mallet", 4096, {"f0": 500, "ratios": (1, 4.0, 10.0), "decay": 0.12}),
    3: ("Xylo1", "mallet", 4096, {"f0": 500, "ratios": (1, 3.0, 6.1), "decay": 0.05, "click": 0.5}),
    4: ("Xylo2", "mallet", 4096, {"f0": 250, "ratios": (1, 3.0, 5.9), "decay": 0.07, "click": 0.6}),
    5: (
        "Log_Bs",
        "mallet",
        8192,
        {"f0": 62.5, "ratios": (1, 2.8, 5.4), "decay": 0.2, "click": 0.3},
    ),
    6: ("Hammer", "hit", 2048, {"band": 900, "q": 0.3, "decay": 0.02}),
    7: (
        "JpnDrm",
        "mallet",
        2048,
        {"f0": 500 / 4, "ratios": (1, 1.5, 2.2), "decay": 0.04, "click": 0.8},
    ),
    8: (
        "Kalmba",
        "mallet",
        2048,
        {"f0": 125 * 2, "ratios": (1, 2.9, 5.3), "decay": 0.05, "click": 0.4},
    ),
    9: ("Pluck", "pluck", 2048, {"f0": 250, "bright": 0.8, "decay": 0.05}),
    10: ("Chink", "hit", 2048, {"band": 4000, "q": 0.5, "decay": 0.012}),
    11: (
        "Agogo",
        "mallet",
        2048,
        {"f0": 125 * 6, "ratios": (1, 1.53, 2.4), "decay": 0.05, "click": 0.3},
    ),
    12: (
        "3angle",
        "mallet",
        4096,
        {"f0": 125 * 20, "ratios": (1, 1.62, 2.7, 3.9), "decay": 0.12, "click": 0.2},
    ),
    13: ("Bells", "mallet", 4096, {"f0": 1000, "ratios": (1, 2.0, 2.76, 5.4), "decay": 0.12}),
    14: ("Nails", "hit", 2048, {"band": 3000, "q": 0.2, "decay": 0.03, "grit": True}),
    15: ("Pick", "hit", 2048, {"band": 2500, "q": 0.4, "decay": 0.008}),
    16: ("Lpiano", "pluck", 8192, {"f0": 62.5, "bright": 0.5, "decay": 0.25, "hammer": 0.5}),
    17: ("Mpiano", "pluck", 8192, {"f0": 250, "bright": 0.6, "decay": 0.22, "hammer": 0.4}),
    18: ("Hpiano", "pluck", 8192, {"f0": 500, "bright": 0.7, "decay": 0.18, "hammer": 0.3}),
    19: ("Harpsi", "pluck", 8192, {"f0": 125, "bright": 1.0, "decay": 0.2, "hammer": 0.2}),
    20: ("Harp", "pluck", 4096, {"f0": 1000, "bright": 0.4, "decay": 0.1}),
    21: ("OrgPrc", "mallet", 4096, {"f0": 250, "ratios": (1, 3.0), "decay": 0.08}),
    22: ("Steel", "pluck", 4096, {"f0": 125, "bright": 0.9, "decay": 0.12, "hammer": 0.3}),
    23: ("Nylon", "pluck", 4096, {"f0": 250, "bright": 0.5, "decay": 0.1, "hammer": 0.2}),
    24: ("Eguit1", "pluck", 4096, {"f0": 125, "bright": 0.7, "decay": 0.12}),
    25: ("Eguit2", "pluck", 4096, {"f0": 250, "bright": 0.6, "decay": 0.12}),
    26: ("Dirt", "pluck", 4096, {"f0": 250, "bright": 1.0, "decay": 0.12, "drive": 4.0}),
    27: ("P_Bass", "pluck", 4096, {"f0": 62.5, "bright": 0.7, "decay": 0.12, "hammer": 0.3}),
    28: ("Pop", "pluck", 4096, {"f0": 62.5, "bright": 1.0, "decay": 0.05, "hammer": 0.8}),
    29: ("Thump", "hit", 4096, {"band": 150, "q": 0.3, "decay": 0.05}),
    30: ("Uprite", "pluck", 4096, {"f0": 125, "bright": 0.3, "decay": 0.12, "hammer": 0.4}),
    31: ("Clarnt", "breath", 4096, {"f0": 500, "odd": True, "noise": 0.3, "rise": 0.03}),
    32: ("Breath", "breath", 4096, {"f0": 1000, "noise": 1.0, "rise": 0.02, "band": 2500}),
    33: ("Steam", "breath", 4096, {"f0": 500, "noise": 0.9, "rise": 0.01, "band": 4000}),
    34: ("FluteH", "breath", 4096, {"f0": 1000, "noise": 0.5, "rise": 0.03}),
    35: ("FluteL", "breath", 4096, {"f0": 500, "noise": 0.5, "rise": 0.04}),
    36: ("Guiro", "hit", 2048, {"band": 2000, "q": 0.6, "decay": 0.04, "grit": True}),
    37: ("IndFlt", "breath", 2048, {"f0": 1000, "noise": 0.6, "rise": 0.015}),
    38: ("Harmo", "breath", 4096, {"f0": 500, "noise": 0.2, "rise": 0.05, "harmonic": 3}),
    39: ("Lips1", "brass", 2048, {"f0": 125, "buzz": 0.8}),
    40: ("Lips2", "brass", 2048, {"f0": 250, "buzz": 0.6}),
    41: ("Trumpt", "brass", 4096, {"f0": 1000, "buzz": 0.4}),
    42: ("Bones", "brass", 4096, {"f0": 1000 / 4, "buzz": 0.5}),
    43: ("Contra", "bow", 4096, {"f0": 31.25 * 2, "scratch": 0.5}),
    44: ("Cello", "bow", 4096, {"f0": 125, "scratch": 0.4}),
    45: ("VioBow", "bow", 4096, {"f0": 500, "scratch": 0.7}),
    46: ("Violns", "bow", 4096, {"f0": 1000, "scratch": 0.3, "ensemble": True}),
    47: ("Pizz", "pluck", 4096, {"f0": 62.5 * 4, "bright": 0.4, "decay": 0.06, "hammer": 0.3}),
    # boucles statiques : (longueur du cycle, amplitudes des harmoniques)
    48: ("Drawbr", "loop", 512, {"amps": (1, 0.8, 0.6, 0.5, 0, 0.4, 0, 0.3)}),
    49: ("Horgan", "loop", 128, {"amps": (1, 0.3, 0.5, 0.2, 0.1, 0.15)}),
    50: ("Lorgan", "loop", 256, {"amps": (1, 0.9, 0.2, 0.6, 0.1, 0.3)}),
    51: ("EP_lp1", "loop", 256, {"amps": (1, 0.15, 0.08, 0.05)}),
    52: ("EP_lp2", "loop", 256, {"amps": (1, 0.3, 0.05, 0.12, 0.03)}),
    53: (
        "CLAVlp",
        "loop",
        512,
        {"amps": tuple(1.0 / k if k % 2 else 0.5 / k for k in range(1, 25))},
    ),
    54: ("HC_lp", "loop", 512, {"amps": tuple(1.0 / k**0.7 for k in range(1, 33))}),
    55: ("EB_lp1", "loop", 512, {"amps": (1, 0.6, 0.3, 0.2, 0.1)}),
    56: ("AB_lp", "loop", 256, {"amps": (1, 0.5, 0.25, 0.1)}),
    57: ("EB_lp2", "loop", 512, {"amps": (1, 0.7, 0.5, 0.3, 0.2, 0.1)}),
    58: ("EB_lp3", "loop", 512, {"amps": (1, 0.4, 0.6, 0.2, 0.3)}),
    59: ("EG_lp", "loop", 256, {"amps": (1, 0.8, 0.5, 0.4, 0.3, 0.2, 0.15)}),
    60: ("CELLlp", "loop", 512, {"amps": tuple(1.0 / k for k in range(1, 20))}),
    61: ("VIOLlp", "loop", 128, {"amps": tuple(1.0 / k**0.8 for k in range(1, 12))}),
    62: (
        "Reedlp",
        "loop",
        512,
        {"amps": tuple((1.0 if k % 2 else 0.3) / k**0.6 for k in range(1, 20))},
    ),
    63: ("SAXlp1", "loop", 256, {"amps": (1, 0.9, 0.7, 0.8, 0.5, 0.4, 0.3, 0.2)}),
    64: ("SAXlp2", "loop", 256, {"amps": (1, 0.6, 0.9, 0.5, 0.6, 0.3, 0.2)}),
    65: ("Aah_lp", "loop", 128, {"formants": ((800, 1.0), (1200, 0.6), (2800, 0.25))}),
    66: ("Ooh_lp", "loop", 1024, {"formants": ((350, 1.0), (700, 0.4), (2500, 0.1))}),
    67: ("Manlp1", "loop", 256, {"formants": ((600, 1.0), (1000, 0.5), (2400, 0.2))}),
    68: (
        "Spect1",
        "loop",
        2048,
        {"amps": tuple(1.0 if k in (1, 2, 3, 5, 8, 13) else 0.0 for k in range(1, 20))},
    ),
    69: (
        "Spect2",
        "loop",
        2048,
        {"amps": tuple(1.0 / k**0.3 if k % 3 == 1 else 0.0 for k in range(1, 40))},
    ),
    70: (
        "Spect3",
        "loop",
        2048,
        {"amps": tuple(1.0 if k in (1, 7, 11, 17, 23) else 0.0 for k in range(1, 30))},
    ),
    71: ("Spect4", "loop", 2048, {"amps": tuple(0.9**k for k in range(1, 60))}),
    72: (
        "Spect5",
        "loop",
        2048,
        {"amps": tuple(1.0 if k % 4 == 0 or k == 1 else 0.0 for k in range(1, 48))},
    ),
    73: (
        "Spect6",
        "loop",
        2048,
        {"amps": tuple(1.0 / k if k % 2 == 0 or k == 1 else 0.0 for k in range(1, 40))},
    ),
    74: (
        "Spect7",
        "loop",
        2048,
        {"amps": tuple(1.0 if k in (1, 3, 9, 27) else 0.0 for k in range(1, 30))},
    ),
    75: ("Manlp2", "loop", 256, {"formants": ((500, 1.0), (1500, 0.7), (2500, 0.3))}),
    76: ("Noise", "noise", 16384, {}),
}

# Boucles composites 77–100 : mélange de deux boucles précédentes (longueur, a, b, poids)
_COMBOS = {
    77: (4096, 48, 51, 0.5),
    78: (4096, 49, 65, 0.5),
    79: (4096, 50, 68, 0.6),
    80: (4096, 53, 59, 0.5),
    81: (4096, 55, 60, 0.5),
    82: (4096, 66, 69, 0.5),
    83: (16384, 61, 65, 0.6),
    84: (16384, 68, 70, 0.5),
    85: (16384, 62, 63, 0.5),
    86: (16384, 54, 71, 0.5),
    87: (8192, 49, 72, 0.5),
    88: (8192, 65, 74, 0.5),
    89: (4096, 56, 57, 0.5),
    90: (16384, 51, 73, 0.6),
    91: (8192, 64, 67, 0.5),
    92: (16384, 61, 69, 0.5),
    93: (8192, 48, 76, 0.85),
    94: (8192, 60, 76, 0.85),
    95: (32768, 68, 76, 0.7),
    96: (32768, 70, 72, 0.5),
    97: (32768, 52, 66, 0.5),
    98: (65536, 71, 76, 0.8),
    99: (65536, 73, 74, 0.5),
    100: (65536, 75, 63, 0.5),
}
for _n, (_len, _a, _b, _w) in _COMBOS.items():
    PCM_TABLE[_n] = (f"Loop{_n - 76:02d}", "combo", _len, {"a": _a, "b": _b, "w": _w})

PCM_NAMES = {n: v[0] for n, v in PCM_TABLE.items()}


def pcm_is_loop(n: int) -> bool:
    return n >= 48


def _bandpass(x: np.ndarray, fc: float, q: float) -> np.ndarray:
    b, a = biquad_coeffs("bp", fc, q, PCM_SR)
    return lfilter(b, a, x)


def _lowpass(x: np.ndarray, fc: float) -> np.ndarray:
    b, a = biquad_coeffs("lp", fc, 0.1, PCM_SR)
    return lfilter(b, a, x)


def _edges(x: np.ndarray, ms: float = 2.0) -> np.ndarray:
    k = max(1, int(PCM_SR * ms / 1000))
    x = x.copy()
    x[:k] *= np.linspace(0, 1, k)
    x[-k:] *= np.linspace(1, 0, k)
    return x


def _norm(x: np.ndarray) -> np.ndarray:
    peak = float(np.abs(x).max()) or 1.0
    return (x / peak * 0.9).astype(np.float32)


def _mallet(n: int, f0: float, ratios, decay: float, click: float = 0.2, rng=None) -> np.ndarray:
    t = np.arange(n) / PCM_SR
    out = np.zeros(n)
    for i, r in enumerate(ratios):
        f = min(f0 * r, PCM_SR * 0.45)
        out += np.sin(2 * np.pi * f * t) * np.exp(-t / (decay / (1 + 0.6 * i))) / (1 + i)
    noise = rng.standard_normal(n) * np.exp(-t / 0.003) * click
    return out + _bandpass(noise, min(f0 * 4, 8000), 0.3)


def _pluck(n, f0, bright, decay, hammer=0.0, drive=0.0, rng=None) -> np.ndarray:
    t = np.arange(n) / PCM_SR
    out = np.zeros(n)
    kmax = int(min(40, PCM_SR * 0.45 / f0))
    for k in range(1, kmax + 1):
        amp = bright ** ((k - 1) / 4) / k**0.6
        out += (
            amp
            * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
            * np.exp(-t * k**0.5 / decay)
        )
    if hammer:
        out += (
            _bandpass(rng.standard_normal(n) * np.exp(-t / 0.004), min(f0 * 6, 6000), 0.2) * hammer
        )
    if drive:
        out = np.tanh(out * drive)
    return out


def _hit(n, band, q, decay, grit=False, rng=None) -> np.ndarray:
    t = np.arange(n) / PCM_SR
    noise = rng.standard_normal(n)
    if grit:
        noise *= rng.random(n) < 0.3
    return _bandpass(noise * np.exp(-t / decay), band, q)


def _breath(n, f0, noise, rise, odd=False, band=None, harmonic=1, rng=None) -> np.ndarray:
    t = np.arange(n) / PCM_SR
    env = 1.0 - np.exp(-t / rise)
    tone = np.zeros(n)
    for k in range(1, 8):
        if odd and k % 2 == 0:
            continue
        tone += np.sin(2 * np.pi * f0 * harmonic * k * t) / k**1.3
    air = _bandpass(rng.standard_normal(n), band or min(f0 * 3, 6000), 0.15)
    return tone * env * (1 - noise) + air * noise * (0.3 + 0.7 * env) * np.exp(-t / 0.15)


def _brass(n, f0, buzz, rng=None) -> np.ndarray:
    t = np.arange(n) / PCM_SR
    out = np.zeros(n)
    kmax = int(min(24, PCM_SR * 0.45 / f0))
    for k in range(1, kmax + 1):
        bright = 1.0 - np.exp(-t / (0.02 + 0.004 * k))  # les harmoniques arrivent après
        out += np.sin(2 * np.pi * f0 * k * t) / k**0.8 * bright
    lips = _bandpass(rng.standard_normal(n), min(f0 * 2, 3000), 0.4) * np.exp(-t / 0.015) * buzz
    return out + lips


def _bow(n, f0, scratch, ensemble=False, rng=None) -> np.ndarray:
    t = np.arange(n) / PCM_SR
    out = np.zeros(n)
    kmax = int(min(20, PCM_SR * 0.45 / f0))
    dets = (0.0, 0.004, -0.005) if ensemble else (0.0,)
    for d in dets:
        for k in range(1, kmax + 1):
            out += np.sin(2 * np.pi * f0 * (1 + d) * k * t) / k
    env = 1.0 - np.exp(-t / 0.05)
    noise = _bandpass(rng.standard_normal(n), min(f0 * 5, 5000), 0.2) * np.exp(-t / 0.04) * scratch
    return out * env / len(dets) + noise


def _loop_cycle(length: int, amps=None, formants=None, rng=None) -> np.ndarray:
    n = np.arange(length)
    out = np.zeros(length)
    if amps is not None:
        for k, a in enumerate(amps, start=1):
            if a and k < length // 2:
                out += a * np.sin(2 * np.pi * k * n / length + rng.uniform(0, 2 * np.pi))
    if formants is not None:
        # cycle riche filtré par formants : la boucle sonne à 32000 / length Hz en lecture native
        base = 32000.0 / length
        rich = np.zeros(length)
        for k in range(1, min(60, length // 2)):
            rich += np.sin(2 * np.pi * k * n / length) / k**0.5
        x = np.tile(rich, 8)  # état de filtre stabilisé sur plusieurs cycles
        for fc, g in formants:
            out += _bandpass(x, max(fc, base), 0.5)[-length:] * g
    return out


def _combo(length: int, a: int, b: int, w: float) -> np.ndarray:
    wa, wb = pcm_wave(a)[0], pcm_wave(b)[0]
    ta = np.resize(wa, length)
    tb = np.resize(wb, length)
    return ta * w + tb * (1.0 - w)


@cache
def pcm_wave(n: int) -> tuple[np.ndarray, bool]:
    """Onde PCM ``n`` (1–100) à 32 kHz, (échantillons float32 ±0.9, bouclée ?)."""
    name, kind, length, prm = PCM_TABLE[n]
    rng = np.random.default_rng(1000 + n)
    if kind == "mallet":
        x = _mallet(length, rng=rng, **prm)
    elif kind == "pluck":
        x = _pluck(length, rng=rng, **prm)
    elif kind == "hit":
        x = _hit(length, rng=rng, **prm)
    elif kind == "breath":
        x = _breath(length, rng=rng, **prm)
    elif kind == "brass":
        x = _brass(length, rng=rng, **prm)
    elif kind == "bow":
        x = _bow(length, rng=rng, **prm)
    elif kind == "loop":
        x = _loop_cycle(length, rng=rng, **prm)
    elif kind == "noise":
        x = rng.standard_normal(length)
    else:
        x = _combo(length, **prm)
    looped = pcm_is_loop(n)
    if not looped:
        x = _edges(x * np.linspace(1.0, 0.0, length) ** 0.3)  # les one-shots meurent en fin de page
    return _norm(x), looped
