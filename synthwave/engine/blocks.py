"""Small per-block helpers shared by the oscillators, LFOs and effects.

Every engine renders in fixed-size blocks, so the sample-index ramps and the ring
window arithmetic are recomputed thousands of times per second. Caching the ramps and
expressing ring windows as contiguous slices removes a large share of the per-block
numpy call overhead without changing any arithmetic.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter as _scipy_lfilter

try:  # chemin rapide : la routine C sous scipy.signal.lfilter
    from scipy.signal import _sigtools

    _linear_filter = _sigtools._linear_filter
except (ImportError, AttributeError):  # pragma: no cover - dépend de la version de scipy
    _linear_filter = None


def lfilter(b, a, x, axis=-1, zi=None):
    """``scipy.signal.lfilter`` sans son enveloppe de validation.

    L'enveloppe Python coûte ~10 µs par appel, et les moteurs en paient une centaine
    par bloc audio. On appelle la routine C directement quand les entrées sont déjà
    les ndarray float64 qu'elle attend, sinon on repasse par scipy.
    """
    if (
        _linear_filter is not None
        and zi is not None
        and type(x) is np.ndarray
        and x.dtype == np.float64
        and type(zi) is np.ndarray
        and zi.dtype == np.float64
        and type(b) is np.ndarray
        and b.dtype == np.float64
        and type(a) is np.ndarray
        and a.dtype == np.float64
    ):
        return _linear_filter(b, a, x, axis, zi)
    return _scipy_lfilter(b, a, x, axis=axis, zi=zi)


_MAX_CACHED = 64  # a handful of block sizes at most (audio blocks, effect chunks)
_ARANGE: dict[int, np.ndarray] = {}
_ARANGE1: dict[int, np.ndarray] = {}


def _cached(cache: dict[int, np.ndarray], n: int, start: int) -> np.ndarray:
    a = cache.get(n)
    if a is None:
        a = np.arange(start, start + n)
        a.flags.writeable = False  # shared: never modify in place
        if len(cache) < _MAX_CACHED:
            cache[n] = a
    return a


def arange(n: int) -> np.ndarray:
    """Read-only ``np.arange(n)``, cached per block size."""
    return _cached(_ARANGE, n, 0)


def arange1(n: int) -> np.ndarray:
    """Read-only ``np.arange(1, n + 1)``, cached per block size."""
    return _cached(_ARANGE1, n, 1)


def segments(pos: int, k: int, n: int) -> tuple[tuple[int, int, int, int], ...]:
    """Split the ring window ``(pos .. pos + k) mod n`` into contiguous slices.

    Returns ``(buf_start, buf_stop, off_start, off_stop)`` tuples, at most two: callers
    always use ``k <= n``. Slicing a ring buffer this way replaces an index array plus a
    fancy-index gather/scatter, which dominated the reverb and delay cost.
    """
    pos %= n
    first = n - pos
    if k <= first:
        return ((pos, pos + k, 0, k),)
    return ((pos, n, 0, first), (0, k - first, first, k))
