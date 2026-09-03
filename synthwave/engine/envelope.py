"""ADSR envelope rendered per block, segment by segment (no per-sample Python loop)."""

from __future__ import annotations

import numpy as np

IDLE, ATTACK, DECAY, SUSTAIN, RELEASE = range(5)


class ADSR:
    """Adsr."""
    def __init__(self, attack: float, decay: float, sustain: float, release: float, sr: int):
        """Initialize."""
        self.a = max(1, int(attack * sr))
        self.d = max(1, int(decay * sr))
        self.s = float(np.clip(sustain, 0.0, 1.0))
        self.r = max(1, int(release * sr))
        self.stage, self.t, self.start, self.level = IDLE, 0, 0.0, 0.0

    @property
    def finished(self) -> bool:
        """Finished."""
        return self.stage == IDLE

    def gate_on(self) -> None:
        """Gate on."""
        self.stage, self.t, self.start = ATTACK, 0, self.level

    def gate_off(self) -> None:
        """Gate off."""
        if self.stage != IDLE:
            self.stage, self.t, self.start = RELEASE, 0, self.level

    def render(self, n: int) -> np.ndarray:
        """Render."""
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
            if self.t >= length:
                # Un update_patch a raccourci l'étape sous nos pieds : on la clôt.
                self._advance()
                continue
            k = min(k, length - self.t)
            ts = self.t + np.arange(1, k + 1)
            if self.stage == ATTACK:
                seg = self.start + (1.0 - self.start) * ts / length
            elif self.stage == DECAY:
                seg = self.s + (1.0 - self.s) * np.exp(-5.0 * ts / length)
            else:
                seg = self.start * np.exp(-6.0 * ts / length)
            out[filled : filled + k] = seg
            self.level = float(seg[-1])
            self.t += k
            filled += k
            if self.t >= length:
                self._advance()
        return out

    def _advance(self) -> None:
        """Advance."""
        self.t = 0
        if self.stage == ATTACK:
            self.stage, self.level = DECAY, 1.0
        elif self.stage == DECAY:
            self.stage, self.level = SUSTAIN, self.s
        else:
            self.stage, self.level = IDLE, 0.0
