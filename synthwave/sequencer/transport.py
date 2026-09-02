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
            ticks.append(StepTick(self.step_index // STEPS_PER_BAR,
                                  self.step_index % STEPS_PER_BAR,
                                  int(self.next_step_time - self.clock)))
            self.step_index += 1
            self.next_step_time += self.samples_per_step
        self.clock = end
        return ticks
