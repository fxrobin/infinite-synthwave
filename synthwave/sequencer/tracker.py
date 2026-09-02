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
