from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class NoteEvent:
    """Noteevent."""

    offset: int  # sample offset inside the block
    note: int  # MIDI note
    velocity: float
    on: bool  # False = note-off
