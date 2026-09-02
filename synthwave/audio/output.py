"""Producer thread renders ahead into a queue; the sounddevice callback only copies blocks."""
from __future__ import annotations

import queue
import threading
import time

from .renderer import Renderer


class Player:
    def __init__(self, renderer: Renderer, blocksize: int = 1024, prefill: int = 6, device=None):
        self.renderer, self.blocksize, self.prefill = renderer, blocksize, prefill
        self.device = device
        self.queue: queue.Queue = queue.Queue(maxsize=prefill)
        self.stop_event = threading.Event()
        self.done_event = threading.Event()
        self.underruns = 0
        self.error: Exception | None = None
        self.thread: threading.Thread | None = None
        self.stream = None

    def _produce(self) -> None:
        try:
            while not self.stop_event.is_set() and not self.renderer.finished:
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
        while (self.queue.qsize() < min(self.prefill, 2) and time.time() < deadline
               and not self.error):
            time.sleep(0.01)
        if self.error:
            raise self.error
        self.stream = sd.OutputStream(samplerate=self.renderer.sr, channels=2, dtype="float32",
                                      blocksize=self.blocksize, device=self.device,
                                      callback=self._callback,
                                      finished_callback=self.done_event.set)
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
