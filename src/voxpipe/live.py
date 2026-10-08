"""The real-time loop: source bytes -> effect chain -> destination bytes."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import BinaryIO

import numpy as np

from voxpipe.audio import BLOCK_SIZE, SAMPLE_RATE

MAX_BACKLOG = 3  # blocks; anything older is discarded so latency never drifts (N-04)
INPUT_TIMEOUT_S = 2.0
_POLL_S = 0.05


class StopReason(Enum):
    INTERRUPTED = "interrupted"
    SOURCE_LOST = "source lost"
    DESTINATION_LOST = "destination lost"


@dataclass
class LoopStats:
    blocks: int = 0
    late_blocks: int = 0
    discarded_blocks: int = 0
    duration_s: float = 0.0


def _read_exact(reader: BinaryIO, size: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < size:
        chunk = reader.read(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


class LiveLoop:
    def __init__(
        self,
        chain,
        reader: BinaryIO,
        writer: BinaryIO,
        block_size: int = BLOCK_SIZE,
        input_timeout_s: float = INPUT_TIMEOUT_S,
    ) -> None:
        self.chain = chain
        self.reader = reader
        self.writer = writer
        self.block_size = block_size
        self.block_bytes = block_size * 4
        self.block_seconds = block_size / SAMPLE_RATE
        self.input_timeout_s = input_timeout_s
        self.stats = LoopStats()
        self._queue: queue.Queue[bytes | None] = queue.Queue()

    def _pump(self) -> None:
        while True:
            data = _read_exact(self.reader, self.block_bytes)
            self._queue.put(data)
            if data is None:
                return

    def run(self, stop: threading.Event) -> StopReason:
        started = time.monotonic()
        try:
            return self._run(stop)
        finally:
            self.stats.duration_s = time.monotonic() - started

    def _run(self, stop: threading.Event) -> StopReason:
        threading.Thread(target=self._pump, daemon=True).start()
        last_input = time.monotonic()
        while not stop.is_set():
            try:
                data = self._queue.get(timeout=_POLL_S)
            except queue.Empty:
                if time.monotonic() - last_input > self.input_timeout_s:
                    return StopReason.SOURCE_LOST
                continue
            last_input = time.monotonic()
            while data is not None and self._queue.qsize() > MAX_BACKLOG:
                self.stats.discarded_blocks += 1
                data = self._queue.get_nowait()
            if data is None:
                return StopReason.SOURCE_LOST

            began = time.perf_counter()
            out = self.chain.process_block(np.frombuffer(data, dtype="<f4"))
            if time.perf_counter() - began > self.block_seconds:
                self.stats.late_blocks += 1
            try:
                self.writer.write(np.asarray(out, dtype="<f4").tobytes())
                self.writer.flush()
            except (BrokenPipeError, OSError, ValueError):
                return StopReason.DESTINATION_LOST
            self.stats.blocks += 1
        return StopReason.INTERRUPTED
