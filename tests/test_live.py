import io
import os
import threading
import time

import numpy as np

from helpers import voice_like
from voxpipe.engine.chain import Chain
from voxpipe.engine.entry import ChainEntry
from voxpipe.live import LiveLoop, StopReason

BLOCK = 480
ENTRIES = [ChainEntry("gain", {"gain_db": 6.0}), ChainEntry("reverb", {"room_size": 0.4, "mix": 0.3})]


def to_bytes(x: np.ndarray) -> bytes:
    return x.astype("<f4").tobytes()


def test_processes_all_blocks_then_reports_source_lost():
    x = voice_like(0.5)  # 50 blocks
    writer = io.BytesIO()
    loop = LiveLoop(Chain(ENTRIES), io.BytesIO(to_bytes(x)), writer)
    assert loop.run(threading.Event()) is StopReason.SOURCE_LOST
    out = np.frombuffer(writer.getvalue(), dtype="<f4")
    expected = Chain(ENTRIES).process_signal(x)
    assert loop.stats.blocks + loop.stats.discarded_blocks == 50
    # Without discards the output is exactly the offline render.
    if loop.stats.discarded_blocks == 0:
        np.testing.assert_allclose(out, expected, atol=1e-5)


def test_partial_trailing_block_is_dropped():
    data = to_bytes(np.zeros(BLOCK * 2 + 100, dtype=np.float32))
    writer = io.BytesIO()
    LiveLoop(Chain(ENTRIES), io.BytesIO(data), writer).run(threading.Event())
    assert len(writer.getvalue()) == BLOCK * 2 * 4


class BrokenWriter(io.RawIOBase):
    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        raise BrokenPipeError


def test_destination_lost():
    loop = LiveLoop(Chain(ENTRIES), io.BytesIO(to_bytes(voice_like(0.1))), BrokenWriter())
    assert loop.run(threading.Event()) is StopReason.DESTINATION_LOST


def test_interrupted_while_waiting_for_input():
    read_fd, write_fd = os.pipe()
    stop = threading.Event()
    loop = LiveLoop(Chain(ENTRIES), os.fdopen(read_fd, "rb", buffering=0), io.BytesIO())
    threading.Timer(0.2, stop.set).start()
    started = time.monotonic()
    assert loop.run(stop) is StopReason.INTERRUPTED
    assert time.monotonic() - started < 1.0
    os.close(write_fd)


def test_input_timeout_counts_as_source_lost():
    read_fd, write_fd = os.pipe()
    loop = LiveLoop(Chain(ENTRIES), os.fdopen(read_fd, "rb", buffering=0), io.BytesIO(),
                    input_timeout_s=0.3)
    assert loop.run(threading.Event()) is StopReason.SOURCE_LOST
    os.close(write_fd)


class SlowChain:
    def process_block(self, block):
        time.sleep(0.02)  # twice the 10 ms block budget
        return block


def test_slow_processing_counts_late_and_discards_backlog():
    data = to_bytes(np.zeros(BLOCK * 20, dtype=np.float32))
    loop = LiveLoop(SlowChain(), io.BytesIO(data), io.BytesIO())
    loop.run(threading.Event())
    assert loop.stats.late_blocks > 0
    assert loop.stats.discarded_blocks > 0
    assert loop.stats.blocks + loop.stats.discarded_blocks == 20
