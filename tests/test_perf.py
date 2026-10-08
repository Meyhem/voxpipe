import time
from pathlib import Path

import numpy as np

from helpers import voice_like
from voxpipe.audio import BLOCK_SIZE
from voxpipe.engine.chain import Chain
from voxpipe.profile import load_profile

BUDGET_S = 0.003  # N-02: one 10 ms block in <= 3 ms with the full chain


def test_full_chain_block_fits_realtime_budget():
    chain = Chain(load_profile(Path(__file__).parent / "data" / "mechanicus.json").chain)
    signal = voice_like(5.0)
    blocks = [signal[i : i + BLOCK_SIZE] for i in range(0, signal.size, BLOCK_SIZE)]
    for block in blocks[:50]:  # warm-up
        chain.process_block(block)
    timings = []
    for block in blocks:
        start = time.perf_counter()
        chain.process_block(block)
        timings.append(time.perf_counter() - start)
    assert np.median(timings) < BUDGET_S, f"median {np.median(timings) * 1000:.2f} ms"
