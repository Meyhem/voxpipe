"""Internal audio format (D-09): float32, mono, 48 kHz."""

SAMPLE_RATE = 48_000
BLOCK_SIZE = 480  # 10 ms, the live block size
OFFLINE_CHUNK = 4_800  # 100 ms, used by render and tune; output is block-size invariant
