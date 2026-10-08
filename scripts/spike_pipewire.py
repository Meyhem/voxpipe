"""Manual spike: measure the pw-play -> virtual mic -> pw-record hop latency.

    uv run python scripts/spike_pipewire.py
Requires the virtual mic from Task 1 Step 1 to exist.
"""

import subprocess
import threading
import time

import numpy as np

NAME = "voxpipe.virtual_mic"
RATE, BLOCK = 48_000, 480
FMT = ["--rate", str(RATE), "--channels", "1", "--format", "f32", "--latency", "120"]  # 2.5 ms; 480 gave ~32 ms hop

rec = subprocess.Popen(["pw-record", "--target", NAME, *FMT, "-"], stdout=subprocess.PIPE)
# --target <virtual mic> does not link (falls back to the default sink); link by hand.
play = subprocess.Popen(
    ["pw-play", "--target", "0", "--volume", "1.0", "-P", "{ node.name=voxpipe-spike }", *FMT, "-"],
    stdin=subprocess.PIPE,
)
arrivals: list[float] = []


def reader() -> None:
    while True:
        data = rec.stdout.read(BLOCK * 4)
        if not data:
            return
        if np.max(np.abs(np.frombuffer(data, dtype="<f4"))) > 0.5:
            arrivals.append(time.perf_counter())


threading.Thread(target=reader, daemon=True).start()
silence = np.zeros(BLOCK, dtype="<f4").tobytes()
click = np.ones(BLOCK, dtype="<f4").tobytes()

time.sleep(0.7)  # let the streams appear
subprocess.run(["pw-link", "voxpipe-spike:output_MONO", f"{NAME}:input_MONO"], check=True)
time.sleep(0.3)
next_time = time.perf_counter()
sent_at = None
for i in range(300):  # 3 s of writes paced at real time; the click goes out at 1 s
    if i == 100:
        sent_at = time.perf_counter()
    play.stdin.write(click if i == 100 else silence)
    play.stdin.flush()
    next_time += BLOCK / RATE
    time.sleep(max(0.0, next_time - time.perf_counter()))
time.sleep(0.3)
play.stdin.close()
rec.terminate()
play.terminate()
if arrivals:
    print(f"hop latency: {(arrivals[0] - sent_at) * 1000:.1f} ms")
else:
    print("click never arrived: pw-play did not feed the virtual mic (see Step 4)")
