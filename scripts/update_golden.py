"""Regenerate tests/golden/mechanicus.npy. Run only when an effect's sound changes on
purpose, and say so in the commit message.

    uv run python scripts/update_golden.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from helpers import voice_like  # noqa: E402
from voxpipe.engine.chain import Chain  # noqa: E402
from voxpipe.profile import load_profile  # noqa: E402

profile = load_profile(ROOT / "tests" / "data" / "mechanicus.json")
out = Chain(profile.chain).process_signal(voice_like())
target = ROOT / "tests" / "golden" / "mechanicus.npy"
target.parent.mkdir(exist_ok=True)
np.save(target, out)
print(f"wrote {target} ({out.size} samples)")
