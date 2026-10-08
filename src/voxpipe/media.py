"""File decoding through ffmpeg and WAV output."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from voxpipe.audio import SAMPLE_RATE
from voxpipe.errors import VoxpipeError


class MediaError(VoxpipeError):
    pass


def decode(path: Path) -> np.ndarray:
    """Decode any audio or video file to float32 mono at SAMPLE_RATE."""
    if not path.is_file():
        raise MediaError(f"{path}: file not found")
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError:
        raise MediaError("ffmpeg not found; install ffmpeg") from None
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip().splitlines()
        reason = detail[-1] if detail else f"ffmpeg exit code {result.returncode}"
        raise MediaError(f"{path}: cannot decode audio: {reason}")
    samples = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32)
    if samples.size == 0:
        raise MediaError(f"{path}: cannot decode audio: no audio stream")
    return samples


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, SAMPLE_RATE, pcm)
