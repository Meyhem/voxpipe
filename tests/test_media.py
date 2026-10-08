import shutil
import subprocess

import numpy as np
import pytest

from helpers import dominant_hz, sine
from voxpipe.audio import SAMPLE_RATE
from voxpipe.media import MediaError, decode, write_wav

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


def test_wav_round_trip(tmp_path):
    path = tmp_path / "tone.wav"
    x = sine(440.0, seconds=0.5)
    write_wav(path, x)
    y = decode(path)
    assert y.dtype == np.float32
    assert y.size == x.size
    np.testing.assert_allclose(y, x, atol=1e-3)


def test_write_wav_clips(tmp_path):
    path = tmp_path / "loud.wav"
    write_wav(path, np.array([2.0, -2.0, 0.5], dtype=np.float32))
    np.testing.assert_allclose(decode(path), [1.0, -1.0, 0.5], atol=1e-3)


def test_decodes_stereo_44k_mp3_to_mono_48k(tmp_path):
    path = tmp_path / "tone.mp3"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=300:duration=1:sample_rate=44100", "-ac", "2", str(path))
    y = decode(path)
    assert abs(y.size - SAMPLE_RATE) < 2_000  # mp3 padding
    assert abs(dominant_hz(y) - 300.0) < 5.0


def test_decodes_audio_from_video(tmp_path):
    path = tmp_path / "clip.mkv"
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=500:duration=1",
        "-c:v", "mpeg4", "-c:a", "flac", "-shortest", str(path),
    )
    assert abs(dominant_hz(decode(path)) - 500.0) < 5.0


def test_video_without_audio_is_an_error(tmp_path):
    path = tmp_path / "silent.mkv"
    ffmpeg("-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10", "-c:v", "mpeg4", str(path))
    with pytest.raises(MediaError, match="silent.mkv"):
        decode(path)


def test_missing_file(tmp_path):
    with pytest.raises(MediaError, match="file not found"):
        decode(tmp_path / "nope.wav")


def test_garbage_file(tmp_path):
    path = tmp_path / "garbage.wav"
    path.write_bytes(b"not audio at all")
    with pytest.raises(MediaError, match="cannot decode"):
        decode(path)
