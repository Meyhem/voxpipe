import numpy as np

from helpers import sine, voice_like
from voxpipe.tuning.features import (
    HOP,
    N_MELS,
    frame_count,
    frames,
    log_mel,
    ltas,
    pitch,
    speech_frames,
)


def test_frame_count_and_padding():
    assert frame_count(0) == 1
    assert frame_count(480) == 1
    assert frame_count(481) == 2
    f = frames(np.ones(500, dtype=np.float32), 1024)
    assert f.shape == (2, 1024)
    assert f[0, :500].sum() == 500 and f[0, 500:].sum() == 0
    assert f[1, 0] == 1.0 and f[1, 20] == 0.0  # second frame starts at sample 480


def test_log_mel_shape_and_band_energy():
    low, high = log_mel(sine(200.0)), log_mel(sine(5000.0))
    assert low.shape == (frame_count(48_000), N_MELS)
    assert np.argmax(low.mean(axis=0)) < np.argmax(high.mean(axis=0))


def test_log_mel_tracks_level():
    x = voice_like(0.5)
    diff = log_mel(x) - log_mel(x * 0.5)
    assert abs(np.median(diff) - 6.02) < 0.1


def test_pitch_of_harmonic_tone():
    f0 = pitch(sum(sine(200.0 * k, amplitude=0.3 / k) for k in range(1, 6)))
    voiced = f0[f0 > 0]
    assert voiced.size > 0.8 * f0.size
    assert abs(np.median(voiced) - 200.0) < 4.0


def test_pitch_silence_is_unvoiced():
    assert np.all(pitch(np.zeros(24_000, dtype=np.float32)) == 0.0)


def test_ltas_is_mean_over_frames():
    mel = log_mel(voice_like(0.5))
    np.testing.assert_allclose(ltas(mel), mel.mean(axis=0))


def test_hop_is_10ms():
    assert HOP == 480


def test_low_pitch_is_detected():
    # Biased autocorrelation decays with lag and used to miss low (machine-like) voices.
    tone = sum(sine(65.0 * k, amplitude=0.3 / k) for k in range(1, 8))
    noisy = tone + np.random.default_rng(0).normal(0.0, 0.15, tone.size)
    f0 = pitch(noisy.astype(np.float32))
    voiced = f0[f0 > 0]
    assert voiced.size > 0.8 * f0.size
    assert abs(np.median(voiced) - 65.0) < 3.0


def test_speech_frames_separate_speech_from_background():
    rng = np.random.default_rng(0)
    voice = voice_like(0.5)
    background = rng.normal(0.0, 0.01, voice.size).astype(np.float32)  # about -40 dB
    signal = np.concatenate([background, voice + background, background])
    active = speech_frames(signal)
    third = active.size // 3
    assert not active[5 : third - 5].any()
    assert active[third + 5 : 2 * third - 5].all()
    assert not active[2 * third + 5 :].any()


def test_speech_frames_on_clean_recording():
    signal = np.concatenate([np.zeros(24_000, dtype=np.float32), voice_like(0.5)])
    active = speech_frames(signal)
    assert not active[:45].any() and active[55:].all()
