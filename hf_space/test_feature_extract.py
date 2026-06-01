"""Tests for ml_pipeline/feature_extract.py and ml_pipeline/preprocess.py helpers."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from ml_pipeline.feature_extract import (
    BEAT_KEYS,
    FEATURE_KEYS,
    LIBROSA_KEYS,
    _default_beat_feats,
    extract_beat_features,
    extract_features,
    extract_librosa_features,
    to_vec,
)

SR = 22050


# ── _default_beat_feats ───────────────────────────────────────────────────────

def test_default_beat_feats_keys():
    feats = _default_beat_feats()
    for k in BEAT_KEYS:
        assert k in feats


def test_default_beat_feats_grid_fit_neutral():
    feats = _default_beat_feats()
    assert feats["grid_fit_error"] == 0.5


def test_default_beat_feats_zeros():
    feats = _default_beat_feats()
    for k in feats:
        if k != "grid_fit_error":
            assert feats[k] == 0.0


# ── extract_librosa_features ──────────────────────────────────────────────────

def test_librosa_features_keys():
    y = np.random.randn(SR).astype(np.float32)
    feats = extract_librosa_features(y, SR)
    for k in LIBROSA_KEYS:
        assert k in feats, f"Missing key: {k}"


def test_librosa_features_mfcc_range():
    y = 0.3 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32)
    feats = extract_librosa_features(y, SR)
    for i in range(1, 9):
        assert np.isfinite(feats[f"mfcc_{i}"])


def test_librosa_features_rms_positive():
    y = 0.5 * np.ones(SR, dtype=np.float32)
    feats = extract_librosa_features(y, SR)
    assert feats["rms_mean"] > 0
    assert feats["crest_factor"] >= 0


def test_librosa_features_spectral_range():
    y = 0.3 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32)
    feats = extract_librosa_features(y, SR)
    assert 0 < feats["centroid_mean"] < SR / 2
    assert feats["bandwidth_mean"] >= 0


def test_librosa_features_silence():
    y = np.zeros(SR, dtype=np.float32)
    feats = extract_librosa_features(y, SR)
    assert feats["rms_mean"] == pytest.approx(0.0, abs=1e-6)


def test_librosa_features_single_onset():
    """Short audio with 0-1 onsets → ioi fallback path."""
    y = np.zeros(SR * 2, dtype=np.float32)
    y[SR // 2] = 0.9
    feats = extract_librosa_features(y, SR)
    assert "ioi_mean" in feats
    assert np.isfinite(feats["ioi_mean"])


# ── to_vec ────────────────────────────────────────────────────────────────────

def test_to_vec_length():
    feats = {k: float(i) for i, k in enumerate(FEATURE_KEYS)}
    vec = to_vec(feats)
    assert len(vec) == len(FEATURE_KEYS)


def test_to_vec_dtype():
    feats = {k: 1.0 for k in FEATURE_KEYS}
    vec = to_vec(feats)
    assert vec.dtype == np.float32


def test_to_vec_order():
    feats = {k: float(i) for i, k in enumerate(FEATURE_KEYS)}
    vec = to_vec(feats)
    for i, k in enumerate(FEATURE_KEYS):
        assert vec[i] == pytest.approx(float(i))


# ── extract_beat_features ─────────────────────────────────────────────────────

def _make_rhythmic_audio(sr=SR, bpm=120.0, duration=4.0):
    t = np.arange(int(sr * duration)) / sr
    y = np.zeros(len(t), dtype=np.float32)
    beat_period = 60.0 / bpm
    for bt in np.arange(0, duration, beat_period):
        idx = int(bt * sr)
        if idx < len(y):
            y[idx : idx + 100] = 0.8
    return y


def test_beat_features_with_valid_beats():
    sr = SR
    y = _make_rhythmic_audio(sr, bpm=120.0, duration=4.0)
    # Provide synthetic beats at 120 BPM (0.5s period)
    beats = np.arange(0.0, 4.0, 0.5, dtype=np.float32)
    downbeats = np.arange(0.0, 4.0, 2.0, dtype=np.float32)
    feats = extract_beat_features(y, sr, beats, downbeats)
    for k in BEAT_KEYS:
        assert k in feats
    assert feats["bpm"] > 0


def test_beat_features_returns_defaults_no_beats():
    sr = SR
    y = _make_rhythmic_audio(sr)
    feats = extract_beat_features(y, sr,
                                   beats_in_chunk=np.array([]),
                                   downbeats_in_chunk=np.array([]))
    # Not enough info → returns defaults
    assert feats["bpm"] == 0.0


def test_beat_features_zero_duration():
    feats = extract_beat_features(np.array([], dtype=np.float32), SR,
                                   np.array([]), np.array([]))
    assert feats["grid_fit_error"] == 0.5


def test_beat_features_bar_sanity_fallback():
    """bar_length outside 1-5s → fallback to beat-derived bar_length."""
    sr = SR
    y = _make_rhythmic_audio(sr, bpm=30.0, duration=8.0)  # very slow → bar > 5s
    # 2-second beat period → bar_length = 8s (> 5s limit)
    beats = np.arange(0.0, 8.0, 2.0, dtype=np.float32)
    downbeats = np.array([0.0, 8.0], dtype=np.float32)  # only 2 downbeats
    feats = extract_beat_features(y, sr, beats, downbeats)
    assert isinstance(feats["grid_fit_error"], float)


def test_beat_features_only_beats_no_downbeats():
    """Falls back to beats when no downbeats available."""
    sr = SR
    y = _make_rhythmic_audio(sr, bpm=120.0, duration=4.0)
    beats = np.arange(0.0, 4.0, 0.5, dtype=np.float32)
    feats = extract_beat_features(y, sr,
                                   beats_in_chunk=beats,
                                   downbeats_in_chunk=np.array([]))
    assert isinstance(feats, dict)


# ── extract_features (combined) ───────────────────────────────────────────────

def test_extract_features_without_beats():
    sr = SR
    y = np.random.randn(sr).astype(np.float32)
    feats = extract_features(y, sr)
    assert len(feats) == len(FEATURE_KEYS)
    # Default beat feats applied
    assert feats["bpm"] == 0.0
    assert feats["grid_fit_error"] == 0.5


def test_extract_features_with_beats():
    sr = SR
    y = _make_rhythmic_audio(sr, bpm=120.0, duration=4.0)
    beats = np.arange(0.0, 4.0, 0.5, dtype=np.float32)
    downbeats = np.array([0.0, 2.0], dtype=np.float32)
    feats = extract_features(y, sr, beats=beats, downbeats=downbeats)
    assert len(feats) == len(FEATURE_KEYS)


def test_extract_features_chunk_offset():
    """chunk_offset shifts which beats belong to the chunk."""
    sr = SR
    y = _make_rhythmic_audio(sr, bpm=120.0, duration=3.0)
    # Absolute beats at 6-9s (offset 6 → local 0-3)
    beats = np.arange(6.0, 9.0, 0.5, dtype=np.float32)
    downbeats = np.array([6.0, 8.0], dtype=np.float32)
    feats = extract_features(y, sr, beats=beats, downbeats=downbeats,
                              chunk_offset=6.0)
    assert len(feats) == len(FEATURE_KEYS)


def test_extract_features_all_keys_finite():
    sr = SR
    y = 0.1 * np.random.randn(sr * 3).astype(np.float32)
    beats = np.arange(0.0, 3.0, 0.5, dtype=np.float32)
    downbeats = np.array([0.0, 2.0], dtype=np.float32)
    feats = extract_features(y, sr, beats=beats, downbeats=downbeats)
    for k, v in feats.items():
        assert np.isfinite(v), f"Feature {k} is not finite: {v}"


# ── preprocess helpers ────────────────────────────────────────────────────────

def test_file_beats_no_beats_returns_none():
    from ml_pipeline.preprocess import _file_beats
    beats, downbeats = _file_beats("/fake/path.wav", np.zeros(SR), SR,
                                   use_beats=False)
    assert beats is None
    assert downbeats is None


def test_file_beats_use_beats_calls_compute(tmp_path):
    """_file_beats with use_beats=True delegates to compute_beats."""
    from ml_pipeline.preprocess import _file_beats

    sr = SR
    y = np.zeros(sr, dtype=np.float32)
    wav_path = str(tmp_path / "song.wav")
    sf.write(wav_path, y, sr)

    expected_beats = np.array([0.5, 1.0], dtype=np.float32)
    expected_db    = np.array([0.5],       dtype=np.float32)

    with patch("ml_pipeline.preprocess.compute_beats",
               return_value=(expected_beats, expected_db)):
        beats, downbeats = _file_beats(wav_path, y, sr, use_beats=True)

    np.testing.assert_array_equal(beats, expected_beats)
    np.testing.assert_array_equal(downbeats, expected_db)


def test_check_ytdlp_cli_found():
    from ml_pipeline.preprocess import _check_ytdlp
    with patch("shutil.which", return_value="/usr/local/bin/yt-dlp"):
        result = _check_ytdlp()
    assert result == "cli"


def test_check_ytdlp_python_package():
    from ml_pipeline.preprocess import _check_ytdlp
    import sys
    mock_yt_dlp = MagicMock()
    with patch("shutil.which", return_value=None), \
         patch.dict("sys.modules", {"yt_dlp": mock_yt_dlp}):
        result = _check_ytdlp()
    assert result == "python"


def test_check_ytdlp_not_found_raises():
    from ml_pipeline.preprocess import _check_ytdlp
    with patch("shutil.which", return_value=None), \
         patch.dict("sys.modules", {"yt_dlp": None}):
        with pytest.raises((RuntimeError, ImportError)):
            _check_ytdlp()


def test_compute_beats_cache_hit(tmp_path):
    """compute_beats returns cached arrays when cache exists."""
    from ml_pipeline.preprocess import compute_beats
    import hashlib

    beats_data = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    downbeats_data = np.array([0.5], dtype=np.float32)

    cache_key = "test_cache_key_abc"
    key_hash = hashlib.md5(cache_key.encode()).hexdigest()
    cache_file = tmp_path / f"{key_hash}.npz"
    np.savez(str(cache_file), beats=beats_data, downbeats=downbeats_data)

    with patch("ml_pipeline.preprocess.BEATS_CACHE_DIR", str(tmp_path)):
        beats_out, db_out = compute_beats(np.zeros(SR), SR,
                                          cache_key=cache_key)

    np.testing.assert_array_equal(beats_out, beats_data)
    np.testing.assert_array_equal(db_out, downbeats_data)


def test_load_audio_no_demucs_resamples(tmp_path):
    """load_audio without demucs resamples to target sr."""
    from ml_pipeline.preprocess import load_audio

    sr_in = 16000
    duration = 1.0
    y_in = 0.3 * np.sin(2 * np.pi * 440 * np.arange(int(sr_in * duration)) / sr_in)
    wav_path = str(tmp_path / "test.wav")
    sf.write(wav_path, y_in.astype(np.float32), sr_in)

    y_out, sr_out = load_audio(wav_path, use_demucs=False, sr=22050)
    assert sr_out == 22050
    assert abs(len(y_out) - 22050) < 200


def test_load_audio_with_demucs_delegates(tmp_path):
    """load_audio with use_demucs=True calls separate_guitar."""
    from ml_pipeline.preprocess import load_audio

    sr = 22050
    y = np.zeros(sr, dtype=np.float32)
    wav_path = str(tmp_path / "audio.wav")
    sf.write(wav_path, y, sr)

    with patch("ml_pipeline.preprocess.separate_guitar", return_value=(y, sr)) as mock_sep:
        y_out, sr_out = load_audio(wav_path, use_demucs=True, sr=sr)

    mock_sep.assert_called_once_with(wav_path, target_sr=sr)
    assert sr_out == sr


def test_separate_guitar_cache_hit(tmp_path):
    """separate_guitar reads from disk cache when available."""
    import hashlib
    from ml_pipeline.preprocess import separate_guitar

    sr = 22050
    y_cached = np.zeros(sr, dtype=np.float32)

    wav_path = str(tmp_path / "song.wav")
    sf.write(wav_path, y_cached, sr)

    key = hashlib.md5(
        f"{os.path.abspath(wav_path)}|{os.path.getmtime(wav_path)}".encode()
    ).hexdigest()
    cache_path = str(tmp_path / f"{key}_guitar.wav")
    sf.write(cache_path, y_cached, sr)

    with patch("ml_pipeline.preprocess.DEMUCS_CACHE_DIR", str(tmp_path)):
        y_out, sr_out = separate_guitar(wav_path, target_sr=sr)

    assert sr_out == sr
    assert len(y_out) == len(y_cached)


def test_download_ytdlp_python_mode(tmp_path):
    """_download_audio_ytdlp: python mode calls YoutubeDL."""
    import sys
    from ml_pipeline.preprocess import _download_audio_ytdlp

    mock_info = {"title": "Test Song", "uploader": "Artist", "duration": 180}
    mock_ydl = MagicMock()
    mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl.__exit__ = MagicMock(return_value=False)
    mock_ydl.extract_info.return_value = mock_info

    mock_yt_dlp_module = MagicMock()
    mock_yt_dlp_module.YoutubeDL.return_value = mock_ydl

    tmp_wav = str(tmp_path / "audio.wav")

    with patch("ml_pipeline.preprocess._check_ytdlp", return_value="python"), \
         patch.dict("sys.modules", {"yt_dlp": mock_yt_dlp_module}):
        title, uploader, dur = _download_audio_ytdlp(
            "https://example.com/song", tmp_wav
        )

    assert title == "Test Song"
    assert uploader == "Artist"
    assert dur == 180


def test_download_ytdlp_python_mode_with_trimming(tmp_path):
    """_download_audio_ytdlp: python mode passes start_sec/duration_sec."""
    import sys
    from ml_pipeline.preprocess import _download_audio_ytdlp

    mock_info = {"title": "T", "uploader": "U", "duration": 30}
    mock_ydl = MagicMock()
    mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl.__exit__ = MagicMock(return_value=False)
    mock_ydl.extract_info.return_value = mock_info

    mock_yt_dlp_module = MagicMock()
    mock_yt_dlp_module.YoutubeDL.return_value = mock_ydl

    tmp_wav = str(tmp_path / "audio.wav")

    with patch("ml_pipeline.preprocess._check_ytdlp", return_value="python"), \
         patch.dict("sys.modules", {"yt_dlp": mock_yt_dlp_module}):
        _download_audio_ytdlp("https://example.com/song", tmp_wav,
                               start_sec=10.0, duration_sec=20.0)

    call_kwargs = mock_yt_dlp_module.YoutubeDL.call_args[0][0]
    ffmpeg_args = call_kwargs.get("postprocessor_args", {}).get("ffmpeg", [])
    assert "-ar" in ffmpeg_args


def test_download_ytdlp_cli_mode(tmp_path):
    """_download_audio_ytdlp: CLI mode calls subprocess."""
    from ml_pipeline.preprocess import _download_audio_ytdlp

    meta_json = '{"title": "CLI Song", "uploader": "Band", "duration": 240}'
    mock_meta = MagicMock(returncode=0, stdout=meta_json)
    mock_dl = MagicMock(returncode=0, stderr="")

    tmp_wav = str(tmp_path / "audio.wav")

    with patch("ml_pipeline.preprocess._check_ytdlp", return_value="cli"), \
         patch("subprocess.run", side_effect=[mock_meta, mock_dl]):
        title, uploader, dur = _download_audio_ytdlp(
            "https://example.com/song", tmp_wav
        )

    assert title == "CLI Song"
    assert uploader == "Band"


def test_download_ytdlp_cli_failure(tmp_path):
    """_download_audio_ytdlp: CLI failure raises RuntimeError."""
    from ml_pipeline.preprocess import _download_audio_ytdlp

    meta_json = "{}"
    mock_meta = MagicMock(returncode=0, stdout=meta_json)
    mock_dl = MagicMock(returncode=1, stderr="network error")

    tmp_wav = str(tmp_path / "audio.wav")

    with patch("ml_pipeline.preprocess._check_ytdlp", return_value="cli"), \
         patch("subprocess.run", side_effect=[mock_meta, mock_dl]):
        with pytest.raises(RuntimeError):
            _download_audio_ytdlp("https://example.com/song", tmp_wav)
