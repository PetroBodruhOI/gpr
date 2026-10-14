"""Tests for ml_pipeline/feature_extract.py."""

import numpy as np
import pytest

from ml_pipeline.feature_extract import (
    BEAT_KEYS,
    FEATURE_KEYS,
    LIBROSA_KEYS,
    _default_beat_feats,
    extract_beat_features,
    extract_features,
    extract_librosa_features,
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
