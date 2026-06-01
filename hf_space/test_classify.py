"""Tests for ml_pipeline/classify.py — pure functions + mocked cmd_*."""

import csv
import os
import pickle
import tempfile
from collections import Counter
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_pipeline.classify import (
    FEATURE_KEYS,
    QUIET_THRESHOLD,
    SUPPORTED,
    _brier_multiclass,
    _ece_top_label,
    _normalized,
    classify_vec,
    cmd_extract,
    cmd_predict,
    cmd_train,
    cosine_distance,
    filter_files_by_noise,
    get_files,
    split_audio_chunks,
)

SR = 22050


# ── split_audio_chunks ────────────────────────────────────────────────────────

def test_split_chunks_no_overlap():
    y = np.zeros(SR * 15, dtype=np.float32)
    chunks = split_audio_chunks(y, SR, max_sec=5.0)
    assert len(chunks) == 3
    for _, t_s, t_e in chunks:
        assert t_e - t_s <= 5.1


def test_split_chunks_with_overlap():
    y = np.zeros(SR * 10, dtype=np.float32)
    no_overlap = split_audio_chunks(y, SR, max_sec=5.0)
    with_overlap = split_audio_chunks(y, SR, max_sec=5.0, hop_sec=2.5)
    assert len(with_overlap) > len(no_overlap)


def test_split_chunks_too_short_returns_empty():
    y = np.zeros(SR // 3, dtype=np.float32)  # < 1 second
    assert split_audio_chunks(y, SR, max_sec=5.0) == []


def test_split_chunks_time_boundaries():
    y = np.zeros(SR * 12, dtype=np.float32)
    chunks = split_audio_chunks(y, SR, max_sec=5.0)
    for _, t_s, t_e in chunks:
        assert t_s >= 0.0
        assert t_e <= 12.1


def test_split_chunks_drop_quiet_first():
    sr = SR
    y = np.zeros(sr * 20, dtype=np.float32)
    # First 5 seconds very quiet, rest loud
    y[sr * 5:] = 0.5
    full = split_audio_chunks(y, sr, max_sec=5.0, drop_quiet_first=False)
    dropped = split_audio_chunks(y, sr, max_sec=5.0, drop_quiet_first=True)
    assert len(dropped) <= len(full)


def test_split_chunks_drop_quiet_only_one_chunk():
    """drop_quiet_first with only 1 chunk → keeps it (no drop)."""
    y = np.zeros(SR * 3, dtype=np.float32)
    with_flag = split_audio_chunks(y, SR, max_sec=5.0, drop_quiet_first=True)
    without_flag = split_audio_chunks(y, SR, max_sec=5.0, drop_quiet_first=False)
    assert len(with_flag) == len(without_flag)


# ── cosine_distance / _normalized ────────────────────────────────────────────

def test_normalized_basic():
    v = np.array([2.0, 4.0])
    scale = np.array([2.0, 2.0])
    result = _normalized(v, scale)
    np.testing.assert_allclose(result, [1.0, 2.0])


def test_cosine_distance_identical():
    v = np.array([1.0, 2.0, 3.0])
    scale = np.ones(3)
    assert cosine_distance(v, v, scale) == pytest.approx(0.0, abs=1e-5)


def test_cosine_distance_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    scale = np.ones(2)
    assert cosine_distance(a, b, scale) == pytest.approx(1.0, abs=1e-5)


def test_cosine_distance_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    scale = np.ones(2)
    d = cosine_distance(a, b, scale)
    assert d > 1.0  # 1 - (-1) = 2


# ── classify_vec ─────────────────────────────────────────────────────────────

def test_classify_vec_picks_closest():
    scale = np.ones(2)
    records = [
        {"label": "6a", "filename": "a.wav", "vec": np.array([1.0, 0.0])},
        {"label": "6b", "filename": "b.wav", "vec": np.array([0.0, 1.0])},
    ]
    query = np.array([0.99, 0.01])
    winner, avg_dist, top = classify_vec(query, records, scale, top_k=1)
    assert winner == "6a"


def test_classify_vec_top_k_majority():
    scale = np.ones(2)
    records = [
        {"label": "6a", "filename": "a1.wav", "vec": np.array([1.0, 0.0])},
        {"label": "6a", "filename": "a2.wav", "vec": np.array([0.9, 0.1])},
        {"label": "6b", "filename": "b.wav",  "vec": np.array([0.0, 1.0])},
    ]
    query = np.array([0.95, 0.05])
    winner, _, _ = classify_vec(query, records, scale, top_k=3)
    assert winner == "6a"


# ── get_files ─────────────────────────────────────────────────────────────────

def test_get_files_returns_supported(tmp_path):
    (tmp_path / "song.wav").write_bytes(b"")
    (tmp_path / "song.mp3").write_bytes(b"")
    (tmp_path / "ignore.txt").write_bytes(b"")
    files = get_files(str(tmp_path))
    assert "song.wav" in files
    assert "song.mp3" in files
    assert "ignore.txt" not in files


def test_get_files_sorted(tmp_path):
    (tmp_path / "b.wav").write_bytes(b"")
    (tmp_path / "a.wav").write_bytes(b"")
    files = get_files(str(tmp_path))
    assert files == sorted(files)


# ── filter_files_by_noise ─────────────────────────────────────────────────────

def test_filter_noise_none_returns_all():
    files = ["6a.wav", "6b_f1.wav", "8a_f2.wav"]
    assert filter_files_by_noise(files, None) == files


def test_filter_noise_0_excludes_noisy():
    files = ["6a.wav", "6b_f1.wav"]
    result = filter_files_by_noise(files, "0")
    assert "6a.wav" in result
    assert "6b_f1.wav" not in result


def test_filter_noise_2_includes_up_to_two():
    files = ["6a.wav", "6b_f1.wav", "8a_f1f2.wav", "8b_f1f2f3.wav"]
    result = filter_files_by_noise(files, "2")
    assert "6a.wav" in result
    assert "6b_f1.wav" in result
    assert "8a_f1f2.wav" in result
    assert "8b_f1f2f3.wav" not in result


def test_filter_noise_m_includes_mf():
    files = ["6a.wav", "6b_mf1.wav", "8a_f1.wav"]
    result = filter_files_by_noise(files, "m")
    assert "6a.wav" in result
    assert "6b_mf1.wav" in result
    assert "8a_f1.wav" not in result


# ── _ece_top_label ────────────────────────────────────────────────────────────

def test_ece_perfect_calibration():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([
        [0.9, 0.1],
        [0.9, 0.1],
        [0.1, 0.9],
        [0.1, 0.9],
    ])
    ece = _ece_top_label(y_true, y_proba)
    assert 0.0 <= ece <= 0.15


def test_ece_confident_wrong():
    y_true = np.array([1, 1, 1, 1])
    y_proba = np.array([[0.95, 0.05]] * 4)
    ece = _ece_top_label(y_true, y_proba)
    assert ece > 0.5


def test_ece_returns_float():
    y_true = np.array([0, 1])
    y_proba = np.array([[0.6, 0.4], [0.4, 0.6]])
    assert isinstance(_ece_top_label(y_true, y_proba), float)


# ── _brier_multiclass ─────────────────────────────────────────────────────────

def test_brier_perfect():
    y_true = np.array([0, 1, 2])
    y_proba = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert _brier_multiclass(y_true, y_proba) == pytest.approx(0.0)


def test_brier_uniform():
    y_true = np.array([0, 1])
    y_proba = np.array([[0.5, 0.5], [0.5, 0.5]])
    score = _brier_multiclass(y_true, y_proba)
    assert 0.0 < score < 1.0


def test_brier_returns_float():
    y_true = np.array([0])
    y_proba = np.array([[0.7, 0.3]])
    assert isinstance(_brier_multiclass(y_true, y_proba), float)


# ── cmd_extract (mocked) ──────────────────────────────────────────────────────

@pytest.mark.slow
def test_cmd_extract_creates_csv(tmp_path):
    sr = SR
    y = (np.random.randn(sr * 6)).astype(np.float32)
    dummy_feats = {k: float(np.random.rand()) for k in FEATURE_KEYS}

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "6a_song.wav").write_bytes(b"")
    (audio_dir / "6b_song.wav").write_bytes(b"")

    output_csv = str(tmp_path / "features.csv")

    with patch("ml_pipeline.classify.load_audio", return_value=(y, sr)), \
         patch("ml_pipeline.classify._file_beats", return_value=(None, None)), \
         patch("ml_pipeline.classify.extract_features", return_value=dummy_feats):
        cmd_extract(str(audio_dir), output_csv,
                    max_sec=6.0, hop_ratio=1.0,
                    use_demucs=False, use_beats=False)

    assert os.path.exists(output_csv)
    with open(output_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert "label" in rows[0]


@pytest.mark.slow
def test_cmd_extract_empty_dir_no_crash(tmp_path):
    audio_dir = tmp_path / "empty"
    audio_dir.mkdir()
    output_csv = str(tmp_path / "out.csv")
    cmd_extract(str(audio_dir), output_csv,
                max_sec=5.0, hop_ratio=1.0,
                use_demucs=False, use_beats=False)
    # No exception raised; CSV may not exist
    assert True


# ── cmd_train (real small data) ───────────────────────────────────────────────

def _make_features_csv(path, n_per_class=4):
    """Write a minimal features.csv with two classes."""
    meta = ["file_idx", "filename", "label", "chunk_idx",
            "chunk_start", "chunk_end", "aug"]
    fieldnames = meta + list(FEATURE_KEYS)
    rows = []
    for file_idx, label in enumerate(["6a"] * n_per_class + ["6b"] * n_per_class):
        row = {
            "file_idx": file_idx,
            "filename": f"{label}_{file_idx}.wav",
            "label": label,
            "chunk_idx": 0,
            "chunk_start": 0.0,
            "chunk_end": 5.0,
            "aug": "orig",
        }
        for k in FEATURE_KEYS:
            row[k] = float(np.random.default_rng(file_idx).standard_normal(1)[0])
        rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


@pytest.mark.slow
def test_cmd_train_creates_bundle(tmp_path):
    pytest.importorskip("lightgbm")
    csv_path = str(tmp_path / "features.csv")
    pkl_path = str(tmp_path / "model.pkl")
    _make_features_csv(csv_path, n_per_class=4)

    cmd_train(csv_path, pkl_path,
              n_estimators=5, learning_rate=0.1, num_leaves=4,
              cv_folds=2, calibrate=False)

    assert os.path.exists(pkl_path)
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    assert "pipeline" in bundle
    assert set(bundle["classes"]) == {"6a", "6b"}
    assert "cv_acc" in bundle


@pytest.mark.slow
def test_cmd_train_with_calibration(tmp_path):
    pytest.importorskip("lightgbm")
    csv_path = str(tmp_path / "features.csv")
    pkl_path = str(tmp_path / "model_cal.pkl")
    # Need more samples for calibrated CV (cv=2 inside CalibratedClassifierCV)
    _make_features_csv(csv_path, n_per_class=6)

    cmd_train(csv_path, pkl_path,
              n_estimators=5, learning_rate=0.1, num_leaves=4,
              cv_folds=2, calibrate=True,
              calibration_method="sigmoid", calibration_cv=2)

    assert os.path.exists(pkl_path)
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    assert bundle["calibration"] is not None
    assert "brier_base" in bundle["calibration"]


# ── cmd_predict (mocked) ──────────────────────────────────────────────────────

@pytest.mark.slow
def test_cmd_predict_returns_label(tmp_path):
    pytest.importorskip("lightgbm")
    # Build a minimal pkl
    csv_path = str(tmp_path / "f.csv")
    pkl_path = str(tmp_path / "m.pkl")
    _make_features_csv(csv_path, n_per_class=4)
    cmd_train(csv_path, pkl_path,
              n_estimators=5, learning_rate=0.1, num_leaves=4,
              cv_folds=2, calibrate=False)

    sr = SR
    y = np.random.randn(sr * 6).astype(np.float32)
    dummy_feats = {k: float(np.random.rand()) for k in FEATURE_KEYS}

    with patch("ml_pipeline.classify.load_audio", return_value=(y, sr)), \
         patch("ml_pipeline.classify._file_beats", return_value=(None, None)), \
         patch("ml_pipeline.classify.extract_features", return_value=dummy_feats):
        label, mean_probs = cmd_predict(
            "/fake/audio.wav", pkl_path,
            max_sec=6.0, hop_ratio=1.0,
        )

    assert label in ["6a", "6b"]
    assert len(mean_probs) == 2
    assert abs(sum(mean_probs) - 1.0) < 0.01


# ── cmd_diagnose (mocked) ──────────────────────────────────────────────────────

@pytest.mark.slow
def test_cmd_diagnose_returns_scale(tmp_path):
    from ml_pipeline.classify import cmd_diagnose

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "6a_song.wav").write_bytes(b"")
    (audio_dir / "6b_other.wav").write_bytes(b"")

    sr = SR
    y = np.random.randn(sr * 3).astype(np.float32)
    dummy_feats = {k: float(np.random.rand()) for k in FEATURE_KEYS}

    with patch("ml_pipeline.classify.load_audio", return_value=(y, sr)), \
         patch("ml_pipeline.classify._file_beats", return_value=(None, None)), \
         patch("ml_pipeline.classify.extract_features", return_value=dummy_feats):
        scale = cmd_diagnose(str(audio_dir), use_demucs=False, use_beats=False)

    assert scale is not None
    assert len(scale) == len(FEATURE_KEYS)


# ── cmd_build (mocked) ────────────────────────────────────────────────────────

@pytest.mark.slow
def test_cmd_build_creates_pkl(tmp_path):
    from ml_pipeline.classify import cmd_build

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "6a_song.wav").write_bytes(b"")
    (audio_dir / "6b_other.wav").write_bytes(b"")

    sr = SR
    y = np.random.randn(sr * 3).astype(np.float32)
    dummy_feats = {k: float(np.random.rand()) for k in FEATURE_KEYS}

    output_pkl = str(tmp_path / "db.pkl")

    with patch("ml_pipeline.classify.load_audio", return_value=(y, sr)), \
         patch("ml_pipeline.classify._file_beats", return_value=(None, None)), \
         patch("ml_pipeline.classify.extract_features", return_value=dummy_feats):
        cmd_build(str(audio_dir), output_pkl,
                  max_sec=5.0, hop_ratio=1.0,
                  use_demucs=False, use_beats=False)

    assert os.path.exists(output_pkl)
    with open(output_pkl, "rb") as f:
        db = pickle.load(f)
    assert "records" in db
    assert len(db["records"]) >= 2
