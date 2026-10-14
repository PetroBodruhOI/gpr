"""Tests for train.py — offline dataset extraction and model training."""

import csv
import pickle
from unittest.mock import patch

import numpy as np
import pytest

import train  # must come first: puts backend/ on sys.path for ml_pipeline
from ml_pipeline.feature_extract import FEATURE_KEYS
from train import (
    META_COLS,
    _brier_multiclass,
    _ece_top_label,
    cmd_extract,
    cmd_train,
    filter_files_by_noise,
    get_files,
)

SR = 22050


# ── Dataset files ─────────────────────────────────────────────────────

def test_get_files_supported_and_sorted(tmp_path):
    for name in ("b.wav", "a.MP3", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert get_files(str(tmp_path)) == ["a.MP3", "b.wav"]


@pytest.mark.parametrize("noise, expected", [
    (None, ["6a.wav", "6b_f1.wav", "8a_f1f2.wav", "8b_mf1.wav"]),
    ("0", ["6a.wav"]),
    ("2", ["6a.wav", "6b_f1.wav", "8a_f1f2.wav", "8b_mf1.wav"]),
    ("m", ["6a.wav", "8b_mf1.wav"]),
])
def test_filter_files_by_noise(noise, expected):
    files = ["6a.wav", "6b_f1.wav", "8a_f1f2.wav", "8b_mf1.wav"]
    assert filter_files_by_noise(files, noise) == expected


# ── Calibration metrics ───────────────────────────────────────────────

def test_ece_well_calibrated_is_low():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
    assert _ece_top_label(y_true, y_proba) == pytest.approx(0.1)


def test_ece_confident_and_wrong_is_high():
    assert _ece_top_label(np.array([1, 1]), np.array([[0.95, 0.05]] * 2)) > 0.9


def test_brier_perfect_and_uniform():
    assert _brier_multiclass(np.array([0, 1]), np.eye(2)) == pytest.approx(0.0)
    assert _brier_multiclass(np.array([0, 1]), np.full((2, 2), 0.5)) == pytest.approx(0.5)


# ── Cached loading ────────────────────────────────────────────────────

def test_load_beats_uses_disk_cache(tmp_path):
    audio = tmp_path / "6a.wav"
    audio.write_bytes(b"")
    with patch.object(train, "BEATS_CACHE_DIR", str(tmp_path / "cache")), \
         patch("train.compute_beats", return_value=(np.array([1.0]), np.array([1.0]))) as cb:
        train.load_beats(str(audio), None, SR, use_beats=True)
        beats, _ = train.load_beats(str(audio), None, SR, use_beats=True)
    assert cb.call_count == 1
    assert beats.tolist() == [1.0]


def test_load_beats_disabled():
    assert train.load_beats("x.wav", None, SR, use_beats=False) == (None, None)


# ── extract ───────────────────────────────────────────────────────────

def test_cmd_extract_writes_three_augmented_rows_per_chunk(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "6a_song.wav").write_bytes(b"")
    (audio_dir / "8b_song.wav").write_bytes(b"")
    out = tmp_path / "features.csv"
    feats = {k: 0.5 for k in FEATURE_KEYS}

    with patch("train.load_audio", return_value=(np.zeros(SR * 6, dtype=np.float32), SR)), \
         patch("train.load_beats", return_value=(None, None)), \
         patch("train.extract_features", return_value=feats):
        cmd_extract(str(audio_dir), str(out), max_sec=6.0, hop_ratio=0.7,
                    use_demucs=False, use_beats=False)

    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == META_COLS + FEATURE_KEYS
    assert [(r["label"], r["aug"]) for r in rows] == [
        ("6a", "orig"), ("6a", "gain"), ("6a", "noise"),
        ("8b", "orig"), ("8b", "gain"), ("8b", "noise"),
    ]


def test_cmd_extract_empty_dir_writes_nothing(tmp_path):
    out = tmp_path / "features.csv"
    cmd_extract(str(tmp_path), str(out), 6.0, 0.7, False, False)
    assert not out.exists()


# ── train ─────────────────────────────────────────────────────────────

def _make_features_csv(path, n_per_class):
    rng = np.random.default_rng(0)
    rows = []
    for file_idx, label in enumerate(["6a"] * n_per_class + ["6b"] * n_per_class):
        shift = 0.0 if label == "6a" else 3.0
        for chunk_idx in range(3):
            row = {"file_idx": file_idx, "filename": f"{label}_{file_idx}.wav",
                   "label": label, "chunk_idx": chunk_idx, "chunk_start": 0.0,
                   "chunk_end": 6.0, "aug": "orig"}
            row.update({k: shift + rng.standard_normal() for k in FEATURE_KEYS})
            rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_COLS + FEATURE_KEYS)
        w.writeheader()
        w.writerows(rows)


@pytest.mark.slow
@pytest.mark.parametrize("calibrate", [False, True])
def test_cmd_train_bundle_is_servable(tmp_path, calibrate):
    """The saved bundle has exactly what app.py / classify() need."""
    pytest.importorskip("lightgbm")
    csv_path, pkl_path = tmp_path / "f.csv", tmp_path / "model.pkl"
    _make_features_csv(csv_path, n_per_class=6)

    cmd_train(str(csv_path), str(pkl_path), n_estimators=10, learning_rate=0.1,
              num_leaves=4, cv_folds=2, calibrate=calibrate,
              calibration_method="sigmoid", calibration_cv=2)

    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    assert bundle["classes"] == ["6a", "6b"]
    assert bundle["feature_keys"] == FEATURE_KEYS
    assert (bundle["calibration"] is not None) == calibrate
    proba = bundle["pipeline"].predict_proba(np.zeros((1, len(FEATURE_KEYS))))
    assert proba.shape == (1, 2)


# ── CLI ───────────────────────────────────────────────────────────────

def test_main_dispatches_train():
    with patch("train.cmd_train") as ct:
        train.main(["train", "--csv", "f.csv", "--no_calibrate"])
    assert ct.call_args.args == ("f.csv", "./model.pkl")
    assert ct.call_args.kwargs["calibrate"] is False


def test_main_dispatches_extract():
    with patch("train.cmd_extract") as ce:
        train.main(["extract", "--use_beats", "--noise", "m"])
    assert ce.call_args.args == ("./audio", "./features.csv", 6.0, 0.7, False, True)
    assert ce.call_args.kwargs == {"noise": "m"}


def test_main_requires_command():
    with pytest.raises(SystemExit):
        train.main([])
