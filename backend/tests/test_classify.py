"""Tests for ml_pipeline/classify.py — chunking and soft-voting inference."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_pipeline.classify import CHUNK_SEC, HOP_SEC, classify, split_audio_chunks

SR = 22050
CLASSES = ["6a", "6b", "8a"]


# ── split_audio_chunks ────────────────────────────────────────────────

def test_split_chunks_no_overlap():
    chunks = split_audio_chunks(np.zeros(SR * 15, dtype=np.float32), SR, max_sec=5.0)
    assert [(ts, te) for _, ts, te in chunks] == [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]


def test_split_chunks_production_settings():
    chunks = split_audio_chunks(np.zeros(SR * 15, dtype=np.float32), SR,
                                max_sec=CHUNK_SEC, hop_sec=HOP_SEC)
    starts = [round(ts, 1) for _, ts, _ in chunks]
    assert starts == [0.0, 4.2, 8.4, 12.6]
    assert chunks[-1][2] == pytest.approx(15.0)


def test_split_chunks_drops_tail_shorter_than_1s():
    chunks = split_audio_chunks(np.zeros(int(SR * 10.5), dtype=np.float32), SR, max_sec=5.0)
    assert len(chunks) == 2


def test_split_chunks_too_short_returns_empty():
    assert split_audio_chunks(np.zeros(SR // 3, dtype=np.float32), SR) == []


def test_split_chunks_drop_quiet_first():
    y = np.zeros(SR * 20, dtype=np.float32)
    y[SR * 5:] = 0.5
    full = split_audio_chunks(y, SR, max_sec=5.0)
    dropped = split_audio_chunks(y, SR, max_sec=5.0, drop_quiet_first=True)
    assert len(dropped) == len(full) - 1
    assert dropped[0][1] == 5.0


def test_split_chunks_drop_quiet_keeps_single_chunk():
    y = np.zeros(SR * 3, dtype=np.float32)
    assert len(split_audio_chunks(y, SR, max_sec=5.0, drop_quiet_first=True)) == 1


# ── classify ──────────────────────────────────────────────────────────

def _bundle(*chunk_probs):
    pipeline = MagicMock()
    pipeline.predict_proba.return_value = np.array(chunk_probs)
    return {"pipeline": pipeline, "classes": CLASSES, "feature_keys": ["rms_mean", "bpm"]}


def _classify(y, bundle, **kw):
    with patch("ml_pipeline.classify.extract_features", return_value={"rms_mean": 0.1}):
        return classify(y, SR, np.array([]), np.array([]), bundle, **kw)


def test_classify_soft_voting_beats_majority():
    # Two chunks lean 6b slightly, one is very confident 8a → mean favours 8a.
    bundle = _bundle([0.30, 0.40, 0.30], [0.30, 0.40, 0.30], [0.0, 0.05, 0.95])
    result = _classify(np.zeros(SR * 14, dtype=np.float32), bundle)
    assert [c["label"] for c in result["chunks"]] == ["6b", "6b", "8a"]
    assert result["final_label"] == "8a"
    assert result["final_conf"] == pytest.approx(0.517, abs=1e-3)
    assert sum(result["mean_probs"].values()) == pytest.approx(1.0, abs=1e-2)


def test_classify_result_format():
    bundle = _bundle([0.9, 0.05, 0.05])
    result = _classify(np.zeros(SR * 6, dtype=np.float32), bundle)
    assert result["n_chunks"] == 1
    assert result["chunks"] == [{
        "chunk_idx": 1, "time_start": 0.0, "time_end": 6.0,
        "label": "6a", "confidence": 0.9,
        "probs": {"6a": 0.9, "6b": 0.05, "8a": 0.05},
    }]


def test_classify_fills_missing_features_with_zero():
    bundle = _bundle([0.9, 0.05, 0.05])
    _classify(np.zeros(SR * 6, dtype=np.float32), bundle)
    X = bundle["pipeline"].predict_proba.call_args.args[0]
    assert list(X.columns) == ["rms_mean", "bpm"]
    assert X.iloc[0].tolist() == [0.1, 0.0]


def test_classify_reports_progress():
    calls = []
    bundle = _bundle([1, 0, 0], [1, 0, 0], [1, 0, 0])
    _classify(np.zeros(SR * 14, dtype=np.float32), bundle,
              progress_cb=lambda d, t: calls.append((d, t)))
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_classify_too_short_raises():
    with pytest.raises(ValueError):
        _classify(np.zeros(SR // 2, dtype=np.float32), _bundle())
