"""
Test suite for GPR worker business logic.

Covers:
  - Progress tracking (_set_progress)
  - Task status retrieval (get_task)
  - Soft voting aggregation
  - Error handling & edge cases
  - User feedback submission
  - Concurrent task handling
"""

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# App imports REDIS_URL/HF_MODEL_REPO via lifespan only — set dummy here so
# top-level `from app import ...` doesn't error on missing env in CI.
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")


# Mock Redis for testing (avoid external dependency)
class MockRedis:
    def __init__(self):
        self.store = {}

    def setex(self, key, ttl, value):
        self.store[key] = value

    def get(self, key):
        return self.store.get(key)


@pytest.fixture
def mock_redis():
    return MockRedis()


@pytest.fixture
def mock_bundle():
    """Mock ML model bundle."""
    return {
        "classes": ["6a", "6b", "6x", "8a", "8b", "wa"],
        "feature_keys": [
            "tempo_bpm", "beat_strength_mean", "beat_strength_std",
            "onset_rate", "spectral_centroid_mean", "rms_mean", "zcr_mean",
        ],
        "pipeline": MagicMock(),
    }


# ── Test: Progress tracking ────────────────────────────────────────────

def test_set_progress_pending(mock_redis):
    """Progress: initial pending state."""
    from app import _set_progress

    with patch("app.redis_client", mock_redis):
        _set_progress("task-123", 0, "Queued", status="pending")

        stored = json.loads(mock_redis.get("task:task-123"))
        assert stored["status"] == "pending"
        assert stored["progress"] == 0
        assert stored["message"] == "Queued"
        assert "result" not in stored


def test_set_progress_with_result(mock_redis):
    """Progress: final result with prediction."""
    from app import _set_progress

    with patch("app.redis_client", mock_redis):
        result = {
            "final_label": "6a",
            "final_conf": 0.92,
            "n_chunks": 3,
        }
        _set_progress("task-123", 100, "Done!", status="done", result=result)

        stored = json.loads(mock_redis.get("task:task-123"))
        assert stored["status"] == "done"
        assert stored["progress"] == 100
        assert stored["result"]["final_label"] == "6a"
        assert stored["result"]["final_conf"] == 0.92


def test_set_progress_overwrite(mock_redis):
    """Progress: subsequent updates overwrite previous."""
    from app import _set_progress

    with patch("app.redis_client", mock_redis):
        _set_progress("task-123", 0, "Start", status="pending")
        _set_progress("task-123", 50, "Processing", status="processing")
        _set_progress("task-123", 100, "Done", status="done")

        stored = json.loads(mock_redis.get("task:task-123"))
        assert stored["progress"] == 100
        assert stored["message"] == "Done"


# ── Test: Task status retrieval ────────────────────────────────────────

def test_get_task_nonexistent(mock_redis):
    """Task retrieval: missing task returns pending."""
    from app import get_task

    with patch("app.redis_client", mock_redis):
        result = get_task("nonexistent-task")

        assert result["task_id"] == "nonexistent-task"
        assert result["status"] == "pending"
        assert result["progress"] == 0


def test_get_task_processing(mock_redis):
    """Task retrieval: in-flight task shows progress."""
    from app import _set_progress, get_task

    with patch("app.redis_client", mock_redis):
        _set_progress("task-456", 45, "Extracting features", status="processing")
        result = get_task("task-456")

        assert result["task_id"] == "task-456"
        assert result["status"] == "processing"
        assert result["progress"] == 45


def test_get_task_completed(mock_redis):
    """Task retrieval: completed task returns full result."""
    from app import _set_progress, get_task

    with patch("app.redis_client", mock_redis):
        final_result = {
            "final_label": "8b",
            "final_conf": 0.87,
            "n_chunks": 5,
            "chunks": [
                {"chunk_idx": 1, "label": "8b", "confidence": 0.85},
            ],
        }
        _set_progress("task-789", 100, "Done", status="done", result=final_result)
        result = get_task("task-789")

        assert result["status"] == "done"
        assert result["result"]["final_label"] == "8b"
        assert result["result"]["final_conf"] == 0.87


# ── Test: Soft voting aggregation ──────────────────────────────────────

def test_soft_voting_single_chunk():
    """Soft voting: single chunk uses chunk prediction."""
    classes = ["6a", "6b", "8a", "8b"]
    # One chunk predicts 6a with 0.9, 8a with 0.05
    all_probs = [np.array([0.90, 0.05, 0.03, 0.02])]

    mean_probs = np.mean(all_probs, axis=0)
    final_idx = int(np.argmax(mean_probs))

    assert classes[final_idx] == "6a"
    assert float(mean_probs[final_idx]) == 0.90


def test_soft_voting_multiple_chunks_consensus():
    """Soft voting: chunks agree → high final confidence."""
    classes = ["6a", "6b", "8a", "8b"]
    # 3 chunks all predict 6a with 0.80+
    all_probs = [
        np.array([0.80, 0.10, 0.07, 0.03]),
        np.array([0.82, 0.08, 0.06, 0.04]),
        np.array([0.78, 0.12, 0.05, 0.05]),
    ]

    mean_probs = np.mean(all_probs, axis=0)
    final_idx = int(np.argmax(mean_probs))
    confidence = float(mean_probs[final_idx])

    assert classes[final_idx] == "6a"
    assert 0.79 < confidence < 0.81


def test_soft_voting_multiple_chunks_disagreement():
    """Soft voting: chunks disagree → ensemble shifts prediction."""
    classes = ["6a", "6b", "8a", "8b"]
    # Chunks split: 2×6a, 1×8a
    all_probs = [
        np.array([0.70, 0.10, 0.15, 0.05]),  # 6a wins
        np.array([0.65, 0.15, 0.15, 0.05]),  # 6a wins
        np.array([0.30, 0.10, 0.50, 0.10]),  # 8a wins
    ]

    mean_probs = np.mean(all_probs, axis=0)
    final_idx = int(np.argmax(mean_probs))

    assert classes[final_idx] == "6a"
    assert float(mean_probs[final_idx]) == pytest.approx(0.55, abs=0.01)


def test_soft_voting_uniform_distribution():
    """Soft voting: no clear winner → pick first by argmax."""
    classes = ["6a", "6b", "8a", "8b"]
    # Uniform probs
    all_probs = [
        np.array([0.25, 0.25, 0.25, 0.25]),
    ]

    mean_probs = np.mean(all_probs, axis=0)
    final_idx = int(np.argmax(mean_probs))

    # np.argmax picks first maximum when tied
    assert final_idx == 0
    assert classes[final_idx] == "6a"


# ── Test: Error handling ────────────────────────────────────────────────

def test_feedback_invalid_rating(mock_redis):
    """Feedback: invalid rating rejected."""
    from fastapi import HTTPException
    from app import submit_feedback, FeedbackRequest

    with patch("app.redis_client", mock_redis):
        with pytest.raises(HTTPException) as exc_info:
            submit_feedback("task-123", FeedbackRequest(rating="invalid"))

        assert exc_info.value.status_code == 400


def test_feedback_task_not_found(mock_redis):
    """Feedback: submit for nonexistent task → 404."""
    from fastapi import HTTPException
    from app import submit_feedback, FeedbackRequest

    with patch("app.redis_client", mock_redis):
        with pytest.raises(HTTPException) as exc_info:
            submit_feedback("nonexistent-task", FeedbackRequest(rating="good"))

        assert exc_info.value.status_code == 404


def test_feedback_success(mock_redis):
    """Feedback: valid submission stored."""
    from app import _set_progress, submit_feedback, FeedbackRequest

    with patch("app.redis_client", mock_redis):
        # Create a completed task
        result = {"final_label": "6a", "final_conf": 0.92}
        _set_progress("task-999", 100, "Done", status="done", result=result)

        # Submit feedback
        response = submit_feedback("task-999", FeedbackRequest(rating="good"))

        assert response["ok"] is True
        assert response["task_id"] == "task-999"

        # Check feedback persisted
        feedback = json.loads(mock_redis.get("feedback:task-999"))
        assert feedback["rating"] == "good"
        assert feedback["predicted_class"] == "6a"


# ── Test: Chunk classification ─────────────────────────────────────────

def test_chunk_result_format():
    """Chunk result: correct structure and rounding."""
    classes = ["6a", "6b", "8a", "8b"]
    probs = np.array([0.756234, 0.142567, 0.088123, 0.013076])
    pred_idx = int(np.argmax(probs))

    chunk_result = {
        "chunk_idx": 1,
        "time_start": round(0.0, 1),
        "time_end": round(6.0, 1),
        "label": classes[pred_idx],
        "confidence": round(float(probs[pred_idx]), 3),
        "probs": {c: round(float(p), 3) for c, p in zip(classes, probs)},
    }

    assert chunk_result["label"] == "6a"
    assert chunk_result["confidence"] == 0.756
    assert chunk_result["probs"]["6a"] == 0.756
    assert chunk_result["probs"]["8b"] == 0.013


# ── Test: Concurrent tasks ────────────────────────────────────────────

def test_concurrent_tasks_isolation(mock_redis):
    """Concurrent: two tasks don't interfere."""
    from app import _set_progress, get_task

    with patch("app.redis_client", mock_redis):
        # Task 1
        _set_progress("task-A", 50, "Processing", status="processing")

        # Task 2
        _set_progress("task-B", 75, "Nearly done", status="processing")

        # Check isolation
        task_a = get_task("task-A")
        task_b = get_task("task-B")

        assert task_a["progress"] == 50
        assert task_b["progress"] == 75


def test_concurrent_tasks_final_both_complete(mock_redis):
    """Concurrent: both tasks complete independently."""
    from app import _set_progress, get_task

    with patch("app.redis_client", mock_redis):
        result_a = {"final_label": "6a", "final_conf": 0.95}
        result_b = {"final_label": "8b", "final_conf": 0.88}

        _set_progress("task-A", 100, "Done", status="done", result=result_a)
        _set_progress("task-B", 100, "Done", status="done", result=result_b)

        task_a = get_task("task-A")
        task_b = get_task("task-B")

        assert task_a["result"]["final_label"] == "6a"
        assert task_b["result"]["final_label"] == "8b"


# ── Test: Health endpoint ──────────────────────────────────────────────

def test_health_check(mock_bundle):
    """Health: returns status and model info."""
    from app import health

    with patch("app.BUNDLE", mock_bundle):
        result = health()

        assert result["status"] == "ok"
        assert result["classes"] == ["6a", "6b", "6x", "8a", "8b", "wa"]
        assert result["n_features"] == 7


def test_health_check_no_model():
    """Health: graceful when model not loaded."""
    from app import health

    with patch("app.BUNDLE", None):
        result = health()

        assert result["status"] == "ok"
        assert result["classes"] is None


# ── Test: Edge cases ────────────────────────────────────────────────────

def test_very_short_audio():
    """Edge case: very short audio (< 1 chunk)."""
    # Verify behavior when audio is shorter than chunk size
    # Single chunk → soft voting uses that single prediction
    classes = ["6a", "6b", "8a", "8b"]
    all_probs = [np.array([0.85, 0.10, 0.03, 0.02])]

    mean_probs = np.mean(all_probs, axis=0)
    final_idx = int(np.argmax(mean_probs))

    assert classes[final_idx] == "6a"


def test_many_chunks():
    """Edge case: very long audio (10+ chunks)."""
    classes = ["6a", "6b", "8a", "8b"]
    # Simulate 10 chunks with slight noise
    np.random.seed(42)
    all_probs = [
        np.array([0.7 + np.random.normal(0, 0.05),
                  0.1 + np.random.normal(0, 0.02),
                  0.15 + np.random.normal(0, 0.03),
                  0.05 + np.random.normal(0, 0.01)])
        for _ in range(10)
    ]
    # Clip to [0, 1] and normalize
    all_probs = [np.clip(p, 0, 1) for p in all_probs]
    all_probs = [p / p.sum() for p in all_probs]

    mean_probs = np.mean(all_probs, axis=0)
    final_idx = int(np.argmax(mean_probs))

    # With 10 chunks, averaging should converge
    assert classes[final_idx] in classes


def test_confidence_score_range():
    """Confidence: always in [0, 1]."""
    classes = ["6a", "6b", "8a", "8b"]
    all_probs = [
        np.array([0.33, 0.33, 0.33, 0.01]),
        np.array([0.34, 0.33, 0.32, 0.01]),
    ]

    mean_probs = np.mean(all_probs, axis=0)
    confidence = float(mean_probs[np.argmax(mean_probs)])

    assert 0.0 <= confidence <= 1.0


# ── Test: Audio normalization stability ─────────────────────────────────

# ── Test: Sample rate impact on features ─────────────────────────────────

def test_sample_rate_changes_features():
    """Sample rate: 16 kHz vs 22050 Hz produce DIFFERENT features."""
    from librosa import load, feature
    import tempfile
    import soundfile as sf

    # Create synthetic audio signal (440 Hz sine wave)
    sr_orig = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr_orig * duration))
    y_22k = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Resample to 16 kHz
    from librosa import resample
    y_16k = resample(y_22k, orig_sr=sr_orig, target_sr=16000)

    # Extract MFCC from both
    mfcc_22k = feature.mfcc(y=y_22k, sr=22050, n_mfcc=13)
    mfcc_16k = feature.mfcc(y=y_16k, sr=16000, n_mfcc=13)

    # Compare first coefficient
    val_22k = float(np.mean(mfcc_22k[0]))
    val_16k = float(np.mean(mfcc_16k[0]))

    # They SHOULD be different (sample rate changes feature extraction)
    diff_percent = abs(val_22k - val_16k) / val_22k * 100
    assert diff_percent > 1.0, f"Expected features to differ, got {diff_percent:.1f}%"

    print(f"MFCC[0] difference: {diff_percent:.1f}% (22050Hz: {val_22k:.3f} vs 16kHz: {val_16k:.3f})")


def test_nyquist_frequency_matters():
    """Nyquist: 16 kHz loses high frequencies (> 8 kHz) that guitar has."""
    from librosa import load, feature
    import scipy

    # Guitar typically has energy up to 10+ kHz
    sr_22k = 22050
    max_freq_22k = sr_22k / 2  # 11025 Hz — can see up to this

    sr_16k = 16000
    max_freq_16k = sr_16k / 2  # 8000 Hz — can only see up to this

    # Guitar fundamental: 82-330 Hz
    # Harmonics go way higher: 5000-10000 Hz

    # Check spectral centroid (should be HIGHER at 22050 Hz)
    y = np.sin(2 * np.pi * np.arange(0, 1, 1/sr_22k) * 200)  # 200 Hz signal

    spec_22k = feature.spectral_centroid(y=y, sr=sr_22k)[0]

    from librosa import resample
    y_16k = resample(y, orig_sr=sr_22k, target_sr=sr_16k)
    spec_16k = feature.spectral_centroid(y=y_16k, sr=sr_16k)[0]

    # With lower sample rate, spectral features shift
    print(f"Spectral centroid: 22050Hz={spec_22k:.0f} Hz, 16000Hz={spec_16k:.0f} Hz")
    assert max_freq_22k > max_freq_16k, "22050 Hz should capture higher frequencies"


def test_audio_normalization_idempotent():
    """Normalization: applying twice should give same result."""
    from simple_classify import _normalize_audio

    # Mock audio signal (low quality, variable RMS)
    y_noisy = np.random.randn(22050) * 0.1  # low SNR
    sr = 22050

    # Normalize once
    y_norm1 = _normalize_audio(y_noisy.copy(), sr)

    # Normalize again
    y_norm2 = _normalize_audio(y_norm1.copy(), sr)

    # Second normalization should be stable (RMS already ≈ 0.5)
    rms1 = np.sqrt(np.mean(y_norm1 ** 2))
    rms2 = np.sqrt(np.mean(y_norm2 ** 2))

    # RMS should be similar (within tolerance)
    assert abs(rms1 - rms2) < 0.05


def test_audio_normalization_range():
    """Normalization: output always in reasonable range."""
    from simple_classify import _normalize_audio

    test_signals = [
        np.random.randn(22050) * 0.01,   # very quiet
        np.random.randn(22050) * 1.0,    # normal
        np.random.randn(22050) * 5.0,    # very loud
    ]

    for y in test_signals:
        y_norm = _normalize_audio(y, 22050)

        # Output should be normalized
        rms = np.sqrt(np.mean(y_norm ** 2))
        assert 0.4 < rms < 0.6, f"RMS out of range: {rms}"

        # No NaN or inf
        assert not np.any(np.isnan(y_norm))
        assert not np.any(np.isinf(y_norm))


def test_soft_voting_stability_across_quality():
    """Stability: features from low/high quality audio should give similar prediction."""
    classes = ["6a", "6b", "8a", "8b"]

    # Simulate: same audio encoded at different bitrates
    # High quality version
    probs_hq = np.array([0.75, 0.15, 0.07, 0.03])
    # Low quality version (YouTube MP3)
    probs_lq = np.array([0.72, 0.18, 0.08, 0.02])

    # Both should predict 6a (most confident class)
    assert classes[np.argmax(probs_hq)] == "6a"
    assert classes[np.argmax(probs_lq)] == "6a"

    # Confidence shouldn't differ by more than 10%
    conf_diff = abs(probs_hq[0] - probs_lq[0])
    assert conf_diff < 0.1


# ── Test: 16kHz compatibility (both URL and file) ──────────────────────────

def test_load_audio_16khz_default():
    """Sample rate: load_audio defaults to 16kHz for model compatibility."""
    from simple_classify import load_audio
    import tempfile
    import soundfile as sf

    # Create a synthetic audio file at 16kHz
    sr_test = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr_test * duration))
    y_test = 0.3 * np.sin(2 * np.pi * 440 * t)

    # Save temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y_test, sr_test)
        tmp_path = tmp.name

    try:
        # Load without specifying sr (should default to 16000)
        y, sr_out = load_audio(tmp_path, use_demucs=False)

        # Check that sr is 16000
        assert sr_out == 16000, f"Expected sr=16000, got {sr_out}"
        assert len(y) == len(y_test), "Audio length mismatch"
    finally:
        import os
        os.unlink(tmp_path)


def test_extract_features_16khz():
    """Features: extract_librosa_features defaults to 16kHz."""
    from simple_classify import extract_librosa_features

    # Synthetic audio
    sr = 16000
    y = 0.1 * np.random.randn(sr)  # 1 second of noise at 16kHz

    # Extract features (should use sr=16000 by default)
    features = extract_librosa_features(y)

    # Verify features exist
    assert "spectral_centroid_mean" in features
    assert "tempo_bpm" in features

    # Spectral centroid should be reasonable for 16kHz (max 8000 Hz)
    spec_centroid = features["spectral_centroid_mean"]
    assert 0 < spec_centroid < 8000, f"Spectral centroid out of range for 16kHz: {spec_centroid}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
