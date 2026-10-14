"""Tests for app.py — HTTP endpoints and the background pipeline (ML stages mocked)."""

import base64
import json
import os
import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import _load_model, _set_progress, check_basic_auth, run_predict

SR = 22050
CLASSES = ["6a", "6b", "8a"]


@pytest.fixture
def bundle():
    pipeline = MagicMock()
    pipeline.predict_proba.side_effect = lambda X: np.tile([0.7, 0.2, 0.1], (len(X), 1))
    return {"pipeline": pipeline, "classes": CLASSES, "feature_keys": ["rms_mean"]}


@pytest.fixture
def client(mock_redis, bundle):
    # TestClient without `with` skips lifespan, so no real Redis / model download.
    with patch("app.redis_client", mock_redis), patch("app.BUNDLE", bundle):
        yield TestClient(app_module.app)


def _stored(mock_redis, task_id):
    return json.loads(mock_redis.get(f"task:{task_id}"))


def _basic(user, password):
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


# ── Progress / tasks ──────────────────────────────────────────────────

def test_set_progress_without_result(mock_redis):
    with patch("app.redis_client", mock_redis):
        _set_progress("t1", 40, "working")
    assert _stored(mock_redis, "t1") == {"status": "processing", "progress": 40,
                                         "message": "working"}


def test_set_progress_with_result(mock_redis):
    with patch("app.redis_client", mock_redis):
        _set_progress("t1", 100, "done", status="done", result={"final_label": "6a"})
    assert _stored(mock_redis, "t1")["result"] == {"final_label": "6a"}


def test_get_task_returns_stored_state(client, mock_redis):
    with patch("app.redis_client", mock_redis):
        _set_progress("t1", 55, "beats")
    r = client.get("/tasks/t1")
    assert r.status_code == 200
    assert r.json() == {"task_id": "t1", "status": "processing", "progress": 55,
                        "message": "beats"}


def test_get_task_unknown_is_404(client):
    # Previously returned "pending" forever, so the frontend polled indefinitely.
    assert client.get("/tasks/nope").status_code == 404


# ── /predict/file ─────────────────────────────────────────────────────

def test_predict_file_queues_task(client, mock_redis):
    with patch("app.run_predict") as run:
        r = client.post("/predict/file", files={"file": ("song.wav", b"RIFF", "audio/wav")})
    assert r.status_code == 200
    task_id = r.json()["task_id"]
    assert _stored(mock_redis, task_id)["status"] == "pending"
    audio_path = run.call_args.kwargs["audio_path"]
    assert os.path.basename(audio_path) == "song.wav"
    assert os.path.dirname(audio_path) == run.call_args.kwargs["cleanup_dir"]


def test_predict_file_rejects_oversized_upload(client):
    with patch("app.MAX_UPLOAD_MB", 1), patch("app.run_predict") as run,          patch("app.shutil.rmtree") as rmtree:
        r = client.post("/predict/file",
                        files={"file": ("big.wav", b"x" * (1024 * 1024 + 1), "audio/wav")})
    assert r.status_code == 413
    run.assert_not_called()
    rmtree.assert_called_once()


def test_predict_file_accepts_upload_at_limit(client):
    with patch("app.MAX_UPLOAD_MB", 1), patch("app.run_predict"):
        r = client.post("/predict/file",
                        files={"file": ("ok.wav", b"x" * (1024 * 1024), "audio/wav")})
    assert r.status_code == 200


def test_predict_file_strips_path_from_filename(client):
    with patch("app.run_predict") as run:
        client.post("/predict/file",
                    files={"file": ("../../evil.wav", b"RIFF", "audio/wav")})
    kwargs = run.call_args.kwargs
    assert os.path.dirname(kwargs["audio_path"]) == kwargs["cleanup_dir"]
    assert os.path.basename(kwargs["audio_path"]) == "evil.wav"


# ── run_predict ───────────────────────────────────────────────────────

def _run(mock_redis, bundle, y, task_id="t1", cleanup_dir=None, **patches):
    with patch("app.redis_client", mock_redis), \
         patch("app.BUNDLE", bundle), \
         patch("app.separate_guitar", patches.get("separate", MagicMock(return_value=(y, SR)))), \
         patch("app.compute_beats", MagicMock(return_value=(np.array([]), np.array([])))), \
         patch("ml_pipeline.classify.extract_features", return_value={"rms_mean": 0.1}):
        run_predict(task_id, audio_path="/fake/audio.wav", cleanup_dir=cleanup_dir)
    return _stored(mock_redis, task_id)


def test_run_predict_success(mock_redis, bundle):
    stored = _run(mock_redis, bundle, np.zeros(SR * 14, dtype=np.float32))
    assert stored["status"] == "done"
    assert stored["progress"] == 100
    assert stored["result"]["final_label"] == "6a"
    assert stored["result"]["final_conf"] == 0.7
    assert stored["result"]["n_chunks"] == 3


def test_run_predict_progress_never_goes_backwards(mock_redis, bundle):
    seen = []
    real_setex = mock_redis.setex
    mock_redis.setex = lambda k, ttl, v: (seen.append(json.loads(v)["progress"]),
                                         real_setex(k, ttl, v))

    def fake_demucs(path, progress_cb):
        for pct in (0, 50, 100):
            progress_cb(pct)
        return np.zeros(SR * 14, dtype=np.float32), SR

    _run(mock_redis, bundle, None, separate=MagicMock(side_effect=fake_demucs))
    assert seen == sorted(seen)
    assert seen[-1] == 100


def test_run_predict_demucs_error(mock_redis, bundle):
    stored = _run(mock_redis, bundle, None,
                  separate=MagicMock(side_effect=RuntimeError("demucs failed")))
    assert stored["status"] == "error"
    assert "Demucs" in stored["message"]


def test_run_predict_too_short_audio_is_error(mock_redis, bundle):
    stored = _run(mock_redis, bundle, np.zeros(SR // 2, dtype=np.float32))
    assert stored["status"] == "error"
    assert "коротке" in stored["message"]


def test_run_predict_removes_cleanup_dir(mock_redis, bundle, tmp_path):
    work = tmp_path / "upload"
    work.mkdir()
    _run(mock_redis, bundle, np.zeros(SR * 6, dtype=np.float32), cleanup_dir=str(work))
    assert not work.exists()


# ── Feedback ──────────────────────────────────────────────────────────

def test_feedback_success(client, mock_redis):
    with patch("app.redis_client", mock_redis):
        _set_progress("t1", 100, status="done", result={"final_label": "8a"})
    r = client.post("/feedback/t1", json={"rating": "good"})
    assert r.status_code == 200
    saved = json.loads(mock_redis.get("feedback:t1"))
    assert saved["predicted_class"] == "8a"
    assert saved["rating"] == "good"


def test_feedback_invalid_rating(client, mock_redis):
    with patch("app.redis_client", mock_redis):
        _set_progress("t1", 100, status="done", result={"final_label": "8a"})
    assert client.post("/feedback/t1", json={"rating": "meh"}).status_code == 422


def test_feedback_task_not_found(client):
    assert client.post("/feedback/nope", json={"rating": "bad"}).status_code == 404


def test_restore_feedback_counters(mock_redis):
    mock_redis.set("feedback:a", json.dumps({"predicted_class": "6a", "rating": "good"}))
    mock_redis.set("feedback:b", json.dumps({"predicted_class": "6a", "rating": "bogus"}))
    with patch("app.redis_client", mock_redis):
        assert app_module._restore_feedback_counters() == 1


# ── Health / metrics ──────────────────────────────────────────────────

def test_health(client):
    assert client.get("/").json() == {"status": "ok", "classes": CLASSES, "n_features": 1}


def test_health_before_model_loaded(mock_redis):
    with patch("app.BUNDLE", None):
        assert TestClient(app_module.app).get("/").json()["classes"] is None


@pytest.fixture
def metrics_creds():
    with patch("app.METRICS_USER", "grafana"), patch("app.METRICS_PASSWORD", "s3cret"):
        yield


def test_metrics_disabled_without_secrets(client):
    with patch("app.METRICS_USER", ""), patch("app.METRICS_PASSWORD", ""):
        assert client.get("/metrics", headers={
            "Authorization": _basic("", "")}).status_code == 404


def test_metrics_requires_auth(client, metrics_creds):
    assert client.get("/metrics").status_code == 401
    ok = client.get("/metrics", headers={"Authorization": _basic("grafana", "s3cret")})
    assert ok.status_code == 200
    assert "gpr_predictions_total" in ok.text


@pytest.mark.parametrize("header", [
    None,
    "",
    "Basic",
    "Bearer abc",
    "Basic !!!not-base64!!!",
    _basic("grafana", "wrong"),
    _basic("someone", "s3cret"),
])
def test_check_basic_auth_rejects(header, metrics_creds):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        check_basic_auth(header)
    assert exc.value.status_code == 401


def test_check_basic_auth_accepts_password_with_colon():
    with patch("app.METRICS_USER", "grafana"), patch("app.METRICS_PASSWORD", "a:b"):
        check_basic_auth(_basic("grafana", "a:b"))


# ── Model loading ─────────────────────────────────────────────────────

def test_load_model_from_local_file(tmp_path):
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps({"classes": CLASSES}))
    with patch.dict(os.environ, {"MODEL_PATH": str(path)}):
        assert _load_model() == {"classes": CLASSES}


def test_load_model_missing_without_repo_raises():
    env = {k: v for k, v in os.environ.items() if k != "HF_MODEL_REPO"}
    env["MODEL_PATH"] = "/nonexistent/model.pkl"
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(RuntimeError, match="HF_MODEL_REPO"):
            _load_model()
