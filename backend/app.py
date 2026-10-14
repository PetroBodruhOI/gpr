"""
GPR worker — FastAPI app for HuggingFace Spaces.

POST /predict/file stores the upload and runs the ML pipeline in a FastAPI
BackgroundTask; progress and results go to Redis so the frontend can poll
GET /tasks/{task_id}.

Monitoring:
  - /metrics             — Prometheus exposition (basic auth; off unless
                           METRICS_USER / METRICS_PASSWORD are set)
  - JSON logs on stdout  — parseable by Grafana Loki / any aggregator
  - /feedback/{task_id}  — user rating of a prediction (good / bad)
"""

import base64
import binascii
import json
import logging
import os
import pickle
import secrets
import shutil
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

import aiofiles
import redis as redis_lib
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from pythonjsonlogger import jsonlogger

from ml_pipeline.classify import classify
from ml_pipeline.preprocess import compute_beats, separate_guitar

TASK_TTL_SEC = 3600
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
UPLOAD_CHUNK_BYTES = 1024 * 1024

# ── Structured logging ────────────────────────────────────────────────

_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.setFormatter(jsonlogger.JsonFormatter(
    "%(asctime)s %(name)s %(levelname)s %(message)s"
))
logging.basicConfig(level=logging.INFO, handlers=[_log_handler], force=True)
logger = logging.getLogger("gpr")

# ── Prometheus metrics ────────────────────────────────────────────────

PREDICTIONS_TOTAL = Counter(
    "gpr_predictions_total",
    "Total predictions made",
    labelnames=["predicted_class", "status"],
)

INFERENCE_DURATION = Histogram(
    "gpr_inference_duration_seconds",
    "End-to-end inference time (upload → final classification)",
    labelnames=["predicted_class"],
    buckets=(5, 10, 20, 30, 60, 120, 180, 300, 600),
)

STAGE_DURATION = Histogram(
    "gpr_stage_duration_seconds",
    "Time spent in each pipeline stage",
    labelnames=["stage"],          # demucs / beat / classify
    buckets=(1, 3, 5, 10, 20, 30, 60, 120, 300),
)

AUDIO_DURATION = Histogram(
    "gpr_audio_duration_seconds",
    "Length of input audio after Demucs",
    buckets=(5, 10, 20, 30, 60, 120, 300),
)

CONFIDENCE = Histogram(
    "gpr_inference_confidence",
    "Final confidence score returned to the user",
    labelnames=["predicted_class"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

FEEDBACK_TOTAL = Counter(
    "gpr_user_feedback_total",
    "User ratings on predictions",
    labelnames=["predicted_class", "rating"],   # rating: good / bad
)


# ── Startup: connect Redis, load model ────────────────────────────────

BUNDLE: Optional[dict] = None
redis_client: Optional[redis_lib.Redis] = None


def _load_model() -> dict:
    path = os.environ.get("MODEL_PATH", "/app/model.pkl")
    if not os.path.exists(path):
        repo = os.environ.get("HF_MODEL_REPO")
        if not repo:
            raise RuntimeError(
                f"No model at {path} and HF_MODEL_REPO env var not set"
            )
        from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]
        logger.info("downloading_model", extra={"hf_repo": repo})
        path = hf_hub_download(
            repo_id=repo,
            filename=os.environ.get("HF_MODEL_FILE", "model.pkl"),
            token=os.environ.get("HF_TOKEN") or None,
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _restore_feedback_counters() -> int:
    """Prometheus counters reset on restart; rebuild FEEDBACK_TOTAL from Redis."""
    restored = 0
    try:
        for key in redis_client.scan_iter("feedback:*"):
            raw = redis_client.get(key)
            if not raw:
                continue
            data = json.loads(raw)
            rating = data.get("rating")
            if rating in ("good", "bad"):
                FEEDBACK_TOTAL.labels(
                    predicted_class=data.get("predicted_class", "unknown"),
                    rating=rating,
                ).inc()
                restored += 1
    except Exception as e:
        logger.warning("feedback_restore_failed", extra={"error": str(e)})
    return restored


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global BUNDLE, redis_client
    redis_client = redis_lib.from_url(os.environ["REDIS_URL"])
    BUNDLE = _load_model()
    restored = _restore_feedback_counters()
    logger.info("startup_complete", extra={
        "classes": BUNDLE["classes"],
        "n_features": len(BUNDLE["feature_keys"]),
        "feedback_restored": restored,
    })
    yield


app = FastAPI(title="GPR Worker", lifespan=lifespan)

_allowed = os.environ.get("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed.split(",") if o.strip()] or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Metrics auth ──────────────────────────────────────────────────────
# Set both as HF Space secrets to enable /metrics; unset → endpoint is off (404).
METRICS_USER = os.environ.get("METRICS_USER", "")
METRICS_PASSWORD = os.environ.get("METRICS_PASSWORD", "")


def check_basic_auth(authorization: Optional[str]) -> None:
    if not (METRICS_USER and METRICS_PASSWORD):
        raise HTTPException(status_code=404)
    try:
        scheme, credentials = (authorization or "").split(" ", 1)
        username, _, password = base64.b64decode(credentials).decode().partition(":")
    except (ValueError, binascii.Error, UnicodeDecodeError):
        raise HTTPException(status_code=401)

    ok = (scheme.lower() == "basic"
          and secrets.compare_digest(username, METRICS_USER)
          and secrets.compare_digest(password, METRICS_PASSWORD))
    if not ok:
        raise HTTPException(status_code=401)


# ── Pipeline ──────────────────────────────────────────────────────────

def _set_progress(task_id: str, progress: int, message: str = "",
                  status: str = "processing", result=None) -> None:
    payload = {"status": status, "progress": progress, "message": message}
    if result:
        payload["result"] = result
    redis_client.setex(f"task:{task_id}", TASK_TTL_SEC, json.dumps(payload))


def _fail(task_id: str, message: str, status: str, error: Exception) -> None:
    _set_progress(task_id, 0, message, status="error")
    PREDICTIONS_TOTAL.labels(predicted_class="none", status=status).inc()
    logger.error("prediction_failed", extra={
        "task_id": task_id, "stage": status, "error": str(error),
    })


def run_predict(task_id: str, audio_path: str, cleanup_dir: Optional[str] = None) -> None:
    """Synchronous pipeline — invoked by FastAPI BackgroundTasks.

    Progress: 5 start → 5-50 Demucs → 55-69 BeatThis → 70-95 features → 100 done.
    """
    t_total = time.perf_counter()
    try:
        _set_progress(task_id, 5, "Виділяю гітарний стем (HTDemucs)…")

        # ── 1. Demucs ──
        def _demucs_cb(pct: int) -> None:
            _set_progress(task_id, 5 + pct * 45 // 100, f"HTDemucs: {pct}%…")

        t_stage = time.perf_counter()
        try:
            y, sr = separate_guitar(audio_path, progress_cb=_demucs_cb)
        except Exception as e:
            _fail(task_id, f"Demucs error: {e}", "demucs_error", e)
            return
        STAGE_DURATION.labels(stage="demucs").observe(time.perf_counter() - t_stage)
        audio_length_s = len(y) / sr
        AUDIO_DURATION.observe(audio_length_s)

        # ── 2. BeatThis ──
        _set_progress(task_id, 55, "Аналізую ритм (BeatThis)…")

        def _beat_cb(elapsed_s: int) -> None:
            # BeatThis typically takes 16-28 s; cap at 69 so 70 means "done"
            pct = min(69, 55 + elapsed_s * 14 // 24)
            _set_progress(task_id, pct, f"Аналізую ритм… ({elapsed_s}с)")

        t_stage = time.perf_counter()
        beats, downbeats = compute_beats(y, sr, progress_cb=_beat_cb)
        STAGE_DURATION.labels(stage="beat").observe(time.perf_counter() - t_stage)

        # ── 3. Features + classification ──
        _set_progress(task_id, 70, "Витягую ознаки…")

        def _chunk_cb(done: int, total: int) -> None:
            _set_progress(task_id, 70 + 25 * done // total,
                          f"Chunk {done}/{total} оброблено…")

        t_stage = time.perf_counter()
        result = classify(y, sr, beats, downbeats, BUNDLE, progress_cb=_chunk_cb)
        STAGE_DURATION.labels(stage="classify").observe(time.perf_counter() - t_stage)

        _set_progress(task_id, 100, "Готово!", status="done", result=result)

        # ── Final metrics ──
        predicted_class = result["final_label"]
        total_duration = time.perf_counter() - t_total
        INFERENCE_DURATION.labels(predicted_class=predicted_class).observe(total_duration)
        CONFIDENCE.labels(predicted_class=predicted_class).observe(result["final_conf"])
        PREDICTIONS_TOTAL.labels(predicted_class=predicted_class, status="done").inc()
        logger.info("prediction_complete", extra={
            "task_id": task_id,
            "predicted_class": predicted_class,
            "confidence": result["final_conf"],
            "audio_duration_s": round(audio_length_s, 1),
            "inference_duration_s": round(total_duration, 1),
            "n_chunks": result["n_chunks"],
        })

    except Exception as e:
        _fail(task_id, str(e), "error", e)
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


# ── HTTP API ──────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    rating: Literal["good", "bad"]


@app.get("/")
def health():
    return {
        "status": "ok",
        "classes": BUNDLE["classes"] if BUNDLE else None,
        "n_features": len(BUNDLE["feature_keys"]) if BUNDLE else None,
    }


@app.get("/metrics")
def metrics(authorization: Optional[str] = Header(default=None)):
    check_basic_auth(authorization)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict/file")
async def predict_file(file: UploadFile, bg: BackgroundTasks):
    tmp_dir = tempfile.mkdtemp()
    # basename() so a crafted filename like "../../app.py" can't escape tmp_dir
    filename = os.path.basename(file.filename or "") or "audio.wav"
    tmp_path = os.path.join(tmp_dir, filename)

    # Stream to disk so a huge upload never sits in memory; stop at the limit.
    written, too_big = 0, False
    async with aiofiles.open(tmp_path, "wb") as f:
        while chunk := await file.read(UPLOAD_CHUNK_BYTES):
            written += len(chunk)
            if written > MAX_UPLOAD_MB * 1024 * 1024:
                too_big = True
                break
            await f.write(chunk)
    if too_big:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=413, detail=f"Файл більший за {MAX_UPLOAD_MB} МБ")

    task_id = str(uuid.uuid4())
    _set_progress(task_id, 0, "Queued", status="pending")
    bg.add_task(run_predict, task_id, audio_path=tmp_path, cleanup_dir=tmp_dir)
    logger.info("predict_file_queued", extra={"task_id": task_id, "file_name": filename})
    return {"task_id": task_id}


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    raw = redis_client.get(f"task:{task_id}")
    if raw is None:
        raise HTTPException(status_code=404, detail="task not found or expired")
    return {"task_id": task_id, **json.loads(raw)}


@app.post("/feedback/{task_id}")
def submit_feedback(task_id: str, req: FeedbackRequest):
    raw = redis_client.get(f"task:{task_id}")
    if raw is None:
        raise HTTPException(status_code=404, detail="task not found")
    task = json.loads(raw)
    predicted_class = (task.get("result") or {}).get("final_label", "unknown")

    FEEDBACK_TOTAL.labels(predicted_class=predicted_class, rating=req.rating).inc()
    redis_client.set(f"feedback:{task_id}", json.dumps({
        "task_id": task_id,
        "predicted_class": predicted_class,
        "rating": req.rating,
        "ts": time.time(),
    }))
    return {"ok": True, "task_id": task_id, "rating": req.rating}
