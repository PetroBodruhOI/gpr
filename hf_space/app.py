"""
GPR worker — FastAPI app for HuggingFace Spaces.
Combines what used to be `backend/` (FastAPI HTTP) and `ml_worker/` (Celery)
into a single process. BackgroundTasks run the ML pipeline; progress and
results are stored in Upstash Redis so the frontend can poll.

Monitoring:
  - `/metrics`   — Prometheus exposition (scrape from Grafana Cloud)
  - JSON logs   — stdout, parseable by Grafana Loki / any aggregator
  - `/feedback/{task_id}` — user rating endpoint (good / bad)
"""

import json
import logging
import os
import pickle
import shutil
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import aiofiles
import numpy as np
import pandas as pd
import redis as redis_lib
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel
from pythonjsonlogger import jsonlogger

from simple_classify import (
    _download_audio_ytdlp,
    _file_beats,
    extract_features,
    load_audio,
    split_audio_chunks,
)

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
    labelnames=["source", "predicted_class", "status"],
)

INFERENCE_DURATION = Histogram(
    "gpr_inference_duration_seconds",
    "End-to-end inference time (audio download → final classification)",
    labelnames=["source", "predicted_class"],
    buckets=(5, 10, 20, 30, 60, 120, 180, 300, 600),
)

STAGE_DURATION = Histogram(
    "gpr_stage_duration_seconds",
    "Time spent in each pipeline stage",
    labelnames=["stage"],          # download / demucs / beat / classify
    buckets=(1, 3, 5, 10, 20, 30, 60, 120, 300),
)

AUDIO_DURATION = Histogram(
    "gpr_audio_duration_seconds",
    "Length of input audio after Demucs",
    labelnames=["source"],
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

BUNDLE = None
redis_client: Optional[redis_lib.Redis] = None


def _load_model() -> dict:
    path = os.environ.get("MODEL_PATH", "/app/model.pkl")
    if not os.path.exists(path):
        repo = os.environ.get("HF_MODEL_REPO")
        if not repo:
            raise RuntimeError(
                f"No model at {path} and HF_MODEL_REPO env var not set"
            )
        from huggingface_hub import hf_hub_download
        logger.info("downloading_model", extra={"hf_repo": repo})
        path = hf_hub_download(
            repo_id=repo,
            filename=os.environ.get("HF_MODEL_FILE", "model.pkl"),
            token=os.environ.get("HF_TOKEN") or None,
        )
    with open(path, "rb") as f:
        return pickle.load(f)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global BUNDLE, redis_client
    redis_client = redis_lib.from_url(os.environ["REDIS_URL"])
    BUNDLE = _load_model()
    logger.info("startup_complete", extra={
        "classes": BUNDLE["classes"],
        "n_features": len(BUNDLE["feature_keys"]),
    })
    yield


app = FastAPI(title="GPR Worker", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())   # Prometheus exposition endpoint

_allowed = os.environ.get("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed.split(",") if o.strip()] or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────

def _set_progress(task_id: str, progress: int, message: str = "",
                  status: str = "processing", result=None) -> None:
    payload = {"status": status, "progress": progress, "message": message}
    if result:
        payload["result"] = result
    redis_client.setex(f"task:{task_id}", 3600, json.dumps(payload))


def run_predict(task_id: str, audio_path: Optional[str] = None,
                url: Optional[str] = None,
                start_sec: Optional[float] = None,
                duration_sec: Optional[float] = None,
                cleanup_dir: Optional[str] = None) -> None:
    """Synchronous pipeline — invoked by FastAPI BackgroundTasks."""
    source = "url" if url else "file"
    predicted_class = "unknown"
    audio_length_s = 0.0
    t_start_total = time.perf_counter()

    try:
        _set_progress(task_id, 5, "Починаю обробку…")

        # ── 1. Download from URL ──
        if url:
            _set_progress(task_id, 10, "Завантажую аудіо з URL…")
            tmp_dir = tempfile.mkdtemp()
            cleanup_dir = tmp_dir
            tmp_wav = os.path.join(tmp_dir, "audio.wav")
            t_dl = time.perf_counter()
            try:
                _download_audio_ytdlp(url, tmp_wav,
                                      start_sec=start_sec,
                                      duration_sec=duration_sec)
                audio_path = tmp_wav
                if not os.path.exists(audio_path):
                    candidates = [f for f in os.listdir(tmp_dir)
                                  if f.startswith("audio")]
                    if candidates:
                        audio_path = os.path.join(tmp_dir, candidates[0])
                STAGE_DURATION.labels(stage="download").observe(
                    time.perf_counter() - t_dl
                )
                _set_progress(task_id, 25, "Аудіо завантажено")
            except Exception as e:
                _set_progress(task_id, 0, f"Помилка завантаження: {e}",
                              status="error")
                PREDICTIONS_TOTAL.labels(source=source, predicted_class="none",
                                         status="download_error").inc()
                logger.error("download_failed", extra={
                    "task_id": task_id, "url": url, "error": str(e),
                })
                return

        # ── 2. Demucs ──
        _set_progress(task_id, 30, "Виділяю гітарний стем (HTDemucs)…")
        t_demucs = time.perf_counter()
        try:
            y, sr = load_audio(audio_path, use_demucs=True)
        except Exception as e:
            _set_progress(task_id, 0, f"Demucs error: {e}", status="error")
            PREDICTIONS_TOTAL.labels(source=source, predicted_class="none",
                                     status="demucs_error").inc()
            logger.error("demucs_failed", extra={
                "task_id": task_id, "error": str(e),
            })
            return
        STAGE_DURATION.labels(stage="demucs").observe(
            time.perf_counter() - t_demucs
        )
        audio_length_s = len(y) / sr if sr else 0.0
        AUDIO_DURATION.labels(source=source).observe(audio_length_s)

        # ── 3. BeatThis ──
        _set_progress(task_id, 55, "Аналізую ритм (BeatThis)…")
        t_beat = time.perf_counter()
        beats, downbeats = _file_beats(audio_path, y, sr, use_beats=True)
        STAGE_DURATION.labels(stage="beat").observe(
            time.perf_counter() - t_beat
        )

        # ── 4. Chunks + features + classify ──
        _set_progress(task_id, 70, "Витягую ознаки…")
        t_classify = time.perf_counter()
        pipeline = BUNDLE["pipeline"]
        classes = BUNDLE["classes"]
        feature_keys = BUNDLE["feature_keys"]

        chunks = split_audio_chunks(y, sr, max_sec=6.0,
                                    hop_sec=4.2, drop_quiet_first=False)
        chunk_results = []
        all_probs = []

        for ci, (chunk_y, t_s, t_e) in enumerate(chunks):
            feat = extract_features(chunk_y, sr, beats=beats,
                                    downbeats=downbeats, chunk_offset=t_s)
            vec = pd.DataFrame(
                [[feat.get(k, 0.0) for k in feature_keys]],
                columns=feature_keys,
            )
            probs = pipeline.predict_proba(vec)[0]
            pred_idx = int(np.argmax(probs))
            chunk_results.append({
                "chunk_idx": ci + 1,
                "time_start": round(t_s, 1),
                "time_end": round(t_e, 1),
                "label": classes[pred_idx],
                "confidence": round(float(probs[pred_idx]), 3),
                "probs": {c: round(float(p), 3)
                          for c, p in zip(classes, probs)},
            })
            all_probs.append(probs)
            prog = 70 + int(25 * (ci + 1) / len(chunks))
            _set_progress(task_id, prog,
                          f"Chunk {ci + 1}/{len(chunks)} оброблено…")
        STAGE_DURATION.labels(stage="classify").observe(
            time.perf_counter() - t_classify
        )

        # ── 5. Soft voting ──
        mean_probs = np.mean(all_probs, axis=0)
        final_idx = int(np.argmax(mean_probs))
        predicted_class = classes[final_idx]
        confidence = float(mean_probs[final_idx])

        result = {
            "final_label": predicted_class,
            "final_conf": round(confidence, 3),
            "mean_probs": {c: round(float(p), 3)
                           for c, p in zip(classes, mean_probs)},
            "chunks": chunk_results,
            "n_chunks": len(chunks),
        }
        _set_progress(task_id, 100, "Готово!", status="done", result=result)

        # ── Final metrics ──
        total_duration = time.perf_counter() - t_start_total
        INFERENCE_DURATION.labels(source=source,
                                  predicted_class=predicted_class
                                  ).observe(total_duration)
        CONFIDENCE.labels(predicted_class=predicted_class).observe(confidence)
        PREDICTIONS_TOTAL.labels(source=source,
                                 predicted_class=predicted_class,
                                 status="done").inc()
        logger.info("prediction_complete", extra={
            "task_id": task_id,
            "source": source,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 3),
            "audio_duration_s": round(audio_length_s, 1),
            "inference_duration_s": round(total_duration, 1),
            "n_chunks": len(chunks),
        })

    except Exception as e:
        _set_progress(task_id, 0, str(e), status="error")
        PREDICTIONS_TOTAL.labels(source=source, predicted_class="none",
                                 status="error").inc()
        logger.error("prediction_failed", extra={
            "task_id": task_id, "source": source, "error": str(e),
        })
    finally:
        if cleanup_dir and os.path.exists(cleanup_dir):
            shutil.rmtree(cleanup_dir, ignore_errors=True)


# ── HTTP API ──────────────────────────────────────────────────────────

class PredictUrlRequest(BaseModel):
    url: str
    start_sec: Optional[float] = None
    duration_sec: Optional[float] = None


class FeedbackRequest(BaseModel):
    rating: str   # "good" or "bad"


@app.get("/")
def health():
    return {
        "status": "ok",
        "classes": BUNDLE["classes"] if BUNDLE else None,
        "n_features": len(BUNDLE["feature_keys"]) if BUNDLE else None,
    }


@app.post("/predict/url")
def predict_url(req: PredictUrlRequest, bg: BackgroundTasks):
    task_id = str(uuid.uuid4())
    _set_progress(task_id, 0, "Queued", status="pending")
    bg.add_task(run_predict, task_id, url=req.url,
                start_sec=req.start_sec, duration_sec=req.duration_sec)
    logger.info("predict_url_queued",
                extra={"task_id": task_id, "url": req.url})
    return {"task_id": task_id}


@app.post("/predict/file")
async def predict_file(file: UploadFile, bg: BackgroundTasks):
    tmp_dir = tempfile.mkdtemp()
    filename = file.filename or "audio.wav"
    tmp_path = os.path.join(tmp_dir, filename)
    async with aiofiles.open(tmp_path, "wb") as f:
        await f.write(await file.read())

    task_id = str(uuid.uuid4())
    _set_progress(task_id, 0, "Queued", status="pending")
    bg.add_task(run_predict, task_id, audio_path=tmp_path, cleanup_dir=tmp_dir)
    logger.info("predict_file_queued",
                extra={"task_id": task_id, "filename": filename})
    return {"task_id": task_id}


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    raw = redis_client.get(f"task:{task_id}")
    if raw is None:
        return {"task_id": task_id, "status": "pending",
                "progress": 0, "message": ""}
    return {"task_id": task_id, **json.loads(raw)}


@app.post("/feedback/{task_id}")
def submit_feedback(task_id: str, req: FeedbackRequest):
    if req.rating not in ("good", "bad"):
        raise HTTPException(status_code=400,
                            detail="rating must be 'good' or 'bad'")

    raw = redis_client.get(f"task:{task_id}")
    if raw is None:
        raise HTTPException(status_code=404, detail="task not found")
    task = json.loads(raw)
    predicted_class = (task.get("result") or {}).get("final_label", "unknown")

    FEEDBACK_TOTAL.labels(predicted_class=predicted_class,
                          rating=req.rating).inc()
    # Persist for later analysis (30-day TTL).
    redis_client.setex(f"feedback:{task_id}", 86400 * 30, json.dumps({
        "task_id": task_id,
        "predicted_class": predicted_class,
        "rating": req.rating,
        "ts": time.time(),
    }))
    logger.info("user_feedback", extra={
        "task_id": task_id,
        "predicted_class": predicted_class,
        "rating": req.rating,
    })
    return {"ok": True, "task_id": task_id, "rating": req.rating}
