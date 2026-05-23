"""
GPR worker — FastAPI app for HuggingFace Spaces.
Combines what used to be `backend/` (FastAPI HTTP) and `ml_worker/` (Celery)
into a single process. BackgroundTasks run the ML pipeline; progress and
results are stored in Upstash Redis so the frontend can poll.
"""

import os
import json
import uuid
import pickle
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import aiofiles
import numpy as np
import pandas as pd
import redis as redis_lib
from fastapi import BackgroundTasks, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from simple_classify import (
    _download_audio_ytdlp,
    _file_beats,
    extract_features,
    load_audio,
    split_audio_chunks,
)

# ── Startup: connect Redis, load model ────────────────────────────────

BUNDLE = None
redis_client: Optional[redis_lib.Redis] = None


def _load_model() -> dict:
    """Local file if present, otherwise download from HF Hub."""
    path = os.environ.get("MODEL_PATH", "/app/model.pkl")
    if not os.path.exists(path):
        repo = os.environ.get("HF_MODEL_REPO")
        if not repo:
            raise RuntimeError(
                f"No model at {path} and HF_MODEL_REPO env var not set"
            )
        from huggingface_hub import hf_hub_download
        print(f"[startup] Fetching model from HF Hub: {repo}", flush=True)
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
    print(f"[startup] Loaded model with classes: {BUNDLE['classes']}", flush=True)
    yield


app = FastAPI(title="GPR Worker", lifespan=lifespan)

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
    try:
        _set_progress(task_id, 5, "Починаю обробку…")

        # ── 1. Download from URL ──
        if url:
            _set_progress(task_id, 10, "Завантажую аудіо з URL…")
            tmp_dir = tempfile.mkdtemp()
            cleanup_dir = tmp_dir
            tmp_wav = os.path.join(tmp_dir, "audio.wav")
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
                _set_progress(task_id, 25, "Аудіо завантажено")
            except Exception as e:
                _set_progress(task_id, 0, f"Помилка завантаження: {e}", status="error")
                return

        # ── 2. Demucs ──
        _set_progress(task_id, 30, "Виділяю гітарний стем (HTDemucs)…")
        try:
            y, sr = load_audio(audio_path, use_demucs=True)
        except Exception as e:
            _set_progress(task_id, 0, f"Demucs error: {e}", status="error")
            return

        # ── 3. BeatThis ──
        _set_progress(task_id, 55, "Аналізую ритм (BeatThis)…")
        beats, downbeats = _file_beats(audio_path, y, sr, use_beats=True)

        # ── 4. Chunks + features + classify ──
        _set_progress(task_id, 70, "Витягую ознаки…")
        pipeline = BUNDLE["pipeline"]
        classes = BUNDLE["classes"]
        feature_keys = BUNDLE["feature_keys"]

        chunks = split_audio_chunks(y, sr, max_sec=6.0,
                                    hop_sec=4.2, drop_quiet_first=False)
        chunk_results = []
        all_probs = []

        for ci, (chunk_y, t_start, t_end) in enumerate(chunks):
            feat = extract_features(chunk_y, sr, beats=beats,
                                    downbeats=downbeats, chunk_offset=t_start)
            # DataFrame keeps feature names → matches pipeline.fit(X=DataFrame).
            vec = pd.DataFrame(
                [[feat.get(k, 0.0) for k in feature_keys]],
                columns=feature_keys,
            )
            probs = pipeline.predict_proba(vec)[0]
            pred_idx = int(np.argmax(probs))
            chunk_results.append({
                "chunk_idx": ci + 1,
                "time_start": round(t_start, 1),
                "time_end": round(t_end, 1),
                "label": classes[pred_idx],
                "confidence": round(float(probs[pred_idx]), 3),
                "probs": {c: round(float(p), 3)
                          for c, p in zip(classes, probs)},
            })
            all_probs.append(probs)
            prog = 70 + int(25 * (ci + 1) / len(chunks))
            _set_progress(task_id, prog,
                          f"Chunk {ci + 1}/{len(chunks)} оброблено…")

        # ── 5. Soft voting ──
        mean_probs = np.mean(all_probs, axis=0)
        final_idx = int(np.argmax(mean_probs))
        result = {
            "final_label": classes[final_idx],
            "final_conf": round(float(mean_probs[final_idx]), 3),
            "mean_probs": {c: round(float(p), 3)
                           for c, p in zip(classes, mean_probs)},
            "chunks": chunk_results,
            "n_chunks": len(chunks),
        }
        _set_progress(task_id, 100, "Готово!", status="done", result=result)

    except Exception as e:
        _set_progress(task_id, 0, str(e), status="error")
    finally:
        if cleanup_dir and os.path.exists(cleanup_dir):
            shutil.rmtree(cleanup_dir, ignore_errors=True)


# ── HTTP API ──────────────────────────────────────────────────────────

class PredictUrlRequest(BaseModel):
    url: str
    start_sec: Optional[float] = None
    duration_sec: Optional[float] = None


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
    return {"task_id": task_id}


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    raw = redis_client.get(f"task:{task_id}")
    if raw is None:
        return {"task_id": task_id, "status": "pending",
                "progress": 0, "message": ""}
    return {"task_id": task_id, **json.loads(raw)}
