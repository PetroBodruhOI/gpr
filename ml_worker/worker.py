import os, json, time
import redis
from celery import Celery
import sys
sys.path.insert(0, ".")
from simple_classify import (
    load_audio, extract_features, split_audio_chunks,
    compute_beats, to_vec, FEATURE_KEYS,
    _download_audio_ytdlp, _file_beats,
)
import pickle, numpy as np, pandas as pd

REDIS_URL = os.environ["REDIS_URL"]
celery_app = Celery("guitar_worker", broker=REDIS_URL, backend=REDIS_URL)
r = redis.from_url(REDIS_URL)

MODEL_PATH = os.environ.get("MODEL_PATH", "./model.pkl")

# If no local file, try downloading from HuggingFace Hub.
# Set HF_MODEL_REPO (e.g. "your-user/gpr-model") and optionally
# HF_MODEL_FILE (default "model.pkl") + HF_TOKEN for private repos.
if not os.path.exists(MODEL_PATH):
    hf_repo = os.environ.get("HF_MODEL_REPO")
    if hf_repo:
        from huggingface_hub import hf_hub_download
        print(f"[startup] Downloading model from HF Hub: {hf_repo}", flush=True)
        MODEL_PATH = hf_hub_download(
            repo_id=hf_repo,
            filename=os.environ.get("HF_MODEL_FILE", "model.pkl"),
            token=os.environ.get("HF_TOKEN") or None,
        )
        print(f"[startup] Model cached at {MODEL_PATH}", flush=True)
    else:
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH} and HF_MODEL_REPO not set"
        )

with open(MODEL_PATH, "rb") as f:
    BUNDLE = pickle.load(f)


def _set_progress(task_id, progress, message="", status="processing", result=None):
    payload = {"status": status, "progress": progress, "message": message}
    if result:
        payload["result"] = result
    r.setex(f"task:{task_id}", 3600, json.dumps(payload))


@celery_app.task(name="run_predict")
def run_predict(task_id, audio_path=None, url=None,
                start_sec=None, duration_sec=None):
    try:
        _set_progress(task_id, 5, "Починаю обробку…")

        # ── 1. Отримати аудіо ──
        if url:
            _set_progress(task_id, 10, "Завантажую аудіо з URL…")
            import tempfile, shutil
            tmp_dir = tempfile.mkdtemp()
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
                print(f"DEBUG: Downloaded to {audio_path}, exists={os.path.exists(audio_path)}", flush=True)
                _set_progress(task_id, 25, "Аудіо завантажено")
            except Exception as e:
                print(f"DEBUG: yt-dlp error: {e}", flush=True)
                _set_progress(task_id, 0, f"Помилка завантаження: {e}",
                              status="error")
                return

        # ── 2. Demucs ──
        _set_progress(task_id, 30, "Виділяю гітарний стем (HTDemucs)…")
        try:
            y, sr = load_audio(audio_path, use_demucs=True)
            print(f"DEBUG: Demucs success, y.shape={y.shape}, sr={sr}")
        except Exception as e:
            print(f"DEBUG: Demucs error: {e}")
            _set_progress(task_id, 0, f"Demucs error: {e}",
                          status="error")
            return

        # ── 3. BeatThis ──
        _set_progress(task_id, 55, "Аналізую ритм (BeatThis)…")
        print("DEBUG: Starting BeatThis…", flush=True)
        beats, downbeats = _file_beats(audio_path, y, sr, use_beats=True)
        print(f"DEBUG: BeatThis done. beats={len(beats)}, downbeats={len(downbeats)}", flush=True)

        # ── 4. Chunks + фічі ──
        _set_progress(task_id, 65, "Розбиваю на чанки…")
        pipeline     = BUNDLE["pipeline"]
        classes      = BUNDLE["classes"]
        feature_keys = BUNDLE["feature_keys"]

        chunks = split_audio_chunks(y, sr, max_sec=6.0,
                                    hop_sec=4.2, drop_quiet_first=False)
        print(f"DEBUG: split_audio_chunks done, n_chunks={len(chunks)}", flush=True)

        _set_progress(task_id, 70, f"Обробляю {len(chunks)} чанків…")

        chunk_results = []
        all_probs     = []

        for ci, (chunk_y, t_start, t_end) in enumerate(chunks):
            print(f"DEBUG: chunk {ci+1}/{len(chunks)} - extract_features start", flush=True)
            feat = extract_features(chunk_y, sr, beats=beats,
                                    downbeats=downbeats, chunk_offset=t_start)
            print(f"DEBUG: chunk {ci+1} - extract_features done, {len(feat)} features", flush=True)

            vec = pd.DataFrame(
                [[feat.get(k, 0.0) for k in feature_keys]],
                columns=feature_keys,
            )
            print(f"DEBUG: chunk {ci+1} - predict_proba start", flush=True)
            probs    = pipeline.predict_proba(vec)[0]
            print(f"DEBUG: chunk {ci+1} - predict_proba done", flush=True)
            pred_idx = int(np.argmax(probs))

            chunk_results.append({
                "chunk_idx":  ci + 1,
                "time_start": round(t_start, 1),
                "time_end":   round(t_end, 1),
                "label":      classes[pred_idx],
                "confidence": round(float(probs[pred_idx]), 3),
                "probs":      {c: round(float(p), 3)
                               for c, p in zip(classes, probs)},
            })
            all_probs.append(probs)
            prog = 70 + int(25 * (ci + 1) / len(chunks))
            _set_progress(task_id, prog,
                          f"Chunk {ci+1}/{len(chunks)} оброблено…")

        # ── 5. Soft voting ──
        mean_probs  = np.mean(all_probs, axis=0)
        final_idx   = int(np.argmax(mean_probs))
        final_label = classes[final_idx]

        result = {
            "final_label":  final_label,
            "final_conf":   round(float(mean_probs[final_idx]), 3),
            "mean_probs":   {c: round(float(p), 3)
                             for c, p in zip(classes, mean_probs)},
            "chunks":       chunk_results,
            "n_chunks":     len(chunks),
        }
        _set_progress(task_id, 100, "Готово!", status="done", result=result)

    except Exception as e:
        _set_progress(task_id, 0, str(e), status="error")

    finally:
        # Видаляємо тимчасові файли
        if url and audio_path and os.path.exists(audio_path):
            import shutil
            shutil.rmtree(os.path.dirname(audio_path), ignore_errors=True)
