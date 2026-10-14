"""
Inference: split audio into overlapping chunks, classify each chunk with the
trained LightGBM pipeline, and combine them by soft voting (mean probability).
"""

import numpy as np
import pandas as pd

from ml_pipeline.feature_extract import extract_features

CHUNK_SEC = 6.0
HOP_SEC = 4.2          # 30% overlap — must match the training split
QUIET_THRESHOLD = 0.65


def split_audio_chunks(y, sr, max_sec=CHUNK_SEC, hop_sec=None, drop_quiet_first=False):
    """
    Returns [(chunk_y, t_start, t_end), ...]. Chunks shorter than 1 s are dropped.

    hop_sec=None → no overlap (hop = max_sec).
    drop_quiet_first → drop the first chunk if it is much quieter than the rest
    (used when building the training set, to skip silent intros).
    """
    if hop_sec is None:
        hop_sec = max_sec
    total_samples = len(y)
    chunk_samples = int(max_sec * sr)
    hop_samples = max(1, int(hop_sec * sr))

    chunks = []
    offset = 0
    while offset < total_samples:
        end = min(offset + chunk_samples, total_samples)
        chunk = y[offset:end]
        if len(chunk) >= sr:
            chunks.append((chunk, offset / sr, end / sr))
        if end >= total_samples:
            break
        offset += hop_samples

    if drop_quiet_first and len(chunks) >= 2:
        rms_vals = [float(np.sqrt(np.mean(c[0] ** 2))) for c in chunks]
        if rms_vals[0] / (np.mean(rms_vals[1:]) + 1e-8) < QUIET_THRESHOLD:
            return chunks[1:]

    return chunks


def classify(y, sr, beats, downbeats, bundle, progress_cb=None) -> dict:
    """
    Classify a whole track. `bundle` is the dict saved by train.py
    (pipeline, classes, feature_keys). progress_cb(done, total) is called after
    each chunk's features are extracted.

    Raises ValueError if the audio is shorter than one chunk (1 s).
    """
    pipeline = bundle["pipeline"]
    classes = bundle["classes"]
    feature_keys = bundle["feature_keys"]

    chunks = split_audio_chunks(y, sr, max_sec=CHUNK_SEC, hop_sec=HOP_SEC)
    if not chunks:
        raise ValueError("Аудіо занадто коротке — потрібно щонайменше 1 секунду")

    rows = []
    for i, (chunk_y, t_start, _) in enumerate(chunks):
        feat = extract_features(chunk_y, sr, beats=beats, downbeats=downbeats,
                                chunk_offset=t_start)
        rows.append([feat.get(k, 0.0) for k in feature_keys])
        if progress_cb:
            progress_cb(i + 1, len(chunks))

    probs = pipeline.predict_proba(pd.DataFrame(rows, columns=feature_keys))

    chunk_results = []
    for i, ((_, t_start, t_end), p) in enumerate(zip(chunks, probs)):
        idx = int(np.argmax(p))
        chunk_results.append({
            "chunk_idx": i + 1,
            "time_start": round(t_start, 1),
            "time_end": round(t_end, 1),
            "label": classes[idx],
            "confidence": round(float(p[idx]), 3),
            "probs": _round_probs(classes, p),
        })

    mean_probs = np.mean(probs, axis=0)
    final_idx = int(np.argmax(mean_probs))
    return {
        "final_label": classes[final_idx],
        "final_conf": round(float(mean_probs[final_idx]), 3),
        "mean_probs": _round_probs(classes, mean_probs),
        "chunks": chunk_results,
        "n_chunks": len(chunks),
    }


def _round_probs(classes, probs) -> dict:
    return {c: round(float(p), 3) for c, p in zip(classes, probs)}
