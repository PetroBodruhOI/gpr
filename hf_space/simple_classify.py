"""
simple_classify.py — пайплайн класифікації гітарних патернів:
  1) HTDemucs (htdemucs_6s) → гітарний стем
  2) BeatThis → beats + downbeats (ритмічний grid)
  3) librosa → ~25 числових ознак (RMS, спектр, MFCC, onset, ZCR, …)
  4) Beat-quantized strum positions:
     librosa onsets квантуються на 8-step grid відносно downbeats;
     для кожного onset рахується STFT-енергія LOW/HIGH → b_low[8], b_high[8].
  5) LightGBM (або будь-яка sklearn-сумісна модель) — навчання на CSV-фічах,
     inference з виводом ймовірностей по кожному chunk.

Команди:
  diagnose  — статистика розділення класів (gap, cosine matrix, Fisher ratio)
  build     — будує pickle-БД для kNN-класифікатора (legacy)
  classify  — класифікує файл через kNN на готовій БД (legacy)
  extract   — витягує фічі для всіх файлів і зберігає у CSV
  train     — навчає LightGBM на CSV, зберігає model.pkl
  predict   — класифікує аудіо через збережену модель;
              виводить результат по кожному chunk + фінальний вердикт

Використання:
  python simple_classify.py extract  --audio_dir ./audio --output ./features.csv \\
                                     --use_demucs --use_beats --max_sec 6 --hop_ratio 0.7

  python simple_classify.py train    --csv ./features.csv --output ./model.pkl

  python simple_classify.py predict  --input ./test.wav --model ./model.pkl \\
                                     --use_demucs --use_beats
"""

import os
import csv
import pickle
import argparse
import hashlib
import numpy as np
import librosa
from collections import Counter
from itertools import combinations


# ─── HTDemucs guitar separation ──────────────────────────────────────────────

DEMUCS_CACHE_DIR = "./_demucs_cache"
_DEMUCS_SEPARATOR = None


def separate_guitar(path: str, target_sr: int = 22050):
    """
    CLI-based Demucs separation (compatible with demucs 4.0.1 from PyPI,
    where demucs.api doesn't yet exist). Calls `python -m demucs -n htdemucs_6s`

    Note: Demucs naturally outputs at 16kHz (for compatibility with trained models)
    and reads the resulting guitar.wav stem from disk.
    """
    import soundfile as sf
    import subprocess, tempfile, shutil

    os.makedirs(DEMUCS_CACHE_DIR, exist_ok=True)
    key = hashlib.md5(
        f"{os.path.abspath(path)}|{os.path.getmtime(path)}".encode()
    ).hexdigest()
    cache_path = os.path.join(DEMUCS_CACHE_DIR, f"{key}_guitar.wav")

    if os.path.exists(cache_path):
        return librosa.load(cache_path, sr=target_sr, mono=True)

    out_dir = tempfile.mkdtemp(prefix="demucs_")
    try:
        print(f"[demucs] htdemucs_6s → {os.path.basename(path)}", flush=True)
        cmd = [
            "python", "-m", "demucs",
            "-n", "htdemucs_6s",
            "--shifts", "0",      # disable test-time aug → ~2x faster on CPU
            "--overlap", "0.1",   # smaller window overlap → ~30% faster
            "-o", out_dir,
            path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"demucs failed (rc={result.returncode}):\n{result.stderr}"
            )

        track_stem = os.path.splitext(os.path.basename(path))[0]
        guitar_path = os.path.join(out_dir, "htdemucs_6s", track_stem, "guitar.wav")
        if not os.path.exists(guitar_path):
            import glob
            found = glob.glob(os.path.join(out_dir, "**", "*.wav"), recursive=True)
            raise FileNotFoundError(
                f"Guitar stem not found at {guitar_path}. "
                f"Available files: {found}"
            )

        guitar, _ = librosa.load(guitar_path, sr=target_sr, mono=True)
        guitar = guitar.astype(np.float32)
        sf.write(cache_path, guitar, target_sr)
        return guitar, target_sr
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def load_audio(path: str, use_demucs: bool, sr: int = 22050):
    """Load audio at target sample rate (22050Hz for trained model compatibility).

    ВАЖЛИВО: НЕ застосовуй тут жодних audio-перетворень
    (preemphasis, RMS-нормалізацію, фільтри тощо).
    Модель навчалась на сирому librosa-аудіо, тому будь-яке перетворення
    зміщує розподіл фіч і ламає predict.
    """
    if use_demucs:
        return separate_guitar(path, target_sr=sr)
    return librosa.load(path, sr=sr, mono=True)


# ─── BeatThis beat tracker ───────────────────────────────────────────────────

BEATS_CACHE_DIR = "./_beats_cache"
_BEAT_TRACKER = None


def _get_beat_tracker():
    global _BEAT_TRACKER
    if _BEAT_TRACKER is None:
        try:
            from beat_this.inference import Audio2Beats
        except ImportError as e:
            raise ImportError(
                "beat-this не встановлено. Запусти: pip install beat-this"
            ) from e
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[beat-this] Завантажую модель (device={device}) …", flush=True)
        _BEAT_TRACKER = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)
    return _BEAT_TRACKER


def compute_beats(y: np.ndarray, sr: int, cache_key: str = None):
    """
    Returns (beats, downbeats) — np.float32 arrays of times in seconds.
    cache_key: persistent disk-cache identifier (e.g. abs_path|mtime).
    """
    cache_path = None
    if cache_key:
        os.makedirs(BEATS_CACHE_DIR, exist_ok=True)
        key = hashlib.md5(cache_key.encode()).hexdigest()
        cache_path = os.path.join(BEATS_CACHE_DIR, f"{key}.npz")
        if os.path.exists(cache_path):
            d = np.load(cache_path)
            return d["beats"], d["downbeats"]

    tracker = _get_beat_tracker()
    beats, downbeats = tracker(y, sr)
    beats = np.asarray(beats, dtype=np.float32)
    downbeats = np.asarray(downbeats, dtype=np.float32)

    if cache_path:
        np.savez(cache_path, beats=beats, downbeats=downbeats)
    return beats, downbeats


# ─── Feature extraction ──────────────────────────────────────────────────────

LIBROSA_KEYS = [
    # Energy / dynamics
    "rms_mean", "rms_std", "dyn_range", "crest_factor",
    # Spectral
    "centroid_mean", "centroid_std",
    "bandwidth_mean", "bandwidth_std",
    "rolloff_mean",
    "flux_mean", "flux_std",
    # Timing (librosa onsets)
    "onset_density", "onset_strength_mean", "ioi_mean", "ioi_cv",
    # Texture
    "zcr_mean", "zcr_std",
    # MFCC (8 коефіцієнтів — середні)
    "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4",
    "mfcc_5", "mfcc_6", "mfcc_7", "mfcc_8",
]

BEAT_GRID_STEPS = 8
BEAT_KEYS = (
    [f"b_low_{i}"  for i in range(BEAT_GRID_STEPS)] +
    [f"b_high_{i}" for i in range(BEAT_GRID_STEPS)] +
    [
        "bpm",
        "downbeats_per_sec",
        "mean_bar_length",
        "grid_fit_error",       # 0 = ідеальний 4/4, ~0.5 = off-grid (waltz, swing…)
        "onset_beat_alignment", # частка onsets що падають у ±50мс від BeatThis-біту
    ]
)

FEATURE_KEYS = LIBROSA_KEYS + BEAT_KEYS  # 25 + 21 = 46


def extract_librosa_features(y: np.ndarray, sr: int = 22050) -> dict:
    hop = 256

    # Energy
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_mean = float(rms.mean())
    rms_std  = float(rms.std())
    dyn_range = float(np.percentile(rms, 90) - np.percentile(rms, 10))
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    crest_factor = peak / (rms_mean + 1e-8)

    # Spectral
    stft_mag = np.abs(librosa.stft(y, hop_length=hop))
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0]
    flux = np.sqrt(np.sum(np.diff(stft_mag, axis=1) ** 2, axis=0))

    # Timing
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_times = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop,
        units="time", delta=0.07
    )
    duration = len(y) / sr if sr > 0 else 0.0
    onset_density = len(onset_times) / duration if duration > 0 else 0.0
    if len(onset_times) >= 2:
        iois = np.diff(onset_times)
        ioi_mean = float(iois.mean())
        ioi_cv   = float(iois.std() / (iois.mean() + 1e-8))
    else:
        ioi_mean = float(duration)
        ioi_cv   = 0.0

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]

    # MFCC (8 coeffs, mean across time)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8, hop_length=hop)
    mfcc_means = mfcc.mean(axis=1)  # (8,)

    feats = {
        "rms_mean":            rms_mean,
        "rms_std":             rms_std,
        "dyn_range":           dyn_range,
        "crest_factor":        float(crest_factor),
        "centroid_mean":       float(centroid.mean()),
        "centroid_std":        float(centroid.std()),
        "bandwidth_mean":      float(bandwidth.mean()),
        "bandwidth_std":       float(bandwidth.std()),
        "rolloff_mean":        float(rolloff.mean()),
        "flux_mean":           float(flux.mean()),
        "flux_std":            float(flux.std()),
        "onset_density":       float(onset_density),
        "onset_strength_mean": float(onset_env.mean()),
        "ioi_mean":            ioi_mean,
        "ioi_cv":              ioi_cv,
        "zcr_mean":            float(zcr.mean()),
        "zcr_std":             float(zcr.std()),
    }
    for i, v in enumerate(mfcc_means, start=1):
        feats[f"mfcc_{i}"] = float(v)
    return feats


def _default_beat_feats():
    f = {k: 0.0 for k in BEAT_KEYS}
    f["grid_fit_error"] = 0.5  # нейтральне середнє замість 0 (бо 0 = ідеальний)
    return f


def extract_beat_features(y: np.ndarray, sr: int,
                          beats_in_chunk: np.ndarray,
                          downbeats_in_chunk: np.ndarray,
                          low_high_cutoff_hz: float = 500.0,
                          beat_match_window: float = 0.05) -> dict:
    """
    Усе у локальному часі chunk (секунди від 0).
    Квантизує librosa-onsets на 8-step grid відносно downbeats.
    Для кожного onset рахує LOW/HIGH STFT-енергію → b_low[8], b_high[8].
    """
    feats = _default_beat_feats()
    n = BEAT_GRID_STEPS
    duration = len(y) / sr if sr > 0 else 0.0
    if duration <= 0:
        return feats

    feats["downbeats_per_sec"] = len(downbeats_in_chunk) / duration

    # --- bar grid (anchor + bar_length) ---
    if len(downbeats_in_chunk) >= 2:
        bar_length = float(np.median(np.diff(downbeats_in_chunk)))
        anchor = float(downbeats_in_chunk[0])
    elif len(beats_in_chunk) >= 2:
        beat_period = float(np.median(np.diff(beats_in_chunk)))
        bar_length = 4 * beat_period
        anchor = float(downbeats_in_chunk[0]) if len(downbeats_in_chunk) else float(beats_in_chunk[0])
    else:
        return feats  # not enough info

    # Sanity: bar in 1.0–5.0 sec, інакше fallback
    if not (1.0 <= bar_length <= 5.0):
        if len(beats_in_chunk) >= 2:
            beat_period = float(np.median(np.diff(beats_in_chunk)))
            bar_length = 4 * beat_period
        if not (1.0 <= bar_length <= 5.0):
            return feats

    feats["mean_bar_length"] = bar_length
    if len(beats_in_chunk) >= 2:
        beat_period = float(np.median(np.diff(beats_in_chunk)))
        feats["bpm"] = 60.0 / beat_period if beat_period > 0 else 0.0
    step_period = bar_length / n

    # --- librosa onsets (це і є "strum positions") ---
    hop = 256
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_times = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop,
        units="time", delta=0.07
    )
    if len(onset_times) == 0:
        return feats

    # --- STFT для LOW/HIGH ---
    n_fft = 1024
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_mask  = freqs <= low_high_cutoff_hz
    high_mask = (freqs > low_high_cutoff_hz) & (freqs <= 5000)
    frame_times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=hop)

    fit_errors = []
    on_beat = 0
    for t_onset in onset_times:
        # квантизація: позиція в grid (0..n-1)
        rel = (t_onset - anchor) / step_period
        pos = int(round(rel)) % n
        fit_errors.append(abs(rel - round(rel)))

        # вирівнювання з BeatThis-бітами
        if len(beats_in_chunk) > 0:
            dt = np.min(np.abs(beats_in_chunk - t_onset))
            if dt <= beat_match_window:
                on_beat += 1

        # STFT-енергія в ±50мс вікні
        i0, i1 = np.searchsorted(frame_times, [t_onset - 0.05, t_onset + 0.05])
        i1 = min(i1, stft.shape[1])
        if i0 >= i1:
            continue
        win = stft[:, i0:i1]
        e_low  = float(win[low_mask].sum())
        e_high = float(win[high_mask].sum())
        if e_low >= e_high:
            feats[f"b_low_{pos}"]  += 1
        else:
            feats[f"b_high_{pos}"] += 1

    n_total = max(len(onset_times), 1)
    for k in list(feats.keys()):
        if k.startswith("b_low_") or k.startswith("b_high_"):
            feats[k] /= n_total

    feats["grid_fit_error"]      = float(np.mean(fit_errors)) if fit_errors else 0.5
    feats["onset_beat_alignment"] = on_beat / n_total
    return feats


def extract_features(y, sr, beats=None, downbeats=None, chunk_offset=0.0):
    """
    Об'єднаний extractor.
    beats/downbeats — абсолютні часи у вихідному файлі;
    chunk_offset — позиція початку chunk у файлі (секунди).
    """
    feats = extract_librosa_features(y, sr)
    if beats is not None and downbeats is not None:
        duration = len(y) / sr if sr > 0 else 0.0
        b_mask = (beats >= chunk_offset) & (beats < chunk_offset + duration)
        d_mask = (downbeats >= chunk_offset) & (downbeats < chunk_offset + duration)
        beats_in     = beats[b_mask]     - chunk_offset
        downbeats_in = downbeats[d_mask] - chunk_offset
        feats.update(extract_beat_features(y, sr, beats_in, downbeats_in))
    else:
        feats.update(_default_beat_feats())
    return feats


def to_vec(feat_dict: dict) -> np.ndarray:
    return np.array([feat_dict[k] for k in FEATURE_KEYS], dtype=np.float32)


# ─── Cosine similarity у нормалізованому просторі (без WEIGHTS) ──────────────

def _normalized(v: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return v / (scale + 1e-8)


def cosine_distance(a: np.ndarray, b: np.ndarray, scale: np.ndarray) -> float:
    a_n = _normalized(a, scale)
    b_n = _normalized(b, scale)
    sim = float(np.dot(a_n, b_n) /
                ((np.linalg.norm(a_n) + 1e-8) * (np.linalg.norm(b_n) + 1e-8)))
    return 1.0 - sim


# ─── Chunk helpers ───────────────────────────────────────────────────────────

QUIET_THRESHOLD = 0.65


def split_audio_chunks(y, sr, max_sec=5.0, hop_sec=None, drop_quiet_first=False):
    """
    hop_sec=None → крок=max_sec (без overlap)
    hop_sec=max_sec/2 → 50% overlap
    """
    if hop_sec is None:
        hop_sec = max_sec
    total_samples = len(y)
    chunk_samples = int(max_sec * sr)
    hop_samples   = max(1, int(hop_sec * sr))

    raw_chunks = []
    offset = 0
    while offset < total_samples:
        end   = min(offset + chunk_samples, total_samples)
        chunk = y[offset:end]
        if len(chunk) >= sr:
            raw_chunks.append((chunk, offset / sr, end / sr))
        if end >= total_samples:
            break
        offset += hop_samples

    if not raw_chunks:
        return raw_chunks

    if drop_quiet_first and len(raw_chunks) >= 2:
        rms_vals   = [float(np.sqrt(np.mean(c[0] ** 2))) for c in raw_chunks]
        rms_others = np.mean(rms_vals[1:])
        if rms_vals[0] / (rms_others + 1e-8) < QUIET_THRESHOLD:
            return raw_chunks[1:]

    return raw_chunks


def classify_vec(vec, records, scale, top_k):
    scored = sorted(
        [{"label": r["label"], "filename": r["filename"],
          "dist": cosine_distance(vec, r["vec"], scale)} for r in records],
        key=lambda x: x["dist"]
    )
    top    = scored[:top_k]
    votes  = Counter(item["label"] for item in top)
    winner = votes.most_common(1)[0][0]

    cs = {}
    for item in scored:
        cs.setdefault(item["label"], []).append(item["dist"])
    avg_dist = {lbl: float(np.mean(dists)) for lbl, dists in cs.items()}
    return winner, avg_dist, top


# ─── Commands ────────────────────────────────────────────────────────────────

SUPPORTED = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

NOISE_CHOICES = ["0", "1", "2", "3", "m"]


def get_files(d):
    return sorted([f for f in os.listdir(d)
                   if os.path.splitext(f)[1].lower() in SUPPORTED])


def filter_files_by_noise(files, noise):
    """
    Фільтр training-файлів за рівнем «забруднення» в назві (case-insensitive,
    розширення не враховується).
    """
    if noise is None:
        return files

    def stem(fname):
        return os.path.splitext(fname)[0].lower()

    if noise == "m":
        return [fn for fn in files
                if ("f" not in stem(fn)) or ("mf" in stem(fn))]
    n = int(noise)
    return [fn for fn in files if stem(fn).count("f") <= n]


def _file_beats(fpath, y, sr, use_beats):
    if not use_beats:
        return None, None
    cache_key = f"{os.path.abspath(fpath)}|{os.path.getmtime(fpath)}|sr{sr}"
    return compute_beats(y, sr, cache_key=cache_key)


# ─── NEW: extract command — фічі у CSV ───────────────────────────────────────

def cmd_extract(audio_dir, output_csv, max_sec, hop_ratio, use_demucs, use_beats,
                noise=None):
    """
    Витягає фічі з усіх файлів у audio_dir і зберігає у CSV.
    Кожен рядок = один chunk. Колонки: метадані + FEATURE_KEYS.
    Це сирий датасет — використовуй у ноутбуці для навчання моделей.
    """
    all_files = get_files(audio_dir)
    files     = filter_files_by_noise(all_files, noise)
    hop_sec   = max_sec * hop_ratio
    noise_tag = f"noise<={noise}" if noise is not None else "noise: OFF"
    print(f"[+] Файлів: {len(files)}/{len(all_files)}  |  max chunk: {max_sec}s  "
          f"|  hop: {hop_sec:.2f}s ({int(hop_ratio*100)}%)  "
          f"|  {noise_tag}  "
          f"|  HTDemucs: {'ON' if use_demucs else 'OFF'}  "
          f"|  BeatThis: {'ON' if use_beats else 'OFF'}\n")

    rows = []
    errors = []
    total_chunks = 0

    for file_idx, fname in enumerate(files):
        label = fname[:2].lower()
        fpath = os.path.join(audio_dir, fname)
        try:
            y, sr = load_audio(fpath, use_demucs=use_demucs)
            duration = len(y) / sr
            beats, downbeats = _file_beats(fpath, y, sr, use_beats)

            if duration <= max_sec:
                chunks = [(y, 0.0, duration)]
            else:
                chunks = split_audio_chunks(
                    y, sr, max_sec=max_sec,
                    hop_sec=hop_sec, drop_quiet_first=True
                )

            print(f"  → {fname:<35} ({label})  {duration:.1f}s → {len(chunks)} chunks")
            for ci, (chunk_y, t_start, t_end) in enumerate(chunks):
                try:
                    feat = extract_features(
                        chunk_y, sr,
                        beats=beats, downbeats=downbeats,
                        chunk_offset=t_start,
                    )
                    row = {
                        "file_idx":    file_idx,
                        "filename":    fname,
                        "label":       label,
                        "chunk_idx":   ci,
                        "chunk_start": round(t_start, 3),
                        "chunk_end":   round(t_end, 3),
                    }
                    for k in FEATURE_KEYS:
                        row[k] = feat[k]
                    rows.append(row)
                    total_chunks += 1
                except Exception as e:
                    print(f"     chunk {ci+1}: ✗ {e}")
        except Exception as e:
            errors.append((fname, str(e)))
            print(f"  → {fname:<35} ✗ {e}")

    if not rows:
        print("[!] Немає даних")
        return

    meta_cols = ["file_idx", "filename", "label",
                 "chunk_idx", "chunk_start", "chunk_end"]
    fieldnames = meta_cols + list(FEATURE_KEYS)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    labels = sorted(set(r["label"] for r in rows))
    label_counts = Counter(r["label"] for r in rows)
    print(f"\n{'═'*55}")
    print(f"  ✅ {output_csv}")
    print(f"  📦 Рядків (chunks)  : {len(rows)}")
    print(f"  📁 Файлів           : {len(files) - len(errors)}/{len(files)}")
    print(f"  🗂  Класи            : {labels}")
    print(f"  📊 По класах        : "
          + ", ".join(f"{l}={c}" for l, c in sorted(label_counts.items())))
    print(f"  🎯 Фічей            : {len(FEATURE_KEYS)}  "
          f"(librosa: {len(LIBROSA_KEYS)}, beats: {len(BEAT_KEYS)})")
    print(f"  🎸 HTDemucs         : {'ON' if use_demucs else 'OFF'}")
    print(f"  🥁 BeatThis         : {'ON' if use_beats else 'OFF'}")
    print(f"  ⏱  Chunk            : {max_sec}s, hop {hop_sec:.2f}s")
    print(f"{'═'*55}")
    if errors:
        print(f"\n⚠️  Файли з помилками ({len(errors)}):")
        for fname, msg in errors:
            print(f"     {fname}: {msg}")



# ─── NEW: train — навчання LightGBM на CSV ───────────────────────────────────

def cmd_train(csv_path, output_pkl, n_estimators=500, learning_rate=0.05,
              num_leaves=63, cv_folds=5):
    """
    Читає features.csv (з cmd_extract), тренує LightGBM через GroupKFold
    (групи = file_idx, без leakage між chunks одного файла), зберігає model.pkl.

    model.pkl містить:
      pipeline      — sklearn Pipeline (StandardScaler + LGBMClassifier)
      classes       — впорядкований список рядкових міток
      feature_keys  — список фіч (порядок колонок)
      cv_scores     — масив accuracy по фолдах
      config        — параметри навчання
    """
    try:
        import pandas as pd
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import GroupKFold, cross_val_score
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        from lightgbm import LGBMClassifier
    except ImportError as e:
        raise ImportError(
            "Встанови залежності: pip install lightgbm scikit-learn pandas"
        ) from e

    print(f"[train] Читаю {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"  Рядків: {len(df)}  |  Колонок: {df.shape[1]}")

    META_COLS = ["file_idx", "filename", "label", "chunk_idx",
                 "chunk_start", "chunk_end"]
    feature_keys = [c for c in df.columns if c not in META_COLS]

    X = df[feature_keys].values.astype(np.float32)
    labels_raw = df["label"].values
    file_ids   = df["file_idx"].values

    le = LabelEncoder()
    y  = le.fit_transform(labels_raw)
    classes = list(le.classes_)

    label_counts = Counter(labels_raw)
    print(f"  Класи: {classes}")
    print(f"  По класах: " + ", ".join(f"{c}={label_counts[c]}" for c in classes))
    print(f"  Фічей: {len(feature_keys)}\n")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )),
    ])

    # ── GroupKFold — chunks одного файла НЕ розділяються між train і val ──
    print(f"[train] GroupKFold CV (k={cv_folds}, groups=file_idx) …")
    gkf = GroupKFold(n_splits=cv_folds)
    cv_acc  = []
    cv_f1   = []
    fold_reports = []

    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=file_ids)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        file_ids_va = file_ids[va_idx]

        pipeline.fit(X_tr, y_tr)

        # ── file-level prediction: голосування chunks ──
        file_true, file_pred = [], []
        for fid in np.unique(file_ids_va):
            mask = file_ids_va == fid
            chunk_preds = pipeline.predict(X_va[mask])
            vote = Counter(chunk_preds).most_common(1)[0][0]
            file_true.append(y_va[mask][0])
            file_pred.append(vote)

        acc = accuracy_score(file_true, file_pred)
        f1  = f1_score(file_true, file_pred, average="macro", zero_division=0)
        cv_acc.append(acc)
        cv_f1.append(f1)
        fold_reports.append((file_true, file_pred))
        print(f"  fold {fold_i+1}/{cv_folds}  file-acc={acc:.4f}  F1m={f1:.4f}")

    cv_acc  = np.array(cv_acc)
    cv_f1   = np.array(cv_f1)
    print(f"\n  CV file-accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"  CV F1 macro      : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # ── Фінальне навчання на ВСІХ даних ──
    print(f"\n[train] Фінальне навчання на всіх {len(X)} chunks …")
    pipeline.fit(X, y)
    print(f"  ✅ Готово")

    bundle = {
        "pipeline":     pipeline,
        "classes":      classes,
        "label_encoder": le,
        "feature_keys": feature_keys,
        "cv_acc":       cv_acc,
        "cv_f1":        cv_f1,
        "config": {
            "n_estimators":  n_estimators,
            "learning_rate": learning_rate,
            "num_leaves":    num_leaves,
            "cv_folds":      cv_folds,
            "csv_path":      csv_path,
        },
    }

    with open(output_pkl, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n{'═'*55}")
    print(f"  ✅ {output_pkl}")
    print(f"  📦 Розмір: {os.path.getsize(output_pkl) // 1024} KB")
    print(f"  🗂  Класи : {classes}")
    print(f"  📊 CV acc : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"  📊 CV F1m : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"{'═'*55}")


# ─── NEW: predict — inference через збережену модель ─────────────────────────

def cmd_predict(input_path, model_pkl, max_sec, hop_ratio,
                use_demucs_override=None, use_beats_override=None):
    """
    Класифікує аудіо через збережену LightGBM-модель.
    Для кожного chunk виводить:
      - передбачений клас
      - ймовірності по всіх класах (рядок з N значень)
    Фінальний вердикт = середнє ймовірностей по всіх chunks (soft voting).
    """
    try:
        import pandas as pd
    except ImportError:
        pass  # pandas потрібен тільки для виводу таблиці, не критично

    print(f"[predict] Завантажую модель з {model_pkl} …")
    with open(model_pkl, "rb") as f:
        bundle = pickle.load(f)

    pipeline     = bundle["pipeline"]
    classes      = bundle["classes"]
    feature_keys = bundle["feature_keys"]
    cfg          = bundle.get("config", {})

    # Параметри аудіо — з model.pkl якщо не перевизначені аргументами
    use_demucs = (use_demucs_override if use_demucs_override is not None
                  else cfg.get("use_demucs", False))
    use_beats  = (use_beats_override  if use_beats_override  is not None
                  else cfg.get("use_beats",  False))

    n_classes = len(classes)
    print(f"  Класи ({n_classes}): {classes}")
    print(f"  Фічей: {len(feature_keys)}")
    print(f"  HTDemucs: {'ON' if use_demucs else 'OFF'}  "
          f"BeatThis: {'ON' if use_beats else 'OFF'}\n")

    # ── Завантаження і нарізка аудіо ──
    print(f"[predict] Обробляю {input_path} …")
    y_audio, sr = load_audio(input_path, use_demucs=use_demucs)
    duration = len(y_audio) / sr
    hop_sec  = max_sec * hop_ratio
    beats, downbeats = _file_beats(input_path, y_audio, sr, use_beats)

    chunks = split_audio_chunks(y_audio, sr, max_sec=max_sec,
                                hop_sec=hop_sec, drop_quiet_first=False)
    n_chunks = len(chunks)
    print(f"  Тривалість: {duration:.1f}s  |  Chunks: {n_chunks}  "
          f"(max={max_sec}s, hop={hop_ratio})\n")

    # ── Header таблиці ──
    col_w    = 8
    cls_hdrs = "".join(f"{c:>{col_w}}" for c in classes)
    sep      = "─" * (42 + col_w * n_classes)
    print(f"{'Chunk':<6} {'Час':>12}  {'Клас':>6}  {'Впевн':>6}  {cls_hdrs}")
    print(sep)

    chunk_probs = []   # (n_chunks, n_classes)
    chunk_preds = []   # str мітки

    import pandas as pd
    for ci, (chunk_y, t_start, t_end) in enumerate(chunks):
        feat = extract_features(chunk_y, sr,
                                beats=beats, downbeats=downbeats,
                                chunk_offset=t_start)
        # DataFrame with column names matches what the pipeline was fit with
        # (silences the "X does not have valid feature names" warning).
        vec = pd.DataFrame(
            [[feat.get(k, 0.0) for k in feature_keys]],
            columns=feature_keys,
        )

        probs     = pipeline.predict_proba(vec)[0]   # (n_classes,)
        pred_idx  = int(np.argmax(probs))
        pred_lbl  = classes[pred_idx]
        conf      = float(probs[pred_idx])

        chunk_probs.append(probs)
        chunk_preds.append(pred_lbl)

        prob_str = "".join(f"{p:>{col_w}.3f}" for p in probs)
        time_str = f"{t_start:.1f}–{t_end:.1f}s"
        print(f"  {ci+1:<4} {time_str:>12}  {pred_lbl:>6}  {conf:>6.3f}  {prob_str}")

    print(sep)

    # ── Soft voting: середнє ймовірностей ──
    mean_probs  = np.mean(chunk_probs, axis=0)   # (n_classes,)
    final_idx   = int(np.argmax(mean_probs))
    final_label = classes[final_idx]
    final_conf  = float(mean_probs[final_idx])

    # Голосування по мітках (hard voting) — для порівняння
    vote_counts = Counter(chunk_preds)
    hard_winner = vote_counts.most_common(1)[0][0]
    hard_conf   = vote_counts[hard_winner] / n_chunks * 100

    print(f"\n  Soft voting (середнє ймовірностей):")
    mean_str = "".join(f"{p:>{col_w}.3f}" for p in mean_probs)
    print(f"  {'MEAN':>6}  {'':>12}  {final_label:>6}  {final_conf:>6.3f}  {mean_str}")

    print(f"\n  Hard voting (більшість chunks): {hard_winner}  "
          f"({vote_counts[hard_winner]}/{n_chunks}  {hard_conf:.0f}%)")

    print(f"\n{'═'*55}")
    if final_label == hard_winner:
        print(f"  ✅ РЕЗУЛЬТАТ: '{final_label}'  "
              f"(soft={final_conf:.3f}, hard={hard_conf:.0f}%)")
    else:
        # Soft і hard не збіглись — попереджаємо
        print(f"  ⚠️  РОЗБІЖНІСТЬ:")
        print(f"     Soft voting → '{final_label}'  (p={final_conf:.3f})")
        print(f"     Hard voting → '{hard_winner}'  ({hard_conf:.0f}% chunks)")
        print(f"     Рекомендація: довіряй soft (враховує впевненість модели)")
    print(f"{'═'*55}")

    return final_label, mean_probs


def cmd_diagnose(audio_dir, use_demucs, use_beats, noise=None):
    all_files = get_files(audio_dir)
    files     = filter_files_by_noise(all_files, noise)
    labels    = sorted(set(f[:2].lower() for f in files))
    noise_tag = f"noise<={noise}" if noise is not None else "noise: OFF"
    print(f"[+] Файлів: {len(files)}/{len(all_files)}  |  Класи: {labels}  "
          f"|  {noise_tag}  "
          f"|  HTDemucs: {'ON' if use_demucs else 'OFF'}  "
          f"|  BeatThis: {'ON' if use_beats else 'OFF'}\n")

    records, raw = [], []
    for fname in files:
        fpath = os.path.join(audio_dir, fname)
        print(f"  → {fname}", end=" ", flush=True)
        try:
            y, sr = load_audio(fpath, use_demucs=use_demucs)
            beats, downbeats = _file_beats(fpath, y, sr, use_beats)
            feat = extract_features(y, sr, beats=beats, downbeats=downbeats)
            vec  = to_vec(feat)
            records.append({"label": fname[:2].lower(), "vec": vec})
            raw.append(vec)
            print(f"✓  rms={feat['rms_mean']:.3f}  ons={feat['onset_density']:.2f}"
                  f"  bpm={feat['bpm']:.0f}  fit={feat['grid_fit_error']:.3f}"
                  f"  aln={feat['onset_beat_alignment']:.2f}")
        except Exception as e:
            print(f"✗ {e}")

    if not records:
        print("[!] Немає даних"); return

    raw_arr = np.stack(raw)
    scale   = raw_arr.std(axis=0)
    scale[scale < 1e-8] = 1.0

    intra, inter = [], []
    for a, b in combinations(records, 2):
        d = cosine_distance(a["vec"], b["vec"], scale)
        (intra if a["label"] == b["label"] else inter).append(d)

    i_m = np.mean(intra) if intra else 0
    e_m = np.mean(inter) if inter else 0
    gap = e_m - i_m
    print(f"\n📊 Розділення (cosine):")
    print(f"   intra={i_m:.4f}  inter={e_m:.4f}  gap={gap:.4f}", end="  ")
    print("✅ ВІДМІННО" if gap > 0.4 else "✅ ДОБРЕ" if gap > 0.15
          else "⚠️  НОРМАЛЬНО" if gap > 0.05 else "❌ СЛАБО")
    return scale


def cmd_build(audio_dir, output, max_sec, hop_ratio, use_demucs, use_beats,
              noise=None):
    all_files = get_files(audio_dir)
    files     = filter_files_by_noise(all_files, noise)
    hop_sec   = max_sec * hop_ratio
    print(f"[+] Файлів: {len(files)}/{len(all_files)}  "
          f"|  max chunk: {max_sec}s  |  hop: {hop_sec:.2f}s\n")

    records, raw = [], []
    for fname in files:
        label = fname[:2].lower()
        fpath = os.path.join(audio_dir, fname)
        try:
            y, sr = load_audio(fpath, use_demucs=use_demucs)
            duration = len(y) / sr
            beats, downbeats = _file_beats(fpath, y, sr, use_beats)

            if duration <= max_sec:
                feat = extract_features(y, sr, beats=beats, downbeats=downbeats,
                                        chunk_offset=0.0)
                vec  = to_vec(feat)
                records.append({"label": label, "filename": fname, "vec": vec})
                raw.append(vec)
            else:
                chunks = split_audio_chunks(y, sr, max_sec=max_sec,
                                            hop_sec=hop_sec, drop_quiet_first=True)
                for ci, (chunk_y, t_start, t_end) in enumerate(chunks):
                    feat = extract_features(chunk_y, sr,
                                            beats=beats, downbeats=downbeats,
                                            chunk_offset=t_start)
                    vec  = to_vec(feat)
                    records.append({
                        "label":    label,
                        "filename": f"{fname}[{t_start:.1f}-{t_end:.1f}]",
                        "vec":      vec,
                    })
                    raw.append(vec)
        except Exception as e:
            print(f"  → {fname}: ✗ {e}")

    if not records:
        print("[!] Немає даних"); return

    raw_arr = np.stack(raw)
    scale = raw_arr.std(axis=0)
    scale[scale < 1e-8] = 1.0

    with open(output, "wb") as f:
        pickle.dump({
            "records":      records,
            "scale":        scale,
            "use_demucs":   use_demucs,
            "use_beats":    use_beats,
            "feature_keys": FEATURE_KEYS,
            "max_sec":      max_sec,
            "hop_ratio":    hop_ratio,
            "noise":        noise,
            "metric":       "cosine",
        }, f)
    print(f"✅ {output} ({len(records)} records)")


def cmd_classify(input_path, db_path, top_k, max_sec, hop_ratio,
                 use_demucs_override=None, use_beats_override=None):
    with open(db_path, "rb") as f:
        data = pickle.load(f)
    records         = data["records"]
    scale           = data["scale"]
    db_use_demucs   = data.get("use_demucs", False)
    db_use_beats    = data.get("use_beats", False)

    use_demucs = db_use_demucs if use_demucs_override is None else use_demucs_override
    use_beats  = db_use_beats  if use_beats_override  is None else use_beats_override

    y, sr = load_audio(input_path, use_demucs=use_demucs)
    duration = len(y) / sr
    hop_sec = max_sec * hop_ratio
    beats, downbeats = _file_beats(input_path, y, sr, use_beats)

    chunks = split_audio_chunks(y, sr, max_sec=max_sec, hop_sec=hop_sec,
                                drop_quiet_first=False)

    all_votes = Counter()
    for ci, (chunk_y, t_start, t_end) in enumerate(chunks):
        feat = extract_features(chunk_y, sr,
                                beats=beats, downbeats=downbeats,
                                chunk_offset=t_start)
        vec = to_vec(feat)
        winner, _, top = classify_vec(vec, records, scale, top_k)
        all_votes[winner] += 1
        print(f"  chunk {ci+1} [{t_start:.1f}-{t_end:.1f}s] → {winner}")

    final_winner = all_votes.most_common(1)[0][0]
    confidence = all_votes[final_winner] / len(chunks) * 100
    print(f"\n✅ РЕЗУЛЬТАТ: '{final_winner}' (впевненість: {confidence:.0f}%)")


# ─── NEW: predict_url — класифікація за URL (YouTube / TikTok / etc.) ────────

def _check_ytdlp():
    """Перевіряє що yt-dlp доступний як CLI-інструмент або Python-пакет."""
    import shutil
    if shutil.which("yt-dlp"):
        return "cli"
    try:
        import yt_dlp  # noqa
        return "python"
    except ImportError:
        pass
    raise RuntimeError(
        "yt-dlp не знайдено.\n"
        "Встанови: pip install yt-dlp\n"
        "або:      brew install yt-dlp  (macOS)\n"
        "або:      choco install yt-dlp  (Windows)"
    )


def _download_audio_ytdlp(url: str, tmp_path: str, start_sec=None, duration_sec=None):
    """
    Завантажує аудіо з URL у тимчасовий WAV-файл.
    start_sec / duration_sec — необов'язкове обрізання прямо при завантаженні
    (через postprocessor ffmpeg, без зайвого трафіку).
    Файл зберігається у tmp_path і ПОВИНЕН бути видалений після використання.
    """
    mode = _check_ytdlp()

    # Postprocessor args для ffmpeg-обрізання (якщо задано)
    pp_args = []
    if start_sec is not None or duration_sec is not None:
        ss  = f"-ss {start_sec}"  if start_sec   is not None else ""
        dur = f"-t {duration_sec}" if duration_sec is not None else ""
        pp_args = [ss, dur]
        pp_args = [a for a in pp_args if a]  # прибираємо порожні

    if mode == "python":
        import yt_dlp

        ydl_opts = {
            "format":            "bestaudio/best",
            "outtmpl":           tmp_path.replace(".wav", ".%(ext)s"),
            "postprocessors": [{
                "key":             "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }],
            "quiet":             True,
            "no_warnings":       True,
        }

        # ЗАВЖДИ додаємо -ar 16000 для сумісності з натренованою моделлю
        ffmpeg_args = ["-ar", "16000"]
        if start_sec is not None:
            ffmpeg_args += ["-ss", str(start_sec)]
        if duration_sec is not None:
            ffmpeg_args += ["-t", str(duration_sec)]
        ydl_opts["postprocessor_args"] = {"ffmpeg": ffmpeg_args}

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title    = info.get("title", url)
            uploader = info.get("uploader", "unknown")
            dur      = info.get("duration", 0)
        return title, uploader, dur

    else:  # CLI
        import subprocess, json
        # Спочатку отримуємо метадані (без завантаження)
        meta_cmd = ["yt-dlp", "--dump-json", "--no-playlist", url]
        meta_out = subprocess.run(
            meta_cmd, capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        meta     = json.loads(meta_out.stdout) if meta_out.returncode == 0 else {}
        title    = meta.get("title", url)
        uploader = meta.get("uploader", "unknown")
        dur      = meta.get("duration", 0)

        dl_cmd = [
            "yt-dlp",
            "--extract-audio", "--audio-format", "wav",
            "--no-playlist",
            "-o", tmp_path.replace(".wav", ".%(ext)s"),
            "--quiet",
        ]
        # ЗАВЖДИ додаємо -ar 16000 для сумісності з натренованою моделлю
        ffmpeg_pp = ["-ar", "16000"] + pp_args
        dl_cmd += ["--postprocessor-args", f"ffmpeg:{' '.join(ffmpeg_pp)}"]
        dl_cmd.append(url)

        result = subprocess.run(
            dl_cmd, capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"yt-dlp завершився з помилкою:\n{result.stderr}"
            )
        return title, uploader, dur


def cmd_predict_url(url, model_pkl, max_sec, hop_ratio,
                    use_demucs_override=None, use_beats_override=None,
                    start_sec=None, duration_sec=None):
    """
    Класифікує аудіо за URL (YouTube, TikTok, SoundCloud, …).

    Алгоритм:
      1. Завантажує аудіо у тимчасовий WAV (тільки аудіо-стрім, без відео)
      2. Запускає cmd_predict
      3. ВИДАЛЯЄ тимчасовий файл — нічого не залишається на диску

    Юридична позиція: особисте некомерційне використання для аналізу,
    аудіо не зберігається після класифікації.
    """
    import tempfile, shutil

    # Попередження користувачу
    print("─" * 60)
    print("  ⚠️  УВАГА: завантаження аудіо з зовнішніх платформ може")
    print("  суперечити їх Terms of Service. Цей інструмент призначений")
    print("  виключно для особистого некомерційного використання.")
    print("  Аудіо-файл буде ВИДАЛЕНО одразу після класифікації.")
    print("─" * 60)
    print()

    # Тимчасова тека — автоматично прибирається навіть при помилці
    tmp_dir = tempfile.mkdtemp(prefix="sc_url_")
    tmp_wav = os.path.join(tmp_dir, "audio.wav")

    try:
        # ── Завантаження ──
        print(f"[url] Завантажую аудіо з:\n  {url}\n")
        if start_sec is not None or duration_sec is not None:
            seg = []
            if start_sec   is not None: seg.append(f"start={start_sec}s")
            if duration_sec is not None: seg.append(f"duration={duration_sec}s")
            print(f"  Сегмент: {', '.join(seg)}")

        title, uploader, total_dur = _download_audio_ytdlp(
            url, tmp_wav, start_sec=start_sec, duration_sec=duration_sec
        )

        # yt-dlp може зберегти файл без .wav якщо конвертація не спрацювала
        if not os.path.exists(tmp_wav):
            candidates = [f for f in os.listdir(tmp_dir)
                          if f.startswith("audio")]
            if not candidates:
                raise FileNotFoundError(
                    f"yt-dlp не створив файл у {tmp_dir}. "
                    "Перевір що ffmpeg встановлений: ffmpeg -version"
                )
            tmp_wav = os.path.join(tmp_dir, candidates[0])

        size_mb = os.path.getsize(tmp_wav) / 1024 / 1024
        print(f"\n  ✅ Завантажено: '{title}'  від {uploader}")
        print(f"     Тривалість відео: {total_dur}s  |  Файл: {size_mb:.1f} MB\n")

        # ── Класифікація ──
        result_label, mean_probs = cmd_predict(
            tmp_wav, model_pkl, max_sec, hop_ratio,
            use_demucs_override=use_demucs_override,
            use_beats_override=use_beats_override,
        )
        return result_label, mean_probs

    finally:
        # ── Гарантоване видалення тимчасових файлів ──
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\n  🗑  Тимчасовий файл видалено.")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub    = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("diagnose")
    p.add_argument("--audio_dir",  default="./audio")
    p.add_argument("--use_demucs", action="store_true")
    p.add_argument("--use_beats",  action="store_true")
    p.add_argument("--noise",      choices=NOISE_CHOICES, default=None)

    p = sub.add_parser("build")
    p.add_argument("--audio_dir",  default="./audio")
    p.add_argument("--output",     default="./db_simple.pkl")
    p.add_argument("--max_sec",    type=float, default=5.0)
    p.add_argument("--hop_ratio",  type=float, default=0.7)
    p.add_argument("--use_demucs", action="store_true")
    p.add_argument("--use_beats",  action="store_true")
    p.add_argument("--noise",      choices=NOISE_CHOICES, default=None)

    p = sub.add_parser("classify")
    p.add_argument("--input",      required=True)
    p.add_argument("--db",         default="./db_simple.pkl")
    p.add_argument("--top_k",      type=int,   default=3)
    p.add_argument("--max_sec",    type=float, default=5.0)
    p.add_argument("--hop_ratio",  type=float, default=0.7)
    p.add_argument("--use_demucs", action="store_true")
    p.add_argument("--no_demucs",  action="store_true")
    p.add_argument("--use_beats",  action="store_true")
    p.add_argument("--no_beats",   action="store_true")

    # NEW: extract — фічі у CSV
    p = sub.add_parser("extract")
    p.add_argument("--audio_dir",  default="./audio")
    p.add_argument("--output",     default="./features.csv")
    p.add_argument("--max_sec",    type=float, default=6.0)
    p.add_argument("--hop_ratio",  type=float, default=0.7,
                   help="0.7 = 30%% overlap (за замовч.)")
    p.add_argument("--use_demucs", action="store_true")
    p.add_argument("--use_beats",  action="store_true")
    p.add_argument("--noise",      choices=NOISE_CHOICES, default=None)

    # train — навчання LightGBM на CSV
    p = sub.add_parser("train")
    p.add_argument("--csv",           default="./features.csv",
                   help="CSV з фічами (з команди extract)")
    p.add_argument("--output",        default="./model.pkl",
                   help="Куди зберегти навчену модель")
    p.add_argument("--n_estimators",  type=int,   default=500)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--num_leaves",    type=int,   default=63)
    p.add_argument("--cv_folds",      type=int,   default=5,
                   help="К-кість фолдів GroupKFold CV (по file_idx)")

    # predict — inference через збережену модель
    p = sub.add_parser("predict")
    p.add_argument("--input",      required=True,
                   help="Аудіо-файл для класифікації")
    p.add_argument("--model",      default="./model.pkl",
                   help="Збережена модель (з команди train)")
    p.add_argument("--max_sec",    type=float, default=6.0)
    p.add_argument("--hop_ratio",  type=float, default=0.7)
    p.add_argument("--use_demucs", action="store_true")
    p.add_argument("--no_demucs",  action="store_true")
    p.add_argument("--use_beats",  action="store_true")
    p.add_argument("--no_beats",   action="store_true")

    # predict_url — класифікація за URL
    p = sub.add_parser("predict_url")
    p.add_argument("--url",          required=True,
                   help="URL відео (YouTube, TikTok, SoundCloud, …)")
    p.add_argument("--model",        default="./model.pkl")
    p.add_argument("--max_sec",      type=float, default=6.0)
    p.add_argument("--hop_ratio",    type=float, default=0.7)
    p.add_argument("--start_sec",    type=float, default=None,
                   help="Початок сегмента (секунди від початку відео)")
    p.add_argument("--duration_sec", type=float, default=None,
                   help="Тривалість сегмента (секунди). "
                        "Якщо не вказано — весь файл")
    p.add_argument("--use_demucs",   action="store_true")
    p.add_argument("--no_demucs",    action="store_true")
    p.add_argument("--use_beats",    action="store_true")
    p.add_argument("--no_beats",     action="store_true")

    args = parser.parse_args()
    if   args.cmd == "build":
        cmd_build(args.audio_dir, args.output, args.max_sec, args.hop_ratio,
                  args.use_demucs, args.use_beats, noise=args.noise)
    elif args.cmd == "classify":
        d_ovr = True if args.use_demucs else (False if args.no_demucs else None)
        b_ovr = True if args.use_beats  else (False if args.no_beats  else None)
        cmd_classify(args.input, args.db, args.top_k, args.max_sec, args.hop_ratio,
                     use_demucs_override=d_ovr, use_beats_override=b_ovr)
    elif args.cmd == "diagnose":
        cmd_diagnose(args.audio_dir, args.use_demucs, args.use_beats,
                     noise=args.noise)
    elif args.cmd == "extract":
        cmd_extract(args.audio_dir, args.output, args.max_sec, args.hop_ratio,
                    args.use_demucs, args.use_beats, noise=args.noise)
    elif args.cmd == "train":
        cmd_train(args.csv, args.output,
                  n_estimators=args.n_estimators,
                  learning_rate=args.learning_rate,
                  num_leaves=args.num_leaves,
                  cv_folds=args.cv_folds)
    elif args.cmd == "predict":
        d_ovr = True if args.use_demucs else (False if args.no_demucs else None)
        b_ovr = True if args.use_beats  else (False if args.no_beats  else None)
        cmd_predict(args.input, args.model, args.max_sec, args.hop_ratio,
                    use_demucs_override=d_ovr, use_beats_override=b_ovr)
    elif args.cmd == "predict_url":
        d_ovr = True if args.use_demucs else (False if args.no_demucs else None)
        b_ovr = True if args.use_beats  else (False if args.no_beats  else None)
        cmd_predict_url(
            args.url, args.model, args.max_sec, args.hop_ratio,
            use_demucs_override=d_ovr, use_beats_override=b_ovr,
            start_sec=args.start_sec, duration_sec=args.duration_sec,
        )
    else:
        parser.print_help()
