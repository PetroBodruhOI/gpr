"""
Offline training CLI — builds the model.pkl that backend/app.py serves.
Reuses backend/ml_pipeline so training and inference compute features with
exactly the same code. Run from this folder:

  python train.py extract --audio_dir ./audio --output features.csv --use_demucs --use_beats
  python train.py train   --csv features.csv --output model.pkl

Training files are named "<label>...<ext>": the first two characters are the
class (e.g. "6a_song_f1.wav" → "6a"); each "f" in the stem marks one noise layer.

Demucs stems and beat grids are cached in ./_demucs_cache and ./_beats_cache,
keyed by file path + mtime, so re-running extract over the dataset is fast.
"""

import argparse
import csv
import hashlib
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from ml_pipeline.classify import split_audio_chunks  # noqa: E402
from ml_pipeline.feature_extract import (  # noqa: E402
    BEAT_KEYS, FEATURE_KEYS, LIBROSA_KEYS, extract_features,
)
from ml_pipeline.preprocess import SAMPLE_RATE, compute_beats, separate_guitar  # noqa: E402

SUPPORTED = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
NOISE_CHOICES = ["0", "1", "2", "3", "m"]
META_COLS = ["file_idx", "filename", "label", "chunk_idx",
             "chunk_start", "chunk_end", "aug"]

DEMUCS_CACHE_DIR = "./_demucs_cache"
BEATS_CACHE_DIR = "./_beats_cache"


# ─── Cached audio loading ────────────────────────────────────────────────────

def _file_key(path: str) -> str:
    return f"{os.path.abspath(path)}|{os.path.getmtime(path)}"


def load_audio(path: str, use_demucs: bool):
    if not use_demucs:
        return librosa.load(path, sr=SAMPLE_RATE, mono=True)

    os.makedirs(DEMUCS_CACHE_DIR, exist_ok=True)
    key = hashlib.md5(_file_key(path).encode()).hexdigest()
    cache_path = os.path.join(DEMUCS_CACHE_DIR, f"{key}_guitar.wav")
    if os.path.exists(cache_path):
        return librosa.load(cache_path, sr=SAMPLE_RATE, mono=True)

    y, sr = separate_guitar(path, sr=SAMPLE_RATE)
    sf.write(cache_path, y, sr)
    return y, sr


def load_beats(path: str, y, sr, use_beats: bool):
    if not use_beats:
        return None, None

    os.makedirs(BEATS_CACHE_DIR, exist_ok=True)
    key = hashlib.md5(f"{_file_key(path)}|sr{sr}".encode()).hexdigest()
    cache_path = os.path.join(BEATS_CACHE_DIR, f"{key}.npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        return d["beats"], d["downbeats"]

    beats, downbeats = compute_beats(y, sr)
    np.savez(cache_path, beats=beats, downbeats=downbeats)
    return beats, downbeats


# ─── Dataset files ───────────────────────────────────────────────────────────

def get_files(d):
    return sorted(f for f in os.listdir(d)
                  if os.path.splitext(f)[1].lower() in SUPPORTED)


def filter_files_by_noise(files, noise):
    """
    Keep files by noise level encoded in the name (case-insensitive, extension ignored):
      None → all files;  "0".."3" → at most N "f" markers;
      "m"  → clean files plus files marked "mf".
    """
    if noise is None:
        return files

    def stem(fname):
        return os.path.splitext(fname)[0].lower()

    if noise == "m":
        return [fn for fn in files if "f" not in stem(fn) or "mf" in stem(fn)]
    n = int(noise)
    return [fn for fn in files if stem(fn).count("f") <= n]


# ─── extract ─────────────────────────────────────────────────────────────────

def cmd_extract(audio_dir, output_csv, max_sec, hop_ratio, use_demucs, use_beats,
                noise=None):
    """
    Extract features from every file in audio_dir into a CSV
    (one row per chunk × augmentation; columns = META_COLS + FEATURE_KEYS).
    """
    all_files = get_files(audio_dir)
    files = filter_files_by_noise(all_files, noise)
    hop_sec = max_sec * hop_ratio
    noise_tag = f"noise<={noise}" if noise is not None else "noise: OFF"
    print(f"[+] Файлів: {len(files)}/{len(all_files)}  |  max chunk: {max_sec}s  "
          f"|  hop: {hop_sec:.2f}s ({int(hop_ratio*100)}%)  "
          f"|  {noise_tag}  "
          f"|  HTDemucs: {'ON' if use_demucs else 'OFF'}  "
          f"|  BeatThis: {'ON' if use_beats else 'OFF'}\n")

    rows = []
    errors = []

    for file_idx, fname in enumerate(files):
        label = fname[:2].lower()
        fpath = os.path.join(audio_dir, fname)
        try:
            y, sr = load_audio(fpath, use_demucs)
            duration = len(y) / sr
            beats, downbeats = load_beats(fpath, y, sr, use_beats)

            if duration <= max_sec:
                chunks = [(y, 0.0, duration)]
            else:
                chunks = split_audio_chunks(y, sr, max_sec=max_sec, hop_sec=hop_sec,
                                            drop_quiet_first=True)

            print(f"  → {fname:<35} ({label})  {duration:.1f}s → {len(chunks)} chunks")
            for ci, (chunk_y, t_start, t_end) in enumerate(chunks):
                try:
                    # Audio-level augmentation: each chunk → 3 variants.
                    # Features are recomputed per variant, so rows genuinely differ.
                    variants = [
                        ("orig", chunk_y),
                        ("gain", chunk_y * np.random.uniform(0.8, 1.2)),
                        ("noise", chunk_y + np.random.normal(
                            0, 0.005, chunk_y.shape).astype(np.float32)),
                    ]
                    for aug_tag, y_aug in variants:
                        feat = extract_features(y_aug, sr, beats=beats, downbeats=downbeats,
                                                chunk_offset=t_start)
                        row = {
                            "file_idx": file_idx,
                            "filename": fname,
                            "label": label,
                            "chunk_idx": ci,
                            "chunk_start": round(t_start, 3),
                            "chunk_end": round(t_end, 3),
                            "aug": aug_tag,
                        }
                        row.update({k: feat[k] for k in FEATURE_KEYS})
                        rows.append(row)
                except Exception as e:
                    print(f"     chunk {ci+1}: ✗ {e}")
        except Exception as e:
            errors.append((fname, str(e)))
            print(f"  → {fname:<35} ✗ {e}")

    if not rows:
        print("[!] Немає даних")
        return

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=META_COLS + list(FEATURE_KEYS))
        writer.writeheader()
        writer.writerows(rows)

    label_counts = Counter(r["label"] for r in rows)
    aug_counts = Counter(r["aug"] for r in rows)
    print(f"\n{'═'*55}")
    print(f"  ✅ {output_csv}")
    print(f"  📦 Рядків (chunks)  : {len(rows)}")
    print(f"  📁 Файлів           : {len(files) - len(errors)}/{len(files)}")
    print(f"  🗂  Класи            : {sorted(label_counts)}")
    print(f"  📊 По класах        : {', '.join(f'{c}={n}' for c, n in sorted(label_counts.items()))}")
    print(f"  🔀 Аугментація      : {', '.join(f'{a}={n}' for a, n in sorted(aug_counts.items()))}")
    print(f"  🎯 Фічей            : {len(FEATURE_KEYS)}  "
          f"(librosa: {len(LIBROSA_KEYS)}, beats: {len(BEAT_KEYS)})")
    print(f"  ⏱  Chunk            : {max_sec}s, hop {hop_sec:.2f}s")
    print(f"{'═'*55}")
    if errors:
        print(f"\n⚠️  Файли з помилками ({len(errors)}):")
        for fname, msg in errors:
            print(f"     {fname}: {msg}")


# ─── Calibration metrics ─────────────────────────────────────────────────────

def _ece_top_label(y_true, y_proba, n_bins=10):
    """Expected Calibration Error (top-label). 0 = perfect; > 0.1 = over/under-confident."""
    confidences = y_proba.max(axis=1)
    accuracies = (y_proba.argmax(axis=1) == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += mask.mean() * abs(confidences[mask].mean() - accuracies[mask].mean())
    return float(ece)


def _brier_multiclass(y_true, y_proba):
    """Multi-class Brier score: mean squared distance to one-hot. 0 = perfect."""
    onehot = np.zeros_like(y_proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))


# ─── train ───────────────────────────────────────────────────────────────────

def cmd_train(csv_path, output_pkl, n_estimators=500, learning_rate=0.05,
              num_leaves=63, cv_folds=5,
              calibrate=True, calibration_method="isotonic",
              calibration_cv=3):
    """
    Train LightGBM on features.csv with GroupKFold CV (groups = file_idx, so
    chunks of one file never leak between train and validation), then fit on
    all data and save the bundle that app.py loads.
    """
    import pandas as pd
    from lightgbm import LGBMClassifier
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    print(f"[train] Читаю {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"  Рядків: {len(df)}  |  Колонок: {df.shape[1]}")

    feature_keys = [c for c in df.columns if c not in META_COLS]
    X = df[feature_keys].values.astype(np.float32)
    labels_raw = df["label"].values
    file_ids = df["file_idx"].values

    le = LabelEncoder()
    y = le.fit_transform(labels_raw)
    classes = list(le.classes_)

    label_counts = Counter(labels_raw)
    print(f"  Класи: {classes}")
    print("  По класах: " + ", ".join(f"{c}={label_counts[c]}" for c in classes))
    print(f"  Фічей: {len(feature_keys)}")
    cal_tag = f"ON ({calibration_method}, cv={calibration_cv})" if calibrate else "OFF"
    print(f"  Калібрування: {cal_tag}\n")

    def make_lgbm():
        return LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )

    def make_pipeline(estimator):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    final_estimator = (
        CalibratedClassifierCV(make_lgbm(), method=calibration_method, cv=calibration_cv)
        if calibrate else make_lgbm()
    )

    print(f"[train] GroupKFold CV (k={cv_folds}, groups=file_idx) …")
    cv_acc, cv_f1 = [], []
    brier_base, brier_cal, ece_base, ece_cal = [], [], [], []

    for fold_i, (tr_idx, va_idx) in enumerate(
            GroupKFold(n_splits=cv_folds).split(X, y, groups=file_ids)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        file_ids_va = file_ids[va_idx]

        pipe = make_pipeline(clone(final_estimator))
        pipe.fit(X_tr, y_tr)

        if calibrate:
            base_pipe = make_pipeline(make_lgbm())
            base_pipe.fit(X_tr, y_tr)
            p_base = base_pipe.predict_proba(X_va)
            p_cal = pipe.predict_proba(X_va)
            brier_base.append(_brier_multiclass(y_va, p_base))
            brier_cal.append(_brier_multiclass(y_va, p_cal))
            ece_base.append(_ece_top_label(y_va, p_base))
            ece_cal.append(_ece_top_label(y_va, p_cal))

        # File-level accuracy: majority vote over each file's chunks
        file_true, file_pred = [], []
        for fid in np.unique(file_ids_va):
            mask = file_ids_va == fid
            file_true.append(y_va[mask][0])
            file_pred.append(Counter(pipe.predict(X_va[mask])).most_common(1)[0][0])

        acc = accuracy_score(file_true, file_pred)
        f1 = f1_score(file_true, file_pred, average="macro", zero_division=0)
        cv_acc.append(acc)
        cv_f1.append(f1)

        extra = ""
        if calibrate:
            extra = (f"  Brier {brier_base[-1]:.4f}→{brier_cal[-1]:.4f}"
                     f"  ECE {ece_base[-1]:.4f}→{ece_cal[-1]:.4f}")
        print(f"  fold {fold_i+1}/{cv_folds}  file-acc={acc:.4f}  F1m={f1:.4f}{extra}")

    cv_acc = np.array(cv_acc)
    cv_f1 = np.array(cv_f1)
    print(f"\n  CV file-accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"  CV F1 macro      : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    calibration_summary = None
    if calibrate:
        calibration_summary = {
            "method": calibration_method,
            "cv": calibration_cv,
            "brier_base": float(np.mean(brier_base)),
            "brier_cal": float(np.mean(brier_cal)),
            "ece_base": float(np.mean(ece_base)),
            "ece_cal": float(np.mean(ece_cal)),
        }
        s = calibration_summary
        print("\n  Калібрування (chunk-level, усереднено по фолдах):")
        print(f"    Brier score : base={s['brier_base']:.4f}  →  calibrated={s['brier_cal']:.4f}")
        print(f"    ECE         : base={s['ece_base']:.4f}  →  calibrated={s['ece_cal']:.4f}")
        print("    (нижче = краще)")

    print(f"\n[train] Фінальне навчання на всіх {len(X)} chunks …")
    pipeline = make_pipeline(final_estimator)
    pipeline.fit(X, y)

    bundle = {
        "pipeline": pipeline,
        "classes": classes,
        "label_encoder": le,
        "feature_keys": feature_keys,
        "cv_acc": cv_acc,
        "cv_f1": cv_f1,
        "calibration": calibration_summary,
        "config": {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "cv_folds": cv_folds,
            "calibrate": calibrate,
            "calibration_method": calibration_method if calibrate else None,
            "calibration_cv": calibration_cv if calibrate else None,
            "csv_path": csv_path,
        },
    }
    with open(output_pkl, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n{'═'*55}")
    print(f"  ✅ {output_pkl}  ({os.path.getsize(output_pkl) // 1024} KB)")
    print(f"  🗂  Класи : {classes}")
    print(f"  📊 CV acc : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"  📊 CV F1m : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"{'═'*55}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="audio dir → features.csv")
    p.add_argument("--audio_dir", default="./audio")
    p.add_argument("--output", default="./features.csv")
    p.add_argument("--max_sec", type=float, default=6.0)
    p.add_argument("--hop_ratio", type=float, default=0.7,
                   help="hop = max_sec × hop_ratio; 0.7 = 30%% overlap")
    p.add_argument("--use_demucs", action="store_true")
    p.add_argument("--use_beats", action="store_true")
    p.add_argument("--noise", choices=NOISE_CHOICES, default=None)

    p = sub.add_parser("train", help="features.csv → model.pkl")
    p.add_argument("--csv", default="./features.csv")
    p.add_argument("--output", default="./model.pkl")
    p.add_argument("--n_estimators", type=int, default=500)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--num_leaves", type=int, default=63)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--no_calibrate", action="store_true")
    p.add_argument("--calibration_method", choices=["isotonic", "sigmoid"],
                   default="isotonic")
    p.add_argument("--calibration_cv", type=int, default=3)

    args = parser.parse_args(argv)
    if args.cmd == "extract":
        cmd_extract(args.audio_dir, args.output, args.max_sec, args.hop_ratio,
                    args.use_demucs, args.use_beats, noise=args.noise)
    else:
        cmd_train(args.csv, args.output,
                  n_estimators=args.n_estimators,
                  learning_rate=args.learning_rate,
                  num_leaves=args.num_leaves,
                  cv_folds=args.cv_folds,
                  calibrate=not args.no_calibrate,
                  calibration_method=args.calibration_method,
                  calibration_cv=args.calibration_cv)


if __name__ == "__main__":
    main()
