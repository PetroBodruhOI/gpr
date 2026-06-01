"""
ML pipeline — LightGBM training/inference, soft voting, chunk splitting, CLI commands.
"""

import csv
import os
import pickle
from collections import Counter
from itertools import combinations

import numpy as np

from ml_pipeline.feature_extract import (
    BEAT_KEYS,
    FEATURE_KEYS,
    LIBROSA_KEYS,
    extract_features,
    to_vec,
)
from ml_pipeline.preprocess import (
    _download_audio_ytdlp,
    _file_beats,
    load_audio,
)


# ─── Cosine similarity ───────────────────────────────────────────────────────

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


# ─── File utilities ──────────────────────────────────────────────────────────

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


# ─── Commands ────────────────────────────────────────────────────────────────

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
                    # Audio-level augmentation: each chunk → 3 variants.
                    # extract_features recomputes all 46 features per variant,
                    # so the rows are genuinely different (not a label trick).
                    variants = [
                        ("orig",  chunk_y),
                        ("gain",  chunk_y * np.random.uniform(0.8, 1.2)),
                        ("noise", chunk_y + np.random.normal(
                            0, 0.005, chunk_y.shape
                        ).astype(np.float32)),
                    ]
                    for aug_tag, y_aug in variants:
                        feat = extract_features(
                            y_aug, sr,
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
                            "aug":         aug_tag,
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
                 "chunk_idx", "chunk_start", "chunk_end", "aug"]
    fieldnames = meta_cols + list(FEATURE_KEYS)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    labels = sorted(set(r["label"] for r in rows))
    label_counts = Counter(r["label"] for r in rows)
    aug_counts = Counter(r.get("aug", "orig") for r in rows)
    print(f"\n{'═'*55}")
    print(f"  ✅ {output_csv}")
    print(f"  📦 Рядків (chunks)  : {len(rows)}")
    print(f"  📁 Файлів           : {len(files) - len(errors)}/{len(files)}")
    print(f"  🗂  Класи            : {labels}")
    print(f"  📊 По класах        : {', '.join(f'{li}={c}' for li, c in sorted(label_counts.items()))}")  # noqa: E741)
    print(f"  🔀 Аугментація      : {', '.join(f'{a}={c}' for a, c in sorted(aug_counts.items()))}")
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


# ─── Calibration helpers ─────────────────────────────────────────────────────

def _ece_top_label(y_true, y_proba, n_bins=10):
    """
    Expected Calibration Error (top-label, multi-class).
    0 = ідеально калібровано, > 0.1 — модель «бреше» про впевненість.
    """
    confidences = y_proba.max(axis=1)
    preds       = y_proba.argmax(axis=1)
    accuracies  = (preds == y_true).astype(float)
    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc  = accuracies[mask].mean()
        ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
    return float(ece)


def _brier_multiclass(y_true, y_proba):
    """
    Multi-class Brier score: середня квадратна відстань predict_proba до one-hot.
    0 = ідеально.
    """
    onehot = np.zeros_like(y_proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))


# ─── train ───────────────────────────────────────────────────────────────────

def cmd_train(csv_path, output_pkl, n_estimators=500, learning_rate=0.05,
              num_leaves=63, cv_folds=5,
              calibrate=True, calibration_method="isotonic",
              calibration_cv=3):
    """
    Читає features.csv (з cmd_extract), тренує LightGBM через GroupKFold
    (групи = file_idx, без leakage між chunks одного файла), зберігає model.pkl.
    """
    try:
        import pandas as pd
        from lightgbm import LGBMClassifier
        from sklearn.base import clone
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder, StandardScaler
    except ImportError as e:
        raise ImportError(
            "Встанови залежності: pip install lightgbm scikit-learn pandas"
        ) from e

    print(f"[train] Читаю {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"  Рядків: {len(df)}  |  Колонок: {df.shape[1]}")

    META_COLS = ["file_idx", "filename", "label", "chunk_idx",
                 "chunk_start", "chunk_end", "aug"]
    feature_keys = [c for c in df.columns if c not in META_COLS]

    X = df[feature_keys].values.astype(np.float32)
    labels_raw = df["label"].values
    file_ids   = df["file_idx"].values

    le = LabelEncoder()
    y  = le.fit_transform(labels_raw)
    classes = list(le.classes_)

    label_counts = Counter(labels_raw)
    print(f"  Класи: {classes}")
    print("  По класах: " + ", ".join(f"{c}={label_counts[c]}" for c in classes))
    print(f"  Фічей: {len(feature_keys)}")
    print(f"  Калібрування: {'ON (' + calibration_method + f', cv={calibration_cv})' if calibrate else 'OFF'}\n")

    def _make_base_lgbm():
        return LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )

    def _make_pipeline(estimator):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   estimator),
        ])

    if calibrate:
        final_estimator = CalibratedClassifierCV(
            _make_base_lgbm(),
            method=calibration_method,
            cv=calibration_cv,
        )
    else:
        final_estimator = _make_base_lgbm()

    pipeline = _make_pipeline(final_estimator)

    print(f"[train] GroupKFold CV (k={cv_folds}, groups=file_idx) …")
    gkf = GroupKFold(n_splits=cv_folds)
    cv_acc, cv_f1 = [], []
    brier_base, brier_cal = [], []
    ece_base,   ece_cal   = [], []
    fold_reports = []

    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=file_ids)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        file_ids_va = file_ids[va_idx]

        pipe = _make_pipeline(clone(final_estimator))
        pipe.fit(X_tr, y_tr)

        if calibrate:
            base_pipe = _make_pipeline(_make_base_lgbm())
            base_pipe.fit(X_tr, y_tr)
            p_base = base_pipe.predict_proba(X_va)
            p_cal  = pipe.predict_proba(X_va)
            brier_base.append(_brier_multiclass(y_va, p_base))
            brier_cal .append(_brier_multiclass(y_va, p_cal))
            ece_base  .append(_ece_top_label(y_va, p_base))
            ece_cal   .append(_ece_top_label(y_va, p_cal))

        file_true, file_pred = [], []
        for fid in np.unique(file_ids_va):
            mask = file_ids_va == fid
            chunk_preds = pipe.predict(X_va[mask])
            vote = Counter(chunk_preds).most_common(1)[0][0]
            file_true.append(y_va[mask][0])
            file_pred.append(vote)

        acc = accuracy_score(file_true, file_pred)
        f1  = f1_score(file_true, file_pred, average="macro", zero_division=0)
        cv_acc.append(acc)
        cv_f1.append(f1)
        fold_reports.append((file_true, file_pred))

        extra = ""
        if calibrate:
            extra = (f"  Brier {brier_base[-1]:.4f}→{brier_cal[-1]:.4f}"
                     f"  ECE {ece_base[-1]:.4f}→{ece_cal[-1]:.4f}")
        print(f"  fold {fold_i+1}/{cv_folds}  file-acc={acc:.4f}  F1m={f1:.4f}{extra}")

    cv_acc = np.array(cv_acc)
    cv_f1  = np.array(cv_f1)
    print(f"\n  CV file-accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"  CV F1 macro      : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    calibration_summary = None
    if calibrate:
        bb, bc = float(np.mean(brier_base)), float(np.mean(brier_cal))
        eb, ec = float(np.mean(ece_base)),   float(np.mean(ece_cal))
        calibration_summary = {
            "method":     calibration_method,
            "cv":         calibration_cv,
            "brier_base": bb, "brier_cal": bc,
            "ece_base":   eb, "ece_cal":   ec,
        }
        print("\n  Калібрування (chunk-level, усереднено по фолдах):")
        print(f"    Brier score : base={bb:.4f}  →  calibrated={bc:.4f}  "
              f"(Δ={bb-bc:+.4f})")
        print(f"    ECE         : base={eb:.4f}  →  calibrated={ec:.4f}  "
              f"(Δ={eb-ec:+.4f})")
        print("    (нижче = краще; від'ємна Δ означає що калібрування погіршило)")

    print(f"\n[train] Фінальне навчання на всіх {len(X)} chunks …")
    pipeline.fit(X, y)
    print("  ✅ Готово")

    bundle = {
        "pipeline":      pipeline,
        "classes":       classes,
        "label_encoder": le,
        "feature_keys":  feature_keys,
        "cv_acc":        cv_acc,
        "cv_f1":         cv_f1,
        "calibration":   calibration_summary,
        "config": {
            "n_estimators":       n_estimators,
            "learning_rate":      learning_rate,
            "num_leaves":         num_leaves,
            "cv_folds":           cv_folds,
            "calibrate":          calibrate,
            "calibration_method": calibration_method if calibrate else None,
            "calibration_cv":     calibration_cv     if calibrate else None,
            "csv_path":           csv_path,
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
    if calibration_summary:
        print(f"  🎯 ECE    : {calibration_summary['ece_base']:.4f}"
              f" → {calibration_summary['ece_cal']:.4f}")
    print(f"{'═'*55}")


# ─── predict ─────────────────────────────────────────────────────────────────

def cmd_predict(input_path, model_pkl, max_sec, hop_ratio,
                use_demucs_override=None, use_beats_override=None):
    """
    Класифікує аудіо через збережену LightGBM-модель.
    Фінальний вердикт = середнє ймовірностей по всіх chunks (soft voting).
    """
    import pandas as pd

    print(f"[predict] Завантажую модель з {model_pkl} …")
    with open(model_pkl, "rb") as f:
        bundle = pickle.load(f)

    pipeline     = bundle["pipeline"]
    classes      = bundle["classes"]
    feature_keys = bundle["feature_keys"]
    cfg          = bundle.get("config", {})

    use_demucs = (use_demucs_override if use_demucs_override is not None
                  else cfg.get("use_demucs", False))
    use_beats  = (use_beats_override  if use_beats_override  is not None
                  else cfg.get("use_beats",  False))

    n_classes = len(classes)
    print(f"  Класи ({n_classes}): {classes}")
    print(f"  Фічей: {len(feature_keys)}")
    print(f"  HTDemucs: {'ON' if use_demucs else 'OFF'}  "
          f"BeatThis: {'ON' if use_beats else 'OFF'}\n")

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

    col_w    = 8
    cls_hdrs = "".join(f"{c:>{col_w}}" for c in classes)
    sep      = "─" * (42 + col_w * n_classes)
    print(f"{'Chunk':<6} {'Час':>12}  {'Клас':>6}  {'Впевн':>6}  {cls_hdrs}")
    print(sep)

    chunk_probs = []
    chunk_preds = []

    for ci, (chunk_y, t_start, t_end) in enumerate(chunks):
        feat = extract_features(chunk_y, sr,
                                beats=beats, downbeats=downbeats,
                                chunk_offset=t_start)
        vec = pd.DataFrame(
            [[feat.get(k, 0.0) for k in feature_keys]],
            columns=feature_keys,
        )

        probs     = pipeline.predict_proba(vec)[0]
        pred_idx  = int(np.argmax(probs))
        pred_lbl  = classes[pred_idx]
        conf      = float(probs[pred_idx])

        chunk_probs.append(probs)
        chunk_preds.append(pred_lbl)

        prob_str = "".join(f"{p:>{col_w}.3f}" for p in probs)
        time_str = f"{t_start:.1f}–{t_end:.1f}s"
        print(f"  {ci+1:<4} {time_str:>12}  {pred_lbl:>6}  {conf:>6.3f}  {prob_str}")

    print(sep)

    mean_probs  = np.mean(chunk_probs, axis=0)
    final_idx   = int(np.argmax(mean_probs))
    final_label = classes[final_idx]
    final_conf  = float(mean_probs[final_idx])

    vote_counts = Counter(chunk_preds)
    hard_winner = vote_counts.most_common(1)[0][0]
    hard_conf   = vote_counts[hard_winner] / n_chunks * 100

    print("\n  Soft voting (середнє ймовірностей):")
    mean_str = "".join(f"{p:>{col_w}.3f}" for p in mean_probs)
    print(f"  {'MEAN':>6}  {'':>12}  {final_label:>6}  {final_conf:>6.3f}  {mean_str}")

    print(f"\n  Hard voting (більшість chunks): {hard_winner}  "
          f"({vote_counts[hard_winner]}/{n_chunks}  {hard_conf:.0f}%)")

    print(f"\n{'═'*55}")
    if final_label == hard_winner:
        print(f"  ✅ РЕЗУЛЬТАТ: '{final_label}'  "
              f"(soft={final_conf:.3f}, hard={hard_conf:.0f}%)")
    else:
        print("  ⚠️  РОЗБІЖНІСТЬ:")
        print(f"     Soft voting → '{final_label}'  (p={final_conf:.3f})")
        print(f"     Hard voting → '{hard_winner}'  ({hard_conf:.0f}% chunks)")
        print("     Рекомендація: довіряй soft (враховує впевненість моделі)")
    print(f"{'═'*55}")

    return final_label, mean_probs


# ─── predict_url ─────────────────────────────────────────────────────────────

def cmd_predict_url(url, model_pkl, max_sec, hop_ratio,
                    use_demucs_override=None, use_beats_override=None,
                    start_sec=None, duration_sec=None):
    """
    Класифікує аудіо за URL (YouTube, TikTok, SoundCloud, …).
    Аудіо завантажується у тимчасовий файл, класифікується і ВИДАЛЯЄТЬСЯ.
    """
    import shutil
    import tempfile

    print("─" * 60)
    print("  ⚠️  УВАГА: завантаження аудіо з зовнішніх платформ може")
    print("  суперечити їх Terms of Service. Цей інструмент призначений")
    print("  виключно для особистого некомерційного використання.")
    print("  Аудіо-файл буде ВИДАЛЕНО одразу після класифікації.")
    print("─" * 60)
    print()

    tmp_dir = tempfile.mkdtemp(prefix="sc_url_")
    tmp_wav = os.path.join(tmp_dir, "audio.wav")

    try:
        print(f"[url] Завантажую аудіо з:\n  {url}\n")
        if start_sec is not None or duration_sec is not None:
            seg = []
            if start_sec   is not None: seg.append(f"start={start_sec}s")  # noqa: E701
            if duration_sec is not None: seg.append(f"duration={duration_sec}s")  # noqa: E701
            print(f"  Сегмент: {', '.join(seg)}")

        title, uploader, total_dur = _download_audio_ytdlp(
            url, tmp_wav, start_sec=start_sec, duration_sec=duration_sec
        )

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

        result_label, mean_probs = cmd_predict(
            tmp_wav, model_pkl, max_sec, hop_ratio,
            use_demucs_override=use_demucs_override,
            use_beats_override=use_beats_override,
        )
        return result_label, mean_probs

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("\n  🗑  Тимчасовий файл видалено.")


# ─── diagnose ────────────────────────────────────────────────────────────────

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
        print("[!] Немає даних")
        return

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
    print("\n📊 Розділення (cosine):")
    print(f"   intra={i_m:.4f}  inter={e_m:.4f}  gap={gap:.4f}", end="  ")
    print("✅ ВІДМІННО" if gap > 0.4 else "✅ ДОБРЕ" if gap > 0.15
          else "⚠️  НОРМАЛЬНО" if gap > 0.05 else "❌ СЛАБО")
    return scale


# ─── build ───────────────────────────────────────────────────────────────────

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
        print("[!] Немає даних")
        return

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


# ─── classify (legacy kNN) ───────────────────────────────────────────────────

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
