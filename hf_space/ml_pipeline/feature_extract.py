"""
ML pipeline — librosa feature extraction (46 ознак: 25 librosa + 21 beat).
"""

import librosa
import numpy as np


# ─── Feature key lists ───────────────────────────────────────────────────────

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


# ─── Librosa features ────────────────────────────────────────────────────────

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


# ─── Beat features ───────────────────────────────────────────────────────────

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
