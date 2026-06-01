"""
ml_pipeline — ML pipeline package for GPR.

Re-exports all public symbols so both forms work:
  from ml_pipeline import load_audio
  from ml_pipeline.preprocess import load_audio
"""

from ml_pipeline.classify import (
    NOISE_CHOICES,
    QUIET_THRESHOLD,
    SUPPORTED,
    classify_vec,
    cmd_build,
    cmd_classify,
    cmd_diagnose,
    cmd_extract,
    cmd_predict,
    cmd_predict_url,
    cmd_train,
    cosine_distance,
    filter_files_by_noise,
    get_files,
    split_audio_chunks,
)
from ml_pipeline.feature_extract import (
    BEAT_GRID_STEPS,
    BEAT_KEYS,
    FEATURE_KEYS,
    LIBROSA_KEYS,
    extract_beat_features,
    extract_features,
    extract_librosa_features,
    to_vec,
)
from ml_pipeline.preprocess import (
    BEATS_CACHE_DIR,
    DEMUCS_CACHE_DIR,
    _download_audio_ytdlp,
    _file_beats,
    compute_beats,
    load_audio,
    separate_guitar,
)

__all__ = [
    # preprocess
    "DEMUCS_CACHE_DIR", "BEATS_CACHE_DIR",
    "separate_guitar", "load_audio",
    "compute_beats", "_file_beats",
    "_download_audio_ytdlp",
    # feature_extract
    "LIBROSA_KEYS", "BEAT_GRID_STEPS", "BEAT_KEYS", "FEATURE_KEYS",
    "extract_librosa_features", "extract_beat_features",
    "extract_features", "to_vec",
    # classify
    "QUIET_THRESHOLD", "SUPPORTED", "NOISE_CHOICES",
    "split_audio_chunks", "classify_vec", "cosine_distance",
    "get_files", "filter_files_by_noise",
    "cmd_extract", "cmd_train", "cmd_predict", "cmd_predict_url",
    "cmd_diagnose", "cmd_build", "cmd_classify",
]
