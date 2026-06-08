"""
simple_classify.py — backward-compatibility shim.

All logic has moved to ml_pipeline/:
  ml_pipeline/preprocess.py     — HTDemucs, BeatThis, yt-dlp
  ml_pipeline/feature_extract.py — librosa features (46 ознак)
  ml_pipeline/classify.py        — LightGBM, soft voting, CLI commands

This file re-exports everything so existing imports and the CLI keep working:
  from simple_classify import load_audio, extract_features, ...
  python simple_classify.py predict --input ...
"""

import argparse

# ── Re-export everything from ml_pipeline ────────────────────────────────────
from ml_pipeline.classify import (  # noqa: F401
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
from ml_pipeline.feature_extract import (  # noqa: F401
    BEAT_GRID_STEPS,
    BEAT_KEYS,
    FEATURE_KEYS,
    LIBROSA_KEYS,
    _default_beat_feats,
    extract_beat_features,
    extract_features,
    extract_librosa_features,
    to_vec,
)
from ml_pipeline.preprocess import (  # noqa: F401
    BEATS_CACHE_DIR,
    DEMUCS_CACHE_DIR,
    _check_ytdlp,
    _download_audio_ytdlp,
    _file_beats,
    compute_beats,
    load_audio,
    separate_guitar,
)

# ── CLI entry point ───────────────────────────────────────────────────────────

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

    p = sub.add_parser("extract")
    p.add_argument("--audio_dir",  default="./audio")
    p.add_argument("--output",     default="./features.csv")
    p.add_argument("--max_sec",    type=float, default=6.0)
    p.add_argument("--hop_ratio",  type=float, default=0.7,
                   help="0.7 = 30%% overlap (за замовч.)")
    p.add_argument("--use_demucs", action="store_true")
    p.add_argument("--use_beats",  action="store_true")
    p.add_argument("--noise",      choices=NOISE_CHOICES, default=None)

    p = sub.add_parser("train")
    p.add_argument("--csv",           default="./features.csv")
    p.add_argument("--output",        default="./model.pkl")
    p.add_argument("--n_estimators",  type=int,   default=500)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--num_leaves",    type=int,   default=63)
    p.add_argument("--cv_folds",      type=int,   default=5)
    p.add_argument("--no_calibrate",  action="store_true")
    p.add_argument("--calibration_method", choices=["isotonic", "sigmoid"],
                   default="isotonic")
    p.add_argument("--calibration_cv", type=int, default=3)

    p = sub.add_parser("predict")
    p.add_argument("--input",      required=True)
    p.add_argument("--model",      default="./model.pkl")
    p.add_argument("--max_sec",    type=float, default=6.0)
    p.add_argument("--hop_ratio",  type=float, default=0.7)
    p.add_argument("--use_demucs", action="store_true")
    p.add_argument("--no_demucs",  action="store_true")
    p.add_argument("--use_beats",  action="store_true")
    p.add_argument("--no_beats",   action="store_true")

    p = sub.add_parser("predict_url")
    p.add_argument("--url",          required=True)
    p.add_argument("--model",        default="./model.pkl")
    p.add_argument("--max_sec",      type=float, default=6.0)
    p.add_argument("--hop_ratio",    type=float, default=0.7)
    p.add_argument("--start_sec",    type=float, default=None)
    p.add_argument("--duration_sec", type=float, default=None)
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
                  cv_folds=args.cv_folds,
                  calibrate=not args.no_calibrate,
                  calibration_method=args.calibration_method,
                  calibration_cv=args.calibration_cv)
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
