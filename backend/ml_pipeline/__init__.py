"""
GPR ML pipeline.

  preprocess.py      — HTDemucs guitar separation, BeatThis beat tracking
  feature_extract.py — 46 per-chunk features (librosa + beat grid)
  classify.py        — chunking + LightGBM soft-voting inference
"""
