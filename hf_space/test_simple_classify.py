"""Tests for simple_classify.py — re-exports and __main__ CLI dispatch."""

import runpy
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import simple_classify as sc


# ── Re-export coverage ────────────────────────────────────────────────────────

def test_public_callables():
    for name in [
        "load_audio", "separate_guitar", "compute_beats",
        "_file_beats", "_download_audio_ytdlp", "_check_ytdlp",
        "extract_features", "extract_librosa_features", "extract_beat_features",
        "_default_beat_feats", "to_vec",
        "split_audio_chunks", "cosine_distance", "filter_files_by_noise",
        "get_files", "classify_vec",
        "cmd_extract", "cmd_train", "cmd_predict", "cmd_predict_url",
        "cmd_diagnose", "cmd_build", "cmd_classify",
    ]:
        assert callable(getattr(sc, name)), f"{name} should be callable"


def test_public_constants():
    assert len(sc.FEATURE_KEYS) == 46
    assert len(sc.LIBROSA_KEYS) > 0
    assert len(sc.BEAT_KEYS) > 0
    assert len(sc.SUPPORTED) > 0
    assert len(sc.NOISE_CHOICES) > 0
    assert isinstance(sc.QUIET_THRESHOLD, float)


# ── __main__ CLI dispatch via runpy ───────────────────────────────────────────
# runpy re-executes the module, re-running all imports.
# "from ml_pipeline.classify import cmd_train" copies the reference at import
# time, so we must patch at the SOURCE (ml_pipeline.classify.*) so that the
# re-import picks up the mock.

@contextmanager
def _run_main_ctx(mock_args):
    """Patch all cmd_* at source level, then yield the mocks dict."""
    mocks = {
        "ml_pipeline.classify.cmd_build":       MagicMock(),
        "ml_pipeline.classify.cmd_classify":    MagicMock(),
        "ml_pipeline.classify.cmd_diagnose":    MagicMock(),
        "ml_pipeline.classify.cmd_extract":     MagicMock(),
        "ml_pipeline.classify.cmd_train":       MagicMock(),
        "ml_pipeline.classify.cmd_predict":     MagicMock(),
        "ml_pipeline.classify.cmd_predict_url": MagicMock(),
    }
    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args), \
         patch("ml_pipeline.classify.cmd_build",       mocks["ml_pipeline.classify.cmd_build"]), \
         patch("ml_pipeline.classify.cmd_classify",    mocks["ml_pipeline.classify.cmd_classify"]), \
         patch("ml_pipeline.classify.cmd_diagnose",    mocks["ml_pipeline.classify.cmd_diagnose"]), \
         patch("ml_pipeline.classify.cmd_extract",     mocks["ml_pipeline.classify.cmd_extract"]), \
         patch("ml_pipeline.classify.cmd_train",       mocks["ml_pipeline.classify.cmd_train"]), \
         patch("ml_pipeline.classify.cmd_predict",     mocks["ml_pipeline.classify.cmd_predict"]), \
         patch("ml_pipeline.classify.cmd_predict_url", mocks["ml_pipeline.classify.cmd_predict_url"]):
        yield mocks


def test_main_no_command_prints_help():
    mock_args = MagicMock()
    mock_args.cmd = None

    with _run_main_ctx(mock_args):
        runpy.run_module("simple_classify", run_name="__main__", alter_sys=False)


def test_main_train_dispatch():
    mock_args = MagicMock()
    mock_args.cmd = "train"
    mock_args.csv = "features.csv"
    mock_args.output = "model.pkl"
    mock_args.n_estimators = 5
    mock_args.learning_rate = 0.1
    mock_args.num_leaves = 4
    mock_args.cv_folds = 2
    mock_args.no_calibrate = True
    mock_args.calibration_method = "isotonic"
    mock_args.calibration_cv = 2

    with _run_main_ctx(mock_args) as mocks:
        runpy.run_module("simple_classify", run_name="__main__", alter_sys=False)

    mocks["ml_pipeline.classify.cmd_train"].assert_called_once()


def test_main_extract_dispatch():
    mock_args = MagicMock()
    mock_args.cmd = "extract"
    mock_args.audio_dir = "./audio"
    mock_args.output = "./features.csv"
    mock_args.max_sec = 6.0
    mock_args.hop_ratio = 0.7
    mock_args.use_demucs = False
    mock_args.use_beats = False
    mock_args.noise = None

    with _run_main_ctx(mock_args) as mocks:
        runpy.run_module("simple_classify", run_name="__main__", alter_sys=False)

    mocks["ml_pipeline.classify.cmd_extract"].assert_called_once()


def test_main_build_dispatch():
    mock_args = MagicMock()
    mock_args.cmd = "build"
    mock_args.audio_dir = "./audio"
    mock_args.output = "./db.pkl"
    mock_args.max_sec = 5.0
    mock_args.hop_ratio = 0.7
    mock_args.use_demucs = False
    mock_args.use_beats = False
    mock_args.noise = None

    with _run_main_ctx(mock_args) as mocks:
        runpy.run_module("simple_classify", run_name="__main__", alter_sys=False)

    mocks["ml_pipeline.classify.cmd_build"].assert_called_once()


def test_main_diagnose_dispatch():
    mock_args = MagicMock()
    mock_args.cmd = "diagnose"
    mock_args.audio_dir = "./audio"
    mock_args.use_demucs = False
    mock_args.use_beats = False
    mock_args.noise = None

    with _run_main_ctx(mock_args) as mocks:
        runpy.run_module("simple_classify", run_name="__main__", alter_sys=False)

    mocks["ml_pipeline.classify.cmd_diagnose"].assert_called_once()


def test_main_classify_dispatch():
    mock_args = MagicMock()
    mock_args.cmd = "classify"
    mock_args.input = "audio.wav"
    mock_args.db = "./db.pkl"
    mock_args.top_k = 3
    mock_args.max_sec = 5.0
    mock_args.hop_ratio = 0.7
    mock_args.use_demucs = False
    mock_args.no_demucs = False
    mock_args.use_beats = False
    mock_args.no_beats = False

    with _run_main_ctx(mock_args) as mocks:
        runpy.run_module("simple_classify", run_name="__main__", alter_sys=False)

    mocks["ml_pipeline.classify.cmd_classify"].assert_called_once()


def test_main_predict_dispatch():
    mock_args = MagicMock()
    mock_args.cmd = "predict"
    mock_args.input = "audio.wav"
    mock_args.model = "model.pkl"
    mock_args.max_sec = 6.0
    mock_args.hop_ratio = 0.7
    mock_args.use_demucs = False
    mock_args.no_demucs = False
    mock_args.use_beats = False
    mock_args.no_beats = False

    with _run_main_ctx(mock_args) as mocks:
        runpy.run_module("simple_classify", run_name="__main__", alter_sys=False)

    mocks["ml_pipeline.classify.cmd_predict"].assert_called_once()


def test_main_predict_url_dispatch():
    mock_args = MagicMock()
    mock_args.cmd = "predict_url"
    mock_args.url = "https://example.com/song"
    mock_args.model = "model.pkl"
    mock_args.max_sec = 6.0
    mock_args.hop_ratio = 0.7
    mock_args.start_sec = None
    mock_args.duration_sec = None
    mock_args.use_demucs = False
    mock_args.no_demucs = False
    mock_args.use_beats = False
    mock_args.no_beats = False

    with _run_main_ctx(mock_args) as mocks:
        runpy.run_module("simple_classify", run_name="__main__", alter_sys=False)

    mocks["ml_pipeline.classify.cmd_predict_url"].assert_called_once()
