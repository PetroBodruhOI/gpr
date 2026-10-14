"""Tests for ml_pipeline/preprocess.py — Demucs/BeatThis wrappers with the models mocked."""

import io
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from ml_pipeline import preprocess
from ml_pipeline.preprocess import (
    SAMPLE_RATE,
    _drain_demucs_stderr,
    compute_beats,
    separate_guitar,
)


class FakeProc:
    def __init__(self, stderr_text, returncode=0, on_wait=None):
        self.stderr = io.StringIO(stderr_text)
        self.returncode = returncode
        self._on_wait = on_wait

    def wait(self):
        if self._on_wait:
            self._on_wait()


# ── Demucs ────────────────────────────────────────────────────────────

def test_drain_reports_each_new_percentage():
    pcts = []
    proc = FakeProc("  0%|\r 10%|\r 10%|\r 55%|\n100%|\ndone")
    lines = _drain_demucs_stderr(proc, pcts.append)
    assert pcts == [0, 10, 55, 100]
    assert lines[-1] == "done"


_REAL_POPEN = subprocess.Popen


def _fake_popen(tmp_dir_holder, returncode=0, write_stem=True):
    def popen(cmd, *args, **kwargs):
        if "demucs" not in cmd:  # e.g. librosa probing for ffmpeg
            return _REAL_POPEN(cmd, *args, **kwargs)
        out_dir = cmd[cmd.index("-o") + 1]
        tmp_dir_holder.append(out_dir)
        track = os.path.splitext(os.path.basename(cmd[-1]))[0]

        def on_wait():
            if write_stem:
                stem_dir = os.path.join(out_dir, "htdemucs_6s", track)
                os.makedirs(stem_dir)
                sf.write(os.path.join(stem_dir, "guitar.wav"),
                         np.zeros(44100, dtype=np.float32), 44100)

        assert cmd[:3] == [sys.executable, "-m", "demucs"]
        return FakeProc(" 50%|\n100%|\n", returncode, on_wait)
    return popen


def test_separate_guitar_loads_stem_and_cleans_up(tmp_path):
    out_dirs, pcts = [], []
    with patch("subprocess.Popen", side_effect=_fake_popen(out_dirs)):
        y, sr = separate_guitar(str(tmp_path / "song.wav"), progress_cb=pcts.append)
    assert sr == SAMPLE_RATE
    assert y.dtype == np.float32
    assert len(y) == SAMPLE_RATE  # 1 s resampled 44.1k → 22.05k
    assert pcts == [50, 100]
    assert not os.path.exists(out_dirs[0])


def test_separate_guitar_nonzero_exit_raises(tmp_path):
    with patch("subprocess.Popen", side_effect=_fake_popen([], returncode=1)):
        with pytest.raises(RuntimeError, match="demucs failed"):
            separate_guitar(str(tmp_path / "song.wav"))


def test_separate_guitar_missing_stem_raises(tmp_path):
    with patch("subprocess.Popen", side_effect=_fake_popen([], write_stem=False)):
        with pytest.raises(FileNotFoundError, match="Guitar stem not found"):
            separate_guitar(str(tmp_path / "song.wav"))


# ── BeatThis ──────────────────────────────────────────────────────────

def test_compute_beats_returns_float32():
    tracker = MagicMock(return_value=([0.5, 1.0], [0.5]))
    with patch("ml_pipeline.preprocess._get_beat_tracker", return_value=tracker):
        beats, downbeats = compute_beats(np.zeros(SAMPLE_RATE), SAMPLE_RATE)
    assert beats.dtype == np.float32 and downbeats.dtype == np.float32
    assert beats.tolist() == [0.5, 1.0]


def test_compute_beats_ticks_progress_while_running():
    import time

    def slow_tracker(y, sr):
        time.sleep(0.25)
        return [1.0], [1.0]

    ticks = []
    with patch("ml_pipeline.preprocess._get_beat_tracker", return_value=slow_tracker), \
         patch.object(preprocess, "_BEAT_TICK_INTERVAL", 0.1):
        beats, _ = compute_beats(np.zeros(10), SAMPLE_RATE, progress_cb=ticks.append)
    assert beats.tolist() == [1.0]
    assert len(ticks) >= 1


def test_compute_beats_propagates_tracker_error():
    def broken(y, sr):
        raise RuntimeError("boom")

    with patch("ml_pipeline.preprocess._get_beat_tracker", return_value=broken):
        with pytest.raises(RuntimeError, match="boom"):
            compute_beats(np.zeros(10), SAMPLE_RATE, progress_cb=lambda s: None)
