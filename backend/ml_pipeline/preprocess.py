"""
Audio preprocessing: HTDemucs guitar-stem separation and BeatThis beat tracking.

IMPORTANT: do not add any audio transforms here (pre-emphasis, RMS
normalisation, filters, ...). The model was trained on raw librosa audio, so
any transform shifts the feature distribution and breaks predictions.
"""

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading

import librosa
import numpy as np

SAMPLE_RATE = 22050


# ─── HTDemucs guitar separation ──────────────────────────────────────────────

def separate_guitar(path: str, sr: int = SAMPLE_RATE, progress_cb=None):
    """
    Extract the guitar stem with `python -m demucs -n htdemucs_6s` and load it
    as mono float32 at `sr`. Returns (y, sr).

    The CLI is used because demucs 4.0.1 on PyPI has no `demucs.api` yet.
    progress_cb(pct: int) is called whenever Demucs reports a new N% on stderr.
    """
    out_dir = tempfile.mkdtemp(prefix="demucs_")
    try:
        cmd = [
            sys.executable, "-m", "demucs",
            "-n", "htdemucs_6s",
            "--shifts", "0",      # disable test-time aug → ~2x faster on CPU
            "--overlap", "0.1",   # smaller window overlap → ~30% faster
            "-o", out_dir,
            path,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stderr_lines = _drain_demucs_stderr(proc, progress_cb)

        if proc.returncode != 0:
            raise RuntimeError(
                f"demucs failed (rc={proc.returncode}):\n" + "\n".join(stderr_lines)
            )

        track_stem = os.path.splitext(os.path.basename(path))[0]
        guitar_path = os.path.join(out_dir, "htdemucs_6s", track_stem, "guitar.wav")
        if not os.path.exists(guitar_path):
            found = glob.glob(os.path.join(out_dir, "**", "*.wav"), recursive=True)
            raise FileNotFoundError(
                f"Guitar stem not found at {guitar_path}. Available files: {found}"
            )

        y, _ = librosa.load(guitar_path, sr=sr, mono=True)
        return y.astype(np.float32), sr
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _drain_demucs_stderr(proc: subprocess.Popen, progress_cb) -> list[str]:
    """Read Demucs stderr until it exits, reporting N% progress. Returns all lines."""
    lines: list[str] = []

    def _drain():
        buf = ""
        last_pct = -1
        while True:
            chunk = proc.stderr.read(128)
            if not chunk:
                break
            buf += chunk
            # tqdm redraws with \r, so split on both \r and \n
            *complete, buf = re.split(r"[\r\n]", buf)
            for part in complete:
                lines.append(part)
                m = re.search(r"(\d+)%", part)
                if progress_cb and m and int(m.group(1)) != last_pct:
                    last_pct = int(m.group(1))
                    progress_cb(last_pct)
        if buf:
            lines.append(buf)

    thread = threading.Thread(target=_drain, daemon=True)
    thread.start()
    proc.wait()
    thread.join()
    return lines


# ─── BeatThis beat tracker ───────────────────────────────────────────────────

_BEAT_TRACKER = None
_BEAT_TICK_INTERVAL = 3  # seconds between BeatThis progress ticks


def _get_beat_tracker():
    global _BEAT_TRACKER
    if _BEAT_TRACKER is None:
        import torch
        from beat_this.inference import Audio2Beats

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _BEAT_TRACKER = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)
    return _BEAT_TRACKER


def compute_beats(y: np.ndarray, sr: int, progress_cb=None):
    """
    Returns (beats, downbeats) — float32 arrays of times in seconds.

    BeatThis has no progress API, so when progress_cb is given the tracker runs
    in a worker thread and progress_cb(elapsed_s: int) is called every
    _BEAT_TICK_INTERVAL seconds until it finishes.
    """
    tracker = _get_beat_tracker()

    if progress_cb is None:
        beats, downbeats = tracker(y, sr)
    else:
        result: dict = {}

        def _run():
            try:
                result["value"] = tracker(y, sr)
            except Exception as e:  # re-raised in the calling thread below
                result["error"] = e

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        elapsed = 0
        worker.join(_BEAT_TICK_INTERVAL)
        while worker.is_alive():
            elapsed += _BEAT_TICK_INTERVAL
            progress_cb(elapsed)
            worker.join(_BEAT_TICK_INTERVAL)
        if "error" in result:
            raise result["error"]
        beats, downbeats = result["value"]

    return np.asarray(beats, dtype=np.float32), np.asarray(downbeats, dtype=np.float32)
