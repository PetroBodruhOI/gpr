"""
ML pipeline — HTDemucs guitar separation, BeatThis beat detection, yt-dlp audio download.
"""

import hashlib
import os
import re
import threading
import time

import librosa
import numpy as np


# ─── HTDemucs guitar separation ──────────────────────────────────────────────

DEMUCS_CACHE_DIR = "./_demucs_cache"
_DEMUCS_SEPARATOR = None


def separate_guitar(path: str, target_sr: int = 22050, progress_cb=None):
    """
    CLI-based Demucs separation (compatible with demucs 4.0.1 from PyPI,
    where demucs.api doesn't yet exist). Calls `python -m demucs -n htdemucs_6s`

    Note: Demucs naturally outputs at 16kHz (for compatibility with trained models)
    and reads the resulting guitar.wav stem from disk.

    progress_cb(pct: int) — called when Demucs reports N% in stderr (0-100).
    """
    import shutil
    import subprocess
    import tempfile

    import soundfile as sf

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
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        stderr_lines = []
        last_pct = [-1]

        def _drain_stderr():
            buf = ""
            while True:
                chunk = proc.stderr.read(128)
                if not chunk:
                    break
                buf += chunk
                parts = re.split(r"[\r\n]", buf)
                buf = parts[-1]
                for part in parts[:-1]:
                    stderr_lines.append(part)
                    if progress_cb:
                        m = re.search(r"(\d+)%", part)
                        if m:
                            pct = int(m.group(1))
                            if pct != last_pct[0]:
                                last_pct[0] = pct
                                progress_cb(pct)
            if buf:
                stderr_lines.append(buf)

        drain_thread = threading.Thread(target=_drain_stderr, daemon=True)
        drain_thread.start()
        proc.wait()
        drain_thread.join()

        if proc.returncode != 0:
            raise RuntimeError(
                f"demucs failed (rc={proc.returncode}):\n" + "\n".join(stderr_lines)
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


def load_audio(path: str, use_demucs: bool, sr: int = 22050, progress_cb=None):
    """Load audio at target sample rate (22050Hz for trained model compatibility).

    ВАЖЛИВО: НЕ застосовуй тут жодних audio-перетворень
    (preemphasis, RMS-нормалізацію, фільтри тощо).
    Модель навчалась на сирому librosa-аудіо, тому будь-яке перетворення
    зміщує розподіл фіч і ламає predict.
    """
    if use_demucs:
        return separate_guitar(path, target_sr=sr, progress_cb=progress_cb)
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


_BEAT_TICK_INTERVAL = 3  # seconds between BeatThis progress ticks


def compute_beats(y: np.ndarray, sr: int, cache_key: str = None, progress_cb=None):
    """
    Returns (beats, downbeats) — np.float32 arrays of times in seconds.
    cache_key: persistent disk-cache identifier (e.g. abs_path|mtime).
    progress_cb(elapsed_s: int) — called every _BEAT_TICK_INTERVAL seconds
    while the tracker runs (BeatThis has no internal progress API).
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

    if progress_cb is None:
        raw = tracker(y, sr)
    else:
        result_box = [None]
        exc_box = [None]

        def _run():
            try:
                result_box[0] = tracker(y, sr)
            except Exception as e:
                exc_box[0] = e

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        elapsed = 0
        while worker.is_alive():
            time.sleep(_BEAT_TICK_INTERVAL)
            elapsed += _BEAT_TICK_INTERVAL
            progress_cb(elapsed)
        worker.join()
        if exc_box[0]:
            raise exc_box[0]
        raw = result_box[0]

    beats = np.asarray(raw[0], dtype=np.float32)
    downbeats = np.asarray(raw[1], dtype=np.float32)

    if cache_path:
        np.savez(cache_path, beats=beats, downbeats=downbeats)
    return beats, downbeats


def _file_beats(fpath, y, sr, use_beats, progress_cb=None):
    if not use_beats:
        return None, None
    cache_key = f"{os.path.abspath(fpath)}|{os.path.getmtime(fpath)}|sr{sr}"
    return compute_beats(y, sr, cache_key=cache_key, progress_cb=progress_cb)


# ─── yt-dlp audio download ───────────────────────────────────────────────────

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
        import json
        import subprocess
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
