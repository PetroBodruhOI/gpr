"""Тести feature extraction pipeline."""

import numpy as np
import pytest

from simple_classify import (
    FEATURE_KEYS,
    extract_features,
    split_audio_chunks,
    to_vec,
)


@pytest.fixture
def synthetic_audio():
    sr = 22050
    t = np.linspace(0, 6.0, int(6.0 * sr), endpoint=False)
    # Сигнал 220 Hz + 440 Hz, легка модуляція = імітація гітари
    y = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
    y = y.astype(np.float32)
    return y, sr


def test_split_audio_chunks_basic(synthetic_audio):
    y, sr = synthetic_audio
    chunks = split_audio_chunks(y, sr, max_sec=2.0, hop_sec=1.5)
    assert len(chunks) > 0
    for c, ts, te in chunks:
        assert te > ts
        assert len(c) == int(2.0 * sr)


def test_extract_features_returns_all_keys(synthetic_audio):
    y, sr = synthetic_audio
    feat = extract_features(y, sr, beats=np.array([0.5, 1.0, 1.5, 2.0]))
    for k in FEATURE_KEYS:
        assert k in feat, f"Відсутня ознака: {k}"
        assert isinstance(feat[k], float)


def test_to_vec_shape(synthetic_audio):
    y, sr = synthetic_audio
    feat = extract_features(y, sr)
    vec = to_vec(feat)
    assert vec.shape == (1, len(FEATURE_KEYS))
    assert vec.dtype == np.float32


def test_to_vec_handles_missing_keys():
    vec = to_vec({"tempo_bpm": 120.0})
    assert vec.shape == (1, len(FEATURE_KEYS))
    assert vec[0, 0] == 120.0
    # решта — нулі
    assert np.all(vec[0, 1:] == 0.0)
