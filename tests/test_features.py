"""特征提取模块的单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import build_feature_table, extract_features


def test_window_count_and_overlap():
    signal = np.zeros(1000)
    table = extract_features(
        signal, sample_rate=100.0, window_sec=1.0, step_sec=0.5
    )
    # 1000 个点，窗口 100 点、步长 50 点：(1000-100)/50+1 = 19
    assert len(table) == 19
    assert table["window_start_s"].is_monotonic_increasing


def test_rms_of_sine():
    sample_rate = 1000.0
    t = np.arange(int(sample_rate * 2)) / sample_rate
    signal = np.sin(2 * np.pi * 50.0 * t)
    table = extract_features(signal, sample_rate, window_sec=1.0, step_sec=1.0)
    assert np.allclose(table["rms"], 1 / np.sqrt(2), atol=1e-6)
    # 均值接近 0，主频应为 50 Hz
    assert table["mean"].abs().max() < 1e-9
    assert np.allclose(table["dominant_hz"], 50.0)


def test_short_signal_raises():
    try:
        extract_features(np.zeros(10), sample_rate=100.0)
    except ValueError:
        return
    raise AssertionError("信号过短时应当抛出 ValueError")


def test_build_feature_table_metadata():
    records = [
        {"file": "a", "condition": "normal", "sr": 100.0, "signal": np.random.randn(500)},
        {"file": "b", "condition": "fault", "sr": 100.0, "signal": np.random.randn(500)},
    ]
    table = build_feature_table(records, window_sec=1.0, step_sec=0.5)
    assert set(table["record"]) == {"a", "b"}
    assert set(table["label"]) == {"normal", "fault"}
    assert isinstance(table, pd.DataFrame)
