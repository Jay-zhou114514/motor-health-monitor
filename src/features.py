"""从振动信号中提取窗口特征。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.fft import rfft, rfftfreq


def _spectral_features(window: np.ndarray, sample_rate: float) -> dict[str, float]:
    """返回少量频域特征（谱质心、主频）。"""
    length = window.size
    spectrum = np.abs(rfft(window - window.mean())) ** 2
    frequencies = rfftfreq(length, d=1.0 / sample_rate)
    total = float(spectrum.sum())
    if total <= 0:
        return {"centroid_hz": 0.0, "dominant_hz": 0.0}
    centroid = float((frequencies * spectrum).sum() / total)
    dominant = float(frequencies[np.argmax(spectrum)])
    return {"centroid_hz": centroid, "dominant_hz": dominant}


def extract_features(
    signal: np.ndarray,
    sample_rate: float,
    window_sec: float = 1.0,
    step_sec: float = 0.5,
    record_id: str | None = None,
    label: str | None = None,
) -> pd.DataFrame:
    """把一维信号切成带重叠的窗口，为每个窗口计算时域/频域特征。

    返回的每一行代表一个窗口，列以 record 开头，方便之后合并标签。
    """
    window_size = max(2, int(round(sample_rate * window_sec)))
    step_size = max(1, int(round(sample_rate * step_sec)))
    if signal.size < window_size:
        raise ValueError("信号太短，无法切出完整窗口")

    rows: list[dict] = []
    for start in range(0, signal.size - window_size + 1, step_size):
        window = signal[start : start + window_size]
        mean = float(window.mean())
        std = float(window.std(ddof=1))
        rms = float(np.sqrt(np.mean(window**2)))
        peak = float(np.max(np.abs(window)))
        peak_to_peak = float(np.max(window) - np.min(window))
        crest = peak / rms if rms > 0 else float("nan")
        skewness = (
            float(np.mean((window - mean) ** 3) / std**3) if std > 0 else float("nan")
        )
        kurtosis = (
            float(np.mean((window - mean) ** 4) / std**4) - 3.0
            if std > 0
            else float("nan")
        )
        spectral = _spectral_features(window, sample_rate)

        row = {
            "record": record_id,
            "label": label,
            "window_start_s": round(start / sample_rate, 4),
            "mean": mean,
            "std": std,
            "rms": rms,
            "peak": peak,
            "peak_to_peak": peak_to_peak,
            "crest_factor": crest,
            "skewness": skewness,
            "kurtosis": kurtosis,
            **spectral,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_table(records: list[dict], **window_kwargs) -> pd.DataFrame:
    """为多条记录批量切窗，返回合并后的特征表。"""
    frames = []
    for record in records:
        frame = extract_features(
            signal=record["signal"],
            sample_rate=record["sr"],
            record_id=record["file"],
            label=record["condition"],
            **window_kwargs,
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
