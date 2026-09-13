"""包络谱故障类型诊断。

思路（轴承故障诊断的经典方法）：
1. 轴承局部故障每次被滚动体撞一下，会产生一串冲击衰减振荡；
2. 这些冲击的重复频率就是故障特征频率（外圈 BPFO、内圈 BPFI 等），
   但在原始频谱里被高频共振淹没，直接看不出来；
3. 对信号做希尔伯特变换取包络，再对包络做频谱（包络谱），
   故障特征频率及其谐波就会变成明显的峰；
4. 比较各特征频率处的"峰值/局部本底"倍数，得分最高的即诊断结论。

这里只用可解释的规则，不训练任何模型——和本项目"低成本、可解释"的定位一致。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

DEFAULT_HARMONICS = 3
DEFAULT_HALF_BAND_HZ = 4.0
DEFAULT_SEARCH_HZ = 40.0
NORMAL_MARGIN = 10.0  # 诊断得分超过正常基线多少倍才判为故障


def envelope_spectrum(
    signal: np.ndarray, sample_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """返回包络谱的（频率轴，功率谱）。"""
    envelope = np.abs(hilbert(signal - signal.mean()))
    spectrum = np.abs(np.fft.rfft(envelope - envelope.mean())) ** 2
    frequencies = np.fft.rfftfreq(envelope.size, d=1.0 / sample_rate)
    return frequencies, spectrum


def fault_feature_score(
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    char_freq: float,
    harmonics: int = DEFAULT_HARMONICS,
    half_band_hz: float = DEFAULT_HALF_BAND_HZ,
    search_hz: float = DEFAULT_SEARCH_HZ,
) -> float:
    """特征频率处的平均"峰值 / 局部本底"倍数（考虑前 n 次谐波）。

    本底取目标频率附近 ±search_hz 的中位数，代表"没有故障时这附近
    大概是什么水平"；峰值明显高于本底（如 >10 倍）才像是真正的故障线。
    """
    score = 0.0
    used = 0
    for k in range(1, harmonics + 1):
        target = char_freq * k
        if target >= frequencies[-1]:
            break
        peak_band = (frequencies > target - half_band_hz) & (
            frequencies < target + half_band_hz
        )
        local_band = (frequencies > target - search_hz) & (
            frequencies < target + search_hz
        )
        peak = float(spectrum[peak_band].max())
        floor = float(np.median(spectrum[local_band]))
        score += peak / max(floor, 1e-30)
        used += 1
    return score / max(1, used)


FAULT_KEYS = ("BPFO", "BPFI", "FTF", "BSF")
FAULT_LABELS = {
    "BPFO": "outer_race_fault",
    "BPFI": "inner_race_fault",
    "FTF": "cage_fault",
    "BSF": "ball_fault",
}


class EnvelopeDiagnoser:
    """先在正常数据上估计"正常也会有"的得分水平，再诊断新信号。

    规则：各特征频率得分中最高者若超过正常基线最高得分的
    NORMAL_MARGIN 倍，就诊断该特征对应的故障类型；否则判为正常。
    """

    def __init__(
        self,
        margin: float = NORMAL_MARGIN,
        harmonics: int = DEFAULT_HARMONICS,
    ) -> None:
        self.margin = margin
        self.harmonics = harmonics
        self.baseline_score_ = float("nan")

    def _scores(
        self, signal: np.ndarray, sample_rate: float, char_freqs: dict[str, float]
    ) -> dict[str, float]:
        frequencies, spectrum = envelope_spectrum(signal, sample_rate)
        return {
            key: fault_feature_score(
                frequencies, spectrum, char_freqs[key], harmonics=self.harmonics
            )
            for key in FAULT_KEYS
            if np.isfinite(char_freqs.get(key, np.nan))
        }

    def fit(
        self, normal_records: list[dict], max_seconds: float = 3.0
    ) -> "EnvelopeDiagnoser":
        """normal_records 每项需含 signal/sr 与特征频率（BPFO 等）。"""
        if not normal_records:
            raise ValueError("至少需要一条正常记录来估计基线")
        max_scores = [
            max(self._scores(self._trim(r, max_seconds), r["sr"], r).values())
            for r in normal_records
        ]
        self.baseline_score_ = float(max(max_scores))
        return self

    @staticmethod
    def _trim(record: dict, max_seconds: float) -> np.ndarray:
        count = min(record["signal"].size, int(record["sr"] * max_seconds))
        return record["signal"][:count]

    def diagnose(
        self,
        signal: np.ndarray,
        sample_rate: float,
        char_freqs: dict[str, float],
    ) -> dict:
        """返回各特征得分、最高得分与诊断结论（normal 或某故障类型）。"""
        scores = self._scores(signal, sample_rate, char_freqs)
        best_key = max(scores, key=scores.get)
        best_score = scores[best_key]
        limit = self.baseline_score_ * self.margin
        predicted = FAULT_LABELS[best_key] if best_score > limit else "normal"
        return {
            "scores": scores,
            "best_key": best_key,
            "best_score": best_score,
            "limit": limit,
            "predicted": predicted,
        }
