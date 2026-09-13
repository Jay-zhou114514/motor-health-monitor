"""生成结果图（英文标签，方便放进英文申请材料）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _first_seconds(signal: np.ndarray, sample_rate: float, seconds: float = 2.0):
    count = min(signal.size, int(round(sample_rate * seconds)))
    return np.arange(count) / sample_rate, signal[:count]


def plot_waveforms(records_by_condition: dict[str, np.ndarray], sample_rate: float, path: Path) -> None:
    """画出正常/外圈故障/内圈故障的原始波形。"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    titles = {
        "normal": "Normal (baseline)",
        "outer_race_fault": "Outer race fault",
        "inner_race_fault": "Inner race fault",
    }
    for ax, condition in zip(axes, titles, strict=False):
        time, values = _first_seconds(records_by_condition[condition], sample_rate)
        ax.plot(time, values, linewidth=0.8)
        ax.set_ylabel("Acceleration (g)")
        ax.set_title(titles[condition])
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Raw vibration signals from the bearing dataset")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_space(
    table,
    true_series,
    prediction_series,
    method_name: str,
    path: Path,
) -> None:
    """用 RMS 与峭度画散点，展示真实标签与预测结果。"""
    fig, ax = plt.subplots(figsize=(7, 5))
    normal = table["label"] == "normal"
    fault = ~normal
    ax.scatter(
        table.loc[normal, "rms"],
        table.loc[normal, "kurtosis"],
        color="#2f80ed",
        alpha=0.7,
        label="True normal",
    )
    ax.scatter(
        table.loc[fault, "rms"],
        table.loc[fault, "kurtosis"],
        color="#eb5757",
        alpha=0.7,
        label="True fault",
    )
    # 名称必须与定义一致，否则容易读反：
    # 漏报 Missed detection (false negative) = 实际故障 & 预测正常
    # 误报 False alarm (false positive)     = 实际正常 & 预测异常
    missed_detection = true_series & ~prediction_series
    false_alarm = ~true_series & prediction_series
    if missed_detection.any():
        ax.scatter(
            table.loc[missed_detection, "rms"],
            table.loc[missed_detection, "kurtosis"],
            marker="X",
            s=120,
            facecolors="none",
            edgecolors="k",
            label="Missed detection (false negative)",
        )
    if false_alarm.any():
        ax.scatter(
            table.loc[false_alarm, "rms"],
            table.loc[false_alarm, "kurtosis"],
            marker="o",
            s=150,
            facecolors="none",
            edgecolors="k",
            label="False alarm (false positive)",
        )
    ax.set_xlabel("RMS (g)")
    ax.set_ylabel("Kurtosis")
    ax.set_title(f"Feature space result — {method_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric_comparison(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    path: Path,
) -> None:
    """对比两种方法的精确率/召回率/F1。"""
    labels = ["Precision", "Recall", "F1"]
    values_a = [metrics_a["precision"], metrics_a["recall"], metrics_a["f1"]]
    values_b = [metrics_b["precision"], metrics_b["recall"], metrics_b["f1"]]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, values_a, width, label="A: 3-sigma threshold")
    ax.bar(x + width / 2, values_b, width, label="B: Mahalanobis distance")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Detection performance comparison (window level)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost_curves(
    curve_a,
    curve_b,
    path: Path,
    label_a: str = "A: 3-sigma threshold",
    label_b: str = "B: Mahalanobis distance",
) -> None:
    """画出两种方法的总代价随报警线变化的曲线（横轴已归一到报警线序号）。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve_a.index, curve_a["cost"], marker=".", label=label_a)
    ax.plot(curve_b.index, curve_b["cost"], marker=".", label=label_b)
    ax.set_xlabel("Alarm threshold setting (index)")
    ax.set_ylabel("Total cost (FP + ratio x FN)")
    ax.set_title("Cost of false alarms vs. missed faults across thresholds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_envelope_spectra(
    spectra: dict[str, tuple],
    char_freqs: dict[str, float],
    path: Path,
    freq_limit_hz: float = 400.0,
) -> None:
    """画几个工况的包络谱，并标出故障特征频率（BPFO/BPFI）的位置。

    spectra: {工况名: (频率轴, 功率谱)}；功率谱取对数便于观察弱峰。
    """
    from matplotlib.lines import Line2D

    colors = {"BPFO": "#eb5757", "BPFI": "#f2994a"}
    fig, axes = plt.subplots(len(spectra), 1, figsize=(9, 2.6 * len(spectra)), sharex=True)
    if len(spectra) == 1:
        axes = [axes]
    for ax, (name, (frequencies, spectrum)) in zip(axes, spectra.items(), strict=True):
        band = frequencies <= freq_limit_hz
        ax.semilogy(frequencies[band], spectrum[band] + 1e-30, linewidth=0.8)
        for key, color in colors.items():
            freq = char_freqs.get(key)
            if freq and freq <= freq_limit_hz:
                ax.axvline(freq, color=color, linestyle="--", alpha=0.8)
        ax.set_ylabel("Envelope power")
        ax.set_title(name)
    handles = [
        Line2D([0], [0], color=color, linestyle="--", label=f"{key} = {char_freqs[key]:.1f} Hz")
        for key, color in colors.items()
        if key in char_freqs
    ]
    axes[-1].set_xlabel("Frequency (Hz)")
    axes[-1].legend(handles=handles)
    fig.suptitle("Envelope spectra: fault characteristic frequencies stand out")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

