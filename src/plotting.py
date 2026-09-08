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
    false_alarm = true_series & ~prediction_series
    missed = ~true_series & prediction_series
    if false_alarm.any():
        ax.scatter(
            table.loc[false_alarm, "rms"],
            table.loc[false_alarm, "kurtosis"],
            marker="X",
            s=120,
            facecolors="none",
            edgecolors="k",
            label="Missed (false negative)",
        )
    if missed.any():
        ax.scatter(
            table.loc[missed, "rms"],
            table.loc[missed, "kurtosis"],
            marker="o",
            s=150,
            facecolors="none",
            edgecolors="k",
            label="False alarm",
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
