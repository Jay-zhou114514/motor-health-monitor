"""Fig. 1：评价不确定性的三个面（概念图）。

不使用外部字体/图片；全部用 matplotlib 基本图元绘制，便于修改与复现。
输出：docs/figures/fig1_three_faces.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_TRAIN = "#8fb8de"
C_TEST = "#f2b880"
C_ALARM = "#d65f5f"
C_OK = "#5f9e6e"


def bearing(ax, x, y, w=0.42, h=0.42, label=None, fmt="train"):
    color = C_TRAIN if fmt == "train" else C_TEST
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.0, edgecolor="#444444", facecolor=color, zorder=3))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7, zorder=4)


def panel_frame(ax, title, subtitle):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.1, 5.55, title, fontsize=11, fontweight="bold", va="top")
    ax.text(0.1, 4.95, subtitle, fontsize=8.5, color="#444444", va="top")


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))

    # ---------------- (a) Scope ----------------
    ax = axes[0]
    panel_frame(ax, "(a) Scope \u2014 are physical units held apart?",
                "same detector, same data, different split")
    ax.text(0.1, 4.35, "within-bearing", fontsize=9, fontweight="bold")
    for i in range(4):
        bearing(ax, 0.3 + i * 1.05, 3.55, label=f"#{i+1}")
    bearing(ax, 4.6, 3.55, label="#1", fmt="test")
    bearing(ax, 5.65, 3.55, label="#2", fmt="test")
    ax.text(0.3, 3.25, "train: recordings 1\u201316 of each bearing", fontsize=7.5, color="#333333")
    ax.text(4.6, 3.25, "test: recordings 17\u201320 of the SAME bearings",
            fontsize=7.5, color="#a05020")
    ax.text(4.9, 2.75, "reported false-alarm rate: 0.00%", fontsize=9.5,
            fontweight="bold", color=C_OK)

    ax.plot([0.3, 9.7], [2.35, 2.35], color="#bbbbbb", linewidth=1)
    ax.text(0.1, 2.02, "bearing-level holdout", fontsize=9, fontweight="bold")
    for i in range(5):
        bearing(ax, 0.3 + i * 1.05, 1.20, label=f"#{i+1}")
    beta = FancyArrowPatch((5.2, 1.45), (6.4, 1.45), arrowstyle="-|>",
                           mutation_scale=14, color="#a05020", linewidth=1.6)
    ax.add_patch(beta)
    bearing(ax, 6.6, 1.20, label="#6", fmt="test")
    ax.text(5.25, 0.86, "held-out bearing", fontsize=7.5, color="#a05020")
    ax.text(0.3, 1.90, "train: all recordings of five bearings", fontsize=7.5, color="#333333")
    ax.text(0.3, 0.35, "reported false-alarm rate: 40.63%  (folds 0.4\u201397.1%)",
            fontsize=9.5, fontweight="bold", color=C_ALARM)

    # ---------------- (b) Units ----------------
    ax = axes[1]
    panel_frame(ax, "(b) Units \u2014 how many independent bearings?",
                "sample size fixed at 20 recordings; one operating condition")
    ax.text(0.1, 4.35, "1 training bearing", fontsize=9, fontweight="bold")
    for i in range(5):
        fmt = "train" if i == 0 else "none"
        if fmt == "train":
            bearing(ax, 0.3 + i * 1.05, 3.55, label="20")
        else:
            ax.add_patch(Rectangle((0.3 + i * 1.05, 3.55), 0.42, 0.42, linewidth=0.6,
                                   edgecolor="#cccccc", facecolor="none", linestyle=":"))
    ax.text(4.4, 3.66, "SD \u2248 23\u201338 pp", fontsize=9, color=C_ALARM, fontweight="bold")
    ax.text(0.3, 3.22, "mean FP at k=1: 54\u201358% (XJTU-SY), 17\u201321% (PRONOSTIA)",
            fontsize=7.5, color="#333333")

    ax.plot([0.3, 9.7], [2.85, 2.85], color="#bbbbbb", linewidth=1)
    ax.text(0.1, 2.52, "4 training bearings", fontsize=9, fontweight="bold")
    for i in range(5):
        if i < 4:
            bearing(ax, 0.3 + i * 1.05, 1.72, label="5")
        else:
            ax.add_patch(Rectangle((0.3 + i * 1.05, 1.72), 0.42, 0.42, linewidth=0.6,
                                   edgecolor="#cccccc", facecolor="none", linestyle=":"))
    ax.text(4.4, 1.83, "SD \u2248 8\u201314 pp", fontsize=9, color=C_OK, fontweight="bold")
    ax.text(0.3, 1.36, "test set: the same held-out bearing in both cases", fontsize=7.5,
            color="#333333")
    ax.text(0.3, 0.72, "3\u03c3 RMS 6/6 and Isolation Forest 6/6 support the effect;",
            fontsize=8, color="#333333")
    ax.text(0.3, 0.40, "Mahalanobis saturates at k=1 (67\u2013100% FP) \u2192 not testable",
            fontsize=8, color="#a05020")

    # ---------------- (c) Definition ----------------
    ax = axes[2]
    panel_frame(ax, "(c) Definition \u2014 where does 'normal' end?",
                "run-to-failure data carry no onset label")
    ax.plot([0.5, 9.5], [3.3, 3.3], color="#888888", linewidth=2)
    ax.plot([0.5, 4.6], [3.3, 3.3], color=C_OK, linewidth=5, solid_capstyle="butt")
    ax.plot([4.6, 9.5], [3.3, 3.3], color=C_ALARM, linewidth=5, solid_capstyle="butt")
    ax.text(0.5, 3.62, "healthy", fontsize=9, color=C_OK, fontweight="bold")
    ax.text(8.6, 3.62, "degraded", fontsize=9, color=C_ALARM, fontweight="bold")
    for x, lab in ((2.6, "5%"), (4.6, "10%"), (6.4, "20%")):
        ax.plot([x, x], [2.95, 3.65], color="#333333", linestyle="--", linewidth=1)
        ax.text(x, 2.72, lab, ha="center", fontsize=8)
    ax.text(0.5, 2.30, "boundary chosen by the analyst (no ground-truth onset)", fontsize=8,
            color="#333333")
    ax.text(0.5, 1.75, "SD of the reported false-alarm rate across five rules:",
            fontsize=8.5)
    ax.text(0.8, 1.30, "\u2022 PRONOSTIA (17 bearings, fixed inclusion set):  2.40 pp",
            fontsize=8.5, color="#333333")
    ax.text(0.8, 0.95, "\u2022 XJTU-SY (inclusion set changes with the rule):  5.19 pp",
            fontsize=8.5, color="#a05020")
    ax.text(0.8, 0.50, "same rule can bind on the floor (short-life) or the cap (long-life)",
            fontsize=7.5, color="#333333")

    fig.suptitle("Three faces of evaluation uncertainty in healthy-data-only bearing "
                 "anomaly detection", fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / "fig1_three_faces.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()