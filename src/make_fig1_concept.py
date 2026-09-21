"""Fig. 1：评价不确定性的三个面（概念图）。

按 nature-figure 规范重做：
- **文字一律不放进填充图形内**（上一版 68 处 text-fill-edge 碰撞的根因）
- 2×2 布局，双栏期刊宽度 183 mm
- 可编辑矢量文字（svg.fonttype=none, pdf.fonttype=42）
- 调用渲染期面板对齐门
- 输出 PNG(600dpi) + PDF + SVG
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7,
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.6,
})

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, r"<WORKDIR>\.codex\skills\nature-figure\scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_TRAIN = "#9fc3e0"
C_TEST = "#f2c9a0"
C_ALARM = "#c85a5a"
C_OK = "#5f9e6e"
C_DIM = "#d9d9d9"


def box(ax, x, y, w, h, color):
    """只画方框，不画方框内的任何文字。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.01,rounding_size=0.06",
                                linewidth=0.8, edgecolor="#555555",
                                facecolor=color, zorder=3))


def frame(ax, tag, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.05, 5.75, tag, fontsize=8, fontweight="bold", va="top")
    ax.text(0.75, 5.75, title, fontsize=7.5, va="top")


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ---------- (a) Scope ----------
    frame(ax_a, "(a)", "Scope: are physical units held apart?")
    ax_a.text(0.05, 4.85, "within-bearing", fontsize=7, fontweight="bold")
    for i in range(5):
        box(ax_a, 0.05 + i * 1.05, 3.95, 0.85, 0.62, C_TRAIN)
    for i in range(2):
        box(ax_a, 5.30 + i * 1.05, 3.95, 0.85, 0.62, C_TEST)
    ax_a.text(0.05, 3.52, "train: recordings 1-16", fontsize=6.5, color="#333333")
    ax_a.text(5.30, 3.50, "test = recordings 17-20\nof the SAME bearings", fontsize=6.5, color="#a05020")
    ax_a.text(0.05, 2.95, "reported false-alarm rate  0.00%", fontsize=8,
              fontweight="bold", color=C_OK)

    ax_a.plot([0.05, 9.6], [2.55, 2.55], color="#cccccc", linewidth=0.8)
    ax_a.text(0.05, 2.15, "bearing-level holdout", fontsize=7, fontweight="bold")
    for i in range(5):
        box(ax_a, 0.05 + i * 1.05, 1.30, 0.85, 0.62, C_TRAIN)
    arrow = FancyArrowPatch((5.45, 1.61), (6.25, 1.61), arrowstyle="-|>",
                            mutation_scale=10, color="#a05020", linewidth=1.2)
    ax_a.add_patch(arrow)
    box(ax_a, 6.40, 1.30, 0.85, 0.62, C_TEST)
    ax_a.text(0.05, 0.98, "train: all recordings of five bearings", fontsize=6.5, color="#333333")
    ax_a.text(6.05, 0.98, "held-out bearing", fontsize=6.5, color="#a05020")
    ax_a.text(0.05, 0.30, "reported false-alarm rate  40.63%  (folds 0.4-97.1%)",
              fontsize=8, fontweight="bold", color=C_ALARM)

    # ---------- (b) Units ----------
    frame(ax_b, "(b)", "Units: how many independent bearings?")
    ax_b.text(0.05, 4.85, "1 training bearing  (20 recordings)", fontsize=7, fontweight="bold")
    box(ax_b, 0.05, 3.95, 0.85, 0.62, C_TRAIN)
    for i in range(1, 5):
        ax_b.add_patch(Rectangle((0.05 + i * 1.05, 3.95), 0.85, 0.62, linewidth=0.5,
                                 edgecolor=C_DIM, facecolor="none", linestyle=":"))
    ax_b.text(0.05, 3.50, "mean FP 17-58%   SD 23-38 pp", fontsize=7, color=C_ALARM)
    ax_b.plot([0.05, 9.6], [3.10, 3.10], color="#cccccc", linewidth=0.8)
    ax_b.text(0.05, 2.70, "4 training bearings  (same 20 recordings)", fontsize=7, fontweight="bold")
    for i in range(4):
        box(ax_b, 0.05 + i * 1.05, 1.85, 0.85, 0.62, C_TRAIN)
    ax_b.add_patch(Rectangle((0.05 + 4 * 1.05, 1.85), 0.85, 0.62, linewidth=0.5,
                             edgecolor=C_DIM, facecolor="none", linestyle=":"))
    ax_b.text(0.05, 1.40, "mean FP 3-14%   SD 8-14 pp", fontsize=7, color=C_OK)
    ax_b.text(0.05, 0.70, "3-sigma RMS 6/6 and Isolation Forest 6/6 support it",
              fontsize=6.5, color="#333333")
    ax_b.text(0.05, 0.30, "Mahalanobis saturates at k=1 (67-100% FP): not testable",
              fontsize=6.5, color="#a05020")

    # ---------- (c) Definition ----------
    frame(ax_c, "(c)", "Definition: where does 'normal' end?")
    ax_c.plot([0.05, 9.6], [3.80, 3.80], color="#888888", linewidth=1.5)
    ax_c.plot([0.05, 4.30], [3.80, 3.80], color=C_OK, linewidth=4, solid_capstyle="butt")
    ax_c.plot([4.30, 9.60], [3.80, 3.80], color=C_ALARM, linewidth=4, solid_capstyle="butt")
    ax_c.text(0.05, 4.05, "healthy", fontsize=7, color=C_OK, fontweight="bold")
    ax_c.text(8.90, 4.05, "degraded", fontsize=7, color=C_ALARM, fontweight="bold")
    for x, lab in ((2.30, "5%"), (4.30, "10%"), (6.30, "20%")):
        ax_c.plot([x, x], [3.35, 4.25], color="#333333", linestyle="--", linewidth=0.8)
        ax_c.text(x - 0.25, 2.95, lab, fontsize=7)
    ax_c.text(0.05, 2.35, "run-to-failure data carry no onset label:", fontsize=6.5)
    ax_c.text(0.05, 1.95, "the boundary is chosen by the analyst", fontsize=6.5, color="#333333")
    ax_c.text(0.05, 1.25, "SD of the reported rate across five rules", fontsize=7, fontweight="bold")
    ax_c.text(0.35, 0.85, "PRONOSTIA (inclusion set fixed):  2.40 pp", fontsize=6.5, color="#333333")
    ax_c.text(0.35, 0.45, "XJTU-SY (inclusion set varies):  5.19 pp", fontsize=6.5, color="#a05020")

    # ---------- (d) Checklist ----------
    frame(ax_d, "(d)", "Reporting checklist")
    items = [
        "1  Report the number of training windows behind the threshold,",
        "    not the dataset size.",
        "2  Report resampling variability from at least two sources:",
        "    split composition and random seed.",
        "3  Report the tie rate when validation resolution is coarser",
        "    than the false-alarm level of interest.",
        "4  Distinguish new recordings from repeated sampling.",
        "5  State the number of independent units, and whether held out.",
    ]
    y = 5.00
    for line in items:
        ax_d.text(0.05, y, line, fontsize=6.5, va="top")
        y -= 0.52

    fig.tight_layout(rect=(0, 0, 1, 1))
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig)

    png = OUT / "fig1_three_faces.png"
    pdf = OUT / "fig1_three_faces.pdf"
    svg = OUT / "fig1_three_faces.svg"
    fig.savefig(png, dpi=600)              # 600 dpi 栅格
    fig.savefig(pdf, format="pdf")         # 矢量：投稿主件
    fig.savefig(svg, format="svg")         # 矢量：可编辑
    plt.close(fig)
    for pth in (png, pdf, svg):
        print(f"saved: {pth}")


if __name__ == "__main__":
    main()