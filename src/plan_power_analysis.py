"""V1.4 规划用功效分析（蒙特卡洛，文件级配对设计）。

依据 statistical-power skill 的方法：
- 效应量来自 EXP-V1-06 的实测差异（先导估计，需按不确定性解读）；
- 用蒙特卡洛估计功效，并报告 95% 蒙特卡洛置信区间；
- 给出敏感性分析（不同效应量 / 样本量下的功效），而不是单一数字；
- 分析单元是"正常文件"（文件级配对），因此不涉及窗口伪重复问题。

输出：
    docs/plans/power_analysis.csv
    docs/plans/figures/power_curves.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = ROOT / "docs" / "plans"
FIGURE_DIR = PLAN_DIR / "figures"

N_SIMS = 5000
ALPHA = 0.05
TARGET_POWER = 0.80
N_GRID = [3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 80]


def paired_power(dz: float, n: int, n_sims: int = N_SIMS, seed: int = 0) -> tuple[float, float, float]:
    """配对 t 检验（单样本等价）的功效，蒙特卡洛估计 + 95% CI。"""
    rng = np.random.default_rng(seed)
    differences = rng.normal(dz, 1.0, size=(n_sims, n))
    mean = differences.mean(axis=1)
    sd = differences.std(axis=1, ddof=1)
    t_stat = mean / (sd / np.sqrt(n))
    # 双侧 p 值
    from scipy.stats import t as t_dist

    p_values = 2 * t_dist.sf(np.abs(t_stat), df=n - 1)
    power = float((p_values < ALPHA).mean())
    standard_error = np.sqrt(power * (1 - power) / n_sims)
    return (
        power,
        max(0.0, power - 1.96 * standard_error),
        min(1.0, power + 1.96 * standard_error),
    )


def required_n(dz: float, target: float = TARGET_POWER, max_n: int = 200) -> int | None:
    for n in range(3, max_n + 1):
        power, _, _ = paired_power(dz, n, n_sims=2000, seed=1)
        if power >= target:
            return n
    return None


def main() -> None:
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 来自 EXP-V1-06 的实测差异（先导估计）----
    h1_differences = np.array([2.9274 - 1.6105, 0.7337 - 0.7232, 1.8102 - 1.8645])
    h2_differences = np.array([1.576 - 0.311, 0.469 - 0.419, 0.797 - 0.253])

    def dz(values: np.ndarray) -> float:
        return float(values.mean() / values.std(ddof=1))

    dz_h1 = dz(h1_differences)
    dz_h2 = dz(h2_differences)

    rows: list[dict] = []
    for label, effect in (("H1_geometry", dz_h1), ("H2_regularization", dz_h2)):
        for n in N_GRID:
            power, low, high = paired_power(effect, n)
            rows.append(
                {
                    "hypothesis": label,
                    "dz": round(effect, 3),
                    "n_normal_files": n,
                    "power": round(power, 3),
                    "power_ci_low": round(low, 3),
                    "power_ci_high": round(high, 3),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(PLAN_DIR / "power_analysis.csv", index=False)

    sensitivity_effects = [0.3, 0.4, dz_h1, 0.7, 1.0]
    sensitivity = {
        round(effect, 3): required_n(effect) for effect in sensitivity_effects
    }

    print("实测（先导）效应量：")
    print(f"  H1 几何归因：差异 {np.round(h1_differences, 3)}，均值 {h1_differences.mean():.3f}，"
          f"SD {h1_differences.std(ddof=1):.3f}，dz = {dz_h1:.3f}")
    print(f"  H2 正则化：  差异 {np.round(h2_differences, 3)}，均值 {h2_differences.mean():.3f}，"
          f"SD {h2_differences.std(ddof=1):.3f}，dz = {dz_h2:.3f}")
    print()
    print("达到 80% 功效所需正常文件数（配对 t 检验，α=0.05 双侧）：")
    for effect, n in sensitivity.items():
        print(f"  dz = {effect:.3f} → n = {n}")
    print()
    print("当前 3 个正常文件下的功效：")
    for label, effect in (("H1", dz_h1), ("H2", dz_h2)):
        power, low, high = paired_power(effect, 3)
        print(f"  {label}: power = {power:.3f} (95% MC CI {low:.3f}-{high:.3f})")

    # ---- 图 ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, effect, color in (
        ("H1 geometry (dz=%.2f)" % dz_h1, dz_h1, "tab:red"),
        ("H2 regularization (dz=%.2f)" % dz_h2, dz_h2, "tab:blue"),
    ):
        ns = np.arange(3, 61)
        powers = [paired_power(effect, int(n), n_sims=1000, seed=2)[0] for n in ns]
        axes[0].plot(ns, powers, color=color, label=label)
    axes[0].axhline(TARGET_POWER, linestyle="--", color="gray", label="80% power")
    axes[0].set_xlabel("Number of normal files")
    axes[0].set_ylabel("Power")
    axes[0].set_title("Power vs number of normal files (pilot effect sizes)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for effect in [0.3, 0.5, 0.7, 1.0]:
        ns = np.arange(3, 61)
        powers = [paired_power(effect, int(n), n_sims=1000, seed=3)[0] for n in ns]
        axes[1].plot(ns, powers, label=f"dz={effect}")
    axes[1].axhline(TARGET_POWER, linestyle="--", color="gray")
    axes[1].set_xlabel("Number of normal files")
    axes[1].set_ylabel("Power")
    axes[1].set_title("Sensitivity analysis")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "power_curves.png", dpi=150)
    plt.close(figure)

    print(f"\n已保存：{PLAN_DIR / 'power_analysis.csv'}")
    print(f"已保存：{FIGURE_DIR / 'power_curves.png'}")


if __name__ == "__main__":
    main()
