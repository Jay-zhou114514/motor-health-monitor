"""EXP-V1-11：阈值估计噪声下限与样本量处方。

预注册：experiments/EXP-V1-11-preregistration.md（先于本脚本运行提交）

输出：
- outputs/exp_v1_11_curve.csv
- outputs/exp_v1_11_run.log
- outputs/figures/exp_v1_11_sample_size_scaling.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from config import FIGURES_DIR, OUTPUT_DIR
from exp_v1_08_baselines import QUANTILE, load_healthy_records
from exp_v1_10_selection_uncertainty import BATCH, TEST_FILES, to_arrays

N_SIZES = [10, 20, 40, 80, 160, 320, 640]
R_REPEAT = 150
MODAL_CFG = (200, 0.5)


class Tee:
    def __init__(self, path: Path):
        self.handle = open(path, "w", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        self.handle.write(text)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.handle.flush()


def fit_and_score(train: np.ndarray, test: np.ndarray) -> tuple[float, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=1)
    std = np.where(std == 0, 1.0, std)
    model = IsolationForest(
        n_estimators=MODAL_CFG[0], max_samples=MODAL_CFG[1],
        random_state=0, n_jobs=1,
    ).fit((train - mean) / std)
    train_scores = -model.score_samples((train - mean) / std)
    threshold = float(np.quantile(train_scores, QUANTILE))
    test_scores = -model.score_samples((test - mean) / std)
    return threshold, test_scores, train_scores


def run_variant(pool_windows: np.ndarray, file_blocks: list[np.ndarray],
                test_block: np.ndarray, n_train: int, rng) -> list[dict]:
    rows = []
    for repeat in range(R_REPEAT):
        if pool_windows is None:
            # V2 记录级：以整文件为单位抽到 >= n_train
            chosen: list[np.ndarray] = []
            total = 0
            while total < n_train:
                chosen.append(file_blocks[rng.integers(0, len(file_blocks))])
                total += len(chosen[-1])
            train = np.vstack(chosen)
        else:
            index = rng.integers(0, len(pool_windows), size=n_train)
            train = pool_windows[index]
        threshold, test_scores, train_scores = fit_and_score(train, test_block)
        pct = np.searchsorted(np.sort(train_scores), test_scores, side="right") / len(train_scores)
        rows.append(
            {
                "n_train": int(n_train),
                "n_train_actual": int(len(train)),
                "repeat": repeat,
                "threshold": threshold,
                "test_fp": float((test_scores > threshold).mean()),
                "test_pct_mean": float(pct.mean()),
            }
        )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUTPUT_DIR / "exp_v1_11_run.log")

    arrays = to_arrays(load_healthy_records()[BATCH])
    pool_names = [name for name in sorted(arrays) if name not in TEST_FILES]
    pool_windows = np.vstack([arrays[name] for name in pool_names])
    file_blocks = [arrays[name] for name in pool_names]
    test_block = np.vstack([arrays[name] for name in TEST_FILES])
    print(f"训练池：{len(pool_names)} 个文件 / {len(pool_windows)} 个窗口；"
          f"固定测试集：{len(test_block)} 个窗口")
    print(f"扫描 n_train = {N_SIZES}，每个规模 R = {R_REPEAT}")

    rows: list[dict] = []
    for variant, label in ((0, "V1_window"), (1, "V2_recording")):
        rng = np.random.default_rng(20260918 + variant)
        for n_train in N_SIZES:
            print(f"  {label}: n_train = {n_train}", flush=True)
            for row in run_variant(
                pool_windows if variant == 0 else None,
                file_blocks, test_block, n_train, rng,
            ):
                row["variant"] = label
                rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "exp_v1_11_curve.csv", index=False)

    q = QUANTILE
    summary = (frame.groupby(["variant", "n_train"])
               .agg(sd_fp=("test_fp", "std"),
                    mean_fp=("test_fp", "mean"),
                    n_repeat=("test_fp", "size"))
               .reset_index())
    summary["sd_ideal"] = np.sqrt(q * (1 - q) / summary["n_train"])
    summary["ratio_obs_ideal"] = summary["sd_fp"] / summary["sd_ideal"]
    summary["n_eff"] = q * (1 - q) / summary["sd_fp"] ** 2
    summary["efficiency"] = summary["n_eff"] / summary["n_train"]
    summary.to_csv(OUTPUT_DIR / "exp_v1_11_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== 结果 ===")
    print(summary.round(4).to_string(index=False))

    print("\n=== 预注册判据 ===")
    for label in ("V1_window", "V2_recording"):
        sub = summary[summary["variant"] == label]
        rho, pvalue = stats.spearmanr(sub["n_train"], sub["sd_fp"])
        print(f"{label}: Spearman(n_train, SD) = {rho:.3f} (p = {pvalue:.4f}) -> "
              f"{'P1 成立' if rho < 0 else 'P1 不成立'}")
    ratios = summary["ratio_obs_ideal"]
    p2 = bool(((ratios >= 0.5) & (ratios <= 5)).mean() >= 0.7)
    print(f"P2：SD_obs / SD_ideal 落在 [0.5, 5] 的比例 = "
          f"{((ratios >= 0.5) & (ratios <= 5)).mean():.2f} -> {'成立' if p2 else '不成立'}")
    v1 = summary[summary["variant"] == "V1_window"].set_index("n_train")["sd_fp"]
    v2 = summary[summary["variant"] == "V2_recording"].set_index("n_train")["sd_fp"]
    p3 = bool((v2.reindex(v1.index) > v1).mean() >= 0.7)
    print(f"P3：V2 > V1 的规模占比 = {(v2.reindex(v1.index) > v1).mean():.2f} -> "
          f"{'成立' if p3 else '不成立'}")
    for epsilon in (0.01, 0.02, 0.05):
        needed_ideal = q * (1 - q) / epsilon ** 2
        print(f"  ε = {epsilon:.2f} → SD_ideal 处方 n ≥ {needed_ideal:.0f} 个窗口")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, marker in (("V1_window", "o"), ("V2_recording", "s")):
        sub = summary[summary["variant"] == label]
        axes[0].plot(sub["n_train"], sub["sd_fp"] * 100, marker=marker, label=label)
    grid = np.unique(np.round(np.logspace(np.log10(min(N_SIZES)), np.log10(max(N_SIZES)), 60)))
    axes[0].plot(grid, np.sqrt(q * (1 - q) / grid) * 100, "k--", label="ideal  sqrt(q(1-q)/n)")
    axes[0].axhline(1.0, color="gray", linestyle=":", label="target 1 pp")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Training windows (nominal)")
    axes[0].set_ylabel("SD of reported false-alarm rate (pp)")
    axes[0].set_title("EXP-V1-11: noise floor vs training size")
    axes[0].legend(fontsize=8)

    for label, marker in (("V1_window", "o"), ("V2_recording", "s")):
        sub = summary[summary["variant"] == label]
        axes[1].plot(sub["n_train"], sub["efficiency"], marker=marker, label=label)
    axes[1].axhline(1.0, color="k", linestyle="--", label="ideal (n_eff = n)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Training windows (nominal)")
    axes[1].set_ylabel("Effective / nominal sample size")
    axes[1].set_title("How far from the ideal i.i.d. assumption?")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "exp_v1_11_sample_size_scaling.png", dpi=150)
    plt.close(figure)
    print("\n已保存曲线与图。")


if __name__ == "__main__":
    main()