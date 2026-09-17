"""EXP-V2-01：跨数据体系下的误报率不确定性缩放（S1 重采样 vs S2 真实新增）。

预注册：experiments/EXP-V2-01-preregistration.md + 修订 1/2/3（均先于运行提交）。

核心问题：**名义样本量 n 的增长，是否等价于有效信息量的增长？**
- S1：n 增长来自同一有限总体内的有放回重采样
- S2：n 增长来自真实新增的不同记录
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from config import FIGURES_DIR, OUTPUT_DIR
from exp_v1_08_baselines import QUANTILE
from paderborn_data import DEFAULT_CONDITION, FEATURES, load_healthy_dataset

N_SIZES = [5, 10, 20, 40, 60, 80, 96]
R_REPEAT = 150
MODAL_CFG = (200, 0.5)
TEST_RECORD_NUMBERS = (17, 18, 19, 20)
S1_SUBSET_SIZE = 20
RANDOM_SEED = 0


class Tee:
    def __init__(self, path: Path):
        self.handle = open(path, "w", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        self.handle.write(text)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.handle.flush()


def fit_threshold_scores(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=1)
    std = np.where(std == 0, 1.0, std)
    standardized_train = (train - mean) / std
    model = IsolationForest(
        n_estimators=MODAL_CFG[0], max_samples=MODAL_CFG[1],
        random_state=RANDOM_SEED, n_jobs=1,
    ).fit(standardized_train)
    train_scores = -model.score_samples(standardized_train)
    threshold = float(np.quantile(train_scores, QUANTILE))
    test_scores = -model.score_samples((test - mean) / std)
    return threshold, train_scores, test_scores


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUTPUT_DIR / "exp_v2_01_run.log")

    arrays, detail = load_healthy_dataset(DEFAULT_CONDITION)
    detail = detail.copy()
    detail["is_test"] = detail["record_number"].isin(TEST_RECORD_NUMBERS)
    test_names = sorted(detail[detail["is_test"]]["record_id"].tolist())
    pool_names = sorted(detail[~detail["is_test"]]["record_id"].tolist())
    print(f"Paderborn {DEFAULT_CONDITION}：{len(arrays)} 条记录")
    print(f"  固定测试集：{len(test_names)} 条（记录号 {TEST_RECORD_NUMBERS}，覆盖 "
          f"{detail[detail['is_test']]['bearing'].nunique()} 个轴承）")
    print(f"  训练池：{len(pool_names)} 条")
    binding = detail[~detail["is_test"]].groupby("bearing")["rms"].agg(["count", "mean"])
    print("  训练池按轴承：")
    print(binding.round(4).to_string())

    test_block = np.vstack([arrays[name] for name in test_names])

    # S1 的固定子集：训练池中编号最小的 S1_SUBSET_SIZE 条
    order = detail[~detail["is_test"]].sort_values(["bearing", "record_number"])
    subset_names = sorted(order["record_id"].tolist()[:S1_SUBSET_SIZE])
    print(f"  S1 固定子集：{len(subset_names)} 条")

    rows: list[dict] = []
    for arm in ("S1_resample", "S2_new_records"):
        rng = np.random.default_rng(20260918 + (0 if arm.startswith("S1") else 1))
        source = subset_names if arm.startswith("S1") else pool_names
        for n_train in N_SIZES:
            if arm.startswith("S2") and n_train > len(pool_names):
                continue
            print(f"  {arm}: n = {n_train}", flush=True)
            for repeat in range(R_REPEAT):
                replace = arm.startswith("S1")
                chosen = rng.choice(len(source), size=n_train, replace=replace)
                blocks = [arrays[source[i]] for i in chosen]
                train = np.vstack(blocks)
                threshold, train_scores, test_scores = fit_threshold_scores(train, test_block)
                pct = (np.searchsorted(np.sort(train_scores), test_scores, side="right")
                       / len(train_scores))
                rows.append({
                    "arm": arm,
                    "n_train": int(n_train),
                    "n_distinct_records": int(len({source[i] for i in chosen})),
                    "repeat": repeat,
                    "threshold": threshold,
                    "test_fp": float((test_scores > threshold).mean()),
                    "test_pct_mean": float(pct.mean()),
                })

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "exp_v2_01_curve.csv", index=False)

    q = QUANTILE
    summary = (frame.groupby(["arm", "n_train"])
               .agg(sd_fp=("test_fp", "std"),
                    mean_fp=("test_fp", "mean"),
                    n_distinct=("n_distinct_records", "mean"),
                    n_repeat=("test_fp", "size"))
               .reset_index())
    summary["sd_ideal"] = np.sqrt(q * (1 - q) / summary["n_train"])
    summary["n_eff"] = q * (1 - q) / summary["sd_fp"] ** 2
    summary["efficiency"] = summary["n_eff"] / summary["n_train"]
    summary.to_csv(OUTPUT_DIR / "exp_v2_01_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== 结果 ===")
    print(summary.round(4).to_string(index=False))

    print("\n=== 预注册判据 ===")
    s1 = summary[summary["arm"] == "S1_resample"].set_index("n_train")
    s2 = summary[summary["arm"] == "S2_new_records"].set_index("n_train")
    rho1, p1v = stats.spearmanr(s1.index, s1["n_eff"])
    rho2, p2v = stats.spearmanr(s2.index, s2["n_eff"])
    p1 = bool(rho2 > 0 and s2["n_eff"].iloc[-1] > 2 * s2["n_eff"].iloc[0])
    print(f"P1（S2 的 n_eff 持续增长）：rho = {rho2:.3f} (p={p2v:.4f}), "
          f"末点/首点 = {s2['n_eff'].iloc[-1] / s2['n_eff'].iloc[0]:.2f} -> "
          f"{'成立' if p1 else '不成立'}")
    last_ratio = s1["n_eff"].iloc[-1] / s1["n_eff"].iloc[0]
    p2 = bool(last_ratio < 2)
    print(f"P2（S1 的 n_eff 饱和）：rho = {rho1:.3f} (p={p1v:.4f}), "
          f"末点/首点 = {last_ratio:.2f} -> {'成立' if p2 else '不成立'}")
    common = sorted(set(s1.index) & set(s2.index))
    wins = sum(1 for n in common if s2.loc[n, "efficiency"] > s1.loc[n, "efficiency"])
    p3 = bool(wins / len(common) >= 0.7)
    print(f"P3（相同 n 下 S2 效率高于 S1）：{wins}/{len(common)} -> "
          f"{'成立' if p3 else '不成立'}")
    slopes = {}
    for arm, sub in (("S1", s1), ("S2", s2)):
        slope, _, r_value, _, _ = stats.linregress(
            np.log(sub.index.astype(float)), np.log(sub["sd_fp"].astype(float))
        )
        slopes[arm] = slope
        print(f"  {arm}: log-log 斜率 = {slope:.3f}（R² = {r_value ** 2:.3f}）")
    p4 = bool(abs(slopes["S2"] + 0.5) < abs(slopes["S1"] + 0.5))
    print(f"P4（S2 斜率更接近 −0.5）：{slopes['S2']:.3f} vs {slopes['S1']:.3f} -> "
          f"{'成立' if p4 else '不成立'}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for arm, marker, label in (("S1_resample", "o", "S1 resampled (same pool)"),
                               ("S2_new_records", "s", "S2 genuinely new records")):
        sub = summary[summary["arm"] == arm]
        axes[0].plot(sub["n_train"], sub["sd_fp"] * 100, marker=marker, label=label)
        axes[1].plot(sub["n_train"], sub["efficiency"], marker=marker, label=label)
    grid = np.unique(np.round(np.logspace(np.log10(5), np.log10(96), 40)))
    axes[0].plot(grid, np.sqrt(q * (1 - q) / grid) * 100, "k--", label="ideal  sqrt(q(1-q)/n)")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Training records (nominal n)")
    axes[0].set_ylabel("SD of reported false-alarm rate (pp)")
    axes[0].set_title("EXP-V2-01: Paderborn, nominal n vs noise")
    axes[0].legend(fontsize=8)
    axes[1].axhline(1.0, color="k", linestyle="--", label="ideal (n_eff = n)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Training records (nominal n)")
    axes[1].set_ylabel("Effective / nominal sample size")
    axes[1].set_title("S1 (resampling) vs S2 (new records)")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "exp_v2_01_cross_dataset_scaling.png", dpi=150)
    plt.close(figure)
    print("\n已保存曲线与图。")


if __name__ == "__main__":
    main()