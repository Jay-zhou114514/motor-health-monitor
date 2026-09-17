"""决定性检验：报告误报率的波动，主要来自「训练集组成」还是「算法随机性」？

这是对"把波动归因于划分组成"这一主张的**可证伪检验**。

设计（全部使用同一固定测试集）：
- A 组分：固定随机种子，改变 fit 集组成（50 个划分）→ 观察阈值与测试误报的波动
- B 组分：固定 fit 集，改变随机种子（50 个种子）→ 观察阈值与测试误报的波动
- 若 SD_A >> SD_B：波动主要来自训练集组成（支持我们的叙事）
- 若 SD_A ≈ SD_B：波动主要是算法随机性（**则叙事必须改写**）

额外：给出阈值的 99% 分位数估计噪声量级，用于解析对照。

输出：experiments/EXP-V1-10-variance-decomposition.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import OUTPUT_DIR
from exp_v1_08_baselines import QUANTILE, load_healthy_records
from exp_v1_10_selection_uncertainty import BATCH, TEST_FILES, resample_split, to_arrays

N_REPEAT = 50
MODAL_CFG = (200, 0.5)


def fit_threshold(train: np.ndarray, seed: int) -> tuple[float, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=1)
    std = np.where(std == 0, 1.0, std)
    standardized = (train - mean) / std
    model = IsolationForest(
        n_estimators=MODAL_CFG[0], max_samples=MODAL_CFG[1],
        random_state=seed, n_jobs=1,
    ).fit(standardized)
    scores = -model.score_samples(standardized)
    return float(np.quantile(scores, QUANTILE)), (mean, std, model)


def test_scores(state, test: np.ndarray) -> np.ndarray:
    mean, std, model = state
    return -model.score_samples((test - mean) / std)


def main() -> None:
    arrays = to_arrays(load_healthy_records()[BATCH])
    pool = [name for name in sorted(arrays) if name not in TEST_FILES]
    test_block = np.vstack([arrays[name] for name in TEST_FILES])

    # ---- A：改变 fit 集组成，种子固定 ----
    rng = np.random.default_rng(20260917)
    rows_a = []
    for index in range(N_REPEAT):
        fit_names, _, _ = resample_split(rng, pool, 7, 3)
        train = np.vstack([arrays[name] for name in fit_names])
        threshold, state = fit_threshold(train, seed=0)
        scores = test_scores(state, test_block)
        rows_a.append({
            "component": "A_fit_composition", "repeat": index, "seed": 0,
            "threshold": threshold,
            "test_fp": float((scores > threshold).mean()),
            "test_pct_mean": float((scores > threshold).mean()),
        })
    frame_a = pd.DataFrame(rows_a)

    # ---- B：固定 fit 集，改变随机种子 ----
    rng_fixed = np.random.default_rng(20260917)
    fit_names, _, _ = resample_split(rng_fixed, pool, 7, 3)
    train_fixed = np.vstack([arrays[name] for name in fit_names])
    rows_b = []
    for seed in range(N_REPEAT):
        threshold, state = fit_threshold(train_fixed, seed=seed)
        scores = test_scores(state, test_block)
        rows_b.append({
            "component": "B_random_seed", "repeat": seed, "seed": seed,
            "threshold": threshold,
            "test_fp": float((scores > threshold).mean()),
            "test_pct_mean": float((scores > threshold).mean()),
        })
    frame_b = pd.DataFrame(rows_b)

    # ---- C：不同划分 × 不同种子（总方差参照）----
    rng_c = np.random.default_rng(20260917)
    rows_c = []
    for index in range(N_REPEAT):
        fit_names_c, _, _ = resample_split(rng_c, pool, 7, 3)
        train_c = np.vstack([arrays[name] for name in fit_names_c])
        threshold, state = fit_threshold(train_c, seed=index)
        scores = test_scores(state, test_block)
        rows_c.append({
            "component": "C_both", "repeat": index, "seed": index,
            "threshold": threshold,
            "test_fp": float((scores > threshold).mean()),
            "test_pct_mean": float((scores > threshold).mean()),
        })
    frame_c = pd.DataFrame(rows_c)

    frame = pd.concat([frame_a, frame_b, frame_c], ignore_index=True)
    frame.to_csv(OUTPUT_DIR / "exp_v1_10_variance_decomposition.csv", index=False)

    print("=== 三组分对比（固定测试集，iForest 200/0.5，99% 分位阈值）===")
    summary = (frame.groupby("component")
               .agg(threshold_sd=("threshold", "std"),
                    test_fp_mean=("test_fp", "mean"),
                    test_fp_sd=("test_fp", "std"),
                    test_fp_min=("test_fp", "min"),
                    test_fp_max=("test_fp", "max"))
               .round(4))
    print(summary.to_string())

    sd_a = float(summary.loc["A_fit_composition", "test_fp_sd"])
    sd_b = float(summary.loc["B_random_seed", "test_fp_sd"])
    print("\n=== 判定 ===")
    print(f"SD_A（训练集组成）= {sd_a:.4f}")
    print(f"SD_B（算法随机性）= {sd_b:.4f}")
    if sd_b == 0:
        print("SD_B = 0 → 算法随机性不贡献波动；全部波动来自训练集组成。")
    else:
        ratio = sd_a / sd_b
        verdict = ("训练集组成主导" if ratio >= 2
                   else "两者相当（叙事必须改写）" if ratio < 1.5
                   else "训练集组成较强但不能忽略随机性")
        print(f"SD_A / SD_B = {ratio:.2f} → {verdict}")

    # ---- 解析对照：49 个训练窗口下 99% 分位数的估计噪声 ----
    print("\n=== 解析对照 ===")
    print("阈值 = 49 个训练窗口分数的 99% 分位（样本内），阈值自身的抽样噪声")
    print(f"  阈值 SD（A 组）= {summary.loc['A_fit_composition', 'threshold_sd']:.4f}")
    print(f"  阈值 SD（B 组）= {summary.loc['B_random_seed', 'threshold_sd']:.4f}")
    print("  测试集二项分辨率 = 1/14 = 7.14 pp（单次观测的量化步长）")
    print(f"  实测（EXP-V1-10 主分析，模式配置内 n=44）SD = 3.58 pp")
    print(f"  本检验 A 组（n=50）SD = {sd_a * 100:.2f} pp")


if __name__ == "__main__":
    main()