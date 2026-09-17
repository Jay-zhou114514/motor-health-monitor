"""EXP-V1-10 补充检查（探索性）。

定位：**探索性分析（证据等级 C）**，不改变任何预注册判据，只用于解释主结果。
产出：outputs/exp_v1_10_extra_checks.csv

E1 划分敏感性是否是"选择过程"特有的？
E2 模态配置内部的波动。
E3 验证集能否预测测试表现。
E4 观察值相对零模型区间的位置。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from config import OUTPUT_DIR
from exp_v1_08_baselines import RMSThreshold, load_healthy_records
from exp_v1_10_selection_uncertainty import (
    BATCH,
    TEST_FILES,
    fit_model,
    percentile_of,
    resample_split,
    score_model,
    to_arrays,
)


def main() -> None:
    datasets = load_healthy_records()
    arrays = to_arrays(datasets[BATCH])
    fixed_test = TEST_FILES
    pool = [name for name in sorted(arrays) if name not in fixed_test]
    test_block = np.vstack([arrays[name] for name in fixed_test])

    rng = np.random.default_rng(20260917)
    rows = []
    for index in range(50):
        fit_names, val_names, _ = resample_split(rng, pool, 7, 3)
        train = np.vstack([arrays[name] for name in fit_names])
        val_block = np.vstack([arrays[name] for name in val_names])
        rms = RMSThreshold().fit(train)
        rms_test = rms.score(test_block)
        rms_val = rms.score(val_block)
        forest = fit_model("forest", train, (200, 0.8))
        forest_test = score_model(forest, test_block)
        rows.append(
            {
                "replicate": index,
                "rms_test_fp": float((rms_test > rms.threshold_).mean()),
                "rms_val_fp": float((rms_val > rms.threshold_).mean()),
                "default_forest_test_fp": float((forest_test > forest["threshold"]).mean()),
                "default_forest_test_pct": float(percentile_of(forest, test_block).mean()),
            }
        )
    extra = pd.DataFrame(rows)
    extra.to_csv(OUTPUT_DIR / "exp_v1_10_extra_checks.csv", index=False)

    replicates = pd.read_csv(OUTPUT_DIR / "exp_v1_10_arm_realA_R50.csv")
    null_n1 = pd.read_csv(OUTPUT_DIR / "exp_v1_10_arm_nullN1.csv")
    null_n2 = pd.read_csv(OUTPUT_DIR / "exp_v1_10_arm_nullN2.csv")

    print("=== E1：同一批 50 个 fit 划分上的“无选择”方法 ===")
    for column, label in (
        ("rms_test_fp", "3-sigma RMS (threshold re-estimated per split)"),
        ("default_forest_test_fp", "default iForest(200,0.8) (refit per split)"),
    ):
        values = extra[column]
        print(f"  {label}: mean={values.mean():.4f} sd={values.std(ddof=1):.4f} "
              f"min={values.min():.4f} max={values.max():.4f} "
              f"values={sorted(values.unique().round(4).tolist())}")
    print(f"  frozen reference (train fixed at 10 files): P3 = 0.0000")
    print(f"  default iForest continuous indicator sd = "
          f"{extra['default_forest_test_pct'].std(ddof=1):.4f}")

    print("\n=== E2：模态配置内部的波动（设计 A）===")
    modal = replicates["cfg"].value_counts().idxmax()
    inside = replicates[replicates["cfg"] == modal]
    print(f"  modal cfg {modal}: n={len(inside)}, "
          f"test FP mean={inside['test_fp'].mean():.4f} sd={inside['test_fp'].std(ddof=1):.4f} "
          f"(values {sorted(inside['test_fp'].unique().round(4).tolist())})")

    print("\n=== E3：验证集能否预测测试表现（设计 A）===")
    rho, pvalue = stats.spearmanr(replicates["val_fp"], replicates["test_fp"])
    print(f"  Spearman(val_fp, test_fp) = {rho:.3f} (p = {pvalue:.3f}, n = {len(replicates)})")

    print("\n=== E4：观察值相对零模型区间 ===")
    observed = {
        "modal_share": replicates["cfg"].value_counts().iloc[0] / len(replicates),
        "test_fp_std": replicates["test_fp"].std(ddof=1),

        "tie_share": float((replicates["n_tied_at_min"] >= 2).mean()),
    }
    for name, value in observed.items():
        lo1, hi1 = np.percentile(null_n1[name], [5, 95])
        lo2, hi2 = np.percentile(null_n2[name], [5, 95])
        flag1 = "inside" if lo1 <= value <= hi1 else "outside"
        flag2 = "inside" if lo2 <= value <= hi2 else "outside"
        print(f"  {name} = {value:.4f} | N1 [{lo1:.4f}, {hi1:.4f}] {flag1} | "
              f"N2 [{lo2:.4f}, {hi2:.4f}] {flag2}")

    print("\n=== 零模型汇总 ===")
    for label, frame in (("N1", null_n1), ("N2", null_n2)):
        print(f"  {label} ({len(frame)} pseudo-datasets x 20 resamples): "
              f"modal_share median={frame['modal_share'].median():.3f}, "
              f"tie_share median={frame['tie_share'].median():.3f}, "
              f"test_fp_std median={frame['test_fp_std'].median():.4f}, "
              f"test_fp_mean median={frame['test_fp_mean'].median():.4f}")

    ncurve = pd.read_csv(OUTPUT_DIR / "exp_v1_10_arm_ncurve.csv")
    print("\n=== 样本量曲线（N1 噪声基准，模态占比）===")
    print(ncurve.groupby("n_files")["modal_share"]
          .agg(["mean", "std", "count"]).round(3).to_string())


if __name__ == "__main__":
    main()