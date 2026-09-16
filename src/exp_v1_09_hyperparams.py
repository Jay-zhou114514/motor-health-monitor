"""EXP-V1-09：训练内超参协议（嵌套 leave-one-file-out）。

预注册见 experiments/EXP-V1-09-preregistration.md。

核心纪律：**外层留出文件（测试）绝不参与超参选择**。
选择只在训练集内部通过内层 leave-one-file-out 完成。

输出：
- outputs/exp_v1_09_folds.csv
- outputs/exp_v1_09_summary.csv
- outputs/figures/exp_v1_09_hyperparams.png
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from config import FIGURES_DIR, OUTPUT_DIR
from exp_v1_08_baselines import (
    FEATURES,
    QUANTILE,
    Mahalanobis,
    RMSThreshold,
    load_healthy_records,
)

# 预注册冻结的候选网格（顺序即并列时的优先顺序：更保守者在前）
OCSVM_GRID = [
    (nu, gamma)
    for nu in (0.001, 0.005, 0.01, 0.05, 0.1)
    for gamma in ("scale", 0.1, 1.0, 10.0)
]
IF_GRID = [
    (n_estimators, max_samples)
    for n_estimators in (200, 100)  # 并列优先更多树
    for max_samples in (0.5, 0.8, 1.0)
]
FP_TARGET = 0.05


def fit_ocsvm(train: np.ndarray, nu: float, gamma):
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    standardized = (train - mean) / std
    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma).fit(standardized)
    scores = -model.decision_function(standardized)
    threshold = float(np.quantile(scores, QUANTILE))
    return mean, std, model, threshold


def score_ocsvm(state, data: np.ndarray) -> np.ndarray:
    mean, std, model, _ = state
    standardized = (data - mean) / std
    return -model.decision_function(standardized)


def fit_iforest(train: np.ndarray, n_estimators: int, max_samples: float):
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    standardized = (train - mean) / std
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=0,
        n_jobs=-1,
    ).fit(standardized)
    scores = -model.score_samples(standardized)
    threshold = float(np.quantile(scores, QUANTILE))
    return mean, std, model, threshold


def score_iforest(state, data: np.ndarray) -> np.ndarray:
    mean, std, model, _ = state
    standardized = (data - mean) / std
    return -model.score_samples(standardized)


def table_to_array(tables) -> np.ndarray:
    return np.vstack([table[FEATURES].to_numpy(float) for table in tables])


def fp_rate_for_hyper(train_tables, test_table, method: str, hyper) -> float:
    train = table_to_array(train_tables)
    test = test_table[FEATURES].to_numpy(float)
    if method == "svm":
        state = fit_ocsvm(train, *hyper)
        scores = score_ocsvm(state, test)
    else:
        state = fit_iforest(train, *hyper)
        scores = score_iforest(state, test)
    threshold = state[3]
    return float((scores > threshold).mean())


def inner_select(train_names, tables, method: str, candidates) -> tuple:
    """在训练集内部做 leave-one-file-out 选超参（不使用外层测试文件）。"""
    best_hyper = candidates[0]
    best_score = None
    for hyper in candidates:
        folds = []
        for held_out in train_names:
            inner_train = [tables[n] for n in train_names if n != held_out]
            folds.append(
                fp_rate_for_hyper(inner_train, tables[held_out], method, hyper)
            )
        score = float(np.mean(folds))
        if best_score is None or score < best_score - 1e-12:
            best_score = score
            best_hyper = hyper
    return best_hyper, best_score


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_healthy_records()
    batches = ["IMS 1st_test", "IMS 2nd_test", "IMS 4th_test"]

    rows: list[dict] = []
    for batch in batches:
        tables = datasets[batch]
        names = sorted(tables)
        print(f"\n=== {batch}（{len(names)} 个文件；外层 {len(names)} 折）===")
        for held_out in names:
            train_names = [n for n in names if n != held_out]
            # 参考方法（无超参）
            train = table_to_array([tables[n] for n in train_names])
            test = tables[held_out][FEATURES].to_numpy(float)
            for detector_class in (RMSThreshold, Mahalanobis):
                detector = detector_class()
                start = time.perf_counter()
                detector.fit(train)
                fp = float((detector.score(test) > detector.threshold_).mean())
                rows.append(
                    {
                        "batch": batch,
                        "held_out": held_out,
                        "method": detector.name,
                        "hyper": "-",
                        "fp_rate": fp,
                        "seconds": time.perf_counter() - start,
                    }
                )
            # 现代方法：内层选参 → 外层评估
            for method, label, candidates in (
                ("svm", "C: One-Class SVM (tuned)", OCSVM_GRID),
                ("forest", "D: Isolation Forest (tuned)", IF_GRID),
            ):
                start = time.perf_counter()
                hyper, _ = inner_select(train_names, tables, method, candidates)
                if method == "svm":
                    state = fit_ocsvm(train, *hyper)
                    scores = score_ocsvm(state, test)
                else:
                    state = fit_iforest(train, *hyper)
                    scores = score_iforest(state, test)
                fp = float((scores > state[3]).mean())
                rows.append(
                    {
                        "batch": batch,
                        "held_out": held_out,
                        "method": label,
                        "hyper": str(hyper),
                        "fp_rate": fp,
                        "seconds": time.perf_counter() - start,
                    }
                )

    folds = pd.DataFrame(rows)
    folds.to_csv(OUTPUT_DIR / "exp_v1_09_folds.csv", index=False)

    summary_rows = []
    for (batch, method), group in folds.groupby(["batch", "method"]):
        hypers = group["hyper"].tolist()
        modal_share = (
            Counter(hypers).most_common(1)[0][1] / len(hypers) if hypers else np.nan
        )
        summary_rows.append(
            {
                "batch": batch,
                "method": method,
                "mean_fp_rate": round(float(group["fp_rate"].mean()), 4),
                "median_fp_rate": round(float(group["fp_rate"].median()), 4),
                "max_fp_rate": round(float(group["fp_rate"].max()), 4),
                "modal_hyper": Counter(hypers).most_common(1)[0][0],
                "modal_share": round(float(modal_share), 3),
                "seconds_per_fold": round(float(group["seconds"].mean()), 4),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["batch", "method"])
    summary.to_csv(OUTPUT_DIR / "exp_v1_09_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nEXP-V1-09 批次均值误报率（内层选参，测试不参与）")
    print(summary.to_string(index=False))

    print("\n=== 预注册判据 ===")
    print(f"判定阈值（预先设定，非方法应有水平）：FP rate ≤ {FP_TARGET:.0%}\n")
    for label in ("C: One-Class SVM (tuned)", "D: Isolation Forest (tuned)"):
        subset = summary[summary["method"] == label]
        passed = int((subset["mean_fp_rate"] <= FP_TARGET).sum())
        print(
            f"{label}: 达标批次 {passed}/{len(subset)}；"
            f"批次均值 = {[round(v,4) for v in subset['mean_fp_rate'].tolist()]}"
        )
    tuned = summary[summary["method"].str.contains("tuned")]
    passed_total = int((tuned["mean_fp_rate"] <= FP_TARGET).sum())
    print(
        f"\nP1/P2/P3：现代方法达标批次数 = {passed_total} "
        f"→ {'支持（3 批次全达标）' if passed_total == 3 else ('部分支持（1-2 批次）' if passed_total else '不支持（0 批次）')}"
    )

    unstable = tuned[tuned["modal_share"] < 0.5]
    print(
        f"\nP4（次要）：折间选参不稳定（模态占比 < 50%）的组合数 = {len(unstable)}"
        f" → {'成立' if len(unstable) else '不成立'}"
    )
    if len(unstable):
        print(unstable[["batch", "method", "modal_share"]].to_string(index=False))

    # ==== 图 ====
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted(summary["method"].unique())
    figure, axis = plt.subplots(figsize=(10, 5))
    width = 0.2
    for index, method in enumerate(methods):
        values = [
            float(
                summary[(summary["batch"] == batch) & (summary["method"] == method)][
                    "mean_fp_rate"
                ].iloc[0]
            )
            for batch in batches
        ]
        axis.bar(
            np.arange(len(batches)) + (index - 1.5) * width,
            values,
            width=width,
            label=method,
        )
    axis.axhline(FP_TARGET, linestyle="--", color="gray", label="pre-set threshold 5%")
    axis.set_xticks(np.arange(len(batches)))
    axis.set_xticklabels(batches, rotation=10, fontsize=9)
    axis.set_ylabel("Batch-mean false-positive rate")
    axis.set_title("EXP-V1-09: hyperparameters selected inside training only")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "exp_v1_09_hyperparams.png", dpi=150)
    plt.close(figure)

    print(f"\n已保存：{OUTPUT_DIR / 'exp_v1_09_folds.csv'}")
    print(f"已保存：{OUTPUT_DIR / 'exp_v1_09_summary.csv'}")
    print(f"已保存：{FIGURES_DIR / 'exp_v1_09_hyperparams.png'}")


if __name__ == "__main__":
    main()
