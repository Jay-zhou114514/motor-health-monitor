"""EXP-V1-05：Covariance Geometry & Regularization。

实验契约
--------
Positioning:
    机制验证优先、修复验证并行；目标不是提高 F1。
Hypothesis:
    H1（方向归因）：问题组的距离膨胀可归因到协方差的最小特征值方向
    ——即该方向与正常文件之间的分布差异对齐。
    H2（正则化）：若 H1 成立，收缩协方差应同时降低距离膨胀与误报。
Control:
    只比较三个特征组；数据、划分、窗口、分位数（0.99）、
    阈值推导方式与评价协议完全一致。
Groups:
    RMS + kurtosis（正常对照）
    RMS + centroid（问题组）
    crest + centroid（高条件数但无严重 FP 的反例组）
Rule:
    协方差与所有超参数（含收缩强度）只由训练数据确定；
    测试数据只用于评价。收缩强度用训练集留一交叉验证选择。
Outputs:
    outputs/exp_v1_05_geometry.csv
    outputs/exp_v1_05_regularization.csv
    outputs/exp_v1_05_window_stats.csv
    outputs/figures/exp_v1_05_geometry.png
    outputs/figures/exp_v1_05_regularization.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import data_loading
import evaluate as evaluate_module
from config import FIGURES_DIR, MAHAL_FEATURES, OUTPUT_DIR, STEP_SEC, WINDOW_SEC
from covariance_geometry import (
    canonicalize_direction,
    center,
    direction_contributions,
    eigen_structure,
    empirical_covariance,
    ledoit_wolf_delta,
    mahalanobis_distances,
    select_shrinkage_by_loo,
    shrinkage_covariance,
)
from features import build_feature_table

QUANTILE = 0.99
JITTER = 1e-12
DELTAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)

GROUPS: list[tuple[str, list[str]]] = [
    ("RMS + kurtosis (control)", ["rms", "kurtosis"]),
    ("RMS + centroid (problem)", ["rms", "centroid_hz"]),
    ("crest + centroid (counterexample)", ["crest_factor", "centroid_hz"]),
]


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = data_loading.load_records(data_loading.MINIMAL_RELATIVE_FILES)
    train_records = [r for r in records if r["split"] == "train"]
    test_records = [r for r in records if r["split"] == "test"]
    window_kwargs = {"window_sec": WINDOW_SEC, "step_sec": STEP_SEC}
    train = build_feature_table(train_records, **window_kwargs).dropna(
        subset=MAHAL_FEATURES
    )
    test = build_feature_table(test_records, **window_kwargs).dropna(
        subset=MAHAL_FEATURES
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def evaluate_estimator(
    train_matrix: np.ndarray,
    test_table: pd.DataFrame,
    features: list[str],
    mean: np.ndarray,
    covariance: np.ndarray,
) -> dict:
    train_distances = mahalanobis_distances(train_matrix, mean, covariance)
    threshold = float(np.quantile(train_distances, QUANTILE))
    test_distances = mahalanobis_distances(
        test_table[features].to_numpy(dtype=float), mean, covariance
    )
    prediction = test_distances > threshold
    y_true = (test_table["label"] != "normal").to_numpy()
    metrics = evaluate_module.evaluate(y_true, prediction)
    normal_distances = test_distances[test_table["label"].to_numpy() == "normal"]
    return {
        "threshold": threshold,
        "fp": int(metrics["fp"]),
        "tn": int(metrics["tn"]),
        "fpr": round(metrics["fpr"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
        "normal_distance_median": round(float(np.median(normal_distances)), 4),
        "inflation_ratio": round(float(np.median(normal_distances)) / threshold, 3),
        "condition_number": float(np.linalg.cond(covariance)),
    }


def main() -> None:
    train, test = build_tables()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    geometry_rows: list[dict] = []
    regularization_rows: list[dict] = []
    window_rows: list[dict] = []

    for name, features in GROUPS:
        train_matrix = train[features].to_numpy(dtype=float)
        mean = train_matrix.mean(axis=0)
        centered = center(train_matrix)

        # ---------- B：方向归因（经验协方差几何） ----------
        empirical = empirical_covariance(centered)
        eigenvalues, eigenvectors = eigen_structure(empirical)
        minimum_direction = canonicalize_direction(eigenvectors[:, 0])
        decomposition = direction_contributions(
            test[features].to_numpy(dtype=float), mean, eigenvalues, eigenvectors
        )
        train_decomposition = direction_contributions(
            train_matrix, mean, eigenvalues, eigenvectors
        )
        labels = test["label"].to_numpy()
        normal_mask = labels == "normal"
        fault_mask = ~normal_mask

        geometry_rows.append(
            {
                "feature_set": name,
                "condition_number": float(np.linalg.cond(empirical)),
                "eigenvalue_min": float(eigenvalues[0]),
                "eigenvalue_max": float(eigenvalues[-1]),
                "min_direction": ", ".join(
                    f"{f}={v:+.3f}" for f, v in zip(features, minimum_direction)
                ),
                "share_min_train_normal": round(
                    float(np.median(train_decomposition["shares"][:, 0])), 4
                ),
                "share_min_test_normal": round(
                    float(np.median(decomposition["shares"][normal_mask, 0])), 4
                ),
                "share_min_fault": round(
                    float(np.median(decomposition["shares"][fault_mask, 0])), 4
                ),
                "stdproj_min_test_normal": round(
                    float(
                        np.median(
                            np.abs(decomposition["standardized_projections"][normal_mask, 0])
                        )
                    ),
                    4,
                ),
                "stdproj_min_train_normal": round(
                    float(
                        np.median(
                            np.abs(
                                train_decomposition["standardized_projections"][:, 0]
                            )
                        )
                    ),
                    4,
                ),
            }
        )

        for index in range(len(test)):
            window_rows.append(
                {
                    "feature_set": name,
                    "record": test["record"].iloc[index],
                    "label": labels[index],
                    "window_start_s": test["window_start_s"].iloc[index],
                    "distance": float(np.sqrt(decomposition["squared_distance"][index])),
                    "share_min": float(decomposition["shares"][index, 0]),
                    "standardized_projection_min": float(
                        decomposition["standardized_projections"][index, 0]
                    ),
                }
            )

        # ---------- A：正则化对比 ----------
        delta_lw = ledoit_wolf_delta(centered)
        delta_loo, _ = select_shrinkage_by_loo(centered, DELTAS)
        estimators = [
            ("empirical", 0.0),
            ("ledoit_wolf", delta_lw),
            ("shrinkage_loo", delta_loo),
        ]
        for estimator_name, delta in estimators:
            covariance = shrinkage_covariance(centered, delta)
            covariance = covariance + np.eye(covariance.shape[0]) * JITTER
            result = evaluate_estimator(train_matrix, test, features, mean, covariance)
            regularization_rows.append(
                {
                    "feature_set": name,
                    "estimator": estimator_name,
                    "delta": round(float(delta), 4),
                    **result,
                }
            )

        print(f"\n=== {name} ===")
        print(
            "  条件数 = {:.3e}；最小特征值方向: {}".format(
                float(np.linalg.cond(empirical)),
                ", ".join(f"{f}={v:+.3f}" for f, v in zip(features, minimum_direction)),
            )
        )
        print(
            "  最小方向占比（中位数）：训练正常 {:.3f} | 测试正常 {:.3f} | 故障 {:.3f}".format(
                float(np.median(train_decomposition["shares"][:, 0])),
                float(np.median(decomposition["shares"][normal_mask, 0])),
                float(np.median(decomposition["shares"][fault_mask, 0])),
            )
        )
        print(
            "  测试正常在最小方向上的标准化投影（中位数，绝对值）={:.3f}".format(
                float(
                    np.median(
                        np.abs(decomposition["standardized_projections"][normal_mask, 0])
                    )
                )
            )
        )
        print(
            "  收缩强度：Ledoit-Wolf δ={:.3f}；LOO 最优 δ={:.3f}".format(
                delta_lw, delta_loo
            )
        )

    geometry = pd.DataFrame(geometry_rows)
    regularization = pd.DataFrame(regularization_rows)
    window_stats = pd.DataFrame(window_rows)
    geometry.to_csv(OUTPUT_DIR / "exp_v1_05_geometry.csv", index=False)
    regularization.to_csv(OUTPUT_DIR / "exp_v1_05_regularization.csv", index=False)
    window_stats.to_csv(OUTPUT_DIR / "exp_v1_05_window_stats.csv", index=False)

    print("\n=== 方向归因汇总 ===")
    print(geometry.to_string(index=False))
    print("\n=== 正则化对比汇总 ===")
    print(
        regularization[
            [
                "feature_set",
                "estimator",
                "delta",
                "condition_number",
                "threshold",
                "fp",
                "fpr",
                "f1",
                "inflation_ratio",
            ]
        ].to_string(index=False)
    )

    # ---------- 图 ----------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    positions = np.arange(len(geometry))
    axes[0].bar(positions - 0.2, geometry["share_min_train_normal"], width=0.35, label="train normal")
    axes[0].bar(positions + 0.2, geometry["share_min_test_normal"], width=0.35, label="test normal")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(geometry["feature_set"], rotation=15, ha="right", fontsize=8)
    axes[0].set_ylabel("Median share of D² in min-eigenvalue direction")
    axes[0].set_title("Directional attribution")
    axes[0].legend(fontsize=8)

    for estimator_name, color in (
        ("empirical", "tab:red"),
        ("ledoit_wolf", "tab:blue"),
        ("shrinkage_loo", "tab:green"),
    ):
        subset = regularization[regularization["estimator"] == estimator_name]
        axes[1].plot(
            subset["feature_set"],
            subset["fp"],
            marker="o",
            color=color,
            label=estimator_name,
        )
    axes[1].set_ylabel("False alarms (FP)")
    axes[1].set_title("Regularization effect on false alarms")
    axes[1].tick_params(axis="x", rotation=15, labelsize=8)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "exp_v1_05_geometry.png", dpi=150)
    plt.close(figure)

    print(f"\n已保存：{OUTPUT_DIR}")
    print(f"图表目录：{FIGURES_DIR}")


if __name__ == "__main__":
    main()
