"""EXP-V1-01：阈值敏感性分析。

实验契约
--------
Hypothesis:
    检测性能可能对报警阈值高度敏感；若敏感，则"最优阈值"很可能只是
    在测试集上碰巧选出来的，不能作为可靠结论。
Dataset:
    当前 MathWorks 轴承数据（8 个文件）。
Train:
    baseline_1 / baseline_2（仅正常数据，用于拟合模型与阈值）。
Test:
    其余 6 个文件（1 正常 + 3 外圈故障 + 2 内圈故障）。
Methods:
    A：3σ RMS 阈值，扫描 k = 2 / 2.5 / 3 / 3.5 / 4
    B：马氏距离，扫描分位数 = 0.90 / 0.95 / 0.99 / 0.995 / 0.999
Metrics:
    Accuracy / Precision / Recall / F1 / FPR / FNR / 混淆计数
Rule:
    所有阈值只由训练正常数据决定；测试数据只用于评价。
Output:
    outputs/threshold_sensitivity.csv 与终端表格。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import data_loading
import evaluate as evaluate_module
from config import MAHAL_FEATURES, OUTPUT_DIR, STEP_SEC, WINDOW_SEC
from detection import MahalanobisDetector, ThresholdDetector
from features import build_feature_table

K_VALUES = [2.0, 2.5, 3.0, 3.5, 4.0]
QUANTILES = [0.90, 0.95, 0.99, 0.995, 0.999]


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


def _row(
    method: str,
    parameter: str,
    threshold: float,
    metrics: dict[str, float],
) -> dict:
    return {
        "method": method,
        "parameter": parameter,
        "threshold": round(float(threshold), 4),
        "accuracy": round(metrics["accuracy"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
        "fpr": round(metrics["fpr"], 4),
        "fnr": round(metrics["fnr"], 4),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "tn": int(metrics["tn"]),
    }


def main() -> None:
    train, test = build_tables()
    y_true = (test["label"] != "normal").to_numpy()

    rows: list[dict] = []
    for k in K_VALUES:
        detector = ThresholdDetector(column="rms", n_std=k).fit(train)
        pred = detector.predict(test).to_numpy()
        rows.append(
            _row(
                "A: 3-sigma RMS",
                f"k={k:g}",
                detector.threshold_value,
                evaluate_module.evaluate(y_true, pred),
            )
        )

    for quantile in QUANTILES:
        detector = MahalanobisDetector(
            features=MAHAL_FEATURES, quantile=quantile
        ).fit(train)
        pred = detector.predict(test).to_numpy()
        rows.append(
            _row(
                "B: Mahalanobis",
                f"q={quantile:g}",
                detector.distance_limit_,
                evaluate_module.evaluate(y_true, pred),
            )
        )

    table = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "threshold_sensitivity.csv"
    table.to_csv(csv_path, index=False)

    print("EXP-V1-01 阈值敏感性（阈值只由训练正常数据决定）")
    print(table.to_string(index=False))
    print(f"\n已保存：{csv_path}")

    # 方法 B 的关键诊断：训练正常距离分布 vs 测试正常文件距离
    detector_b = MahalanobisDetector(features=MAHAL_FEATURES, quantile=0.99).fit(train)
    train_distances = detector_b.decision_function(train).to_numpy()
    test_distances = detector_b.decision_function(test).to_numpy()
    test_with_distance = test.assign(distance=test_distances)
    print("\n方法 B 诊断信息：")
    print(
        "  训练正常窗口距离分位数："
        + ", ".join(
            f"p{q:g}={np.quantile(train_distances, q):.3f}"
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        )
    )
    print(
        "  测试窗口距离中位数："
        + ", ".join(
            f"{label}={group['distance'].median():.3f}"
            for label, group in test_with_distance.groupby("label")
        )
    )


if __name__ == "__main__":
    main()
