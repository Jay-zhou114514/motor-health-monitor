"""EXP-V1-02：特征消融（Feature Ablation）与特征分布偏移分析。

实验契约
--------
Hypothesis:
    方法 B（马氏距离）的误报来自某些特征在"训练正常文件"与"测试正常文件"
    之间的分布偏移，而不是阈值；逐个/组合移除特征可以定位到具体特征。
Dataset:
    当前 MathWorks 轴承数据（8 个文件）。
Train:
    baseline_1 / baseline_2（仅正常数据）。
Test:
    其余 6 个文件（1 正常 + 3 外圈故障 + 2 内圈故障）。
Rule:
    阈值只由训练正常数据决定（分位数 0.99）；测试只用于评价。
Output:
    outputs/feature_ablation.csv
    outputs/feature_shift.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import data_loading
import evaluate as evaluate_module
from config import MAHAL_FEATURES, OUTPUT_DIR, STEP_SEC, WINDOW_SEC
from detection import MahalanobisDetector
from features import build_feature_table

FEATURE_SETS: list[tuple[str, list[str]]] = [
    ("RMS only", ["rms"]),
    ("RMS + Kurtosis", ["rms", "kurtosis"]),
    ("RMS + Crest Factor", ["rms", "crest_factor"]),
    ("RMS + Centroid", ["rms", "centroid_hz"]),
    ("RMS + Kurtosis + Crest Factor", ["rms", "kurtosis", "crest_factor"]),
    ("All (RMS+Crest+Kurtosis+Centroid)", MAHAL_FEATURES),
]

SHIFT_FEATURES = ["rms", "crest_factor", "kurtosis", "centroid_hz", "std", "peak"]


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


def ablation_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    y_true = (test["label"] != "normal").to_numpy()
    rows = []
    for name, features in FEATURE_SETS:
        detector = MahalanobisDetector(features=features, quantile=0.99).fit(train)
        prediction = detector.predict(test).to_numpy()
        metrics = evaluate_module.evaluate(y_true, prediction)
        rows.append(
            {
                "feature_set": name,
                "n_features": len(features),
                "threshold": round(float(detector.distance_limit_), 4),
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
        )
    return pd.DataFrame(rows)


def shift_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """每个特征：训练正常的均值/标准差 vs 测试正常文件均值。"""
    normal_test = test[test["label"] == "normal"]
    fault_test = test[test["label"] != "normal"]
    rows = []
    for feature in SHIFT_FEATURES:
        train_values = train[feature]
        train_mean = float(train_values.mean())
        train_std = float(train_values.std(ddof=1))
        normal_mean = float(normal_test[feature].mean())
        shift = (normal_mean - train_mean) / train_std if train_std > 0 else float("nan")
        rows.append(
            {
                "feature": feature,
                "train_mean": round(train_mean, 4),
                "train_std": round(train_std, 4),
                "test_normal_mean": round(normal_mean, 4),
                "shift_in_train_sigma": round(shift, 2),
                "test_normal_min": round(float(normal_test[feature].min()), 4),
                "test_normal_max": round(float(normal_test[feature].max()), 4),
                "fault_mean": round(float(fault_test[feature].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    train, test = build_tables()
    ablation = ablation_table(train, test)
    shift = shift_table(train, test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ablation.to_csv(OUTPUT_DIR / "feature_ablation.csv", index=False)
    shift.to_csv(OUTPUT_DIR / "feature_shift.csv", index=False)

    print("EXP-V1-02 特征消融（马氏距离，分位数 0.99，阈值仅由训练正常数据决定）")
    print(ablation.to_string(index=False))
    print("\n特征分布偏移（训练正常 → 测试正常，单位：训练标准差）")
    print(shift.to_string(index=False))
    print(f"\n已保存：{OUTPUT_DIR / 'feature_ablation.csv'}")
    print(f"已保存：{OUTPUT_DIR / 'feature_shift.csv'}")

    # 图：不同特征集对 F1 / 精确率 / 误报数的影响
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = list(range(len(ablation)))
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar([i - 0.2 for i in positions], ablation["f1"], width=0.4, label="F1")
    ax1.bar([i + 0.2 for i in positions], ablation["precision"], width=0.4, label="Precision")
    ax1.set_xticks(positions)
    ax1.set_xticklabels(ablation["feature_set"], rotation=25, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score")
    ax2 = ax1.twinx()
    ax2.plot(positions, ablation["fp"], color="tab:red", marker="o", label="False alarms (FP)")
    ax2.set_ylabel("False alarms", color="tab:red")
    ax1.set_title("EXP-V1-02 Feature Ablation (Mahalanobis, q = 0.99)")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower left", fontsize=8)
    fig.tight_layout()
    figure_dir = OUTPUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "feature_ablation.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"已保存：{figure_path}")


if __name__ == "__main__":
    main()

