"""EXP-V1-04：良态特征集受控对照。

实验契约
--------
Hypothesis:
    如果马氏距离的大量误报部分来自协方差矩阵病态（近共线特征 + 小样本），
    那么条件数明显更低的特征组合应当（预测 1）误报更少，
    并且（预测 2）阈值估计更稳定。
Control:
    数据集、train/test 划分、窗口参数、检测方法（马氏距离）、
    协方差估计方式（经验协方差）、分位数（0.99）、评价协议全部保持一致，
    只改变"使用哪些特征"。
Dataset:
    MathWorks 轴承数据；train = baseline_1/2（22 个正常窗口），
    test = 1 正常 + 3 外圈 + 2 内圈（42 个窗口，其中 31 个故障）。
Metrics:
    condition number、threshold、threshold CV（留一训练窗口重拟合）、
    FP、TN、precision、recall、F1、FPR、距离分布、距离膨胀比。
Judgement:
    Spearman(log10 condition number, FP) 与
    Spearman(log10 condition number, threshold CV) 的方向与显著性。
Output:
    outputs/conditioning_control.csv
    outputs/figures/conditioning_control.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import data_loading
import evaluate as evaluate_module
from config import FIGURES_DIR, MAHAL_FEATURES, OUTPUT_DIR, STEP_SEC, WINDOW_SEC
from detection import MahalanobisDetector
from features import build_feature_table

QUANTILE = 0.99

# 覆盖不同维度与不同条件数的组合，其中包含"同维度、不同条件数"的对照
FEATURE_SETS: list[tuple[str, list[str]]] = [
    ("RMS only", ["rms"]),
    ("RMS + kurtosis", ["rms", "kurtosis"]),
    ("RMS + crest", ["rms", "crest_factor"]),
    ("crest + kurtosis", ["crest_factor", "kurtosis"]),
    ("RMS + centroid", ["rms", "centroid_hz"]),
    ("crest + centroid", ["crest_factor", "centroid_hz"]),
    ("kurtosis + centroid", ["kurtosis", "centroid_hz"]),
    ("RMS + kurtosis + crest", ["rms", "kurtosis", "crest_factor"]),
    ("RMS + kurtosis + centroid", ["rms", "kurtosis", "centroid_hz"]),
    ("RMS + crest + centroid", ["rms", "crest_factor", "centroid_hz"]),
    ("All (4 features)", MAHAL_FEATURES),
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


def condition_number(frame: pd.DataFrame, columns: list[str]) -> float:
    covariance = np.atleast_2d(
        np.cov(frame[columns].to_numpy(dtype=float), rowvar=False, ddof=1)
    )
    return float(np.linalg.cond(covariance))


def threshold_stability(
    train: pd.DataFrame, columns: list[str], quantile: float = QUANTILE
) -> tuple[float, float, float, float]:
    """留一训练窗口重拟合，衡量阈值估计的稳定性。

    返回 (阈值标准差, 阈值均值, 最小阈值, 最大阈值)。
    """
    thresholds = []
    for index in range(len(train)):
        subset = train.drop(index=index)
        detector = MahalanobisDetector(features=columns, quantile=quantile).fit(subset)
        thresholds.append(float(detector.distance_limit_))
    values = np.asarray(thresholds)
    return (
        float(values.std(ddof=1)),
        float(values.mean()),
        float(values.min()),
        float(values.max()),
    )


def evaluate_feature_set(
    train: pd.DataFrame, test: pd.DataFrame, name: str, columns: list[str]
) -> dict:
    detector = MahalanobisDetector(features=columns, quantile=QUANTILE).fit(train)
    prediction = detector.predict(test)
    y_true = (test["label"] != "normal").to_numpy()
    metrics = evaluate_module.evaluate(y_true, prediction.to_numpy())

    distances = pd.Series(
        detector.decision_function(test).to_numpy(), index=test.index, dtype=float
    )
    normal_distances = distances[test["label"] == "normal"]
    fault_distances = distances[test["label"] != "normal"]
    threshold = float(detector.distance_limit_)
    stability_std, stability_mean, _, _ = threshold_stability(train, columns)

    return {
        "feature_set": name,
        "n_features": len(columns),
        "condition_number": condition_number(train, columns),
        "threshold": threshold,
        "threshold_cv": stability_std / stability_mean if stability_mean else float("nan"),
        "fp": int(metrics["fp"]),
        "tn": int(metrics["tn"]),
        "tp": int(metrics["tp"]),
        "fn": int(metrics["fn"]),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
        "fpr": round(metrics["fpr"], 4),
        "normal_distance_median": round(float(normal_distances.median()), 4),
        "normal_distance_max": round(float(normal_distances.max()), 4),
        "fault_distance_median": round(float(fault_distances.median()), 4),
        "inflation_ratio": round(float(normal_distances.median()) / threshold, 3),
    }


def plot_results(table: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    log_condition = np.log10(table["condition_number"].clip(lower=1e-12))

    axes[0].scatter(log_condition, table["fp"], color="tab:red", s=45)
    for _, row in table.iterrows():
        axes[0].annotate(
            row["feature_set"],
            (np.log10(max(row["condition_number"], 1e-12)), row["fp"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    axes[0].set_xlabel("log10(condition number)")
    axes[0].set_ylabel("False alarms (FP)")
    axes[0].set_title("Condition number vs false alarms")
    axes[0].grid(alpha=0.3)

    order = table.sort_values("fp")
    axes[1].barh(order["feature_set"], order["fp"], color="tab:orange")
    axes[1].set_xlabel("False alarms (FP)")
    axes[1].set_title("False alarms by feature set")
    axes[1].tick_params(axis="y", labelsize=7)
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / "conditioning_control.png", dpi=150)
    plt.close(figure)


def main() -> None:
    train, test = build_tables()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        evaluate_feature_set(train, test, name, columns)
        for name, columns in FEATURE_SETS
    ]
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "conditioning_control.csv", index=False)
    plot_results(table)

    display_columns = [
        "feature_set",
        "n_features",
        "condition_number",
        "threshold",
        "threshold_cv",
        "fp",
        "fpr",
        "f1",
        "inflation_ratio",
    ]
    print("EXP-V1-04 良态特征集受控对照（马氏距离，q=0.99，仅改变特征集）")
    print(table[display_columns].to_string(index=False))

    log_condition = np.log10(table["condition_number"].clip(lower=1e-12))
    rho_fp, p_fp = spearmanr(log_condition, table["fp"])
    rho_cv, p_cv = spearmanr(log_condition, table["threshold_cv"])
    print("\n可检验预测：")
    print(
        f"  log10(condition number) vs FP:          "
        f"Spearman rho = {rho_fp:.3f}, p = {p_fp:.4f}"
    )
    print(
        f"  log10(condition number) vs threshold CV: "
        f"Spearman rho = {rho_cv:.3f}, p = {p_cv:.4f}"
    )

    print("\n同维度对照（2 特征）：")
    two_feature = table[table["n_features"] == 2].sort_values("condition_number")
    print(
        two_feature[
            ["feature_set", "condition_number", "fp", "fpr", "f1", "threshold_cv"]
        ].to_string(index=False)
    )

    print(f"\n已保存：{OUTPUT_DIR / 'conditioning_control.csv'}")
    print(f"已保存：{FIGURES_DIR / 'conditioning_control.png'}")


if __name__ == "__main__":
    main()
