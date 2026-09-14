"""EXP-V1-03：误差定位与依赖结构变化验证。

实验契约
--------
Question:
    误报是否与正常文件内部的时间变化、以及多变量依赖结构变化有关？
    现有相关系数差异是"真实差异"还是"小样本波动"？
Dataset:
    当前 MathWorks 轴承数据（8 个文件）。
Train:
    baseline_1 / baseline_2（仅正常数据）。
Test:
    其余 6 个文件；误差定位聚焦唯一正常文件 baseline_3。
Parts:
    A. 误报时间定位（含逐特征对马氏距离的贡献占比）
    B. 逐个正常文件的相关矩阵（baseline_1 / baseline_2 / baseline_3）
    C. 关键相关性的 bootstrap 95% 置信区间（训练 vs 测试）
    D. Leave-one-feature-out（从四特征中依次删掉一个）
Rule:
    阈值只由训练正常数据决定；测试只用于评价。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import data_loading
import evaluate as evaluate_module
from config import FIGURES_DIR, MAHAL_FEATURES, OUTPUT_DIR, STEP_SEC, WINDOW_SEC
from detection import MahalanobisDetector
from features import build_feature_table, extract_features

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
KEY_PAIRS = [
    ("rms", "kurtosis"),
    ("rms", "crest_factor"),
    ("crest_factor", "centroid_hz"),
    ("rms", "centroid_hz"),
]


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
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
    return train.reset_index(drop=True), test.reset_index(drop=True), records


def fit_detector(train: pd.DataFrame) -> MahalanobisDetector:
    return MahalanobisDetector(features=MAHAL_FEATURES, quantile=0.99).fit(train)


def part_a_false_positive_timeline(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[MahalanobisDetector, pd.DataFrame]:
    """A. baseline_3 每个窗口的距离、是否误报、以及各特征贡献占比。"""
    detector = fit_detector(train)
    normal = test[test["label"] == "normal"].reset_index(drop=True).copy()
    normal["distance"] = detector.decision_function(normal).to_numpy()
    normal["threshold"] = float(detector.distance_limit_)
    normal["false_alarm"] = normal["distance"] > normal["threshold"]

    data = normal[MAHAL_FEATURES].to_numpy(dtype=float)
    centered = data - detector.mean_
    contributions = np.abs(centered * (centered @ detector.inv_covariance_))
    totals = contributions.sum(axis=1, keepdims=True)
    shares = np.divide(
        contributions, totals, out=np.zeros_like(contributions), where=totals > 0
    )
    for index, feature in enumerate(MAHAL_FEATURES):
        normal[f"share_{feature}"] = shares[:, index]
    return detector, normal


def part_b_correlation_per_file(
    records: list[dict],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """B. 每个文件单独计算相关矩阵。"""
    rows: list[dict] = []
    tables: dict[str, pd.DataFrame] = {}
    for record in records:
        table = extract_features(
            record["signal"],
            record["sr"],
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
            record_id=record["file"],
            label=record["condition"],
        ).dropna(subset=MAHAL_FEATURES)
        tables[record["file"]] = table
        correlation = table[MAHAL_FEATURES].corr()
        for i, first in enumerate(MAHAL_FEATURES):
            for second in MAHAL_FEATURES[i + 1 :]:
                rows.append(
                    {
                        "file": record["file"],
                        "condition": record["condition"],
                        "n_windows": int(len(table)),
                        "pair": f"{first}|{second}",
                        "r": round(float(correlation.loc[first, second]), 3),
                    }
                )
    return pd.DataFrame(rows), tables


def _bootstrap_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    correlations: list[float] = []
    for _ in range(N_BOOTSTRAP):
        index = rng.integers(0, x.size, x.size)
        xs, ys = x[index], y[index]
        if xs.std() == 0 or ys.std() == 0:
            continue
        correlations.append(float(np.corrcoef(xs, ys)[0, 1]))
    values = np.array(correlations)
    point = float(np.corrcoef(x, y)[0, 1])
    low, high = np.percentile(values, [2.5, 97.5])
    return point, float(low), float(high)


def part_c_bootstrap(
    train: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """C. 关键特征对在训练正常 vs 测试正常上的 bootstrap 置信区间。"""
    normal = test[test["label"] == "normal"]
    rows = []
    for first, second in KEY_PAIRS:
        r_train, lo_train, hi_train = _bootstrap_ci(
            train[first].to_numpy(), train[second].to_numpy()
        )
        r_test, lo_test, hi_test = _bootstrap_ci(
            normal[first].to_numpy(), normal[second].to_numpy()
        )
        overlap = not (hi_train < lo_test or hi_test < lo_train)
        rows.append(
            {
                "pair": f"{first}|{second}",
                "train_r": round(r_train, 3),
                "train_ci_low": round(lo_train, 3),
                "train_ci_high": round(hi_train, 3),
                "test_r": round(r_test, 3),
                "test_ci_low": round(lo_test, 3),
                "test_ci_high": round(hi_test, 3),
                "ci_overlap": bool(overlap),
            }
        )
    return pd.DataFrame(rows)


def part_d_leave_one_feature_out(
    train: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """D. 依次删掉一个特征，观察误报如何变化。"""
    y_true = (test["label"] != "normal").to_numpy()
    feature_sets: list[tuple[str, list[str]]] = []
    for dropped in MAHAL_FEATURES:
        feature_sets.append(
            (f"All - {dropped}", [f for f in MAHAL_FEATURES if f != dropped])
        )
    feature_sets.append(("All (4 features)", MAHAL_FEATURES))
    feature_sets.append(("RMS only", ["rms"]))

    rows = []
    for name, features in feature_sets:
        detector = MahalanobisDetector(features=features, quantile=0.99).fit(train)
        prediction = detector.predict(test).to_numpy()
        metrics = evaluate_module.evaluate(y_true, prediction)
        rows.append(
            {
                "feature_set": name,
                "n_features": len(features),
                "threshold": round(float(detector.distance_limit_), 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
                "fpr": round(metrics["fpr"], 4),
                "false_alarms": int(metrics["fp"]),
                "true_normal": int(metrics["tn"]),
            }
        )
    return pd.DataFrame(rows)


def plot_error_localization(
    train: pd.DataFrame, timeline: pd.DataFrame
) -> None:
    """画出 baseline_3 的特征时间线与马氏距离，并标出误报窗口。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean = train[MAHAL_FEATURES].mean()
    std = train[MAHAL_FEATURES].std(ddof=1)
    standardized = (timeline[MAHAL_FEATURES] - mean) / std

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for feature in MAHAL_FEATURES:
        axes[0].plot(
            timeline["window_start_s"],
            standardized[feature],
            marker="o",
            markersize=3,
            label=feature,
        )
    axes[0].set_ylabel("Feature (train sigma)")
    axes[0].set_title("baseline_3: feature timeline (standardized by training normals)")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        timeline["window_start_s"],
        timeline["distance"],
        marker="o",
        color="tab:blue",
        label="Mahalanobis distance",
    )
    axes[1].axhline(
        float(timeline["threshold"].iloc[0]),
        color="tab:red",
        linestyle="--",
        label="threshold (train q=0.99)",
    )
    false_alarms = timeline[timeline["false_alarm"]]
    axes[1].scatter(
        false_alarms["window_start_s"],
        false_alarms["distance"],
        color="tab:red",
        marker="x",
        s=60,
        label="false alarm",
    )
    axes[1].set_xlabel("Window start (s)")
    axes[1].set_ylabel("Distance")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / "error_localization.png", dpi=150)
    plt.close(figure)


def plot_correlation_matrices(tables: dict[str, pd.DataFrame]) -> None:
    """画 baseline_1 / baseline_2 / baseline_3 的相关矩阵热图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [name for name in ("baseline_1.mat", "baseline_2.mat", "baseline_3.mat") if name in tables]
    if not names:
        return
    figure, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    for axis, name in zip(axes, names):
        correlation = tables[name][MAHAL_FEATURES].corr().to_numpy()
        image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(MAHAL_FEATURES)))
        axis.set_xticklabels(MAHAL_FEATURES, rotation=45, ha="right", fontsize=7)
        axis.set_yticks(range(len(MAHAL_FEATURES)))
        axis.set_yticklabels(MAHAL_FEATURES, fontsize=7)
        axis.set_title(f"{name}\n(n={len(tables[name])} windows)", fontsize=9)
        for i in range(len(MAHAL_FEATURES)):
            for j in range(len(MAHAL_FEATURES)):
                axis.text(
                    j,
                    i,
                    f"{correlation[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
    figure.colorbar(image, ax=axes, shrink=0.8)
    figure.suptitle("Feature correlation matrices per normal file")
    figure.savefig(FIGURES_DIR / "correlation_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    train, test, records = build_tables()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # A. 误报时间定位
    detector, timeline = part_a_false_positive_timeline(train, test)
    timeline_columns = [
        "window_start_s",
        *MAHAL_FEATURES,
        "distance",
        "threshold",
        "false_alarm",
        *[f"share_{feature}" for feature in MAHAL_FEATURES],
    ]
    timeline[timeline_columns].to_csv(
        OUTPUT_DIR / "error_localization.csv", index=False
    )
    plot_error_localization(train, timeline)
    print("A. baseline_3 误报时间定位（窗口级）")
    print(
        timeline[
            ["window_start_s", "distance", "threshold", "false_alarm"]
        ].round(3).to_string(index=False)
    )
    false_alarm_windows = timeline[timeline["false_alarm"]]
    print(f"   误报窗口数：{len(false_alarm_windows)} / {len(timeline)}")
    if len(false_alarm_windows):
        share_columns = [f"share_{feature}" for feature in MAHAL_FEATURES]
        print("   误报窗口的平均特征贡献占比：")
        print(
            false_alarm_windows[share_columns]
            .mean()
            .round(3)
            .to_string()
        )

    # B. 逐文件相关矩阵
    correlation_table, per_file_tables = part_b_correlation_per_file(records)
    correlation_table.to_csv(OUTPUT_DIR / "correlation_per_file.csv", index=False)
    plot_correlation_matrices(per_file_tables)
    print("\nB. 各正常文件的关键相关性（rms|kurtosis、rms|crest_factor 等）")
    key_pairs = {"rms|kurtosis", "rms|crest_factor", "crest_factor|centroid_hz"}
    print(
        correlation_table[correlation_table["pair"].isin(key_pairs)]
        .pivot(index="file", columns="pair", values="r")
        .to_string()
    )

    # C. bootstrap 置信区间
    bootstrap_table = part_c_bootstrap(train, test)
    bootstrap_table.to_csv(OUTPUT_DIR / "bootstrap_correlations.csv", index=False)
    print("\nC. 关键相关性的 bootstrap 95% 置信区间")
    print(bootstrap_table.to_string(index=False))

    # D. leave-one-feature-out
    leave_one_out = part_d_leave_one_feature_out(train, test)
    leave_one_out.to_csv(OUTPUT_DIR / "leave_one_feature_out.csv", index=False)
    print("\nD. Leave-one-feature-out（马氏距离，q=0.99）")
    print(leave_one_out.to_string(index=False))

    # E. 协方差条件数（数值稳定性）
    def condition_number(frame: pd.DataFrame, columns: list[str]) -> float:
        covariance = np.atleast_2d(
            np.cov(frame[columns].to_numpy(dtype=float), rowvar=False, ddof=1)
        )
        return float(np.linalg.cond(covariance))

    condition_sets = [
        ("All (4 features)", MAHAL_FEATURES),
        ("rms + centroid_hz", ["rms", "centroid_hz"]),
        ("crest_factor + centroid_hz", ["crest_factor", "centroid_hz"]),
        ("crest_factor + kurtosis", ["crest_factor", "kurtosis"]),
        ("rms + kurtosis", ["rms", "kurtosis"]),
        ("RMS only", ["rms"]),
    ]
    normal_test = test[test["label"] == "normal"]
    condition_table = pd.DataFrame(
        [
            {
                "feature_set": name,
                "train_condition_number": round(condition_number(train, columns), 1),
                "baseline_3_condition_number": round(
                    condition_number(normal_test, columns), 1
                ),
            }
            for name, columns in condition_sets
        ]
    )
    condition_table.to_csv(OUTPUT_DIR / "covariance_condition.csv", index=False)
    print("\nE. 协方差条件数（越大越病态，1 左右最健康）")
    print(condition_table.to_string(index=False))

    print(f"\n结果目录：{OUTPUT_DIR}")
    print(f"图表目录：{FIGURES_DIR}")


if __name__ == "__main__":
    main()

