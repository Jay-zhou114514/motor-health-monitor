"""EXP-V1-07：H2 的跨批次 / 跨数据集验证。

严格遵循：
- 主预注册 experiments/EXP-V1-07-preregistration.md
- 修订 1（IMS 窗口 0.25 s / 0.125 s）
- 修订 2（独立单元 = 试验批次；批次内 LOFO；不做文件级显著检验）

数据：
- IMS 1st_test：12 个健康文件（2003-10-22 前 1 小时）
- IMS 2nd_test：6 个健康文件
- IMS 4th_test：6 个健康文件
- MFPT（MathWorks 子集）：3 个正常基线文件

分析：
1. 每个批次内部做 leave-one-file-out；
2. 每个折叠、每个特征组、每种协方差方案，记录
   条件数、阈值、膨胀比（留出正常窗口距离中位数 / 阈值）、FP rate；
3. 批次间比较"收缩是否降低膨胀比"的方向一致性（独立单元 = 批次）；
4. 强制稳健性检查：IMS 用 0.5 s 窗口重跑一遍（修订 1 的要求）。

输出：
- outputs/exp_v1_07_folds.csv
- outputs/exp_v1_07_summary.csv
- outputs/exp_v1_07_folds_w0.5.csv（稳健性）
- outputs/figures/exp_v1_07_cross_dataset.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import data_loading
from config import FIGURES_DIR, OUTPUT_DIR
from covariance_geometry import (
    center,
    empirical_covariance,
    ledoit_wolf_delta,
    mahalanobis_distances,
    shrinkage_covariance,
)
from features import extract_features

ROOT = Path(__file__).resolve().parents[1]
IMS_ROOT = ROOT / "data" / "raw" / "ims"
IMS_SR = 20_000.0
QUANTILE = 0.99
JITTER = 1e-12
COND_TARGET = 100.0
DELTAS = np.round(np.arange(0.0, 1.0001, 0.01), 2)

GROUPS: list[tuple[str, list[str]]] = [
    ("RMS + kurtosis (control)", ["rms", "kurtosis"]),
    ("RMS + centroid (problem)", ["rms", "centroid_hz"]),
    ("crest + centroid (counterexample)", ["crest_factor", "centroid_hz"]),
]

SCHEMES = ["empirical", "ledoit_wolf", "cond_target"]


def ims_files(subset: str) -> list[Path]:
    files = sorted((IMS_ROOT / subset).glob("*.txt"))
    if subset == "1st_test":
        files = [path for path in files if path.name.startswith("2003.10.22")]
    return files


def load_ims_signal(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t", header=None, usecols=[0])
    return frame.iloc[:, 0].to_numpy(dtype=float)


def build_records() -> list[dict]:
    records: list[dict] = []

    for subset in ("1st_test", "2nd_test", "4th_test"):
        for path in ims_files(subset):
            records.append(
                {
                    "dataset": f"IMS {subset}",
                    "run": subset,
                    "file": path.name,
                    "signal": load_ims_signal(path),
                    "sr": IMS_SR,
                    "label": "normal",
                }
            )

    for record in data_loading.load_records():
        if record["split"] == "train" or record["file"] == "baseline_3.mat":
            records.append(
                {
                    "dataset": "MFPT",
                    "run": "MFPT",
                    "file": record["file"],
                    "signal": record["signal"],
                    "sr": record["sr"],
                    "label": "normal",
                }
            )
    return records


def window_params(dataset: str, fast_window: bool) -> tuple[float, float]:
    if dataset.startswith("IMS"):
        return (0.5, 0.25) if fast_window else (0.25, 0.125)
    return (1.0, 0.5)


def condition_target_delta(centered: np.ndarray) -> float:
    for delta in DELTAS:
        covariance = shrinkage_covariance(centered, float(delta))
        if np.linalg.cond(covariance) <= COND_TARGET:
            return float(delta)
    return 1.0


def fit_and_evaluate(
    train_windows: np.ndarray, test_windows: np.ndarray, scheme: str
) -> dict:
    centered_train = center(train_windows)
    if scheme == "empirical":
        delta = 0.0
    elif scheme == "ledoit_wolf":
        delta = ledoit_wolf_delta(centered_train)
    else:
        delta = condition_target_delta(centered_train)

    mean = train_windows.mean(axis=0)
    covariance = shrinkage_covariance(centered_train, float(delta))
    covariance = covariance + np.eye(covariance.shape[0]) * JITTER

    train_distances = mahalanobis_distances(train_windows, mean, covariance)
    threshold = float(np.quantile(train_distances, QUANTILE))
    test_distances = mahalanobis_distances(test_windows, mean, covariance)
    fp = int((test_distances > threshold).sum())
    return {
        "delta": round(float(delta), 4),
        "condition_number": float(np.linalg.cond(covariance)),
        "threshold": threshold,
        "inflation_ratio": float(np.median(test_distances)) / threshold,
        "fp": fp,
        "n_windows": int(test_distances.size),
        "fp_rate": fp / max(1, test_distances.size),
    }


def analyse(fast_window: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = build_records()
    fold_rows: list[dict] = []

    datasets = sorted({record["dataset"] for record in records}, key=str)
    for dataset in datasets:
        group_records = [r for r in records if r["dataset"] == dataset]
        window_sec, step_sec = window_params(dataset, fast_window)
        # 每个文件的特征表
        tables: dict[str, pd.DataFrame] = {}
        for record in group_records:
            table = extract_features(
                record["signal"],
                record["sr"],
                window_sec=window_sec,
                step_sec=step_sec,
                record_id=record["file"],
                label=record["label"],
            ).dropna(subset=["rms", "crest_factor", "kurtosis", "centroid_hz"])
            if len(table) >= 1:
                tables[record["file"]] = table

        names = sorted(tables)
        if len(names) < 2:
            continue

        for held_out in names:
            train_names = [name for name in names if name != held_out]
            for group_name, features in GROUPS:
                train_windows = np.vstack(
                    [tables[name][features].to_numpy(dtype=float) for name in train_names]
                )
                test_windows = tables[held_out][features].to_numpy(dtype=float)
                for scheme in SCHEMES:
                    result = fit_and_evaluate(train_windows, test_windows, scheme)
                    fold_rows.append(
                        {
                            "dataset": dataset,
                            "run": group_records[0]["run"],
                            "window_sec": window_sec,
                            "held_out": held_out,
                            "feature_group": group_name,
                            "scheme": scheme,
                            "n_train_windows": int(train_windows.shape[0]),
                            **result,
                        }
                    )

    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby(["dataset", "feature_group", "scheme"], as_index=False)
        .agg(
            median_inflation=("inflation_ratio", "median"),
            mean_fp_rate=("fp_rate", "mean"),
            median_condition=("condition_number", "median"),
            folds=("inflation_ratio", "size"),
        )
        .sort_values(["dataset", "feature_group", "scheme"])
    )
    return folds, summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    folds, summary = analyse(fast_window=False)
    folds.to_csv(OUTPUT_DIR / "exp_v1_07_folds.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "exp_v1_07_summary.csv", index=False)

    folds_w05, summary_w05 = analyse(fast_window=True)
    folds_w05.to_csv(OUTPUT_DIR / "exp_v1_07_folds_w0.5.csv", index=False)

    pd.set_option("display.width", 200)
    print("EXP-V1-07 主分析（IMS 0.25 s 窗 / MFPT 1.0 s 窗）")
    print(summary.to_string(index=False))

    problem = "RMS + centroid (problem)"
    control = "RMS + kurtosis (control)"

    print("\n=== 预注册判据 ===")
    # P1：IMS 上问题组出现 inflation > 1 的折占比
    ims_problem = folds[
        (folds["feature_group"] == problem) & (folds["dataset"].str.startswith("IMS"))
    ]
    for scheme in ("empirical",):
        subset = ims_problem[ims_problem["scheme"] == scheme]
        ratio = float((subset["inflation_ratio"] > 1.0).mean()) if len(subset) else float("nan")
        print(
            f"P1（{scheme}）问题组膨胀比 > 1 的折占比 = {ratio:.2%} "
            f"（{int((subset['inflation_ratio'] > 1.0).sum())}/{len(subset)}）"
            f" → {'成立' if ratio >= 0.25 else '不成立'}"
        )

    # P2：收缩是否降低问题组膨胀比（批次内配对，按批次报告方向）
    print("\nP2：各批次上收缩对问题组膨胀比的改变（正 = 降低）")
    per_run_rows = []
    for dataset in sorted(folds["dataset"].unique()):
        for scheme in ("ledoit_wolf", "cond_target"):
            empirical = folds[
                (folds["dataset"] == dataset)
                & (folds["feature_group"] == problem)
                & (folds["scheme"] == "empirical")
            ].set_index("held_out")["inflation_ratio"]
            shrunk = folds[
                (folds["dataset"] == dataset)
                & (folds["feature_group"] == problem)
                & (folds["scheme"] == scheme)
            ].set_index("held_out")["inflation_ratio"]
            common = empirical.index.intersection(shrunk.index)
            delta = (empirical.loc[common] - shrunk.loc[common]).mean()
            per_run_rows.append(
                {
                    "dataset": dataset,
                    "scheme": scheme,
                    "mean_inflation_reduction": float(delta),
                }
            )
            print(f"  {dataset:18s} {scheme:12s} Δ膨胀比 = {delta:+.3f}")

    per_run = pd.DataFrame(per_run_rows)
    per_run.to_csv(OUTPUT_DIR / "exp_v1_07_inflation_reduction.csv", index=False)

    ims_rows = per_run[per_run["dataset"].str.startswith("IMS")]
    for scheme in ("ledoit_wolf", "cond_target"):
        subset = ims_rows[ims_rows["scheme"] == scheme]
        positive = int((subset["mean_inflation_reduction"] > 0).sum())
        print(
            f"  → {scheme}: {positive}/{len(subset)} 个 IMS 批次方向为正"
            f" → {'成立' if positive >= len(subset) else '部分/不成立'}"
        )

    # P3：跨数据集方向一致性（IMS vs MFPT）
    print("\nP3：跨数据集方向一致性（问题组，Δ膨胀比 > 0）")
    for scheme in ("ledoit_wolf", "cond_target"):
        ims_positive = int(
            (ims_rows[ims_rows["scheme"] == scheme]["mean_inflation_reduction"] > 0).sum()
        )
        mfpt_rows = per_run[
            (per_run["dataset"] == "MFPT") & (per_run["scheme"] == scheme)
        ]
        mfpt_positive = int((mfpt_rows["mean_inflation_reduction"] > 0).sum())
        consistent = ims_positive == len(ims_rows[ims_rows["scheme"] == scheme]) and mfpt_positive >= 1
        print(
            f"  {scheme}: IMS {ims_positive}/3 为正，MFPT {mfpt_positive}/1 为正"
            f" → {'一致' if consistent else '不一致'}"
        )

    print("\n=== 强制稳健性检查（IMS 用 0.5 s 窗重跑）===")
    summary_w05 = (
        folds_w05.groupby(["dataset", "feature_group", "scheme"], as_index=False)
        .agg(
            median_inflation=("inflation_ratio", "median"),
            mean_fp_rate=("fp_rate", "mean"),
            folds=("inflation_ratio", "size"),
        )
    )
    ims_problem_w05 = summary_w05[
        (summary_w05["feature_group"] == problem)
        & (summary_w05["dataset"].str.startswith("IMS"))
        & (summary_w05["scheme"].isin(["empirical", "ledoit_wolf", "cond_target"]))
    ]
    print(ims_problem_w05.to_string(index=False))

    # ==== 图 ====
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted(summary["dataset"].unique())
    schemes = ["empirical", "ledoit_wolf", "cond_target"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.25
    for axis, group_name, title in (
        (axes[0], problem, "Problem group: inflation ratio (median over folds)"),
        (axes[1], control, "Control group: false-positive rate (mean over folds)"),
    ):
        for index, scheme in enumerate(schemes):
            values = []
            for dataset in datasets:
                row = summary[
                    (summary["dataset"] == dataset)
                    & (summary["feature_group"] == group_name)
                    & (summary["scheme"] == scheme)
                ]
                values.append(
                    float(row["median_inflation"].iloc[0])
                    if axis is axes[0] and len(row)
                    else (float(row["mean_fp_rate"].iloc[0]) if len(row) else 0.0)
                )
            axis.bar(
                np.arange(len(datasets)) + (index - 1) * width,
                values,
                width=width,
                label=scheme,
            )
        axis.set_xticks(np.arange(len(datasets)))
        axis.set_xticklabels(datasets, rotation=15, ha="right", fontsize=8)
        axis.axhline(1.0, linestyle="--", color="gray")
        axis.set_title(title)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "exp_v1_07_cross_dataset.png", dpi=150)
    plt.close(figure)

    print(f"\n已保存：{OUTPUT_DIR / 'exp_v1_07_folds.csv'}")
    print(f"已保存：{OUTPUT_DIR / 'exp_v1_07_summary.csv'}")
    print(f"已保存：{OUTPUT_DIR / 'exp_v1_07_folds_w0.5.csv'}")
    print(f"已保存：{FIGURES_DIR / 'exp_v1_07_cross_dataset.png'}")


if __name__ == "__main__":
    main()
