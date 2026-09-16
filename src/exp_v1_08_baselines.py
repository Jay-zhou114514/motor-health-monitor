"""EXP-V1-08：单类检测器基线对比（预注册见
experiments/EXP-V1-08-preregistration.md）。

四种检测器（全部只用健康窗口训练）：
  A 3σ RMS 阈值 / B 马氏距离 / C One-Class SVM / D Isolation Forest

评测：
- IMS 1st/2nd/4th：批次内 leave-one-file-out，评"误报率"
- MFPT：留出一个正常文件 + 全部故障文件，评 FP rate / AUROC / 10% 误报预算下的召回 / F1

输出：
- outputs/exp_v1_08_folds.csv
- outputs/exp_v1_08_summary.csv
- outputs/figures/exp_v1_08_baselines.png
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM

import data_loading
from config import FIGURES_DIR, OUTPUT_DIR
from covariance_geometry import (
    center,
    empirical_covariance,
    mahalanobis_distances,
)
from features import extract_features

ROOT = Path(__file__).resolve().parents[1]
IMS_ROOT = ROOT / "data" / "raw" / "ims"
IMS_SR = 20_000.0
QUANTILE = 0.99
FP_BUDGET = 0.10
FEATURES = ["rms", "crest_factor", "kurtosis", "centroid_hz"]


def ims_files(subset: str) -> list[Path]:
    files = sorted((IMS_ROOT / subset).glob("*.txt"))
    if subset == "1st_test":
        files = [p for p in files if p.name.startswith("2003.10.22")]
    return files


def load_ims_signal(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t", header=None, usecols=[0])
    return frame.iloc[:, 0].to_numpy(dtype=float)


def load_healthy_records() -> dict[str, dict[str, pd.DataFrame]]:
    """返回 {dataset: {file: feature_table}}。"""
    datasets: dict[str, dict[str, pd.DataFrame]] = {}
    for subset in ("1st_test", "2nd_test", "4th_test"):
        tables: dict[str, pd.DataFrame] = {}
        for path in ims_files(subset):
            table = extract_features(
                load_ims_signal(path),
                IMS_SR,
                window_sec=0.25,
                step_sec=0.125,
                record_id=path.name,
                label="normal",
            ).dropna(subset=FEATURES)
            if len(table):
                tables[path.name] = table
        datasets[f"IMS {subset}"] = tables

    mfpt_normal: dict[str, pd.DataFrame] = {}
    mfpt_fault: dict[str, pd.DataFrame] = {}
    for record in data_loading.load_records():
        table = extract_features(
            record["signal"],
            record["sr"],
            window_sec=1.0,
            step_sec=0.5,
            record_id=record["file"],
            label=record["condition"],
        ).dropna(subset=FEATURES)
        if not len(table):
            continue
        if record["condition"] == "normal":
            mfpt_normal[record["file"]] = table
        elif record["split"] == "test":
            mfpt_fault[record["file"]] = table
    datasets["MFPT"] = {"normal": mfpt_normal, "fault": mfpt_fault}
    return datasets


class RMSThreshold:
    name = "A: 3-sigma RMS"

    def fit(self, train: np.ndarray):
        scores = train[:, 0]
        self.threshold_ = float(np.quantile(scores, QUANTILE))
        return self

    def score(self, data: np.ndarray) -> np.ndarray:
        return data[:, 0]


class Mahalanobis:
    name = "B: Mahalanobis"

    def fit(self, train: np.ndarray):
        self.mean_ = train.mean(axis=0)
        covariance = empirical_covariance(center(train))
        covariance += np.eye(covariance.shape[0]) * 1e-12
        self.covariance_ = covariance
        distances = mahalanobis_distances(train, self.mean_, covariance)
        self.threshold_ = float(np.quantile(distances, QUANTILE))
        return self

    def score(self, data: np.ndarray) -> np.ndarray:
        return mahalanobis_distances(data, self.mean_, self.covariance_)


class OCSVM:
    name = "C: One-Class SVM"

    def fit(self, train: np.ndarray):
        self.mean_ = train.mean(axis=0)
        self.std_ = train.std(axis=0, ddof=1)
        self.std_[self.std_ == 0] = 1.0
        standardized = (train - self.mean_) / self.std_
        self.model_ = OneClassSVM(kernel="rbf", nu=0.01, gamma="scale").fit(standardized)
        scores = -self.model_.decision_function(standardized)
        self.threshold_ = float(np.quantile(scores, QUANTILE))
        return self

    def score(self, data: np.ndarray) -> np.ndarray:
        standardized = (data - self.mean_) / self.std_
        return -self.model_.decision_function(standardized)


class IForest:
    name = "D: Isolation Forest"

    def fit(self, train: np.ndarray):
        self.mean_ = train.mean(axis=0)
        self.std_ = train.std(axis=0, ddof=1)
        self.std_[self.std_ == 0] = 1.0
        standardized = (train - self.mean_) / self.std_
        self.model_ = IsolationForest(
            n_estimators=200, random_state=0, contamination="auto"
        ).fit(standardized)
        scores = -self.model_.score_samples(standardized)
        self.threshold_ = float(np.quantile(scores, QUANTILE))
        return self

    def score(self, data: np.ndarray) -> np.ndarray:
        standardized = (data - self.mean_) / self.std_
        return -self.model_.score_samples(standardized)


DETECTORS = [RMSThreshold, Mahalanobis, OCSVM, IForest]


def run_healthy_only(tables: dict[str, pd.DataFrame]) -> list[dict]:
    rows: list[dict] = []
    names = sorted(tables)
    for held_out in names:
        train_names = [n for n in names if n != held_out]
        train = np.vstack([tables[n][FEATURES].to_numpy(float) for n in train_names])
        test = tables[held_out][FEATURES].to_numpy(float)
        for detector_class in DETECTORS:
            detector = detector_class()
            start = time.perf_counter()
            detector.fit(train)
            scores = detector.score(test)
            elapsed = time.perf_counter() - start
            fp = int((scores > detector.threshold_).sum())
            rows.append(
                {
                    "dataset": None,
                    "held_out": held_out,
                    "detector": detector.name,
                    "n_test_windows": len(test),
                    "fp": fp,
                    "fp_rate": fp / len(test),
                    "threshold": detector.threshold_,
                    "seconds": elapsed,
                }
            )
    return rows


def run_mfpt(normal: dict[str, pd.DataFrame], fault: dict[str, pd.DataFrame]) -> list[dict]:
    rows: list[dict] = []
    names = sorted(normal)
    for held_out in names:
        train_names = [n for n in names if n != held_out]
        train = np.vstack([normal[n][FEATURES].to_numpy(float) for n in train_names])
        holdout = normal[held_out][FEATURES].to_numpy(float)
        faults = np.vstack([t[FEATURES].to_numpy(float) for t in fault.values()])
        for detector_class in DETECTORS:
            detector = detector_class()
            start = time.perf_counter()
            detector.fit(train)
            score_normal = detector.score(holdout)
            score_fault = detector.score(faults)
            elapsed = time.perf_counter() - start
            fp = int((score_normal > detector.threshold_).sum())
            tp = int((score_fault > detector.threshold_).sum())
            labels = np.concatenate(
                [np.zeros(len(score_normal)), np.ones(len(score_fault))]
            )
            scores = np.concatenate([score_normal, score_fault])
            auroc = float(roc_auc_score(labels, scores))
            budget_threshold = float(np.quantile(score_normal, 1.0 - FP_BUDGET))
            recall_at_budget = float((score_fault > budget_threshold).mean())
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + len(score_fault))
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
            rows.append(
                {
                    "dataset": "MFPT",
                    "held_out": held_out,
                    "detector": detector.name,
                    "n_test_windows": len(score_normal) + len(score_fault),
                    "fp": fp,
                    "fp_rate": fp / len(score_normal),
                    "auroc": auroc,
                    "recall_at_10pct_fp": recall_at_budget,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "threshold": detector.threshold_,
                    "seconds": elapsed,
                }
            )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_healthy_records()
    rows: list[dict] = []
    for name in ("IMS 1st_test", "IMS 2nd_test", "IMS 4th_test"):
        for row in run_healthy_only(datasets[name]):
            row["dataset"] = name
            rows.append(row)
    rows.extend(run_mfpt(datasets["MFPT"]["normal"], datasets["MFPT"]["fault"]))
    folds = pd.DataFrame(rows)
    folds.to_csv(OUTPUT_DIR / "exp_v1_08_folds.csv", index=False)

    summary = (
        folds.groupby(["dataset", "detector"], as_index=False)
        .agg(
            mean_fp_rate=("fp_rate", "mean"),
            median_fp_rate=("fp_rate", "median"),
            std_fp_rate=("fp_rate", "std"),
            mean_auroc=("auroc", "mean"),
            mean_recall_at_10pct=("recall_at_10pct_fp", "mean"),
            mean_f1=("f1", "mean"),
            seconds=("seconds", "mean"),
        )
        .round(4)
    )
    summary.to_csv(OUTPUT_DIR / "exp_v1_08_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("EXP-V1-08 单类检测器基线对比（全部只用健康窗口训练）")
    print(summary.to_string(index=False))

    print("\n=== 预注册判据 ===")
    ims = summary[summary["dataset"].str.startswith("IMS")]
    modern = ["C: One-Class SVM", "D: Isolation Forest"]
    simple = ["A: 3-sigma RMS", "B: Mahalanobis"]
    p1_hits = 0
    p1_total = 0
    for dataset in ims["dataset"].unique():
        subset = ims[ims["dataset"] == dataset]
        for modern_name in modern:
            modern_fp = float(
                subset[subset["detector"] == modern_name]["mean_fp_rate"].iloc[0]
            )
            best_simple = min(
                float(subset[subset["detector"] == name]["mean_fp_rate"].iloc[0])
                for name in simple
            )
            p1_total += 1
            p1_hits += int(modern_fp <= best_simple)
    print(
        f"P1（IMS 上现代单类方法 FP ≤ 最好的简单方法）：{p1_hits}/{p1_total} "
        f"→ {'成立' if p1_hits == p1_total else ('部分成立' if p1_hits else '不成立')}"
    )

    mfpt = summary[summary["dataset"] == "MFPT"]
    rows_out = []
    for modern_name in modern:
        modern_recall = float(
            mfpt[mfpt["detector"] == modern_name]["mean_recall_at_10pct"].iloc[0]
        )
        best_simple_recall = max(
            float(mfpt[mfpt["detector"] == name]["mean_recall_at_10pct"].iloc[0])
            for name in simple
        )
        rows_out.append((modern_name, modern_recall, best_simple_recall))
        print(
            f"  {modern_name}: 10% 误报预算下召回 = {modern_recall:.3f} "
            f"（最好的简单方法 = {best_simple_recall:.3f}）"
        )
    p2_hits = sum(1 for _, m, s in rows_out if m >= s)
    print(
        f"P2（MFPT 上现代方法在 10% 误报预算下召回 ≥ 简单方法）：{p2_hits}/{len(rows_out)} "
        f"→ {'成立' if p2_hits == len(rows_out) else ('部分成立' if p2_hits else '不成立')}"
    )

    # P3：是否存在单一方法同时 FP 最低 + 召回最高
    best_fp = {}
    best_recall = {}
    for dataset in summary["dataset"].unique():
        subset = summary[summary["dataset"] == dataset]
        best_fp[dataset] = subset.loc[subset["mean_fp_rate"].idxmin(), "detector"]
        if subset["mean_recall_at_10pct"].notna().any():
            best_recall[dataset] = subset.loc[
                subset["mean_recall_at_10pct"].idxmax(), "detector"
            ]
    print(f"\n各数据集误报最低的方法：{best_fp}")
    print(f"MFPT 召回最高的方法：{best_recall}")
    p3 = len(set(best_fp.values())) > 1
    print(f"P3（不存在普适最优方法）：{'成立' if p3 else '不成立'}")

    # ==== 图 ====
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets_sorted = list(summary["dataset"].unique())
    detectors = [d().name for d in DETECTORS]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.2
    for index, detector in enumerate(detectors):
        values = [
            float(
                summary[
                    (summary["dataset"] == dataset) & (summary["detector"] == detector)
                ]["mean_fp_rate"].iloc[0]
            )
            for dataset in datasets_sorted
        ]
        axes[0].bar(
            np.arange(len(datasets_sorted)) + (index - 1.5) * width,
            values,
            width=width,
            label=detector,
        )
    axes[0].set_xticks(np.arange(len(datasets_sorted)))
    axes[0].set_xticklabels(datasets_sorted, rotation=15, ha="right", fontsize=8)
    axes[0].set_ylabel("Mean false-positive rate")
    axes[0].set_title("False-alarm rate on held-out healthy recordings")
    axes[0].legend(fontsize=7)

    mfpt_rows = summary[summary["dataset"] == "MFPT"]
    axes[1].bar(
        np.arange(len(detectors)),
        [
            float(mfpt_rows[mfpt_rows["detector"] == d]["mean_auroc"].iloc[0])
            for d in detectors
        ],
        color="tab:blue",
    )
    axes[1].set_xticks(np.arange(len(detectors)))
    axes[1].set_xticklabels(detectors, rotation=20, ha="right", fontsize=7)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("MFPT discrimination (normal vs fault)")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "exp_v1_08_baselines.png", dpi=150)
    plt.close(figure)

    print(f"\n已保存：{OUTPUT_DIR / 'exp_v1_08_folds.csv'}")
    print(f"已保存：{OUTPUT_DIR / 'exp_v1_08_summary.csv'}")
    print(f"已保存：{FIGURES_DIR / 'exp_v1_08_baselines.png'}")


if __name__ == "__main__":
    main()
