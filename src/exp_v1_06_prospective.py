"""EXP-V1-06：前瞻性验证（预注册见 experiments/EXP-V1-06-preregistration.md）。

V1.06-A：三折 leave-one-normal-file-out 轮换，验证冻结指标
         s(x) = |(x − μ)·v_min| / √λ₁ 是否复现。
V1.06-B：比较五个预定义的 training-only 准则选择收缩强度 δ。

规则（来自预注册）：
- 指标定义已冻结，不得修改；
- 阈值与协方差只用训练窗口确定；
- 被留出正常文件与故障文件只用于评价；
- 折设计与判据（P1–P3、Q1–Q3）见预注册文件。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import data_loading
import evaluate as evaluate_module
from config import FIGURES_DIR, MAHAL_FEATURES, OUTPUT_DIR, STEP_SEC, WINDOW_SEC
from covariance_geometry import (
    canonicalize_direction,
    center,
    empirical_covariance,
    ledoit_wolf_delta,
    mahalanobis_distances,
    select_shrinkage_by_loo,
    shrinkage_covariance,
)
from features import extract_features

QUANTILE = 0.99
JITTER = 1e-12
DELTAS = np.round(np.arange(0.0, 1.0001, 0.01), 2)

GROUPS: list[tuple[str, list[str]]] = [
    ("RMS + kurtosis (control)", ["rms", "kurtosis"]),
    ("RMS + centroid (problem)", ["rms", "centroid_hz"]),
    ("crest + centroid (counterexample)", ["crest_factor", "centroid_hz"]),
]

NORMAL_FILES = ["baseline_1.mat", "baseline_2.mat", "baseline_3.mat"]
FOLDS = [
    ("F1", ["baseline_1.mat", "baseline_2.mat"], "baseline_3.mat"),
    ("F2", ["baseline_1.mat", "baseline_3.mat"], "baseline_2.mat"),
    ("F3", ["baseline_2.mat", "baseline_3.mat"], "baseline_1.mat"),
]


def build_file_tables() -> dict[str, pd.DataFrame]:
    records = data_loading.load_records(data_loading.MINIMAL_RELATIVE_FILES)
    tables: dict[str, pd.DataFrame] = {}
    for record in records:
        tables[record["file"]] = extract_features(
            record["signal"],
            record["sr"],
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
            record_id=record["file"],
            label=record["condition"],
        ).dropna(subset=MAHAL_FEATURES)
    return tables


def fault_matrices(
    tables: dict[str, pd.DataFrame], features: list[str]
) -> tuple[np.ndarray, list[str]]:
    fault_files = [
        name for name, table in tables.items() if name not in NORMAL_FILES
    ]
    blocks = [tables[name][features].to_numpy(dtype=float) for name in fault_files]
    return np.vstack(blocks), fault_files


def fold_geometry(
    tables: dict[str, pd.DataFrame],
    train_files: list[str],
    held_out: str,
    features: list[str],
) -> dict:
    train_matrix = np.vstack(
        [tables[name][features].to_numpy(dtype=float) for name in train_files]
    )
    mean = train_matrix.mean(axis=0)
    centered = center(train_matrix)
    covariance = empirical_covariance(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    minimum_direction = canonicalize_direction(eigenvectors[:, 0])
    lam_min = float(eigenvalues[0])

    def metric(samples: np.ndarray) -> np.ndarray:
        projections = (samples - mean) @ minimum_direction
        return np.abs(projections) / np.sqrt(max(lam_min, 1e-300))

    train_distances = mahalanobis_distances(train_matrix, mean, covariance)
    threshold = float(np.quantile(train_distances, QUANTILE))
    normal_samples = tables[held_out][features].to_numpy(dtype=float)
    faults, fault_files = fault_matrices(tables, features)
    normal_distances = mahalanobis_distances(normal_samples, mean, covariance)
    fault_distances = mahalanobis_distances(faults, mean, covariance)

    predictions = np.concatenate(
        [normal_distances > threshold, fault_distances > threshold]
    )
    labels = np.concatenate([np.zeros(len(normal_distances), bool), np.ones(len(fault_distances), bool)])
    metrics = evaluate_module.evaluate(labels, predictions)

    return {
        "train_files": train_files,
        "held_out": held_out,
        "n_train_windows": len(train_matrix),
        "n_normal_windows": len(normal_samples),
        "condition_number": float(np.linalg.cond(covariance)),
        "lambda_min": lam_min,
        "min_direction": minimum_direction,
        "mean": mean,
        "centered_train": centered,
        "threshold": threshold,
        "metric_normal": metric(normal_samples),
        "metric_fault": metric(faults),
        "inflation_ratio": float(np.median(normal_distances)) / threshold,
        "fp": int(metrics["fp"]),
        "fpr": float(metrics["fpr"]),
        "f1": float(metrics["f1"]),
        "recall": float(metrics["recall"]),
        "metric": metric,
    }


def file_loo_delta(
    tables: dict[str, pd.DataFrame], train_files: list[str], features: list[str]
) -> float:
    """文件级留一（只用训练文件）选择 δ。"""
    blocks = {
        name: tables[name][features].to_numpy(dtype=float) for name in train_files
    }
    best_delta, best_score = 0.0, float("-inf")
    for delta in DELTAS:
        scores = []
        for held in train_files:
            others = [name for name in train_files if name != held]
            fit_matrix = np.vstack([blocks[name] for name in others])
            mean = fit_matrix.mean(axis=0)
            covariance = shrinkage_covariance(center(fit_matrix), float(delta))
            covariance = covariance + np.eye(covariance.shape[0]) * JITTER
            inverse = np.linalg.inv(covariance)
            sign, log_det = np.linalg.slogdet(covariance)
            if sign <= 0:
                continue
            centered = blocks[held] - mean
            quadratic = np.einsum("ij,jk,ik->i", centered, inverse, centered)
            dimension = covariance.shape[0]
            log_likelihood = -0.5 * (
                dimension * np.log(2 * np.pi) + log_det + quadratic
            )
            scores.append(float(log_likelihood.mean()))
        score = float(np.mean(scores))
        if score > best_score:
            best_delta, best_score = float(delta), score
    return best_delta


def condition_target_delta(centered: np.ndarray, target: float = 100.0) -> float:
    for delta in DELTAS:
        covariance = shrinkage_covariance(centered, float(delta))
        if np.linalg.cond(covariance) <= target:
            return float(delta)
    return 1.0


def select_deltas(
    tables: dict[str, pd.DataFrame], train_files: list[str], features: list[str]
) -> dict[str, float]:
    train_matrix = np.vstack(
        [tables[name][features].to_numpy(dtype=float) for name in train_files]
    )
    centered = center(train_matrix)
    loo_delta, _ = select_shrinkage_by_loo(centered, DELTAS)
    return {
        "C1_empirical": 0.0,
        "C2_loo_window": float(loo_delta),
        "C3_ledoit_wolf": float(ledoit_wolf_delta(centered)),
        "C4_file_loo": float(file_loo_delta(tables, train_files, features)),
        "C5_cond_target": float(condition_target_delta(centered)),
    }


def evaluate_delta(
    tables: dict[str, pd.DataFrame],
    train_files: list[str],
    held_out: str,
    features: list[str],
    delta: float,
) -> dict:
    train_matrix = np.vstack(
        [tables[name][features].to_numpy(dtype=float) for name in train_files]
    )
    mean = train_matrix.mean(axis=0)
    covariance = shrinkage_covariance(center(train_matrix), float(delta))
    covariance = covariance + np.eye(covariance.shape[0]) * JITTER
    train_distances = mahalanobis_distances(train_matrix, mean, covariance)
    threshold = float(np.quantile(train_distances, QUANTILE))
    normal_samples = tables[held_out][features].to_numpy(dtype=float)
    faults, _ = fault_matrices(tables, features)
    normal_distances = mahalanobis_distances(normal_samples, mean, covariance)
    fault_distances = mahalanobis_distances(faults, mean, covariance)
    predictions = np.concatenate(
        [normal_distances > threshold, fault_distances > threshold]
    )
    labels = np.concatenate(
        [np.zeros(len(normal_distances), bool), np.ones(len(fault_distances), bool)]
    )
    metrics = evaluate_module.evaluate(labels, predictions)
    return {
        "delta": float(delta),
        "condition_number": float(np.linalg.cond(covariance)),
        "threshold": threshold,
        "fp": int(metrics["fp"]),
        "fpr": float(metrics["fpr"]),
        "f1": float(metrics["f1"]),
        "inflation_ratio": float(np.median(normal_distances)) / threshold,
    }


def main() -> None:
    tables = build_file_tables()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    geometry_rows: list[dict] = []
    delta_rows: list[dict] = []
    metric_by_fold: dict[tuple[str, str], tuple[float, float]] = {}

    for fold_name, train_files, held_out in FOLDS:
        print(f"\n########## {fold_name}: train {train_files} → hold out {held_out} ##########")
        for group_name, features in GROUPS:
            geometry = fold_geometry(tables, train_files, held_out, features)
            metric_normal_median = float(np.median(geometry["metric_normal"]))
            metric_fault_median = float(np.median(geometry["metric_fault"]))
            metric_by_fold[(fold_name, group_name)] = (
                metric_normal_median,
                geometry["inflation_ratio"],
            )
            geometry_rows.append(
                {
                    "fold": fold_name,
                    "held_out": held_out,
                    "feature_set": group_name,
                    "n_train_windows": geometry["n_train_windows"],
                    "n_normal_windows": geometry["n_normal_windows"],
                    "condition_number": geometry["condition_number"],
                    "lambda_min": geometry["lambda_min"],
                    "min_direction": ", ".join(
                        f"{f}={v:+.3f}" for f, v in zip(features, geometry["min_direction"])
                    ),
                    "metric_normal_median": round(metric_normal_median, 4),
                    "metric_fault_median": round(metric_fault_median, 4),
                    "threshold": round(geometry["threshold"], 4),
                    "inflation_ratio": round(geometry["inflation_ratio"], 3),
                    "fp": geometry["fp"],
                    "fpr": round(geometry["fpr"], 4),
                    "recall": round(geometry["recall"], 4),
                    "f1": round(geometry["f1"], 4),
                }
            )
            print(
                f"  [{group_name}] cond={geometry['condition_number']:.3e} "
                f"| s(normal)={metric_normal_median:.3f} "
                f"| s(fault)={metric_fault_median:.3f} "
                f"| inflation={geometry['inflation_ratio']:.3f} "
                f"| FP={geometry['fp']}/{geometry['n_normal_windows']} | F1={geometry['f1']:.3f}"
            )

            deltas = select_deltas(tables, train_files, features)
            for criterion, delta in deltas.items():
                result = evaluate_delta(tables, train_files, held_out, features, delta)
                delta_rows.append(
                    {
                        "fold": fold_name,
                        "feature_set": group_name,
                        "criterion": criterion,
                        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()},
                    }
                )

    geometry = pd.DataFrame(geometry_rows)
    delta_table = pd.DataFrame(delta_rows)
    geometry.to_csv(OUTPUT_DIR / "exp_v1_06_geometry.csv", index=False)
    delta_table.to_csv(OUTPUT_DIR / "exp_v1_06_delta_selection.csv", index=False)

    # ---------- 预注册判据 ----------
    print("\n########## 预注册判据 ##########")
    control = "RMS + kurtosis (control)"
    problem = "RMS + centroid (problem)"
    counter = "crest + centroid (counterexample)"

    p1_folds = 0
    for fold_name, _, _ in FOLDS:
        s_problem, _ = metric_by_fold[(fold_name, problem)]
        s_control, _ = metric_by_fold[(fold_name, control)]
        if s_problem > s_control:
            p1_folds += 1
    p1 = p1_folds >= 2
    print(f"P1（问题组 s > 对照组 s，3 折中至少 2 折）：{p1_folds}/3 → {'成立' if p1 else '不成立'}")

    order_matches = 0
    for fold_name, _, _ in FOLDS:
        s_values = {
            group: metric_by_fold[(fold_name, group)][0] for group, _ in GROUPS
        }
        inflation_values = {
            group: metric_by_fold[(fold_name, group)][1] for group, _ in GROUPS
        }
        s_rank = pd.Series(s_values).rank()
        inflation_rank = pd.Series(inflation_values).rank()
        if float((s_rank - inflation_rank).abs().max()) <= 1e-9:
            order_matches += 1
    p2 = order_matches >= 2
    print(f"P2（三组 s 排序 = 膨胀比排序，3 折中至少 2 折）：{order_matches}/3 → {'成立' if p2 else '不成立'}")

    problem_s = np.array([metric_by_fold[(f, problem)][0] for f, _, _ in FOLDS])
    problem_inflation = np.array([metric_by_fold[(f, problem)][1] for f, _, _ in FOLDS])
    rho_p3, p_p3 = spearmanr(problem_s, problem_inflation)
    p3 = bool(rho_p3 > 0)
    print(f"P3（问题组 s 与膨胀比同向）：Spearman rho={rho_p3:.3f}, p={p_p3:.3f} → {'成立' if p3 else '不成立'}")

    if p1 and p2:
        verdict_a = "支持"
    elif p1 or p2:
        verdict_a = "部分支持"
    else:
        verdict_a = "不支持"
    print(f"V1.06-A 总体判定：{verdict_a}")

    problem_deltas = delta_table[
        (delta_table["feature_set"] == problem) & (delta_table["criterion"] != "C1_empirical")
    ]
    stable_criteria = []
    for criterion in problem_deltas["criterion"].unique():
        subset = problem_deltas[problem_deltas["criterion"] == criterion]
        if (subset["delta"] > 0).all():
            stable_criteria.append(criterion)
    q1 = len(stable_criteria) > 0
    print(f"Q1（至少一个准则在 3 折都选出 δ>0）：{'成立' if q1 else '不成立'}；稳定准则={stable_criteria}")

    q2 = False
    for criterion in stable_criteria:
        subset = delta_table[delta_table["criterion"] == criterion]
        problem_rows = subset[subset["feature_set"] == problem]
        control_rows = subset[subset["feature_set"] == control]
        if (
            (problem_rows["inflation_ratio"] < 1.0).all()
            and control_rows["fp"].sum()
            <= delta_table[
                (delta_table["feature_set"] == control)
                & (delta_table["criterion"] == "C1_empirical")
            ]["fp"].sum()
        ):
            q2 = True
            break
    print(f"Q2（稳定准则把问题组膨胀比降到 <1 且不明显增加对照 FP）：{'成立' if q2 else '不成立'}")

    pivot = delta_table.pivot_table(
        index=["feature_set", "fold"], columns="criterion", values="delta"
    )
    spread = (pivot.max(axis=1) - pivot.min(axis=1)).mean()
    q3 = bool(spread > 0.1)
    print(f"Q3（不同准则选出的 δ 差异较大，平均极差={spread:.3f}）：{'成立' if q3 else '不成立'}")

    if q1 and q2:
        verdict_b = "支持"
    elif q1 or q2:
        verdict_b = "部分支持"
    else:
        verdict_b = "不支持"
    print(f"V1.06-B 总体判定：{verdict_b}")

    # ---------- 图 ----------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, column, title in (
        (axes[0], "metric_normal_median", "Frozen metric s(x) on held-out normal file"),
        (axes[1], "inflation_ratio", "Inflation ratio on held-out normal file"),
    ):
        folds = [fold for fold, _, _ in FOLDS]
        width = 0.25
        for index, (group_name, _) in enumerate(GROUPS):
            values = [
                float(
                    geometry[
                        (geometry["fold"] == fold) & (geometry["feature_set"] == group_name)
                    ][column].iloc[0]
                )
                for fold in folds
            ]
            axis.bar(
                np.arange(len(folds)) + (index - 1) * width,
                values,
                width=width,
                label=group_name,
            )
        axis.set_xticks(np.arange(len(folds)))
        axis.set_xticklabels(folds)
        axis.set_title(title)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "exp_v1_06_prospective.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    criteria = sorted(delta_table["criterion"].unique())
    width = 0.25
    for index, (group_name, _) in enumerate(GROUPS):
        values = [
            float(
                delta_table[
                    (delta_table["fold"] == fold)
                    & (delta_table["feature_set"] == group_name)
                ]
                .set_index("criterion")
                .loc[criteria, "delta"]
                .mean()
            )
            for fold in [f for f, _, _ in FOLDS]
        ]
        axis.bar(
            np.arange(len(FOLDS)) + (index - 1) * width,
            values,
            width=width,
            label=group_name,
        )
    axis.set_xticks(np.arange(len(FOLDS)))
    axis.set_xticklabels([f for f, _, _ in FOLDS])
    axis.set_ylabel("Mean selected delta over criteria")
    axis.set_title("Selected shrinkage strength by fold (mean over criteria)")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "exp_v1_06_delta_selection.png", dpi=150)
    plt.close(figure)

    print(f"\n已保存：{OUTPUT_DIR / 'exp_v1_06_geometry.csv'}")
    print(f"已保存：{OUTPUT_DIR / 'exp_v1_06_delta_selection.csv'}")
    print(f"图表目录：{FIGURES_DIR}")


if __name__ == "__main__":
    main()
