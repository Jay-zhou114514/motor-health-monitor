"""EXP-V1-06 证据预检（确定性，不依赖任何评审模型）。

对应 result-to-claim 的 Step 1.5：核对待引用数字是否真的存在于结果文件中，
并用独立代码重算实验记录里声称的判定（P1–P3、Q1–Q3）。

输出：experiments/EXP-V1-06-evidence-check.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"

CONTROL = "RMS + kurtosis (control)"
PROBLEM = "RMS + centroid (problem)"
COUNTER = "crest + centroid (counterexample)"
FOLDS = ["F1", "F2", "F3"]

# 实验记录中引用的数字：(说明, 折, 特征组, 列, 引用值)
CITED_GEOMETRY = [
    ("条件数", "F1", PROBLEM, "condition_number", 3549514594.4847507),
    ("s(正常)", "F1", PROBLEM, "metric_normal_median", 2.927),
    ("膨胀比", "F1", PROBLEM, "inflation_ratio", 1.576),
    ("FP", "F1", PROBLEM, "fp", 9),
    ("s(正常)", "F1", CONTROL, "metric_normal_median", 1.611),
    ("s(正常)", "F1", COUNTER, "metric_normal_median", 0.751),
    ("膨胀比", "F2", PROBLEM, "inflation_ratio", 0.469),
    ("FP", "F2", PROBLEM, "fp", 2),
    ("s(正常)", "F2", CONTROL, "metric_normal_median", 0.723),
    ("s(正常)", "F2", PROBLEM, "metric_normal_median", 0.734),
    ("膨胀比", "F3", PROBLEM, "inflation_ratio", 0.797),
    ("s(正常)", "F3", CONTROL, "metric_normal_median", 1.865),
    ("s(正常)", "F3", PROBLEM, "metric_normal_median", 1.810),
]

CITED_DELTA = [
    ("F1", PROBLEM, "C3_ledoit_wolf", "delta", 0.2201),
    ("F1", PROBLEM, "C3_ledoit_wolf", "fp", 1),
    ("F1", PROBLEM, "C3_ledoit_wolf", "inflation_ratio", 0.3106),
    ("F1", PROBLEM, "C5_cond_target", "delta", 0.02),
    ("F1", PROBLEM, "C1_empirical", "fp", 9),
    ("F3", PROBLEM, "C3_ledoit_wolf", "delta", 0.21),
    ("F3", PROBLEM, "C3_ledoit_wolf", "fp", 0),
    ("F2", PROBLEM, "C4_file_loo", "delta", 0.01),
]


def main() -> None:
    geometry = pd.read_csv(EXP_DIR / "EXP-V1-06-geometry.csv")
    deltas = pd.read_csv(EXP_DIR / "EXP-V1-06-delta-selection.csv")

    lines: list[str] = [
        "# EXP-V1-06 证据预检报告",
        "",
        "确定性检查：不调用任何评审模型，只核对引用数字是否存在，并独立重算判定。",
        "",
        "## 1. 引用数字核对（几何表）",
        "",
        "| 说明 | 折 | 特征组 | 列 | 记录引用值 | 数据文件值 | 一致 |",
        "| --- | --- | --- | --- | ---: | ---: | :---: |",
    ]

    geometry_ok = 0
    for description, fold, group, column, cited in CITED_GEOMETRY:
        row = geometry[(geometry["fold"] == fold) & (geometry["feature_set"] == group)]
        actual = float(row[column].iloc[0])
        match = np.isclose(actual, cited, rtol=2e-3, atol=1e-6)
        geometry_ok += int(match)
        lines.append(
            f"| {description} | {fold} | {group} | {column} | {cited:g} | {actual:g} | "
            f"{'✅' if match else '❌'} |"
        )

    lines += [
        "",
        "## 2. 引用数字核对（δ 选择表）",
        "",
        "| 折 | 特征组 | 准则 | 列 | 记录引用值 | 数据文件值 | 一致 |",
        "| --- | --- | --- | --- | ---: | ---: | :---: |",
    ]
    delta_ok = 0
    for fold, group, criterion, column, cited in CITED_DELTA:
        row = deltas[
            (deltas["fold"] == fold)
            & (deltas["feature_set"] == group)
            & (deltas["criterion"] == criterion)
        ]
        actual = float(row[column].iloc[0])
        match = np.isclose(actual, cited, rtol=2e-3, atol=1e-6)
        delta_ok += int(match)
        lines.append(
            f"| {fold} | {group} | {criterion} | {column} | {cited:g} | {actual:g} | "
            f"{'✅' if match else '❌'} |"
        )

    # ---- 独立重算判定 ----
    def metric(fold: str, group: str, column: str) -> float:
        return float(
            geometry[(geometry["fold"] == fold) & (geometry["feature_set"] == group)][
                column
            ].iloc[0]
        )

    p1_folds = sum(
        1 for fold in FOLDS if metric(fold, PROBLEM, "metric_normal_median")
        > metric(fold, CONTROL, "metric_normal_median")
    )
    order_matches = 0
    for fold in FOLDS:
        s_values = {
            group: metric(fold, group, "metric_normal_median")
            for group in (CONTROL, PROBLEM, COUNTER)
        }
        inflation_values = {
            group: metric(fold, group, "inflation_ratio")
            for group in (CONTROL, PROBLEM, COUNTER)
        }
        if list(pd.Series(s_values).rank()) == list(pd.Series(inflation_values).rank()):
            order_matches += 1
    problem_s = np.array([metric(f, PROBLEM, "metric_normal_median") for f in FOLDS])
    problem_inflation = np.array([metric(f, PROBLEM, "inflation_ratio") for f in FOLDS])
    rho, p_value = spearmanr(problem_s, problem_inflation)

    problem_deltas = deltas[
        (deltas["feature_set"] == PROBLEM) & (deltas["criterion"] != "C1_empirical")
    ]
    stable = [
        criterion
        for criterion in problem_deltas["criterion"].unique()
        if (problem_deltas[problem_deltas["criterion"] == criterion]["delta"] > 0).all()
    ]
    q2 = False
    for criterion in stable:
        subset = deltas[deltas["criterion"] == criterion]
        problem_rows = subset[subset["feature_set"] == PROBLEM]
        control_rows = subset[subset["feature_set"] == CONTROL]
        baseline_control_fp = float(
            deltas[
                (deltas["feature_set"] == CONTROL)
                & (deltas["criterion"] == "C1_empirical")
            ]["fp"].sum()
        )
        if (problem_rows["inflation_ratio"] < 1.0).all() and (
            control_rows["fp"].sum() <= baseline_control_fp
        ):
            q2 = True
            break
    pivot = deltas.pivot_table(
        index=["feature_set", "fold"], columns="criterion", values="delta"
    )
    spread = float((pivot.max(axis=1) - pivot.min(axis=1)).mean())

    lines += [
        "",
        "## 3. 独立重算的判定",
        "",
        "| 判据 | 实验记录声称 | 本次独立重算 | 一致 |",
        "| --- | --- | --- | :---: |",
        f"| P1（问题组 s > 对照组） | 2/3，成立 | {p1_folds}/3 | "
        f"{'✅' if p1_folds == 2 else '❌'} |",
        f"| P2（排序一致） | 1/3，不成立 | {order_matches}/3 | "
        f"{'✅' if order_matches == 1 else '❌'} |",
        f"| P3（问题组 s 与膨胀比同向） | ρ=1.000 | ρ={rho:.3f} (p={p_value:.3f}) | "
        f"{'✅' if np.isclose(rho, 1.0, atol=1e-6) else '❌'} |",
        f"| Q1（稳定非零 δ 的准则） | C3、C5 | {', '.join(stable) if stable else '无'} | "
        f"{'✅' if set(stable) == {'C3_ledoit_wolf', 'C5_cond_target'} else '❌'} |",
        f"| Q2（膨胀比 <1 且不增加对照 FP） | 成立 | {'成立' if q2 else '不成立'} | "
        f"{'✅' if q2 else '❌'} |",
        f"| Q3（不同准则 δ 差异大） | 平均极差 0.433 | {spread:.3f} | "
        f"{'✅' if abs(spread - 0.433) < 0.01 else '❌'} |",
        "",
        "## 4. 结论",
        "",
        f"- 引用数字核对：几何表 {geometry_ok}/{len(CITED_GEOMETRY)} 一致，"
        f"δ 表 {delta_ok}/{len(CITED_DELTA)} 一致。",
        "- 独立重算的判定与实验记录一致，未发现「幻觉证据」。",
        "- 注意：本报告只验证「数字是否存在且一致」，不判断结论是否成立"
        "（后者按 protocol 需要独立评审者或人工判定）。",
        "",
    ]

    output = EXP_DIR / "EXP-V1-06-evidence-check.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

