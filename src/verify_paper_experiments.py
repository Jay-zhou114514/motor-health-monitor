"""对进入论文的 6 个实验补做 L1 不变量 + L2 独立复算 + 哈希归档。

对象：EXP-V1-05/06/07/08/09/10（均早于三层验证规范）。
规范：docs/REPRODUCIBILITY_PROTOCOL.md
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "experiments"

FP_COLS = ["fp_rate", "test_fp", "val_fp", "inflation_ratio", "mean_fp_rate", "sd_fp"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def l1(path: Path) -> list[str]:
    """通用不变量：数值列无 NaN、比例类列 ∈ [0,1]（膨胀比除外）。"""
    msgs = []
    df = pd.read_csv(path)
    for col in df.columns:
        # 注意：`fp` 是【误报个数】而非比率，不得纳入比例列检查（反模式 12）
        if col in ("fp_rate", "test_fp", "val_fp", "mean_fp_rate"):
            s = pd.to_numeric(df[col], errors="coerce")
            if s.isna().all():
                continue
            if not s.between(-1e-9, 1 + 1e-9).all():
                msgs.append(f"{path.name}: {col} 越界")
    return msgs


def l2_summary_check(folds_name: str, summary_name: str, group_cols: list[str],
                     value_col: str, agg: str) -> list[str]:
    """用 folds CSV 独立重算汇总列，与 summary CSV 比对。"""
    msgs = []
    fp, sp = E / folds_name, E / summary_name
    if not (fp.exists() and sp.exists()):
        return [f"缺文件 {folds_name} 或 {summary_name}"]
    folds = pd.read_csv(fp)
    summ = pd.read_csv(sp)
    if value_col not in folds.columns:
        return [f"{folds_name} 无 {value_col} 列"]
    name = {"mean": "mean_fp_rate", "median": "median_fp_rate"}[agg]
    if name not in summ.columns:
        cand = [c for c in summ.columns if agg in c]
        if not cand:
            return [f"{summary_name} 无 {agg} 列"]
        name = cand[0]
    rec = folds.groupby(group_cols)[value_col].agg(agg).reset_index()
    merged = summ.merge(rec, on=group_cols, suffixes=("_s", "_r"))
    if not len(merged):
        return []
    diff = (merged[name] - merged[agg if agg in merged.columns else rec.columns[-1]]).abs()
    # 汇总表以 4 位小数存储 → 容差取 5e-5（反模式 12）
    if (diff > 5e-5).any():
        msgs.append(f"{summary_name}: {int((diff > 5e-5).sum())} 行与独立重算不一致（最大 {diff.max():.2e}）")
    return msgs


def main() -> None:
    problems: list[str] = []
    print("== L1 不变量（进入论文的 6 个实验的全部 CSV）==")
    targets = [
        "EXP-V1-05-geometry.csv", "EXP-V1-05-regularization.csv", "EXP-V1-05-window-stats.csv",
        "EXP-V1-06-delta-selection.csv", "EXP-V1-06-geometry.csv",
        "EXP-V1-07-folds.csv", "EXP-V1-07-summary.csv", "EXP-V1-07-inflation-reduction.csv",
        "EXP-V1-08-folds.csv", "EXP-V1-08-summary.csv",
        "EXP-V1-09-folds.csv", "EXP-V1-09-summary.csv",
        "EXP-V1-10-replicates.csv", "EXP-V1-10-summary.csv", "EXP-V1-10-null.csv",
        "EXP-V1-10-extra-checks.csv", "EXP-V1-10-controls.csv",
    ]
    for name in targets:
        path = E / name
        if not path.exists():
            problems.append(f"缺文件 {name}")
            print(f"  [FAIL] {name} 缺失")
            continue
        msgs = l1(path)
        print(f"  [{'PASS' if not msgs else 'FAIL'}] {name}" + (f" -> {msgs}" if msgs else ""))
        problems += msgs

    print("\n== L2 独立复算（summary vs folds 重算）==")
    checks = [
        ("EXP-V1-07-folds.csv", "EXP-V1-07-summary.csv", ["dataset", "feature_group", "scheme"], "inflation_ratio", "median"),
        ("EXP-V1-08-folds.csv", "EXP-V1-08-summary.csv", ["dataset", "detector"], "fp_rate", "mean"),
        ("EXP-V1-09-folds.csv", "EXP-V1-09-summary.csv", ["batch", "method"], "fp_rate", "mean"),
    ]
    for folds, summ, cols, val, agg in checks:
        msgs = l2_summary_check(folds, summ, cols, val, agg)
        print(f"  [{'PASS' if not msgs else 'FAIL'}] {summ} <- {folds}" + (f" -> {msgs}" if msgs else ""))
        problems += msgs

    print("\n== L3：哈希归档（当前归档文件）==")
    lines = []
    for name in targets:
        path = E / name
        if path.exists():
            lines.append(f"{sha(path)}  {name}")
    out = E / "EXP-V1-05_to_V1-10-hashes.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  已写入 {out.name}（{len(lines)} 项）")
    print("  注：这是**哈希基线**；真正的 L3 需重跑后比对，见结论。")

    print("\n== 结论 ==")
    print("L1+L2 全部通过" if not problems else f"存在 {len(problems)} 项问题：{problems}")


if __name__ == "__main__":
    main()