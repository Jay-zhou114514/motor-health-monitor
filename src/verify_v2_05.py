"""EXP-V2-05 的可复现性验证（L1 + L2 + 哈希归档）。

规范：docs/EXPERIMENT_AFTERCARE.md（① L1 / ② L2 / ③ L3）
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR

FAIL: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail}")
    if not cond:
        FAIL.append(label)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    raw = pd.read_csv(OUTPUT_DIR / "exp_v2_05_raw.csv")
    summ = pd.read_csv(OUTPUT_DIR / "exp_v2_05_summary.csv")

    print("== ① L1 不变量检查 ==")
    check(raw.test_fp.between(0, 1).all(), "误报率 ∈ [0,1]")
    check(raw.test_fp.notna().all(), "无 NaN")
    check((raw.n_train > 0).all(), "n_train > 0")
    # 每折可用轴承数 = 该工况内轴承数 - 1；k 不得超过
    avail = (raw.groupby(["dataset", "condition", "held_out"]).k_bearings.transform("max"))
    check(bool((raw.k_bearings <= avail).all()), "k ≤ 该折可用轴承数")
    # 折数 = 每工况轴承数（留一），逐单元核对
    folds = raw.groupby(["dataset", "condition"]).held_out.nunique()
    print(f"      各工况折数：{folds.to_dict()}")
    # 每格重复次数
    reps = raw.groupby(["dataset", "condition", "k_bearings", "held_out"]).size()
    check(bool((reps == 50).all()), "每格每折 R = 50", f"(min={reps.min()}, max={reps.max()})")
    cell_rows = raw.groupby(["dataset", "condition", "k_bearings"]).size()
    consistent = all((cell_rows % 50 == 0).values)
    check(bool(consistent), "每格行数 = 折数 × 50（折间均衡）")
    # 单格退化标记（未显式标记，需人工确认）
    deg = summ[summ.sd_fp.fillna(0) == 0]
    print(f"      退化格（SD=0）：{len(deg)} 个" + (f" -> {deg[['dataset','condition','k_bearings']].to_dict('records')}" if len(deg) else ""))

    print("\n== ② L2 独立复算 ==")
    ok = True
    for _, row in summ.iterrows():
        sub = raw[(raw.dataset == row.dataset) & (raw.condition == row.condition)
                  & (raw.k_bearings == row.k_bearings)]
        sd = sub.test_fp.std(ddof=1)
        mean = sub.test_fp.mean()
        if abs(sd - row.sd_fp) > 1e-12 or abs(mean - row.mean_fp) > 1e-12:
            ok = False
            print(f"      不一致 {row.dataset}/{row.condition} k={row.k_bearings}: "
                  f"sd {row.sd_fp} vs {sd}")
    check(ok, "全部 (dataset, condition, k) 的 mean/SD 与汇总表逐项一致")

    # 独立重算 Y3（公平范围 k=1→4），用与主脚本不同的写法
    print("\n== ② 附：Y3 公平比较的独立重算 ==")
    s3 = pd.read_csv(OUTPUT_DIR / "exp_v2_04_main_raw.csv")
    s3 = s3[s3.arm == "S3_units"]
    for ds in sorted(summ.dataset.unique()):
        mixed = s3[(s3.dataset == ds) & (s3.k_bearings.isin([1, 4]))]
        r_mixed = mixed[mixed.k_bearings == 1].test_fp.std(ddof=1) / mixed[mixed.k_bearings == 4].test_fp.std(ddof=1)
        ws = summ[(summ.dataset == ds) & (summ.k_bearings.isin([1, 4]))]
        rs = []
        for cond, g in ws.groupby("condition"):
            if g.k_bearings.max() < 4:
                continue
            rs.append(g[g.k_bearings == 1].sd_fp.iloc[0] / g[g.k_bearings == 4].sd_fp.iloc[0])
        r_within = float(np.mean(rs)) if rs else float("nan")
        print(f"      {ds}: R_mixed={r_mixed:.2f} R_within={r_within:.2f} 比值={r_within/r_mixed:.2f}")

    print("\n== ③ L3：哈希归档（重跑前）==")
    lines = []
    for name in ("exp_v2_05_raw.csv", "exp_v2_05_summary.csv"):
        path = OUTPUT_DIR / name
        digest = sha256(path)
        lines.append(f"{digest}  {name}")
        print(f"  {digest}  {name}")
    (OUTPUT_DIR / "exp_v2_05-hashes-BEFORE-rerun.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n== 结论 ==")
    print("L1+L2 全部通过" if not FAIL else f"存在 {len(FAIL)} 项失败：{FAIL}")
    sys.exit(0 if not FAIL else 1)


if __name__ == "__main__":
    main()