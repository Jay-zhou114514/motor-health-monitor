"""EXP-V3-01d 的 L1 不变量与 L2 独立复算。

预注册：experiments/EXP-V3-01-preregistration-amendment-4.md §4
L3（确定性重跑逐字节比对）由单独的重跑完成，本脚本只做 L1 与 L2。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR

ALPHA = 0.01
LAMBDA_GRID = {0.0, 0.25, 0.5, 0.75, 1.0}


class Tee:
    def __init__(self, path: Path):
        self.handle = open(path, "w", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        self.handle.write(text)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.handle.flush()


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> None:
    sys.stdout = Tee(OUTPUT_DIR / "EXP-V3-01d-L1L2.txt")
    runs = pd.read_csv(OUTPUT_DIR / "exp_v3_01d_runs.csv")
    summary = pd.read_csv(OUTPUT_DIR / "exp_v3_01d_summary.csv")
    all_ok = True

    print("=== L1 不变量 ===")
    all_ok &= check("率在 [0,1] 内", bool(((runs.rate >= 0) & (runs.rate <= 1)).all()),
                    f"min={runs.rate.min():.4f} max={runs.rate.max():.4f}")
    all_ok &= check("无 NaN 率", bool(runs.rate.notna().all()))
    lam = runs[runs.method == "PP_partial_pooling"].param
    all_ok &= check("λ 在网格内", bool(set(np.round(lam.unique(), 2)) <= LAMBDA_GRID),
                    f"取值={sorted(set(np.round(lam.unique(), 2)))}")
    all_ok &= check("每实例窗口数 > 500", bool((runs.n_test_windows > 500).all()),
                    f"min={int(runs.n_test_windows.min())} median={int(runs.n_test_windows.median())}")
    for prot, expect in (("P1_strict", 20), ("P2_leaky", 23)):
        sub = runs[runs.protocol == prot]
        all_ok &= check(f"{prot} 训练实例数 == {expect}", bool((sub.n_train_units == expect).all()),
                        f"取值={sorted(sub.n_train_units.unique())}")
        all_ok &= check(f"{prot} 行数 == 144", len(sub) == 144, f"实际={len(sub)}")
        all_ok &= check(f"{prot} 测试实例数 == 24", sub.test_instance.nunique() == 24,
                        f"实际={sub.test_instance.nunique()}")

    print("\n=== L2 独立复算（另一条代码路径，直接由 runs.csv 重算） ===")
    mismatches = 0
    for (prot, det, method), g in runs.groupby(["protocol", "detector", "method"]):
        sd_pp = float(np.std(g.rate.to_numpy(), ddof=1) * 100)
        mean_rate = float(g.rate.mean())
        row = summary[(summary.protocol == prot) & (summary.detector == det)
                      & (summary.method == method)]
        if not len(row):
            print(f"  [FAIL] 汇总缺少 {prot}/{det}/{method}")
            mismatches += 1
            continue
        d_sd = abs(sd_pp - float(row.sd_pp.iloc[0]))
        d_mean = abs(mean_rate - float(row.mean_rate.iloc[0]))
        if d_sd > 1e-6 or d_mean > 1e-9:
            print(f"  [FAIL] {prot}/{det}/{method}: ΔSD={d_sd:.2e} Δmean={d_mean:.2e}")
            mismatches += 1
    all_ok &= check("汇总与复算逐项一致", mismatches == 0, f"不一致项={mismatches}")

    print("\n=== 判定（H5/H10 由 P1 单独判定） ===")
    for det, g in runs[runs.protocol == "P1_strict"].groupby("detector"):
        b1 = g[g.method == "B1_pooled_quantile"].rate.to_numpy()
        pp = g[g.method == "PP_partial_pooling"].rate.to_numpy()
        sd1, sdp = np.std(b1, ddof=1) * 100, np.std(pp, ddof=1) * 100
        drop = (1 - sdp / sd1) * 100 if sd1 > 0 else float("nan")
        cover = 0.5 * ALPHA <= pp.mean() <= 2 * ALPHA
        print(f"  {det:<14} drop={drop:7.1f}%  mean_rate={pp.mean():.4f}  coverage={'OK' if cover else 'FAIL'}")

    print(f"\n总判定：{'L1/L2 全部通过' if all_ok else '存在未通过项（见上）'}")
    print("L3：待确定性重跑完成后比对（排除计时列）")


if __name__ == "__main__":
    main()
