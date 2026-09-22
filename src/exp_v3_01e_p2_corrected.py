"""EXP-V3-01 批次⑤：P2（轴承级泄漏上界）的正确实现——逐个测试实例排除。

前两次 P2 的错误：
  批次③：排除测试轴承的全部 4 个工况 → 与 P1 相同；
  批次④：仍是"排除该轴承的全部 4 个测试实例" → 训练 20，仍与 P1 相同（L1 抓出）。
本脚本：对**每一个测试实例**单独构造训练集 = 其余 23 个实例（含同轴承的其他 3 个工况），
这才是"轴承级泄漏"的乐观上界。

判据 H6（来自修订 4）：P1 与 P2 的降幅差 ≤20 pp。P1 结果取 `exp_v3_01d_summary.csv`。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR
from exp_v1_08_baselines import QUANTILE
from exp_v3_01_unit_calibration import make_scorer
from exp_v3_01d_paderborn_fixed import ALPHA, calibrate, build_instances, stable_seed


class Tee:
    def __init__(self, path: Path):
        self.handle = open(path, "w", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        self.handle.write(text)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.handle.flush()


def main() -> None:
    sys.stdout = Tee(OUTPUT_DIR / "exp_v3_01e_run.log")
    print("构建实例（W-B 窗口口径，620 窗/实例）…", flush=True)
    instances = build_instances()
    keys = sorted(instances)
    print(f"实例总数 = {len(keys)}（每个测试实例的训练集 = 其余 {len(keys) - 1} 个）", flush=True)

    rows = []
    for test_key in keys:
        train_keys = [k for k in keys if k != test_key]          # ← 正确的 P2
        assert test_key not in train_keys, "训练集含测试实例"
        train_units = [instances[k] for k in train_keys]
        test_block = instances[test_key]
        for kind in ("rms", "mahalanobis", "iforest"):
            pooled = np.vstack(train_units)
            scorer = make_scorer(kind, pooled)
            b1 = float(np.quantile(scorer(pooled), QUANTILE))
            rows.append(dict(protocol="P2_leaky_corrected",
                             test_instance=f"{test_key[0]}|{test_key[1]}", detector=kind,
                             method="B1_pooled_quantile", param=np.nan,
                             rate=float((scorer(test_block) > b1).mean()),
                             n_train_units=len(train_units), n_test_windows=len(test_block)))
            lam, diag = calibrate(kind, train_units,
                                  stable_seed("V301e", test_key[0], test_key[1], kind))
            thr = diag["t_pooled"] + lam * (diag["t_bar"] - diag["t_pooled"])
            rows.append(dict(protocol="P2_leaky_corrected",
                             test_instance=f"{test_key[0]}|{test_key[1]}", detector=kind,
                             method="PP_partial_pooling", param=lam,
                             rate=float((scorer(test_block) > thr).mean()),
                             n_train_units=len(train_units), n_test_windows=len(test_block)))
        print(f"  {test_key[0]}|{test_key[1]} done", flush=True)
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "exp_v3_01e_partial.csv", index=False)

    raw = pd.DataFrame(rows)
    raw["saturated"] = raw.rate.isin([0.0, 1.0])
    raw.to_csv(OUTPUT_DIR / "exp_v3_01e_runs.csv", index=False)

    summary = (raw.groupby(["protocol", "detector", "method"])
               .agg(n=("rate", "size"),
                    sd_pp=("rate", lambda s: float(np.std(s, ddof=1) * 100)),
                    mean_rate=("rate", "mean"),
                    median_lambda=("param", "median"),
                    n_train_units=("n_train_units", "first"),
                    saturated=("saturated", "sum")).reset_index())
    summary.to_csv(OUTPUT_DIR / "exp_v3_01e_summary.csv", index=False)

    print("\n=== 批次⑤（正确的 P2）汇总 ===")
    p2 = dict(zip(summary[summary.method == "B1_pooled_quantile"].detector,
                  summary[summary.method == "B1_pooled_quantile"].sd_pp))
    p2pp = dict(zip(summary[summary.method == "PP_partial_pooling"].detector,
                    summary[summary.method == "PP_partial_pooling"].sd_pp))
    drops = {}
    for det in p2:
        drops[det] = (1 - p2pp[det] / p2[det]) * 100 if p2[det] > 0 else float("nan")
        print(f"  {det:<14} B1 {p2[det]:7.2f} pp | PP {p2pp[det]:7.2f} pp | 降 {drops[det]:7.1f}%")

    p1 = pd.read_csv(OUTPUT_DIR / "exp_v3_01d_summary.csv")
    p1sd = dict(zip(p1[p1.method == "B1_pooled_quantile"].detector,
                    p1[p1.method == "B1_pooled_quantile"].sd_pp))
    p1pp = dict(zip(p1[p1.method == "PP_partial_pooling"].detector,
                    p1[p1.method == "PP_partial_pooling"].sd_pp))
    print("\n=== H6：P1 与 P2 的降幅差 ===")
    diffs = []
    for det in p1sd:
        d1 = (1 - p1pp[det] / p1sd[det]) * 100 if p1sd[det] > 0 else float("nan")
        diffs.append(abs(d1 - drops[det]))
        print(f"  {det:<14} P1 降 {d1:7.1f}% | P2 降 {drops[det]:7.1f}% | 差 {abs(d1 - drops[det]):5.1f} pp")
    print("H6 判定:", "支持" if max(diffs) <= 20 else "不支持", f"（最大差 {max(diffs):.1f} pp）")


if __name__ == "__main__":
    main()
