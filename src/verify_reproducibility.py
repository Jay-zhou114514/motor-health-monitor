"""可复现性验证（L1 不变量 + L2 独立复算 + 哈希归档）。

规范：docs/REPRODUCIBILITY_PROTOCOL.md
用法：python verify_reproducibility.py --experiment exp_v2_04
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATASET_ROOT, OUTPUT_DIR
from runtofailure_data import FEATURES, read_pronostia_csv, healthy_size, included

FAIL: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label} {detail}")
    if not condition:
        FAIL.append(label)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def l1_invariants(prefix: str) -> None:
    print("\n== L1 不变量检查 ==")
    for name in ("main_raw", "sensitivity_raw"):
        path = OUTPUT_DIR / f"{prefix}_{name}.csv"
        if not path.exists():
            check(False, f"{name} 存在")
            continue
        d = pd.read_csv(path)
        check(d.test_fp.between(0, 1).all(), f"{name}: 误报率 ∈ [0,1]")
        check(d.test_fp.notna().all(), f"{name}: 无 NaN")
        check((d.n_train > 0).all(), f"{name}: n_train > 0")
        bad_k = d[d.k_bearings.notna() & (d.k_bearings > d.groupby(["dataset", "held_out"]).n_train.transform("max"))]
        check(len(bad_k) == 0, f"{name}: k_bearings ≤ 训练池规模", f"(违规 {len(bad_k)})")
        counts = d.groupby(["arm"]).size()
        print(f"      {name}: arm 计数 {counts.to_dict()}")


def l2_recompute(prefix: str) -> None:
    print("\n== L2 独立复算 ==")
    raw = pd.read_csv(OUTPUT_DIR / f"{prefix}_main_raw.csv")
    summ = pd.read_csv(OUTPUT_DIR / f"{prefix}_summary.csv")
    ok = True
    for _, row in summ[summ.arm == "S3_units"].iterrows():
        sub = raw[(raw.dataset == row.dataset) & (raw.arm == "S3_units")
                  & (raw.k_bearings == row.k_bearings)]
        sd = sub.test_fp.std(ddof=1)
        if abs(sd - row.sd_fp) > 1e-12:
            ok = False
            print(f"      不一致: {row.dataset} k={row.k_bearings} "
                  f"summary={row.sd_fp:.6f} recompute={sd:.6f}")
    check(ok, "S3 的 SD 与汇总表逐项一致")

    # 特征层抽检：直接重读源文件独立重算
    cache = pd.read_csv(OUTPUT_DIR / "r2f_features.csv")
    rng = np.random.default_rng(20260918)
    for dataset, loader in (("XJTU-SY", "xjtu"), ("PRONOSTIA", "pronostia")):
        sub = cache[cache.dataset == dataset]
        rows = sub.sample(3, random_state=0)
        for _, r in rows.iterrows():
            if loader == "xjtu":
                folder = DATASET_ROOT / "xjtu_sy" / "XJTU-SY_Bearing_Datasets" / r.condition / r.bearing
                path = folder / f"{int(r.record_index)}.csv"
                signal = pd.read_csv(path).iloc[:, 0].to_numpy(float)
            else:
                folder = (DATASET_ROOT / "pronostia" / "extracted"
                          / "phm-ieee-2012-data-challenge-dataset-master" / r.condition / r.bearing)
                path = folder / f"acc_{int(r.record_index):05d}.csv"
                signal = read_pronostia_csv(path).iloc[:, 4].to_numpy(float)
            rms = float(np.sqrt(np.mean(signal ** 2)))
            crest = float(np.max(np.abs(signal)) / rms) if rms > 0 else np.nan
            centered = signal - signal.mean()
            # 必须与冻结实现一致：分母用【样本标准差 ddof=1】
            std = float(np.std(signal, ddof=1))
            kurt = (float(np.mean(centered ** 4) / std ** 4) - 3.0) if std > 0 else np.nan
            match = (abs(rms - r["rms"]) < 1e-9 and abs(crest - r["crest_factor"]) < 1e-9
                     and abs(kurt - r["kurtosis"]) < 1e-6)
            check(match, f"{dataset} {r.bearing}#{int(r.record_index)} 特征独立复算",
                  f"rms {r['rms']:.6f} vs {rms:.6f}")
    _ = rng


def write_hashes(prefix: str) -> None:
    print("\n== 哈希归档 ==")
    lines = []
    for name in ("main_raw", "sensitivity_raw", "summary"):
        path = OUTPUT_DIR / f"{prefix}_{name}.csv"
        if path.exists():
            digest = sha256(path)
            lines.append(f"{digest}  {path.name}")
            print(f"  {digest[:16]}…  {path.name}")
    out = OUTPUT_DIR / f"{prefix}-hashes.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  已写入 {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="exp_v2_04")
    args = ap.parse_args()
    prefix = args.experiment
    l1_invariants(prefix)
    l2_recompute(prefix)
    write_hashes(prefix)
    print("\n== 结论 ==")
    print("全部通过" if not FAIL else f"存在 {len(FAIL)} 项失败：{FAIL}")
    sys.exit(0 if not FAIL else 1)


if __name__ == "__main__":
    main()