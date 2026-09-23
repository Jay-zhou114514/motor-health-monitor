"""EXP-V3-03 修订 1：M5 自校准的目标数据量扫描（c = 10/31/155/310/465 窗）。

预注册：experiments/EXP-V3-03-preregistration-amendment-1.md（先于运行提交）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import FIGURES_DIR, OUTPUT_DIR
from exp_v1_08_baselines import QUANTILE
from exp_v3_01_unit_calibration import make_scorer
from exp_v3_01d_paderborn_fixed import ALPHA, build_instances

C_GRID = (10, 31, 155, 310, 465)
DETECTORS = ("rms", "mahalanobis", "iforest")


class Tee:
    def __init__(self, path: Path):
        self.handle = open(path, "w", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        self.handle.write(text)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.handle.flush()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(centre - half), float(centre + half)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUTPUT_DIR / "exp_v3_03b_run.log")
    print("构建实例（620 窗/实例）…", flush=True)
    instances = build_instances()
    keys = sorted(instances)
    print(f"实例数 = {len(keys)}；c 网格 = {C_GRID}", flush=True)

    rows = []
    for test_key in keys:
        block = instances[test_key]
        n_total = len(block)
        for c in C_GRID:
            if c >= n_total - 10:
                continue
            calib, heldout = block[:c], block[c:]
            assert len(calib) + len(heldout) == n_total, "窗口数不守恒"
            for kind in DETECTORS:
                sc = make_scorer(kind, calib)
                thr = float(np.quantile(sc(calib), QUANTILE))
                rate = float((sc(heldout) > thr).mean())
                rows.append(dict(test_instance=f"{test_key[0]}|{test_key[1]}", detector=kind,
                                 c_windows=c, c_records=round(c / 31, 2), threshold=thr,
                                 rate=rate, n_eval=len(heldout)))
        print(f"  {test_key[0]}|{test_key[1]} done", flush=True)
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "exp_v3_03b_partial.csv", index=False)

    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT_DIR / "exp_v3_03b_runs.csv", index=False)

    out = []
    for (det, c), g in raw.groupby(["detector", "c_windows"]):
        r = g.rate.to_numpy(float)
        k = int(((r >= 0.5 * ALPHA) & (r <= 1.5 * ALPHA)).sum())
        lo, hi = wilson(k, len(r))
        out.append(dict(detector=det, c_windows=c, c_records=round(c / 31, 2), n=len(r),
                        B_tight=k / len(r), wilson_lo=lo, wilson_hi=hi,
                        B_wide=float(((r >= 0.25 * ALPHA) & (r <= 2 * ALPHA)).mean()),
                        mean_dev_pp=float(abs(r - ALPHA).mean() * 100),
                        over_conservative=float((r == 0).mean()),
                        sd_pp=float(r.std(ddof=1) * 100)))
    table = pd.DataFrame(out).sort_values(["detector", "c_windows"])
    table.to_csv(OUTPUT_DIR / "exp_v3_03b_selfcal_sweep.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== 自校准扫描：目标数据量 → 双侧带宽命中率 ===")
    print(table.round(3).to_string(index=False))

    print("\n=== 判据 ===")
    piv = table.pivot_table(index="c_windows", columns="detector", values="B_tight")
    ok = {c: int((row >= 0.5).sum()) for c, row in piv.iterrows()}
    c_star = next((c for c, n in sorted(ok.items()) if n >= 2), None)
    print(f"H20：B_tight ≥ 0.50 且 ≥2/3 检测器的最小 c* = {c_star}（各 c 的达标检测器数：{ok}）")
    for det, sub in table.groupby("detector"):
        rho = np.corrcoef(sub.c_windows, sub.mean_dev_pp)[0, 1]
        print(f"  {det:<14} mean_dev 与 c 的相关系数 = {rho:+.3f}（H21 要求 < 0）")
    if c_star is not None:
        print(f"H22：c* = {c_star} 窗（≈ {c_star/31:.1f} 条记录），是否 ≤ 310 窗（半数）：{c_star <= 310}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for det, sub in table.groupby("detector"):
            ax.plot(sub.c_windows, sub.B_tight, marker="o", label=det)
        ax.axhline(0.5, ls=":", color="grey")
        ax.set_xlabel("target-unit windows used for calibration (c)")
        ax.set_ylabel("share within [0.5a, 1.5a]")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "exp_v3_03b_selfcal_curve.png", dpi=200)
        print("图已写出: docs/figures/exp_v3_03b_selfcal_curve.png")
    except Exception as exc:  # noqa: BLE001
        print("绘图失败:", exc)


if __name__ == "__main__":
    main()
