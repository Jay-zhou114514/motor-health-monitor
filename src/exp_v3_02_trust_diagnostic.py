"""EXP-V3-02：报告误报率的信任判定器（元分析）。

预注册：experiments/EXP-V3-02-preregistration.md（先于运行提交）

模型：SD(n) = a * n^(-gamma) + c；判据 H7（gamma 的 95% CI 不含 0）、
H8（LOO-CV 平均绝对误差 <= 8 pp）、H9（优于常数模型）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from config import FIGURES_DIR, OUTPUT_DIR

SEED = 20260922
BOOT = 10_000
TAUS = (2.0, 5.0, 10.0)


class Tee:
    def __init__(self, path: Path):
        self.handle = open(path, "w", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        self.handle.write(text)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.handle.flush()


def model(n, a, gamma, c):
    return a * np.power(n, -gamma) + c


def collect_cells() -> pd.DataFrame:
    rows: list[dict] = []

    v203 = pd.read_csv(OUTPUT_DIR / "exp_v2_03_summary.csv")
    sub = v203[(v203.regime == "R2_bearing_holdout") & (v203.arm == "S2_new_records")
               & (v203.n_train == 96)]
    if len(sub):
        rows.append(dict(source="EXP-V2-03", cell="Paderborn_holdout_n96", detector="iforest",
                         units=int(sub.fold.nunique()), sd_pp=float(sub.sd_fp.iloc[0] * 100)))

    for name, detector in (("exp_v2_05_summary.csv", "iforest"),
                           ("exp_v2_06_summary.csv", None)):
        df = pd.read_csv(OUTPUT_DIR / name)
        for (det, ds, cond), g in df.groupby(["detector", "dataset", "condition"]) if detector is None \
                else [(detector, k[0], k[1]) for k in df.groupby(["dataset", "condition"]).groups]:
            sub = df[(df.dataset == ds) & (df.condition == cond)]
            if detector is None:
                sub = sub[sub.detector == det]
            for _, r in sub.iterrows():
                if float(r.sd_fp) > 0:
                    rows.append(dict(source=name.replace("exp_", "EXP-").replace("_summary.csv", ""),
                                     cell=f"{ds}/{cond}", detector=det,
                                     units=int(r.k_bearings), sd_pp=float(r.sd_fp) * 100))

    v301b = pd.read_csv(OUTPUT_DIR / "exp_v3_01b_summary.csv")
    for _, r in v301b.iterrows():
        if r.sd_pp > 0:
            rows.append(dict(source="EXP-V3-01b", cell=f"{r.dataset}/{r.condition}",
                             detector=r.detector, units=int(r.units), sd_pp=float(r.sd_pp)))

    v301c = pd.read_csv(OUTPUT_DIR / "exp_v3_01c_summary.csv")
    for _, r in v301c.iterrows():
        if r.protocol == "P1_strict" and r.method == "B1_pooled_quantile" and r.sd_pp > 0:
            rows.append(dict(source="EXP-V3-01c", cell="Paderborn_instances",
                             detector=r.detector, units=6, sd_pp=float(r.sd_pp)))

    return pd.DataFrame(rows)


def fit(cells: pd.DataFrame):
    n, y = cells.units.to_numpy(float), cells.sd_pp.to_numpy(float)
    popt, _ = curve_fit(model, n, y, p0=[50.0, 0.5, 1.0],
                        bounds=([0, 0, 0], [np.inf, 5, np.inf]), maxfev=20000)
    return popt


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUTPUT_DIR / "exp_v3_02_run.log")

    cells = collect_cells()
    cells.to_csv(OUTPUT_DIR / "exp_v3_02_cells.csv", index=False)
    print(f"纳入单元集合: {len(cells)} 个；单元数范围 {cells.units.min()}–{cells.units.max()}")
    print(cells.groupby("source").size().to_string())

    a, gamma, c = fit(cells)
    print(f"\n主模型拟合: a={a:.2f}, gamma={gamma:.3f}, c={c:.2f}")

    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(BOOT):
        idx = rng.integers(0, len(cells), len(cells))
        try:
            boot.append(fit(cells.iloc[idx]))
        except Exception:  # noqa: BLE001
            continue
    boot = np.array(boot)
    lo, hi = np.percentile(boot[:, 1], [2.5, 97.5])
    print(f"gamma bootstrap 95% CI = [{lo:.3f}, {hi:.3f}]  (H7: 不含 0 -> {'支持' if lo > 0 else '不支持'})")

    # LOO-CV
    errs, const_errs, preds = [], [], []
    for i in range(len(cells)):
        train = cells.drop(index=i)
        try:
            p = fit(train)
            pred = float(model(cells.units.iloc[i], *p))
        except Exception:  # noqa: BLE001
            pred = float(train.sd_pp.mean())
        errs.append(abs(pred - cells.sd_pp.iloc[i]))
        const_errs.append(abs(train.sd_pp.mean() - cells.sd_pp.iloc[i]))
        preds.append(pred)
    mae, const_mae = float(np.mean(errs)), float(np.mean(const_errs))
    print(f"\nLOO-CV MAE = {mae:.2f} pp (H8: <=8pp -> {'支持' if mae <= 8 else '不支持'})")
    print(f"常数模型 MAE = {const_mae:.2f} pp (H9: 主模型更优 -> {'支持' if mae < const_mae else '不支持'})")

    loo = cells.copy()
    loo["pred_pp"] = preds
    loo["abs_err_pp"] = errs
    loo.to_csv(OUTPUT_DIR / "exp_v3_02_loo.csv", index=False)

    print("\n=== 设计表：达到目标可信度所需的最少独立单元数 ===")
    grid = np.arange(2, 41)
    design = {}
    for tau in TAUS:
        ok = grid[model(grid, a, gamma, c) <= tau]
        design[tau] = int(ok.min()) if len(ok) else None
        print(f"  tau = {tau:>4} pp -> n_min = {design[tau] if design[tau] else '数据范围内不可达'}")

    pd.DataFrame([dict(a=a, gamma=gamma, c=c, gamma_ci_low=lo, gamma_ci_high=hi,
                       loo_mae_pp=mae, const_mae_pp=const_mae,
                       n_min_tau2=design[2.0], n_min_tau5=design[5.0], n_min_tau10=design[10.0])]
                 ).to_csv(OUTPUT_DIR / "exp_v3_02_fit.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(cells.units, cells.sd_pp, s=18, alpha=0.7, label="archived unit sets")
        xs = np.linspace(cells.units.min(), max(40, cells.units.max()), 200)
        ax.plot(xs, model(xs, a, gamma, c), color="crimson", label="SD(n) fit")
        for tau in TAUS:
            ax.axhline(tau, ls=":", lw=0.8, color="grey")
        ax.set_xlabel("independent units n")
        ax.set_ylabel("across-unit SD of the reported rate (pp)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "exp_v3_02_sd_vs_units.png", dpi=200)
        print("\n图已写出: docs/figures/exp_v3_02_sd_vs_units.png")
    except Exception as exc:  # noqa: BLE001
        print("绘图失败:", exc)


if __name__ == "__main__":
    main()
