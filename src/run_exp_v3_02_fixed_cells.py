"""修正 EXP-V3-02 的单元集合采集（不改动已提交脚本的其余部分）。"""

from __future__ import annotations

import pandas as pd

import exp_v3_02_trust_diagnostic as exp
from config import OUTPUT_DIR


def collect_cells() -> pd.DataFrame:
    rows: list[dict] = []

    v203 = pd.read_csv(OUTPUT_DIR / "exp_v2_03_summary.csv")
    sub = v203[(v203.regime == "R2_bearing_holdout") & (v203.arm == "S2_new_records")
               & (v203.n_train == 96)]
    if len(sub):
        rows.append(dict(source="EXP-V2-03", cell="Paderborn_holdout_n96", detector="iforest",
                         units=int(sub.fold.nunique()), sd_pp=float(sub.sd_fp.iloc[0] * 100)))

    v205 = pd.read_csv(OUTPUT_DIR / "exp_v2_05_summary.csv")
    for (ds, cond), g in v205.groupby(["dataset", "condition"]):
        for _, r in g.iterrows():
            if float(r.sd_fp) > 0:
                rows.append(dict(source="EXP-V2-05", cell=f"{ds}/{cond}", detector="iforest",
                                 units=int(r.k_bearings), sd_pp=float(r.sd_fp) * 100))

    v206 = pd.read_csv(OUTPUT_DIR / "exp_v2_06_summary.csv")
    for (det, ds, cond), g in v206.groupby(["detector", "dataset", "condition"]):
        for _, r in g.iterrows():
            if float(r.sd_fp) > 0:
                rows.append(dict(source="EXP-V2-06", cell=f"{ds}/{cond}", detector=det,
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


exp.collect_cells = collect_cells
exp.main()
