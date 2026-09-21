"""EXP-V3-01 修订 2：部分池化阈值校准。

预注册：experiments/EXP-V3-01-preregistration-amendment-2.md（先于运行提交）
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR
from exp_v1_08_baselines import QUANTILE
from runtofailure_data import FEATURES, healthy_size, included
from exp_v3_01_unit_calibration import make_scorer

BUDGET = 20
ALPHA = 1.0 - QUANTILE
LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


class Tee:
    def __init__(self, path: Path):
        self.handle = open(path, "w", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        self.handle.write(text)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.handle.flush()


def stable_seed(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode("utf-8"))


def operating_condition(dataset: str, condition: str, bearing: str) -> str:
    if dataset == "XJTU-SY":
        return condition
    return "C" + bearing.split("_")[0].replace("Bearing", "")


def pooled_threshold(kind: str, train_units: list[np.ndarray]) -> float:
    pooled = np.vstack(train_units)
    return float(np.quantile(make_scorer(kind, pooled)(pooled), QUANTILE))


def unit_quantiles(kind: str, train_units: list[np.ndarray]) -> list[float]:
    ts = []
    for i, unit in enumerate(train_units):
        others = np.vstack([u for j, u in enumerate(train_units) if j != i])
        ts.append(float(np.quantile(make_scorer(kind, others)(unit), QUANTILE)))
    return ts


def threshold(kind: str, train_units: list[np.ndarray], lam: float) -> float:
    t_pool = pooled_threshold(kind, train_units)
    ts = unit_quantiles(kind, train_units)
    return float(t_pool + lam * (np.mean(ts) - t_pool))


def choose_lambda(kind: str, train_units: list[np.ndarray]) -> float:
    """训练内 leave-one-unit-out：选使 |实际率 − 名义率| 平均最小的 lambda。"""
    best, best_dev = 0.0, np.inf
    for lam in LAMBDA_GRID:
        devs = []
        for i, unit in enumerate(train_units):
            inner = [u for j, u in enumerate(train_units) if j != i]
            if len(inner) < 2:
                continue
            thr = threshold(kind, inner, lam)
            scorer = make_scorer(kind, np.vstack(inner))
            devs.append(abs(float((scorer(unit) > thr).mean()) - ALPHA))
        if devs:
            dev = float(np.mean(devs))
            if dev < best_dev - 1e-12:
                best, best_dev = lam, dev
    return best


def build_pools(frame: pd.DataFrame):
    pools: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for (dataset, condition, bearing), g in frame.groupby(["dataset", "condition", "bearing"]):
        n_total = int(g.n_records_total.iloc[0])
        h = healthy_size(n_total)
        if not included(n_total, h):
            continue
        ordered = g.sort_values("record_index").head(h)
        if len(ordered) < h:
            continue
        pools.setdefault((dataset, operating_condition(dataset, condition, bearing)), {})[
            bearing] = ordered[FEATURES].to_numpy(float)
    return pools


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUTPUT_DIR / "exp_v3_01b_run.log")
    frame = pd.read_csv(OUTPUT_DIR / "r2f_features.csv")
    pools = build_pools(frame)

    rows: list[dict] = []
    for (dataset, cond), bearings in sorted(pools.items()):
        names = sorted(bearings)
        if len(names) < 3:
            print(f"[{dataset}/{cond}] 单元 {len(names)} 颗，跳过", flush=True)
            continue
        for held in names:
            train_names = [b for b in names if b != held]
            rng = np.random.default_rng(stable_seed("V301b", dataset, cond, held))
            train_units = []
            per = max(1, BUDGET // max(1, len(train_names)))
            for name in train_names:
                arr = bearings[name]
                train_units.append(arr[rng.choice(len(arr), size=min(per, len(arr)),
                                                 replace=False)])
            test_block = bearings[held][rng.choice(len(bearings[held]),
                                                   size=min(BUDGET, len(bearings[held])),
                                                   replace=False)]
            for kind in ("rms", "mahalanobis", "iforest"):
                lam = choose_lambda(kind, train_units)
                thr = threshold(kind, train_units, lam)
                scorer = make_scorer(kind, np.vstack(train_units))
                rate = float((scorer(test_block) > thr).mean())
                rows.append(dict(dataset=dataset, condition=cond, held_out=held, detector=kind,
                                 lambda_=lam, threshold=thr, rate=rate,
                                 n_train=sum(len(u) for u in train_units),
                                 n_test=len(test_block)))
        print(f"    [{dataset}/{cond}] done", flush=True)

    raw = pd.DataFrame(rows)
    raw["saturated"] = raw.rate.isin([0.0, 1.0])
    raw.to_csv(OUTPUT_DIR / "exp_v3_01b_runs.csv", index=False)

    summary = (raw.groupby(["dataset", "condition", "detector"])
               .agg(units=("rate", "size"),
                    sd_pp=("rate", lambda s: float(np.std(s, ddof=1) * 100)),
                    mean_dev_pp=("rate", lambda s: float(np.mean(np.abs(s - ALPHA)) * 100)),
                    median_lambda=("lambda_", "median")).reset_index())
    summary.to_csv(OUTPUT_DIR / "exp_v3_01b_summary.csv", index=False)

    print("\n=== 部分池化：跨单元 SD（pp）与 B1 对比 ===")
    base = pd.read_csv(OUTPUT_DIR / "exp_v3_01_summary.csv")
    base = base[base.method == "B1_pooled_quantile"][
        ["dataset", "condition", "detector", "sd_pp"]].rename(columns={"sd_pp": "b1_sd_pp"})
    merged = summary.merge(base, on=["dataset", "condition", "detector"], how="left")
    merged["drop_pct"] = (1 - merged.sd_pp / merged.b1_sd_pp) * 100
    pd.set_option("display.width", 200)
    print(merged.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
