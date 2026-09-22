"""EXP-V3-01 修订 4：修正准则（SD + 覆盖约束）、提高分辨率（W-B 窗口）、修复 P2。

预注册：experiments/EXP-V3-01-preregistration-amendment-4.md（先于运行提交）
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import OUTPUT_DIR
from exp_v1_08_baselines import QUANTILE
from exp_v3_01_unit_calibration import cond_number, make_scorer
from paderborn_data import CONDITIONS, HEALTHY_BEARINGS, load_healthy_dataset

ALPHA = 1.0 - QUANTILE
LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
RECORDS_PER_INSTANCE = 20
INNER_SAMPLE = 6
WINDOW_SEC, STEP_SEC = 0.25, 0.125
DELTA_GRID_WIDE = np.round(np.arange(0.0, 0.91, 0.05), 2)


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


def pooled_threshold(kind: str, units: list[np.ndarray]) -> float:
    pooled = np.vstack(units)
    return float(np.quantile(make_scorer(kind, pooled)(pooled), QUANTILE))


def unit_quantiles(kind: str, units: list[np.ndarray]) -> list[float]:
    out = []
    for i, unit in enumerate(units):
        others = np.vstack([u for j, u in enumerate(units) if j != i])
        out.append(float(np.quantile(make_scorer(kind, others)(unit), QUANTILE)))
    return out


def calibrate(kind: str, train_units: list[np.ndarray], seed: int) -> tuple[float, dict]:
    """返回 (lambda, 诊断)。准则：覆盖约束下最小化跨单元 SD。"""
    n = len(train_units)
    pooled = np.vstack(train_units)
    t_pool = float(np.quantile(make_scorer(kind, pooled)(pooled), QUANTILE))
    t_bar = float(np.mean(unit_quantiles(kind, train_units)))

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(INNER_SAMPLE, n), replace=False)
    inner_cache: dict[int, tuple[float, float]] = {}
    for j in idx:
        inner = [u for k, u in enumerate(train_units) if k != j]
        inner_cache[j] = (pooled_threshold(kind, inner),
                          float(np.mean(unit_quantiles(kind, inner))))

    eligible, stats = [], {}
    for lam in LAMBDA_GRID:
        rates = []
        for j in idx:
            inner = [u for k, u in enumerate(train_units) if k != j]
            thr = inner_cache[j][0] + lam * (inner_cache[j][1] - inner_cache[j][0])
            scorer = make_scorer(kind, np.vstack(inner))
            rates.append(float((scorer(train_units[j]) > thr).mean()))
        rates = np.asarray(rates)
        sd, mean = float(np.std(rates, ddof=1)), float(rates.mean())
        stats[lam] = (sd, mean)
        if 0.5 * ALPHA <= mean <= 2.0 * ALPHA:
            eligible.append((sd, lam))
    if eligible:
        sd_best, lam_best = min(eligible, key=lambda t: (t[0], t[1]))
    else:
        sd_best, lam_best = stats[0.0][0], 0.0
    return lam_best, {"t_pooled": t_pool, "t_bar": t_bar, "inner_sd": sd_best,
                      "n_eligible": len(eligible)}


def delta_wide(units: list[np.ndarray]) -> float:
    train = np.vstack(units)
    base = np.cov(train, rowvar=False)
    lw = LedoitWolf().fit(train).covariance_
    for delta in DELTA_GRID_WIDE:
        if cond_number((1.0 - delta) * base + delta * lw) <= 100.0:
            return float(delta)
    return float(DELTA_GRID_WIDE[-1])


def build_instances() -> dict[tuple[str, str], np.ndarray]:
    instances: dict[tuple[str, str], np.ndarray] = {}
    for cond in CONDITIONS:
        arrays, _ = load_healthy_dataset(condition=cond, window_sec=WINDOW_SEC,
                                         step_sec=STEP_SEC, aggregate=False)
        per_bearing: dict[str, list[tuple[int, np.ndarray]]] = {}
        for record_name, arr in arrays.items():
            stem = record_name.rsplit(".", 1)[0]
            parts = stem.split("_")
            per_bearing.setdefault(parts[-2], []).append((int(parts[-1]), np.asarray(arr, float)))
        for bearing, recs in per_bearing.items():
            recs.sort(key=lambda t: t[0])
            block = np.vstack([r for _, r in recs[:RECORDS_PER_INSTANCE]])
            if len(recs) >= RECORDS_PER_INSTANCE:
                instances[(bearing, cond)] = block
        print(f"  {cond}: 实例累计 {len(instances)}，窗口/实例 ≈ {block.shape[0]}", flush=True)
    return instances


def run_protocol(instances: dict[tuple[str, str], np.ndarray], strict: bool, label: str):
    rows = []
    for test_bearing in HEALTHY_BEARINGS:
        if strict:
            train_keys = [k for k in instances if k[0] != test_bearing]
            test_keys = [k for k in instances if k[0] == test_bearing]
        else:
            test_keys = [(test_bearing, c) for c in CONDITIONS if (test_bearing, c) in instances]
            train_keys = [k for k in instances if k not in test_keys]      # 修复后的 P2
        train_units = [instances[k] for k in sorted(train_keys)]
        assert not strict or all(k[0] != test_bearing for k in train_keys), "P1 训练集含测试轴承"
        for test_key in sorted(test_keys):
            test_block = instances[test_key]
            for kind in ("rms", "mahalanobis", "iforest"):
                pooled = np.vstack(train_units)
                scorer = make_scorer(kind, pooled)
                b1 = float(np.quantile(scorer(pooled), QUANTILE))
                rows.append(dict(protocol=label, test_instance=f"{test_key[0]}|{test_key[1]}",
                                 detector=kind, method="B1_pooled_quantile", param=np.nan,
                                 rate=float((scorer(test_block) > b1).mean()),
                                 n_train_units=len(train_units), n_test_windows=len(test_block)))
                lam, diag = calibrate(kind, train_units,
                                      stable_seed("V301d", label, test_key[0], test_key[1], kind))
                thr = diag["t_pooled"] + lam * (diag["t_bar"] - diag["t_pooled"])
                rows.append(dict(protocol=label, test_instance=f"{test_key[0]}|{test_key[1]}",
                                 detector=kind, method="PP_partial_pooling", param=lam,
                                 rate=float((scorer(test_block) > thr).mean()),
                                 n_train_units=len(train_units), n_test_windows=len(test_block)))
        print(f"    [{label}] {test_bearing} done", flush=True)
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUTPUT_DIR / "exp_v3_01d_run.log")
    print(f"W-B 窗口口径：{WINDOW_SEC}s / {STEP_SEC}s", flush=True)
    instances = build_instances()
    print(f"实例总数 = {len(instances)}", flush=True)
    rows = run_protocol(instances, True, "P1_strict")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "exp_v3_01d_partial_P1_strict.csv", index=False)
    print("[检查点] P1 已写出", flush=True)
    rows += run_protocol(instances, False, "P2_leaky")
    raw = pd.DataFrame(rows)
    raw["saturated"] = raw.rate.isin([0.0, 1.0])
    raw.to_csv(OUTPUT_DIR / "exp_v3_01d_runs.csv", index=False)
    summary = (raw.groupby(["protocol", "detector", "method"])
               .agg(n=("rate", "size"),
                    sd_pp=("rate", lambda s: float(np.std(s, ddof=1) * 100)),
                    mean_rate=("rate", "mean"),
                    median_lambda=("param", "median"),
                    saturated=("saturated", "sum")).reset_index())
    summary.to_csv(OUTPUT_DIR / "exp_v3_01d_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== 汇总（率已按窗口口径，分辨率 ~0.16%）===")
    print(summary.round(4).to_string(index=False))
    print("\n=== PP 相对 B1 的 SD 降幅 ===")
    for (prot, det), sub in summary.groupby(["protocol", "detector"]):
        d = dict(zip(sub.method, sub.sd_pp))
        base, pp = d.get("B1_pooled_quantile"), d.get("PP_partial_pooling")
        if base and pp is not None:
            print(f"  {prot} {det}: {pp:.2f} vs {base:.2f} -> 降 {(1 - pp / base) * 100:.1f}%")


if __name__ == "__main__":
    main()
