"""EXP-V3-01：面向未见物理单元的阈值校准（ULTC）。

预注册：experiments/EXP-V3-01-preregistration.md（2026-09-21，commit 0a73bda）
修订 1：experiments/EXP-V3-01-preregistration-amendment-1.md（第一版仅 XJTU-SY + PRONOSTIA）

方法：阈值不再取"训练分数池化的 0.99 分位"，而是先在训练单元之间校准
（leave-one-unit-out 得到每颗训练单元的分位 t_i），再取 mean(t_i) 或 max(t_i) 作为阈值。
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest

from config import OUTPUT_DIR
from exp_v1_08_baselines import QUANTILE
from runtofailure_data import FEATURES, healthy_size, included

R_REPEAT = 50
BUDGET = 20
ALPHA = 1.0 - QUANTILE          # 名义误报目标 = 1%
DELTA_GRID = np.linspace(0.0, 0.5, 26)
COND_LIMIT = 100.0


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


# ----------------------------------------------------------------- 检测器
def make_scorer(kind: str, train: np.ndarray, delta: float | None = None):
    """返回 score(X)->ndarray；分数越大越可能报警。"""
    if kind == "rms":
        mu, sd = train[:, 0].mean(), train[:, 0].std(ddof=1) or 1.0
        return lambda x: (x[:, 0] - mu) / sd

    if kind == "mahalanobis":
        mu = train.mean(axis=0)
        cov = np.cov(train, rowvar=False)
        if delta is not None:
            lw = LedoitWolf().fit(train).covariance_
            cov = (1.0 - delta) * cov + delta * lw
        inv = np.linalg.pinv(cov + 1e-12 * np.eye(cov.shape[0]))
        return lambda x: np.sqrt(np.einsum("ij,jk,ik->i", x - mu, inv, x - mu))

    if kind == "iforest":
        mu, sd = train.mean(axis=0), train.std(axis=0, ddof=1)
        sd = np.where(sd == 0, 1.0, sd)
        model = IsolationForest(n_estimators=200, max_samples=0.5, random_state=0,
                                n_jobs=1).fit((train - mu) / sd)
        return lambda x: -model.score_samples((x - mu) / sd)

    raise ValueError(kind)


def cond_number(cov: np.ndarray) -> float:
    return float(np.linalg.cond(cov + 1e-12 * np.eye(cov.shape[0])))


def delta_by_condition(train: np.ndarray) -> float:
    """训练内准则：最小 delta 使收缩后协方差的条件数 <= COND_LIMIT。"""
    base = np.cov(train, rowvar=False)
    lw = LedoitWolf().fit(train).covariance_
    for delta in DELTA_GRID:
        if cond_number((1.0 - delta) * base + delta * lw) <= COND_LIMIT:
            return float(delta)
    return 0.5


def calibration_thresholds(kind: str, train_units: list[np.ndarray]) -> dict:
    """leave-one-unit-out 得到每颗训练单元的分位 t_i，再聚合。"""
    ts = []
    for i, unit in enumerate(train_units):
        others = np.vstack([u for j, u in enumerate(train_units) if j != i])
        scores = make_scorer(kind, others)(unit)
        ts.append(float(np.quantile(scores, QUANTILE)))
    return {"mean": float(np.mean(ts)), "max": float(np.max(ts)), "ts": ts}


# ----------------------------------------------------------------- 数据池
def operating_condition(dataset: str, condition: str, bearing: str) -> str:
    if dataset == "XJTU-SY":
        return condition
    return "C" + bearing.split("_")[0].replace("Bearing", "")


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
        key = (dataset, operating_condition(dataset, condition, bearing))
        pools.setdefault(key, {})[bearing] = ordered[FEATURES].to_numpy(float)
    return pools


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUTPUT_DIR / "exp_v3_01_run.log")

    cache = OUTPUT_DIR / "r2f_features.csv"
    if not cache.exists():
        raise SystemExit(f"缺少特征缓存：{cache}（需先运行 runtofailure_data.load_feature_cache）")
    frame = pd.read_csv(cache)
    pools = build_pools(frame)
    print(f"单元集合：{len(pools)} 个；名义误报目标 alpha={ALPHA:.3f}", flush=True)

    rows: list[dict] = []
    for (dataset, cond), bearings in sorted(pools.items()):
        names = sorted(bearings)
        if len(names) < 3:                 # 需要 >=2 颗训练单元才能做 leave-one-unit-out
            print(f"[{dataset}/{cond}] 仅 {len(names)} 颗单元，ULTC 不可检验，跳过", flush=True)
            continue
        print(f"[{dataset}/{cond}] 单元 {len(names)} 颗", flush=True)
        for held in names:
            train_names = [b for b in names if b != held]
            rng = np.random.default_rng(stable_seed("V301", dataset, cond, held))
            train_units = []
            for name in train_names:
                arr = bearings[name]
                take = min(BUDGET // max(1, len(train_names)), len(arr))
                train_units.append(arr[rng.choice(len(arr), size=take, replace=False)])
            test_block = bearings[held][rng.choice(len(bearings[held]),
                                                   size=min(BUDGET, len(bearings[held])),
                                                   replace=False)]
            n_test = len(test_block)

            for kind in ("rms", "mahalanobis", "iforest"):
                pooled = np.vstack(train_units)
                # B1 池化分位
                s = make_scorer(kind, pooled)(pooled)
                thr = float(np.quantile(s, QUANTILE))
                rate = float((make_scorer(kind, pooled)(test_block) > thr).mean())
                rows.append(dict(dataset=dataset, condition=cond, held_out=held, detector=kind,
                                 method="B1_pooled_quantile", threshold=thr, rate=rate,
                                 n_train=len(pooled), n_test=n_test))
                # ULTC
                cal = calibration_thresholds(kind, train_units)
                for label in ("mean", "max"):
                    thr_c = cal[label]
                    rate_c = float((make_scorer(kind, pooled)(test_block) > thr_c).mean())
                    rows.append(dict(dataset=dataset, condition=cond, held_out=held, detector=kind,
                                     method=f"ULTC_{label}", threshold=thr_c, rate=rate_c,
                                     n_train=len(pooled), n_test=n_test))
                # 仅马氏：Ledoit-Wolf 与条件数准则
                if kind == "mahalanobis":
                    lw = LedoitWolf().fit(pooled)
                    mu = pooled.mean(axis=0)
                    inv = np.linalg.pinv(lw.covariance_ + 1e-12 * np.eye(pooled.shape[1]))
                    sc = lambda x: np.sqrt(np.einsum("ij,jk,ik->i", x - mu, inv, x - mu))
                    thr_lw = float(np.quantile(sc(pooled), QUANTILE))
                    rows.append(dict(dataset=dataset, condition=cond, held_out=held,
                                     detector=kind, method="B2_ledoit_wolf",
                                     threshold=thr_lw, rate=float((sc(test_block) > thr_lw).mean()),
                                     n_train=len(pooled), n_test=n_test))
                    d = delta_by_condition(pooled)
                    sc3 = make_scorer(kind, pooled, delta=d)
                    thr_c3 = float(np.quantile(sc3(pooled), QUANTILE))
                    rows.append(dict(dataset=dataset, condition=cond, held_out=held,
                                     detector=kind, method=f"B3_condnum(delta={d:.2f})",
                                     threshold=thr_c3, rate=float((sc3(test_block) > thr_c3).mean()),
                                     n_train=len(pooled), n_test=n_test))
        print(f"    [{dataset}/{cond}] done", flush=True)

    raw = pd.DataFrame(rows)
    raw["saturated"] = raw.rate.isin([0.0, 1.0])
    raw["resolution"] = 1.0 / raw.n_test
    raw.to_csv(OUTPUT_DIR / "exp_v3_01_runs.csv", index=False)

    def agg(sub: pd.DataFrame) -> pd.Series:
        rates = sub.rate.to_numpy(float)
        return pd.Series({
            "units": len(rates),
            "sd_pp": float(np.std(rates, ddof=1) * 100) if len(rates) > 1 else np.nan,
            "mean_dev_pp": float(np.mean(np.abs(rates - ALPHA)) * 100),
            "hit_share": float(np.mean((rates >= 0.5 * ALPHA) & (rates <= 2.0 * ALPHA))),
            "saturated_units": int(sub.saturated.sum()),
        })

    summary = (raw.groupby(["dataset", "condition", "detector", "method"])
               .apply(agg, include_groups=False).reset_index())
    summary.to_csv(OUTPUT_DIR / "exp_v3_01_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== M1 跨单元 SD（pp，排除饱和单元的读数另计）===")
    piv = summary.pivot_table(index=["dataset", "condition", "detector"], columns="method",
                              values="sd_pp")
    print(piv.round(2).to_string())
    print("\n=== 每个检测器：ULTC 相对 B1 的 SD 降幅（%）===")
    for (dataset, cond, det), sub in summary.groupby(["dataset", "condition", "detector"]):
        row = dict(zip(sub.method, sub.sd_pp))
        base = row.get("B1_pooled_quantile")
        for name in ("ULTC_mean", "ULTC_max"):
            if base and name in row and not np.isnan(row[name]):
                drop = (1 - row[name] / base) * 100
                print(f"  {dataset}/{cond} {det}: {name} SD {row[name]:.2f} vs B1 {base:.2f} -> 降 {drop:.1f}%")


if __name__ == "__main__":
    main()
