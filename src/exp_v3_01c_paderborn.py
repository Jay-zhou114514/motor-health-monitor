"""EXP-V3-01 修订 3：Paderborn 多工况实例上的部分池化校准（P1 严格 / P2 泄漏）。

预注册：experiments/EXP-V3-01-preregistration-amendment-3.md（先于运行提交）

实例 = 轴承 × 工况（6 × 4 = 24）；每实例取前 20 条记录，每条记录 = 1 个特征向量
（整条 4 s 记录作为单窗口，与 r2f 数据集的"每条记录 1 维特征向量"约定一致）。

实现说明（写进日志，属于实现细节而非判据改动）：
lambda 的训练内选择采用**子采样内层 LOO**（每个外层折随机取 6 个训练实例做内层留一，
种子由稳定哈希决定），以把拟合次数从 O(n^3) 降到可行量级；子采样只用于选 lambda，
阈值本身仍用全部训练实例计算。
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
from exp_v3_01_unit_calibration import DELTA_GRID, cond_number, make_scorer
from paderborn_data import CONDITIONS, HEALTHY_BEARINGS, load_healthy_dataset

ALPHA = 1.0 - QUANTILE
LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
RECORDS_PER_INSTANCE = 20
INNER_SAMPLE = 6
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


def threshold(kind: str, units: list[np.ndarray], lam: float,
              cache: dict | None = None) -> float:
    key = (kind, len(units))
    if cache is not None and key in cache:
        t_pool, t_bar = cache[key]
    else:
        t_pool = pooled_threshold(kind, units)
        t_bar = float(np.mean(unit_quantiles(kind, units)))
        if cache is not None:
            cache[key] = (t_pool, t_bar)
    return float(t_pool + lam * (t_bar - t_pool))


def choose_lambda(kind: str, units: list[np.ndarray], seed: int) -> float:
    """训练内子采样 LOO：选使 |实际率 − 名义率| 平均最小的 lambda。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(units), size=min(INNER_SAMPLE, len(units)), replace=False)
    best, best_dev = 0.0, np.inf
    for lam in LAMBDA_GRID:
        devs = []
        for j in idx:
            inner = [u for k, u in enumerate(units) if k != j]
            thr = threshold(kind, inner, lam)
            scorer = make_scorer(kind, np.vstack(inner))
            devs.append(abs(float((scorer(units[j]) > thr).mean()) - ALPHA))
        dev = float(np.mean(devs)) if devs else np.inf
        if dev < best_dev - 1e-12:
            best, best_dev = lam, dev
    return best


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
        arrays, _ = load_healthy_dataset(condition=cond, window_sec=None, step_sec=None,
                                         aggregate=False)
        per_bearing: dict[str, list[tuple[int, np.ndarray]]] = {}
        for record_name, arr in arrays.items():
            stem = record_name.rsplit(".", 1)[0]
            parts = stem.split("_")
            bearing, number = parts[-2], int(parts[-1])
            per_bearing.setdefault(bearing, []).append((number, np.asarray(arr, float)))
        for bearing, recs in per_bearing.items():
            recs.sort(key=lambda t: t[0])
            block = np.vstack([r for _, r in recs[:RECORDS_PER_INSTANCE]])
            if len(block) >= RECORDS_PER_INSTANCE:
                instances[(bearing, cond)] = block
        print(f"  {cond}: {len(per_bearing)} 颗轴承，实例累计 {len(instances)}", flush=True)
    return instances


def run_protocol(instances: dict[tuple[str, str], np.ndarray], strict: bool,
                 label: str) -> list[dict]:
    rows: list[dict] = []
    for test_bearing in HEALTHY_BEARINGS:
        if strict:
            train_keys = [k for k in instances if k[0] != test_bearing]
            test_keys = [k for k in instances if k[0] == test_bearing]
        else:
            test_keys = [(test_bearing, c) for c in CONDITIONS if (test_bearing, c) in instances]
            train_keys = [k for k in instances if k not in test_keys]
        train_units = [instances[k] for k in sorted(train_keys)]
        assert strict is False or all(k[0] != test_bearing for k in train_keys), "P1 训练集含测试轴承"
        for test_key in sorted(test_keys):
            test_block = instances[test_key]
            for kind in ("rms", "mahalanobis", "iforest"):
                b1 = pooled_threshold(kind, train_units)
                scorer = make_scorer(kind, np.vstack(train_units))
                rows.append(dict(protocol=label, test_instance=f"{test_key[0]}|{test_key[1]}",
                                 detector=kind, method="B1_pooled_quantile", param=np.nan,
                                 threshold=b1, rate=float((scorer(test_block) > b1).mean()),
                                 n_train_units=len(train_units)))
                lam = choose_lambda(kind, train_units,
                                    stable_seed("V301c", label, test_key[0], test_key[1], kind))
                thr_pp = threshold(kind, train_units, lam)
                rows.append(dict(protocol=label, test_instance=f"{test_key[0]}|{test_key[1]}",
                                 detector=kind, method="PP_partial_pooling", param=lam,
                                 threshold=thr_pp,
                                 rate=float((scorer(test_block) > thr_pp).mean()),
                                 n_train_units=len(train_units)))
                if kind == "mahalanobis":
                    pooled = np.vstack(train_units)
                    lw = LedoitWolf().fit(pooled)
                    mu = pooled.mean(axis=0)
                    inv = np.linalg.pinv(lw.covariance_ + 1e-12 * np.eye(pooled.shape[1]))
                    sc = lambda x: np.sqrt(np.einsum("ij,jk,ik->i", x - mu, inv, x - mu))
                    thr_lw = float(np.quantile(sc(pooled), QUANTILE))
                    rows.append(dict(protocol=label, test_instance=f"{test_key[0]}|{test_key[1]}",
                                     detector=kind, method="B2_ledoit_wolf", param=np.nan,
                                     threshold=thr_lw,
                                     rate=float((sc(test_block) > thr_lw).mean()),
                                     n_train_units=len(train_units)))
                    d = delta_wide(train_units)
                    sc3 = make_scorer(kind, pooled, delta=d)
                    thr_c = float(np.quantile(sc3(pooled), QUANTILE))
                    rows.append(dict(protocol=label, test_instance=f"{test_key[0]}|{test_key[1]}",
                                     detector=kind, method="B3_condnum_wide", param=d,
                                     threshold=thr_c,
                                     rate=float((sc3(test_block) > thr_c).mean()),
                                     n_train_units=len(train_units)))
        print(f"    [{label}] 留出轴承 {test_bearing} done", flush=True)
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUTPUT_DIR / "exp_v3_01c_run.log")
    print("构建 Paderborn 实例（轴承 × 工况，前 20 条记录，单窗口模式）…", flush=True)
    instances = build_instances()
    print(f"实例总数 = {len(instances)}", flush=True)
    if len(instances) < 20:
        raise SystemExit("实例数不足，检查数据根目录是否可读")

    rows = run_protocol(instances, strict=True, label="P1_strict") + \
        run_protocol(instances, strict=False, label="P2_leaky")
    raw = pd.DataFrame(rows)
    raw["saturated"] = raw.rate.isin([0.0, 1.0])
    raw.to_csv(OUTPUT_DIR / "exp_v3_01c_runs.csv", index=False)

    summary = (raw.groupby(["protocol", "detector", "method"])
               .agg(instances=("rate", "size"),
                    sd_pp=("rate", lambda s: float(np.std(s, ddof=1) * 100)),
                    mean_dev_pp=("rate", lambda s: float(np.mean(np.abs(s - ALPHA)) * 100)),
                    median_param=("param", "median"),
                    saturated=("saturated", "sum")).reset_index())
    summary.to_csv(OUTPUT_DIR / "exp_v3_01c_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== 汇总（SD 单位 pp） ===")
    print(summary.round(3).to_string(index=False))
    print("\n=== PP 相对 B1 的 SD 降幅（%）===")
    for (prot, det), sub in summary.groupby(["protocol", "detector"]):
        d = dict(zip(sub.method, sub.sd_pp))
        base = d.get("B1_pooled_quantile")
        pp = d.get("PP_partial_pooling")
        if base and pp is not None and base > 0:
            print(f"  {prot} {det}: {pp:.2f} vs {base:.2f} -> 降 {(1 - pp / base) * 100:.1f}%")
        else:
            print(f"  {prot} {det}: 基线 SD=0 或缺失，降幅不可计算（pp={pp}, b1={base}）")


if __name__ == "__main__":
    main()
