"""EXP-V2-04：run-to-failure 数据上的健康阶段定义不确定性与独立单元数。

预注册：experiments/EXP-V2-04-preregistration.md + 修订 1（双数据集）+ 修订 2（口径更正）

用法：
  python exp_v2_04_xjtu_pronostia.py --part main
  python exp_v2_04_xjtu_pronostia.py --part sensitivity
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import OUTPUT_DIR
from exp_v1_08_baselines import QUANTILE
from runtofailure_data import FEATURES, healthy_size, included

R_MAIN = 50
R_SENS = 30
BUDGET = 20
MODAL_CFG = (200, 0.5)
N_GRID = [20, 60, 160]
K_GRID = [1, 4, 99]          # 99 = 该数据集的 k_max（占位）
RULES = [(0.05, 20, 60), (0.10, 20, 60), (0.20, 20, 60),
         (0.10, 10, 60), (0.10, 30, 60)]


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


def fp_rate(train: np.ndarray, test: np.ndarray) -> float:
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=1)
    std = np.where(std == 0, 1.0, std)
    z = (train - mean) / std
    model = IsolationForest(n_estimators=MODAL_CFG[0], max_samples=MODAL_CFG[1],
                            random_state=0, n_jobs=1).fit(z)
    scores = -model.score_samples(z)
    threshold = float(np.quantile(scores, QUANTILE))
    return float((-model.score_samples((test - mean) / std) > threshold).mean())


def build_pools(frame: pd.DataFrame, dataset: str, rule) -> dict[str, np.ndarray]:
    """按健康阶段规则构造每颗轴承的健康特征数组；返回 {bearing: (H,4)}。"""
    fraction, floor, cap = rule
    sub = frame[frame.dataset == dataset]
    pools: dict[str, np.ndarray] = {}
    for (condition, bearing), g in sub.groupby(["condition", "bearing"]):
        n_total = int(g.n_records_total.iloc[0])
        h = healthy_size(n_total, fraction=fraction, floor=floor, cap=cap)
        if not included(n_total, h):
            continue
        ordered = g.sort_values("record_index").head(h)
        if len(ordered) < h:
            continue
        pools[bearing] = ordered[FEATURES].to_numpy(float)
    return pools


def run_s2(pools, folds, rng_tag, r_repeat) -> list[dict]:
    rows = []
    for held in folds:
        train_names = [b for b in pools if b != held]
        pool = np.vstack([pools[b] for b in train_names])
        test = pools[held]
        rng = np.random.default_rng(stable_seed("S2", rng_tag, held))
        for n in N_GRID:
            n_use = min(n, len(pool))
            for _ in range(r_repeat):
                idx = rng.choice(len(pool), size=n_use, replace=False)
                rows.append({"arm": "S2_new_records", "held_out": held,
                             "n_train": n_use, "k_bearings": np.nan,
                             "test_fp": fp_rate(pool[idx], test)})
        print(f"    S2 {rng_tag} held={held} done", flush=True)
    return rows


def run_s3(pools, folds, rng_tag, r_repeat, k_grid) -> list[dict]:
    rows = []
    for held in folds:
        train_names = sorted(b for b in pools if b != held)
        k_max = min(max(k_grid), len(train_names))
        rng = np.random.default_rng(stable_seed("S3", rng_tag, held))
        for k in k_grid:
            k_use = k_max if k == 99 else min(k, len(train_names))
            per = max(1, BUDGET // k_use)
            for _ in range(r_repeat):
                picked = rng.choice(len(train_names), size=k_use, replace=False)
                chosen: list[np.ndarray] = []
                for pi in picked:
                    arr = pools[train_names[pi]]
                    take = min(per, len(arr))
                    sub = rng.choice(len(arr), size=take, replace=False)
                    chosen.append(arr[sub])
                train = np.vstack(chosen)
                rows.append({"arm": "S3_units", "held_out": held,
                             "n_train": int(len(train)), "k_bearings": k_use,
                             "test_fp": fp_rate(train, pools[held])})
        print(f"    S3 {rng_tag} held={held} (kmax={k_max}) done", flush=True)
    return rows


def summarize(frame: pd.DataFrame, tag: str) -> pd.DataFrame:
    out = (frame.groupby(["dataset", "arm", "n_train", "k_bearings"], dropna=False)
           .agg(mean_fp=("test_fp", "mean"), sd_fp=("test_fp", "std"),
                folds=("held_out", "nunique"), n=("test_fp", "size"))
           .reset_index())
    out["tag"] = tag
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["main", "sensitivity"], default="main")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUTPUT_DIR / f"exp_v2_04_{args.part}.log")
    frame = pd.read_csv(OUTPUT_DIR / "r2f_features.csv")
    print(f"特征缓存：{len(frame)} 行；数据集 {sorted(frame.dataset.unique())}", flush=True)

    if args.part == "main":
        rows = []
        for dataset in sorted(frame.dataset.unique()):
            pools = build_pools(frame, dataset, RULES[1])
            folds = sorted(pools)
            print(f"{dataset}: 纳入 {len(folds)} 颗，健康记录 "
                  f"{sum(len(v) for v in pools.values())} 条", flush=True)
            rows += [dict(r, dataset=dataset) for r in run_s2(pools, folds, f"{dataset}_main", R_MAIN)]
            rows += [dict(r, dataset=dataset) for r in run_s3(pools, folds, f"{dataset}_main", R_MAIN, K_GRID)]
        raw = pd.DataFrame(rows)
        raw.to_csv(OUTPUT_DIR / "exp_v2_04_main_raw.csv", index=False)
        summary = summarize(raw, "main")
        summary.to_csv(OUTPUT_DIR / "exp_v2_04_summary.csv", index=False)
        pd.set_option("display.width", 240)
        print("\n=== 主分析汇总 ===")
        print(summary.round(4).to_string(index=False))
    else:
        rows = []
        for rule in RULES:
            for dataset in sorted(frame.dataset.unique()):
                pools = build_pools(frame, dataset, rule)
                folds = sorted(pools)
                tag = f"{dataset}_f{rule[0]}_lo{rule[1]}"
                print(f"{tag}: 纳入 {len(folds)} 颗", flush=True)
                rows += [dict(r, dataset=dataset, rule=str(rule)) for r in
                         run_s3(pools, folds, tag, R_SENS, [1, 99])]
        raw = pd.DataFrame(rows)
        raw.to_csv(OUTPUT_DIR / "exp_v2_04_sensitivity_raw.csv", index=False)
        summary = summarize(raw, "sensitivity")
        summary.to_csv(OUTPUT_DIR / "exp_v2_04_sensitivity_summary.csv", index=False)
        print("\n=== 敏感性汇总 ===")
        print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()