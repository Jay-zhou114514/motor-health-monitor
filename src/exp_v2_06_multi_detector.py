"""EXP-V2-06：在 run-to-failure 数据集上补齐第二个/第三个检测器。

预注册：experiments/EXP-V2-06-preregistration.md（先于运行提交）
复用 EXP-V2-05 的采样设计，只替换检测器（A 3σ RMS / B 马氏距离）。
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR
from exp_v1_08_baselines import QUANTILE
from covariance_geometry import center, empirical_covariance, mahalanobis_distances
from runtofailure_data import FEATURES, healthy_size, included

R_REPEAT = 50
BUDGET = 20
MAX_K = 4


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


def fit_rms(train):
    scores = train[:, 0]
    return float(np.quantile(scores, QUANTILE)), scores


def score_rms(state, data):
    return data[:, 0]


def fit_maha(train):
    mean = train.mean(axis=0)
    cov = empirical_covariance(center(train)) + np.eye(train.shape[1]) * 1e-12
    d = mahalanobis_distances(train, mean, cov)
    return float(np.quantile(d, QUANTILE)), (mean, cov)


def score_maha(state, data):
    mean, cov = state
    return mahalanobis_distances(data, mean, cov)


DETECTORS = {"A_rms_3sigma": (fit_rms, score_rms), "B_mahalanobis": (fit_maha, score_maha)}


def operating_condition(dataset: str, condition: str, bearing: str) -> str:
    return condition if dataset == "XJTU-SY" else "C" + bearing.split("_")[0].replace("Bearing", "")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUTPUT_DIR / "exp_v2_06_run.log")
    frame = pd.read_csv(OUTPUT_DIR / "r2f_features.csv")

    pools: dict[tuple[str, str], dict[str, dict]] = {}
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

    rows: list[dict] = []
    for (dataset, cond), bearings in sorted(pools.items()):
        names = sorted(bearings)
        if len(names) < 2:
            continue
        k_max = min(MAX_K, len(names) - 1)
        print(f"[{dataset}/{cond}] 轴承 {len(names)}，k_max={k_max}", flush=True)
        for held in names:
            train_names = [b for b in names if b != held]
            rng = np.random.default_rng(stable_seed("V206", dataset, cond, held))
            for k in range(1, k_max + 1):
                per = max(1, BUDGET // k)
                for _ in range(R_REPEAT):
                    picked = rng.choice(len(train_names), size=k, replace=False)
                    blocks = []
                    for pi in picked:
                        arr = bearings[train_names[pi]]
                        take = min(per, len(arr))
                        blocks.append(arr[rng.choice(len(arr), size=take, replace=False)])
                    train = np.vstack(blocks)
                    test = bearings[held]
                    for name, (fit, score) in DETECTORS.items():
                        thr, state = fit(train)
                        rows.append({"dataset": dataset, "condition": cond, "held_out": held,
                                     "detector": name, "k_bearings": k,
                                     "n_train": int(len(train)),
                                     "test_fp": float((score(state, test) > thr).mean())})
        print(f"    [{dataset}/{cond}] done", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT_DIR / "exp_v2_06_raw.csv", index=False)
    summary = (raw.groupby(["detector", "dataset", "condition", "k_bearings"])
               .agg(mean_fp=("test_fp", "mean"), sd_fp=("test_fp", "std"),
                    folds=("held_out", "nunique"), n=("test_fp", "size")).reset_index())
    summary.to_csv(OUTPUT_DIR / "exp_v2_06_summary.csv", index=False)

    pd.set_option("display.width", 240)
    print("\n=== SD（pp）随 k 的变化 ===")
    for det in sorted(summary.detector.unique()):
        sub = summary[summary.detector == det]
        piv = sub.pivot_table(index=["dataset", "condition"], columns="k_bearings", values="sd_fp") * 100
        print(f"\n[{det}]")
        print(piv.round(2).to_string())
        print("  均值误报（%）：")
        piv2 = sub.pivot_table(index=["dataset", "condition"], columns="k_bearings", values="mean_fp") * 100
        print(piv2.round(2).to_string())

    print("\n=== 判据 ===")
    for det in sorted(summary.detector.unique()):
        z1 = z2 = tot = 0
        for (ds, cond), g in summary[summary.detector == det].groupby(["dataset", "condition"]):
            kmax = int(g.k_bearings.max())
            s1 = g[g.k_bearings == 1].sd_fp.iloc[0]
            sk = g[g.k_bearings == kmax].sd_fp.iloc[0]
            rho = np.corrcoef(g.k_bearings, g.sd_fp)[0, 1]
            z1 += int(rho < 0); z2 += int(sk < s1); tot += 1
        print(f"  {det}: Z1={z1}/{tot}  Z2={z2}/{tot}")


if __name__ == "__main__":
    main()