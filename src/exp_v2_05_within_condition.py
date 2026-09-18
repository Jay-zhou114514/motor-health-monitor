"""EXP-V2-05：工况内分层的独立单元对照。

预注册：experiments/EXP-V2-05-preregistration.md（先于运行提交）

目的：拆开 EXP-V2-04 的混杂——"训练轴承数 k"与"工况多样性"。
做法：训练轴承**只取自与被测轴承相同的工况**，使工况数恒定 = 1。
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import OUTPUT_DIR
from exp_v1_08_baselines import QUANTILE
from runtofailure_data import FEATURES, healthy_size, included

R_REPEAT = 50
BUDGET = 20
MAX_K = 4
MODAL_CFG = (200, 0.5)


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


def operating_condition(dataset: str, condition: str, bearing: str) -> str:
    """统一"工况"口径：XJTU-SY 用目录名；PRONOSTIA 用轴承名前缀（Bearing1_x → C1）。"""
    if dataset == "XJTU-SY":
        return condition
    return "C" + bearing.split("_")[0].replace("Bearing", "")


def build(frame: pd.DataFrame):
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
    sys.stdout = Tee(OUTPUT_DIR / "exp_v2_05_run.log")
    frame = pd.read_csv(OUTPUT_DIR / "r2f_features.csv")
    pools = build(frame)

    rows: list[dict] = []
    for (dataset, cond), bearings in sorted(pools.items()):
        names = sorted(bearings)
        if len(names) < 2:
            print(f"[{dataset}/{cond}] 只有 {len(names)} 颗，跳过", flush=True)
            continue
        k_max = min(MAX_K, len(names) - 1)
        print(f"[{dataset}/{cond}] 轴承 {len(names)} 颗，k_max={k_max}", flush=True)
        for held in names:
            train_names = [b for b in names if b != held]
            rng = np.random.default_rng(stable_seed("V205", dataset, cond, held))
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
                    rows.append({"dataset": dataset, "condition": cond, "held_out": held,
                                 "k_bearings": k, "n_train": int(len(train)),
                                 "test_fp": fp_rate(train, bearings[held])})
        print(f"    [{dataset}/{cond}] done", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT_DIR / "exp_v2_05_raw.csv", index=False)
    summary = (raw.groupby(["dataset", "condition", "k_bearings"])
               .agg(mean_fp=("test_fp", "mean"), sd_fp=("test_fp", "std"),
                    folds=("held_out", "nunique"), n=("test_fp", "size")).reset_index())
    summary.to_csv(OUTPUT_DIR / "exp_v2_05_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== 工况内 S3：SD（pp）随 k 的变化 ===")
    piv = summary.pivot_table(index=["dataset", "condition"], columns="k_bearings", values="sd_fp")
    print((piv * 100).round(2).to_string())

    # Y1 / Y2 / Y3
    print("\n=== 判据 ===")
    y1 = y2 = total = 0
    ratios = {}
    for (dataset, cond), sub in summary.groupby(["dataset", "condition"]):
        kmax = int(sub.k_bearings.max())
        s1 = float(sub[sub.k_bearings == 1].sd_fp.iloc[0])
        sk = float(sub[sub.k_bearings == kmax].sd_fp.iloc[0])
        rho = np.corrcoef(sub.k_bearings, sub.sd_fp)[0, 1]
        y1 += int(rho < 0)
        y2 += int(sk < s1)
        total += 1
        ratios[(dataset, cond)] = (s1 / sk) if sk > 0 else np.inf
        print(f"  {dataset}/{cond}: k=1 SD={s1*100:.2f}  k={kmax} SD={sk*100:.2f}  R={ratios[(dataset,cond)]:.2f}")
    print(f"Y1（SD 随 k 下降）: {y1}/{total}")
    print(f"Y2（SD(kmax) < SD(k=1)）: {y2}/{total}")

    main_path = OUTPUT_DIR / "exp_v2_04_main_raw.csv"
    if main_path.exists():
        m = pd.read_csv(main_path)
        s3 = m[m.arm == "S3_units"]
        print("\n=== Y3：工况内 R_within vs 跨工况 R_mixed（V2-04）===")
        for dataset in sorted({k[0] for k in ratios}):
            sub = s3[s3.dataset == dataset]
            kmax = int(sub.k_bearings.max())
            s1 = sub[sub.k_bearings == 1].test_fp.std(ddof=1)
            sk = sub[sub.k_bearings == kmax].test_fp.std(ddof=1)
            r_mixed = s1 / sk
            r_within = float(np.mean([v for (d, _), v in ratios.items() if d == dataset and np.isfinite(v)]))
            verdict = ("独立单元效应独立成立" if r_within / r_mixed >= 0.6
                       else "效应主要来自工况多样性" if r_within < r_mixed * 0.6 else "混合")
            print(f"  {dataset}: R_within={r_within:.2f}  R_mixed={r_mixed:.2f}  "
                  f"比值={r_within/r_mixed:.2f} -> {verdict}")


if __name__ == "__main__":
    main()