"""EXP-V2-03：within-bearing vs bearing-level holdout 评价。

预注册：experiments/EXP-V2-03-preregistration.md + 修订 1（均先于运行提交）

三种抽样臂：
- S1 重采样：固定 20 条子集内有放回抽 n
- S2 真实新增：训练池内无放回抽 n 条不同记录
- S3 独立单元数（仅 R2）：固定 20 条记录预算，改变训练轴承数 k = 1..5
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from config import FIGURES_DIR, OUTPUT_DIR
from exp_v1_08_baselines import QUANTILE
from paderborn_data import DEFAULT_CONDITION, load_healthy_dataset

CONDITION = DEFAULT_CONDITION
TEST_RECORD_NUMBERS = (17, 18, 19, 20)
N_SIZES = [10, 20, 40, 60, 80, 96]
K_VALUES = [1, 2, 3, 4, 5]
R_REPEAT = 100
BUDGET = 20
MODAL_CFG = (200, 0.5)
SEED = 0


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
    """确定性种子（跨进程可复现；不用 Python 的 hash()，它受 PYTHONHASHSEED 影响）。"""
    return zlib.crc32("|".join(parts).encode("utf-8"))


def test_fp(train: np.ndarray, test: np.ndarray) -> float:
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=1)
    std = np.where(std == 0, 1.0, std)
    z = (train - mean) / std
    model = IsolationForest(
        n_estimators=MODAL_CFG[0], max_samples=MODAL_CFG[1],
        random_state=SEED, n_jobs=1,
    ).fit(z)
    scores = -model.score_samples(z)
    threshold = float(np.quantile(scores, QUANTILE))
    return float((-model.score_samples((test - mean) / std) > threshold).mean())


def build_units(detail: pd.DataFrame):
    """返回 [(regime, fold, train_names, test_names)]。"""
    units = []
    is_test = detail["record_number"].isin(TEST_RECORD_NUMBERS)
    # R1 within-bearing：训练=各轴承 1-16，测试=各轴承 17-20
    tr = sorted(detail[~is_test]["record_id"].tolist())
    te = sorted(detail[is_test]["record_id"].tolist())
    units.append(("R1_within_bearing", "all", tr, te))
    # R2 bearing-level holdout：逐轴承留出，训练=其余 5 颗的全部记录
    for bearing in sorted(detail["bearing"].unique()):
        tr2 = sorted(detail[detail["bearing"] != bearing]["record_id"].tolist())
        te2 = sorted(detail[detail["bearing"] == bearing]["record_id"].tolist())
        units.append(("R2_bearing_holdout", bearing, tr2, te2))
    return units


def draw_balanced(by_bearing: dict[str, list[str]], k: int, budget: int, rng) -> list[str]:
    bearings = list(by_bearing)
    picked = rng.choice(len(bearings), size=k, replace=False)
    per = budget // k
    remainder = budget - per * k
    chosen: list[str] = []
    for position, index in enumerate(picked):
        take = per + (1 if position < remainder else 0)
        pool = by_bearing[bearings[index]]
        take = min(take, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        chosen.extend(pool[j] for j in idx)
    return chosen


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUTPUT_DIR / "exp_v2_03_run.log")

    arrays, detail = load_healthy_dataset(CONDITION)
    units = build_units(detail)
    print(f"Paderborn {CONDITION}：{len(arrays)} 条记录，{len(units)} 个 unit", flush=True)
    for regime, fold, tr, te in units:
        print(f"  {regime:22s} fold={fold:6s} train={len(tr):3d} test={len(te):3d}", flush=True)

    rows: list[dict] = []
    for regime, fold, train_names, test_names in units:
        test_block = np.vstack([arrays[n] for n in test_names])
        by_bearing: dict[str, list[str]] = {}
        for name in train_names:
            by_bearing.setdefault(name.split("_")[3], []).append(name)
        subset = sorted(train_names)[:BUDGET]

        for arm in ("S1_resample", "S2_new_records"):
            rng = np.random.default_rng(stable_seed(regime, fold, arm))
            source = subset if arm == "S1_resample" else train_names
            for n in N_SIZES:
                if n > len(train_names):
                    continue
                for repeat in range(R_REPEAT):
                    idx = rng.integers(0, len(source), size=n) if arm == "S1_resample" \
                        else rng.choice(len(source), size=n, replace=False)
                    train = np.vstack([arrays[source[i]] for i in idx])
                    rows.append({
                        "regime": regime, "fold": fold, "arm": arm, "n_train": n,
                        "k_bearings": np.nan, "repeat": repeat,
                        "test_fp": test_fp(train, test_block),
                    })
            print(f"    {regime}/{fold}/{arm} done", flush=True)

        if regime.startswith("R2"):
            rng = np.random.default_rng(stable_seed(regime, fold, "S3_units"))
            for k in K_VALUES:
                if k > len(by_bearing):
                    continue
                for repeat in range(R_REPEAT):
                    chosen = draw_balanced(by_bearing, k, BUDGET, rng)
                    train = np.vstack([arrays[name] for name in chosen])
                    rows.append({
                        "regime": regime, "fold": fold, "arm": "S3_units", "n_train": len(chosen),
                        "k_bearings": k, "repeat": repeat,
                        "test_fp": test_fp(train, test_block),
                    })
            print(f"    {regime}/{fold}/S3 done", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "exp_v2_03_curve.csv", index=False)

    summary = (frame.groupby(["regime", "fold", "arm", "n_train", "k_bearings"], dropna=False)
               .agg(mean_fp=("test_fp", "mean"), sd_fp=("test_fp", "std"),
                    n_repeat=("test_fp", "size"))
               .reset_index())
    summary.to_csv(OUTPUT_DIR / "exp_v2_03_summary.csv", index=False)

    pd.set_option("display.width", 240)
    print("\n=== R2 各折 n=96 对比 ===")
    r2 = frame[(frame.regime == "R2_bearing_holdout") & (frame.n_train == 96)]
    piv = r2.pivot_table(index="fold", columns="arm", values="test_fp",
                         aggfunc=["mean", "std"])
    print((piv * 100).round(2).to_string())

    print("\n=== S3：固定 20 条记录，改变训练轴承数 ===")
    s3 = frame[frame.arm == "S3_units"]
    piv3 = s3.pivot_table(index="fold", columns="k_bearings", values="test_fp", aggfunc="std")
    print((piv3 * 100).round(2).to_string())

    print("\n=== 预注册判据 ===")
    folds = sorted(r2.fold.unique())
    p1 = 0
    for fold in folds:
        sub = r2[r2.fold == fold]
        s1 = float(sub[sub.arm == "S1_resample"].test_fp.std(ddof=1))
        s2 = float(sub[sub.arm == "S2_new_records"].test_fp.std(ddof=1))
        p1 += int(s2 < s1)
    print(f"P1（R2 中 SD96(S2) < SD96(S1)）：{p1}/{len(folds)}")
    p4 = 0
    for fold in folds:
        sub = frame[(frame.regime == "R2_bearing_holdout") & (frame.fold == fold)
                    & (frame.arm == "S2_new_records")]
        g = sub.groupby("n_train")["test_fp"].std(ddof=1)
        rho, _ = stats.spearmanr(g.index, g.values)
        p4 += int(rho < 0)
    print(f"P4（R2 中 S2 的 SD 随 n 下降）：{p4}/{len(folds)}")
    p5 = p6 = 0
    ratios = []
    for fold in folds:
        sub = s3[s3.fold == fold].groupby("k_bearings")["test_fp"].std(ddof=1)
        rho, _ = stats.spearmanr(sub.index.astype(float), sub.values)
        p5 += int(rho < 0)
        p6 += int(sub.iloc[-1] < sub.iloc[0])
        ratios.append(float(sub.iloc[0] / sub.iloc[-1]) if sub.iloc[-1] > 0 else np.nan)
    print(f"P5（S3 的 SD 随轴承数下降）：{p5}/{len(folds)}")
    print(f"P6（SD(k=5) < SD(k=1)）：{p6}/{len(folds)}")
    print(f"P7（SD(k=1)/SD(k=5) 每折）：{[None if np.isnan(r) else round(r,2) for r in ratios]}")

    # R1 vs R2 绝对水平对比（同 n、同臂）
    print("\n=== P2：绝对误报率水平对比（%）===")
    comp = []
    for arm in ("S1_resample", "S2_new_records"):
        for n in N_SIZES:
            r1v = frame[(frame.regime == "R1_within_bearing") & (frame.arm == arm)
                        & (frame.n_train == n)]["test_fp"].mean()
            r2v = frame[(frame.regime == "R2_bearing_holdout") & (frame.arm == arm)
                        & (frame.n_train == n)]["test_fp"].mean()
            comp.append({"arm": arm, "n": n, "R1_within_%": round(r1v * 100, 2),
                         "R2_holdout_%": round(r2v * 100, 2)})
    comp_frame = pd.DataFrame(comp)
    comp_frame.to_csv(OUTPUT_DIR / "exp_v2_03_regime_comparison.csv", index=False)
    print(comp_frame.to_string(index=False))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for arm, marker in (("S1_resample", "o"), ("S2_new_records", "s")):
        for regime, style in (("R1_within_bearing", ":"), ("R2_bearing_holdout", "-")):
            sub = frame[(frame.regime == regime) & (frame.arm == arm)]
            g = sub.groupby("n_train")["test_fp"].std(ddof=1)
            axes[0].plot(g.index, g.values * 100, marker=marker, linestyle=style,
                         label=f"{regime[:2]} {arm[:2]}")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].axhline(1.0, color="gray", linestyle=":")
    axes[0].set_xlabel("Training records (nominal n)")
    axes[0].set_ylabel("SD of reported false-alarm rate (pp)")
    axes[0].set_title("Within-bearing vs bearing-level holdout")
    axes[0].legend(fontsize=7)

    axes[1].bar(np.arange(len(folds)) - 0.2,
                [s3[s3.fold == f].groupby("k_bearings")["test_fp"].std(ddof=1).iloc[-1] * 100 for f in folds],
                width=0.4, label="k=5 bearings")
    axes[1].bar(np.arange(len(folds)) + 0.2,
                [s3[s3.fold == f].groupby("k_bearings")["test_fp"].std(ddof=1).iloc[0] * 100 for f in folds],
                width=0.4, label="k=1 bearing")
    axes[1].set_xticks(np.arange(len(folds))); axes[1].set_xticklabels(folds, rotation=30, fontsize=7)
    axes[1].set_ylabel("SD (pp) at fixed 20 records")
    axes[1].set_title("S3: independent units at fixed sample size")
    axes[1].legend(fontsize=8)

    for fold in folds:
        sub = s3[s3.fold == fold].groupby("k_bearings")["test_fp"].std(ddof=1)
        axes[2].plot(sub.index, sub.values * 100, marker="o", alpha=0.7, label=fold)
    axes[2].set_xlabel("Number of distinct training bearings")
    axes[2].set_ylabel("SD (pp)")
    axes[2].set_title("S3 per fold")
    axes[2].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "exp_v2_03_bearing_level_holdout.png", dpi=150)
    plt.close(figure)
    print("\n已保存曲线与图。")


if __name__ == "__main__":
    main()