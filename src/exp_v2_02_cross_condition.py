"""EXP-V2-02：S1 vs S2 对比在其余 3 个工况上的稳健性。

预注册：experiments/EXP-V2-02-preregistration.md（先于运行提交）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import FIGURES_DIR, OUTPUT_DIR
from exp_v1_08_baselines import QUANTILE
from paderborn_data import CONDITIONS, FEATURES, load_healthy_dataset

MAIN_CONDITION = "N15_M07_F10"
ROBUSTNESS_CONDITIONS = [c for c in CONDITIONS if c != MAIN_CONDITION]
N_SIZES = [10, 20, 40, 60, 80, 96]
R_REPEAT = 100
MODAL_CFG = (200, 0.5)
TEST_RECORD_NUMBERS = (17, 18, 19, 20)
S1_SUBSET_SIZE = 20
RANDOM_SEED = 0


class Tee:
    def __init__(self, path: Path):
        self.handle = open(path, "w", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        self.handle.write(text)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.handle.flush()


def fit_threshold_scores(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=1)
    std = np.where(std == 0, 1.0, std)
    standardized = (train - mean) / std
    model = IsolationForest(
        n_estimators=MODAL_CFG[0], max_samples=MODAL_CFG[1],
        random_state=RANDOM_SEED, n_jobs=1,
    ).fit(standardized)
    train_scores = -model.score_samples(standardized)
    threshold = float(np.quantile(train_scores, QUANTILE))
    test_scores = -model.score_samples((test - mean) / std)
    return threshold, train_scores, test_scores


def run_condition(condition: str) -> list[dict]:
    arrays, detail = load_healthy_dataset(condition)
    detail = detail.copy()
    is_test = detail["record_number"].isin(TEST_RECORD_NUMBERS)
    test_names = sorted(detail[is_test]["record_id"].tolist())
    pool_names = sorted(detail[~is_test]["record_id"].tolist())
    order = detail[~is_test].sort_values(["bearing", "record_number"])
    subset_names = sorted(order["record_id"].tolist()[:S1_SUBSET_SIZE])
    test_block = np.vstack([arrays[name] for name in test_names])
    print(f"[{condition}] 记录 {len(arrays)}，测试 {len(test_names)}，训练池 {len(pool_names)}", flush=True)

    rows: list[dict] = []
    for arm in ("S1_resample", "S2_new_records"):
        rng = np.random.default_rng(20260919 + (0 if arm.startswith("S1") else 1))
        source = subset_names if arm.startswith("S1") else pool_names
        for n_train in N_SIZES:
            for repeat in range(R_REPEAT):
                chosen = rng.choice(len(source), size=n_train, replace=arm.startswith("S1"))
                train = np.vstack([arrays[source[i]] for i in chosen])
                threshold, _, test_scores = fit_threshold_scores(train, test_block)
                rows.append({
                    "condition": condition,
                    "arm": arm,
                    "n_train": int(n_train),
                    "repeat": repeat,
                    "test_fp": float((test_scores > threshold).mean()),
                })
        print(f"    {arm} done", flush=True)
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUTPUT_DIR / "exp_v2_02_run.log")

    rows: list[dict] = []
    for condition in ROBUSTNESS_CONDITIONS:
        rows.extend(run_condition(condition))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "exp_v2_02_curve.csv", index=False)

    q = QUANTILE
    summary = (frame.groupby(["condition", "arm", "n_train"])
               .agg(sd_fp=("test_fp", "std"), mean_fp=("test_fp", "mean"),
                    n_repeat=("test_fp", "size"))
               .reset_index())
    summary["n_eff"] = q * (1 - q) / summary["sd_fp"] ** 2
    summary["efficiency"] = summary["n_eff"] / summary["n_train"]

    # 合并主分析（N15_M07_F10）以便同一张表比较
    main_path = OUTPUT_DIR / "exp_v2_01_summary.csv"
    if main_path.exists():
        main = pd.read_csv(main_path)
        main["condition"] = MAIN_CONDITION
        main = main[(main["n_train"].isin(N_SIZES))]
        keep = ["condition", "arm", "n_train", "sd_fp", "mean_fp", "n_eff", "efficiency"]
        summary = pd.concat([main[keep], summary[keep]], ignore_index=True)
        summary = summary.sort_values(["condition", "arm", "n_train"]).reset_index(drop=True)

    summary.to_csv(OUTPUT_DIR / "exp_v2_02_summary.csv", index=False)
    pd.set_option("display.width", 240)
    print("\n=== 汇总（含主分析工况）===")
    print(summary.round(4).to_string(index=False))

    print("\n=== 预注册判据 ===")
    conditions = sorted(summary["condition"].unique())
    r1 = r2 = r3 = r4 = 0
    for condition in conditions:
        sub = summary[(summary["condition"] == condition) & (summary["n_train"] == 96)]
        s1 = float(sub[sub["arm"] == "S1_resample"]["sd_fp"].iloc[0])
        s2 = float(sub[sub["arm"] == "S2_new_records"]["sd_fp"].iloc[0])
        r1 += int(s1 > s2)
        big = summary[(summary["condition"] == condition) & (summary["arm"] == "S2_new_records")
                      & (summary["n_train"] >= 60)]
        r2 += int((big["sd_fp"] <= 0.01).all())
        s1sub = summary[(summary["condition"] == condition) & (summary["arm"] == "S1_resample")]
        last = float(s1sub[s1sub["n_train"] == 96]["sd_fp"].iloc[0])
        mid = float(s1sub[s1sub["n_train"] == 40]["sd_fp"].iloc[0])
        r3 += int(last <= 1.2 * mid)
        e80 = summary[(summary["condition"] == condition) & (summary["arm"] == "S2_new_records")
                      & (summary["n_train"] == 80)]["efficiency"]
        r4 += int(len(e80) and float(e80.iloc[0]) > 1)
        print(f"  {condition}: SD96(S1)={s1*100:.2f}pp vs SD96(S2)={s2*100:.2f}pp | "
              f"S1末/中={last/mid:.2f} | eff80(S2)={float(e80.iloc[0]) if len(e80) else float('nan'):.2f}")
    n = len(conditions)
    print(f"R1（SD96 S1>S2）：{r1}/{n} -> {'稳健' if r1 == n else ('部分' if r1 >= 2 else '不稳健')}")
    print(f"R2（S2 在 n>=60 均 <=1pp）：{r2}/{n}")
    print(f"R3（S1 饱和）：{r3}/{n}")
    print(f"R4（描述性，eff80(S2)>1）：{r4}/{n}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for condition in conditions:
        for arm, marker, style in (("S1_resample", "o", "--"), ("S2_new_records", "s", "-")):
            sub = summary[(summary["condition"] == condition) & (summary["arm"] == arm)]
            axes[0].plot(sub["n_train"], sub["sd_fp"] * 100, marker=marker, linestyle=style,
                         label=f"{condition} {arm.split('_')[0]}", alpha=0.8)
    grid = np.unique(np.round(np.logspace(np.log10(10), np.log10(96), 40)))
    axes[0].plot(grid, np.sqrt(q * (1 - q) / grid) * 100, "k:", label="ideal 1/sqrt(n)")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].axhline(1.0, color="gray", linestyle=":", label="1 pp")
    axes[0].set_xlabel("Training records (nominal n)")
    axes[0].set_ylabel("SD of reported false-alarm rate (pp)")
    axes[0].set_title("EXP-V2-02: cross-condition robustness")
    axes[0].legend(fontsize=6, ncol=2)
    for condition in conditions:
        sub = summary[(summary["condition"] == condition) & (summary["arm"] == "S2_new_records")]
        axes[1].plot(sub["n_train"], sub["efficiency"], marker="s", label=f"{condition} S2")
    axes[1].axhline(1.0, color="k", linestyle="--")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Training records (nominal n)")
    axes[1].set_ylabel("Effective / nominal (S2)")
    axes[1].set_title("Genuinely new records: efficiency by condition")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "exp_v2_02_cross_condition.png", dpi=150)
    plt.close(figure)
    print("\n已保存。")


if __name__ == "__main__":
    main()