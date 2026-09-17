"""EXP-V1-10：模型选择过程不确定性（量化）。

预注册：experiments/EXP-V1-10-preregistration.md
修订 1：experiments/EXP-V1-10-preregistration-amendment-1.md（零模型 + 分数层指标）
修订 2：experiments/EXP-V1-10-preregistration-amendment-2.md（计算规模）

核心纪律：
- 固定测试集（IMS 1st_test 前 2 个文件）只用于最终评估；
- 所有选择只发生在训练池内部的 fit / validation；
- 零模型使用完全相同的协议，只替换数据生成方式；
- 每个实验臂各自落盘，支持断点续跑（--force 可强制重算）。

输出：
- outputs/exp_v1_10_arm_*.csv（逐臂检查点）
- outputs/exp_v1_10_replicates.csv / _null.csv / _ncurve.csv / _summary.csv
- outputs/figures/exp_v1_10_selection_uncertainty.png
- outputs/exp_v1_10_run.log
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from config import FIGURES_DIR, OUTPUT_DIR
from exp_v1_08_baselines import FEATURES, QUANTILE, RMSThreshold, load_healthy_records

IF_GRID = [(n, m) for n in (200, 100) for m in (0.5, 0.8, 1.0)]
OCSVM_GRID = [
    (nu, gamma)
    for nu in (0.001, 0.005, 0.01, 0.05, 0.1)
    for gamma in ("scale", 0.1, 1.0, 10.0)
]

TEST_FILES = ("2003.10.22.12.06.24.txt", "2003.10.22.12.09.13.txt")
BATCH = "IMS 1st_test"


class Tee:
    def __init__(self, path: Path):
        self.path = path
        self.handle = open(path, "w", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        self.handle.write(text)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.handle.flush()


def to_arrays(tables: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    return {name: table[FEATURES].to_numpy(float) for name, table in tables.items()}


def fit_model(method: str, train: np.ndarray, cfg):
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=1)
    std = np.where(std == 0, 1.0, std)
    standardized = (train - mean) / std
    if method == "forest":
        n_estimators, max_samples = cfg
        model = IsolationForest(
            n_estimators=int(n_estimators),
            max_samples=float(max_samples),
            random_state=0,
            n_jobs=1,
        ).fit(standardized)
    else:
        nu, gamma = cfg
        model = OneClassSVM(kernel="rbf", nu=float(nu), gamma=gamma).fit(standardized)
    scores = raw_scores(model, standardized)
    threshold = float(np.quantile(scores, QUANTILE))
    return {
        "mean": mean,
        "std": std,
        "model": model,
        "threshold": threshold,
        "fit_scores": np.sort(scores),
    }


def raw_scores(model, standardized: np.ndarray) -> np.ndarray:
    if isinstance(model, IsolationForest):
        return -model.score_samples(standardized)
    return -model.decision_function(standardized)


def score_model(state, data: np.ndarray) -> np.ndarray:
    standardized = (data - state["mean"]) / state["std"]
    return raw_scores(state["model"], standardized)


def percentile_of(state, data: np.ndarray) -> np.ndarray:
    scores = score_model(state, data)
    fit_scores = state["fit_scores"]
    return np.searchsorted(fit_scores, scores, side="right") / len(fit_scores)


def run_selection(method: str, grid, fit_tables, val_tables, test_tables) -> dict:
    fit_blocks = list(fit_tables.values())
    val_block = np.vstack(list(val_tables.values()))
    test_block = np.vstack(list(test_tables.values()))
    start = time.perf_counter()
    outcomes = []
    for cfg in grid:
        train = np.vstack(fit_blocks)
        state = fit_model(method, train, cfg)
        val_scores = score_model(state, val_block)
        val_fp = float((val_scores > state["threshold"]).mean())
        val_pct = percentile_of(state, val_block)
        test_scores = score_model(state, test_block)
        test_pct = percentile_of(state, test_block)
        outcomes.append(
            {
                "cfg": cfg,
                "val_fp": val_fp,
                "val_pct_mean": float(val_pct.mean()),
                "test_fp": float((test_scores > state["threshold"]).mean()),
                "test_pct_mean": float(test_pct.mean()),
                "test_pct_max": float(test_pct.max()),
                "test_pct_q90": float(np.quantile(test_pct, 0.9)),
            }
        )
    best_val = min(item["val_fp"] for item in outcomes)
    tied = [item for item in outcomes if item["val_fp"] <= best_val + 1e-12]
    chosen = dict(tied[0])
    chosen["cfg"] = str(chosen["cfg"])
    chosen["n_tied_at_min"] = len(tied)
    chosen["seconds"] = time.perf_counter() - start
    return chosen


def resample_split(rng, names, n_fit: int, n_val: int, n_test: int = 0):
    order = rng.permutation(len(names))
    picked = [names[i] for i in order]
    return (
        picked[:n_fit],
        picked[n_fit : n_fit + n_val],
        picked[n_fit + n_val : n_fit + n_val + n_test],
    )


def subset(tables, names):
    return {name: tables[name] for name in names}


def run_real_arm(arrays, method, grid, r, seed, design, fixed_test) -> list[dict]:
    names_all = sorted(arrays)
    pool = [name for name in names_all if name not in fixed_test] if design == "A" else names_all
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    for index in range(r):
        if design == "A":
            fit_names, val_names, _ = resample_split(rng, pool, 7, 3)
            test_tables = subset(arrays, fixed_test)
        else:
            fit_names, val_names, test_names = resample_split(rng, pool, 7, 3, 2)
            test_tables = subset(arrays, test_names)
        outcome = run_selection(
            method, grid, subset(arrays, fit_names), subset(arrays, val_names), test_tables
        )
        outcome.update(
            {
                "design": design,
                "method": method,
                "replicate": index,
                "fit_files": len(fit_names),
                "val_files": len(val_names),
                "test_files": len(test_tables),
                "test_windows": int(sum(len(t) for t in test_tables.values())),
            }
        )
        rows.append(outcome)
        if (index + 1) % 25 == 0:
            print(f"    design {design} / {method}: {index + 1}/{r}", flush=True)
    return rows


def build_null_dataset(rng, kind, shapes, pool_windows, mu, cov):
    pseudo = {}
    for name, count in shapes.items():
        if kind == "n1":
            index = rng.integers(0, len(pool_windows), size=count)
            pseudo[name] = pool_windows[index]
        else:
            pseudo[name] = rng.multivariate_normal(mu, cov, size=count)
    return pseudo


def run_null_arm(arrays, method, grid, m_datasets, r, seed, kind, fixed_test,
                 checkpoint: Path | None = None) -> list[dict]:
    pool = sorted(arrays)
    pool_windows = np.vstack([arrays[name] for name in pool])
    shapes = {name: len(arrays[name]) for name in pool}
    if kind == "n2":
        lw = LedoitWolf().fit(pool_windows)
        mu, cov = lw.location_, lw.covariance_
    else:
        mu = cov = None
    rows: list[dict] = []
    start_index = 0
    if checkpoint is not None and checkpoint.exists():
        existing = pd.read_csv(checkpoint).to_dict("records")
        if existing:
            rows = existing
            start_index = int(max(row["dataset_index"] for row in rows)) + 1
            print(f"    [resume] 零模型 {kind}：从数据集 {start_index} 继续", flush=True)
    for dataset_index in range(start_index, m_datasets):
        rng = np.random.default_rng(seed + 100_003 * dataset_index)
        pseudo = build_null_dataset(rng, kind, shapes, pool_windows, mu, cov)
        cfg_counts: Counter = Counter()
        val_fps, test_fps, tie_flags = [], [], []
        for _ in range(r):
            fit_names, val_names, _ = resample_split(rng, pool, 7, 3)
            outcome = run_selection(
                method, grid, subset(pseudo, fit_names), subset(pseudo, val_names),
                subset(arrays, fixed_test),
            )
            cfg_counts[outcome["cfg"]] += 1
            val_fps.append(outcome["val_fp"])
            test_fps.append(outcome["test_fp"])
            tie_flags.append(outcome["n_tied_at_min"] >= 2)
        rows.append(
            {
                "null_kind": kind,
                "dataset_index": dataset_index,
                "n_replicates": r,
                "modal_share": cfg_counts.most_common(1)[0][1] / r,
                "n_distinct_cfg": len(cfg_counts),
                "tie_share": float(np.mean(tie_flags)),
                "val_fp_mean": float(np.mean(val_fps)),
                "test_fp_mean": float(np.mean(test_fps)),
                "test_fp_std": float(np.std(test_fps, ddof=1)),
            }
        )
        if checkpoint is not None and ((dataset_index + 1) % 5 == 0 or dataset_index == m_datasets - 1):
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
        if (dataset_index + 1) % 5 == 0:
            print(f"    null {kind}: {dataset_index + 1}/{m_datasets}", flush=True)
    return rows


def run_ncurve_arm(arrays, method, grid, ks, m_datasets, r, seed, fixed_test,
                   checkpoint: Path | None = None) -> list[dict]:
    pool = sorted(arrays)
    pool_windows = np.vstack([arrays[name] for name in pool])
    rows: list[dict] = []
    done: set[tuple[int, int]] = set()
    if checkpoint is not None and checkpoint.exists():
        existing = pd.read_csv(checkpoint).to_dict("records")
        if existing:
            rows = existing
            done = {(int(row["n_files"]), int(row["dataset_index"])) for row in rows}
            print(f"    [resume] 样本量曲线：已完成 {len(done)} 个组合", flush=True)
    for k in ks:
        for dataset_index in range(m_datasets):
            if (k, dataset_index) in done:
                continue
            rng = np.random.default_rng(seed + 7_919 * k + 101 * dataset_index)
            shapes = {f"pseudo_{i:03d}": 7 for i in range(k)}
            pseudo = build_null_dataset(rng, "n1", shapes, pool_windows, None, None)
            names = sorted(pseudo)
            cfg_counts: Counter = Counter()
            test_fps = []
            for _ in range(r):
                fit_names, val_names, _ = resample_split(rng, names, max(3, k - 3), 3)
                outcome = run_selection(
                    method, grid, subset(pseudo, fit_names), subset(pseudo, val_names),
                    subset(arrays, fixed_test),
                )
                cfg_counts[outcome["cfg"]] += 1
                test_fps.append(outcome["test_fp"])
            rows.append(
                {
                    "n_files": k,
                    "dataset_index": dataset_index,
                    "n_replicates": r,
                    "modal_share": cfg_counts.most_common(1)[0][1] / r,
                    "n_distinct_cfg": len(cfg_counts),
                    "test_fp_mean": float(np.mean(test_fps)),
                    "test_fp_std": float(np.std(test_fps, ddof=1)),
                }
            )
            if checkpoint is not None:
                pd.DataFrame(rows).to_csv(checkpoint, index=False)
        subset_rows = [row for row in rows if row["n_files"] == k]
        if subset_rows:
            print(
                f"    n-curve k={k}: mean modal share = "
                f"{np.mean([row['modal_share'] for row in subset_rows]):.3f}",
                flush=True,
            )
    return rows


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = z * np.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(rows: list[dict], label: str) -> dict:
    frame = pd.DataFrame(rows)
    counts = Counter(frame["cfg"].tolist())
    modal_cfg, modal_count = counts.most_common(1)[0]
    total = len(frame)
    if len(counts) == 1:
        entropy = 0.0
    else:
        props = np.array(list(counts.values())) / total
        entropy = float(-np.sum(props * np.log(props)) / np.log(len(counts)))
    lo, hi = wilson_interval(modal_count, total)
    return {
        "arm": label,
        "n_replicates": total,
        "modal_cfg": modal_cfg,
        "modal_share": modal_count / total,
        "modal_share_lo": lo,
        "modal_share_hi": hi,
        "n_distinct_cfg": len(counts),
        "entropy_norm": entropy,
        "tie_share": float((frame["n_tied_at_min"] >= 2).mean()),
        "test_fp_mean": float(frame["test_fp"].mean()),
        "test_fp_std": float(frame["test_fp"].std(ddof=1)),
        "test_fp_min": float(frame["test_fp"].min()),
        "test_fp_max": float(frame["test_fp"].max()),
        "test_pct_mean_mean": float(frame["test_pct_mean"].mean()),
        "test_pct_mean_std": float(frame["test_pct_mean"].std(ddof=1)),
        "val_fp_mean": float(frame["val_fp"].mean()),
        "seconds_total": float(frame["seconds"].sum()),
    }


def reference_controls(arrays, fixed_test) -> list[dict]:
    pool = [name for name in sorted(arrays) if name not in fixed_test]
    train = np.vstack([arrays[name] for name in pool])
    test = np.vstack([arrays[name] for name in fixed_test])
    rows = []
    rms = RMSThreshold().fit(train)
    scores = rms.score(test)
    rows.append(
        {
            "control": "3-sigma RMS (no selection)",
            "cfg": "-",
            "test_fp": float((scores > rms.threshold_).mean()),
            "n_test_windows": len(test),
        }
    )
    state = fit_model("forest", train, (200, 0.8))
    scores = score_model(state, test)
    rows.append(
        {
            "control": "Isolation Forest (default cfg, no search)",
            "cfg": "(200, 0.8)",
            "test_fp": float((scores > state["threshold"]).mean()),
            "n_test_windows": len(test),
        }
    )
    return rows


def make_figure(replicates, null_rows, ncurve_rows, summary, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    forest_a = pd.DataFrame(
        [row for row in replicates if row["design"] == "A" and row["method"] == "forest"]
    )
    frame_summary = summary.set_index("arm")
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))

    labels = list(frame_summary.index)
    axes[0, 0].barh(
        np.arange(len(labels)),
        [frame_summary.loc[label, "modal_share"] for label in labels],
        color="tab:blue",
    )
    for index, label in enumerate(labels):
        value = frame_summary.loc[label, "modal_share"]
        axes[0, 0].text(
            value + 0.01, index,
            f"{value:.2f} ({int(frame_summary.loc[label, 'n_distinct_cfg'])} cfgs)",
            va="center", fontsize=8,
        )
    axes[0, 0].set_yticks(np.arange(len(labels)))
    axes[0, 0].set_yticklabels(labels, fontsize=8)
    axes[0, 0].set_xlim(0, 1.35)
    axes[0, 0].set_xlabel("Modal share of selected config")
    axes[0, 0].set_title("Selection stability")

    n1 = pd.DataFrame([row for row in null_rows if row["null_kind"] == "n1"])
    if len(n1):
        axes[0, 1].hist(n1["test_fp_std"] * 100, bins=15, alpha=0.6,
                        label=f"null N1 ({len(n1)} datasets)")
    axes[0, 1].axvline(
        forest_a["test_fp"].std(ddof=1) * 100, color="crimson", linewidth=2,
        label="observed (design A)",
    )
    axes[0, 1].set_xlabel("SD of test FP rate across replicates (%)")
    axes[0, 1].set_title("Test-FP spread: observed vs noise baseline")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].hist(forest_a["test_pct_mean"], bins=15, alpha=0.7, color="tab:green")
    axes[1, 0].axvline(0.99, color="gray", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("Mean training-percentile of test windows")
    axes[1, 0].set_title("Continuous-scale test score (design A)")

    if ncurve_rows:
        ncurve = pd.DataFrame(ncurve_rows)
        grouped = ncurve.groupby("n_files")["modal_share"]
        axes[1, 1].errorbar(
            grouped.mean().index, grouped.mean().values,
            yerr=grouped.std(ddof=1).fillna(0).values, marker="o",
        )
        axes[1, 1].set_xscale("log")
        axes[1, 1].set_xlabel("Number of files available for selection")
        axes[1, 1].set_ylabel("Modal share")
        axes[1, 1].set_title("Null noise baseline vs sample size (N1)")
        axes[1, 1].set_ylim(0, 1.05)

    figure.suptitle("EXP-V1-10: selection-process uncertainty", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def load_or_run(tag: str, loader, force: bool, cache_dir: Path):
    path = cache_dir / f"exp_v1_10_arm_{tag}.csv"
    if path.exists() and not force:
        rows = pd.read_csv(path).to_dict("records")
        print(f"  [resume] 复用 {path.name}（{len(rows)} 行）", flush=True)
        return rows
    start = time.perf_counter()
    rows = loader()
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  [{tag}] 完成 {len(rows)} 行，耗时 {time.perf_counter() - start:.1f} s", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r-primary", type=int, default=50)
    parser.add_argument("--r-large", type=int, default=200)
    parser.add_argument("--null-datasets", type=int, default=40)
    parser.add_argument("--null-r", type=int, default=20)
    parser.add_argument("--ncurve-datasets", type=int, default=5)
    parser.add_argument("--ncurve-r", type=int, default=15)
    parser.add_argument("--skip-ncurve", action="store_true")
    parser.add_argument("--skip-svm", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / f"exp_v1_10_run{args.tag}.log"
    sys.stdout = Tee(log_path)

    print("EXP-V1-10 选择过程不确定性")
    print(f"参数：R={args.r_primary}, R_large={args.r_large}, "
          f"null={args.null_datasets}x{args.null_r}, "
          f"ncurve={args.ncurve_datasets}x{args.ncurve_r}, force={args.force}")
    print(f"固定测试集：{TEST_FILES}")

    datasets = load_healthy_records()
    tables = datasets[BATCH]
    missing = [name for name in TEST_FILES if name not in tables]
    if missing:
        raise SystemExit(f"固定测试文件缺失：{missing}")
    arrays = to_arrays(tables)
    print(f"{BATCH}：{len(arrays)} 个健康文件，{sum(len(v) for v in arrays.values())} 个窗口")

    controls = reference_controls(arrays, TEST_FILES)
    print("\n=== 无选择过程的参考值（同一固定测试集）===")
    for row in controls:
        print(f"  {row['control']}: FP rate = {row['test_fp']:.4f}"
              f"（{row['n_test_windows']} 个窗口）")

    print("\n=== 真实数据：设计 A（测试集固定）===")
    replicate_frame = pd.DataFrame(load_or_run(
        f"realA_R{args.r_primary}",
        lambda: run_real_arm(arrays, "forest", IF_GRID, args.r_primary, 20260917, "A", TEST_FILES),
        args.force, OUTPUT_DIR,
    ))
    print(f"\n=== 真实数据：设计 A 精度检查（R = {args.r_large}）===")
    large_frame = pd.DataFrame(load_or_run(
        f"realA_R{args.r_large}",
        lambda: run_real_arm(arrays, "forest", IF_GRID, args.r_large, 777001, "A", TEST_FILES),
        args.force, OUTPUT_DIR,
    ))
    print("\n=== 真实数据：设计 B（测试集也重采样）===")
    design_b_frame = pd.DataFrame(load_or_run(
        f"realB_R{args.r_primary}",
        lambda: run_real_arm(arrays, "forest", IF_GRID, args.r_primary, 55501, "B", TEST_FILES),
        args.force, OUTPUT_DIR,
    ))
    svm_frame = pd.DataFrame()
    if not args.skip_svm:
        print("\n=== 真实数据：设计 A / One-Class SVM（次要）===")
        svm_frame = pd.DataFrame(load_or_run(
            f"svmA_R{args.r_primary}",
            lambda: run_real_arm(arrays, "svm", OCSVM_GRID, args.r_primary, 31337, "A", TEST_FILES),
            args.force, OUTPUT_DIR,
        ))

    print("\n=== 零模型 N1（非参，iForest）===")
    null_n1 = run_null_arm(
        arrays, "forest", IF_GRID, args.null_datasets, args.null_r, 909, "n1", TEST_FILES,
        checkpoint=OUTPUT_DIR / "exp_v1_10_arm_nullN1.csv",
    )
    print("\n=== 零模型 N2（参数高斯，iForest）===")
    null_n2 = run_null_arm(
        arrays, "forest", IF_GRID, args.null_datasets, args.null_r, 1234, "n2", TEST_FILES,
        checkpoint=OUTPUT_DIR / "exp_v1_10_arm_nullN2.csv",
    )

    ncurve_rows: list[dict] = []
    if not args.skip_ncurve:
        print("\n=== 零模型 N1：样本量曲线 ===")
        ncurve_rows = run_ncurve_arm(
            arrays, "forest", IF_GRID, [6, 12, 24, 48],
            args.ncurve_datasets, args.ncurve_r, 4321, TEST_FILES,
            checkpoint=OUTPUT_DIR / "exp_v1_10_arm_ncurve.csv",
        )

    all_replicates = pd.concat(
        [replicate_frame, large_frame, design_b_frame, svm_frame], ignore_index=True
    )
    all_replicates.to_csv(OUTPUT_DIR / "exp_v1_10_replicates.csv", index=False)

    null_frame = pd.DataFrame(null_n1 + null_n2)
    null_frame.to_csv(OUTPUT_DIR / "exp_v1_10_null.csv", index=False)
    ncurve_frame = pd.DataFrame(ncurve_rows)
    if len(ncurve_frame):
        ncurve_frame.to_csv(OUTPUT_DIR / "exp_v1_10_ncurve.csv", index=False)
    pd.DataFrame(controls).to_csv(OUTPUT_DIR / "exp_v1_10_controls.csv", index=False)

    summary_rows = [
        summarize(replicate_frame.to_dict("records"), f"forest / design A (R={args.r_primary}, frozen)"),
        summarize(large_frame.to_dict("records"), f"forest / design A (R={args.r_large})"),
        summarize(design_b_frame.to_dict("records"), f"forest / design B (R={args.r_primary}, test resampled)"),
    ]
    if len(svm_frame):
        summary_rows.append(
            summarize(svm_frame.to_dict("records"), f"OC-SVM / design A (R={args.r_primary}, secondary)")
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "exp_v1_10_summary.csv", index=False)

    pd.set_option("display.width", 240)
    print("\n=== 汇总 ===")
    print(summary.round(4).to_string(index=False))

    n1_share = np.array([row["modal_share"] for row in null_n1])
    n2_share = np.array([row["modal_share"] for row in null_n2])
    observed_share = float(summary.loc[0, "modal_share"])
    lo_n1, hi_n1 = np.percentile(n1_share, [5, 95])
    lo_n2, hi_n2 = np.percentile(n2_share, [5, 95])

    print("\n=== 预注册判据（含修订 1 / 修订 2）===")
    p1a = observed_share < 0.5
    p1b = int(summary.loc[0, "n_distinct_cfg"]) >= 3
    print(f"P1a 模态占比 < 50%：{observed_share:.3f} -> {'成立' if p1a else '不成立'}")
    print(f"P1b 不同配置数 >= 3：{int(summary.loc[0, 'n_distinct_cfg'])} -> {'成立' if p1b else '不成立'}")
    p2 = float(summary.loc[0, "test_fp_std"]) >= 0.02
    print(f"P2 测试 FP 标准差 >= 2%：{summary.loc[0, 'test_fp_std']:.4f} -> "
          f"{'成立' if p2 else '不成立'}（分辨率 1/14 = 7.1%）")
    print(f"P3 3σ RMS 为单一确定值：{controls[0]['test_fp']:.4f} -> 成立")
    p4 = float(summary.loc[0, "test_pct_mean_std"]) > 0.02
    print(f"P4 连续尺度 test_pct_mean 标准差 > 0.02：{summary.loc[0, 'test_pct_mean_std']:.4f} -> "
          f"{'成立' if p4 else '不成立'}")
    p5 = float(summary.loc[0, "tie_share"]) >= 0.25
    print(f"P5 并列占比 >= 25%：{summary.loc[0, 'tie_share']:.3f} -> {'成立' if p5 else '不成立'}")
    p6 = lo_n1 <= observed_share <= hi_n1
    print(f"P6 观察值落在 N1 零模型 5–95% 区间 [{lo_n1:.3f}, {hi_n1:.3f}]：{observed_share:.3f} -> "
          f"{'成立（与噪声不可区分）' if p6 else '不成立（超出噪声基准）'}")
    print(f"   N2 零模型 5–95% 区间：[{lo_n2:.3f}, {hi_n2:.3f}]")
    p7 = float(summary.loc[2, "test_fp_std"]) >= float(summary.loc[0, "test_fp_std"])
    print(f"P7 设计 B 标准差 >= 设计 A：{summary.loc[2, 'test_fp_std']:.4f} vs "
          f"{summary.loc[0, 'test_fp_std']:.4f} -> {'成立' if p7 else '不成立'}")

    figure_path = FIGURES_DIR / "exp_v1_10_selection_uncertainty.png"
    make_figure(replicate_frame.to_dict("records"), null_n1 + null_n2, ncurve_rows,
                summary, figure_path)
    print(f"\n已保存：{OUTPUT_DIR / 'exp_v1_10_replicates.csv'}")
    print(f"已保存：{OUTPUT_DIR / 'exp_v1_10_null.csv'}")
    print(f"已保存：{OUTPUT_DIR / 'exp_v1_10_summary.csv'}")
    print(f"已保存：{figure_path}")
    print(f"已保存：{log_path}")


if __name__ == "__main__":
    main()