"""EXP-V3-03：从 marginal 到 unit-level 覆盖——六种校准的公平对照。

预注册：experiments/EXP-V3-03-preregistration.md（先于运行提交）

M0 全局池化分位（marginal 基线）
M1 Mondrian / group-conditional（组 = 工况）
M2 weighted（权重由交叉拟合 logistic 估计训练 vs 目标单元的密度比；只用目标单元的特征，不用标签）
M3 localized（取与目标单元特征均值最近的 k 个训练实例）
M4 hierarchical（全局 + 组级收缩，λ 由训练内 LOO 选）
M5 自校准（用目标单元自身前 10 条记录校准；标注为上界）

主指标：C1 单元级覆盖率（率 ≤ α 的实例比例）+ Wilson 95% CI；C2 平均绝对偏差；C3 分层；C4 饱和计数；
次要指标：跨实例 SD（与 EXP-V3-01 对照）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import OUTPUT_DIR
from exp_v1_08_baselines import QUANTILE
from exp_v3_01_unit_calibration import make_scorer
from exp_v3_01d_paderborn_fixed import ALPHA, build_instances, stable_seed
from paderborn_data import CONDITIONS, HEALTHY_BEARINGS

LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
CALIB_RECORDS_FOR_SELF = 10          # M5：目标单元前 10 条记录用于自校准
LOCAL_K = 5                          # M3：邻域大小
DETECTORS = ("rms", "mahalanobis", "iforest")


class Tee:
    def __init__(self, path: Path):
        self.handle = open(path, "w", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        self.handle.write(text)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.handle.flush()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(centre - half), float(centre + half)


def quantile_threshold(scores: np.ndarray) -> float:
    return float(np.quantile(scores, QUANTILE))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUTPUT_DIR / "exp_v3_03_run.log")
    print("构建实例（W-B 窗口，620 窗/实例）…", flush=True)
    instances = build_instances()
    keys = sorted(instances)

    rows: list[dict] = []
    for test_bearing in HEALTHY_BEARINGS:
        train_keys = [k for k in keys if k[0] != test_bearing]
        test_keys = [k for k in keys if k[0] == test_bearing]
        assert all(k[0] != test_bearing for k in train_keys), "P1 训练集含测试轴承"
        train_units = [instances[k] for k in sorted(train_keys)]
        pooled = np.vstack(train_units)
        for kind in DETECTORS:
            # 训练分数（一次拟合，供 M0/M1/M3/M4/M2 复用）
            scorer = make_scorer(kind, pooled)
            train_scores = scorer(pooled)
            t_global = quantile_threshold(train_scores)
            scores_per_unit = [scorer(u) for u in train_units]
            cond_of = {k: k[1] for k in sorted(train_keys)}

            # M1：按工况分组的 Mondrian 阈值
            t_group = {}
            for cond in CONDITIONS:
                idx = [i for i, k in enumerate(sorted(train_keys)) if cond_of[k] == cond]
                if idx:
                    t_group[cond] = quantile_threshold(np.concatenate([scores_per_unit[i] for i in idx]))

            # M3：局部邻域（按特征均值距离）
            means = np.array([u.mean(axis=0) for u in train_units])
            std = means.std(axis=0)
            std = np.where(std == 0, 1.0, std)

            # M4：λ 由训练内 LOO 选（在训练实例上评估，不含测试单元）
            lam_best, lam_dev = 0.0, np.inf
            for lam in LAMBDA_GRID:
                devs = []
                for i, k in enumerate(sorted(train_keys)):
                    others = [j for j in range(len(train_units)) if j != i]
                    others_pool = np.vstack([train_units[j] for j in others])
                    sc = make_scorer(kind, others_pool)
                    s_all = sc(others_pool)
                    idx_g = [j for j in others if cond_of[sorted(train_keys)[j]] == k[1]]
                    if not idx_g:
                        continue
                    s_g = np.concatenate([sc(train_units[j]) for j in idx_g])
                    thr = quantile_threshold(s_all) + lam * (quantile_threshold(s_g)
                                                             - quantile_threshold(s_all))
                    devs.append(abs(float((sc(train_units[i]) > thr).mean()) - ALPHA))
                if devs:
                    dev = float(np.mean(devs))
                    if dev < lam_dev - 1e-12:
                        lam_best, lam_dev = lam, dev

            for test_key in sorted(test_keys):
                test_block = instances[test_key]
                cond = test_key[1]
                n_records = max(1, len(test_block) // 20)          # 每记录约 31 窗
                methods = {
                    "M0_global": t_global,
                    "M1_mondrian": t_group.get(cond, t_global),
                    "M3_localized": quantile_threshold(
                        np.concatenate([scores_per_unit[i]
                                        for i in np.argsort(np.linalg.norm(
                                            (means - test_block.mean(axis=0)) / std, axis=1))[:LOCAL_K]])),
                    "M4_hierarchical": t_global + lam_best * (t_group.get(cond, t_global) - t_global),
                }
                # M2：weighted conformal（权重由训练 vs 目标单元特征的 logistic 估计）
                try:
                    X = np.vstack([pooled[:: max(1, len(pooled) // 4000)],
                                   test_block[:: max(1, len(test_block) // 4000)]])
                    y = np.r_[np.zeros(min(4000, len(pooled))),
                              np.ones(len(X) - min(4000, len(pooled)))]
                    clf = LogisticRegression(max_iter=200).fit(X, y)
                    w = clf.predict_proba(pooled)[:, 1] / np.maximum(
                        1 - clf.predict_proba(pooled)[:, 1], 1e-9)
                    order = np.argsort(train_scores)
                    cw = np.cumsum(w[order]) / max(w.sum(), 1e-12)
                    methods["M2_weighted"] = float(train_scores[order][np.searchsorted(cw, QUANTILE)])
                except Exception as exc:  # noqa: BLE001
                    print(f"    M2 失败（{test_key} {kind}）: {exc}", flush=True)
                    methods["M2_weighted"] = np.nan
                # M5：自校准（前 10 条记录）
                cut = min(CALIB_RECORDS_FOR_SELF * n_records, len(test_block) // 2)
                calib, heldout = test_block[:cut], test_block[cut:]
                if len(calib) > 10 and len(heldout) > 10:
                    sc_self = make_scorer(kind, calib)
                    methods["M5_selfcalibrated"] = quantile_threshold(sc_self(calib))
                    scored = (sc_self(heldout) > methods["M5_selfcalibrated"])
                else:
                    scored = None

                for name, thr in methods.items():
                    if not np.isfinite(thr):
                        continue
                    if name == "M5_selfcalibrated":
                        rate = float(scored.mean()) if scored is not None else np.nan
                    else:
                        rate = float((scorer(test_block) > thr).mean())
                    rows.append(dict(test_instance=f"{test_key[0]}|{test_key[1]}", detector=kind,
                                     method=name, threshold=thr, rate=rate,
                                     n_test_windows=len(test_block), n_train_units=len(train_units)))
        print(f"  [{test_bearing}] done", flush=True)
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "exp_v3_03_partial.csv", index=False)

    raw = pd.DataFrame(rows)
    raw["saturated"] = raw.rate.isin([0.0, 1.0])
    raw.to_csv(OUTPUT_DIR / "exp_v3_03_runs.csv", index=False)

    out = []
    for (det, method), g in raw.groupby(["detector", "method"]):
        covered = int((g.rate <= ALPHA).sum())
        lo, hi = wilson(covered, len(g))
        out.append(dict(detector=det, method=method, n=len(g),
                        coverage=covered / len(g), wilson_lo=lo, wilson_hi=hi,
                        mean_dev_pp=float(np.mean(np.abs(g.rate - ALPHA)) * 100),
                        sd_pp=float(np.std(g.rate.to_numpy(), ddof=1) * 100),
                        saturated=int(g.saturated.sum())))
    summary = pd.DataFrame(out).sort_values(["detector", "method"])
    summary.to_csv(OUTPUT_DIR / "exp_v3_03_coverage.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== 覆盖率（C1）与偏差（C2） ===")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
