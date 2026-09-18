"""构建并检查 run-to-failure 特征缓存（EXP-V2-04 的前置健全性检查）。

产出：
- outputs/r2f_features.csv（每颗轴承前 60 条记录的特征）
- 控制台打印：各数据集特征分布、每颗轴承的纳入判定
"""
from __future__ import annotations

import sys

import pandas as pd

from config import OUTPUT_DIR
from runtofailure_data import load_feature_cache, healthy_size, included, FEATURES


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_feature_cache()
    frame.to_csv(OUTPUT_DIR / "r2f_features.csv", index=False)

    print("\n=== 特征健全性检查（必须非退化）===")
    for dataset, sub in frame.groupby("dataset"):
        print(f"\n[{dataset}] 轴承 {sub.bearing.nunique()} 颗，记录 {len(sub)} 条")
        stats = sub[FEATURES].describe().loc[["mean", "std", "min", "max"]].round(4)
        print(stats.to_string())
        ratio = (sub[FEATURES].std() / sub[FEATURES].mean().abs()).round(4)
        print(f"  变异系数(CV) = {ratio.to_dict()}")
        degenerate = [f for f in FEATURES if ratio[f] < 0.01]
        print(f"  退化特征(<1% CV): {degenerate if degenerate else '无'}")

    print("\n=== 健康阶段规则与纳入判定（H = clip(max(20,10%N),20,60)，要求 H/N≤25%）===")
    summary = (frame.groupby(["dataset", "condition", "bearing"])["n_records_total"]
               .first().reset_index())
    rows = []
    for _, r in summary.iterrows():
        h = healthy_size(int(r["n_records_total"]))
        rows.append({"dataset": r["dataset"], "bearing": r["bearing"],
                     "N": int(r["n_records_total"]), "H": h,
                     "H/N%": round(h / r["n_records_total"] * 100, 1),
                     "纳入主分析": included(int(r["n_records_total"]), h)})
    detail = pd.DataFrame(rows)
    detail.to_csv(OUTPUT_DIR / "r2f_inclusion.csv", index=False)
    for dataset, sub in detail.groupby("dataset"):
        ok = sub[sub["纳入主分析"]]
        bad = sub[~sub["纳入主分析"]]
        print(f"\n[{dataset}] 纳入 {len(ok)}/{len(sub)}；H 范围 {ok.H.min()}–{ok.H.max()}"
              f"；健康记录合计 {int(ok.H.sum())}")
        if len(bad):
            print("  未纳入（进敏感性分析）：")
            for _, r in bad.iterrows():
                print(f"    {r.bearing}: N={r.N}, H={r.H} ({r['H/N%']}%)")


if __name__ == "__main__":
    main()