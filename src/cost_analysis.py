"""成本敏感的报警线选择。

F1 只是把误报和漏报看得同样重，但真实工厂里两者代价不同：
误报（FP）浪费一次停机检查，漏报（FN）可能导致设备损坏与非计划停机。
本模块在给定"漏报:误报 代价倍率"下，扫描报警线并选出总代价最低的一条。

注意：如果在测试集上扫描再报告最优值，得到的是"事后最优"结果，
只能用来理解代价结构；真实系统应基于历史数据或领域知识预先固定代价倍率。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cost_curve(
    scores: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
    fn_over_fp: float = 10.0,
) -> pd.DataFrame:
    """扫描报警线，计算每条报警线下的误报/漏报与总代价。

    scores 越大越像异常；判定规则为 score > threshold。
    fn_over_fp 是一次漏报相当于多少次误报的代价。
    """
    true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    rows = []
    for threshold in thresholds:
        pred = scores > threshold
        fp = int(np.sum(~true & pred))
        fn = int(np.sum(true & ~pred))
        rows.append(
            {
                "threshold": float(threshold),
                "fp": fp,
                "fn": fn,
                "cost": fp + fn_over_fp * fn,
            }
        )
    return pd.DataFrame(rows)


def select_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
    fn_over_fp: float = 10.0,
) -> dict[str, float]:
    """返回总代价最低的报警线及其混淆计数。"""
    curve = cost_curve(scores, y_true, thresholds, fn_over_fp)
    best = curve.loc[curve["cost"].idxmin()]
    return {
        "threshold": float(best["threshold"]),
        "fp": int(best["fp"]),
        "fn": int(best["fn"]),
        "cost": float(best["cost"]),
    }


def sweep_cost_ratios(
    scores: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
    ratios: list[float] | None = None,
) -> pd.DataFrame:
    """对多个代价倍率分别选最优报警线，观察报警线如何随成本偏好移动。"""
    if ratios is None:
        ratios = [1, 5, 10, 20, 50]
    rows = []
    for ratio in ratios:
        best = select_threshold(scores, y_true, thresholds, ratio)
        rows.append({"fn_over_fp": ratio, **best})
    return pd.DataFrame(rows)
