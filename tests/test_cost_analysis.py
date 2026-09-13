"""成本敏感阈值选择的单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cost_analysis import cost_curve, select_threshold, sweep_cost_ratios


def _scores_and_labels():
    # 正常分数集中在 0 附近，异常分数集中在 10 附近
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 1.0, 40)
    fault = rng.normal(10.0, 1.0, 40)
    scores = np.concatenate([normal, fault])
    labels = np.concatenate([np.zeros(40, bool), np.ones(40, bool)])
    return scores, labels


def test_curve_counts():
    scores, labels = _scores_and_labels()
    thresholds = np.array([2.0, 20.0])
    curve = cost_curve(scores, labels, thresholds, fn_over_fp=10.0)
    row_low, row_high = curve.iloc[0], curve.iloc[1]
    # 低阈值：漏报为 0，可能有少量误报
    assert row_low["fn"] == 0
    # 高阈值：误报为 0，漏报为 40
    assert row_high["fp"] == 0 and row_high["fn"] == 40
    assert row_high["cost"] == 400.0


def test_select_threshold_finds_separator():
    scores, labels = _scores_and_labels()
    best = select_threshold(scores, labels, np.linspace(-3, 15, 100), fn_over_fp=10.0)
    assert 0 < best["threshold"] < 10
    assert best["fn"] == 0
    assert best["cost"] == best["fp"]


def test_high_cost_ratio_favors_recall():
    """漏报代价越高，最优阈值越低（更倾向报警）。"""
    scores, labels = _scores_and_labels()
    thresholds = np.linspace(-3, 15, 100)
    low = select_threshold(scores, labels, thresholds, fn_over_fp=1.0)
    high = select_threshold(scores, labels, thresholds, fn_over_fp=50.0)
    assert high["threshold"] <= low["threshold"]


def test_sweep_ratios_output():
    scores, labels = _scores_and_labels()
    table = sweep_cost_ratios(scores, labels, np.linspace(-3, 15, 50))
    assert list(table.columns) == ["fn_over_fp", "threshold", "fp", "fn", "cost"]
    assert len(table) == 5
    assert isinstance(table, pd.DataFrame)
