"""评估指标的单元测试。"""

from __future__ import annotations

import numpy as np

from evaluate import evaluate


def test_perfect_prediction():
    y = np.array([True] * 5 + [False] * 5)
    metrics = evaluate(y, y)
    assert metrics["tp"] == 5 and metrics["tn"] == 5
    assert metrics["fp"] == 0 and metrics["fn"] == 0
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_counts_and_metrics():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 0])
    metrics = evaluate(y_true, y_pred)
    assert (metrics["tp"], metrics["fn"], metrics["fp"], metrics["tn"]) == (2, 1, 1, 2)
    assert np.isclose(metrics["precision"], 2 / 3)
    assert np.isclose(metrics["recall"], 2 / 3)
    assert np.isclose(metrics["accuracy"], 4 / 6)


def test_all_wrong_still_defined():
    y_true = np.array([1, 1])
    y_pred = np.array([0, 0])
    metrics = evaluate(y_true, y_pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
