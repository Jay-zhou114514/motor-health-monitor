"""二分类评估：混淆矩阵、准确率、精确率、召回率、F1。"""

from __future__ import annotations

import numpy as np


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """把布尔标签按 异常=1、正常=0 计算指标。"""
    true = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    tp = float(np.sum(true & pred))
    fp = float(np.sum(~true & pred))
    fn = float(np.sum(true & ~pred))
    tn = float(np.sum(~true & ~pred))

    accuracy = (tp + tn) / max(1, true.size)
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
