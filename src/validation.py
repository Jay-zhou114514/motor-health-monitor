"""泄漏安全的阈值选择：leave-one-file-out 交叉验证。

为什么需要：
直接在测试集上扫描所有阈值、再挑代价最低的那条，属于"事后最优"，
报告出来的结果会偏乐观。正确做法是：选阈值时不能看到被评估文件自己的标签。

这里采用 leave-one-file-out：每次留出一个文件，用其余文件选阈值，
再在被留出的文件上评估。所有被留出文件的预测拼起来，就是一份
没有数据泄漏的评估结果。
"""

from __future__ import annotations

import numpy as np

from cost_analysis import select_threshold
from evaluate import evaluate


def leave_one_file_out(
    scores: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    thresholds: np.ndarray,
    fn_over_fp: float = 10.0,
) -> dict:
    """对每个文件用其余文件选阈值，再在该文件上评估。

    参数
    ----
    scores：每个窗口的报警分数（越大越像异常）。
    y_true：每个窗口的真实标签（True = 故障）。
    groups：每个窗口所属的文件名（用于分组）。
    thresholds：候选阈值网格。
    fn_over_fp：一次漏报相当于多少次误报的代价（需事先固定，不能用测试集调）。

    返回
    ----
    dict：包含聚合后的混淆计数、指标，以及每个文件被选中的阈值。
    """
    scores = np.asarray(scores, dtype=float)
    true = np.asarray(y_true, dtype=bool)
    groups = np.asarray(groups)
    thresholds = np.asarray(thresholds, dtype=float)

    predictions = np.zeros(true.shape, dtype=bool)
    chosen: list[dict] = []

    for group in np.unique(groups):
        held_out = groups == group
        rest = ~held_out
        rest_labels = true[rest]

        # 若其余文件只有单一类别，无法做代价权衡，退回到阈值网格的中位数
        if rest_labels.size == 0 or rest_labels.all() or (~rest_labels).all():
            chosen_threshold = float(np.median(thresholds))
        else:
            best = select_threshold(
                scores[rest], rest_labels, thresholds, fn_over_fp=fn_over_fp
            )
            chosen_threshold = float(best["threshold"])

        predictions[held_out] = scores[held_out] > chosen_threshold
        chosen.append(
            {
                "file": str(group),
                "windows": int(held_out.sum()),
                "threshold": chosen_threshold,
            }
        )

    return {
        "predictions": predictions,
        "chosen_thresholds": chosen,
        "metrics": evaluate(true, predictions),
        "fn_over_fp": fn_over_fp,
    }


def markdown_summary(name: str, result: dict) -> list[str]:
    """把 LOFO 结果整理成 Markdown 行。"""
    metrics = result["metrics"]
    lines = [
        f"**{name}（leave-one-file-out，漏报:误报 = {result['fn_over_fp']:g}）**",
        "",
        f"- 精确率 = {metrics['precision']:.3f}，召回率 = {metrics['recall']:.3f}，"
        f"F1 = {metrics['f1']:.3f}，FPR = {metrics['fpr']:.3f}，FNR = {metrics['fnr']:.3f}",
        f"- 混淆计数：TP={metrics['tp']:.0f}，FP={metrics['fp']:.0f}，"
        f"FN={metrics['fn']:.0f}，TN={metrics['tn']:.0f}",
        "",
        "| 文件 | 窗口数 | 该文件被选中的阈值 |",
        "| --- | ---: | ---: |",
    ]
    for row in result["chosen_thresholds"]:
        lines.append(f"| {row['file']} | {row['windows']} | {row['threshold']:.3f} |")
    return lines
