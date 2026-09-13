"""leave-one-file-out 泄漏安全阈值选择的单元测试。"""

from __future__ import annotations

import numpy as np

from validation import leave_one_file_out


def _toy_scores():
    # 3 个文件：一个正常文件（分数低）、两个故障文件（分数高）
    scores = np.array([0.1, 0.2, 0.15, 9.0, 9.5, 8.8, 10.0, 9.2, 9.9])
    labels = np.array([False, False, False, True, True, True, True, True, True])
    groups = np.array(
        ["normal_a"] * 3 + ["fault_a"] * 3 + ["fault_b"] * 3
    )
    return scores, labels, groups


def _threshold_for(result, file_name):
    return next(
        row["threshold"]
        for row in result["chosen_thresholds"]
        if row["file"] == file_name
    )


def test_lofo_produces_predictions_for_all_windows():
    scores, labels, groups = _toy_scores()
    result = leave_one_file_out(scores, labels, groups, np.linspace(0.0, 12.0, 25))
    assert result["predictions"].shape == labels.shape
    # 分数分得很开，留一法应该仍能正确分开
    assert result["metrics"]["f1"] == 1.0
    assert len(result["chosen_thresholds"]) == 3


def test_lofo_threshold_never_uses_held_out_labels():
    """被留出文件自己的标签不能影响它自己被选中的阈值。"""
    scores, labels, groups = _toy_scores()
    changed = labels.copy()
    changed[groups == "fault_b"] = False  # 只改 fault_b 的标签

    result_a = leave_one_file_out(scores, labels, groups, np.linspace(0.0, 12.0, 25))
    result_b = leave_one_file_out(scores, changed, groups, np.linspace(0.0, 12.0, 25))

    # fault_b 被留出时，选阈值只用 normal_a 与 fault_a，因此它的阈值不应改变
    assert _threshold_for(result_a, "fault_b") == _threshold_for(result_b, "fault_b")
    # 但 fault_b 自己的预测结果应当因为标签改变而改变（这里只检查阈值不变性）
    assert result_a["metrics"] != result_b["metrics"]
