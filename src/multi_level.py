"""多层评价：把窗口级预测聚合成文件级（记录级）结论。

为什么需要：
窗口之间有 50% 重叠，相邻窗口高度相关，所以 Window-level 指标会高估
真实的检测能力。把同一个文件的窗口结果聚合起来，能回答更实际的问题：
“这段记录是不是被稳定地判成故障？”
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def file_level_table(
    table: pd.DataFrame, prediction: pd.Series, method: str
) -> pd.DataFrame:
    """按文件聚合窗口预测。

    输出每个文件一行，包含：条件、窗口数、被报警窗口数、检测率、
    误报率、首次报警时间、最长连续报警窗口数。
    """
    data = table.copy()
    data["prediction"] = prediction.reindex(data.index).fillna(False).astype(bool)

    rows: list[dict] = []
    for record, group in data.groupby("record", sort=True):
        total = int(len(group))
        fault_windows = int((group["label"] != "normal").sum())
        normal_windows = total - fault_windows
        detected = int(group["prediction"].sum())
        alarm_ratio = detected / total if total else 0.0

        detection_rate = alarm_ratio if fault_windows > 0 else float("nan")
        false_alarm_rate = alarm_ratio if normal_windows > 0 else float("nan")

        alarm_positions = group.index[group["prediction"]]
        first_alarm_s = (
            float(group.loc[alarm_positions[0], "window_start_s"])
            if len(alarm_positions)
            else float("nan")
        )

        longest_run = 0
        current_run = 0
        for flagged in group["prediction"].to_numpy():
            current_run = current_run + 1 if flagged else 0
            longest_run = max(longest_run, current_run)

        rows.append(
            {
                "method": method,
                "file": record,
                "condition": group["label"].iloc[0],
                "total_windows": total,
                "fault_windows": fault_windows,
                "normal_windows": normal_windows,
                "detected_windows": detected,
                "detection_rate": detection_rate,
                "false_alarm_rate": false_alarm_rate,
                "first_alarm_s": first_alarm_s,
                "longest_alarm_run": longest_run,
            }
        )
    return pd.DataFrame(rows)


def file_level_summary(
    table: pd.DataFrame,
    prediction: pd.Series,
    method: str,
    file_flag_ratio: float = 0.5,
) -> dict:
    """文件级汇总指标。

    file_flag_ratio：一个故障文件被判定为"检出"，至少需要多大比例的
    窗口报警。默认 50%，属于可讨论的实验选择（不是唯一标准）。
    """
    detail = file_level_table(table, prediction, method)
    fault_files = detail[detail["fault_windows"] > 0]
    normal_files = detail[detail["normal_windows"] > 0]

    detected_files = int((fault_files["detection_rate"] >= file_flag_ratio).sum())
    flagged_normal_files = int((normal_files["false_alarm_rate"] > 0).sum())

    return {
        "method": method,
        "file_flag_ratio": file_flag_ratio,
        "fault_files": int(len(fault_files)),
        "fault_files_detected": detected_files,
        "file_detection_rate": (
            detected_files / len(fault_files) if len(fault_files) else float("nan")
        ),
        "normal_files": int(len(normal_files)),
        "normal_files_flagged": flagged_normal_files,
        "file_false_alarm_rate": (
            flagged_normal_files / len(normal_files)
            if len(normal_files)
            else float("nan")
        ),
        "mean_window_detection_rate": (
            float(fault_files["detection_rate"].mean())
            if len(fault_files)
            else float("nan")
        ),
        "mean_window_false_alarm_rate": (
            float(normal_files["false_alarm_rate"].mean())
            if len(normal_files)
            else float("nan")
        ),
    }


def file_level_markdown(detail: pd.DataFrame) -> list[str]:
    """把文件级结果转成 Markdown 表格行。"""
    lines = [
        "| File | Condition | Total windows | Fault windows | Detected | Detection rate | False alarm rate | First alarm (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in detail.iterrows():
        detection = "-" if pd.isna(row["detection_rate"]) else f"{row['detection_rate']:.3f}"
        false_alarm = "-" if pd.isna(row["false_alarm_rate"]) else f"{row['false_alarm_rate']:.3f}"
        first_alarm = "-" if np.isnan(row["first_alarm_s"]) else f"{row['first_alarm_s']:.2f}"
        lines.append(
            f"| {row['file']} | {row['condition']} | {row['total_windows']} | "
            f"{row['fault_windows']} | {row['detected_windows']} | {detection} | "
            f"{false_alarm} | {first_alarm} |"
        )
    return lines
