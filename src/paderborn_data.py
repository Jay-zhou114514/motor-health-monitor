"""Paderborn 健康轴承数据装载与特征提取（EXP-V2-01 用）。

已核实的事实（2026-09-17，见 EXP-V2-01-preregistration-amendment-2.md）：
- `.mat` 顶层：`Info / X / Y / Description`；
  **`X` 是时间轴（1×3）**，**`Y` 是测量通道（1×7）**；
- 振动信号 = **`Y['vibration_1']`**，`HostService` 栅格，**256,001 点 / 4 s → 64 kHz**；
- 力 / 转速 / 扭矩 = `Mech_4kHz`（16,001 点）；相电流 = `HostService`；
- 每轴承每工况 20 条记录；健康轴承 K001–K006；共 4 个工况。

窗口约定：见 `EXP-V2-01-preregistration-amendment-2.md` §4（W-A / W-B 两候选）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from config import DATASET_ROOT
from features import extract_features

PADERBORN_ROOT = DATASET_ROOT / "paderborn" / "extracted"
HEALTHY_BEARINGS = ("K001", "K002", "K003", "K004", "K005", "K006")
CONDITIONS = ("N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04")
DEFAULT_CONDITION = "N15_M07_F10"

PADERBORN_SR = 64_000.0
MAX_SAMPLES = 256_000
VIBRATION_CHANNEL = "vibration_1"
RECORD_SECONDS = 4.0

FEATURES = ["rms", "crest_factor", "kurtosis", "centroid_hz"]


def recording_paths(bearing: str, condition: str = DEFAULT_CONDITION) -> list[Path]:
    return sorted((PADERBORN_ROOT / bearing).glob(f"{condition}_{bearing}_*.mat"))


def load_signal(
    path: Path, channel: str = VIBRATION_CHANNEL, max_samples: int | None = MAX_SAMPLES
) -> np.ndarray:
    """读取振动通道（默认 `vibration_1`），并按 MAX_SAMPLES 截断（见 EXP-V2-02 修订 1）。"""
    content = sio.loadmat(path, simplify_cells=False)
    key = [k for k in content if not k.startswith("__")][0]
    record = content[key][0, 0]
    channels = record["Y"]
    for index in range(channels.shape[1]):
        if str(channels[0, index]["Name"][0]).strip().lower() == channel:
            signal = np.ravel(channels[0, index]["Data"]).astype(float)
    if max_samples is not None and signal.size > max_samples:
        signal = signal[:max_samples]
    return signal
    available = [str(channels[0, i]["Name"][0]) for i in range(channels.shape[1])]
    raise KeyError(f"通道 {channel} 不存在；可用通道：{available}")


def record_feature_table(
    path: Path, condition: str, window_sec: float, step_sec: float, aggregate: bool
) -> pd.DataFrame:
    signal = load_signal(path)
    if window_sec is None:  # 单窗口模式：整条（截断后）记录 = 1 个窗口
        window_sec = signal.size / PADERBORN_SR
        step_sec = window_sec
    table = extract_features(
        signal,
        PADERBORN_SR,
        window_sec=window_sec,
        step_sec=step_sec,
        record_id=path.name,
        label="normal",
    ).dropna(subset=FEATURES)
    table["bearing"] = path.parent.name
    table["condition"] = condition
    table["record_number"] = int(path.stem.rsplit("_", 1)[-1])
    table["record_id"] = path.name
    if aggregate and len(table):
        row = table[FEATURES].mean().to_frame().T
        row["bearing"] = path.parent.name
        row["condition"] = condition
        row["record_number"] = int(path.stem.rsplit("_", 1)[-1])
        row["record_id"] = path.name
        row["n_windows"] = len(table)
        table = row
    return table


def load_healthy_dataset(
    condition: str = DEFAULT_CONDITION,
    bearings: tuple[str, ...] = HEALTHY_BEARINGS,
    window_sec: float | None = None,
    step_sec: float | None = None,
    aggregate: bool = False,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """返回 ({记录名: 特征数组}, 明细表)。

    window_sec/step_sec/aggregate 决定窗口约定（W-A：4 s / 4 s / False；W-B：0.25 s / 0.125 s / True）。
    """
    arrays: dict[str, np.ndarray] = {}
    tables: list[pd.DataFrame] = []
    for bearing in bearings:
        for path in recording_paths(bearing, condition):
            table = record_feature_table(path, condition, window_sec, step_sec, aggregate)
            if len(table):
                arrays[path.name] = table[FEATURES].to_numpy(float)
                tables.append(table)
    return arrays, pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()