"""Run-to-failure 数据集装载（XJTU-SY 与 PRONOSTIA），供 EXP-V2-04 使用。

设计依据：experiments/EXP-V2-04-preregistration.md + 修订 1（双数据集）。

要点：
- 所有健康阶段规则下 H ≤ 60（上限 60）→ **只读每颗轴承前 60 条记录**，不必读全量；
- 1 条记录 = 1 个特征向量（整条记录上计算冻结的 4 个特征）；
- XJTU-SY：CSV **有表头**，列序 = 水平, 垂直；
- PRONOSTIA：CSV **无表头、6 列**（时,分,秒,微秒, 水平, 垂直）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import DATASET_ROOT
from features import extract_features

FEATURES = ["rms", "crest_factor", "kurtosis", "centroid_hz"]
MAX_HEALTHY = 60          # H 的上限（预注册）
SR = 25_600.0

XJTU_ROOT = DATASET_ROOT / "xjtu_sy" / "XJTU-SY_Bearing_Datasets"
XJTU_CONDITIONS = ("35Hz12kN", "37.5Hz11kN", "40Hz10kN")

PRONOSTIA_ROOT = (
    DATASET_ROOT / "pronostia" / "extracted"
    / "phm-ieee-2012-data-challenge-dataset-master"
)
PRONOSTIA_DIRS = ("Learning_set", "Full_Test_Set")   # Test_set 为截断版，不纳入
PRONOSTIA_SEPARATOR_COUNTS: dict[str, int] = {}


def healthy_size(n_records: int, fraction: float = 0.10, floor: int = 20,
                 cap: int = MAX_HEALTHY) -> int:
    """预注册冻结的健康阶段规则：H = clip(max(floor, fraction*N), floor, cap)。"""
    return int(min(max(floor, fraction * n_records), cap))


def included(n_records: int, h: int) -> bool:
    """纳入门槛：H / N <= 0.25。"""
    return h / n_records <= 0.25


def _features_from_signal(signal: np.ndarray, record_sec: float, name: str) -> dict | None:
    table = extract_features(signal, SR, window_sec=record_sec, step_sec=record_sec,
                             record_id=name, label="healthy").dropna(subset=FEATURES)
    if not len(table):
        return None
    row = {f: float(table[f].iloc[0]) for f in FEATURES}
    row["n_windows"] = int(len(table))
    return row


def _xjtu_paths(condition: str, bearing: str) -> list[Path]:
    folder = XJTU_ROOT / condition / bearing
    return sorted(folder.glob("*.csv"), key=lambda p: int(p.stem))


def _pronostia_paths(folder: str, bearing: str) -> list[Path]:
    return sorted((PRONOSTIA_ROOT / folder / bearing).glob("acc_*.csv"))


def read_pronostia_csv(path: Path) -> pd.DataFrame:
    """读取 PRONOSTIA 的 6 列加速度文件。

    镜像中存在**分隔符不一致**：绝大多数文件用逗号，
    但 `Full_Test_Set/Bearing1_4/` 使用分号（实测：`8;8;0;4.2504e+05;0.065;-0.058`）。
    因此按首行自动识别分隔符，并统计使用情况（供实验记录披露）。
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        first = handle.readline()
    separator = ";" if (";" in first and "," not in first) else ","
    PRONOSTIA_SEPARATOR_COUNTS[separator] = PRONOSTIA_SEPARATOR_COUNTS.get(separator, 0) + 1
    return pd.read_csv(path, header=None, sep=separator)


def load_feature_cache(max_records: int = MAX_HEALTHY) -> pd.DataFrame:
    """读取每颗轴承的前 max_records 条记录并计算特征；返回长表。"""
    rows: list[dict] = []

    for condition in XJTU_CONDITIONS:
        base = XJTU_ROOT / condition
        if not base.exists():
            continue
        for bearing_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            paths = _xjtu_paths(condition, bearing_dir.name)
            n_total = len(paths)
            for index, path in enumerate(paths[:max_records], start=1):
                frame = pd.read_csv(path)                     # 有表头
                signal = frame.iloc[:, 0].to_numpy(float)     # 水平通道
                feat = _features_from_signal(signal, signal.size / SR, path.name)
                if feat is None:
                    continue
                rows.append({
                    "dataset": "XJTU-SY", "condition": condition,
                    "bearing": bearing_dir.name, "record_index": index,
                    "n_records_total": n_total, "horizontal": True, **feat,
                })
            print(f"  XJTU-SY {condition}/{bearing_dir.name}: N={n_total}", flush=True)

    for folder in PRONOSTIA_DIRS:
        base = PRONOSTIA_ROOT / folder
        if not base.exists():
            continue
        for bearing_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            paths = _pronostia_paths(folder, bearing_dir.name)
            n_total = len(paths)
            for index, path in enumerate(paths[:max_records], start=1):
                frame = read_pronostia_csv(path)              # 无表头，6 列，分隔符自动识别
                signal = frame.iloc[:, 4].to_numpy(float)     # 水平通道
                feat = _features_from_signal(signal, signal.size / SR, path.name)
                if feat is None:
                    continue
                rows.append({
                    "dataset": "PRONOSTIA", "condition": folder,
                    "bearing": bearing_dir.name, "record_index": index,
                    "n_records_total": n_total, "horizontal": True, **feat,
                })
            print(f"  PRONOSTIA {folder}/{bearing_dir.name}: N={n_total}", flush=True)

    return pd.DataFrame(rows)