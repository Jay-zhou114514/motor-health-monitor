"""读取 MathWorks 滚动轴承故障数据集。

数据来源：https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data
许可：Creative Commons Attribution-NonCommercial-ShareAlike 4.0
"""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np

from config import RAW_DATA_DIR


DATASET_BASE_URL = (
    "https://raw.githubusercontent.com/mathworks/"
    "RollingElementBearingFaultDiagnosis-Data/master"
)

# 本次实验只使用其中一部分文件：
# 训练（正常基线）2 个文件，测试 1 个正常 + 4 个故障文件。
MINIMAL_RELATIVE_FILES = [
    "train_data/baseline_1.mat",
    "train_data/baseline_2.mat",
    "test_data/baseline_3.mat",
    "test_data/OuterRaceFault_3.mat",
    "test_data/OuterRaceFault_vload_6.mat",
    "test_data/OuterRaceFault_vload_7.mat",
    "test_data/InnerRaceFault_vload_6.mat",
    "test_data/InnerRaceFault_vload_7.mat",
]

ALL_RELATIVE_FILES = [
    "train_data/baseline_1.mat",
    "train_data/baseline_2.mat",
    "test_data/baseline_3.mat",
    "train_data/OuterRaceFault_1.mat",
    "train_data/OuterRaceFault_2.mat",
    "train_data/OuterRaceFault_vload_1.mat",
    "train_data/OuterRaceFault_vload_2.mat",
    "train_data/OuterRaceFault_vload_3.mat",
    "train_data/OuterRaceFault_vload_4.mat",
    "train_data/OuterRaceFault_vload_5.mat",
    "test_data/OuterRaceFault_3.mat",
    "test_data/OuterRaceFault_vload_6.mat",
    "test_data/OuterRaceFault_vload_7.mat",
    "train_data/InnerRaceFault_vload_1.mat",
    "train_data/InnerRaceFault_vload_2.mat",
    "train_data/InnerRaceFault_vload_3.mat",
    "train_data/InnerRaceFault_vload_4.mat",
    "train_data/InnerRaceFault_vload_5.mat",
    "test_data/InnerRaceFault_vload_6.mat",
    "test_data/InnerRaceFault_vload_7.mat",
]


def condition_from_name(name: str) -> str:
    """根据文件名判断健康状态。"""
    lowered = name.lower()
    if "baseline" in lowered:
        return "normal"
    if "outerracefault" in lowered:
        return "outer_race_fault"
    if "innerracefault" in lowered:
        return "inner_race_fault"
    return "unknown"


def _url(relative_file: str) -> str:
    return f"{DATASET_BASE_URL}/{relative_file}"


def _download_file(url: str, dest: Path) -> bool:
    """下载单个文件；已存在且非空则跳过。返回是否真的下载了。"""
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temp.open(
                "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file)
            temp.replace(dest)
            return True
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
    raise RuntimeError(f"下载失败（已重试 3 次）: {url}\n原因: {last_error}")


def ensure_dataset_files(
    relative_files: Iterable[str] | None = None, force: bool = False
) -> list[Path]:
    """下载所需数据文件到 data/raw/ 目录。"""
    if relative_files is None:
        relative_files = MINIMAL_RELATIVE_FILES
    paths: list[Path] = []
    for relative in relative_files:
        dest = RAW_DATA_DIR / relative
        if force:
            dest.unlink(missing_ok=True)
        _download_file(_url(relative), dest)
        paths.append(dest)
    return paths


def _load_mat_variables(path: Path) -> dict[str, object]:
    """读取 .mat 文件并整理成普通字典。"""
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "需要 scipy 才能读取 .mat 文件。请先运行："
            "python -m pip install scipy"
        ) from exc

    raw = loadmat(str(path), squeeze_me=False, struct_as_record=False)
    variables: dict[str, object] = {}
    sources: dict[str, object] = dict(raw)

    # 数据文件把信号与实验参数放在名为 bearing 的结构体里，
    # 故障特征频率（BPFO 等）则直接放在文件顶层，这里合并处理。
    bearing = raw.get("bearing")
    if bearing is not None:
        struct = np.asarray(bearing).reshape(-1)
        if struct.size and hasattr(struct[0], "_fieldnames"):
            for key in struct[0]._fieldnames:
                sources[key] = getattr(struct[0], key)

    for key in ("gs", "sr", "rate", "load", "BPFO", "BPFI", "FTF", "BSF"):
        if key not in sources:
            continue
        array = np.asarray(sources[key]).squeeze()
        if array.dtype.kind in "fc":
            variables[key] = float(array) if array.ndim == 0 else array.flatten().astype(float)
        elif array.dtype.kind in "iu":
            variables[key] = float(array) if array.ndim == 0 else array.flatten().astype(float)
    return variables


def load_records(
    relative_files: Iterable[str] | None = None,
) -> list[dict]:
    """把 .mat 文件读成记录列表。

    每条记录包含：文件信息、健康状态标签、采样率 sr、转速 rate、
    载荷 load、故障特征频率，以及一维振动信号 gs。
    """
    if relative_files is None:
        relative_files = MINIMAL_RELATIVE_FILES
    records: list[dict] = []
    for relative in relative_files:
        path = RAW_DATA_DIR / relative
        if not path.exists():
            raise FileNotFoundError(
                f"缺少数据文件 {path}。请先运行：python src/download_data.py"
            )
        variables = _load_mat_variables(path)
        signal = variables.get("gs")
        if signal is None or len(np.asarray(signal)) < 2:
            raise ValueError(f"{path.name} 中没有找到有效的 gs 振动信号")
        record = {
            "file": path.name,
            "condition": condition_from_name(path.name),
            "split": "train" if "train_data" in relative else "test",
            "sr": float(variables.get("sr", np.nan)),
            "rate": float(variables.get("rate", np.nan)),
            "load": float(variables.get("load", np.nan)),
            "BPFO": float(variables.get("BPFO", np.nan)),
            "BPFI": float(variables.get("BPFI", np.nan)),
            "FTF": float(variables.get("FTF", np.nan)),
            "BSF": float(variables.get("BSF", np.nan)),
            "signal": np.asarray(signal, dtype=float).reshape(-1),
        }
        records.append(record)
    return records

