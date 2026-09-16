"""为 EXP-V1-07 下载 IMS 1st test 的小子集。

数据来源：NASA/University of Cincinnati IMS bearing run-to-failure 数据集
（GitHub 镜像：younesmekouar25/IMACAB-predictive-maintenance，Git LFS）。
通过 GitHub 的 LFS 媒体端点获取真实内容（raw.githubusercontent 只返回 LFS 指针）。

用途：
- 前 12 个文件（2003-10-22 前 1 小时）→ 健康记录池（正常）
- 最后 3 个文件（2003-11-25 深夜）→ 退化阶段记录（仅作描述性参考，非精确标签）

格式：制表符分隔，20480 行 × 8 通道，采样率 20 kHz，每行一个时间点。
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "data" / "raw" / "ims" / "1st_test"

MEDIA_BASE = (
    "https://media.githubusercontent.com/media/younesmekouar25/"
    "IMACAB-predictive-maintenance/main/data/ims_raw/1st_test"
)

EARLY_HEALTHY = [
    "2003.10.22.12.06.24.txt",
    "2003.10.22.12.09.13.txt",
    "2003.10.22.12.14.13.txt",
    "2003.10.22.12.19.13.txt",
    "2003.10.22.12.24.13.txt",
    "2003.10.22.12.29.13.txt",
    "2003.10.22.12.34.13.txt",
    "2003.10.22.12.39.13.txt",
    "2003.10.22.12.44.13.txt",
    "2003.10.22.12.49.13.txt",
    "2003.10.22.12.54.13.txt",
    "2003.10.22.12.59.13.txt",
]

LATE_DEGRADED = [
    "2003.11.25.23.19.56.txt",
    "2003.11.25.23.29.56.txt",
    "2003.11.25.23.39.56.txt",
]


def download(name: str, overwrite: bool = False) -> Path:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    destination = TARGET_DIR / name
    if destination.exists() and destination.stat().st_size > 1000 and not overwrite:
        print(f"skip (exists): {name}")
        return destination

    url = f"{MEDIA_BASE}/{name}"
    temp = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temp.open(
                "wb"
            ) as out_file:
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    out_file.write(chunk)
            if temp.stat().st_size < 100_000:
                raise RuntimeError(
                    f"下载内容过小（{temp.stat().st_size} 字节），可能是 LFS 指针"
                )
            temp.replace(destination)
            print(f"downloaded: {name} ({destination.stat().st_size/1e6:.2f} MB)")
            return destination
        except (urllib.error.URLError, TimeoutError, ConnectionError, RuntimeError) as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"下载失败: {name}\n原因: {last_error}")


def main() -> None:
    print("下载 IMS 1st test 子集（健康 12 个 + 退化 3 个）……")
    for name in EARLY_HEALTHY:
        download(name)
    for name in LATE_DEGRADED:
        download(name)
    files = sorted(TARGET_DIR.glob("*.txt"))
    total_mb = sum(path.stat().st_size for path in files) / 1e6
    print(f"\n完成：{len(files)} 个文件，共 {total_mb:.1f} MB")
    print(f"目录：{TARGET_DIR}")


if __name__ == "__main__":
    main()
