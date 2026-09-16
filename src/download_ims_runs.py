"""下载 IMS 2nd 与 4th 试验批次的早期健康文件。

目的：解决"同一批次相邻时间点不独立"的限制。
1st / 2nd / 4th 是三次不同的试验（不同轴承与运行周期），
因此提供 3 个相互独立的运行单元（run-level independence）。

来源与访问方式同 `download_ims.py`（GitHub LFS 媒体端点）。
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "ims"

MEDIA_BASE = (
    "https://media.githubusercontent.com/media/younesmekouar25/"
    "IMACAB-predictive-maintenance/main/data/ims_raw"
)

SUBSETS: dict[str, list[str]] = {
    "2nd_test": [
        "2004.02.12.10.32.39.txt",
        "2004.02.12.10.42.39.txt",
        "2004.02.12.10.52.39.txt",
        "2004.02.12.11.02.39.txt",
        "2004.02.12.11.12.39.txt",
        "2004.02.12.11.22.39.txt",
    ],
    "4th_test": [
        "2004.03.04.09.27.46.txt",
        "2004.03.04.09.32.46.txt",
        "2004.03.04.09.42.46.txt",
        "2004.03.04.09.52.46.txt",
        "2004.03.04.10.02.46.txt",
        "2004.03.04.10.12.46.txt",
    ],
}


def download(subset: str, name: str) -> Path:
    target_dir = RAW_ROOT / subset
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / name
    if destination.exists() and destination.stat().st_size > 1000:
        print(f"skip (exists): {subset}/{name}")
        return destination

    url = f"{MEDIA_BASE}/{subset}/{name}"
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
                raise RuntimeError("下载内容过小，可能是 LFS 指针")
            temp.replace(destination)
            print(
                f"downloaded: {subset}/{name} "
                f"({destination.stat().st_size/1e6:.2f} MB)"
            )
            return destination
        except (urllib.error.URLError, TimeoutError, ConnectionError, RuntimeError) as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"下载失败: {subset}/{name}\n原因: {last_error}")


def main() -> None:
    for subset, names in SUBSETS.items():
        print(f"\n===== {subset} =====")
        for name in names:
            download(subset, name)
    print("\n各批次文件数：")
    for subset in ("1st_test", "2nd_test", "4th_test"):
        directory = RAW_ROOT / subset
        if directory.exists():
            files = sorted(directory.glob("*.txt"))
            size_mb = sum(path.stat().st_size for path in files) / 1e6
            print(f"  {subset}: {len(files)} 个文件, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
