"""数据下载脚本。

用法：
    python src/download_data.py            # 只下载本次实验需要的 8 个文件
    python src/download_data.py --all      # 下载仓库里的全部 20 个文件
"""

from __future__ import annotations

import argparse

import data_loading


def main() -> None:
    parser = argparse.ArgumentParser(description="下载滚动轴承故障数据集")
    parser.add_argument(
        "--all",
        action="store_true",
        help="下载数据仓库里的全部文件（约 41 MB）",
    )
    args = parser.parse_args()
    files = data_loading.ALL_RELATIVE_FILES if args.all else data_loading.MINIMAL_RELATIVE_FILES
    paths = data_loading.ensure_dataset_files(files)
    print(f"共 {len(paths)} 个文件已就绪：")
    for path in paths:
        print(f"  {path.relative_to(data_loading.RAW_DATA_DIR)}")


if __name__ == "__main__":
    main()
