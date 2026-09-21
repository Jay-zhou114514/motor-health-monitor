"""Scopus 文献检索（可复现）。

依据 academic-research-suite / deep-research 的 bibliography_agent 规范：
- 先定义检索策略（数据库、关键词、布尔式、时间范围、纳入/排除标准）
- 记录每条检索式的命中数（PRISMA 流程的原始计数）
- 抓取代表性论文（按被引排序）用于注释性书目

API key 读取顺序（均不写入仓库）：
1. 环境变量 SCOPUS_API_KEY
2. 本机文件 C:\\Users\\<user>\\.codex\\scopus_api_key.txt

输出：
- docs/literature/scopus_query_counts.csv
- docs/literature/scopus_top_papers.csv
"""

from __future__ import annotations

import csv
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "literature"
KEY_FILE = Path.home() / ".codex" / "scopus_api_key.txt"

ENDPOINT = "https://api.elsevier.com/content/search/scopus"
FIELDS = "dc:title,prism:coverDate,citedby-count,prism:doi,prism:publicationName"

# 检索策略：围绕"简单检测器在状态监测中的评价有效性"
QUERIES: list[tuple[str, str]] = [
    # —— 拥挤区（确认不可作为创新点）——
    ("BASE_bearing", 'TITLE-ABS-KEY("bearing fault diagnosis")'),
    ("BASE_mahalanobis", 'TITLE-ABS-KEY("Mahalanobis distance" AND "fault diagnosis")'),
    ("BASE_shrinkage", 'TITLE-ABS-KEY("covariance" AND "shrinkage" AND "anomaly detection")'),
    # —— 功效与样本量 ——
    ("POWER_cms", 'TITLE-ABS-KEY("power analysis" AND ("fault diagnosis" OR "condition monitoring"))'),
    ("POWER_ml", 'TITLE-ABS-KEY(("statistical power" OR "sample size") AND "machine learning" AND ("condition monitoring" OR "fault diagnosis"))'),
    ("POWER_under", 'TITLE-ABS-KEY("underpowered" AND "machine learning")'),
    # —— 评价有效性 ——
    ("EVAL_leakage", 'TITLE-ABS-KEY(("data leakage" OR "leakage-safe" OR "leakage-aware") AND ("fault diagnosis" OR "bearing"))'),
    ("EVAL_pitfall", 'TITLE-ABS-KEY(("evaluation" AND ("pitfall" OR "validity")) AND ("machine learning" OR "fault diagnosis"))'),
    ("EVAL_repro", 'TITLE-ABS-KEY("reproducibility" AND ("fault diagnosis" OR "condition monitoring"))'),
    # —— 多层级评价 ——
    ("LEVEL_window_file", 'TITLE-ABS-KEY(("window-level" OR "file-level" OR "record-level" OR "segment-level") AND ("fault diagnosis" OR "anomaly detection"))'),
    # —— 跨设备/跨数据集 ——
    ("GENERAL_cross", 'TITLE-ABS-KEY(("cross-dataset" OR "cross-machine" OR "cross-rig") AND ("fault diagnosis" OR "condition monitoring"))'),
    ("GENERAL_domain", 'TITLE-ABS-KEY("domain shift" AND ("fault diagnosis" OR "bearing"))'),
    # —— 单类基线 ——
    ("BASELINE_oneclass", 'TITLE-ABS-KEY(("one-class SVM" OR "novelty detection" OR "Isolation Forest") AND ("bearing" OR "fault diagnosis" OR "condition monitoring"))'),
    # —— 负结果与预注册 ——
    ("NEG_prereg", 'TITLE-ABS-KEY("preregistration" AND ("machine learning" OR "anomaly detection"))'),
    ("NEG_negative", 'TITLE-ABS-KEY("negative results" AND ("machine learning" OR "fault diagnosis"))'),
]


def api_key() -> str:
    key = os.environ.get("SCOPUS_API_KEY", "").strip()
    if key:
        return key
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    raise SystemExit(
        "未找到 Scopus API key：请设置环境变量 SCOPUS_API_KEY，"
        f"或写入 {KEY_FILE}"
    )


def fetch(query: str, key: str, count: int = 1, sort: str | None = None):
    params = {"query": query, "count": str(count), "field": FIELDS}
    if sort:
        params["sort"] = sort
    url = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url, headers={"X-ELS-APIKey": key, "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key = api_key()

    counts: list[dict] = []
    papers: list[dict] = []

    for query_id, query in QUERIES:
        try:
            result = fetch(query, key, count=1)
            total = int(result["search-results"]["opensearch:totalResults"])
        except Exception as exc:  # noqa: BLE001
            print(f"[{query_id}] count failed: {exc}")
            continue
        counts.append({"query_id": query_id, "query": query, "total": total})
        print(f"[{query_id}] total={total}")
        time.sleep(0.4)

        # 抓取被引最高的 5 篇代表作（用于注释性书目）
        try:
            top = fetch(query, key, count=5, sort="-citedby-count")
            for entry in top.get("search-results", {}).get("entry", []):
                date = str(entry.get("prism:coverDate", ""))[:4]
                papers.append(
                    {
                        "query_id": query_id,
                        "year": date,
                        "citations": entry.get("citedby-count", ""),
                        "title": (entry.get("dc:title", "") or "").replace("\n", " "),
                        "venue": entry.get("prism:publicationName", ""),
                        "doi": entry.get("prism:doi", ""),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[{query_id}] top papers failed: {exc}")
        time.sleep(0.4)

    with (OUT_DIR / "scopus_query_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "query", "total"])
        writer.writeheader()
        writer.writerows(counts)

    with (OUT_DIR / "scopus_top_papers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "year", "citations", "title", "venue", "doi"],
        )
        writer.writeheader()
        writer.writerows(papers)

    print(f"\n已保存：{OUT_DIR / 'scopus_query_counts.csv'}")
    print(f"已保存：{OUT_DIR / 'scopus_top_papers.csv'}")


if __name__ == "__main__":
    main()
