"""Scopus 补检索：直接检验本项目的三条核心主张是否已有工作占据。

输出 docs/literature/AUDIT_scopus_gap.csv 并打印摘要。
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "literature"
KEY_FILE = Path.home() / ".codex" / "scopus_api_key.txt"
ENDPOINT = "https://api.elsevier.com/content/search/scopus"
FIELDS = ("dc:title,dc:creator,prism:publicationName,prism:coverDate,"
          "citedby-count,prism:doi,dc:description,eid")

QUERIES: dict[str, str] = {
    "S1 误报与振动异常检测":
        'TITLE-ABS-KEY("false alarm" AND "vibration" AND ("anomaly detection" OR "novelty detection"))',
    "S2 误报率与旋转机械监测":
        'TITLE-ABS-KEY(("false alarm rate" OR "nuisance alarm") AND ("wind turbine" '
        'OR "rotating machinery" OR "bearing") AND ("condition monitoring" OR "monitoring"))',
    "S3 样本量与异常检测":
        'TITLE-ABS-KEY(("sample size" OR "number of recordings" OR "training set size") '
        'AND ("anomaly detection" OR "novelty detection") AND ("vibration" OR "bearing" '
        'OR "machinery" OR "industrial"))',
    "S4 单一划分与比较":
        'TITLE-ABS-KEY(("single split" OR "single train-test" OR "one split") '
        'AND ("evaluation" OR "benchmark" OR "comparison"))',
    "S5 划分方差与故障诊断":
        'TITLE-ABS-KEY("split" AND ("variance" OR "variability") AND ("fault diagnosis" '
        'OR "condition monitoring"))',
    "S6 检测性能的置信区间":
        'TITLE-ABS-KEY(("confidence interval" OR "uncertainty") AND ("false alarm rate" '
        'OR "detection performance") AND ("condition monitoring" OR "anomaly detection"))',
    "S7 健康基线漂移":
        'TITLE-ABS-KEY(("healthy baseline" OR "baseline model") AND ("drift" OR "aging" '
        'OR "seasonal") AND ("condition monitoring" OR "anomaly detection"))',
    "S8 独立单元/轴承级划分":
        'TITLE-ABS-KEY(("bearing-wise" OR "unit-wise" OR "by bearing" OR "per bearing") '
        'AND ("split" OR "validation" OR "evaluation") AND ("fault" OR "anomaly"))',
}


def api_key() -> str:
    key = os.environ.get("SCOPUS_API_KEY", "").strip()
    if key:
        return key
    return KEY_FILE.read_text(encoding="utf-8").strip()


def search(query: str, key: str, count: int = 10) -> tuple[str, list[dict]]:
    params = urllib.parse.urlencode({"query": query, "count": count, "field": FIELDS})
    request = urllib.request.Request(
        f"{ENDPOINT}?{params}", headers={"X-ELS-APIKey": key, "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    block = payload.get("search-results") or {}
    rows = []
    for item in block.get("entry") or []:
        if "error" in item:
            continue
        rows.append(
            {
                "query": query,
                "title": item.get("dc:title"),
                "authors": item.get("dc:creator"),
                "venue": item.get("prism:publicationName"),
                "date": item.get("prism:coverDate"),
                "cited_by": item.get("citedby-count"),
                "doi": item.get("prism:doi"),
                "abstract": (item.get("dc:description") or "").replace("\n", " ").strip(),
            }
        )
    return block.get("opensearch:totalResults", "0"), rows


def main() -> None:
    key = api_key()
    all_rows: list[dict] = []
    for label, query in QUERIES.items():
        try:
            total, rows = search(query, key)
        except Exception as exc:  # noqa: BLE001
            print(f"== {label}: FAILED {exc}")
            continue
        print("=" * 100)
        print(f"== {label}  (总命中 {total}，显示前 {len(rows)})")
        for row in rows:
            year = str(row["date"])[:7] if row["date"] else "?"
            print(f"  [{year}] {str(row['title'])[:120]}")
            print(f"        {str(row['venue'])[:70]} | cited={row['cited_by']} | {row['doi']}")
            abstract = str(row["abstract"])
            if len(abstract) > 60:
                print("        ABS: " + abstract[:420])
            all_rows.extend([{**row, "label": label}])
        time.sleep(1.2)

    pd.DataFrame(all_rows).to_csv(OUT / "AUDIT_scopus_gap.csv", index=False)
    print(f"\nsaved -> {OUT / 'AUDIT_scopus_gap.csv'}")


if __name__ == "__main__":
    main()