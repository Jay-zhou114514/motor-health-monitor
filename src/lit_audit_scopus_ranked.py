"""Scopus 针对性审计（按相关度与被引排序），输出可直接阅读的 Markdown。

检索式围绕本项目的可主张结论；每条同时给出
「相关度前 10」与「被引前 10」，避免只看最新论文造成的偏差。
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
    "Q1 split 策略与故障诊断":
        'TITLE-ABS-KEY(("train-test split" OR "data split" OR "splitting strategy" '
        'OR "random split" OR "holdout") AND ("fault diagnosis" OR "condition monitoring" '
        'OR "bearing"))',
    "Q2 统计功效与状态监测":
        'TITLE-ABS-KEY(("statistical power" OR "power analysis" OR "underpowered") '
        'AND ("fault diagnosis" OR "condition monitoring" OR "predictive maintenance"))',
    "Q3 时间序列异常检测的评价批评":
        'TITLE-ABS-KEY(("time series" AND "anomaly detection") AND ("benchmark" OR "evaluation") '
        'AND ("pitfall" OR "flawed" OR "critique" OR "rethink" OR "misleading"))',
    "Q4 超参/模型选择的不稳定性":
        'TITLE-ABS-KEY(("hyperparameter" OR "model selection") AND ("instability" OR "stability") '
        'AND ("anomaly detection" OR "fault diagnosis" OR "machine learning"))',
    "Q5 normal-only 训练":
        'TITLE-ABS-KEY(("normal data only" OR "normal-only" OR "healthy data only" '
        'OR "only normal data") AND ("anomaly detection" OR "condition monitoring"))',
    "Q6 重复重采样/嵌套交叉验证的方差":
        'TITLE-ABS-KEY(("repeated cross-validation" OR "repeated random subsampling" '
        'OR "nested cross-validation") AND ("variance" OR "uncertainty"))',
    "Q7 评估协议（状态监测）":
        'TITLE-ABS-KEY(("evaluation protocol" OR "validation protocol" OR "experimental protocol") '
        'AND ("fault diagnosis" OR "condition monitoring" OR "predictive maintenance"))',
    "Q8 单类方法在轴承上的对比":
        'TITLE-ABS-KEY(("one-class" OR "Isolation Forest" OR "novelty detection") '
        'AND ("bearing" OR "rotating machinery"))',
    "Q9 健康数据误报":
        'TITLE-ABS-KEY(("false alarm" OR "false positive rate") AND ("bearing" OR "gearbox" '
        'OR "rotating machinery") AND ("unsupervised" OR "normal" OR "healthy"))',
    "Q10 样本量与深度学习诊断":
        'TITLE-ABS-KEY(("sample size" OR "number of samples" OR "training set size") '
        'AND "deep learning" AND ("bearing" OR "fault diagnosis"))',
    "Q11 基准方差与报告":
        'TITLE-ABS-KEY(("benchmark" OR "reproducibility") AND ("variance" OR "variability" '
        'OR "random seed") AND "machine learning")',
    "Q12 泄漏与时间序列验证":
        'TITLE-ABS-KEY(("data leakage" OR "leakage") AND ("time series" OR "sensor") '
        'AND ("validation" OR "evaluation"))',
}


def api_key() -> str:
    key = os.environ.get("SCOPUS_API_KEY", "").strip()
    if key:
        return key
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    raise SystemExit("未找到 Scopus API key")


def search(query: str, sort: str | None, key: str, count: int = 10) -> tuple[str, list[dict]]:
    params = {"query": query, "count": count, "field": FIELDS}
    if sort:
        params["sort"] = sort
    request = urllib.request.Request(
        f"{ENDPOINT}?{urllib.parse.urlencode(params)}",
        headers={"X-ELS-APIKey": key, "Accept": "application/json"},
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
    OUT.mkdir(parents=True, exist_ok=True)
    key = api_key()
    lines = ["# Scopus 针对性审计（按相关度 / 被引排序）", ""]
    all_rows: list[dict] = []
    for label, query in QUERIES.items():
        print(f"== {label}")
        lines.append(f"## {label}")
        lines.append("")
        lines.append("```text")
        lines.append(query)
        lines.append("```")
        lines.append("")
        for sort, sort_label in ((None, "相关度/默认"), ("-citedby-count", "被引最多")):
            try:
                total, rows = search(query, sort, key)
            except Exception as exc:  # noqa: BLE001
                print(f"   {sort_label} FAILED: {exc}")
                continue
            print(f"   {sort_label}: total={total}")
            lines.append(f"**{sort_label}**（总命中 {total}）")
            lines.append("")
            lines.append("| # | 年份 | 标题 | 来源 | 被引 | DOI |")
            lines.append("| ---: | --- | --- | --- | ---: | --- |")
            for index, row in enumerate(rows, 1):
                year = str(row["date"])[:7] if row["date"] else "?"
                title = str(row["title"]).replace("|", "/")[:150]
                venue = str(row["venue"]).replace("|", "/")[:60]
                lines.append(
                    f"| {index} | {year} | {title} | {venue} | {row['cited_by']} | {row['doi']} |"
                )
                record = dict(row)
                record["query"] = label
                record["rank_type"] = sort_label
                record["rank"] = index
                all_rows.append(record)
            lines.append("")
            time.sleep(1.2)
        lines.append("")
        time.sleep(1.2)

    (OUT / "AUDIT_scopus_ranked.md").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(OUT / "AUDIT_scopus_ranked.csv", index=False)
    print(f"\nsaved -> {OUT / 'AUDIT_scopus_ranked.md'}")


if __name__ == "__main__":
    main()