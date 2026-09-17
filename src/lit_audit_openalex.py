"""面向本项目研究问题的针对性文献审计（OpenAlex + arXiv）。

目的：不用于"参考文献表"，而是回答一个具体问题：
**我们要主张的东西，是否已经有人在本领域做过？**

输出：
- docs/literature/AUDIT_openalex.csv
- docs/literature/AUDIT_arxiv.csv
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "literature"
MAILTO = "jay-zhou114514@users.noreply.github.com"

OPENALEX_QUERIES = {
    "OA1 bearing fault diagnosis data splitting evaluation":
        "bearing fault diagnosis data splitting evaluation",
    "OA2 sample size statistical power fault diagnosis deep learning":
        "sample size statistical power fault diagnosis deep learning",
    "OA3 time series anomaly detection benchmark evaluation flawed":
        "time series anomaly detection benchmark evaluation flawed",
    "OA4 hyperparameter selection instability unsupervised anomaly detection":
        "hyperparameter selection instability unsupervised anomaly detection",
    "OA5 reproducibility machine learning benchmarks random seed variance":
        "reproducibility machine learning benchmarks random seed variance",
    "OA6 one-class condition monitoring false alarm":
        "one-class classification condition monitoring false alarm",
    "OA7 normal-only anomaly detection evaluation predictive maintenance":
        "normal-only anomaly detection evaluation predictive maintenance",
    "OA8 effect of data split on fault diagnosis accuracy":
        "effect of data splitting on fault diagnosis accuracy",
    "OA9 repeated random subsampling variance model evaluation":
        "repeated random subsampling variance model evaluation cross validation",
    "OA10 recording-level evaluation bearing dataset leakage":
        "data leakage bearing dataset evaluation deep learning",
}

ARXIV_QUERIES = {
    "AX1": 'all:"time series anomaly detection" AND all:"benchmark" AND all:"evaluation"',
    "AX2": 'all:"anomaly detection" AND all:"hyperparameter" AND all:"selection"',
    "AX3": 'all:"predictive maintenance" AND all:"evaluation" AND all:"benchmark"',
    "AX4": 'all:"machine learning benchmarks" AND all:"variance"',
    "AX5": 'all:"one-class" AND all:"anomaly detection" AND all:"false alarm"',
}


def http_json(url: str, tries: int = 5) -> dict:
    delay = 2.0
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": f"lit-audit ({MAILTO})"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            print(f"    retry {attempt + 1} after {exc}")
            time.sleep(delay)
            delay *= 2
    return {}


def reconstruct_abstract(inverted: dict | None) -> str:
    if not inverted:
        return ""
    positions = []
    for word, spots in inverted.items():
        for spot in spots:
            positions.append((spot, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def openalex_search(query: str, per_page: int = 25) -> list[dict]:
    params = urllib.parse.urlencode(
        {"search": query, "per_page": per_page, "mailto": MAILTO}
    )
    data = http_json(f"https://api.openalex.org/works?{params}")
    rows = []
    for item in data.get("results", []):
        source = ((item.get("primary_location") or {}).get("source") or {}).get("display_name")
        rows.append(
            {
                "query": query,
                "title": (item.get("title") or "").strip(),
                "year": item.get("publication_year"),
                "venue": source,
                "doi": item.get("doi"),
                "cited_by": item.get("cited_by_count"),
                "type": item.get("type"),
                "abstract": reconstruct_abstract(item.get("abstract_inverted_index")),
                "openalex_id": item.get("id"),
            }
        )
    return rows


def arxiv_search(query: str, max_results: int = 25) -> list[dict]:
    ns = "http://www.w3.org/2005/Atom"
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    request = urllib.request.Request(
        f"http://export.arxiv.org/api/query?{params}",
        headers={"User-Agent": f"lit-audit ({MAILTO})"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        root = ET.fromstring(response.read())
    rows = []
    for entry in root.findall(f"{{{ns}}}entry"):
        rows.append(
            {
                "query": query,
                "title": (entry.findtext(f"{{{ns}}}title", "") or "").strip().replace("\n", " "),
                "year": (entry.findtext(f"{{{ns}}}published", "") or "")[:10],
                "authors": "; ".join(
                    a.findtext(f"{{{ns}}}name", "") for a in entry.findall(f"{{{ns}}}author")
                )[:200],
                "arxiv_id": (entry.findtext(f"{{{ns}}}id", "") or "").split("/abs/")[-1],
                "abstract": (entry.findtext(f"{{{ns}}}summary", "") or "").strip().replace("\n", " "),
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    openalex_rows: list[dict] = []
    for label, query in OPENALEX_QUERIES.items():
        print(f"[OpenAlex] {label}")
        try:
            rows = openalex_search(query)
            print(f"    {len(rows)} hits")
            openalex_rows.extend(rows)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {exc}")
        time.sleep(1.5)

    arxiv_rows: list[dict] = []
    for label, query in ARXIV_QUERIES.items():
        print(f"[arXiv] {label}")
        try:
            rows = arxiv_search(query)
            print(f"    {len(rows)} hits")
            arxiv_rows.extend(rows)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {exc}")
        time.sleep(3.0)

    pd.DataFrame(openalex_rows).to_csv(OUT / "AUDIT_openalex.csv", index=False)
    pd.DataFrame(arxiv_rows).to_csv(OUT / "AUDIT_arxiv.csv", index=False)
    print(f"\nsaved {len(openalex_rows)} OpenAlex rows, {len(arxiv_rows)} arXiv rows -> {OUT}")


if __name__ == "__main__":
    main()