"""按 DOI 或标题抓取摘要（Semantic Scholar -> Crossref 回退）。

用于判定"最接近的既有工作"与本研究主张的重叠程度。
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "literature"

DOIS = [
    ("minimum sample size vibration", "10.1016/j.eswa.2010.06.068"),
    ("minimum sample size brake", "10.1016/j.jestch.2014.09.007"),
    ("Wu Keogh flawed benchmarks", "10.1109/TKDE.2021.3112126"),
    ("CV pitfalls", "10.1186/1758-2946-6-10"),
    ("robust framework eval unsupervised", "10.1007/978-3-031-78395-1_4"),
    ("taxonomy eval metrics TSAD", "10.1016/j.neucom.2026.134547"),
    ("UCR anomaly archive guidelines", "10.1016/j.iswa.2026.200717"),
    ("PATE proximity aware", "10.1145/3637528.3671971"),
]

TITLES = [
    "Multivariate Time Series Anomaly Detection: Fancy Algorithms and Flawed Evaluation Methodology",
    "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark",
    "Current Time Series Anomaly Detection Benchmarks are Flawed",
]


def get(url: str, tries: int = 4) -> dict | None:
    delay = 2.0
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "lit-audit/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            if attempt == tries - 1:
                print(f"    FAIL {exc}")
                return None
            time.sleep(delay)
            delay *= 2
    return None


def s2_by_doi(doi: str) -> dict | None:
    url = (f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi)}"
           "?fields=title,abstract,year,venue,citationCount,externalIds")
    return get(url)


def s2_search(title: str) -> dict | None:
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?query="
           + urllib.parse.quote(title)
           + "&fields=title,abstract,year,venue,citationCount,externalIds&limit=3")
    data = get(url)
    if not data or not data.get("data"):
        return None
    for item in data["data"]:
        if item.get("abstract"):
            return item
    return data["data"][0]


def crossref(doi: str) -> dict | None:
    data = get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    if not data:
        return None
    message = data.get("message", {})
    abstract = message.get("abstract", "")
    for tag in ("<jats:p>", "</jats:p>", "<jats:title>", "</jats:title>", "Abstract"):
        abstract = abstract.replace(tag, " ")
    return {
        "title": (message.get("title") or [""])[0],
        "abstract": abstract.strip(),
        "year": (message.get("issued", {}).get("date-parts") or [[None]])[0][0],
        "venue": (message.get("container-title") or [""])[0],
        "citationCount": message.get("is-referenced-by-count"),
    }


def main() -> None:
    rows = []
    for label, doi in DOIS:
        print(f"== {label} ({doi})")
        record = s2_by_doi(doi) or {}
        if not record.get("abstract"):
            fallback = crossref(doi) or {}
            if fallback.get("abstract") and len(fallback["abstract"]) > len(record.get("abstract") or ""):
                record.update(fallback)
        rows.append({"label": label, "doi": doi, "title": record.get("title"),
                     "year": record.get("year"), "venue": record.get("venue"),
                     "cited": record.get("citationCount"), "abstract": record.get("abstract")})
        time.sleep(1.5)
    for title in TITLES:
        print(f"== {title[:60]}")
        record = s2_search(title) or {}
        rows.append({"label": title[:40], "doi": (record.get("externalIds") or {}).get("DOI"),
                     "title": record.get("title"), "year": record.get("year"),
                     "venue": record.get("venue"), "cited": record.get("citationCount"),
                     "abstract": record.get("abstract")})
        time.sleep(1.5)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "AUDIT_abstracts.csv", index=False)
    for row in rows:
        print("=" * 100)
        print(f"{row['title']} [{row['year']}] {row['venue']} cited={row['cited']}")
        abstract = row.get("abstract")
        print("ABS:", (str(abstract)[:1100] if abstract else "(未获取到摘要)"))


if __name__ == "__main__":
    main()