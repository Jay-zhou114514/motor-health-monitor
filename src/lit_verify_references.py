"""核实拟引用文献的元数据（Crossref + Semantic Scholar + arXiv）。

只输出**能核实的**条目；核不到的显式标为 unverified，禁止进入正文引用。
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
UA = {"User-Agent": "lit-verify/1.0 (jay-zhou114514@users.noreply.github.com)"}

DOIS = {
    "wu_keogh_2021": "10.1109/TKDE.2021.3112126",
    "cv_pitfalls_2014": "10.1186/1758-2946-6-10",
    "min_sample_eswa_2010": "10.1016/j.eswa.2010.06.068",
    "min_sample_jestch_2015": "10.1016/j.jestch.2014.09.007",
    "robust_eval_2024": "10.1007/978-3-031-78395-1_4",
    "pate_2024": "10.1145/3637528.3671971",
    "taxonomy_2026": "10.1016/j.neucom.2026.134547",
    "vieira_2026": "10.1016/j.ymssp.2025.114640",
}

ARXIV_IDS = {
    "kim_2021": "2109.05257",
    "rank_instability_2026": "2608.04613",
    "msad_2025": "2510.26643",
    "mtsbench_2025": "2506.21550",
    "tab_2025": "2506.18046",
}

TITLE_QUERIES = {
    "bouthillier_2021": "Accounting for Variance in Machine Learning Benchmarks",
    "fancy_flawed_2024": "Multivariate Time Series Anomaly Detection Fancy Algorithms and Flawed Evaluation Methodology",
    "knap_2026": "Cross-domain evaluation of bearing fault diagnosis recording-level",
}


def get(url: str, tries: int = 3):
    delay = 2.0
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if attempt == tries - 1:
                return None
            time.sleep(delay)
            delay *= 2
    return None


def crossref(doi: str) -> dict:
    raw = get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    if not raw:
        return {}
    msg = json.loads(raw.decode())["message"]
    authors = "; ".join(
        f"{a.get('family','')} {a.get('given','')}".strip() for a in msg.get("author", [])
    )
    return {
        "title": (msg.get("title") or [""])[0],
        "authors": authors,
        "year": (msg.get("issued", {}).get("date-parts") or [[None]])[0][0],
        "venue": (msg.get("container-title") or [""])[0],
        "doi": msg.get("DOI"),
        "type": msg.get("type"),
        "source": "crossref",
    }


def arxiv(aid: str) -> dict:
    ns = "http://www.w3.org/2005/Atom"
    params = urllib.parse.urlencode({"id_list": aid})
    raw = get(f"http://export.arxiv.org/api/query?{params}")
    if not raw:
        return {}
    root = ET.fromstring(raw)
    entry = root.find(f"{{{ns}}}entry")
    if entry is None:
        return {}
    return {
        "title": (entry.findtext(f"{{{ns}}}title", "") or "").strip().replace("\n", " "),
        "authors": "; ".join(a.findtext(f"{{{ns}}}name", "") for a in entry.findall(f"{{{ns}}}author")),
        "year": (entry.findtext(f"{{{ns}}}published", "") or "")[:4],
        "venue": "arXiv",
        "doi": f"arXiv:{aid}",
        "type": "preprint",
        "source": "arxiv",
    }


def s2_by_title(title: str) -> dict:
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?query="
           + urllib.parse.quote(title)
           + "&fields=title,authors,year,venue,externalIds,citationCount&limit=3")
    raw = get(url)
    if not raw:
        return {}
    data = json.loads(raw.decode())
    for item in data.get("data", []):
        ext = item.get("externalIds") or {}
        return {
            "title": item.get("title"),
            "authors": "; ".join(a.get("name", "") for a in item.get("authors", [])),
            "year": item.get("year"),
            "venue": item.get("venue"),
            "doi": ext.get("DOI") or (f"arXiv:{ext['ArXiv']}" if ext.get("ArXiv") else None),
            "type": "paper",
            "source": "semanticscholar",
            "cited": item.get("citationCount"),
        }
    return {}


def main() -> None:
    rows = []
    for key, doi in DOIS.items():
        rec = crossref(doi)
        rec.update({"key": key, "queried_doi": doi, "verified": bool(rec.get("title"))})
        rows.append(rec)
        print(f"[doi]    {key:26s} {rec.get('year')} | {str(rec.get('title'))[:80]}")
        time.sleep(1.2)
    for key, aid in ARXIV_IDS.items():
        rec = arxiv(aid)
        rec.update({"key": key, "queried_doi": f"arXiv:{aid}", "verified": bool(rec.get("title"))})
        rows.append(rec)
        print(f"[arxiv]  {key:26s} {rec.get('year')} | {str(rec.get('title'))[:80]}")
        time.sleep(2.0)
    for key, title in TITLE_QUERIES.items():
        rec = s2_by_title(title)
        rec.update({"key": key, "queried_doi": None, "verified": bool(rec.get("title"))})
        rows.append(rec)
        print(f"[title]  {key:26s} {rec.get('year')} | {str(rec.get('title'))[:80]} | {rec.get('doi')}")
        time.sleep(2.0)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "REFERENCES_VERIFIED.csv", index=False)
    bad = frame[~frame["verified"].astype(bool)]
    print(f"\nverified {int(frame['verified'].sum())}/{len(frame)}")
    if len(bad):
        print("UNVERIFIED:", bad["key"].tolist())


if __name__ == "__main__":
    main()