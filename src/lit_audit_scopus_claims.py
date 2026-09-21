"""针对性文献审计（Scopus）：逐条检验本项目的主张是否已被做过。

与 `lit_search_scopus.py` 的区别：
- 前者是"领域地图"（找拥挤区）；本脚本是"主张检验"（找最接近的竞争工作）；
- 每条检索式对应本项目的一条可主张结论，便于逐条判定"可声称 / 不可声称"。

API key 读取顺序（**不写入仓库**）：
1. 环境变量 SCOPUS_API_KEY
2. C:\\Users\\<user>\\.codex\\scopus_api_key.txt

输出：
- docs/literature/AUDIT_scopus.csv         逐条命中（含摘要）
- docs/literature/AUDIT_scopus_summary.csv 每条检索式的命中数
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
          "citedby-count,prism:doi,dc:description,eid,subtypeDescription")

CLAIMS: dict[str, str] = {
    # 主张 A：评估报告的误报率主要由划分组成决定（不是方法/调参）
    "A1_train_test_split_fd":
        'TITLE-ABS-KEY(("train-test split" OR "data split" OR "splitting strategy" '
        'OR "random split" OR "split variability") AND ("fault diagnosis" '
        'OR "condition monitoring" OR "bearing" OR "predictive maintenance"))',
    "A2_split_effect_accuracy":
        'TITLE-ABS-KEY((("effect of" OR "impact of" OR "influence of") AND '
        '("data split" OR "splitting" OR "partition") AND ("accuracy" OR "performance")) '
        'AND TITLE-ABS-KEY(("machine learning" OR "deep learning")))',
    "A3_repeated_resampling_variance":
        'TITLE-ABS-KEY(("repeated cross-validation" OR "repeated random subsampling" '
        'OR "repeated holdout" OR "nested cross-validation") AND ("variance" '
        'OR "uncertainty" OR "instability"))',
    "A4_benchmark_variance_ml":
        'TITLE-ABS-KEY(("benchmark" OR "evaluation") AND ("variance" OR "variability") '
        'AND ("machine learning") AND ("reporting" OR "protocol" OR "protocols"))',
    # 主张 B：小样本下训练内选择准则不判别 / 选参不稳
    "B1_hyperparameter_instability_ad":
        'TITLE-ABS-KEY(("hyperparameter" OR "model selection" OR "hyper-parameter") '
        'AND ("instability" OR "stability" OR "sensitivity") AND '
        '("anomaly detection" OR "fault diagnosis" OR "condition monitoring"))',
    "B2_small_sample_anomaly":
        'TITLE-ABS-KEY(("small sample" OR "small-sample" OR "limited data" '
        'OR "few samples" OR "data scarcity") AND ("anomaly detection" OR "one-class" '
        'OR "novelty detection"))',
    "B3_small_sample_bearing":
        'TITLE-ABS-KEY(("small sample" OR "small-sample" OR "limited data") '
        'AND ("bearing" OR "gearbox") AND ("diagnosis" OR "detection"))',
    "B4_statistical_power_cm":
        'TITLE-ABS-KEY(("statistical power" OR "power analysis" OR "underpowered") '
        'AND ("fault diagnosis" OR "condition monitoring" OR "predictive maintenance"))',
    # 主张 C：normal-only / 单类方法在健康记录上的误报是主要风险
    "C1_false_alarm_normal_only":
        'TITLE-ABS-KEY(("false alarm" OR "false positive") AND ("condition monitoring" '
        'OR "anomaly detection") AND ("vibration" OR "bearing" OR "machine"))',
    "C2_oneclass_vs_simple":
        'TITLE-ABS-KEY(("one-class" OR "novelty detection" OR "Isolation Forest") '
        'AND ("bearing" OR "rotating machinery") AND ("false alarm" OR "comparison" '
        'OR "baseline"))',
    "C3_normal_only_training":
        'TITLE-ABS-KEY(("normal data only" OR "healthy data only" OR "normal-only" '
        'OR "only normal data") AND ("anomaly detection" OR "condition monitoring" '
        'OR "predictive maintenance"))',
    # 主张 D：时间序列异常检测的评价协议本身有问题（跨领域先例）
    "D1_tsad_evaluation_critique":
        'TITLE-ABS-KEY(("time series" AND "anomaly detection") AND ("benchmark" '
        'OR "evaluation") AND ("pitfall" OR "flawed" OR "critique" OR "rethink" '
        'OR "misleading"))',
    "D2_tsad_point_adjust":
        'TITLE-ABS-KEY("point adjustment" AND "anomaly detection")',
    # 主张 E：可复现性 / 泄漏 / 基准问题
    "E1_leakage_fd":
        'TITLE-ABS-KEY(("data leakage" OR "leakage") AND ("fault diagnosis" '
        'OR "bearing" OR "condition monitoring") AND ("evaluation" OR "benchmark" '
        'OR "validation"))',
    "E2_reproducibility_cm":
        'TITLE-ABS-KEY(("reproducibility" OR "replicability") AND ("fault diagnosis" '
        'OR "condition monitoring" OR "predictive maintenance" OR "prognostics"))',
    "E3_negative_results_ml":
        'TITLE-ABS-KEY(("negative results" OR "null results" OR "failure to replicate") '
        'AND ("machine learning" OR "artificial intelligence" OR "data science"))',
    # 主张 F：本项目统一问题的直接检验
    "F1_reliability_of_conclusions":
        'TITLE-ABS-KEY(("how reliable" OR "reliability of" OR "trustworthiness") '
        'AND ("fault diagnosis" OR "condition monitoring" OR "anomaly detection") '
        'AND ("conclusions" OR "findings" OR "performance evaluation"))',
    "F2_evaluation_protocol_cm":
        'TITLE-ABS-KEY(("evaluation protocol" OR "experimental protocol" OR '
        '"validation protocol") AND ("fault diagnosis" OR "condition monitoring" '
        'OR "predictive maintenance"))',
}


def api_key() -> str:
    key = os.environ.get("SCOPUS_API_KEY", "").strip()
    if key:
        return key
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    raise SystemExit("未找到 Scopus API key")


def search(query: str, count: int, key: str) -> dict:
    params = urllib.parse.urlencode({"query": query, "count": count, "field": FIELDS})
    request = urllib.request.Request(
        f"{ENDPOINT}?{params}",
        headers={"X-ELS-APIKey": key, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def parse(payload: dict, label: str, query: str) -> list[dict]:
    block = (payload.get("search-results") or {})
    entries = block.get("entry") or []
    rows = []
    for item in entries:
        if "error" in item:
            continue
        rows.append(
            {
                "claim": label,
                "query": query,
                "title": item.get("dc:title"),
                "authors": item.get("dc:creator"),
                "venue": item.get("prism:publicationName"),
                "date": item.get("prism:coverDate"),
                "cited_by": item.get("citedby-count"),
                "doi": item.get("prism:doi"),
                "eid": item.get("eid"),
                "type": item.get("subtypeDescription"),
                "abstract": (item.get("dc:description") or "").replace("\n", " ").strip(),
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    key = api_key()
    rows: list[dict] = []
    summary: list[dict] = []
    for label, query in CLAIMS.items():
        try:
            payload = search(query, 25, key)
            block = payload.get("search-results") or {}
            total = block.get("opensearch:totalResults", "0")
            parsed = parse(payload, label, query)
            rows.extend(parsed)
            summary.append({"claim": label, "total_hits": total, "fetched": len(parsed)})
            print(f"{label:38s} total={total:>7} fetched={len(parsed)}")
        except Exception as exc:  # noqa: BLE001
            summary.append({"claim": label, "total_hits": "ERROR", "fetched": 0})
            print(f"{label:38s} FAILED: {exc}")
        time.sleep(1.2)

    pd.DataFrame(rows).to_csv(OUT / "AUDIT_scopus.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "AUDIT_scopus_summary.csv", index=False)
    print(f"\nsaved {len(rows)} rows -> {OUT / 'AUDIT_scopus.csv'}")


if __name__ == "__main__":
    main()