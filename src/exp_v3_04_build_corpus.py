"""EXP-V3-04 第 1 步：用 OpenAlex 构建候选语料并分层抽样 30 篇（预注册的抽样框）。

预注册：experiments/EXP-V3-04-preregistration.md
输出：experiments/EXP-V3-04-corpus.csv（全部候选）与 EXP-V3-04-sample.csv（分层随机 30 篇）
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import urllib.parse
import urllib.request

EXPERIMENTS = r'C:\Users\32597\Documents\Codex\2026-09-06\github\motor-health-monitor\experiments'
SEED = 20260922
FROM_DATE, TO_DATE = '2021-01-01', '2026-08-31'
QUERIES = [
    'bearing anomaly detection',
    'rolling element bearing fault detection',
    'bearing condition monitoring',
    'one-class bearing fault',
]
MAILTO = 'research@example.org'


def fetch(query: str) -> list[dict]:
    params = {
        'filter': f'from_publication_date:{FROM_DATE},to_publication_date:{TO_DATE},'
                  f'title_and_abstract.search:{query},type:article',
        'per-page': '200',
        'mailto': MAILTO,
    }
    url = 'https://api.openalex.org/works?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': 'mhm-audit/1.0'})
    with urllib.request.urlopen(req, timeout=90) as fh:
        return json.load(fh).get('results', [])


def main() -> None:
    seen: dict[str, dict] = {}
    for query in QUERIES:
        try:
            results = fetch(query)
            print(f'query "{query}": {len(results)} hits')
        except Exception as exc:  # noqa: BLE001
            print(f'query "{query}" failed: {exc}')
            continue
        for work in results:
            doi = (work.get('doi') or '').replace('https://doi.org/', '')
            if not doi or doi in seen:
                continue
            venue = ''
            primary = work.get('primary_location') or {}
            source = (primary.get('source') or {})
            venue = source.get('display_name') or ''
            seen[doi] = {
                'doi': doi,
                'title': (work.get('title') or '').strip().replace('\n', ' '),
                'year': work.get('publication_year'),
                'venue': venue,
                'is_oa': bool((work.get('open_access') or {}).get('is_oa')),
                'oa_url': (work.get('open_access') or {}).get('oa_url') or '',
                'openalex_id': work.get('id', ''),
            }

    rows = sorted(seen.values(), key=lambda r: (r['year'] or 0, r['title']))
    all_path = os.path.join(EXPERIMENTS, 'EXP-V3-04-corpus.csv')
    with io.open(all_path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'candidates written: {len(rows)} -> {os.path.basename(all_path)}')

    # 分层随机抽样（按年份）
    rng = random.Random(SEED)
    by_year: dict[int, list[dict]] = {}
    for r in rows:
        by_year.setdefault(r['year'] or 0, []).append(r)
    sample = []
    years = sorted(by_year)
    per_year = max(1, 30 // max(1, len(years)))
    for year in years:
        pool = by_year[year][:]
        rng.shuffle(pool)
        sample.extend(pool[:per_year])
    if len(sample) < 30:
        rest = [r for r in rows if r not in sample]
        rng.shuffle(rest)
        sample.extend(rest[: 30 - len(sample)])
    sample = sample[:30]
    sample_path = os.path.join(EXPERIMENTS, 'EXP-V3-04-sample.csv')
    with io.open(sample_path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample)
    oa = sum(1 for r in sample if r['is_oa'])
    print(f'sample written: {len(sample)} (OA {oa}, paywalled {len(sample) - oa}) '
          f'-> {os.path.basename(sample_path)}')
    print('\n样本（年份 | OA | 期刊 | 标题前 60 字）:')
    for r in sample:
        print(f"  {r['year']} | {'OA ' if r['is_oa'] else '付费'} | {r['venue'][:28]:<28} | {r['title'][:60]}")


if __name__ == '__main__':
    main()
