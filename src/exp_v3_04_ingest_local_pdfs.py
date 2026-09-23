"""把桌面 manual_pdfs 里的 PDF 变成机器可读的语料清单（审计与相关工作的共同输入）。

输出：experiments/EXP-V3-04-local-corpus.csv
列：file, pages, title_guess, year, doi, venue_guess, snippet(1200 字)
"""

from __future__ import annotations

import csv
import io
import os
import re

from pypdf import PdfReader

SRC_DIR = r'C:\Users\32597\Desktop\manual_pdfs'
OUT_CSV = (r'C:\Users\32597\Documents\Codex\2026-09-06\github\motor-health-monitor'
           r'\experiments\EXP-V3-04-local-corpus.csv')


def first_text(path: str) -> tuple[int, str]:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages[:2]:
        try:
            texts.append(page.extract_text() or '')
        except Exception:  # noqa: BLE001
            continue
    return len(reader.pages), '\n'.join(texts)


def main() -> None:
    files = sorted(f for f in os.listdir(SRC_DIR) if f.lower().endswith('.pdf'))
    print(f'发现 {len(files)} 个 PDF')
    rows = []
    for name in files:
        path = os.path.join(SRC_DIR, name)
        try:
            pages, text = first_text(path)
        except Exception as exc:  # noqa: BLE001
            print(f'  [FAIL] {name}: {exc}')
            rows.append(dict(file=name, pages=0, title_guess='', year='', doi='',
                             venue_guess='', snippet=f'EXTRACT_FAILED: {exc}'))
            continue
        flat = re.sub(r'\s+', ' ', text).strip()
        lines = [ln.strip() for ln in text.split('\n') if len(ln.strip()) > 15]
        title = lines[0] if lines else ''
        year = ''
        years = re.findall(r'\b(19[89]\d|20[0-2]\d)\b', flat)
        if years:
            year = max(years)
        doi = ''
        m = re.search(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', flat)
        if m:
            doi = m.group(0).rstrip('.,')
        venue = ''
        for key in ('Mechanical Systems and Signal Processing', 'Measurement',
                    'IEEE Transactions', 'Reliability Engineering', 'Sensors',
                    'PHM', 'Condition Monitoring', 'Applied Sciences', 'Machines',
                    'Structural Health Monitoring', 'Expert Systems'):
            if key.lower() in flat.lower():
                venue = key
                break
        rows.append(dict(file=name, pages=pages, title_guess=title[:200], year=year, doi=doi,
                         venue_guess=venue, snippet=flat[:1200]))
        print(f'  {pages:>3}p | {year or "----"} | {title[:78]}')

    with io.open(OUT_CSV, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['file', 'pages', 'title_guess', 'year', 'doi',
                                                'venue_guess', 'snippet'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n已写出 {OUT_CSV}（{len(rows)} 行）')


if __name__ == '__main__':
    main()
