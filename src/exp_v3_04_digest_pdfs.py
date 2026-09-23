"""把 manual_pdfs 里的论文压成"与五点清单相关的关键句摘要"，供编码使用。

输出：experiments/EXP-V3-04-digests.md（每篇 ≤ 1600 字的关键句）
"""

from __future__ import annotations

import io
import os
import re

from pypdf import PdfReader

SRC = r'C:\Users\32597\Desktop\manual_pdfs'
OUT = (r'C:\Users\32597\Documents\Codex\2026-09-06\github\motor-health-monitor'
       r'\experiments\EXP-V3-04-digests.md')

KEYS = [
    'window', 'windows', 'sample', 'samples', 'training set', 'test set', 'hold-out',
    'holdout', 'split', 'random seed', 'seed', 'resampl', 'tie', 'tied', 'quantile',
    'threshold', 'bearings', 'bearing', 'unit', 'units', 'rig', 'dataset',
    'false alarm', 'false-alarm', 'false positive', 'CWRU', 'Paderborn', 'XJTU',
    'PRONOSTIA', 'IMS', 'coverage', 'conformal', 'calibrat', 'label', 'healthy',
]


def sentences(text: str) -> list[str]:
    flat = re.sub(r'\s+', ' ', text)
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', flat) if len(s.strip()) > 30]


def main() -> None:
    names = sorted(f for f in os.listdir(SRC) if f.lower().endswith('.pdf'))
    chunks = ['# EXP-V3-04 关键句摘要（用于五点清单编码）', '']
    for name in names:
        path = os.path.join(SRC, name)
        try:
            reader = PdfReader(path)
            text = '\n'.join((p.extract_text() or '') for p in reader.pages)
        except Exception as exc:  # noqa: BLE001
            chunks.append(f'## {name}\n\n(提取失败: {exc})\n')
            continue
        scored = []
        for sent in sentences(text):
            low = sent.lower()
            hits = sum(1 for k in KEYS if k.lower() in low)
            if hits:
                scored.append((hits, sent))
        scored.sort(key=lambda t: -t[0])
        picked, total = [], 0
        for _hits, sent in scored:
            if total + len(sent) > 1600:
                continue
            picked.append(sent)
            total += len(sent)
            if len(picked) >= 14:
                break
        chunks.append(f'## {name}\n')
        for sent in picked:
            chunks.append(f'- {sent}')
        chunks.append('')
        print(f'{name}: {len(picked)} sentences, {total} chars')

    with io.open(OUT, 'w', encoding='utf-8', newline='') as fh:
        fh.write('\n'.join(chunks) + '\n')
    print(f'\n已写出 {OUT}')


if __name__ == '__main__':
    main()
