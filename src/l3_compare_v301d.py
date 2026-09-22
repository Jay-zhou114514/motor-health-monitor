"""L3 比对：EXP-V3-01d 重跑产物 vs 重跑前快照（排除非确定性列），写哈希记录。"""

from __future__ import annotations

import csv
import hashlib
import io
import os

MASTER = r'C:\Users\32597\Documents\Codex\2026-09-06\github\motor-health-monitor'
OUT = os.path.join(MASTER, 'outputs')
PAIRS = [('L3_v301d_before_exp_v3_01d_runs.csv', 'exp_v3_01d_runs.csv'),
         ('L3_v301d_before_exp_v3_01d_summary.csv', 'exp_v3_01d_summary.csv'),
         ('L3_v301d_before_exp_v3_01d_partial_P1_strict.csv',
          'exp_v3_01d_partial_P1_strict.csv')]
NONDET = ('seconds', 'elapsed', 'time', 'duration', 'wall', 'cpu')

lines = ['# EXP-V3-01d  L3 确定性重跑比对', '',
         '快照：outputs/L3_v301d_before_*.csv（重跑前）',
         '重跑：outputs/exp_v3_01d_*.csv（同脚本、同种子）',
         '规则：先逐字节比对；若不同，则按列比对并排除非确定性列（计时/CPU）。', '']

all_ok = True
for old_name, new_name in PAIRS:
    old_path, new_path = os.path.join(OUT, old_name), os.path.join(OUT, new_name)
    if not (os.path.isfile(old_path) and os.path.isfile(new_path)):
        lines.append(f'{new_name}: 缺少文件（old={os.path.isfile(old_path)}, new={os.path.isfile(new_path)}）')
        all_ok = False
        continue
    with io.open(old_path, 'rb') as fh:
        old = fh.read()
    with io.open(new_path, 'rb') as fh:
        new = fh.read()
    same = old == new
    verdict = 'byte-identical' if same else ''
    if not same:
        rows_old = list(csv.DictReader(io.StringIO(old.decode('utf-8-sig', 'replace'))))
        rows_new = list(csv.DictReader(io.StringIO(new.decode('utf-8-sig', 'replace'))))
        if rows_old and rows_new and list(rows_old[0]) == list(rows_new[0]):
            cols = [c for c in rows_old[0] if not any(k in c.lower() for k in NONDET)]
            diffs = []
            if len(rows_old) != len(rows_new):
                diffs.append(f'行数 {len(rows_old)} -> {len(rows_new)}')
            for i, (a, b) in enumerate(zip(rows_old, rows_new)):
                for c in cols:
                    if a.get(c) != b.get(c):
                        diffs.append(f'row {i} col {c}: {a.get(c)} -> {b.get(c)}')
            verdict = ('identical on all scientific columns' if not diffs
                       else 'scientific differences: ' + '; '.join(diffs[:5]))
            all_ok &= not diffs
        else:
            verdict = 'differs and header changed'
            all_ok = False
    lines += [f'{new_name}',
              f'  sha256 (rerun)  {hashlib.sha256(new).hexdigest()}',
              f'  sha256 (before) {hashlib.sha256(old).hexdigest()}',
              f'  verdict: {verdict}', '']

lines.append(f'总判定：{"L3 通过" if all_ok else "L3 未通过（见上）"}')
with io.open(os.path.join(MASTER, 'experiments', 'EXP-V3-01d-hashes.txt'), 'w',
             encoding='utf-8', newline='') as fh:
    fh.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
