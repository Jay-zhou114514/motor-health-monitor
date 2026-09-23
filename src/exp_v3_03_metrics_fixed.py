"""EXP-V3-03 指标修订：把单侧覆盖率改成**双侧带宽**并重算（不重跑实验）。

原指标 C1 = P(rate <= alpha) 会奖励"过度保守"的方法（例如把率压到 0）。
本脚本改为：
  B_tight = P(rate in [0.5a, 1.5a])     严格双侧
  B_wide  = P(rate in [0.25a, 2.0a])    宽双侧
  C2      = E|rate - alpha|  (pp)
  over    = P(rate == 0)                过度保守比例（单独报告）
数据：outputs/exp_v3_03_runs.csv（六种方法 × 三检测器 × 24 实例）
"""

from __future__ import annotations

import io
import os

import pandas as pd

ALPHA = 0.01
OUT = r'C:\Users\32597\Documents\Codex\2026-09-06\github\motor-health-monitor\outputs'


rows = pd.read_csv(os.path.join(OUT, 'exp_v3_03_runs.csv'))
records = []
for (det, method), g in rows.groupby(['detector', 'method']):
    r = g.rate.to_numpy(float)
    records.append({
        'detector': det,
        'method': method,
        'n': len(r),
        'B_tight': float(((r >= 0.5 * ALPHA) & (r <= 1.5 * ALPHA)).mean()),
        'B_wide': float(((r >= 0.25 * ALPHA) & (r <= 2.0 * ALPHA)).mean()),
        'mean_dev_pp': float(abs(r - ALPHA).mean() * 100),
        'sd_pp': float(r.std(ddof=1) * 100),
        'over_conservative': float((r == 0).mean()),
        'saturated': int(g.saturated.sum()),
    })

table = pd.DataFrame(records).sort_values(['detector', 'B_tight', 'mean_dev_pp'],
                                          ascending=[True, False, True])
table.to_csv(os.path.join(OUT, 'exp_v3_03_coverage_fixed.csv'), index=False)

pd.set_option('display.width', 200)
print('=== 双侧带宽下的校准表现（B_tight = 率落在 [0.5%, 1.5%] 的比例）===')
for det, sub in table.groupby('detector'):
    print(f'\n-- {det} --')
    print(sub[['method', 'B_tight', 'B_wide', 'mean_dev_pp', 'over_conservative', 'sd_pp']]
          .to_string(index=False, float_format=lambda x: f'{x:.3f}'))

print('\n=== 按 B_tight 的最佳方法（每检测器）===')
for det, sub in table.groupby('detector'):
    best = sub.iloc[0]
    base = sub[sub.method == 'M0_global'].iloc[0]
    print(f'  {det:<14} 最佳 {best.method:<18} B_tight={best.B_tight:.3f} '
          f'(M0 {base.B_tight:.3f}) | dev {best.mean_dev_pp:.2f} pp (M0 {base.mean_dev_pp:.2f})')
