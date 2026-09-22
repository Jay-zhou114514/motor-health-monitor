# 串行队列：等待批次⑤（pid 3144）→ L3 重跑与比对 → EXP-V3-03 → 写总记录
$ErrorActionPreference = 'Continue'
$m = 'C:\Users\32597\Documents\Codex\2026-09-06\github\motor-health-monitor'
$out = Join-Path $m 'outputs'
$log = Join-Path $out 'CHAIN_v301e_l3_v303.log'

function W($t) { $line = "$(Get-Date -Format 'HH:mm:ss')  $t"; $line | Out-File -FilePath $log -Append -Encoding utf8 }

W 'chain started; waiting for batch 5 (pid 3144)'
Wait-Process -Id 3144 -ErrorAction SilentlyContinue
W 'batch 5 finished'

# 1) L3 rerun of EXP-V3-01d
Push-Location (Join-Path $m 'src')
W 'L3 rerun start (exp_v3_01d_paderborn_fixed.py)'
python exp_v3_01d_paderborn_fixed.py *> (Join-Path $out 'L3_v301d_rerun.log')
W "L3 rerun exit=$LASTEXITCODE"

# 2) L3 comparison + hash record
python l3_compare_v301d.py *> (Join-Path $out 'L3_v301d_compare.txt')
W "L3 compare exit=$LASTEXITCODE"

# 3) EXP-V3-03 coverage hierarchy
W 'EXP-V3-03 start'
python exp_v3_03_coverage_hierarchy.py *> (Join-Path $out 'exp_v3_03_console.log')
W "EXP-V3-03 exit=$LASTEXITCODE"
Pop-Location

W 'chain finished'
