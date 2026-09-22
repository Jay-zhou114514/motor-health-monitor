# 关机交接单（2026-09-22）

- 场景：用户**现在要关机**，重启后继续。本文件**取代** `PAUSE_2026-09-22.md` 中"正在运行"一节。
- 新会话使用方式：让代理先读本文件（以及 `PAUSE_2026-09-22.md` 的 §3–§7）。

## 1. 关机时的状态（重要）

| 进程 | 关机前状态 | 关机后 |
| --- | --- | --- |
| `pid 3144` — 批次⑤（正确的 P2） | 运行中（CPU ≈2555 s，进度约 K003–K004/24 实例） | **会被关机终止**；已有检查点 `outputs/exp_v3_01e_partial.csv` |
| `pid 20124` — 串行队列（→ L3 重跑 → EXP-V3-03） | 仍在等待 3144 | **会被终止**；队列从未开始后续步骤 |

**因此：**

- `outputs/exp_v3_01d_*.csv`（批次④的完整结果）**未被覆盖，保持原样**（L3 重跑尚未开始）；
- 重跑前快照仍在：`outputs/L3_v301d_before_exp_v3_01d_{runs,summary,partial_P1_strict}.csv`（可保留，日后 L3 仍可用）；
- 批次⑤ 目前只有**部分结果**（`exp_v3_01e_partial.csv`），**H6 尚未判定**。

## 2. 重启后的推荐恢复顺序（直接命令）

工作目录一律用**绝对路径**；沙箱无法启动进程，所有 shell 命令需 `sandbox_permissions: "require_escalated"`。

```powershell
$m = 'C:\Users\32597\Documents\Codex\2026-09-06\github\motor-health-monitor'
cd "$m\src"

# ① 批次⑤（正确 P2 → H6）；约 50–70 分钟，逐实例写检查点
python exp_v3_01e_p2_corrected.py

# ② L3：重跑 EXP-V3-01d 并与快照比对（快照已在 outputs/L3_v301d_before_*）
python exp_v3_01d_paderborn_fixed.py          # 约 1.5–2 小时
python l3_compare_v301d.py                    # 生成 experiments/EXP-V3-01d-hashes.txt
python verify_exp_v3_01d.py                   # L1/L2（集合断言、λ 网格、窗口数、独立复算）

# ③ EXP-V3-03 覆盖层级（M0–M5，约 1–2 小时）
python exp_v3_03_coverage_hierarchy.py
```

**注意**：`tools/run_chain_after_01e.ps1` 里用 `Wait-Process -Id 3144`，**关机重启后该 pid 可能被别的进程占用**，
所以**不要再直接跑那个脚本**；按上面三条直接顺序执行更安全（或改脚本去掉等待）。

## 3. 恢复后要读的三份结果

| 文件 | 回答什么 |
| --- | --- |
| `outputs/exp_v3_01e_summary.csv` | **H6**：P1 与正确 P2 的降幅差 ≤20 pp？ |
| `experiments/EXP-V3-01d-hashes.txt` | **L3**：EXP-V3-01d 是否逐字节复现（未过不得进入结果审查） |
| `outputs/exp_v3_03_coverage.csv` | **H11–H14**：M0–M5 的单元级覆盖率（Wilson CI） |

## 4. 已确立、不要重做的部分

- **方法路线已终止**（H5 0/3、H10 0/3，批次④在"修正准则 + 620 窗分辨率 + 20 训练实例"下）；
  批次①②③的部分结论**已撤回**，理由写在 `experiments/EXP-V3-01d-verdict-addendum.md`；
- **措辞修正（必须遵守）**：禁止写"阈值校准救不了单元级不确定性"；
  改写为 **"仅靠全局阈值校准，一般不能保证异质或分布漂移下的单元级条件覆盖率"**；
  "口径依赖" → **"覆盖率保证的层级依赖性"**（marginal → group/unit-level → conditional）；
- 预注册已冻结 8 份：`EXP-V3-01`（原版 + 修订 1–4）、`EXP-V3-02`、`EXP-V3-03`、`EXP-V3-04`、`EXP-V3-05`。

## 5. 未决事项（重启后先定）

1. **投稿会场**：ICPHM（EI，1 月）/ IJPHM（Scopus，免费滚动）/ MDPI Sensors（SCI Q2，有 APC）/
   QR2MSE-ICRMS（EI，春季）；CCDC 已评估为最差匹配（25–35%）。
2. **EXP-V3-05 多分析者实验**：需用户招 5–10 位参与者；材料（brief / 数据打包 / 提交模板 / 分析脚本）**尚未编写**。
3. **EXP-V3-04 报告缺口审计**：已移交独立任务（threadId `01a0c961-962f-7ae2-8234-632821d66f1d`），
   走用户已登录的浏览器；本机对境外 API/站点网络受限（OpenAlex 拉取已实测失败）。
4. **可选增强**：扩独立单元（Ottawa 等）、消融 A4/A6。

## 6. 第一件事（重启后）

按 §2 跑三条命令（或从 ① 开始），跑完按 §3 读三份结果 → 更新判定附录 → 写 EXP-V3-05 材料 →
按"覆盖层级 + 缺口审计 + 多分析者"重写稿件。
