# 新会话入口（NEXT SESSION BRIEF）

- 更新：2026-09-20
- **用途**：上下文用尽或换新会话时，把这一份文件交给新的代理/会话即可继续，无需依赖聊天记录。
- 使用方式：在新任务里只说一句——
  「读 `docs/plans/NEXT_SESSION_BRIEF.md`，然后按里面的"下一步"继续。」

## 0. 一句话状态

项目与实验**已完成**（EXP-V1-00 ~ V2-06，共 17 个）；论文 v1.0 稿**已完成并经过三轮模拟审稿的修正**；
现在处在"**投稿前收尾**"阶段，剩三件事（见 §4）。

## 1. 仓库与数据

| 用途 | 路径 |
| --- | --- |
| Master（唯一事实来源） | `C:\Users\32597\Documents\Codex\2026-09-06\github\motor-health-monitor` |
| 论文库（会议线/期刊线） | `C:\Users\32597\Documents\Codex\2026-09-15\github\motor-health-monitor-papers` |
| 数据根 | `E:\MotorHealthMonitorData`（paderborn 4.9 GB / pronostia 3.0 GB / xjtu_sy 11.4 GB） |
| 兜底 | `src/config.py` 的 `DATASET_ROOT` 在 E 盘缺失时**回退**到仓库内 `data/raw` |

**环境注意**：本机沙箱无法直接启进程（`windows elevated sandbox ... carveouts`），
所有 shell 命令都需要**提权执行**；GitHub 会间歇性断连，推送需重试。

## 2. 必须先读的规则文件（新会话务必遵守）

| 文件 | 作用 |
| --- | --- |
| `docs/FROZEN_PROTOCOL.md` | 数据/特征/检测器/阈值/选择协议，全量冻结 |
| `docs/REPRODUCIBILITY_PROTOCOL.md` | **L1 不变量 → L2 独立复算 → L3 确定性重跑**，未通过不得进结果审查 |
| `docs/EXPERIMENT_AFTERCARE.md` | **实验后置 8 步清单 + 15 条反模式**（含"必须区分计数与比率""L3 排除计时列""子代理结果必须落盘"） |
| `docs/plans/COMPLETENESS_AUDIT_v2.md` | **6 个审计维度**与"可投"门槛（D3 数字可复算 / D5 比较须同尺度） |
| `docs/plans/RESEARCH_PLAN_v1.7.md` | 当前计划：三层评价不确定性 + 风险登记 R1–R7 |
| `docs/plans/MAINLINE_DRIFT_AUDIT.md` | 与最初总目标的对照；A+C 决定（自采数据作展示项，不进论文） |

## 3. 论文现状（论文库 `conference-track/`）

- **稿件**：`PAPER_v1.0_submission-draft.md`（Abstract / Intro / Protocol / Results 含 §3.6 / Related Work / Discussion 5.1–5.5 / Limitations 8 条 / Conclusion / Fig.1 图注 / 数据集引用）
- **主张-证据映射**：`CLAIM_EVIDENCE_MAP.md`（C1–C10 + 禁写清单）
- **引用**：`REFERENCES.md`（19 条，已 Crossref/arXiv 核实）＋ `CITATION_AUDIT.md`
- **图**：`figures/fig1_three_faces.{png,pdf,svg}` + 三份 QA 报告（碰撞 69 → 0）
- **审稿记录**：`REVIEW_round1_reviewer_scope.md`、`REVIEW_round2_serial_reviewerA.md`、`REVIEW_round3_reviewerB.md`

**已处理的三轮审稿**：S-M1~M4 + R1-M1~M8 + RB-M1~M3（Major 共 21 条）＋ 29 条 Minor。

## 4. 下一步（按优先级，三件事）

1. **跑第 4 位审稿人（串行）**——验证三轮修正是否收敛、是否还有新问题。
   **关键操作**：一次只启动**一个**子代理；**要求它把报告写入
   `conference-track/REVIEW_round4_reviewerC.md`**，并在同一回合确认文件存在
   （反模式第 15 条：首个 reviewerB 因未落盘而丢失）。
   侧重建议：**报告完整性与读者可复现性**（前三轮分别覆盖了 scope、统计、内部效度）。
2. **语言润色**（`nature-polishing` skill）——正文已多轮改动，现在才适合润色。
3. **投稿会议决定**——PHM Europe / PHM Conference / ICPHM；并决定是否等 V2 跨体系补齐。

## 5. 已知且已接受的缺口（不要再重复披露，也不要试图关闭）

- V1-00~04、V2-01~03 为 **pre-protocol**，未做三层验证（已声明）；
- V1-05~09 有 L1+L2 与哈希基线，**未做 L3 重跑**；
- **MFPT 官方著录未确认**（官方页未提供、MathWorks 403）→ 用分发方脚注；
- **中文文献与 PHM 会议集未覆盖** → 已在检索范围声明中写明；
- **无自采数据**（A+C 决定：自采作红鸟展示项，不进本论文）。

## 6. 有效的方法学结论（供新会话快速校准）

1. **口径决定结论**：within-bearing 0.00% vs bearing-level holdout **双峰**（三折 ≤1.55%、三折 ≥69.40%，均值 40.63%）；
2. **单元数效应**：工况内固定 20 条记录，k=1 时 SD 23.9–48.2 pp，k_max 时显著下降；3σ RMS 与 iForest 各 6/6，**马氏饱和（66.8–100%）不可检验**；
3. **健康阶段定义是自由参数**：PRONOSTIA 极差 **2.40 pp**（干净）、XJTU-SY **5.19 pp**（含纳入与单元数混杂）；
4. **不能声称**：新检测器、新的退化起点检测法、跨设备泛化、"分析者选择比检测器更重要"（无同尺度测量）。