# 新会话入口（NEXT SESSION BRIEF）

- 更新：2026-09-20
- **用途**：上下文用尽或换新会话时，把这一份文件交给新的代理/会话即可继续，无需依赖聊天记录。
- 使用方式：在新任务里只说一句——
  「读 `docs/plans/NEXT_SESSION_BRIEF.md`，然后按里面的"下一步"继续。」

## 0. 一句话状态

项目与实验**已完成**（EXP-V1-00 ~ V2-06，共 17 个）；论文 v1.0 稿**已完成并经过四轮模拟审稿的修正**（第 4 轮独立报告于 2026-09-20 完成，并暴露出一个待人工裁决的口径问题，见 §4）；
现在处在"**投稿前收尾**"阶段：第 4 轮审稿已完成，剩两件需人工决定的事（见 §4）。

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

## 4. 下一步（按优先级）

1. ~~跑第 4 位审稿人（串行）~~ **已完成（2026-09-20）**。
   - **独立报告**：`conference-track/REVIEW_round4_reviewerC.md`（46 KB，Major 4 / Minor 10，
     侧重报告完整性与读者可复现性）。子代理的消息通道本次**仍然失效**（4 个实例全部没收到
     任务文本），但**孙代理从落盘的任务包 `REVIEW_round4_TASK_C.md` 恢复了任务**并完成审稿
     ——反模式第 15 条的"任务包落盘"生效了。主代理的自检版另存为
     `REVIEW_round4_selfcheck_primary.md`。
   - **RC-M1（最重要，已复算证实）**：§3.2 的 SD 从未定义；归档 `sd_fp` 是"折 × 重复"池化
     SD，改用 §3.1 的折间 SD 后判据翻转：iForest 6/6→**2/6**、3σ RMS 6/6→**4/6**、
     马氏 2/6→**0/6**（`REVIEW_round4_sd_definition_check{,2}.py/.txt`）。稿件已补口径与两套
     读数，但**"EXP-V2-05 的 Y2 是否改判"是需要人工裁决的假说判定问题**。
   - **RC-M2（已证实）**：EXP-V1-11/V2-01/V2-02 并无 L1+L2/哈希基线记录
     （`verify_paper_experiments.py` 只覆盖 V1-05~V1-10）；稿件已改为如实声明。
   - RC-M3 / RC-M4 与 §5.5 残句已按最小改法修正。

2. **补两处归档产物**（第 4 轮 RC-M3 / RC-m2 的残留，需要写代码或补列）：
   - 把五条健康阶段规则的 SD(k_max) 以**带规则标签的 CSV** 落盘（当前只有散文表格）；
   - 在 `EXP-V1-10` 臂级汇总里补 zero-minimum 占比列，并给复制文件加 arm 列
     （现在 R=50 与 R=200 两臂无法被第三方分离）。
3. **语言润色**——注意：brief 原先引用的 `nature-polishing` skill **在本机并不存在**
   （`~/.codex/skills` 下无该目录）。可选替代：① 新安装的 `humanizer` / `humanizer-zh`
   （去 AI 写作痕迹，偏通用散文）；② `nature-writing`、`research-paper-writing`
   （偏学术结构，非纯语言）；③ 先安装 `nature-polishing` 再润色。**需人工选定后再动手**。
4. **投稿会议决定**——PHM Europe / PHM Conference / ICPHM；并决定是否等 V2 跨体系补齐。
   这会决定篇幅上限与润色力度，建议先定这个再润色。
5. **待办**：本会话已提交并推送 4 个 commit（`ce1e81a`、`80e1f5e`、`0745b8d`、`4403120`）；
   2026-09-20 两次 `git push` 均因 GitHub 连接被重置而失败，需稍后重推。


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