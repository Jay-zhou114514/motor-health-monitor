# Scopus 文献扫描（2026-09-16）

- 目的：为"创新性是否足够"提供可核查的文献依据（此前项目只有 12 篇人工整理文献）
- 数据源：Scopus Search API（`content/search/scopus`）
- 检索字段：`TITLE-ABS-KEY(...)`（标题/摘要/关键词）
- 检索日期：2026-09-16
- 可复现性：查询式与命中数见 `scopus_counts.csv`；API key 不入库（存于本机非仓库路径）
- 局限：仅标题/摘要/关键词层面，未做全文筛选；Scopus 覆盖不等于全部文献

## 1. 文献地形（命中数）

### 1.1 拥挤区（已被大量研究）

| 检索式 | 命中 |
| --- | ---: |
| `TITLE-ABS-KEY("bearing fault diagnosis")` | 9,117 |
| `TITLE-ABS-KEY("negative results" AND "machine learning")` | 1,891 |
| `TITLE-ABS-KEY("leakage-safe" OR "leakage-aware")` | 652 |
| `TITLE-ABS-KEY("condition monitoring" AND "false alarm")` | 477 |
| `TITLE-ABS-KEY("statistical power" AND "machine learning")` | 482 |
| `TITLE-ABS-KEY("benchmark" AND "bearing fault")` | 417 |
| `TITLE-ABS-KEY("domain shift" AND bearing)` | 280 |
| `TITLE-ABS-KEY("Mahalanobis distance" AND "fault diagnosis")` | 248 |
| `TITLE-ABS-KEY("Mahalanobis" AND bearing)` | 221 |

### 1.2 中等覆盖（有工作，但未饱和）

| 检索式 | 命中 |
| --- | ---: |
| `TITLE-ABS-KEY("reproducibility" AND "fault diagnosis")` | 119 |
| `TITLE-ABS-KEY("underpowered" AND "machine learning")` | 81 |
| `TITLE-ABS-KEY("Ledoit-Wolf")` | 72 |
| `TITLE-ABS-KEY("power analysis" AND ("anomaly detection" OR "condition monitoring"))` | **55** |
| `TITLE-ABS-KEY("cross-dataset" AND bearing)` | 48 |
| `TITLE-ABS-KEY("data leakage" AND "fault diagnosis")` | 44 |
| `TITLE-ABS-KEY("preregistration" AND "machine learning")` | 28 |

### 1.3 稀疏区（近乎空白）

| 检索式 | 命中 |
| --- | ---: |
| `TITLE-ABS-KEY("window-level" AND "fault diagnosis")` | 14 |
| `TITLE-ABS-KEY("file-level" AND "fault diagnosis")` | 12 |
| `TITLE-ABS-KEY("collinearity" AND "anomaly detection")` | 9 |
| `TITLE-ABS-KEY("covariance" AND "shrinkage" AND "anomaly detection")` | 6 |
| `TITLE-ABS-KEY("evaluation pitfalls" AND "machine learning")` | 6 |
| **`TITLE-ABS-KEY("Mahalanobis" AND "shrinkage" AND bearing)`** | **1** |

## 2. 最接近的先行工作（按被引排序抽样）

### 2.1 `Mahalanobis + shrinkage + bearing`（唯一命中）

- 〔2026, 被引 0〕*Unsupervised performance degradation assessment of rolling
  bearings via time-constrained metric learning* —— 属度量学习路线，
  不是"协方差估计稳定性 / 误报机制"研究。

### 2.2 `covariance + shrinkage + anomaly detection`（6 篇，抽样）

- 〔2011, 42〕Sparse matrix transform for hyperspectral image processing
- 〔2014, 8〕Regularized block Toeplitz covariance matrix estimation
- 〔2024, 0〕A likelihood ratio test for shrinkage covariance estimators
- 〔2026, 0〕An unsupervised approach to anomaly detection in near-infrared
  spectroscopy via Covariance-Shrunk Slow Feature Analysis

→ **没有一篇属于旋转机械 / 工业振动监测领域**。

### 2.3 `Mahalanobis distance + fault diagnosis`（248 篇，被引最高的样本）

- 〔2006, 249〕EMD + AR 模型的滚动轴承故障诊断
- 〔2021, 234〕红外热成像 + 机器学习的感应电机轴承故障诊断
- 〔2020, 181〕基于 SVM 的非接触轴承故障诊断
- 〔2012, 169〕旋转机械红外热像智能诊断
- 〔2011, 162〕齿轮多故障诊断（小波-AR + PCA）

→ 该领域主流是"特征/深度模型做诊断"，马氏距离多作为工具而非研究对象。

## 3. 差距陈述（Gap Statement）

1. **技术交叉点几乎空白**：`Mahalanobis + shrinkage + bearing` 仅 1 篇，
   且不属于本主题；`covariance + shrinkage + anomaly detection` 的 6 篇
   不在工业振动领域。
2. **但"空白"不等于"有价值"**：收缩协方差是统计学的标准补救手段，
   该交叉点稀疏更可能说明"领域认为它不值得作为研究对象"。
   本项目 EXP-V1-07 的负结果（膨胀未跨批次复现）与该判断一致。
3. **真正薄弱且有需求的是评价维度**：
   - `power analysis` +（异常检测/状态监测）仅 **55** 篇；
   - `evaluation pitfalls` + 机器学习仅 **6** 篇；
   - `file-level` / `window-level` + 故障诊断仅 12 / 14 篇；
   - `preregistration` + 机器学习仅 **28** 篇。
4. 已有文献确认"数据泄漏 / 评价不可信"是真实问题
   （2026 MSSP 的现实评估论文；PHM 的 leakage-safe benchmark），
   但**把"功效不足 ≠ 被反驳"与多层级（window/file/批次）评价结合**的工作极少。

## 4. 创新性结论（据本次扫描）

| 声称 | 是否成立 | 依据 |
| --- | --- | --- |
| "首次用马氏距离做轴承诊断" | ❌ 不成立 | 248 篇 |
| "首次用收缩协方差做异常检测" | ❌ 不成立 | 技术成熟，跨领域 6 篇 |
| "首次关注该领域的评价可信度" | ❌ 不成立 | 泄漏/复现类 44 / 119 篇 |
| "把功效分析、批次级独立性与多层级评价结合用于低成本监测检测器" | ✅ **可能成立**（需全文精读确认） | 相关交叉点均在 6–55 篇量级，且未见直接组合 |
| "报告一个跨批次不复现的失效模式，并给出评价建议" | ✅ 可能成立（负结果 + 方法学） | `negative results` 有 1,891 篇但该细分稀疏 |

## 5. 对项目的直接含义

1. **放弃"几何失效 + 收缩修复"的技术叙事**（EXP-V1-07 已证伪其普遍性）。
2. **主线改为评价方法学**：
   > 在低成本状态监测中，检测器的失效模式必须经过跨批次/跨设备验证与功效分析才能声称成立；
   > 本文给出一个"看似显著但不复现"的实例，并提出可操作的评价清单。
3. **必须补的对比**：经典 ML 基线（One-Class SVM / Isolation Forest），
   否则无法回应"简单 vs 现代方法"的问题。
4. 下一步：对上述稀疏区（6–55 篇）做**全文精读与引用核对**，
   形成正式 Related Work 与引用列表（目前仅到标题/摘要层）。

## 6. 复现方式

```text
端点：https://api.elsevier.com/content/search/scopus
参数：query=TITLE-ABS-KEY(...)，count=1（取总数）或 count=5&sort=-citedby-count（取代表论文）
字段：dc:title, prism:coverDate, citedby-count, prism:doi
密钥：不入库；运行时通过环境变量传入
```
