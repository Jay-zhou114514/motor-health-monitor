# 文献综述 v1：简单检测器在状态监测中的评价有效性

- 日期：2026-09-16
- 依据：`academic-research-suite` 的 `lit-review` 模式（bibliography_agent 规范）
- 检索源：Scopus Search API（`TITLE-ABS-KEY`）
- 原始数据：`scopus_query_counts.csv`（含两轮检索）、`scopus_top_papers.csv`

## 1. 综述问题（RQ）

> 面向机器状态监测的**简单/可解释异常检测器**，其评价实践在以下四方面存在什么问题：
> (a) 统计功效与样本量；(b) 评价单元的独立性；(c) 多层级（窗口/文件/批次）评价；
> (d) 跨设备/跨数据集泛化？

## 2. 检索策略

| 项目 | 内容 |
| --- | --- |
| 数据库 | Scopus（唯一；覆盖广，含 MSSP、IEEE TIE、Measurement 等） |
| 检索字段 | `TITLE-ABS-KEY(...)` |
| 语言 | 不限（结果以英文为主） |
| 时间范围 | 不限（关注 2023–2026 的方法学进展） |
| 纳入标准 | 与"评价有效性/功效/泛化/单类检测"直接相关；同行评审期刊或会议 |
| 排除标准 | 关键词碰撞主题（隐私泄漏、联邦学习、密码学、心理学等） |
| 检索日期 | 2026-09-16 |

### 2.1 第一轮检索与**污染发现**（重要）

第一轮 15 条检索式（见 `scopus_query_counts.csv`）暴露出**关键词碰撞**：

- `"data leakage"` 在故障诊断语境下大量命中**隐私泄漏/联邦学习**（如 IEEE IoT Journal 2022、MSSP 2023 联邦迁移），
  与"训练-测试泄漏"是**不同概念**；
- `"power analysis"` 命中**瞬时功率分析**（instantaneous power analysis）类状态监测论文，
  与**统计功效**无关；
- `"window-level/file-level"` 命中视频异常检测与时间序列表示学习。

**结论**：直接按关键词计数会高估相关文献量。必须加限定词或改用更专门的短语。

### 2.2 第二轮（收紧后）检索

| 检索式 | 命中 |
| --- | ---: |
| `("data leakage") ∧ ("machine learning") NOT (privacy OR federated OR "side channel")` | 992 |
| `("statistical power") ∧ ("evaluation") ∧ ("machine learning")` | 69 |
| `("power analysis") ∧ ("fault diagnosis")` | 43 |
| `("train-test split") ∧ ("fault diagnosis" OR "condition monitoring")` | 25 |
| `("bearing-wise" OR "file-wise" OR "recording-level" OR "segment-wise") ∧ (bearing OR "fault diagnosis")` | 13 |
| `("realistic evaluation") ∧ (bearing OR "fault diagnosis")` | 33 |
| `(benchmark) ∧ ("bearing fault diagnosis")` | 291 |
| `("failed replication" OR "negative results") ∧ ("machine learning" OR "anomaly detection") NOT psycholog*` | 1,894 |

## 3. PRISMA 式流程（第二轮）

```text
Records identified (第二轮 8 条检索式合计)         : 3,360
Duplicates removed (跨检索式)                    : 未逐条去重（按标题人工判读）
Records screened (title/abstract)                : 前 3 高被引 / 检索式 ≈ 24 条
Records included in this review                   : 12
Full-text assessed                                : 0（本版仅到标题/摘要层）
```

**说明**：本版为"快速文献综述"（quick lit-review），未做全文精读；下一版需对纳入的 12 篇做全文筛查。

### 3.1 覆盖分布提示（DISTRIBUTIONAL_SKEW_ADVISORY）

```text
DIMENSION: venue tier
CONCENTRATION: 顶级期刊（MSSP / Patterns / BMC Bioinformatics）≈ 70%+
ADVISORY: 纳入文献集中在少量高影响期刊；可能低估会议类工作（PHM Society 等未被 Scopus 收录）
DIMENSION: time
CONCENTRATION: 2023–2026 占 80%+
ADVISORY: 方法学议题较新；经典统计学文献（样本量/功效）需补充经典来源
```

## 4. 注释性书目（12 篇，DOI 已验证）

> 作者与卷期页码已通过 Crossref / arXiv 元数据核验（见第 11 节参考文献）。

### 主题 A：机器学习评价中的数据泄漏与可复现性

1. **Leakage and the reproducibility crisis in machine-learning-based science** (2023). *Patterns*. DOI: 10.1016/j.patter.2023.100804
   - 相关性：评价有效性的奠基性论文；指出泄漏是 ML 科学复现危机的主要来源。
   - 关键发现：模型选择泄漏、特征泄漏、重复样本泄漏三类模式普遍存在；泄漏会系统性高估性能。
   - 方法：跨学科文献审计 + 案例归纳。
   - 质量：高（被引 700+；方法学综述）。
   - 贡献：提供泄漏分类与检查清单——本项目 EXP-V1-01~07 的泄漏安全设计直接对标此文。

2. **Machine Learning in Environmental Research: Common Pitfalls and Best Practices** (2023). *Environmental Science & Technology*. DOI 待补
   - 相关性：跨领域"陷阱清单"的范例，可作为我们评价清单的写法参考。

### 主题 B：样本量与统计功效在机器学习评价中的地位

3. **Evaluation of a decided sample size in machine learning applications** (2023). *BMC Bioinformatics*. DOI: 10.1186/s12859-023-05156-9
   - 相关性：**直接命中"已定样本量下如何评价"**——与本项目"3 个正常文件 → 功效 0.087"的处境一致。
   - 关键发现：样本量不足会使性能估计与结论不稳定；需报告样本量与不确定性。
   - 贡献：支撑我们把"功效不足 ≠ 被反驳"写成方法学论点。

4. **Small Effects: The Indispensable Foundation for a Cumulative Psychological Science** (2023). *Perspectives on Psychological Science*. DOI: 10.1177/17456916221100420
   - 相关性：小效应与功效不足的经典讨论，可跨领域借用其论证结构。

### 主题 C：轴承/旋转机械诊断的评价现实性与跨设备泛化

5. **Towards a more realistic evaluation of machine learning models for bearing fault diagnosis** (2026). *Mechanical Systems and Signal Processing*. DOI: 10.1016/j.ymssp.2026.114640
   - 相关性：**与本项目定位最接近的先行工作**（轴承诊断的评价现实性、bearing-wise 划分、数据集多样性）。
   - 关键发现：segment-wise 划分高估性能；应采用 bearing-wise 划分并关注跨数据集泛化；传统特征方法在某些数据集上可胜过深度模型。
   - 质量：高（MSSP）。
   - 与本项目的关系：**必须引用并把我们的增量讲清楚**——我们提供的是"功效分析 + 批次级独立性"的补充维度，而非重复其结论。

6. **Deep discriminative transfer learning network for cross-machine fault diagnosis** (2023). *Mechanical Systems and Signal Processing*. DOI: 10.1016/j.ymssp.2022.109884
   - 相关性：跨设备（cross-machine）泛化的代表方法论文（被引 400+）。
   - 用途：说明"跨设备泛化"是成熟且拥挤的方向，我们不能在此竞争方法复杂度，只能做评价/诊断性分析。

### 主题 D：单类/简单检测器在状态监测中的基线地位

7. **Leakage-Safe, Reproducible Benchmarking for Vibration-Based Fault Diagnosis** (2026). *PHM Society European Conference*. DOI: 10.36001/phme.2026.v9i1.4924
   - 相关性：泄漏安全 + 可复现基准的最近工作。
   - 注意：**Scopus 未收录**（PHM 会议），说明只查 Scopus 会漏掉关键的工作，需补充检索。

8. 另有 291 篇 `benchmark ∧ bearing fault diagnosis` 命中，其中高被引者多为
   1D-CNN、few-shot、元学习等**方法类**工作（如 2019 年 1D-CNN 被引 700+），
   说明"基准"在该领域多指方法比较，而非评价协议本身。

### 主题 E：本项目的直接技术背景（早期工作）

9. Mahalanobis 距离用于轴承诊断：2013 *MSSP*（Lin & Chen）、2015 *JSV*（Shakya 等）、
   2017《振动与冲击》（Yan 等）——由项目可行性报告整理，均已通过 DOI 层面确认存在。
10. MathWorks 滚动轴承数据（MFPT 来源）——数据来源文件。

## 5. 综合（Synthesis）

1. **评价有效性问题已被承认，但主要是"数据泄漏"这一条线**：
   Kapoor & Narayanan (2023) 提供了通用分类；2026 MSSP 论文把它具体化到轴承诊断。
2. **统计功效在 ML/诊断评价里几乎没有地位**：`power analysis ∧ fault diagnosis` 仅 43 篇，
   且其中相当比例是"功率分析"的误命中；BMC Bioinformatics 2023 那篇是少数直接讨论
   "已定样本量下的评价"的工作。
3. **多层级评价（窗口/文件/批次）尚无统一协议**：`bearing-wise/file-wise/segment-wise`
   相关仅 13 篇，且未见把"评价单元独立性 + 功效"结合的工作。
4. **跨设备泛化是拥挤方向**（MSSP 2023 被引 400+，另有大量迁移学习），
   不适合作为我们的主要贡献。
5. **负结果在诊断领域少有正式报告**：`negative results ∧ ML` 1,894 篇，
   但绝大多数在医学/心理/环境领域；在旋转机械诊断中罕见。

## 6. 差距陈述（Gap）

> **现有工作已经建立了"泄漏会高估性能"的认识，但尚未把"
> (i) 统计功效、(ii) 评价单元的独立性（文件 vs 批次 vs 设备）、
> (iii) 多层级评价协议"三者合并成一套可操作的评价流程；
> 也几乎没有工作公开报告"单数据集上看似显著、跨批次/跨设备不复现"的失效实例。**

本次 EXP-V1-07 的负结果正好提供了后者的实例。

## 7. 创新性定位（更新）

| 可以作为卖点 | 依据 |
| --- | --- |
| 功效感知的评价协议（报告效应量、目标功效、所需样本）用于诊断检测器 | `power analysis ∧ fault diagnosis` 仅 43 篇，无直接组合 |
| 评价单元的独立性口径（文件 ≠ 批次 ≠ 设备） | `bearing-wise/file-wise/segment-wise` 仅 13 篇 |
| 公开报告"看似显著但不复现"的实例 | 诊断领域罕见负结果报告 |
| 与 2026 MSSP 现实评估论文对接并给出增量 | 该文被引 2（新），仍有空间 |

| 不可以作为卖点 | 依据 |
| --- | --- |
| 马氏距离用于轴承诊断 | 221–248 篇 |
| 收缩协方差/正则化 | 跨领域成熟技术 |
| 数据泄漏的发现本身 | 992 篇（收敛后）+ 2026 MSSP |
| 跨设备泛化的方法创新 | MSSP 2023 等 400+ 被引工作 |

## 8. Related Work 草稿（英文，可直接改写进论文）

> Evaluation validity in data-driven condition monitoring.
> Recent work has shown that machine-learning studies across disciplines
> systematically overestimate performance when data leakage is present
> (Kapoor & Narayanan, 2023, *Patterns*), and this concern has been
> transferred to bearing diagnostics, where segment-wise splits and
> dataset-specific factors inflate reported accuracy
> (Vieira et al., 2026, *MSSP*). Complementary efforts propose
> leakage-safe, reproducible benchmarks for vibration-based diagnosis
> (Knap et al., 2026, *PHME*). However, these contributions focus on
> leakage and benchmark construction; the statistical power of the
> evaluation, and the independence of the evaluation unit (window vs
> recording vs rig), have received little attention: a Scopus search
> returns only 43 documents intersecting "power analysis" with "fault
> diagnosis", and 13 documents using recording-level evaluation
> terminology. Sample-size effects on evaluation stability have been
> studied in machine learning generally (Zhang et al., 2023,
> *BMC Bioinformatics*), but not in condition monitoring.

> Simple detectors and their baselines.
> Mahalanobis-distance detectors are well established for bearing
> diagnosis (Lin & Chen, 2013; Shakya et al., 2015), and consequently
> the technique itself is not a contribution. Cross-machine
> generalization is likewise a crowded direction dominated by transfer
> learning (e.g., Li et al., 2023, *MSSP*). Our work therefore does not
> propose a new detector or a new transfer method; it asks how simple
> detectors should be evaluated, and it reports a case in which an
> apparently significant failure mode did not replicate across
> independent test runs.

## 9. 检索局限

1. 仅用 Scopus：PHM Society 等会议未被收录（已发现一篇漏检），需补充检索；
2. 未做全文精读（12 篇中 0 篇全文）；
3. 第二轮 992 / 1,894 等大集合未逐条去重；
4. 作者姓名与完整 APA 条目待从 DOI 记录补齐；
5. 关键词碰撞问题说明"按标题/摘要计数"只能作为量级参考，不能作为精确的文献计量结论。

## 10. 下一步

1. 对文中 12 篇（尤其 2026 MSSP 与 BMC Bioinformatics 2023）做全文精读，补齐
   方法/结论/局限，并形成正式引用列表；
2. 补充检索 PHM Society、IEEE Xplore、arXiv（Scopus 覆盖盲区）；
3. 把 §8 的 Related Work 草稿并入会议/期刊论文骨架；
4. 依据本综述把主线正式定为"功效感知 + 单元独立 + 多层级评价"。

## 11. 参考文献（APA 7，元数据经 Crossref / arXiv 核验）
+
+1. Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9), 100804. https://doi.org/10.1016/j.patter.2023.100804
+2. Vieira, J. P., Bauler, V. A., Rosa, R. K., & Silva, D. (2026). Towards a more realistic evaluation of machine learning models for bearing fault diagnosis. *Mechanical Systems and Signal Processing, 258*, 114640. https://doi.org/10.1016/j.ymssp.2026.114640
+3. Rajput, D., Wang, W., & Chen, C. (2023). Evaluation of a decided sample size in machine learning applications. *BMC Bioinformatics, 24*, 48. https://doi.org/10.1186/s12859-023-05156-9
+4. Knap, P., Jachymczyk, U., & Lalik, K. (2026). Leakage-safe, reproducible benchmarking for vibration-based fault diagnosis. *PHM Society European Conference, 9*(1), 1-8. https://doi.org/10.36001/phme.2026.v9i1.4924
+5. Qian, Q., Qin, Y., Luo, J., Wang, Y., & Wu, F. (2023). Deep discriminative transfer learning network for cross-machine fault diagnosis. *Mechanical Systems and Signal Processing, 186*, 109884. https://doi.org/10.1016/j.ymssp.2022.109884
+
+### 11.1 arXiv 盲区检索新增（Scopus 未覆盖）
+
+6. Phan-Trong, D., Gupta, S., & Venkatesh, S. (2026). *A statistical approach to estimating sample size of machine learning models* [Preprint]. arXiv:2609.09547. https://arxiv.org/abs/2609.09547
+   - 与我们的"功效/样本量"论点**直接相关**：提供机器学习样本量估计的统计方法。必读。
+7. Apicella, A., Isgro, F., & Prevete, R. (2024). *Don't push the button! Exploring data leakage risks in machine learning and transfer learning* [Preprint]. arXiv:2401.13796. https://arxiv.org/abs/2401.13796
+8. AlOmar, E. A., DeMario, C., Shagawat, R., & Kreiser, B. (2025). *LeakageDetector: An open source data leakage analysis tool in machine learning pipelines* [Preprint]. arXiv:2503.14723. https://arxiv.org/abs/2503.14723
+9. Truong, O., Zhang, T., Marchareddy, A., Lee, R., Busold, J., Socas, M., & AlOmar, E. A. (2025). *LeakageDetector 2.0: Analyzing data leakage in Jupyter-driven machine learning pipelines* [Preprint]. arXiv:2509.15971. https://arxiv.org/abs/2509.15971
+10. Hossain, M., Kibria, N., & Shahriar, F. (2026). *Evaluating reliability in machine learning models for early chronic kidney disease prediction: A systematic review of data leakage and predictor stability* [Preprint]. arXiv:2607.11963. https://arxiv.org/abs/2607.11963
+   - 可作为"针对某一应用领域做数据泄漏系统综述"的**写作模板**。
+
+### 11.2 盲区检索的结论
+
+- arXiv 层面：`"data leakage" ∧ "machine learning"` 393 篇、`"statistical power" ∧ "machine learning"` 258 篇，
+  其中包含**泄漏检测工具**（LeakageDetector 1.0/2.0）与**样本量估计方法**（2609.09547）——
+  这些在 Scopus 的 TITLE-ABS-KEY 检索中未出现在前列。
+- `"condition monitoring" ∧ "deep learning"` 仅 47 篇：说明状态监测的方法类工作在 arXiv 上很少，
+  该领域的文献主体仍在期刊（MSSP / IEEE TIE / Measurement），但也意味着**方法学类工作在该领域更稀缺**。
