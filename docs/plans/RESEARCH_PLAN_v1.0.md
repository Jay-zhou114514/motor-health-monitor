# Motor Health Monitor｜阶段性科研计划

## 一、总目标

将 `motor-health-monitor` 从一个真实数据驱动的轴承故障检测项目，逐步发展为：

**可复现、可解释、经过严格验证、具有跨数据集泛化证据，并可由独立真实实验验证的论文级研究项目。**

最终路线：

```text
阶段 0
建立科研项目基础
        ↓
阶段 1
V1 正确性与实验审计
        ↓
阶段 2
V1 严谨实验
        ↓
阶段 3
形成第一版研究问题
        ↓
阶段 4
Cross-Dataset 泛化
        ↓
阶段 5
Failure Analysis
        ↓
阶段 6
个人真实实验
        ↓
阶段 7
最终实验整合
        ↓
阶段 8
论文写作
        ↓
阶段 9
投稿准备
        ↓
阶段 10
可选：ESP32 实时部署
```

---

# 阶段 0：建立科研项目基础

### 时间
3–7 天

### 目标
把 GitHub 项目从“代码仓库”升级成“研究项目”。

### 建议建立
- `PROJECT_CHARTER.md`
- `PROJECT_STATUS.md`
- `WORK_PROTOCOL.md`
- `EXPERIMENT_REGISTRY.md`

并建立：
```text
experiments/
results/
figures/
reports/
paper/
```

### 核心内容
明确：
- Research Question
- Hypothesis
- Experiment
- Evidence
- Conclusion
- Limitation

### 阶段产出
一个明确研究问题 + 一个实验路线图 + 一个实验记录系统。

### 完成标准
能够清楚回答：

> 我到底想研究什么？

---

# 阶段 1：V1 正确性审计

### 时间
1–2 周

### 目标
确认当前 V1 结果可信，而不是单纯追求更高分。

### 检查
```text
代码逻辑
数据读取
Train/Test split
Label alignment
NaN
Windowing
Feature extraction
Prediction
Evaluation
Plotting
```

重点：
```text
False Alarm
False Negative
Data Leakage
```

### 正确的数据流程
```text
训练数据
→ 建模
→ 确定 threshold
→ 冻结

测试数据
→ prediction
→ evaluation
```

### 阶段产出
`V1 Audit Report`

内容包括：
- Bug
- Data leakage risk
- Evaluation risk
- Reproducibility risk

### 完成标准
**我们敢相信当前 V1 的结果。**

---

# 阶段 2：V1 严谨实验

### 时间
2–4 周

## A. Window-level
输出：
- Accuracy
- Precision
- Recall
- F1
- FPR
- FNR
- Confusion Matrix

## B. File-level
增加：
- Detection Rate
- False Alarm Rate
- Missed Detection Rate

## C. Threshold Sensitivity

3-Sigma：
- 2σ
- 2.5σ
- 3σ
- 3.5σ
- 4σ

Mahalanobis：
- 95%
- 97.5%
- 99%
- 99.5%
- 99.9%

重点观察：
- Precision
- Recall
- F1
- FPR
- FNR

目标不是找到“最高分阈值”，而是寻找**稳定性能区域**。

## D. Feature Ablation
逐步测试：
- RMS
- RMS + Kurtosis
- RMS + Crest Factor
- RMS + Centroid
- All Features

## E. Error Analysis
寻找：
- False Positive
- False Negative
- Hard cases
- Easy cases

### 阶段产出
`V1 Experimental Report`

包括：
- Tables
- Figures
- Error Analysis

### 完成标准
能够回答：

> 当前 V1 到底有多可靠？

---

# 阶段 3：确定论文研究问题

### 时间
1–2 周

### 原则
不要提前假设某个方法一定最好。

根据阶段 1–2 的真实结果决定论文故事。

可能的问题：

```text
简单方法在同数据集很强，
但跨数据集明显下降。
```

→ 研究 Domain Shift / Generalization

或者：

```text
Window-level 很高，
但 File-level 明显下降。
```

→ 研究 Window-level Evaluation 是否高估真实检测能力。

### 阶段产出
- Research Question
- Hypothesis
- Contribution candidates
- Paper story

### 完成标准
能够用 3–5 句话讲清楚：

> 这个项目为什么值得成为一篇论文？

---

# 阶段 4：Cross-Dataset Generalization

### 时间
3–5 周

### 目标
测试：
```text
Dataset A
→ Train
→ Frozen Model
→ Dataset B
→ Test
```

优先考虑：
- MFPT
- NASA IMS

### 核心比较
```text
Same Dataset
vs
Cross Dataset
```

### 重点记录
- Performance drop
- Feature distribution shift
- Threshold transferability
- False alarms
- Missed detection
- Method stability

### 阶段产出
- Cross-dataset experiment
- Distribution plots
- Performance comparison
- Initial explanation

### 完成标准
回答：

> 这个方法换到没见过的数据上，还可靠吗？

---

# 阶段 5：Failure Analysis

### 时间
2–3 周

### 目标
不仅研究“为什么成功”，还主动研究：

> **什么时候失败？为什么失败？**

### 重点变量
- Load
- Noise
- Sampling rate
- Amplitude
- Sensor condition
- Machine condition
- Feature distribution
- Threshold

### 分析流程
```text
Failure
↓
Locate
↓
Characterize
↓
Hypothesis
↓
Experiment
↓
Explanation
```

### 阶段产出
- Failure taxonomy
- Failure examples
- Cause analysis

### 完成标准
能够解释至少一部分：

> 为什么模型会失败。

---

# 阶段 6：个人真实实验

### 时间
3–6 周

### 目标
建立独立于公开 benchmark 的外部验证。

### 核心原则
先冻结：
- Model
- Feature set
- Threshold
- Evaluation protocol

然后：
```text
你的设备
↓
你的传感器
↓
你的采样
↓
你的数据
↓
External Test
```

### 最低可行版本
可以考虑：
- 电机 / 旋转设备
- 加速度传感器
- 健康状态
- 若干可控异常状态

记录：
- Speed
- Load
- Sampling Rate
- Sensor Position
- Bearing / Machine
- Condition
- Timestamp

### 阶段产出
- Independent Experimental Dataset
- Experimental Protocol
- External Validation Results

### 完成标准
能够说：

> 这个方法不是只在别人公开的数据上有效，我还在独立实验条件下对它进行了验证。

---

# 阶段 7：最终实验整合

### 时间
2–4 周

最终实验矩阵：

```text
EXP-01
Baseline

EXP-02
Leakage-safe evaluation

EXP-03
File-level evaluation

EXP-04
Threshold sensitivity

EXP-05
Feature ablation

EXP-06
Classical ML baselines

EXP-07
Cross-dataset

EXP-08
Failure analysis

EXP-09
Independent real experiment
```

可选：
`EXP-10 Run-to-failure`

### 任务
统一：
- Metrics
- Tables
- Figures
- Seeds
- Parameters
- Naming
- Dataset descriptions

### 阶段产出
`Final Experimental Package`

包括：
- 最终结果
- 最终表格
- 最终图片
- 实验记录
- 参数与数据说明

### 完成标准
**论文需要的证据已经齐全。**

---

# 阶段 8：论文写作

### 时间
3–5 周

### 推荐写作顺序
```text
Figures
↓
Tables
↓
Results
↓
Methods
↓
Discussion
↓
Introduction
↓
Abstract
```

### 论文结构
1. Introduction
2. Related Work
3. Dataset and Methodology
4. Experimental Protocol
5. Results
6. Cross-Dataset Analysis
7. Failure Analysis
8. Independent Experimental Validation
9. Discussion
10. Limitations
11. Conclusion

### 阶段产出
`Paper Draft v1`

### 完成标准
一个不认识项目的人，仅根据论文，就能理解并复现实验逻辑。

---

# 阶段 9：投稿准备

### 时间
2–4 周

检查：
- Technical proofreading
- Statistical review
- Citation check
- Figure check
- Table check
- Reproducibility check
- Plagiarism-risk check
- AI disclosure
- Author contribution
- Code/data availability

然后根据论文真正的贡献选择目标期刊/会议。

### 完成标准
不是“我觉得能投”，而是：

> 已按照目标 venue 的标准检查过，可以正式投稿。

---

# 阶段 10：ESP32 实时部署

### 时间
4–8 周，可选

**不应该阻塞第一篇论文。**

系统：
```text
Sensor
↓
ESP32
↓
Window
↓
Feature extraction
↓
Detector
↓
Alarm
```

测试：
- Latency
- Memory
- CPU
- Power
- Sampling
- Stability
- False Alarm

最终可作为：
- 工程原型
- 后续论文
- 项目扩展

---

# 阶段闯关规则

| 阶段 | 必须回答的问题 |
|---|---|
| 0 | 我们到底在研究什么？ |
| 1 | 现在的结果可信吗？ |
| 2 | V1 到底有多可靠？ |
| 3 | 什么问题值得写论文？ |
| 4 | 换数据还能不能工作？ |
| 5 | 为什么会失败？ |
| 6 | 在真实实验里还能不能工作？ |
| 7 | 所有证据是否完整？ |
| 8 | 能不能写成论文？ |
| 9 | 能不能正式投稿？ |
| 10 | 能不能真正部署？ |

---

# 建议时间线

如果保持稳定投入：

### Month 1
阶段 0 + 阶段 1 + 阶段 2 前半

### Month 2
阶段 2 完成 + 阶段 3

### Month 3
阶段 4 Cross-Dataset

### Month 4
阶段 5 Failure Analysis
+
阶段 6 自采实验准备

### Month 5
阶段 6 自采实验
+
阶段 7 最终实验整合

### Month 6
阶段 8 论文初稿

### Month 7
阶段 9 投稿准备

---

# 总体目标

推荐目标：

**约 6–7 个月完成一套完整研究和可投稿论文。**

如果进展顺利：
**4–5 个月**

如果跨数据集或自采实验需要大量补充：
**7–9 个月**

---

# 项目推进方式

以后不需要一次完成整个项目。

每次只推进当前阶段：

```text
你：开始阶段 X
↓
确定当前阶段任务
↓
执行
↓
验证
↓
判断是否通过
↓
进入下一阶段
```

你的职责：
- 实际执行
- 提供/确认真实实验数据
- 验证结果
- 对最终科研结论负责

我的职责：
- 研究问题拆解
- 文献分析
- 实验设计
- 代码与分析协助
- 结果审查
- 论文结构与语言
- 识别数据泄漏、统计问题和研究风险

---

# 最重要的原则

> **先证明实验是对的，再证明模型是好的，再证明模型能泛化，最后证明它能落地。**
