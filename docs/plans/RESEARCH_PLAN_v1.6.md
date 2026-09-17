# Motor Health Monitor 阶段性研究计划 V1.6

- Version：v1.6
- Update basis：EXP-V1-09 + 针对性文献审计（selection uncertainty / hyperparameter instability）
- Current stage：Evaluation-Reliability Study（三层框架）
- Next experiment：**EXP-V1-10 — 模型选择过程不确定性（可量化实验）**
- Update principle：Experimental-result-driven iterative planning

## 一、统一研究问题（本版确立）

> **在小样本、仅健康数据（normal-only）的轴承异常检测实验中，
> 从这类实验得出的结论有多可靠？**

这个问题比"哪个模型准确率最高"高一个层级，也是本版三条证据线共同的落点。

## 二、三层可靠性框架

### Layer 1 — 数据/划分可靠性（EXP-V1-05 → V1-07）

- 单折上观察到的"协方差几何失效"机制，在 3 个独立批次、24 个健康文件上 **0/24 复现**；
- 结论：**单次划分上观察到的机制不能直接升级为一般性机制。**

### Layer 2 — 模型选择可靠性（EXP-V1-08 → V1-09）

- 固定超参时：OC-SVM 误报 27–50%，iForest 3.6–4.8%，3σ RMS 2.4%；
- 训练内调参后：**iForest 3/3 批次达到预先设定的 ≤5%**；
  **OC-SVM 0/3**（13.1% / 35.7% / 26.2%）；P4 成立（1st_test 选参稳定性仅 33%）；
- 结论（严格限定）：**在本研究的小样本、normal-only、RBF OC-SVM 设定与预定义训练内选择协议下，
  调参没有消除其不稳定性**；iForest 可以改善，但**选择成本（每折 6.4–14.0 s，约 10⁵ 倍）
  与选择稳定性（33%）成为新的问题**。

### Layer 3 — 结果报告可靠性（本版新增）

- 可辩护的表述：
  > **若只报告经过模型选择后的单一测试集性能，而不报告候选配置、选择稳定性与选择成本，
  > 结果可能掩盖模型选择过程带来的不确定性。**
- 不写成"调参会把随机波动读成方法有效"（机制性因果断言，证据不足）。
- **EXP-V1-10 量化结果（2026-09-17 口径细化，不升版本号）**：
  固定测试集 + 50 次重采样下，92% 的复制存在并列、86% 的最小验证误报恰为 0；
  报告的误报率仍在 0%–14.29% 之间波动（SD 4.13 pp，均值 3.86%），
  且该波动与同规模非参噪声基准不可区分（观察到的模态占比反而**更集中**）。
  → 本层在论文中统一表述为：**报告的误报率主要由"划分与样本量"决定**，而不是"模型选择过程"；
  "调参把随机波动读成有效"这类因果表述**不得使用**。

## 三、五维评价框架（本版提出，逐步实现）

| 维度 | 含义 | 当前状态 |
| --- | --- | --- |
| Performance | FP rate / Recall / F1（分记录级） | ✅ 已实现（EXP-V1-08/09） |
| Selection stability | 不同训练/验证划分下，最终选中的配置是否一致 | 🔶 已有单点观察（33%）→ EXP-V1-10 量化 |
| Selection cost | 搜索了多少配置、耗时多少 | ✅ 已记录（EXP-V1-09） |
| Selection sensitivity | 验证划分微扰后，最终配置是否改变 | ⬜ EXP-V1-10 覆盖 |
| Generalization | 选出的配置到新批次是否仍有效 | ⬜ EXP-V1-10 部分覆盖 |

## 四、文献定位（本版审计结论）

| 方向 | 命中 | 含义 |
| --- | ---: | --- |
| stability selection | 463 | **通用方法论已成熟 —— 概念非我们原创** |
| nested cross-validation | 2,064 | 同上 |
| hyperparameter optimization + reproducibility/variance | 369 | 同上 |
| model selection + uncertainty + benchmark/evaluation | 599 | 同上 |
| hyperparameter + stability/instability + AD/FD | 181 | 但前 5 篇均为"用超参优化提升诊断性能"，非研究选择过程 |
| small sample + one-class/anomaly + evaluation/hyperparameter | **24** | **无一篇属于轴承/状态监测** |
| evaluation protocol + FD/CM | **27** | 无一篇涉及选择不确定性 |

**结论**：概念层面**不可声称原创**（必须引用 stability selection、nested CV、
hyperparameter-variance 等既有工作）；**贡献定位为"成熟方法学在状态监测领域的应用
+ 案例研究 + 报告规范"**。

## 五、原则：不再增加模型

保留三个**具有不同风险特征**的代表：

| 方法 | 类型 | 调参成本 | 选择稳定性 | 误报表现 |
| --- | --- | --- | --- | --- |
| 3σ RMS | 极简单统计 | ≈0 | 无选择过程 | 三批次 2.38%（稳定） |
| OC-SVM | 非线性单类 | 中 | 不稳定 | 不理想（调参后仍 13–36%） |
| iForest | 集成单类 | 很高 | 33%（大批次） | 可改善（1.2–4.8%） |

新增 RF / XGBoost / AE / LSTM 等只会把研究变回 benchmark，**本轮不做**。

## 六、下一实验：EXP-V1-10（选择过程不确定性，已冻结）

见 `experiments/EXP-V1-10-preregistration.md`：
固定独立测试集，重复 R 次「训练/验证重采样 → 超参搜索 → 冻结 → 测试」，
量化 ①选择的配置有多稳定、②测试表现是否随选择波动、③两者是否相关。

## 七、风险

1. 每个 IMS 文件约 7 个窗口 → 单折 FP rate 分辨率 ≈14.3%（须持续声明）；
2. 独立单元仅 3 个 IMS 批次（同试验台）+ 1 个 MFPT 试验台；
3. 无深度基线、无真实数据；
4. 概念层面非原创，论文必须把"应用与案例"写清楚，不能暗示发现新现象。
