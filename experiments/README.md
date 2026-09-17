# 实验记录目录

每个正式实验一个编号，一个 Markdown 记录 + 一个结果文件。

命名规则：

```text
EXP-V1-01-threshold-sensitivity.md     实验记录（契约、结果、分析、局限）
EXP-V1-01-threshold-sensitivity.csv    原始结果表（可复算）
```

每条记录必须包含：

1. 实验契约：Hypothesis / Dataset / Train / Test / Method / Metrics / Rule
2. 结果表
3. Observation（观察到什么）
4. Interpretation（如何解释）
5. Limitation（不能证明什么）

不允许把 Interpretation 写成 Observation。

## 实验索引

| 编号 | 名称 | 状态 | 脚本 |
| --- | --- | --- | --- |
| EXP-V1-00 | v1 基线流水线 | 已完成（待正式登记） | `src/run_pipeline.py` |
| EXP-V1-01 | 阈值敏感性分析 | 已完成（2026-09-14） | `src/exp_threshold_sensitivity.py` |
| EXP-V1-02 | 特征消融与分布偏移分析 | 已完成（2026-09-14） | `src/exp_feature_ablation.py` |
| EXP-V1-03 | 误差定位与依赖结构验证 | 已完成（2026-09-14） | `src/exp_error_localization.py` |
| EXP-V1-04 | 良态特征集受控对照 | 已完成（2026-09-15） | `src/exp_conditioning_control.py` |
| EXP-V1-05 | Covariance Geometry & Regularization | 已完成并审查（判定 B，2026-09-15） | `src/exp_v1_05_geometry.py` |
| EXP-V1-06 | 前瞻性验证（H1 复现 + δ 准则比较） | 已完成 + 证据预检通过（2026-09-16） | `src/exp_v1_06_prospective.py` |
| EXP-V1-07 | 跨批次/跨数据集验证（H2） | 已完成（2026-09-16） | src/exp_v1_07_cross_dataset.py |
| EXP-V1-08 | 单类检测器基线对比（3σ / 马氏 / OC-SVM / iForest） | 已完成（2026-09-17） | src/exp_v1_08_baselines.py |
| EXP-V1-09 | 训练内超参协议（嵌套 leave-one-file-out） | 已完成（2026-09-17） | src/exp_v1_09_hyperparams.py |
| EXP-V1-10 | 模型选择过程不确定性（固定测试集 + 重采样 + 零模型 + 分数层指标） | 已完成（2026-09-17） | src/exp_v1_10_selection_uncertainty.py（+ src/exp_v1_10_extra_checks.py） |
| EXP-V1-11 | 阈值估计噪声下限与样本量处方 | 已完成（2026-09-17；P1 成立、P2/P3 不成立） | src/exp_v1_11_sample_size_scaling.py |








