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
| EXP-V1-01 | 阈值敏感性分析 | 已完成（2026-09-14） | `src/exp_threshold_sensitivity.py` |
| EXP-V1-02 | 特征消融与分布偏移分析 | 已完成（2026-09-14） | `src/exp_feature_ablation.py` |
| EXP-V1-03 | 误差定位与依赖结构验证 | 已完成（2026-09-14） | `src/exp_error_localization.py` |


