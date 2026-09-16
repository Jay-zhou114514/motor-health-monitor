# EXP-V1-06 证据预检报告

确定性检查：不调用任何评审模型，只核对引用数字是否存在，并独立重算判定。

## 1. 引用数字核对（几何表）

| 说明 | 折 | 特征组 | 列 | 记录引用值 | 数据文件值 | 一致 |
| --- | --- | --- | --- | ---: | ---: | :---: |
| 条件数 | F1 | RMS + centroid (problem) | condition_number | 3.54951e+09 | 3.54951e+09 | ✅ |
| s(正常) | F1 | RMS + centroid (problem) | metric_normal_median | 2.927 | 2.9274 | ✅ |
| 膨胀比 | F1 | RMS + centroid (problem) | inflation_ratio | 1.576 | 1.576 | ✅ |
| FP | F1 | RMS + centroid (problem) | fp | 9 | 9 | ✅ |
| s(正常) | F1 | RMS + kurtosis (control) | metric_normal_median | 1.611 | 1.6105 | ✅ |
| s(正常) | F1 | crest + centroid (counterexample) | metric_normal_median | 0.751 | 0.7509 | ✅ |
| 膨胀比 | F2 | RMS + centroid (problem) | inflation_ratio | 0.469 | 0.469 | ✅ |
| FP | F2 | RMS + centroid (problem) | fp | 2 | 2 | ✅ |
| s(正常) | F2 | RMS + kurtosis (control) | metric_normal_median | 0.723 | 0.7232 | ✅ |
| s(正常) | F2 | RMS + centroid (problem) | metric_normal_median | 0.734 | 0.7337 | ✅ |
| 膨胀比 | F3 | RMS + centroid (problem) | inflation_ratio | 0.797 | 0.797 | ✅ |
| s(正常) | F3 | RMS + kurtosis (control) | metric_normal_median | 1.865 | 1.8645 | ✅ |
| s(正常) | F3 | RMS + centroid (problem) | metric_normal_median | 1.81 | 1.8102 | ✅ |

## 2. 引用数字核对（δ 选择表）

| 折 | 特征组 | 准则 | 列 | 记录引用值 | 数据文件值 | 一致 |
| --- | --- | --- | --- | ---: | ---: | :---: |
| F1 | RMS + centroid (problem) | C3_ledoit_wolf | delta | 0.2201 | 0.2201 | ✅ |
| F1 | RMS + centroid (problem) | C3_ledoit_wolf | fp | 1 | 1 | ✅ |
| F1 | RMS + centroid (problem) | C3_ledoit_wolf | inflation_ratio | 0.3106 | 0.3106 | ✅ |
| F1 | RMS + centroid (problem) | C5_cond_target | delta | 0.02 | 0.02 | ✅ |
| F1 | RMS + centroid (problem) | C1_empirical | fp | 9 | 9 | ✅ |
| F3 | RMS + centroid (problem) | C3_ledoit_wolf | delta | 0.21 | 0.21 | ✅ |
| F3 | RMS + centroid (problem) | C3_ledoit_wolf | fp | 0 | 0 | ✅ |
| F2 | RMS + centroid (problem) | C4_file_loo | delta | 0.01 | 0.01 | ✅ |

## 3. 独立重算的判定

| 判据 | 实验记录声称 | 本次独立重算 | 一致 |
| --- | --- | --- | :---: |
| P1（问题组 s > 对照组） | 2/3，成立 | 2/3 | ✅ |
| P2（排序一致） | 1/3，不成立 | 1/3 | ✅ |
| P3（问题组 s 与膨胀比同向） | ρ=1.000 | ρ=1.000 (p=0.000) | ✅ |
| Q1（稳定非零 δ 的准则） | C3、C5 | C3_ledoit_wolf, C5_cond_target | ✅ |
| Q2（膨胀比 <1 且不增加对照 FP） | 成立 | 成立 | ✅ |
| Q3（不同准则 δ 差异大） | 平均极差 0.433 | 0.433 | ✅ |

## 4. 结论

- 引用数字核对：几何表 13/13 一致，δ 表 8/8 一致。
- 独立重算的判定与实验记录一致，未发现「幻觉证据」。
- 注意：本报告只验证「数字是否存在且一致」，不判断结论是否成立（后者按 protocol 需要独立评审者或人工判定）。
