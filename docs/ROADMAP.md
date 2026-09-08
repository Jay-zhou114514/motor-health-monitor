# Motor Health Monitor · 项目路线图

> 版本说明：v0 = 模拟数据入门原型（已完成）；v1 = 真实公开数据 + 异常检测（当前目标）；v2/v3 见下。

## v1 完成标准

1. 接入真实滚动轴承振动数据（MathWorks 数据，含正常/外圈故障/内圈故障）。
2. 按窗口提取时域与频域特征。
3. 对比两种异常检测方法：统计阈值基线 vs 马氏距离/MAD 等轻量方法。
4. 在测试集上输出准确率/召回率/F1，并给出局限性。
5. 自动生成实验报告与图表。
6. 文档齐全，推送到 GitHub 新仓库 `motor-health-monitor`。

## v1 目录规划

```text
motor-health-monitor/
├── README.md            # 项目介绍 + 运行方法
├── requirements.txt
├── .gitignore           # 忽略 data/ outputs/ .venv/ 等
├── docs/
│   └── ROADMAP.md       # 本文件
├── src/
│   ├── lesson/          # 旧模拟原型（学习记录）
│   ├── data_loading.py
│   ├── features.py
│   ├── detection.py
│   ├── evaluate.py
│   ├── plotting.py
│   └── run_pipeline.py  # 一键运行入口
└── (data/ outputs/ 不入库)
```

## 分步任务

- [x] v0：模拟信号生成、画图、简单阈值报告
- [ ] v1.1 数据加载模块（读取 .mat，整理成表）
- [ ] v1.2 特征提取模块（RMS、峰值、峭度、频域能量等）
- [ ] v1.3 方法 A：正常数据统计阈值
- [ ] v1.4 方法 B：马氏距离 / MAD 轻量检测
- [ ] v1.5 评估：混淆矩阵与指标
- [ ] v1.6 报告与图表输出
- [ ] v1.7 README / 数据许可说明 / 上传 GitHub

## v2 / v3 方向

- v2：NASA IMS 退化数据、故障类型分类、包络谱/时频分析。
- v3：ESP32 + 低成本传感器搭建验证台。
