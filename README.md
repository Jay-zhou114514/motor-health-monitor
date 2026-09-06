# Motor Health Monitor

一个零硬件起步的自动化练习项目：读取电机振动信号，计算基础统计指标，并标记可能的异常片段。

> 当前数据为程序生成的**模拟数据**，仅用于学习信号处理和异常检测流程，不代表真实设备诊断结果。

## 运行

在项目文件夹中打开 PowerShell，运行：

```powershell
python src/analyze_signal.py
```

程序会生成：

- `data/sample_vibration.csv`：模拟振动数据
- `outputs/signal_plot.svg`：振动信号图
- `outputs/report.md`：分析报告

## 下一步

1. 用公开轴承或电机振动数据替换模拟数据。
2. 对比不同异常检测方法。
3. 记录实验设置、结果和局限性。
