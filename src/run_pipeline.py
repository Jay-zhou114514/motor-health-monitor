"""一键运行：下载数据 → 特征提取 → 两种异常检测 → 评估 → 报告。"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

import data_loading
import evaluate as evaluate_module
import plotting
from config import (
    FIGURES_DIR,
    MAHAL_FEATURES,
    MAHAL_QUANTILE,
    N_STD,
    REPORT_FILE,
    STEP_SEC,
    WINDOW_SEC,
)
from detection import MahalanobisDetector, ThresholdDetector
from features import build_feature_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Motor Health Monitor 一键流水线")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="数据已经下载过时使用，跳过下载步骤",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 1. 数据准备
    if not args.skip_download:
        print("正在准备数据文件……")
        data_loading.ensure_dataset_files(data_loading.MINIMAL_RELATIVE_FILES)
    else:
        missing = [
            rel
            for rel in data_loading.MINIMAL_RELATIVE_FILES
            if not (data_loading.RAW_DATA_DIR / rel).exists()
        ]
        if missing:
            raise SystemExit(
                "检测到数据缺失，请去掉 --skip-download 重新运行，"
                "或先运行 python src/download_data.py"
            )

    records = data_loading.load_records(data_loading.MINIMAL_RELATIVE_FILES)
    train_records = [r for r in records if r["split"] == "train"]
    test_records = [r for r in records if r["split"] == "test"]
    print(
        f"读取 {len(records)} 个文件："
        f"训练 {len(train_records)} 个（正常），测试 {len(test_records)} 个"
    )

    # 2. 特征提取
    window_kwargs = {"window_sec": WINDOW_SEC, "step_sec": STEP_SEC}
    train_table = build_feature_table(train_records, **window_kwargs)
    test_table = build_feature_table(test_records, **window_kwargs)
    train_table = train_table.dropna(
        subset=["rms", "crest_factor", "kurtosis", "centroid_hz"]
    ).reset_index(drop=True)
    test_table = test_table.dropna(
        subset=["rms", "crest_factor", "kurtosis", "centroid_hz"]
    ).reset_index(drop=True)
    y_true = pd.Series(test_table["label"] != "normal", index=test_table.index)
    print(
        f"窗口数：训练 {len(train_table)}，测试 {len(test_table)}"
        f"（其中异常 {int(y_true.sum())}）"
    )

    # 3. 训练两种检测器（只用正常数据）
    detector_a = ThresholdDetector(column="rms", n_std=N_STD)
    detector_a.fit(train_table)
    detector_b = MahalanobisDetector(
        features=MAHAL_FEATURES,
        quantile=MAHAL_QUANTILE,
    )
    detector_b.fit(train_table)
    print(
        "方法 A 报警线（RMS）: "
        f"{detector_a.threshold_value:.4f} g"
        f"（均值 {detector_a.mean_:.4f} + {N_STD} 倍标准差）"
    )

    # 4. 测试与评估
    prediction_a = detector_a.predict(test_table).to_numpy()
    prediction_b = detector_b.predict(test_table).to_numpy()
    metrics_a = evaluate_module.evaluate(y_true.to_numpy(), prediction_a)
    metrics_b = evaluate_module.evaluate(y_true.to_numpy(), prediction_b)

    def show(name: str, metrics: dict[str, float]) -> None:
        print(
            f"{name}: accuracy={metrics['accuracy']:.3f} "
            f"precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f}"
        )

    show("方法 A（3σ 阈值）", metrics_a)
    show("方法 B（马氏距离）", metrics_b)

    # 5. 画图
    signals = {
        condition: next(
            r["signal"]
            for r in test_records
            if r["condition"] == condition
        )
        for condition in ("normal", "outer_race_fault", "inner_race_fault")
    }
    sample_rate = next(r["sr"] for r in test_records)
    waveform_path = FIGURES_DIR / "waveforms.png"
    plotting.plot_waveforms(signals, sample_rate, waveform_path)

    scatter_a_path = FIGURES_DIR / "feature_space_method_a.png"
    plotting.plot_feature_space(
        test_table,
        y_true,
        pd.Series(prediction_a, index=test_table.index),
        "Method A: 3-sigma threshold",
        scatter_a_path,
    )
    scatter_b_path = FIGURES_DIR / "feature_space_method_b.png"
    plotting.plot_feature_space(
        test_table,
        y_true,
        pd.Series(prediction_b, index=test_table.index),
        "Method B: Mahalanobis distance",
        scatter_b_path,
    )
    compare_path = FIGURES_DIR / "method_comparison.png"
    plotting.plot_metric_comparison(metrics_a, metrics_b, compare_path)

    # 6. 生成 Markdown 报告
    report_lines = [
        "# Motor Health Monitor v1 实验报告",
        "",
        f"> 生成日期：{date.today().isoformat()}",
        "",
        "## 1. 目标",
        "",
        "用真实轴承振动数据判断设备是否异常，对比两种不需要人工标注故障数据的方法：",
        "只用健康（正常）数据训练，再在包含真实故障的测试数据上验证。",
        "",
        "## 2. 数据",
        "",
        "- 来源：MathWorks Rolling Element Bearing Fault Diagnosis 数据",
        "  （原始数据来自 data-acoustics.com，由 Eric Bechhoefer 提供）。",
        "- 许可：Creative Commons Attribution-NonCommercial-ShareAlike 4.0。",
        "- 内容：每个文件包含振动信号 gs、采样率 sr、转速 rate、载荷 load",
        "  与四种故障特征频率（BPFO/BPFI/FTF/BSF）。",
        "",
        f"- 采样率：{sample_rate:g} Hz",
        f"- 窗口长度：{WINDOW_SEC} 秒，步长 {STEP_SEC} 秒（50% 重叠）",
        "- 训练：2 个正常基线文件；测试：1 个正常基线 + 2 个外圈故障 + 2 个内圈故障文件。",
        "",
        "## 3. 方法",
        "",
        "**方法 A：3 倍标准差阈值法（基线）**",
        "",
        "只用均方根 RMS 一个特征。先用正常窗口估计 RMS 的均值与标准差，",
        f"报警线为 均值 + {N_STD} × 标准差 = "
        f"{detector_a.threshold_value:.4f} g。",
        "",
        "**方法 B：马氏距离法**",
        "",
        "综合使用 RMS、峰均比、峭度、谱质心四个特征。先估计正常窗口的",
        "均值与协方差，再用马氏距离衡量新窗口偏离正常分布中心的程度；",
        f"以正常训练窗口马氏距离的 {MAHAL_QUANTILE:.0%} 分位数为报警线。",
        "",
        "## 4. 结果（窗口级）",
        "",
        "| 指标 | 方法 A | 方法 B |",
        "| --- | ---: | ---: |",
        f"| 准确率 | {metrics_a['accuracy']:.3f} | {metrics_b['accuracy']:.3f} |",
        f"| 精确率 | {metrics_a['precision']:.3f} | {metrics_b['precision']:.3f} |",
        f"| 召回率 | {metrics_a['recall']:.3f} | {metrics_b['recall']:.3f} |",
        f"| F1 | {metrics_a['f1']:.3f} | {metrics_b['f1']:.3f} |",
        "",
        "### 混淆矩阵",
        "",
        "| 方法 A | 预测正常 | 预测异常 |",
        "| --- | ---: | ---: |",
        f"| 实际正常 | {metrics_a['tn']:.0f} | {metrics_a['fp']:.0f} |",
        f"| 实际异常 | {metrics_a['fn']:.0f} | {metrics_a['tp']:.0f} |",
        "",
        "| 方法 B | 预测正常 | 预测异常 |",
        "| --- | ---: | ---: |",
        f"| 实际正常 | {metrics_b['tn']:.0f} | {metrics_b['fp']:.0f} |",
        f"| 实际异常 | {metrics_b['fn']:.0f} | {metrics_b['tp']:.0f} |",
        "",
        "## 5. 图表",
        "",
        "- 原始波形：`outputs/figures/waveforms.png`",
        "- 方法 A 特征空间：`outputs/figures/feature_space_method_a.png`",
        "- 方法 B 特征空间：`outputs/figures/feature_space_method_b.png`",
        "- 指标对比：`outputs/figures/method_comparison.png`",
        "",
        "## 6. 局限与下一步",
        "",
        "- 局限：数据来自单一试验台与单一传感器；本实验只在文件级已知故障上验证，",
        "  没有覆盖早期退化、变转速、变载荷等情况。",
        "- 下一步：接入更大规模的 NASA IMS 退化数据；尝试故障类型分类；",
        "  条件允许后用低成本传感器做真实采集验证。",
        "",
        "## 7. 运行方法",
        "",
        "```powershell",
        "python src/download_data.py",
        "python src/run_pipeline.py",
        "```",
        "",
    ]
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"报告已生成：{REPORT_FILE}")
    print(f"图表目录：{FIGURES_DIR}")


if __name__ == "__main__":
    main()

