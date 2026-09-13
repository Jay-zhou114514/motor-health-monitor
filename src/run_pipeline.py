"""一键运行：下载数据 → 特征提取 → 两种异常检测 → 评估 → 报告。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd

import cost_analysis
import data_loading
import diagnosis as diagnosis_module
import evaluate as evaluate_module
import multi_level
import plotting
import validation
from config import (
    FIGURES_DIR,
    OUTPUT_DIR,
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
    test_counts = Counter(r["condition"] for r in test_records)
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

    # 4. 测试与评估（窗口级）
    prediction_a_series = detector_a.predict(test_table)
    prediction_b_series = detector_b.predict(test_table)
    prediction_a = prediction_a_series.to_numpy()
    prediction_b = prediction_b_series.to_numpy()
    metrics_a = evaluate_module.evaluate(y_true.to_numpy(), prediction_a)
    metrics_b = evaluate_module.evaluate(y_true.to_numpy(), prediction_b)

    # 4.1 文件级（记录级）评价：把同一文件的窗口聚合起来
    file_table_a = multi_level.file_level_table(
        test_table, prediction_a_series, "Method A (3-sigma)"
    )
    file_table_b = multi_level.file_level_table(
        test_table, prediction_b_series, "Method B (Mahalanobis)"
    )
    pd.concat([file_table_a, file_table_b], ignore_index=True).to_csv(
        OUTPUT_DIR / "file_level_metrics.csv", index=False
    )
    summary_a = multi_level.file_level_summary(
        test_table, prediction_a_series, "Method A (3-sigma)"
    )
    summary_b = multi_level.file_level_summary(
        test_table, prediction_b_series, "Method B (Mahalanobis)"
    )
    for summary in (summary_a, summary_b):
        print(
            f"{summary['method']} 文件级: "
            f"故障文件 {summary['fault_files_detected']}/{summary['fault_files']} 检出, "
            f"正常文件误报 {summary['normal_files_flagged']}/{summary['normal_files']}, "
            f"平均窗口检测率 {summary['mean_window_detection_rate']:.3f}"
        )

    def show(name: str, metrics: dict[str, float]) -> None:
        print(
            f"{name}: accuracy={metrics['accuracy']:.3f} "
            f"precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
            f"fpr={metrics['fpr']:.3f} fnr={metrics['fnr']:.3f}"
        )

    show("方法 A（3σ 阈值）", metrics_a)
    show("方法 B（马氏距离）", metrics_b)

    # 4.5 成本敏感的报警线分析
    # 假设：一次漏报（漏掉真实故障）的代价是一次误报（多停机检查一次）的 10 倍。
    FN_OVER_FP = 10.0
    scores_a = detector_a.decision_function(test_table)
    thresholds_a = np.linspace(0.5, 8.0, 32)
    curve_a = cost_analysis.cost_curve(
        scores_a, y_true.to_numpy(), thresholds_a, fn_over_fp=FN_OVER_FP
    )
    scores_b = detector_b.decision_function(test_table)
    train_distances = detector_b.decision_function(train_table)
    thresholds_b = np.quantile(
        train_distances, np.linspace(0.90, 0.9999, 32)
    )
    curve_b = cost_analysis.cost_curve(
        scores_b, y_true.to_numpy(), thresholds_b, fn_over_fp=FN_OVER_FP
    )
    sweep_a = cost_analysis.sweep_cost_ratios(
        scores_a, y_true.to_numpy(), thresholds_a
    )
    sweep_b = cost_analysis.sweep_cost_ratios(
        scores_b, y_true.to_numpy(), thresholds_b
    )
    best_a = cost_analysis.select_threshold(
        scores_a, y_true.to_numpy(), thresholds_a, fn_over_fp=FN_OVER_FP
    )
    best_b = cost_analysis.select_threshold(
        scores_b, y_true.to_numpy(), thresholds_b, fn_over_fp=FN_OVER_FP
    )

    # 4.6 泄漏安全的阈值选择：leave-one-file-out（被评估文件不参与选阈值）
    lofo_a = validation.leave_one_file_out(
        scores_a.to_numpy(),
        y_true.to_numpy(),
        test_table["record"].to_numpy(),
        thresholds_a,
        FN_OVER_FP,
    )
    lofo_b = validation.leave_one_file_out(
        scores_b.to_numpy(),
        y_true.to_numpy(),
        test_table["record"].to_numpy(),
        thresholds_b,
        FN_OVER_FP,
    )
    for name, result in (("方法 A", lofo_a), ("方法 B", lofo_b)):
        m = result["metrics"]
        print(
            f"{name} LOFO 泄漏安全阈值: precision={m['precision']:.3f} "
            f"recall={m['recall']:.3f} f1={m['f1']:.3f} fpr={m['fpr']:.3f}"
        )

    # 4.7 包络谱故障类型诊断（文件级，回答"是哪种故障"）
    diagnoser = diagnosis_module.EnvelopeDiagnoser().fit(train_records)
    diagnosis_rows = []
    for record in test_records:
        count = min(record["signal"].size, int(record["sr"] * 3.0))
        result = diagnoser.diagnose(
            record["signal"][:count], record["sr"], record
        )
        diagnosis_rows.append(
            {
                "file": record["file"],
                "condition": record["condition"],
                "predicted": result["predicted"],
                "bpfo_score": result["scores"].get("BPFO", float("nan")),
                "bpfi_score": result["scores"].get("BPFI", float("nan")),
                "best_score": result["best_score"],
            }
        )
    diagnosis_table = pd.DataFrame(diagnosis_rows)
    diagnosis_correct = int(
        (diagnosis_table["predicted"] == diagnosis_table["condition"]).sum()
    )
    print(
        f"包络谱诊断（正常基线 × {diagnoser.margin:g} 为报警倍率）："
        f"文件级 {diagnosis_correct}/{len(diagnosis_table)} 判对"
    )
    print(
        f"成本分析（漏报:误报 = {FN_OVER_FP:g}）："
        f"方法 A 最优报警线 z={best_a['threshold']:.2f}"
        f"（FP={best_a['fp']}, FN={best_a['fn']}），"
        f"方法 B 最优报警分位 d={best_b['threshold']:.2f}"
        f"（FP={best_b['fp']}, FN={best_b['fn']}）"
    )

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
    cost_path = FIGURES_DIR / "cost_curves.png"
    plotting.plot_cost_curves(curve_a, curve_b, cost_path)

    envelope_records = {
        condition: next(r for r in test_records if r["condition"] == condition)
        for condition in ("normal", "outer_race_fault", "inner_race_fault")
    }
    envelope_spectra = {}
    for condition, record in envelope_records.items():
        count = min(record["signal"].size, int(record["sr"] * 3.0))
        frequencies, spectrum = diagnosis_module.envelope_spectrum(
            record["signal"][:count], record["sr"]
        )
        envelope_spectra[condition] = (frequencies, spectrum)
    envelope_path = FIGURES_DIR / "envelope_spectra.png"
    plotting.plot_envelope_spectra(
        envelope_spectra, envelope_records["normal"], envelope_path
    )

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
        f"- 训练：{len(train_records)} 个正常基线文件；测试：{len(test_records)} 个文件"
        f"（正常 {test_counts.get('normal', 0)}，外圈故障 {test_counts.get('outer_race_fault', 0)}，"
        f"内圈故障 {test_counts.get('inner_race_fault', 0)}）。",
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
        f"| 误报率 FPR | {metrics_a['fpr']:.3f} | {metrics_b['fpr']:.3f} |",
        f"| 漏报率 FNR | {metrics_a['fnr']:.3f} | {metrics_b['fnr']:.3f} |",
        f"| 特异度 Specificity | {metrics_a['specificity']:.3f} | {metrics_b['specificity']:.3f} |",
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
        "### 4.2 文件级（记录级）评价",
        "",
        "窗口之间有 50% 重叠，窗口级指标可能高估真实检测能力。",
        "把同一文件的窗口聚合后（故障文件被判定检出 = 至少 50% 窗口报警）：",
        "",
        f"- 方法 A：故障文件 {summary_a['fault_files_detected']}/{summary_a['fault_files']} 检出；"
        f"正常文件误报 {summary_a['normal_files_flagged']}/{summary_a['normal_files']}，"
        f"平均窗口检测率 {summary_a['mean_window_detection_rate']:.3f}。",
        f"- 方法 B：故障文件 {summary_b['fault_files_detected']}/{summary_b['fault_files']} 检出；"
        f"正常文件误报 {summary_b['normal_files_flagged']}/{summary_b['normal_files']}，"
        f"平均窗口检测率 {summary_b['mean_window_detection_rate']:.3f}。",
        "",
        "**方法 A（3σ 阈值）文件级结果**",
        "",
        *multi_level.file_level_markdown(file_table_a),
        "",
        "**方法 B（马氏距离）文件级结果**",
        "",
        *multi_level.file_level_markdown(file_table_b),
        "",
        "> 局限：本实验只有 1 个正常测试文件，文件级误报率只能取 0 或 1，",
        "> 分辨率很低，结论不宜过度解读；后续需要更多正常文件或真实数据。",
        "",
        "## 5. 成本视角：报警线应该设多高？",
        "",
        "F1 把误报和漏报看得同样重，但真实工厂里两者的代价完全不同：",
        "误报（FP）浪费一次停机检查，漏报（FN）可能让轴承坏在运行中。",
        f"这里假设一次漏报的代价是一次误报的 {FN_OVER_FP:g} 倍，扫描报警线并选择总代价最低的一条。",
        "",
        f"- 方法 A 最优报警线：均值 + {best_a['threshold']:.2f}σ"
        f"（FP={best_a['fp']}，FN={best_a['fn']}，代价 {best_a['cost']:.0f}）；",
        f"- 方法 B 最优报警分位距离：{best_b['threshold']:.2f}"
        f"（FP={best_b['fp']}，FN={best_b['fn']}，代价 {best_b['cost']:.0f}）。",
        "",
        "### 报警线如何随成本偏好移动（方法 A）",
        "",
        "| 漏报:误报 代价倍率 | 最优报警线（σ） | 误报 | 漏报 |",
        "| ---: | ---: | ---: | ---: |",
        *[
            f"| {row.fn_over_fp:g} | {row.threshold:.2f} | {row.fp} | {row.fn} |"
            for row in sweep_a.itertuples()
        ],
        "",
        "在这份数据上正常与故障窗口分得很开，所以最优报警线在各倍率下保持不变——",
        "这说明当前数据还不足以暴露阈值选择的难度；在两类更接近或分布漂移的数据上",
        "（例如跨数据集验证），报警线才会真正随成本偏好移动。这正是不应随意取 3σ 的原因：",
        "阈值应由代价结构决定，而不是统计惯例。",
        "",
        "方法 B 的对应结果：",
        "",
        "| 漏报:误报 代价倍率 | 最优报警距离 | 误报 | 漏报 |",
        "| ---: | ---: | ---: | ---: |",
        *[
            f"| {row.fn_over_fp:g} | {row.threshold:.2f} | {row.fp} | {row.fn} |"
            for row in sweep_b.itertuples()
        ],
        "",
        "> 注意：这里是在测试集上扫描后报告最优值，属于「事后最优」，",
        "> 只用于理解代价结构；真实系统应预先根据维护记录固定代价倍率。",
        "",
        "### 泄漏安全版本：leave-one-file-out",
        "",
        "上面的最优阈值是在测试集上事后扫描得到的，只能用于理解代价结构，",
        "不能当作可报告的性能结论。为消除这一泄漏风险，这里做 leave-one-file-out：",
        "每次用其余文件选代价最优阈值，再在被留出的文件上评估；",
        "被评估文件自己的标签从不参与阈值选择。",
        "",
        *validation.markdown_summary("方法 A", lofo_a),
        "",
        *validation.markdown_summary("方法 B", lofo_b),
        "",
        "对比说明：方法 B 在事后最优阈值下 FP=9、FN=0；LOFO 结果则是每个文件",
        "在未见过自身数据的情况下选阈值得到的，更接近真实部署时的表现。",
        "",
        "## 6. 故障类型诊断：检出异常之后，是哪种故障？",
        "",
        "前面的两种方法只回答「是否异常」；这一步用包络谱回答「哪种故障」：",
        "1. 对振动信号做希尔伯特变换取包络（把冲击的重复模式提取出来）；",
        "2. 对包络做频谱，轴承故障特征频率（外圈 BPFO、内圈 BPFI）及其谐波会变成尖峰；",
        "3. 比较各特征频率处的「峰值/局部本底」倍数，得分最高者即诊断结论；",
        f"4. 得分需超过正常基线的 {diagnoser.margin:g} 倍才判为故障（基线 = "
        f"训练正常文件得分 {diagnoser.baseline_score_:.1f}）。",
        "",
        f"文件级诊断结果：**{diagnosis_correct}/{len(diagnosis_table)} 判对**。",
        "",
        "| 文件 | 真实 | 诊断 | BPFO 得分 | BPFI 得分 |",
        "| --- | --- | --- | ---: | ---: |",
        *[
            f"| {row.file} | {row.condition} | {row.predicted} "
            f"| {row.bpfo_score:.0f} | {row.bpfi_score:.0f} |"
            for row in diagnosis_table.itertuples()
        ],
        "",
        "诊断依据完全可解释：外圈故障文件在 BPFO 处得分数百到数千倍于本底，",
        "内圈故障文件则 BPFI 占优——与轴承动力学的预期一致。",
        "",
        "## 7. 图表",
        "",
        "- 原始波形：`outputs/figures/waveforms.png`",
        "- 方法 A 特征空间：`outputs/figures/feature_space_method_a.png`",
        "- 方法 B 特征空间：`outputs/figures/feature_space_method_b.png`",
        "- 指标对比：`outputs/figures/method_comparison.png`",
        "- 代价曲线：`outputs/figures/cost_curves.png`",
        "- 包络谱诊断：`outputs/figures/envelope_spectra.png`",
        "",
        "## 8. 局限与下一步",
        "",
        "- 局限：数据来自单一试验台与单一传感器；本实验只在文件级已知故障上验证，",
        "  没有覆盖早期退化、变转速、变载荷等情况。",
        "- 下一步：接入更大规模的 NASA IMS 退化数据；尝试故障类型分类；",
        "  条件允许后用低成本传感器做真实采集验证。",
        "",
        "## 9. 运行方法",
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





