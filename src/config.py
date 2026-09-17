"""项目路径与实验参数。"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# 大数据集存放在 C 盘之外（磁盘容量原因；见 docs/DATA_LAYOUT.md）。
# 若外部目录不存在，自动回退到仓库内的 data/raw，保证在新机器上仍可运行。
EXTERNAL_DATA_ROOT = Path(r"E:\MotorHealthMonitorData")
DATASET_ROOT = EXTERNAL_DATA_ROOT if EXTERNAL_DATA_ROOT.exists() else RAW_DATA_DIR
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

REPORT_FILE = OUTPUT_DIR / "report.md"
FEATURES_FILE = DATA_DIR / "features.csv"

# 实验参数（可以按需修改）
WINDOW_SEC = 1.0      # 特征窗口长度（秒）
STEP_SEC = 0.5        # 窗口步长（秒），0.5 表示相邻窗口有 50% 重叠
N_STD = 3.0           # 方法 A：正常均值之上几个标准差作为报警线
MAHAL_QUANTILE = 0.99 # 方法 B：马氏距离的报警分位点
MAHAL_FEATURES = [
    "rms",
    "crest_factor",
    "kurtosis",
    "centroid_hz",
]
