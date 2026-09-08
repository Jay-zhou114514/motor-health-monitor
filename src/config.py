"""项目路径与实验参数。"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
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
