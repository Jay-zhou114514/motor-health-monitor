"""Generate and analyze a simulated motor-vibration signal.

This is a learning prototype, not a real equipment-diagnosis tool.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from statistics import fmean, pstdev


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATA_FILE = DATA_DIR / "sample_vibration.csv"
PLOT_FILE = OUTPUT_DIR / "signal_plot.svg"
REPORT_FILE = OUTPUT_DIR / "report.md"


def create_sample_data() -> list[tuple[float, float]]:
    """Create a signal with a deliberately injected high-vibration segment."""
    random.seed(42)
    samples: list[tuple[float, float]] = []
    for index in range(600):
        time_s = index / 100
        vibration = 0.30 * math.sin(2 * math.pi * 8 * time_s)
        vibration += random.gauss(0, 0.035)
        if 3.8 <= time_s <= 4.7:
            vibration += 0.55 * math.sin(2 * math.pi * 22 * time_s)
        samples.append((time_s, vibration))
    return samples


def save_csv(samples: list[tuple[float, float]]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with DATA_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "vibration"])
        writer.writerows(samples)


def write_svg(samples: list[tuple[float, float]], threshold: float) -> None:
    width, height, padding = 960, 360, 48
    values = [value for _, value in samples]
    minimum, maximum = min(values), max(values)
    span = maximum - minimum or 1

    def point(index: int, value: float) -> str:
        x = padding + index * (width - 2 * padding) / (len(samples) - 1)
        y = height - padding - (value - minimum) * (height - 2 * padding) / span
        return f"{x:.1f},{y:.1f}"

    points = " ".join(point(index, value) for index, (_, value) in enumerate(samples))
    threshold_y = height - padding - (threshold - minimum) * (height - 2 * padding) / span
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOT_FILE.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="48" y="28" font-family="Arial" font-size="18">Simulated motor vibration signal</text>
  <line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="#555"/>
  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="#555"/>
  <line x1="{padding}" y1="{threshold_y:.1f}" x2="{width-padding}" y2="{threshold_y:.1f}" stroke="#d23" stroke-dasharray="6,4"/>
  <polyline fill="none" stroke="#1769aa" stroke-width="1.5" points="{points}"/>
  <text x="{width-250}" y="{threshold_y-8:.1f}" font-family="Arial" font-size="13" fill="#b11">mean + 2 standard deviations</text>
  <text x="{width-110}" y="{height-18}" font-family="Arial" font-size="13">time (s)</text>
</svg>''',
        encoding="utf-8",
    )


def main() -> None:
    samples = create_sample_data()
    save_csv(samples)
    values = [value for _, value in samples]
    average = fmean(values)
    deviation = pstdev(values)
    threshold = average + 2 * deviation
    abnormal = [(time_s, value) for time_s, value in samples if abs(value - average) > 2 * deviation]
    write_svg(samples, threshold)
    REPORT_FILE.write_text(
        "# Signal analysis report\n\n"
        "This report uses simulated data only.\n\n"
        f"- Samples: {len(samples)}\n"
        f"- Mean vibration: {average:.4f}\n"
        f"- Standard deviation: {deviation:.4f}\n"
        f"- Detected abnormal samples: {len(abnormal)}\n"
        f"- Rule: absolute deviation exceeds two standard deviations\n",
        encoding="utf-8",
    )
    print(f"Created: {DATA_FILE.relative_to(ROOT)}")
    print(f"Created: {PLOT_FILE.relative_to(ROOT)}")
    print(f"Created: {REPORT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
