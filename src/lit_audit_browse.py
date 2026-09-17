"""浏览 Scopus 审计结果：按主张列出命中标题。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "docs" / "literature" / "AUDIT_scopus.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", nargs="*", default=None)
    parser.add_argument("--abstracts", action="store_true")
    parser.add_argument("--max", type=int, default=25)
    args = parser.parse_args()

    frame = pd.read_csv(CSV)
    claims = args.claims or sorted(frame["claim"].unique())
    for claim in claims:
        sub = frame[frame["claim"] == claim].head(args.max)
        print("=" * 100)
        print(f"{claim}  ({len(sub)} shown)")
        for _, row in sub.iterrows():
            date = str(row["date"])[:7] if pd.notna(row["date"]) else "?"
            title = str(row["title"])[:118]
            print(f"  [{date}] {title}")
            print(f"        {str(row['venue'])[:72]} | cited={row['cited_by']} | doi={row['doi']}")
            if args.abstracts and isinstance(row["abstract"], str) and len(row["abstract"]) > 40:
                print("        ABS: " + row["abstract"][:600].replace("\n", " "))


if __name__ == "__main__":
    main()