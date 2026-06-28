from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from swiss_labour_break_even.data import make_synthetic_dataset


def main() -> None:
    output_path = PROJECT_ROOT / "data" / "raw" / "swiss_labour_market_sample.csv"
    data = make_synthetic_dataset()
    data.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
