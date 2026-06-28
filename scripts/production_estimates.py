from __future__ import annotations

from pathlib import Path
import sys

"""
Purpose
    Run the project's core time-varying labour-market estimates from the prepared
    quarterly dataset and save tidy CSV outputs for historical analysis.

Inputs
    data/prepared/analysis_dataset_quarterly.csv
    with:
        Y = q/q real GDP growth
        E = q/q employment growth
        U = quarterly change in the unemployment rate

Outputs
    outputs/production/
        y_to_u.csv
        y_to_e.csv
        e_to_u.csv
        implied_gdp_from_labour_market.csv

How to read this file
    Python reads this file from top to bottom, but the code inside each function
    only runs once main() is called at the bottom.
"""


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data" / "prepared" / "analysis_dataset_quarterly.csv"
OUT = ROOT / "outputs" / "production"

if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from swiss_labour_break_even.production_common import load_analysis_data, save_result
from swiss_labour_break_even.production_inverse import run_inverse_tool
from swiss_labour_break_even.production_models import run_e_to_u, run_y_to_e, run_y_to_u


def main() -> None:
    df = load_analysis_data(DATA)
    save_result(run_y_to_u(df), OUT, "y_to_u.csv")
    save_result(run_y_to_e(df), OUT, "y_to_e.csv")
    save_result(run_e_to_u(df), OUT, "e_to_u.csv")
    save_result(run_inverse_tool(df), OUT, "implied_gdp_from_labour_market.csv")
    print(f"Wrote results to {OUT}")


if __name__ == "__main__":
    main()
