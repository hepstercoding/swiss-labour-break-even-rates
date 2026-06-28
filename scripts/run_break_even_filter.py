from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from swiss_labour_break_even.data import load_labour_market_data, make_synthetic_dataset
from swiss_labour_break_even.filtering import FilterConfig, estimate_break_even_rate
from swiss_labour_break_even.plotting import plot_break_even_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to a monthly CSV with date, unemployment_rate, and vacancy_growth or vacancy_level.",
    )
    parser.add_argument("--beta", type=float, default=1.0, help="Slope linking slack to vacancy growth.")
    parser.add_argument(
        "--measurement-var",
        type=float,
        default=0.10,
        help="Measurement-error variance in the vacancy-growth equation.",
    )
    parser.add_argument(
        "--state-var",
        type=float,
        default=0.01,
        help="Innovation variance of the latent break-even rate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.data:
        data = load_labour_market_data(args.data)
    else:
        data = make_synthetic_dataset()

    config = FilterConfig(
        beta=args.beta,
        measurement_var=args.measurement_var,
        state_var=args.state_var,
        initial_state=float(data["unemployment_rate"].iloc[0]),
        initial_var=0.25,
    )
    results = estimate_break_even_rate(data, config=config)

    csv_path = PROJECT_ROOT / "outputs" / "break_even_rate_estimates.csv"
    chart_path = PROJECT_ROOT / "outputs" / "break_even_rate_chart.png"

    results.to_csv(csv_path, index=False)
    plot_break_even_rate(results, chart_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {chart_path}")
    print(f"Log likelihood: {results.attrs['loglik']:.2f}")


if __name__ == "__main__":
    main()
