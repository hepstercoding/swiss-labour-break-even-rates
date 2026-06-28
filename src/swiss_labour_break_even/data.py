from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic_dataset(
    periods: int = 120,
    start: str = "2015-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Create a monthly synthetic dataset for dry runs."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq="MS")

    latent_break_even = np.empty(periods)
    unemployment_rate = np.empty(periods)
    vacancy_growth = np.empty(periods)

    latent_break_even[0] = 3.2
    unemployment_rate[0] = 3.5

    beta = 1.2

    for t in range(1, periods):
        latent_break_even[t] = latent_break_even[t - 1] + rng.normal(0.0, 0.03)
        unemployment_rate[t] = (
            0.92 * unemployment_rate[t - 1]
            + 0.08 * latent_break_even[t]
            + rng.normal(0.0, 0.05)
        )

    vacancy_growth[:] = beta * (latent_break_even - unemployment_rate) + rng.normal(
        0.0,
        0.18,
        periods,
    )

    vacancy_level = 100 * np.exp(np.cumsum(vacancy_growth / 100))

    return pd.DataFrame(
        {
            "date": dates,
            "unemployment_rate": unemployment_rate,
            "vacancy_growth": vacancy_growth,
            "vacancy_level": vacancy_level,
            "break_even_rate_true": latent_break_even,
        }
    )


def load_labour_market_data(path: str | Path) -> pd.DataFrame:
    """Load monthly labour-market data and standardize core columns."""
    data = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(
        drop=True
    )

    required = {"date", "unemployment_rate"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "vacancy_growth" not in data.columns:
        if "vacancy_level" not in data.columns:
            raise ValueError(
                "Need either 'vacancy_growth' or 'vacancy_level' in the input data."
            )
        data["vacancy_growth"] = 100 * np.log(data["vacancy_level"]).diff()

    return data
