from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ETA_RATIO = 0.05
GDP_SCENARIO_QOQ = 100 * ((1.02) ** 0.25 - 1.0)
EMPLOYMENT_SCENARIO_QOQ = 100 * ((1.01) ** 0.25 - 1.0)


def annualize(qoq_pct: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return ((1.0 + np.asarray(qoq_pct) / 100.0) ** 4 - 1.0) * 100.0


def load_analysis_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    # Build lags and pulse dummies once so all specifications use the same conventions.
    df["Y_lag1"] = df["Y"].shift(1)
    df["Y_lag2"] = df["Y"].shift(2)
    df["E_lag1"] = df["E"].shift(1)
    df["E_lag2"] = df["E"].shift(2)
    df["d2020q2"] = (df["date"] == pd.Timestamp("2020-04-01")).astype(float)
    df["d2020q3"] = (df["date"] == pd.Timestamp("2020-07-01")).astype(float)
    return df


def save_result(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / name, index=False)
