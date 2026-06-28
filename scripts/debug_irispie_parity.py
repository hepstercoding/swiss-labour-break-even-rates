from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data" / "prepared" / "analysis_dataset_quarterly.csv"
OUT = ROOT / "outputs" / "debug_irispie_parity"

if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from swiss_labour_break_even.production_common import ETA_RATIO, load_analysis_data
from swiss_labour_break_even.production_models import fit_local_level


def build_y_to_u_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = df.dropna(subset=["U", "Y", "Y_lag1", "Y_lag2"]).copy()
    exog = sample[["Y", "Y_lag1", "Y_lag2", "d2020q2", "d2020q3"]].copy()
    return sample, exog


def make_parameter_table(ols_result, tv_result, sigma_ols: float, sigma_eta: float) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = [
        {"model": "meta", "parameter": "eta_ratio", "value": ETA_RATIO},
        {"model": "ols", "parameter": "sigma_ols", "value": sigma_ols},
        {"model": "local_level", "parameter": "sigma_eta_fixed", "value": sigma_eta},
        {"model": "local_level", "parameter": "log_likelihood", "value": float(tv_result.llf)},
    ]

    rows.extend(
        {"model": "ols", "parameter": str(name), "value": float(value)}
        for name, value in ols_result.params.items()
    )
    rows.extend(
        {"model": "local_level", "parameter": str(name), "value": float(value)}
        for name, value in tv_result.params.items()
    )
    return pd.DataFrame(rows)


def make_decomposition_table(
    sample: pd.DataFrame,
    exog: pd.DataFrame,
    ols_result,
    tv_result,
) -> pd.DataFrame:
    out = sample[["date", "U", "Y", "Y_lag1", "Y_lag2", "d2020q2", "d2020q3"]].copy()

    out["ols_const"] = float(ols_result.params["const"])
    for name in exog.columns:
        out[f"ols_contrib_{name}"] = float(ols_result.params[name]) * exog[name]
    out["ols_fitted"] = ols_result.predict(sm.add_constant(exog, has_constant="add"))
    out["ols_resid"] = out["U"] - out["ols_fitted"]

    out["tv_level_smoothed"] = tv_result.level.smoothed
    for name in exog.columns:
        out[f"tv_contrib_{name}"] = float(tv_result.params[f"beta.{name}"]) * exog[name]
    out["tv_fitted"] = tv_result.fittedvalues
    out["tv_resid"] = out["U"] - out["tv_fitted"]
    out["tv_one_step_resid"] = tv_result.standardized_forecasts_error[0]

    return out


def main() -> None:
    df = load_analysis_data(DATA)
    sample, exog = build_y_to_u_sample(df)

    ols_result = sm.OLS(sample["U"], sm.add_constant(exog)).fit()
    tv_result, sigma_ols, sigma_eta = fit_local_level(sample["U"], exog)

    OUT.mkdir(parents=True, exist_ok=True)

    sample.to_csv(OUT / "y_to_u_sample.csv", index=False)
    exog.to_csv(OUT / "y_to_u_exog.csv", index=False)

    parameter_table = make_parameter_table(ols_result, tv_result, sigma_ols, sigma_eta)
    parameter_table.to_csv(OUT / "y_to_u_parameter_table.csv", index=False)

    decomposition = make_decomposition_table(sample, exog, ols_result, tv_result)
    decomposition.to_csv(OUT / "y_to_u_state_decomposition.csv", index=False)

    print("Wrote IRIS parity debug exports:")
    print(f"  sample: {OUT / 'y_to_u_sample.csv'}")
    print(f"  exog: {OUT / 'y_to_u_exog.csv'}")
    print(f"  params: {OUT / 'y_to_u_parameter_table.csv'}")
    print(f"  decomposition: {OUT / 'y_to_u_state_decomposition.csv'}")
    print(
        "Sample span: "
        f"{sample['date'].min().date()} to {sample['date'].max().date()} "
        f"({len(sample)} observations)"
    )


if __name__ == "__main__":
    main()
