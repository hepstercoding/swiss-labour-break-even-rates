from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.structural import UnobservedComponents

from .production_common import (
    EMPLOYMENT_SCENARIO_QOQ,
    ETA_RATIO,
    GDP_SCENARIO_QOQ,
    annualize,
)


def fit_local_level(endog: pd.Series, exog: pd.DataFrame, eta_ratio: float = ETA_RATIO):
    ols = sm.OLS(endog, sm.add_constant(exog)).fit()
    sigma_ols = float(np.sqrt(ols.scale))
    sigma_eta = eta_ratio * sigma_ols
    # Only the intercept varies over time; slopes stay fixed.
    # We use smoothed states later on, so the outputs are ex post rather than real-time.
    model = UnobservedComponents(endog, level="local level", exog=exog)
    result = model.fit_constrained({"sigma2.level": sigma_eta**2}, disp=False)
    return result, sigma_ols, sigma_eta


def run_y_to_u(df: pd.DataFrame) -> pd.DataFrame:
    sample = df.dropna(subset=["U", "Y", "Y_lag1", "Y_lag2"]).copy()
    x = sample[["Y", "Y_lag1", "Y_lag2", "d2020q2", "d2020q3"]]
    ols = sm.OLS(sample["U"], sm.add_constant(x)).fit()
    res, sigma_ols, sigma_eta = fit_local_level(sample["U"], x)
    beta_sum = res.params["beta.Y"] + res.params["beta.Y_lag1"] + res.params["beta.Y_lag2"]
    ols_beta_sum = ols.params["Y"] + ols.params["Y_lag1"] + ols.params["Y_lag2"]

    out = sample[["date", "U", "Y", "Y_lag1", "Y_lag2", "d2020q2", "d2020q3"]].copy()
    out["c_t"] = res.level.smoothed
    # Break-even GDP growth is the growth rate that sets predicted unemployment change to zero.
    out["break_even_growth_qoq_pct"] = -out["c_t"] / beta_sum
    out["break_even_growth_annualized_pct"] = annualize(out["break_even_growth_qoq_pct"])
    static_break_even = -ols.params["const"] / ols_beta_sum
    out["static_break_even_qoq_pct"] = static_break_even
    out["static_break_even_annualized_pct"] = annualize(static_break_even)
    out["implied_U_if_gdp_0_qoq_pp"] = out["c_t"]
    out["implied_U_if_gdp_2pct_ann_qoq_pp"] = out["c_t"] + beta_sum * GDP_SCENARIO_QOQ
    out.attrs["sigma_ols"] = sigma_ols
    out.attrs["sigma_eta"] = sigma_eta
    return out


def run_y_to_e(df: pd.DataFrame) -> pd.DataFrame:
    sample = df.dropna(subset=["E", "Y", "Y_lag1", "Y_lag2"]).copy()
    x = sample[["Y", "Y_lag1", "Y_lag2", "d2020q2", "d2020q3"]]
    ols = sm.OLS(sample["E"], sm.add_constant(x)).fit()
    res, sigma_ols, sigma_eta = fit_local_level(sample["E"], x)
    beta_sum = res.params["beta.Y"] + res.params["beta.Y_lag1"] + res.params["beta.Y_lag2"]
    ols_beta_sum = ols.params["Y"] + ols.params["Y_lag1"] + ols.params["Y_lag2"]

    out = sample[["date", "E", "Y", "Y_lag1", "Y_lag2", "d2020q2", "d2020q3"]].copy()
    out["c_t"] = res.level.smoothed
    # Here the break-even concept is GDP growth consistent with zero employment growth.
    out["break_even_growth_qoq_pct"] = -out["c_t"] / beta_sum
    out["break_even_growth_annualized_pct"] = annualize(out["break_even_growth_qoq_pct"])
    static_break_even = -ols.params["const"] / ols_beta_sum
    out["static_break_even_qoq_pct"] = static_break_even
    out["static_break_even_annualized_pct"] = annualize(static_break_even)
    out["implied_E_if_gdp_0_qoq_pct"] = out["c_t"]
    out["implied_E_if_gdp_2pct_ann_qoq_pct"] = out["c_t"] + beta_sum * GDP_SCENARIO_QOQ
    out["implied_E_if_gdp_0_annualized_pct"] = annualize(out["implied_E_if_gdp_0_qoq_pct"])
    out["implied_E_if_gdp_2pct_ann_annualized_pct"] = annualize(out["implied_E_if_gdp_2pct_ann_qoq_pct"])
    out.attrs["sigma_ols"] = sigma_ols
    out.attrs["sigma_eta"] = sigma_eta
    return out


def run_e_to_u(df: pd.DataFrame) -> pd.DataFrame:
    sample = df.dropna(subset=["U", "E", "E_lag1", "E_lag2"]).copy()
    x = sample[["E", "E_lag1", "E_lag2", "d2020q2", "d2020q3"]]
    ols = sm.OLS(sample["U"], sm.add_constant(x)).fit()
    res, sigma_ols, sigma_eta = fit_local_level(sample["U"], x)
    beta_sum = res.params["beta.E"] + res.params["beta.E_lag1"] + res.params["beta.E_lag2"]
    ols_beta_sum = ols.params["E"] + ols.params["E_lag1"] + ols.params["E_lag2"]

    out = sample[["date", "U", "E", "E_lag1", "E_lag2", "d2020q2", "d2020q3"]].copy()
    out["c_t"] = res.level.smoothed
    # This specification produces a break-even employment growth rate rather than a GDP threshold.
    out["break_even_employment_growth_qoq_pct"] = -out["c_t"] / beta_sum
    out["break_even_employment_growth_annualized_pct"] = annualize(out["break_even_employment_growth_qoq_pct"])
    static_break_even = -ols.params["const"] / ols_beta_sum
    out["static_break_even_qoq_pct"] = static_break_even
    out["static_break_even_annualized_pct"] = annualize(static_break_even)
    out["implied_U_if_E_0_qoq_pp"] = out["c_t"]
    out["implied_U_if_E_1pct_ann_qoq_pp"] = out["c_t"] + beta_sum * EMPLOYMENT_SCENARIO_QOQ
    out.attrs["sigma_ols"] = sigma_ols
    out.attrs["sigma_eta"] = sigma_eta
    return out
