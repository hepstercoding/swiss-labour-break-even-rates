from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.mlemodel import MLEModel

from .production_common import annualize
from .production_models import fit_local_level


class InverseGDPModel(MLEModel):
    def __init__(
        self,
        endog: np.ndarray,
        obs_intercept: np.ndarray,
        beta: np.ndarray,
        ar_params: np.ndarray,
        measurement_var: float,
        state_var: float,
    ) -> None:
        super().__init__(endog, k_states=3, k_posdef=1)
        # State vector: [Y_t, Y_{t-1}, Y_{t-2}].
        self.ssm["design"] = np.asarray(beta, dtype=float).reshape(1, 3)
        self.ssm["transition"] = np.array(
            [[ar_params[1], ar_params[2], 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        )
        self.ssm["selection"] = np.array([[1.0], [0.0], [0.0]], dtype=float)
        self.ssm["state_cov"] = np.array([[state_var]], dtype=float)
        self.ssm["obs_cov"] = np.array([[measurement_var]], dtype=float)
        self.ssm["state_intercept"] = np.array([[ar_params[0]], [0.0], [0.0]], dtype=float)
        self.ssm["obs_intercept"] = np.asarray(obs_intercept, dtype=float).reshape(1, -1)
        mean, cov = stationary_ar2_init(ar_params, state_var)
        self.ssm.initialize_known(mean, cov)

    @property
    def start_params(self) -> np.ndarray:
        return np.array([], dtype=float)

    def update(self, params, **kwargs) -> None:
        return None


def stationary_ar2_init(ar_params: np.ndarray, state_var: float) -> tuple[np.ndarray, np.ndarray]:
    const, phi1, phi2 = map(float, ar_params)
    mean_y = const / (1.0 - phi1 - phi2)
    mean = np.array([mean_y, mean_y, mean_y], dtype=float)
    f = np.array([[phi1, phi2], [1.0, 0.0]], dtype=float)
    q = np.array([[state_var, 0.0], [0.0, 0.0]], dtype=float)
    cov = np.zeros((2, 2), dtype=float)
    for _ in range(10_000):
        new_cov = f @ cov @ f.T + q
        if np.max(np.abs(new_cov - cov)) < 1e-12:
            cov = new_cov
            break
        cov = new_cov
    gamma0 = cov[0, 0]
    gamma1 = cov[0, 1]
    gamma2 = phi1 * gamma1 + phi2 * gamma0
    full_cov = np.array(
        [[gamma0, gamma1, gamma2], [gamma1, gamma0, gamma1], [gamma2, gamma1, gamma0]],
        dtype=float,
    )
    return mean, full_cov


def estimate_gdp_ar2(df: pd.DataFrame):
    # Pandemic dummies enter the GDP law of motion, but not the inverse filter itself.
    x = sm.add_constant(df[["Y_lag1", "Y_lag2", "d2020q2", "d2020q3"]])
    return sm.OLS(df["Y"], x).fit()


def run_inverse_gdp(df: pd.DataFrame, observable_col: str) -> pd.DataFrame:
    sample = df.dropna(subset=[observable_col, "Y", "Y_lag1", "Y_lag2"]).copy()
    x = sample[["Y", "Y_lag1", "Y_lag2", "d2020q2", "d2020q3"]]
    meas_res, _, _ = fit_local_level(sample[observable_col], x)
    ar2 = estimate_gdp_ar2(sample)
    obs_intercept = (
        meas_res.level.smoothed
        + meas_res.params["beta.d2020q2"] * sample["d2020q2"].to_numpy()
        + meas_res.params["beta.d2020q3"] * sample["d2020q3"].to_numpy()
    )
    # Invert the labour equation: observed U or E on the left, latent GDP growth in the state.
    inv = InverseGDPModel(
        endog=sample[observable_col].to_numpy(),
        obs_intercept=obs_intercept,
        beta=np.array([meas_res.params["beta.Y"], meas_res.params["beta.Y_lag1"], meas_res.params["beta.Y_lag2"]]),
        ar_params=np.array([ar2.params["const"], ar2.params["Y_lag1"], ar2.params["Y_lag2"]]),
        measurement_var=float(meas_res.params["sigma2.irregular"]),
        state_var=float(ar2.scale),
    ).smooth([])
    out = sample[["date", observable_col, "Y"]].copy()
    out["gdp_implied_smoothed"] = inv.smoothed_state[0]
    out["gdp_implied_annualized_pct"] = annualize(out["gdp_implied_smoothed"])
    out["gdp_actual_annualized_pct"] = annualize(out["Y"])
    return out


def run_inverse_tool(df: pd.DataFrame) -> pd.DataFrame:
    # Build the two labour-implied GDP series, one from unemployment and one from employment.
    u = run_inverse_gdp(df, "U").rename(
        columns={"gdp_implied_smoothed": "gdp_u_implied_smoothed", "gdp_implied_annualized_pct": "gdp_u_implied_annualized_pct"}
    )
    e = run_inverse_gdp(df, "E").rename(
        columns={"gdp_implied_smoothed": "gdp_e_implied_smoothed", "gdp_implied_annualized_pct": "gdp_e_implied_annualized_pct"}
    )
    return u[["date", "Y", "gdp_u_implied_smoothed", "gdp_u_implied_annualized_pct", "gdp_actual_annualized_pct"]].merge(
        e[["date", "gdp_e_implied_smoothed", "gdp_e_implied_annualized_pct"]],
        on="date",
        how="inner",
    )
