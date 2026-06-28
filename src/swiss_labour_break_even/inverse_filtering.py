from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.structural import UnobservedComponentsResultsWrapper


@dataclass
class InverseGDPConfig:
    eta_ratio: float = 0.05
    observable_col: str = "U"
    gdp_col: str = "Y"


class FixedInverseGDPModel(MLEModel):
    """Recover latent GDP growth from observed unemployment changes."""

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

        self.ssm["design"] = np.asarray(beta, dtype=float).reshape(1, 3)
        self.ssm["transition"] = np.array(
            [
                [ar_params[1], ar_params[2], 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        self.ssm["selection"] = np.array([[1.0], [0.0], [0.0]], dtype=float)
        self.ssm["state_cov"] = np.array([[state_var]], dtype=float)
        self.ssm["obs_cov"] = np.array([[measurement_var]], dtype=float)
        self.ssm["state_intercept"] = np.array([[ar_params[0]], [0.0], [0.0]], dtype=float)
        self.ssm["obs_intercept"] = np.asarray(obs_intercept, dtype=float).reshape(1, -1)

        init_mean, init_cov = _stationary_ar2_initialization(ar_params, state_var)
        self.ssm.initialize_known(init_mean, init_cov)

    @property
    def start_params(self) -> np.ndarray:
        return np.array([], dtype=float)

    def update(self, params, **kwargs) -> None:
        return None


def _stationary_ar2_initialization(ar_params: np.ndarray, state_var: float) -> tuple[np.ndarray, np.ndarray]:
    const, phi1, phi2 = map(float, ar_params)
    mean = const / (1.0 - phi1 - phi2)
    init_mean = np.array([mean, mean, mean], dtype=float)

    transition = np.array([[phi1, phi2], [1.0, 0.0]], dtype=float)
    shock_cov = np.array([[state_var, 0.0], [0.0, 0.0]], dtype=float)
    cov = np.zeros((2, 2), dtype=float)

    for _ in range(10_000):
        new_cov = transition @ cov @ transition.T + shock_cov
        if np.max(np.abs(new_cov - cov)) < 1e-12:
            cov = new_cov
            break
        cov = new_cov

    init_cov = np.array(
        [
            [cov[0, 0], cov[0, 1], cov[1, 1]],
            [cov[1, 0], cov[1, 1], cov[1, 0]],
            [cov[1, 1], cov[0, 1], cov[0, 0]],
        ],
        dtype=float,
    )
    return init_mean, init_cov


def prepare_inverse_frame(
    data: pd.DataFrame,
    observable_col: str = "U",
    gdp_col: str = "Y",
) -> pd.DataFrame:
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["Y_lag1"] = frame[gdp_col].shift(1)
    frame["Y_lag2"] = frame[gdp_col].shift(2)
    frame["d2020q2"] = (frame["date"] == pd.Timestamp("2020-04-01")).astype(float)
    frame["d2020q3"] = (frame["date"] == pd.Timestamp("2020-07-01")).astype(float)
    return frame.dropna(subset=[observable_col, gdp_col, "Y_lag1", "Y_lag2"]).reset_index(drop=True)


def fit_measurement_model(
    data: pd.DataFrame,
    config: InverseGDPConfig | None = None,
) -> tuple[pd.DataFrame, RegressionResultsWrapper, float, UnobservedComponentsResultsWrapper]:
    if config is None:
        config = InverseGDPConfig()

    frame = prepare_inverse_frame(data, observable_col=config.observable_col, gdp_col=config.gdp_col)
    exog_cols = [config.gdp_col, "Y_lag1", "Y_lag2", "d2020q2", "d2020q3"]
    exog = frame[exog_cols]

    ols_result = sm.OLS(frame[config.observable_col], sm.add_constant(exog)).fit()
    sigma_ols = float(np.sqrt(ols_result.scale))
    state_std = config.eta_ratio * sigma_ols

    model = UnobservedComponents(frame[config.observable_col], level="local level", exog=exog)
    uc_result = model.fit_constrained({"sigma2.level": state_std**2}, disp=False)
    return frame, ols_result, state_std, uc_result


def fit_gdp_ar2(
    frame: pd.DataFrame,
    gdp_col: str = "Y",
    include_pandemic_dummies: bool = True,
) -> RegressionResultsWrapper:
    regressors = frame[["Y_lag1", "Y_lag2"]].copy()
    if include_pandemic_dummies:
        regressors["d2020q2"] = frame["d2020q2"]
        regressors["d2020q3"] = frame["d2020q3"]
    regressors = sm.add_constant(regressors)
    return sm.OLS(frame[gdp_col], regressors).fit()


def filter_implied_gdp(
    data: pd.DataFrame,
    config: InverseGDPConfig | None = None,
) -> tuple[pd.DataFrame, RegressionResultsWrapper, UnobservedComponentsResultsWrapper, RegressionResultsWrapper]:
    if config is None:
        config = InverseGDPConfig()

    frame, ols_result, state_std, uc_result = fit_measurement_model(data, config=config)
    ar2_result = fit_gdp_ar2(frame, gdp_col=config.gdp_col)

    obs_intercept = (
        uc_result.level.smoothed
        + uc_result.params["beta.d2020q2"] * frame["d2020q2"].to_numpy()
        + uc_result.params["beta.d2020q3"] * frame["d2020q3"].to_numpy()
    )

    inverse_model = FixedInverseGDPModel(
        endog=frame[config.observable_col].to_numpy(),
        obs_intercept=obs_intercept,
        beta=np.array(
            [
                uc_result.params["beta.Y"],
                uc_result.params["beta.Y_lag1"],
                uc_result.params["beta.Y_lag2"],
            ]
        ),
        ar_params=np.array(
            [
                ar2_result.params["const"],
                ar2_result.params["Y_lag1"],
                ar2_result.params["Y_lag2"],
            ]
        ),
        measurement_var=float(uc_result.params["sigma2.irregular"]),
        state_var=float(ar2_result.scale),
    )
    inverse_result = inverse_model.smooth([])

    output = frame.copy()
    output["measurement_intercept"] = obs_intercept
    output["gdp_implied_filtered"] = inverse_result.filtered_state[0]
    output["gdp_implied_smoothed"] = inverse_result.smoothed_state[0]
    output["gdp_gap_smoothed"] = output["gdp_implied_smoothed"] - output[config.gdp_col]
    output["gdp_gap_filtered"] = output["gdp_implied_filtered"] - output[config.gdp_col]
    output["gdp_implied_annualized_pct"] = ((1.0 + output["gdp_implied_smoothed"] / 100.0) ** 4 - 1.0) * 100.0
    output["gdp_actual_annualized_pct"] = ((1.0 + output[config.gdp_col] / 100.0) ** 4 - 1.0) * 100.0

    return output, ols_result, uc_result, ar2_result


def prepare_y_to_u_frame(
    data: pd.DataFrame,
    unemployment_col: str = "U",
    gdp_col: str = "Y",
) -> pd.DataFrame:
    return prepare_inverse_frame(data, observable_col=unemployment_col, gdp_col=gdp_col)


def fit_y_to_u_measurement_model(
    data: pd.DataFrame,
    config: InverseGDPConfig | None = None,
) -> tuple[pd.DataFrame, RegressionResultsWrapper, float, UnobservedComponentsResultsWrapper]:
    if config is None:
        config = InverseGDPConfig(observable_col="U")
    return fit_measurement_model(data, config=config)


def filter_implied_gdp_from_unemployment(
    data: pd.DataFrame,
    config: InverseGDPConfig | None = None,
) -> tuple[pd.DataFrame, RegressionResultsWrapper, UnobservedComponentsResultsWrapper, RegressionResultsWrapper]:
    if config is None:
        config = InverseGDPConfig(observable_col="U")
    return filter_implied_gdp(data, config=config)
