from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FilterConfig:
    beta: float = 1.0
    measurement_var: float = 0.10
    state_var: float = 0.01
    initial_state: float = 3.0
    initial_var: float = 0.25


def estimate_break_even_rate(
    data: pd.DataFrame,
    config: FilterConfig | None = None,
    unemployment_col: str = "unemployment_rate",
    vacancy_growth_col: str = "vacancy_growth",
) -> pd.DataFrame:
    """
    Estimate a latent break-even unemployment rate with a random-walk Kalman filter.

    Measurement equation:
        vacancy_growth_t = beta * (break_even_t - unemployment_t) + eps_t

    Rearranged:
        vacancy_growth_t + beta * unemployment_t = beta * break_even_t + eps_t
    """
    if config is None:
        config = FilterConfig()

    frame = data.copy()
    frame = frame.dropna(subset=["date", unemployment_col, vacancy_growth_col]).reset_index(
        drop=True
    )

    y = (
        frame[vacancy_growth_col].to_numpy(dtype=float)
        + config.beta * frame[unemployment_col].to_numpy(dtype=float)
    )
    h = config.beta
    q = config.state_var
    r = config.measurement_var

    nobs = len(frame)
    a_pred = np.empty(nobs)
    p_pred = np.empty(nobs)
    a_filt = np.empty(nobs)
    p_filt = np.empty(nobs)
    innov = np.empty(nobs)
    innov_var = np.empty(nobs)
    gain = np.empty(nobs)
    loglik = 0.0

    a_prev = config.initial_state
    p_prev = config.initial_var

    for t in range(nobs):
        a_pred[t] = a_prev
        p_pred[t] = p_prev + q

        innov[t] = y[t] - h * a_pred[t]
        innov_var[t] = h * p_pred[t] * h + r
        gain[t] = p_pred[t] * h / innov_var[t]

        a_filt[t] = a_pred[t] + gain[t] * innov[t]
        p_filt[t] = (1.0 - gain[t] * h) * p_pred[t]

        loglik += -0.5 * (
            np.log(2.0 * np.pi) + np.log(innov_var[t]) + innov[t] ** 2 / innov_var[t]
        )

        a_prev = a_filt[t]
        p_prev = p_filt[t]

    a_smooth = a_filt.copy()
    p_smooth = p_filt.copy()

    for t in range(nobs - 2, -1, -1):
        smoother_gain = p_filt[t] / p_pred[t + 1]
        a_smooth[t] = a_filt[t] + smoother_gain * (a_smooth[t + 1] - a_pred[t + 1])
        p_smooth[t] = p_filt[t] + smoother_gain**2 * (p_smooth[t + 1] - p_pred[t + 1])

    frame["break_even_rate_filtered"] = a_filt
    frame["break_even_rate_smoothed"] = a_smooth
    frame["unemployment_gap_smoothed"] = (
        frame[unemployment_col] - frame["break_even_rate_smoothed"]
    )
    frame["vacancy_growth_fitted"] = config.beta * (
        frame["break_even_rate_smoothed"] - frame[unemployment_col]
    )
    frame["filter_innovation"] = innov
    frame["filter_innovation_var"] = innov_var
    frame["loglik_contribution"] = -0.5 * (
        np.log(2.0 * np.pi) + np.log(innov_var) + innov**2 / innov_var
    )
    frame.attrs["loglik"] = loglik
    frame.attrs["config"] = config

    return frame
