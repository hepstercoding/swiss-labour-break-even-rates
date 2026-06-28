from __future__ import annotations

from pathlib import Path

import irispie as ir
from irispie import ModelSource
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "debug_irispie_minimal_repro"


def make_synthetic_local_level(n: int = 80, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    q = 0.04**2
    r = 0.18**2

    state = np.empty(n, dtype=float)
    obs = np.empty(n, dtype=float)
    state[0] = 0.5
    obs[0] = state[0] + rng.normal(0.0, np.sqrt(r))

    for t in range(1, n):
        state[t] = state[t - 1] + rng.normal(0.0, np.sqrt(q))
        obs[t] = state[t] + rng.normal(0.0, np.sqrt(r))

    dates = pd.period_range("2000Q1", periods=n, freq="Q").to_timestamp()
    return pd.DataFrame({"date": dates, "y": obs, "state_true": state})


def manual_rw_smoother(y: np.ndarray, q: float, r: float, a0: float, p0: float) -> np.ndarray:
    n = len(y)
    a_pred = np.empty(n, dtype=float)
    p_pred = np.empty(n, dtype=float)
    a_filt = np.empty(n, dtype=float)
    p_filt = np.empty(n, dtype=float)

    a_prev = a0
    p_prev = p0
    for t in range(n):
        a_pred[t] = a_prev
        p_pred[t] = p_prev + q
        v = y[t] - a_pred[t]
        f = p_pred[t] + r
        k = p_pred[t] / f
        a_filt[t] = a_pred[t] + k * v
        p_filt[t] = (1.0 - k) * p_pred[t]
        a_prev = a_filt[t]
        p_prev = p_filt[t]

    a_smooth = a_filt.copy()
    p_smooth = p_filt.copy()
    for t in range(n - 2, -1, -1):
        j = p_filt[t] / p_pred[t + 1]
        a_smooth[t] = a_filt[t] + j * (a_smooth[t + 1] - a_pred[t + 1])
        p_smooth[t] = p_filt[t] + (j**2) * (p_smooth[t + 1] - p_pred[t + 1])
    return a_smooth


def fit_statsmodels(y: pd.Series, q: float) -> tuple[np.ndarray, float]:
    model = sm.tsa.UnobservedComponents(y, level="local level")
    result = model.fit_constrained({"sigma2.level": q}, disp=False)
    return np.asarray(result.level.smoothed, dtype=float), float(result.params["sigma2.irregular"])


def fit_irispie(y: np.ndarray, q: float, start_period) -> tuple[np.ndarray, float]:
    source = ModelSource.from_lists(
        transition_variables=[("", "c", ())],
        measurement_variables=[("", "y_obs", ())],
        parameters=[],
        unanticipated_shocks=[("", "eta", ())],
        measurement_shocks=[("", "eps", ())],
        transition_equations=[("", ("c = c[-1] + eta", "c = c"), ())],
        measurement_equations=[("", ("y_obs = c + eps", "y_obs = c"), ())],
    )

    input_db = ir.Databox.from_dict({"y_obs": ir.Series.from_start_and_array(start_period, y)})

    def build_model(log_std_eps: float):
        model = ir.Model.from_source(source, linear=True, flat=True)
        model.assign(c=0.0, y_obs=0.0, std_eta=float(np.sqrt(q)), std_eps=float(np.exp(log_std_eps)))
        model.steady(linear=True, flat=True)
        model.solve(linear=True, flat=True)
        return model

    def objective(theta: np.ndarray) -> float:
        try:
            model = build_model(float(theta[0]))
            _, info = model.kalman_filter(
                input_db,
                span,
                deviation=False,
                return_=("smooth",),
                return_info=True,
                diffuse_method="fixed_unknown",
            )
            return float(info["neg_log_likelihood"])
        except Exception:
            return 1e9

    span = ir.periods_from_to(start_period, start_period + len(y) - 1)
    res = minimize(
        objective,
        np.array([np.log(max(np.std(y, ddof=1), 1e-4))], dtype=float),
        method="L-BFGS-B",
        bounds=[(np.log(1e-4), np.log(5.0))],
        options={"maxiter": 200},
    )
    model = build_model(float(res.x[0]))
    output, _ = model.kalman_filter(
        input_db,
        span,
        deviation=False,
        return_=("smooth",),
        return_info=True,
        diffuse_method="fixed_unknown",
    )
    return output["smooth_med"]["c"].get_data_from_until(span).flatten(), float(np.exp(res.x[0]) ** 2)


def score(label: str, series: np.ndarray, target: np.ndarray) -> dict[str, float | str]:
    gap = series - target
    return {
        "series": label,
        "max_abs_gap": float(np.abs(gap).max()),
        "mean_abs_gap": float(np.abs(gap).mean()),
        "rmse_gap": float(np.sqrt(np.mean(gap**2))),
        "corr": float(np.corrcoef(series, target)[0, 1]),
        "first_value": float(series[0]),
        "first_gap": float(gap[0]),
    }


def main() -> None:
    df = make_synthetic_local_level()
    q = 0.04**2
    start_period = ir.qq(int(df.loc[0, "date"].year), int((df.loc[0, "date"].month - 1) / 3 + 1))

    smoothed_sm, r_sm = fit_statsmodels(df["y"], q=q)
    smoothed_manual = manual_rw_smoother(df["y"].to_numpy(dtype=float), q=q, r=r_sm, a0=0.0, p0=1e7)
    smoothed_iris, r_iris = fit_irispie(df["y"].to_numpy(dtype=float), q=q, start_period=start_period)

    summary = pd.DataFrame(
        [
            score("manual_vs_statsmodels", smoothed_manual, smoothed_sm),
            score("irispie_vs_statsmodels", smoothed_iris, smoothed_sm),
            score("manual_vs_irispie", smoothed_manual, smoothed_iris),
        ]
    )

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False)

    paths = df.copy()
    paths["smoothed_statsmodels"] = smoothed_sm
    paths["smoothed_manual"] = smoothed_manual
    paths["smoothed_irispie"] = smoothed_iris
    paths.to_csv(OUT / "paths.csv", index=False)

    meta = pd.DataFrame(
        {
            "quantity": ["q_fixed", "r_statsmodels", "r_irispie"],
            "value": [q, r_sm, r_iris],
        }
    )
    meta.to_csv(OUT / "meta.csv", index=False)

    print(summary.to_string(index=False))
    print()
    print(meta.to_string(index=False))
    print()
    print(f"Summary CSV: {OUT / 'summary.csv'}")
    print(f"Paths CSV: {OUT / 'paths.csv'}")


if __name__ == "__main__":
    main()
