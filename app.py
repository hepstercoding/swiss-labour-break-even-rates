from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.structural import UnobservedComponents


ROOT = Path(__file__).resolve().parent
PRODUCTION_DIR = ROOT / "outputs" / "production"
PREPARED_DIR = ROOT / "data" / "prepared"


st.set_page_config(
    page_title="Swiss Labour Break-Even Dashboard",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --paper: #f5f0e6;
        --ink: #112233;
        --muted: #5b6b75;
        --card: #fffaf1;
        --teal: #0f766e;
        --blue: #124e78;
        --orange: #bf6c2d;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(18,78,120,0.09), transparent 28%),
            radial-gradient(circle at bottom right, rgba(15,118,110,0.08), transparent 32%),
            var(--paper);
        color: var(--ink);
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 22px;
        background: linear-gradient(145deg, rgba(255,250,241,0.98), rgba(246,238,224,0.96));
        border: 1px solid rgba(17,34,51,0.08);
        box-shadow: 0 16px 40px rgba(17,34,51,0.08);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        letter-spacing: -0.02em;
    }
    .hero p {
        color: var(--muted);
        margin: 0.45rem 0 0 0;
        font-size: 1rem;
    }
    .metric-card {
        padding: 0.9rem 1rem;
        border-radius: 18px;
        background: rgba(255,250,241,0.92);
        border: 1px solid rgba(17,34,51,0.08);
        min-height: 125px;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 0.45rem;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-sub {
        color: var(--muted);
        margin-top: 0.35rem;
        font-size: 0.95rem;
    }
    .section-note {
        color: var(--muted);
        font-size: 0.95rem;
        margin-bottom: 0.6rem;
    }
    .stCheckbox label,
    .stCheckbox span,
    .stCheckbox p,
    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] span,
    div[data-testid="stCheckbox"] p,
    div[data-baseweb="checkbox"] label,
    div[data-baseweb="checkbox"] span,
    div[data-baseweb="checkbox"] p,
    div[role="checkbox"] + div,
    div[role="checkbox"] + div span,
    div[role="checkbox"] + div p {
        color: var(--ink) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_outputs() -> dict[str, pd.DataFrame]:
    files = {
        "y_to_u": "y_to_u.csv",
        "y_to_e": "y_to_e.csv",
        "e_to_u": "e_to_u.csv",
        "inverse": "implied_gdp_from_labour_market.csv",
    }
    outputs: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        df = pd.read_csv(PRODUCTION_DIR / filename, parse_dates=["date"])
        outputs[key] = df.sort_values("date").reset_index(drop=True)
    return outputs


@st.cache_data
def load_prepared() -> dict[str, pd.DataFrame]:
    files = {
        "Analysis dataset": "analysis_dataset_quarterly.csv",
        "Merged quarterly panel": "labour_market_quarterly_merged.csv",
        "Household employment": "household_employment_quarterly.csv",
        "Firm employment": "firm_employment_quarterly.csv",
        "Monthly unemployment": "unemployment_monthly_sa.csv",
        "Quarterly GDP": "seco_gdp_quarterly_selected.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for label, filename in files.items():
        df = pd.read_csv(PREPARED_DIR / filename, parse_dates=["date"])
        out[label] = df.sort_values("date").reset_index(drop=True)
    return out


def fmt_pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%"


def fmt_pp(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f} pp"


def annualize(qoq_pct: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return ((1.0 + np.asarray(qoq_pct) / 100.0) ** 4 - 1.0) * 100.0


def metric_card(label: str, value: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18)


def plot_break_even(df: pd.DataFrame, value_col: str, static_col: str, ylabel: str, color: str):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(df["date"], df[value_col], color=color, linewidth=2.6)
    ax.axhline(df[static_col].iloc[-1], color="#2f2f2f", linestyle="--", linewidth=1.1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    style_ax(ax)
    fig.tight_layout()
    return fig


def plot_two_scenarios(df: pd.DataFrame, col_a: str, col_b: str, label_a: str, label_b: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(df["date"], df[col_a], linewidth=2.3, color="#124e78", label=label_a)
    ax.plot(df["date"], df[col_b], linewidth=2.3, color="#bf6c2d", label=label_b)
    ax.axhline(0, color="#2f2f2f", linestyle="--", linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    style_ax(ax)
    fig.tight_layout()
    return fig


def plot_implied_gdp(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width_days = 70
    ax.bar(df["date"], df["gdp_actual_annualized_pct"], width=width_days, color="#d5dfdd", edgecolor="none", label="Actual GDP")
    ax.plot(df["date"], df["gdp_u_implied_annualized_pct"], linewidth=2.4, color="#124e78", label="U-implied GDP")
    ax.plot(df["date"], df["gdp_e_implied_annualized_pct"], linewidth=2.4, color="#0f766e", label="E-implied GDP")
    ax.axhline(0, color="#2f2f2f", linestyle="--", linewidth=1.0)
    ax.set_ylabel("% annualised")
    ax.set_xlabel("")
    ax.set_ylim(-4.5, 4.5)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    style_ax(ax)
    fig.tight_layout()
    return fig


def plot_all_implied_gdp(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    width_days = 70
    ax.bar(df["date"], df["gdp_actual_annualized_pct"], width=width_days, color="#d5dfdd", edgecolor="none", label="Actual GDP")
    ax.fill_between(
        df["date"],
        df["gdp_implied_min_annualized_pct"],
        df["gdp_implied_max_annualized_pct"],
        color="#9db7c5",
        alpha=0.35,
        label="Implied GDP range",
    )
    ax.plot(df["date"], df["gdp_implied_mean_annualized_pct"], linewidth=2.8, color="#124e78", label="Implied GDP mean")
    ax.axhline(0, color="#2f2f2f", linestyle="--", linewidth=1.0)
    ax.set_ylabel("% annualised")
    ax.set_xlabel("")
    ax.set_ylim(-4.5, 4.5)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    style_ax(ax)
    fig.tight_layout()
    return fig


def plot_all_implied_gdp_interactive(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_bar(
        x=df["date"],
        y=df["gdp_actual_annualized_pct"],
        name="Actual GDP",
        marker_color="#8fa3ad",
        opacity=0.95,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["gdp_implied_max_annualized_pct"],
            mode="lines",
            line=dict(width=0),
            name="Implied GDP max",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["gdp_implied_min_annualized_pct"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(83, 116, 132, 0.35)",
            name="Implied GDP range",
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Min=%{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["gdp_implied_mean_annualized_pct"],
            mode="lines",
            line=dict(color="#124e78", width=3),
            name="Implied GDP mean",
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Mean=%{y:.2f}%<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#2f2f2f")
    fig.update_layout(
        height=500,
        barmode="overlay",
        hovermode="x unified",
        margin=dict(l=80, r=30, t=30, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color="#112233"),
            bgcolor="rgba(245,240,230,0.85)",
        ),
        xaxis=dict(
            title=dict(text="Date"),
            rangeslider=dict(visible=True),
            type="date",
            showline=True,
            linecolor="#5b6b75",
            tickfont=dict(color="#112233"),
            title_font=dict(color="#112233"),
            automargin=True,
        ),
        yaxis=dict(
            title=dict(text="% annualised"),
            range=[-5, 10],
            showline=True,
            linecolor="#5b6b75",
            tickfont=dict(color="#112233"),
            title_font=dict(color="#112233"),
            automargin=True,
        ),
        font=dict(color="#112233"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def summarize_implied_gdp_selection(df: pd.DataFrame, selected_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["gdp_implied_min_annualized_pct"] = out[selected_cols].min(axis=1)
    out["gdp_implied_max_annualized_pct"] = out[selected_cols].max(axis=1)
    out["gdp_implied_mean_annualized_pct"] = out[selected_cols].mean(axis=1)
    return out


def build_model_frame(df: pd.DataFrame, lhs: str, rhs: str, num_lags: int, num_leads: int) -> pd.DataFrame:
    work = df[["date", lhs, rhs]].copy()
    for lead in range(1, num_leads + 1):
        work[f"{rhs}_lead{lead}"] = work[rhs].shift(-lead)
    for lag in range(1, num_lags + 1):
        work[f"{rhs}_lag{lag}"] = work[rhs].shift(lag)
    work = work.dropna().reset_index(drop=True)
    return work


def make_rhs_terms(rhs: str, num_lags: int, num_leads: int) -> list[str]:
    leads = [f"{rhs}_lead{lead}" for lead in range(num_leads, 0, -1)]
    lags = [f"{rhs}_lag{lag}" for lag in range(1, num_lags + 1)]
    return leads + [rhs] + lags


def estimate_distributed_lag(df: pd.DataFrame, lhs: str, rhs: str, num_lags: int, num_leads: int):
    work = build_model_frame(df, lhs, rhs, num_lags, num_leads)
    regressors = make_rhs_terms(rhs, num_lags, num_leads)
    X = sm.add_constant(work[regressors])
    result = sm.OLS(work[lhs], X).fit()
    beta_sum = result.params[regressors].sum()
    break_even = np.nan if np.isclose(beta_sum, 0.0) else -result.params["const"] / beta_sum
    work["fitted_lhs"] = result.fittedvalues
    return work, result, break_even


def estimate_time_varying_threshold(df: pd.DataFrame, lhs: str, rhs: str, num_lags: int, num_leads: int):
    work = build_model_frame(df, lhs, rhs, num_lags, num_leads)
    work["d2020q2"] = (work["date"] == pd.Timestamp("2020-04-01")).astype(float)
    work["d2020q3"] = (work["date"] == pd.Timestamp("2020-07-01")).astype(float)
    rhs_terms = make_rhs_terms(rhs, num_lags, num_leads)
    regressors = rhs_terms + ["d2020q2", "d2020q3"]
    X = work[regressors]
    ols = sm.OLS(work[lhs], sm.add_constant(X)).fit()
    sigma_ols = float(np.sqrt(ols.scale))
    model = UnobservedComponents(work[lhs], level="local level", exog=X)
    result = model.fit_constrained({"sigma2.level": (0.05 * sigma_ols) ** 2}, disp=False)
    beta_sum = result.params[[f"beta.{term}" for term in rhs_terms]].sum()
    work["c_t"] = result.level.smoothed
    work["dummy_effect"] = (
        result.params.get("beta.d2020q2", 0.0) * work["d2020q2"]
        + result.params.get("beta.d2020q3", 0.0) * work["d2020q3"]
    )
    work["full_intercept_t"] = work["c_t"] + work["dummy_effect"]
    work["time_varying_threshold"] = -work["c_t"] / beta_sum if not np.isclose(beta_sum, 0.0) else np.nan
    work["time_varying_lhs_if_rhs_zero"] = work["c_t"]
    return work, result, beta_sum


@st.cache_data
def build_labour_implied_gdp_views(df: pd.DataFrame) -> pd.DataFrame:
    tv_u, _, beta_sum_u = estimate_time_varying_threshold(df, "U", "Y", num_lags=2, num_leads=0)
    tv_e, _, beta_sum_e = estimate_time_varying_threshold(df, "E", "Y", num_lags=2, num_leads=0)
    inv_u = estimate_inverse_rhs(df, "U", "Y", num_lags=2)
    inv_e = estimate_inverse_rhs(df, "E", "Y", num_lags=2)

    direct_u = tv_u[["date", "U", "Y", "full_intercept_t"]].copy()
    direct_u["gdp_u_direct_qoq_pct"] = (direct_u["U"] - direct_u["full_intercept_t"]) / beta_sum_u
    direct_u["gdp_actual_annualized_pct"] = annualize(direct_u["Y"])
    direct_u["gdp_u_direct_annualized_pct"] = annualize(direct_u["gdp_u_direct_qoq_pct"])

    direct_e = tv_e[["date", "E", "full_intercept_t"]].copy()
    direct_e["gdp_e_direct_qoq_pct"] = (direct_e["E"] - direct_e["full_intercept_t"]) / beta_sum_e
    direct_e["gdp_e_direct_annualized_pct"] = annualize(direct_e["gdp_e_direct_qoq_pct"])

    inv_u = inv_u.rename(columns={"inverse_implied_rhs": "gdp_u_inverse_qoq_pct"})[["date", "gdp_u_inverse_qoq_pct"]]
    inv_u["gdp_u_inverse_annualized_pct"] = annualize(inv_u["gdp_u_inverse_qoq_pct"])
    inv_e = inv_e.rename(columns={"inverse_implied_rhs": "gdp_e_inverse_qoq_pct"})[["date", "gdp_e_inverse_qoq_pct"]]
    inv_e["gdp_e_inverse_annualized_pct"] = annualize(inv_e["gdp_e_inverse_qoq_pct"])

    out = direct_u[
        ["date", "Y", "gdp_actual_annualized_pct", "gdp_u_direct_qoq_pct", "gdp_u_direct_annualized_pct"]
    ].merge(
        direct_e[["date", "gdp_e_direct_qoq_pct", "gdp_e_direct_annualized_pct"]],
        on="date",
        how="inner",
    ).merge(
        inv_u[["date", "gdp_u_inverse_qoq_pct", "gdp_u_inverse_annualized_pct"]],
        on="date",
        how="inner",
    ).merge(
        inv_e[["date", "gdp_e_inverse_qoq_pct", "gdp_e_inverse_annualized_pct"]],
        on="date",
        how="inner",
    )
    implied_cols = [
        "gdp_u_direct_annualized_pct",
        "gdp_u_inverse_annualized_pct",
        "gdp_e_direct_annualized_pct",
        "gdp_e_inverse_annualized_pct",
    ]
    out["gdp_implied_min_annualized_pct"] = out[implied_cols].min(axis=1)
    out["gdp_implied_max_annualized_pct"] = out[implied_cols].max(axis=1)
    out["gdp_implied_mean_annualized_pct"] = out[implied_cols].mean(axis=1)
    return out.sort_values("date").reset_index(drop=True)


class InverseVariableModel(MLEModel):
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


def estimate_rhs_ar2(df: pd.DataFrame, rhs: str):
    work = df[["date", rhs]].copy()
    work[f"{rhs}_lag1"] = work[rhs].shift(1)
    work[f"{rhs}_lag2"] = work[rhs].shift(2)
    work["d2020q2"] = (work["date"] == pd.Timestamp("2020-04-01")).astype(float)
    work["d2020q3"] = (work["date"] == pd.Timestamp("2020-07-01")).astype(float)
    work = work.dropna().reset_index(drop=True)
    x = sm.add_constant(work[[f"{rhs}_lag1", f"{rhs}_lag2", "d2020q2", "d2020q3"]])
    result = sm.OLS(work[rhs], x).fit()
    return work, result


def estimate_inverse_rhs(df: pd.DataFrame, lhs: str, rhs: str, num_lags: int):
    if num_lags != 2:
        return None

    sample = df[[lhs, rhs, "date"]].copy()
    sample[f"{rhs}_lag1"] = sample[rhs].shift(1)
    sample[f"{rhs}_lag2"] = sample[rhs].shift(2)
    sample["d2020q2"] = (sample["date"] == pd.Timestamp("2020-04-01")).astype(float)
    sample["d2020q3"] = (sample["date"] == pd.Timestamp("2020-07-01")).astype(float)
    sample = sample.dropna().reset_index(drop=True)

    x = sample[[rhs, f"{rhs}_lag1", f"{rhs}_lag2", "d2020q2", "d2020q3"]]
    meas_res = sm.OLS(sample[lhs], sm.add_constant(x)).fit()
    sigma_ols = float(np.sqrt(meas_res.scale))
    tv_model = UnobservedComponents(sample[lhs], level="local level", exog=x)
    tv_res = tv_model.fit_constrained({"sigma2.level": (0.05 * sigma_ols) ** 2}, disp=False)

    _, ar2 = estimate_rhs_ar2(df, rhs)
    obs_intercept = (
        tv_res.level.smoothed
        + tv_res.params["beta.d2020q2"] * sample["d2020q2"].to_numpy()
        + tv_res.params["beta.d2020q3"] * sample["d2020q3"].to_numpy()
    )

    inv = InverseVariableModel(
        endog=sample[lhs].to_numpy(),
        obs_intercept=obs_intercept,
        beta=np.array(
            [
                tv_res.params[f"beta.{rhs}"],
                tv_res.params[f"beta.{rhs}_lag1"],
                tv_res.params[f"beta.{rhs}_lag2"],
            ]
        ),
        ar_params=np.array([ar2.params["const"], ar2.params[f"{rhs}_lag1"], ar2.params[f"{rhs}_lag2"]]),
        measurement_var=float(tv_res.params["sigma2.irregular"]),
        state_var=float(ar2.scale),
    ).smooth([])

    out = sample[["date", lhs, rhs]].copy()
    out["inverse_implied_rhs"] = inv.smoothed_state[0]
    return out


def plot_scatter(work: pd.DataFrame, lhs: str, rhs: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.scatter(work[rhs], work[lhs], s=35, alpha=0.72, color="#124e78", edgecolor="none")
    x = work[rhs].to_numpy()
    y = work[lhs].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    x_grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_grid, intercept + slope * x_grid, color="#bf6c2d", linewidth=2.0)
    ax.axhline(0, color="#2f2f2f", linestyle="--", linewidth=0.9)
    ax.axvline(0, color="#2f2f2f", linestyle="--", linewidth=0.9)
    ax.set_xlabel(rhs)
    ax.set_ylabel(lhs)
    style_ax(ax)
    fig.tight_layout()
    return fig


def plot_actual_fitted(work: pd.DataFrame, lhs: str):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(work["date"], work[lhs], width=70, color="#d5dfdd", edgecolor="none", label=f"Actual {lhs}")
    ax.plot(work["date"], work["fitted_lhs"], linewidth=2.2, color="#0f766e", label=f"Fitted {lhs}")
    ax.axhline(0, color="#2f2f2f", linestyle="--", linewidth=0.9)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.set_ylabel(lhs)
    style_ax(ax)
    fig.tight_layout()
    return fig


def plot_actual_inverse_rhs(work: pd.DataFrame, rhs: str):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(work["date"], work[rhs], width=70, color="#d5dfdd", edgecolor="none", label=f"Actual {rhs}")
    ax.plot(work["date"], work["inverse_implied_rhs"], linewidth=2.2, color="#bf6c2d", label=f"Inverse-implied {rhs}")
    ax.axhline(0, color="#2f2f2f", linestyle="--", linewidth=0.9)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.set_ylabel(rhs)
    style_ax(ax)
    fig.tight_layout()
    return fig


def plot_threshold(work: pd.DataFrame, col: str, ylabel: str, color: str):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(work["date"], work[col], linewidth=2.4, color=color)
    ax.axhline(0, color="#2f2f2f", linestyle="--", linewidth=0.9)
    ax.set_ylabel(ylabel)
    style_ax(ax)
    fig.tight_layout()
    return fig


def plot_two_variable_views(
    dates: pd.Series,
    series_a: pd.Series,
    label_a: str,
    color_a: str,
    series_b: pd.Series,
    label_b: str,
    color_b: str,
    ylabel: str,
):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(dates, series_a, linewidth=2.4, color=color_a, label=label_a)
    ax.plot(dates, series_b, linewidth=2.4, color=color_b, label=label_b)
    ax.axhline(0, color="#2f2f2f", linestyle="--", linewidth=0.9)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    style_ax(ax)
    fig.tight_layout()
    return fig


def plot_variable_fit_views(
    dates: pd.Series,
    actual: pd.Series,
    actual_label: str,
    direct_fit: pd.Series,
    direct_label: str,
    inverse_fit: pd.Series | None,
    inverse_label: str | None,
    ylabel: str,
):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(dates, actual, width=70, color="#d5dfdd", edgecolor="none", label=actual_label)
    ax.plot(dates, direct_fit, linewidth=2.2, color="#0f766e", label=direct_label)
    if inverse_fit is not None and inverse_label is not None:
        ax.plot(dates, inverse_fit, linewidth=2.2, color="#bf6c2d", label=inverse_label)
    ax.axhline(0, color="#2f2f2f", linestyle="--", linewidth=0.9)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    style_ax(ax)
    fig.tight_layout()
    return fig


def format_equation(result, lhs: str, rhs: str, num_lags: int, num_leads: int) -> str:
    p = result.params
    parts = [f"{lhs}_t = {p['const']:.3f}"]
    for lead in range(num_leads, 0, -1):
        parts.append(f"{p[f'{rhs}_lead{lead}']:.3f}·{rhs}_{{t+{lead}}}")
    parts.append(f"{p[rhs]:.3f}·{rhs}_t")
    for lag in range(1, num_lags + 1):
        parts.append(f"{p[f'{rhs}_lag{lag}']:.3f}·{rhs}_{{t-{lag}}}")
    if "d2020q2" in p.index:
        parts.append(f"{p['d2020q2']:.3f}·d2020q2_t")
    if "d2020q3" in p.index:
        parts.append(f"{p['d2020q3']:.3f}·d2020q3_t")
    return " + ".join(parts) + " + ε_t"


def plot_raw_series(df: pd.DataFrame, columns: list[str]):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    palette = ["#124e78", "#0f766e", "#bf6c2d", "#8a6f47", "#3c7a89"]
    for color, col in zip(palette * 4, columns):
        ax.plot(df["date"], df[col], linewidth=2.0, label=col, color=color)
    ax.legend(frameon=False, ncol=min(3, len(columns)), loc="upper left")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.set_xlabel("")
    style_ax(ax)
    fig.tight_layout()
    return fig


outputs = load_outputs()
prepared = load_prepared()
analysis = prepared["Analysis dataset"]
y_to_u = outputs["y_to_u"]
y_to_e = outputs["y_to_e"]
e_to_u = outputs["e_to_u"]
inverse = outputs["inverse"]
implied_gdp_all = build_labour_implied_gdp_views(analysis)

latest_y_to_u = y_to_u.iloc[-1]
latest_y_to_e = y_to_e.iloc[-1]
latest_e_to_u = e_to_u.iloc[-1]
latest_inverse = inverse.iloc[-1]
latest_date_label = latest_y_to_u["date"].to_period("Q").strftime("%Y Q%q")

st.markdown(
    f"""
    <div class="hero">
        <h1>Swiss Labour Break-Even Dashboard</h1>
        <p>Time-varying break-even growth thresholds, raw data views, and a compact model explorer built from the latest quarterly production outputs. Latest full quarter: {latest_date_label}.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Navigation")
page = st.sidebar.radio("Page", ["Model Explorer", "Implied GDP", "Methodology", "Data"], index=0)
st.sidebar.header("About")
st.sidebar.write(
    """
    This dashboard reads the production CSVs and prepared datasets generated by the
    Swiss labour-market estimation pipeline. The time-varying models are quarterly
    and use smoothed ex post state estimates.
    """
)

if page == "Data":
    st.markdown("### Raw and Prepared Data")
    st.markdown('<div class="section-note">Use this page to inspect the prepared inputs behind the models. Choose a dataset, select one or more columns, and compare the latest values directly.</div>', unsafe_allow_html=True)
    dataset_name = st.selectbox("Dataset", list(prepared.keys()), index=0)
    data_df = prepared[dataset_name].copy()
    numeric_cols = [c for c in data_df.columns if c != "date" and pd.api.types.is_numeric_dtype(data_df[c])]
    default_cols = numeric_cols[: min(3, len(numeric_cols))]
    selected_cols = st.multiselect("Series", numeric_cols, default=default_cols)
    if selected_cols:
        st.pyplot(plot_raw_series(data_df.dropna(subset=selected_cols, how="all"), selected_cols), clear_figure=True)
    st.markdown("#### Latest observations")
    st.dataframe(data_df.tail(12), use_container_width=True)

elif page == "Implied GDP":
    st.markdown("### Labour-Market-Implied GDP")
    st.markdown(
        '<div class="section-note">This page summarizes the four labour-market-implied GDP growth measures against actual GDP growth. The shaded band shows the range across the four implied measures, and the dark line shows their mean. The direct versions solve the labour equation quarter by quarter; the inverse versions recover GDP as a latent state under a GDP AR(2) transition law.</div>',
        unsafe_allow_html=True,
    )
    model_map = {
        "U-direct": "gdp_u_direct_annualized_pct",
        "U-inverse": "gdp_u_inverse_annualized_pct",
        "E-direct": "gdp_e_direct_annualized_pct",
        "E-inverse": "gdp_e_inverse_annualized_pct",
    }
    st.markdown("#### Models included in band and mean")
    c1, c2, c3, c4 = st.columns(4)
    selected_labels: list[str] = []
    with c1:
        if st.checkbox("U-direct", value=True):
            selected_labels.append("U-direct")
    with c2:
        if st.checkbox("U-inverse", value=True):
            selected_labels.append("U-inverse")
    with c3:
        if st.checkbox("E-direct", value=True):
            selected_labels.append("E-direct")
    with c4:
        if st.checkbox("E-inverse", value=True):
            selected_labels.append("E-inverse")

    if not selected_labels:
        st.warning("Select at least one model to compute the implied-GDP band and mean.")
    else:
        selected_cols = [model_map[label] for label in selected_labels]
        selected_implied = summarize_implied_gdp_selection(implied_gdp_all, selected_cols)
        st.caption(f"Included models: {', '.join(selected_labels)}")
        st.plotly_chart(plot_all_implied_gdp_interactive(selected_implied), use_container_width=True)
        latest_implied = selected_implied.iloc[-1]
        latest_table = pd.DataFrame(
            {
                "Series": ["Actual GDP", "Implied GDP mean", "Implied GDP min", "Implied GDP max"],
                "Latest reading": [
                    fmt_pct(latest_implied["gdp_actual_annualized_pct"]),
                    fmt_pct(latest_implied["gdp_implied_mean_annualized_pct"]),
                    fmt_pct(latest_implied["gdp_implied_min_annualized_pct"]),
                    fmt_pct(latest_implied["gdp_implied_max_annualized_pct"]),
                ],
            }
        )
        st.markdown("#### Latest quarter")
        st.dataframe(latest_table, use_container_width=True, hide_index=True)

elif page == "Methodology":
    st.markdown("### Methodology")
    st.markdown(
        '<div class="section-note">This page summarizes how the dashboard constructs the labour-market series, estimates the time-varying relationships, and backs out implied GDP growth from labour-market information.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### 1. Data construction")
    st.markdown(
        """
        - `U` is the quarter-on-quarter change in the quarterly average of the seasonally adjusted monthly unemployment rate.
        - `Y` is quarter-on-quarter GDP growth from the seasonally, calendar, and sport-adjusted SECO GDP series.
        - `E` is the constructed employment growth series used in the project.
        - The employment level combines the seasonally adjusted household and firm signals:
          1. average their q/q growth rates,
          2. build an index from that average growth rate,
          3. scale the resulting level to match the household level in 2019 on average.
        """
    )

    st.markdown("#### 2. Baseline time-varying equations")
    st.markdown("For any left-hand-side variable $L_t$ and right-hand-side variable $R_t$, the dashboard uses a distributed-lag equation with a time-varying intercept:")
    st.latex(r"L_t = c_t + \beta_0 R_t + \beta_1 R_{t-1} + \beta_2 R_{t-2} + \delta_1 d_{2020Q2,t} + \delta_2 d_{2020Q3,t} + \varepsilon_t")
    st.markdown("The slope coefficients are fixed, while the intercept moves over time:")
    st.latex(r"c_t = c_{t-1} + \eta_t")
    st.markdown("Here, `d2020q2` and `d2020q3` are pulse dummies for the two pandemic quarters.")

    st.markdown("#### 3. What is time-varying")
    st.markdown(
        """
        - Only the intercept is allowed to vary over time.
        - The slope coefficients on the rhs variable and its lags stay fixed.
        - The state variance of the local-level intercept is calibrated as `0.05 × sigma_ols`, where `sigma_ols` is the residual standard deviation from the corresponding static regression.
        - The dashboard uses smoothed state estimates, so these are ex post rather than real-time estimates.
        """
    )

    st.markdown("#### 4. Break-even concepts")
    st.markdown("The dashboard works with two related break-even concepts.")
    st.markdown("First, the break-even value for the rhs variable is the value that sets the lhs variable to zero.")
    st.markdown("If the rhs variable is assumed constant across current and lagged terms, the break-even rhs value is:")
    st.latex(r"R_t^{BE} = -\frac{c_t}{\beta_0+\beta_1+\beta_2}")
    st.markdown("Second, we can set the rhs variable to zero and ask what lhs value is implied by the equation.")
    st.markdown("That mirror break-even concept is:")
    st.latex(r"L_t^{BE \mid R=0} = c_t")
    st.markdown("If the pandemic dummy terms are included explicitly in the period, this becomes:")
    st.latex(r"L_t^{BE \mid R=0} = c_t + \delta' d_t")
    st.markdown(
        """
This is how the dashboard obtains:

- GDP growth consistent with unchanged unemployment
- GDP growth consistent with zero employment growth
- employment growth consistent with unchanged unemployment

and, symmetrically, the lhs values implied when the rhs variable is set to zero.
        """
    )

    st.markdown("#### 5. Direct implied paths")
    st.markdown("The direct approach solves the labour equation algebraically for the rhs variable using the observed lhs variable.")
    st.markdown("For example, with unemployment changes on the left-hand side and GDP on the right-hand side:")
    st.latex(r"U_t = c_t + \beta_0 Y_t + \beta_1 Y_{t-1} + \beta_2 Y_{t-2} + \delta' d_t + \varepsilon_t")
    st.markdown("If we impose:")
    st.latex(r"Y_t = Y_{t-1} = Y_{t-2}")
    st.markdown("then the direct labour-implied GDP rate is:")
    st.latex(r"Y_t^{direct} = \frac{U_t - c_t - \delta' d_t}{\beta_0+\beta_1+\beta_2}")
    st.markdown("The same logic is applied using employment instead of unemployment.")

    st.markdown("#### 6. Inverse state-space implied paths")
    st.markdown("The inverse approach treats the labour-market variable as observed and GDP as latent.")
    st.markdown("It combines:")
    st.markdown(
        """
1. the estimated labour equation, and
2. a separate AR(2) law of motion for GDP:
        """
    )
    st.latex(r"Y_t = a + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \kappa' d_t + \nu_t")
    st.markdown(
        """
Given these two blocks, a state-space smoother recovers the GDP path most consistent with both the observed labour-market series and the GDP transition law.

This produces:

- unemployment-based inverse implied GDP
- employment-based inverse implied GDP
        """
    )

    st.markdown("#### 7. Interpretation of the pages")
    st.markdown(
        """
        - `Overview` shows the latest production readings for the core time-varying break-even relationships.
        - `Implied GDP` summarizes the GDP signals extracted from unemployment and employment, both direct and inverse.
        - `Model Explorer` lets you choose two variables, estimate both directions, and inspect scatterplots, direct fits, inverse fits, and break-even paths.
        """
    )

    st.markdown("#### 8. Important caveats")
    st.markdown(
        """
        - The time-varying intercept is estimated with full-sample smoothing, so it benefits from future information.
        - The direct implied GDP concept depends on the assumption that current and lagged GDP terms are set equal when solving the equation.
        - The inverse implied GDP concept depends on the estimated GDP AR(2) transition law.
        - Results should therefore be read as structured summary indicators rather than as a fully identified structural model.
        """
    )

elif page == "Model Explorer":
    st.markdown("### Model Explorer")
    st.markdown('<div class="section-note">Choose two variables from the quarterly analysis dataset. The explorer automatically estimates both directions, once with Variable A on the left-hand side and once with Variable B on the left-hand side. For each variable, the charts then compare two implied paths: the level implied when the other variable is set to zero, and the level required for the other equation to set its left-hand side to zero.</div>', unsafe_allow_html=True)
    choices = ["U", "E", "Y"]
    col_a, col_b, col_lags, col_leads = st.columns(4)
    with col_a:
        var_a = st.selectbox("Variable A", choices, index=0)
    with col_b:
        var_b_options = [c for c in choices if c != var_a]
        default_b = 0 if var_a != "E" else min(1, len(var_b_options) - 1)
        var_b = st.selectbox("Variable B", var_b_options, index=default_b)
    with col_lags:
        num_lags = st.slider("Number of lags", min_value=0, max_value=6, value=2, step=1)
    with col_leads:
        num_leads = st.slider("Number of leads (diagnostic)", min_value=0, max_value=4, value=0, step=1)

    if num_leads > 0:
        st.info(
            "Lead terms are included for timing diagnostics only. The resulting equation and threshold should be read as exploratory rather than as the baseline break-even specification."
        )

    work_ab, result_ab, break_even_b_static = estimate_distributed_lag(analysis, var_a, var_b, num_lags, num_leads)
    tv_ab, tv_result_ab, _ = estimate_time_varying_threshold(analysis, var_a, var_b, num_lags, num_leads)
    work_ba, result_ba, break_even_a_static = estimate_distributed_lag(analysis, var_b, var_a, num_lags, num_leads)
    tv_ba, tv_result_ba, _ = estimate_time_varying_threshold(analysis, var_b, var_a, num_lags, num_leads)
    inverse_ab = estimate_inverse_rhs(analysis, var_a, var_b, num_lags) if num_leads == 0 else None
    inverse_ba = estimate_inverse_rhs(analysis, var_b, var_a, num_lags) if num_leads == 0 else None

    st.markdown("#### Direction A <- B")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Sample", str(len(work_ab)), "Quarterly observations")
    with c2:
        be_text = "Not defined" if np.isnan(break_even_b_static) else f"{break_even_b_static:.3f}"
        metric_card(f"Static break-even {var_b}", be_text, f"Sets {var_a} = 0")
    with c3:
        metric_card("R-squared", f"{result_ab.rsquared:.3f}", f"lhs = {var_a}, rhs = {var_b}")
    st.code(format_equation(result_ab, var_a, var_b, num_lags, num_leads), language="text")

    eq_col1, eq_col2 = st.columns(2)
    with eq_col1:
        st.markdown(f"##### Scatter: {var_a} on {var_b}")
        st.pyplot(plot_scatter(work_ab, var_a, var_b), clear_figure=True)
    with eq_col2:
        st.markdown(f"##### Actual and fitted {var_a}")
        st.pyplot(plot_actual_fitted(work_ab, var_a), clear_figure=True)

    st.markdown(f"##### Inverse state-space estimate of {var_b}")
    if inverse_ab is None:
        st.info("The inverse state-space view is shown only for a no-lead AR(2) specification, that is, 2 lags and 0 leads.")
    else:
        st.pyplot(plot_actual_inverse_rhs(inverse_ab, var_b), clear_figure=True)

    st.markdown("#### Direction B <- A")
    c4, c5, c6 = st.columns(3)
    with c4:
        metric_card("Sample", str(len(work_ba)), "Quarterly observations")
    with c5:
        be_text = "Not defined" if np.isnan(break_even_a_static) else f"{break_even_a_static:.3f}"
        metric_card(f"Static break-even {var_a}", be_text, f"Sets {var_b} = 0")
    with c6:
        metric_card("R-squared", f"{result_ba.rsquared:.3f}", f"lhs = {var_b}, rhs = {var_a}")
    st.code(format_equation(result_ba, var_b, var_a, num_lags, num_leads), language="text")

    eq_col3, eq_col4 = st.columns(2)
    with eq_col3:
        st.markdown(f"##### Scatter: {var_b} on {var_a}")
        st.pyplot(plot_scatter(work_ba, var_b, var_a), clear_figure=True)
    with eq_col4:
        st.markdown(f"##### Actual and fitted {var_b}")
        st.pyplot(plot_actual_fitted(work_ba, var_b), clear_figure=True)

    st.markdown(f"##### Inverse state-space estimate of {var_a}")
    if inverse_ba is None:
        st.info("The inverse state-space view is shown only for a no-lead AR(2) specification, that is, 2 lags and 0 leads.")
    else:
        st.pyplot(plot_actual_inverse_rhs(inverse_ba, var_a), clear_figure=True)

    st.markdown(f"#### Implied paths for {var_a}")
    st.markdown(
        f'<div class="section-note">Blue: the level of {var_a} implied when {var_b} is set to zero in the {var_a} ← {var_b} equation. Green: the level of {var_a} required to set {var_b} to zero in the {var_b} ← {var_a} equation.</div>',
        unsafe_allow_html=True,
    )
    merged_a = tv_ab[["date", "time_varying_lhs_if_rhs_zero"]].merge(
        tv_ba[["date", "time_varying_threshold"]],
        on="date",
        how="inner",
        suffixes=("_lhs_zero", "_break_even"),
    )
    st.pyplot(
        plot_two_variable_views(
            merged_a["date"],
            merged_a["time_varying_lhs_if_rhs_zero"],
            f"{var_a} implied when {var_b}=0",
            "#124e78",
            merged_a["time_varying_threshold"],
            f"{var_a} needed for {var_b}=0",
            "#0f766e",
            var_a,
        ),
        clear_figure=True,
    )

    st.markdown(f"#### Fitted paths for {var_a}")
    st.markdown(
        f'<div class="section-note">Blue: actual {var_a}. Green: direct fitted {var_a} from the {var_a} ← {var_b} equation. Orange: inverse-implied {var_a} recovered from the {var_b} ← {var_a} equation.</div>',
        unsafe_allow_html=True,
    )
    direct_a = work_ab[["date", var_a, "fitted_lhs"]].rename(columns={"fitted_lhs": "direct_fit"})
    if inverse_ba is not None:
        fit_a = direct_a.merge(
            inverse_ba[["date", "inverse_implied_rhs"]].rename(columns={"inverse_implied_rhs": "inverse_fit"}),
            on="date",
            how="inner",
        )
        inverse_series_a = fit_a["inverse_fit"]
        inverse_label_a = f"Inverse-implied {var_a}"
    else:
        fit_a = direct_a.copy()
        inverse_series_a = None
        inverse_label_a = None
    st.pyplot(
        plot_variable_fit_views(
            fit_a["date"],
            fit_a[var_a],
            f"Actual {var_a}",
            fit_a["direct_fit"],
            f"Direct fitted {var_a}",
            inverse_series_a,
            inverse_label_a,
            var_a,
        ),
        clear_figure=True,
    )

    st.markdown(f"#### Implied paths for {var_b}")
    st.markdown(
        f'<div class="section-note">Blue: the level of {var_b} implied when {var_a} is set to zero in the {var_b} ← {var_a} equation. Green: the level of {var_b} required to set {var_a} to zero in the {var_a} ← {var_b} equation.</div>',
        unsafe_allow_html=True,
    )
    merged_b = tv_ba[["date", "time_varying_lhs_if_rhs_zero"]].merge(
        tv_ab[["date", "time_varying_threshold"]],
        on="date",
        how="inner",
        suffixes=("_lhs_zero", "_break_even"),
    )
    st.pyplot(
        plot_two_variable_views(
            merged_b["date"],
            merged_b["time_varying_lhs_if_rhs_zero"],
            f"{var_b} implied when {var_a}=0",
            "#124e78",
            merged_b["time_varying_threshold"],
            f"{var_b} needed for {var_a}=0",
            "#0f766e",
            var_b,
        ),
        clear_figure=True,
    )

    st.markdown(f"#### Fitted paths for {var_b}")
    st.markdown(
        f'<div class="section-note">Blue: actual {var_b}. Green: direct fitted {var_b} from the {var_b} ← {var_a} equation. Orange: inverse-implied {var_b} recovered from the {var_a} ← {var_b} equation.</div>',
        unsafe_allow_html=True,
    )
    direct_b = work_ba[["date", var_b, "fitted_lhs"]].rename(columns={"fitted_lhs": "direct_fit"})
    if inverse_ab is not None:
        fit_b = direct_b.merge(
            inverse_ab[["date", "inverse_implied_rhs"]].rename(columns={"inverse_implied_rhs": "inverse_fit"}),
            on="date",
            how="inner",
        )
        inverse_series_b = fit_b["inverse_fit"]
        inverse_label_b = f"Inverse-implied {var_b}"
    else:
        fit_b = direct_b.copy()
        inverse_series_b = None
        inverse_label_b = None
    st.pyplot(
        plot_variable_fit_views(
            fit_b["date"],
            fit_b[var_b],
            f"Actual {var_b}",
            fit_b["direct_fit"],
            f"Direct fitted {var_b}",
            inverse_series_b,
            inverse_label_b,
            var_b,
        ),
        clear_figure=True,
    )

    st.markdown("#### Coefficients")
    coef_col1, coef_col2 = st.columns(2)
    with coef_col1:
        coef_table_ab = pd.DataFrame(
            {
                "term": result_ab.params.index,
                "estimate": result_ab.params.values,
                "std_error": result_ab.bse.values,
                "t_stat": result_ab.tvalues.values,
            }
        )
        st.dataframe(coef_table_ab, use_container_width=True, hide_index=True)
    with coef_col2:
        coef_table_ba = pd.DataFrame(
            {
                "term": result_ba.params.index,
                "estimate": result_ba.params.values,
                "std_error": result_ba.bse.values,
                "t_stat": result_ba.tvalues.values,
            }
        )
        st.dataframe(coef_table_ba, use_container_width=True, hide_index=True)
