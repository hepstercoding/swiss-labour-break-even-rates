from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _setup():
    plt.style.use("seaborn-v0_8-whitegrid")


def plot_y_to_u():
    df = pd.read_csv(OUTPUTS_DIR / "time_varying_break_even_statsmodels.csv", parse_dates=["date"])
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)

    axes[0].plot(df["date"], df["break_even_growth_annualized_pct"], color="#2a9d8f", linewidth=2.4)
    axes[0].axhline(df["static_break_even_annualized_pct"].iloc[-1], color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("Y -> U: Break-Even GDP Growth")
    axes[0].set_ylabel("% annualized")

    axes[1].plot(df["date"], df["implied_U_if_gdp_0_qoq_pp"], label="GDP = 0% q/q", linewidth=2.0)
    axes[1].plot(df["date"], df["implied_U_if_gdp_2pct_ann_qoq_pp"], label="GDP = 2% annualized", linewidth=2.0)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.9)
    axes[1].set_title("Y -> U: Implied Unemployment Change Under GDP Scenarios")
    axes[1].set_ylabel("pp")
    axes[1].set_xlabel("Quarter")
    axes[1].legend(frameon=False, ncol=2, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "memo_y_to_u.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_y_to_e():
    df = pd.read_csv(OUTPUTS_DIR / "time_varying_break_even_statsmodels_E.csv", parse_dates=["date"])
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)

    axes[0].plot(df["date"], df["break_even_growth_annualized_pct"], color="#2a9d8f", linewidth=2.4)
    axes[0].axhline(df["static_break_even_annualized_pct"].iloc[-1], color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("Y -> E: Break-Even GDP Growth")
    axes[0].set_ylabel("% annualized")

    axes[1].plot(df["date"], df["implied_E_if_gdp_0_annualized_pct"], label="GDP = 0% q/q", linewidth=2.0)
    axes[1].plot(df["date"], df["implied_E_if_gdp_2pct_ann_annualized_pct"], label="GDP = 2% annualized", linewidth=2.0)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.9)
    axes[1].set_title("Y -> E: Implied Employment Growth Under GDP Scenarios")
    axes[1].set_ylabel("% annualized")
    axes[1].set_xlabel("Quarter")
    axes[1].legend(frameon=False, ncol=2, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "memo_y_to_e.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_u_on_e():
    df = pd.read_csv(OUTPUTS_DIR / "time_varying_break_even_statsmodels_U_on_E.csv", parse_dates=["date"])
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)

    axes[0].plot(df["date"], df["break_even_employment_growth_annualized_pct"], color="#2a9d8f", linewidth=2.4)
    axes[0].axhline(df["static_break_even_annualized_pct"].iloc[-1], color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("E -> U: Break-Even Employment Growth")
    axes[0].set_ylabel("% annualized")

    axes[1].plot(df["date"], df["implied_U_if_E_0_qoq_pp"], label="E = 0% q/q", linewidth=2.0)
    axes[1].plot(df["date"], df["implied_U_if_E_1pct_ann_qoq_pp"], label="E = 1% annualized", linewidth=2.0)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.9)
    axes[1].set_title("E -> U: Implied Unemployment Change Under Employment Scenarios")
    axes[1].set_ylabel("pp")
    axes[1].set_xlabel("Quarter")
    axes[1].legend(frameon=False, ncol=2, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "memo_u_on_e.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_direct_vs_chained():
    df = pd.read_csv(OUTPUTS_DIR / "compare_chained_and_direct.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(df["date"], df["direct_break_even_annualized_pct"], label="Direct Y -> U", linewidth=2.4, color="#1f4e79")
    ax.plot(df["date"], df["chained_break_even_annualized_pct"], label="Chained Y -> E -> U", linewidth=2.1, color="#2a9d8f")
    ax.axhline(df["direct_break_even_annualized_pct"].mean(), color="black", linestyle="--", linewidth=1.0, label="Direct mean")
    ax.set_title("Direct Versus Chained Break-Even GDP Growth")
    ax.set_ylabel("% annualized")
    ax.set_xlabel("Quarter")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "memo_direct_vs_chained.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_implied_gdp_tool():
    df = pd.read_csv(OUTPUTS_DIR / "implied_gdp_from_labour_market.csv", parse_dates=["date"])
    plot_df = df[df["date"] >= "1998-01-01"].copy()

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(
        plot_df["date"],
        plot_df["gdp_actual_annualized_pct"],
        width=70,
        color="#c9d6df",
        edgecolor="none",
        label="Actual GDP growth",
    )
    ax.plot(
        plot_df["date"],
        plot_df["gdp_u_implied_annualized_pct"],
        linewidth=2.3,
        color="#1f4e79",
        label="U-implied GDP growth",
    )
    ax.plot(
        plot_df["date"],
        plot_df["gdp_e_implied_annualized_pct"],
        linewidth=2.3,
        color="#2a9d8f",
        label="E-implied GDP growth",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Labour-Implied GDP Growth")
    ax.set_ylabel("% annualized")
    ax.set_xlabel("Quarter")
    ax.set_ylim(-4.5, 4.5)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "memo_implied_gdp_tool.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_motivation_scatters():
    df = pd.read_csv(OUTPUTS_DIR.parent / "data" / "prepared" / "analysis_dataset_quarterly.csv", parse_dates=["date"])
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.8))

    pairs = [
        ("Y", "U", "GDP Growth vs Unemployment Change", "% q/q", "pp"),
        ("Y", "E", "GDP Growth vs Employment Growth", "% q/q", "% q/q"),
        ("E", "U", "Employment Growth vs Unemployment Change", "% q/q", "pp"),
    ]
    colors = ["#1f4e79", "#2a9d8f", "#e07a5f"]
    for ax, (x, y, title, xlabel, ylabel), color in zip(axes, pairs, colors):
        ax.scatter(df[x], df[y], s=18, alpha=0.65, color=color, edgecolor="none")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "memo_motivation_scatters.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_data_construction():
    df = pd.read_csv(OUTPUTS_DIR.parent / "data" / "prepared" / "labour_market_quarterly_merged.csv", parse_dates=["date"])
    plot_df = df[df["date"] >= "1992-01-01"].copy()
    plot_df["household_qoq"] = plot_df["household_employment_total_sa_spliced_chain"].pct_change() * 100
    plot_df["firm_qoq"] = plot_df["firm_employment_total_sa"].pct_change() * 100
    plot_df["blended_qoq"] = plot_df["employment_blended_level"].pct_change() * 100

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.8), sharex=True)

    axes[0].plot(plot_df["date"], plot_df["household_employment_total_sa_spliced_chain"], label="Household SA spliced", linewidth=2.0, color="#1f4e79")
    axes[0].plot(plot_df["date"], plot_df["employment_blended_level"], label="Blended employment", linewidth=2.4, color="#2a9d8f")
    axes[0].set_title("Employment Levels Used in the Project")
    axes[0].set_ylabel("Thousands")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")

    axes[1].plot(plot_df["date"], plot_df["household_qoq"], label="Household SA spliced", linewidth=1.8, color="#1f4e79")
    axes[1].plot(plot_df["date"], plot_df["firm_qoq"], label="Firm SA official", linewidth=1.8, color="#7a8fa6")
    axes[1].plot(plot_df["date"], plot_df["blended_qoq"], label="Blended employment", linewidth=2.6, color="#2a9d8f")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Quarterly Employment Growth Used for the Blend")
    axes[1].set_ylabel("% q/q")
    axes[1].set_xlabel("Quarter")
    axes[1].legend(frameon=False, ncol=3, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "memo_data_construction.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_analysis_dataset():
    df = pd.read_csv(OUTPUTS_DIR.parent / "data" / "prepared" / "analysis_dataset_quarterly.csv", parse_dates=["date"])
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.6), sharex=True)

    axes[0].plot(df["date"], df["E"], linewidth=2.0, color="#2a9d8f")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("E: Employment Growth")
    axes[0].set_ylabel("% q/q")

    axes[1].plot(df["date"], df["U"], linewidth=2.0, color="#1f4e79")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("U: Change in Unemployment")
    axes[1].set_ylabel("pp")

    axes[2].plot(df["date"], df["Y"], linewidth=2.0, color="#e07a5f")
    axes[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_title("Y: GDP Growth")
    axes[2].set_ylabel("% q/q")
    axes[2].set_xlabel("Quarter")

    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "memo_analysis_dataset.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _setup()
    plot_motivation_scatters()
    plot_data_construction()
    plot_analysis_dataset()
    plot_y_to_u()
    plot_y_to_e()
    plot_u_on_e()
    plot_direct_vs_chained()
    plot_implied_gdp_tool()


if __name__ == "__main__":
    main()
