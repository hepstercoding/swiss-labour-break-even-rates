from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_break_even_rate(results: pd.DataFrame, output_path: str | Path) -> None:
    """Plot observed unemployment and the estimated break-even rate."""
    output_path = Path(output_path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(
        results["date"],
        results["unemployment_rate"],
        label="Unemployment rate",
        color="#1b4965",
        linewidth=2,
    )
    axes[0].plot(
        results["date"],
        results["break_even_rate_smoothed"],
        label="Break-even rate",
        color="#c1121f",
        linewidth=2,
    )
    if "break_even_rate_true" in results.columns:
        axes[0].plot(
            results["date"],
            results["break_even_rate_true"],
            label="True latent rate",
            color="#6c757d",
            linewidth=1.5,
            linestyle="--",
        )
    axes[0].set_ylabel("Percent")
    axes[0].set_title("Swiss Labour-Market Break-Even Rate")
    axes[0].legend(frameon=False)

    axes[1].plot(
        results["date"],
        results["vacancy_growth"],
        color="#2a9d8f",
        linewidth=1.8,
        label="Observed vacancy growth",
    )
    axes[1].plot(
        results["date"],
        results["vacancy_growth_fitted"],
        color="#f4a261",
        linewidth=1.8,
        label="Fitted vacancy growth",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel("Percent")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
