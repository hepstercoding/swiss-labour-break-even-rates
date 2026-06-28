# Swiss Labour Break-Even Rates

This repo estimates and visualizes time-varying break-even relationships in the Swiss labour market. The current production app focuses on three quarterly links:

- GDP growth to unemployment changes
- GDP growth to employment growth
- employment growth to unemployment changes

The repository contains the underlying notebooks, production scripts, memo outputs, and a Streamlit dashboard for interactive exploration.

## What The App Shows

The published dashboard includes:

- an overview of the latest break-even readings
- raw and prepared data pages
- an implied-GDP page combining unemployment-based and employment-based signals
- a model explorer that lets you switch left-hand-side and right-hand-side variables, vary lags, inspect fitted relationships, and compare direct and inverse implied paths

## Repo Layout

- `app.py`: Streamlit dashboard entrypoint
- `data/prepared/`: prepared quarterly and monthly inputs used by the app
- `notebooks/`: research notebooks documenting the data construction and estimation steps
- `outputs/production/`: production CSV outputs consumed by the dashboard
- `reports/`: memo and chart-generation files
- `scripts/production_estimates.py`: compact script to regenerate the production model outputs
- `src/swiss_labour_break_even/`: shared code for data handling and model estimation

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app expects the prepared data files in `data/prepared/` and the production outputs in `outputs/production/` to be present.

## Refresh The Production Outputs

If you want to regenerate the model outputs used by the dashboard:

```bash
source .venv/bin/activate
python scripts/production_estimates.py
```

This updates the CSVs in `outputs/production/`.

## Publish On Streamlit Community Cloud

The easiest public deployment path is Streamlit Community Cloud.

1. Push this repo to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Create a new app from the GitHub repo.
4. Set the main file path to `app.py`.
5. Deploy.

The repo already includes:

- `requirements.txt` with the runtime dependencies needed by the app
- `.streamlit/config.toml` with a stable light-theme configuration

## Notes On Dependencies

The deployment requirements are intentionally limited to the dashboard and production pipeline dependencies. If you want to reproduce some of the earlier IRISpie-based notebook experiments, install `irispie` separately in your local research environment.

## Suggested Publish Checklist

- confirm that `app.py` runs locally from a fresh virtual environment
- make sure the latest `data/prepared/` and `outputs/production/` files are committed
- review the app text and chart labels one last time
- commit and push the final deployment branch
- deploy from GitHub and test the live URL
