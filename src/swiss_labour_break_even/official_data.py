from __future__ import annotations

import io
import json
import re
import csv
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


BFS_HOUSEHOLD_URL = "https://dam-api.bfs.admin.ch/hub/api/dam/assets/36346767/master"
BFS_HOUSEHOLD_ASSET_PAGE_URL = "https://www.bfs.admin.ch/bfs/en/home/statistics/work-income/surveys/ets.assetdetail.36346767.html"
BFS_HOUSEHOLD_TOTALS_URL = "https://dam-api.bfs.admin.ch/hub/api/dam/assets/36710247/master"
BFS_HOUSEHOLD_TOTALS_ASSET_PAGE_URL = "https://www.bfs.admin.ch/asset/en/ts-x-03.02.01.02a"
BFS_FIRM_URL = "https://dam-api.bfs.admin.ch/hub/api/dam/assets/36412419/master"
BFS_FIRM_ASSET_PAGE_URL = "https://www.bfs.admin.ch/asset/en/px-x-0602000000_101"
SECO_GDP_URLS = (
    "https://www.seco.admin.ch/dam/seco/en/dokumente/Wirtschaft/Wirtschaftslage/"
    "BIP_Daten/ch_seco_gdp_csv.csv.download.csv/ch_seco_gdp.csv",
    "https://scheduler.swissdatas.ch/scheduled/ch-seco-gdp.csv",
)
SNB_APP_PROPERTIES_URL = "https://data.snb.ch/json/application/properties"
SNB_CUBE_EXPORT_URL = "https://data.snb.ch/json/file/cube"
SNB_URATE_CUBE_ID = "amarbma"
SNB_URATE_SERIES_CODE = "S1"


RAW_FILE_NAMES = {
    "household_xlsx": "bfs_household_ets_36346767.xlsx",
    "household_csv": "bfs_household_ets_current_totals.csv",
    "firm_xlsx": "bfs_firm_besta_36412419.xlsx",
    "gdp_csv": "seco_gdp_ch_seco_gdp.csv",
    "unemployment_csv": "snb_amarbma_unemployment.csv",
}


def download_raw_data(data_dir: str | Path) -> dict[str, Path]:
    """Download the current official source files used in the notebook."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "household_xlsx": data_dir / RAW_FILE_NAMES["household_xlsx"],
        "household_csv": data_dir / RAW_FILE_NAMES["household_csv"],
        "firm_xlsx": data_dir / RAW_FILE_NAMES["firm_xlsx"],
        "gdp_csv": data_dir / RAW_FILE_NAMES["gdp_csv"],
        "unemployment_csv": data_dir / RAW_FILE_NAMES["unemployment_csv"],
    }

    household_url = _resolve_bfs_master_url(BFS_HOUSEHOLD_ASSET_PAGE_URL, BFS_HOUSEHOLD_URL)
    household_totals_url = _resolve_bfs_master_url(
        BFS_HOUSEHOLD_TOTALS_ASSET_PAGE_URL,
        BFS_HOUSEHOLD_TOTALS_URL,
    )
    firm_url = _resolve_bfs_master_url(BFS_FIRM_ASSET_PAGE_URL, BFS_FIRM_URL)

    outputs["household_xlsx"].write_bytes(urlopen(household_url).read())
    outputs["household_csv"].write_bytes(urlopen(household_totals_url).read())
    outputs["firm_xlsx"].write_bytes(urlopen(firm_url).read())
    outputs["gdp_csv"].write_bytes(_download_first_working(SECO_GDP_URLS))
    outputs["unemployment_csv"].write_bytes(fetch_snb_unemployment_raw_csv().encode("utf-8"))

    return outputs


def _resolve_bfs_master_url(asset_page_url: str, fallback_master_url: str) -> str:
    """
    Resolve the latest BFS DAM asset id from a public asset page.

    Some BFS content pages move to a new underlying DAM asset id when a table is
    updated. Scraping the page metadata keeps us on the current release while
    preserving a stable fallback when the page lookup fails.
    """
    try:
        html = urlopen(asset_page_url).read().decode("utf-8", errors="ignore")
    except Exception:
        return fallback_master_url

    match = re.search(r"dam/assets/(\d+)/(?:thumbnail|master)", html)
    if not match:
        match = re.search(r"/asset/[a-z]{2}/(\d+)", html)
    if not match:
        return fallback_master_url
    return f"https://dam-api.bfs.admin.ch/hub/api/dam/assets/{match.group(1)}/master"


def _download_first_working(urls: tuple[str, ...]) -> bytes:
    last_error: Exception | None = None
    for url in urls:
        try:
            payload = urlopen(url).read()
            if payload.startswith(b"502 Bad Gateway"):
                raise ValueError(f"Bad gateway response from {url}")
            return payload
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("No download URLs supplied.")


def fetch_snb_unemployment_raw_csv() -> str:
    """Fetch the SNB cube export that republishes the SECO unemployment series."""
    with urlopen(SNB_APP_PROPERTIES_URL) as response:
        payload = json.load(response)

    query = {
        "fileType": "CSV",
        "lang": "en",
        "isWarehouse": "false",
        "cubeId": SNB_URATE_CUBE_ID,
        "pageViewTime": payload["pageViewTime"],
        "applicationId": payload["applicationId"],
        "environmentId": payload["environmentId"],
        "userName": payload["userName"],
    }
    request = Request(
        f"{SNB_CUBE_EXPORT_URL}?{urlencode(query)}",
        data=json.dumps({"getAllData": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        return response.read().decode("utf-8-sig")


def load_household_employment_quarterly(
    path: str | Path,
    live_totals_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load quarterly BFS household-survey employment totals.

    The file contains both raw totals and seasonally adjusted totals on the
    `Quartalswerte` sheet.
    """
    if live_totals_path is not None:
        official = _load_household_employment_xlsx(path)
        live = _load_household_employment_live_csv(live_totals_path)
        merged = live.merge(
            official.loc[:, ["date", "household_employment_total_sa"]],
            on="date",
            how="left",
        )
        return merged.sort_values("date").reset_index(drop=True)

    if Path(path).suffix.lower() == ".csv":
        live = _load_household_employment_live_csv(path)
        sibling_xlsx = Path(path).with_name(RAW_FILE_NAMES["household_xlsx"])
        if sibling_xlsx.exists():
            official = _load_household_employment_xlsx(sibling_xlsx)
            live = live.merge(
                official.loc[:, ["date", "household_employment_total_sa"]],
                on="date",
                how="left",
            )
        else:
            live["household_employment_total_sa"] = np.nan
        return live.sort_values("date").reset_index(drop=True)

    return _load_household_employment_xlsx(path)


def _load_household_employment_xlsx(path: str | Path) -> pd.DataFrame:
    """Load the legacy ETS workbook with raw and official SA totals."""
    raw = pd.read_excel(path, sheet_name="Quartalswerte", header=None)
    periods = raw.iloc[2, 1:].dropna()
    dates = pd.Index([_parse_bfs_quarter_label(x) for x in periods], name="date")

    total_raw = _extract_row_series(raw, "Erwerbstätige,Total", 1, len(dates))
    total_sa = _extract_row_series(raw, "Total, saisonbereinigte Werte", 1, len(dates))

    return pd.DataFrame(
        {
            "date": dates,
            "household_employment_total": total_raw.to_numpy(),
            "household_employment_total_sa": total_sa.to_numpy(),
        }
    ).sort_values("date").reset_index(drop=True)


def _load_household_employment_live_csv(path: str | Path) -> pd.DataFrame:
    """
    Load the newer ETS open-data CSV with total household employment.

    The current open-data release provides the up-to-date raw total series,
    while the official SA overlap is still sourced from the legacy workbook.
    """
    raw = pd.read_csv(path)
    mask = (
        (raw["FREQ"] == "Q")
        & (raw["UNIT_MES"] == "UM_1")
        & (raw["GENDER_DE"] == "Total")
        & (raw["DETAILS_DE"] == "Total")
        & (raw["MEASURE_DE"] == "Durchschnittliche Quartalswerte")
    )
    out = raw.loc[mask, ["PERIOD", "VALUE"]].copy()
    out["date"] = pd.PeriodIndex(out["PERIOD"], freq="Q").to_timestamp(how="start")
    # The open-data CSV is published in persons, while the legacy ETS workbook
    # used elsewhere in the project is in thousands of persons.
    out["household_employment_total"] = pd.to_numeric(out["VALUE"], errors="coerce") / 1000.0
    out = out.loc[:, ["date", "household_employment_total"]].dropna(subset=["household_employment_total"])
    return out.sort_values("date").reset_index(drop=True)


def load_firm_employment_quarterly(path: str | Path) -> pd.DataFrame:
    """Load quarterly BFS firm-survey total employment, raw and seasonally adjusted."""
    try:
        raw = _load_besta_total_sheet(path, "Total")
        sa = _load_besta_total_sheet(path, "Total saisonbereinigt")
        merged = raw.merge(sa, on="date", how="outer")
        return merged.sort_values("date").reset_index(drop=True)
    except Exception:
        return _load_besta_total_text_export(path)


def _load_besta_total_sheet(path: str | Path, sheet_name: str) -> pd.DataFrame:
    table = pd.read_excel(path, sheet_name=sheet_name, header=None)
    dates = _parse_besta_quarter_columns(table.iloc[5, 3:], table.iloc[6, 3:])
    total_row = table.loc[
        (table.iloc[:, 0] == "B-S")
        & (table.iloc[:, 1] == "5-96")
        & (table.iloc[:, 2] == "Total")
    ]
    if total_row.empty:
        raise ValueError(f"Could not find BESTA total row in sheet '{sheet_name}'.")

    values = pd.to_numeric(total_row.iloc[0, 3 : 3 + len(dates)], errors="coerce")
    value_name = (
        "firm_employment_total_sa"
        if "saisonbereinigt" in sheet_name.lower()
        else "firm_employment_total"
    )
    return pd.DataFrame({"date": dates, value_name: values.to_numpy()})


def _load_besta_total_text_export(path: str | Path) -> pd.DataFrame:
    """
    Load the newer BESTA text export published from the BFS PX-Web asset page.

    The live asset currently comes as a semicolon-delimited text table rather
    than the older multi-sheet Excel workbook.
    """
    with Path(path).open("r", encoding="latin1", newline="") as handle:
        rows = list(csv.reader(handle, delimiter=";"))

    if len(rows) < 7:
        raise ValueError("BESTA text export is unexpectedly short.")

    total_col = 4
    sa_col = 22
    data_rows = rows[6:]
    out_rows: list[dict[str, object]] = []
    for row in data_rows:
        if len(row) <= sa_col:
            continue
        code = row[0].strip()
        quarter = row[2].strip()
        if code != "5-96" or not quarter:
            continue
        out_rows.append(
            {
                "date": pd.Period(quarter, freq="Q").to_timestamp(how="start"),
                "firm_employment_total": pd.to_numeric(row[total_col], errors="coerce"),
                "firm_employment_total_sa": pd.to_numeric(row[sa_col], errors="coerce"),
            }
        )

    if not out_rows:
        raise ValueError("Could not find the BESTA total row in the text export.")

    return pd.DataFrame(out_rows).sort_values("date").reset_index(drop=True)


def load_unemployment_monthly(path: str | Path) -> pd.DataFrame:
    """
    Load seasonally adjusted unemployment from the SNB cube export.

    Code `S1` is the seasonally adjusted unemployment rate published in the
    SNB `amarbma` cube and sourced from SECO labour-market statistics.
    """
    raw = pd.read_csv(path, sep=";", quotechar='"', skiprows=3)
    raw.columns = [col.strip('"') for col in raw.columns]
    sa = raw.loc[raw["D0"] == SNB_URATE_SERIES_CODE, ["Date", "Value"]].copy()
    sa["date"] = pd.to_datetime(sa["Date"], format="%Y-%m")
    sa["unemployment_rate_sa"] = pd.to_numeric(sa["Value"], errors="coerce")
    sa = sa.loc[:, ["date", "unemployment_rate_sa"]].dropna(subset=["unemployment_rate_sa"])
    return sa.sort_values("date").reset_index(drop=True)


def load_seco_gdp_quarterly(path: str | Path) -> pd.DataFrame:
    """
    Load the SECO machine-readable quarterly GDP file.

    We keep real and nominal GDP for:
    - `csa`: seasonally and calendar adjusted
    - `nasa`: sport event adjusted
    - `cssa`: seasonally, calendar and sport event adjusted
    """
    gdp = pd.read_csv(path, parse_dates=["date"])
    gdp = gdp.loc[
        (gdp["structure"] == "gdp")
        & (gdp["type"].isin(["real", "nom"]))
        & (gdp["seas_adj"].isin(["csa", "nasa", "cssa"])),
        ["date", "type", "seas_adj", "value"],
    ].copy()
    gdp["series_name"] = "gdp_" + gdp["type"] + "_" + gdp["seas_adj"]
    wide = gdp.pivot(index="date", columns="series_name", values="value").reset_index()
    wide.columns.name = None
    return wide.sort_values("date").reset_index(drop=True)


def append_gdp_flash_bridge(
    gdp: pd.DataFrame,
    flash_date: str | pd.Timestamp,
    flash_qoq_pct: float,
    target_col: str = "gdp_real_cssa",
) -> pd.DataFrame:
    """
    Append a temporary GDP flash bridge to the detailed quarterly GDP panel.

    The flash is interpreted as q/q growth for the target series. We only fill
    the bridged observation when the quarter is not already present.
    """
    result = gdp.copy().sort_values("date").reset_index(drop=True)
    flash_date = pd.Timestamp(flash_date)
    if (result["date"] == flash_date).any():
        return result

    history = result.loc[result[target_col].notna(), ["date", target_col]].copy()
    if history.empty:
        raise ValueError(f"Cannot append GDP flash because '{target_col}' has no history.")

    last = history.iloc[-1]
    if flash_date <= last["date"]:
        raise ValueError("Flash date must be later than the latest observed GDP quarter.")

    new_row = {col: np.nan for col in result.columns}
    new_row["date"] = flash_date
    new_row[target_col] = float(last[target_col]) * (1.0 + float(flash_qoq_pct) / 100.0)
    if "gdp_source" in result.columns:
        new_row["gdp_source"] = "flash"

    result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
    return result.sort_values("date").reset_index(drop=True)


def build_quarterly_panel(
    household: pd.DataFrame,
    firm: pd.DataFrame,
    unemployment_monthly: pd.DataFrame,
    gdp: pd.DataFrame,
) -> pd.DataFrame:
    """Combine the labour-market and GDP series into one quarterly panel."""
    unemployment = unemployment_monthly.copy()
    unemployment["quarter"] = unemployment["date"].dt.to_period("Q").dt.to_timestamp(how="start")
    unemployment_q = (
        unemployment.groupby("quarter", as_index=False)
        .agg(
            unemployment_rate_sa_qavg=("unemployment_rate_sa", "mean"),
            unemployment_rate_sa_qend=("unemployment_rate_sa", "last"),
        )
        .rename(columns={"quarter": "date"})
    )

    panel = household.merge(firm, on="date", how="outer")
    panel = panel.merge(unemployment_q, on="date", how="outer")
    panel = panel.merge(gdp, on="date", how="outer")
    return panel.sort_values("date").reset_index(drop=True)


def seasonally_adjust_household_employment(
    household: pd.DataFrame,
    seasonal: int = 7,
    trend: int = 15,
) -> pd.DataFrame:
    """
    Create a back-extended seasonally adjusted household employment series.

    The adjustment uses STL on log levels with quarterly seasonality. The
    resulting series is benchmarked in the notebook against the official BFS
    seasonally adjusted overlap.
    """
    result = household.copy().sort_values("date").reset_index(drop=True)

    observed = result["household_employment_total"].astype(float)
    log_observed = np.log(observed)
    stl = STL(log_observed, period=4, seasonal=seasonal, trend=trend, robust=True)
    fitted = stl.fit()

    result["household_employment_total_sa_stl"] = np.exp(log_observed - fitted.seasonal)
    result["household_employment_seasonal_factor_stl"] = np.exp(fitted.seasonal)

    overlap = result["household_employment_total_sa"].notna()
    result["household_sa_gap_official_vs_stl"] = np.where(
        overlap,
        result["household_employment_total_sa"] - result["household_employment_total_sa_stl"],
        np.nan,
    )
    result["household_sa_gap_pct_official_vs_stl"] = np.where(
        overlap,
        100
        * (
            result["household_employment_total_sa_stl"]
            / result["household_employment_total_sa"]
            - 1.0
        ),
        np.nan,
    )

    return result


def splice_household_sa_series(household_sa: pd.DataFrame) -> pd.DataFrame:
    """
    Splice the STL back extension to the official BFS SA series.

    Option 2: scale the pre-official STL segment so it matches the first
    official observation exactly, then use the official series from that date on.
    """
    result = household_sa.copy().sort_values("date").reset_index(drop=True)
    official_mask = result["household_employment_total_sa"].notna()
    if not official_mask.any():
        raise ValueError("Cannot splice because the official SA series is entirely missing.")

    splice_start = result.loc[official_mask, "date"].min()
    first_official = result.loc[result["date"] == splice_start, "household_employment_total_sa"].iloc[0]
    first_model = result.loc[result["date"] == splice_start, "household_employment_total_sa_stl"].iloc[0]
    scale = first_official / first_model

    result["household_employment_total_sa_stl_chained"] = (
        result["household_employment_total_sa_stl"] * scale
    )
    result["household_employment_total_sa_spliced_chain"] = (
        result["household_employment_total_sa_stl_chained"]
    )
    result.loc[official_mask, "household_employment_total_sa_spliced_chain"] = result.loc[
        official_mask, "household_employment_total_sa"
    ]
    result["household_employment_total_sa_spliced"] = result["household_employment_total_sa_spliced_chain"]
    result.attrs["splice_start"] = splice_start
    result.attrs["splice_scale"] = scale
    return result


def save_prepared_tables(
    output_dir: str | Path,
    household: pd.DataFrame,
    firm: pd.DataFrame,
    unemployment_monthly: pd.DataFrame,
    gdp: pd.DataFrame,
    panel: pd.DataFrame,
) -> dict[str, Path]:
    """Write the cleaned tables used by the notebook."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "household": output_dir / "household_employment_quarterly.csv",
        "firm": output_dir / "firm_employment_quarterly.csv",
        "unemployment": output_dir / "unemployment_monthly_sa.csv",
        "gdp": output_dir / "seco_gdp_quarterly_selected.csv",
        "panel": output_dir / "labour_market_quarterly_merged.csv",
    }
    household.to_csv(outputs["household"], index=False)
    firm.to_csv(outputs["firm"], index=False)
    unemployment_monthly.to_csv(outputs["unemployment"], index=False)
    gdp.to_csv(outputs["gdp"], index=False)
    panel.to_csv(outputs["panel"], index=False)
    return outputs


def _extract_row_series(
    table: pd.DataFrame,
    label: str,
    start_col: int,
    length: int,
) -> pd.Series:
    row = table.loc[table.iloc[:, 0] == label]
    if row.empty:
        raise ValueError(f"Could not find row '{label}'.")
    values = pd.to_numeric(row.iloc[0, start_col : start_col + length], errors="coerce")
    return values.reset_index(drop=True)


def _parse_bfs_quarter_label(label: object) -> pd.Timestamp:
    quarter_token, year_token = str(label).split("-")
    quarter_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
    quarter = quarter_map[quarter_token.strip()]
    year = int(year_token)
    month = 3 * quarter - 2
    return pd.Timestamp(year=year, month=month, day=1)


def _parse_besta_quarter_columns(year_row: pd.Series, quarter_row: pd.Series) -> pd.Index:
    dates: list[pd.Timestamp] = []
    current_year: int | None = None
    for year_token, quarter_token in zip(year_row.tolist(), quarter_row.tolist()):
        if pd.notna(year_token):
            current_year = int(year_token)
        if current_year is None or pd.isna(quarter_token):
            continue
        quarter = _quarter_from_roman(str(quarter_token))
        month = 3 * quarter - 2
        dates.append(pd.Timestamp(year=current_year, month=month, day=1))
    return pd.Index(dates, name="date")


def _quarter_from_roman(token: str) -> int:
    quarter_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
    return quarter_map[token.strip()]
