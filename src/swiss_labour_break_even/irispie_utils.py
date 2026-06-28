from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def _import_irispie():
    try:
        import irispie as ir
    except ImportError as exc:
        raise ImportError(
            "irispie is not installed. Run `pip install -r requirements.txt` in a Python 3.11+ environment."
        ) from exc
    return ir


def dataframe_to_irispie_series(
    frame: pd.DataFrame,
    columns: Iterable[str],
    date_col: str = "date",
):
    """
    Convert selected pandas columns into IRISpie monthly Series objects.

    This helper assumes the data are monthly and date-stamped at month start.
    """
    ir = _import_irispie()
    monthly = frame.sort_values(date_col).reset_index(drop=True)
    start = monthly.loc[0, date_col]
    start_period = ir.mm(int(start.year), int(start.month))

    result = {}
    for column in columns:
        values = monthly[column].to_numpy(dtype=float).reshape(-1, 1)
        result[column] = ir.Series(
            start=start_period,
            values=values,
            description=column,
        )

    return result


def results_to_irispie_databox(results: pd.DataFrame):
    """
    Return an IRISpie-style container of the main result series.

    If `Databox` is available in the installed IRISpie version, this returns one.
    Otherwise it falls back to a plain dictionary of `Series` objects.
    """
    ir = _import_irispie()
    series_map = dataframe_to_irispie_series(
        results,
        columns=[
            "unemployment_rate",
            "vacancy_growth",
            "break_even_rate_smoothed",
            "unemployment_gap_smoothed",
        ],
    )

    databox_ctor = getattr(ir, "Databox", None)
    if databox_ctor is None:
        return series_map

    databox = databox_ctor()
    for name, series in series_map.items():
        databox[name] = series
    return databox
