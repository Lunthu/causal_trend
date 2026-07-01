"""
stats_analysis.py
------------------
Correlation, lag/cross-correlation, and Granger causality between
Meta Ads reach and Google Trends interest for a single brand/geo series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller


# ---------------------------------------------------------------------------
# Simple correlation
# ---------------------------------------------------------------------------

def compute_correlation(series_x: pd.Series, series_y: pd.Series) -> dict:
    """Pearson + Spearman correlation between two aligned series."""
    x, y = series_x.align(series_y, join="inner")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]

    if len(x) < 3:
        return {"n": len(x), "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan}

    pearson_r, pearson_p = sstats.pearsonr(x, y)
    spearman_r, spearman_p = sstats.spearmanr(x, y)
    return {
        "n": len(x),
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "spearman_r": spearman_r, "spearman_p": spearman_p,
    }


# ---------------------------------------------------------------------------
# Lag / cross-correlation
# ---------------------------------------------------------------------------

def cross_correlation(
    series_x: pd.Series,
    series_y: pd.Series,
    max_lag: int = 8,
) -> pd.DataFrame:
    """
    Compute Pearson correlation between x(t) and y(t+lag) for lag in
    [-max_lag, +max_lag] (periods, matching the granularity the series
    are indexed at — e.g. weeks if the panel was built with freq='W').

    Positive lag  -> x leads y   (x today predicts y `lag` periods later)
    Negative lag  -> y leads x

    In this project x = reach, y = interest, so a positive best-lag means
    "reach moves first, interest follows" — the expected causal direction
    for a paid-media-drives-awareness hypothesis.
    """
    x, y = series_x.align(series_y, join="inner")
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)

    records = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            xs = x[: len(x) - lag] if lag > 0 else x
            ys = y[lag:]
        else:
            xs = x[-lag:]
            ys = y[: len(y) + lag]

        xs = xs.reset_index(drop=True)
        ys = ys.reset_index(drop=True)
        n = min(len(xs), len(ys))
        if n < 3:
            records.append({"lag": lag, "correlation": np.nan, "n": n})
            continue

        xs, ys = xs[:n], ys[:n]
        if xs.std() == 0 or ys.std() == 0:
            r = np.nan
        else:
            r = np.corrcoef(xs, ys)[0, 1]
        records.append({"lag": lag, "correlation": r, "n": n})

    return pd.DataFrame(records)


def best_lag(cross_corr_df: pd.DataFrame) -> dict:
    """Return the lag with the strongest absolute correlation."""
    valid = cross_corr_df.dropna(subset=["correlation"])
    if valid.empty:
        return {"lag": None, "correlation": np.nan}
    row = valid.iloc[valid["correlation"].abs().argmax()]
    return {"lag": int(row["lag"]), "correlation": float(row["correlation"])}


# ---------------------------------------------------------------------------
# Granger causality
# ---------------------------------------------------------------------------

def _make_stationary(series: pd.Series, alpha: float = 0.05) -> tuple[pd.Series, bool]:
    """Difference a series once if the ADF test fails to reject a unit root."""
    series = series.dropna()
    if len(series) < 8 or series.std() == 0:
        return series, False
    try:
        p_value = adfuller(series)[1]
    except Exception:
        return series, False
    if p_value < alpha:
        return series, False
    return series.diff().dropna(), True


def granger_causality(
    series_x: pd.Series,
    series_y: pd.Series,
    max_lag: int = 4,
    verbose: bool = False,
) -> dict:
    """
    Test whether x (reach) Granger-causes y (interest), i.e. whether past
    values of reach improve the prediction of interest beyond interest's
    own past values.

    Both series are difference-stationarized if needed (ADF test) since
    Granger causality assumes stationary inputs.

    Returns a dict with the p-value at each lag and the best (lowest-p) lag.
    """
    x, y = series_x.align(series_y, join="inner")
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)

    x_stat, x_diffed = _make_stationary(x)
    y_stat, y_diffed = _make_stationary(y)

    n = min(len(x_stat), len(y_stat))
    if n < max_lag * 3 + 5:
        return {
            "ok": False,
            "reason": f"Not enough observations ({n}) for max_lag={max_lag}. "
                      f"Need at least {max_lag * 3 + 5}.",
        }

    data = pd.DataFrame({
        "y": y_stat.tail(n).reset_index(drop=True),
        "x": x_stat.tail(n).reset_index(drop=True),
    })

    try:
        results = grangercausalitytests(data[["y", "x"]], maxlag=max_lag, verbose=verbose)
    except Exception as e:
        return {"ok": False, "reason": str(e)}

    p_values = {lag: round(res[0]["ssr_ftest"][1], 4) for lag, res in results.items()}
    best = min(p_values, key=p_values.get)

    return {
        "ok": True,
        "x_differenced": x_diffed,
        "y_differenced": y_diffed,
        "p_values_by_lag": p_values,
        "best_lag": best,
        "best_p_value": p_values[best],
        "significant_at_0.05": p_values[best] < 0.05,
    }
