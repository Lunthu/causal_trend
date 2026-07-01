"""
data_loader.py
--------------
Loading, cleaning, and merging of the Google Trends and Meta Ads datasets.

Expected raw schemas
=====================
Google Trends CSV
    brand        : str   - keyword tracked
    Country      : str   - country name
    Date         : datetime
    ParsedDate   : datetime
    interest     : float - Google Trends interest score (0-100)
    Geo          : str   - ISO country code

Meta Ads CSV
    campaign name : str
    Date           : datetime
    Parsed date    : datetime
    Reach          : int
    brand          : str
    Country        : str  - ISO country code (matches Geo in Trends data)

Both datasets get normalized into a common schema so the rest of the
pipeline never has to care about the original column names:

    date, brand, geo, interest        (from Trends)
    date, brand, geo, reach, campaign (from Ads, aggregated across campaigns)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Raw loaders
# ---------------------------------------------------------------------------

def load_google_trends(path: str | Path) -> pd.DataFrame:
    """Load and normalize a Google Trends export."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "brand": "brand",
        "Country": "country",
        "Date": "date_raw",
        "ParsedDate": "date",
        "interest": "interest",
        "Geo": "geo",
    }
    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        raise ValueError(f"Google Trends file is missing expected columns: {missing}")

    df = df.rename(columns=rename_map)
    df["date"] = pd.to_datetime(df["date"])
    df["brand"] = df["brand"].astype(str).str.strip()
    df["geo"] = df["geo"].astype(str).str.strip().str.upper()
    df["interest"] = pd.to_numeric(df["interest"], errors="coerce")

    df = df[["date", "brand", "country", "geo", "interest"]].drop_duplicates()
    df = df.sort_values(["brand", "geo", "date"]).reset_index(drop=True)
    return df


def load_meta_ads(path: str | Path) -> pd.DataFrame:
    """Load and normalize a Meta Ads export."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "campaign name": "campaign",
        "Date": "date_raw",
        "Parsed date": "date",
        "Reach": "reach",
        "brand": "brand",
        "Country": "geo",
    }
    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        raise ValueError(f"Meta Ads file is missing expected columns: {missing}")

    df = df.rename(columns=rename_map)
    df["date"] = pd.to_datetime(df["date"])
    df["brand"] = df["brand"].astype(str).str.strip()
    df["geo"] = df["geo"].astype(str).str.strip().str.upper()
    df["reach"] = pd.to_numeric(df["reach"], errors="coerce")

    df = df[["date", "brand", "geo", "campaign", "reach"]].drop_duplicates()
    df = df.sort_values(["brand", "geo", "date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Aggregation / merge
# ---------------------------------------------------------------------------

def aggregate_ads_daily(ads_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple campaigns per brand/geo/day into a single reach figure."""
    agg = (
        ads_df.groupby(["date", "brand", "geo"], as_index=False)
        .agg(reach=("reach", "sum"), n_campaigns=("campaign", "nunique"))
    )
    return agg


def merge_datasets(
    trends_df: pd.DataFrame,
    ads_df: pd.DataFrame,
    freq: str = "W",
) -> pd.DataFrame:
    """
    Align both datasets to a common time granularity (default weekly, matching
    Google Trends' native resolution for most lookback windows) and merge on
    brand + geo + date.

    Returns a tidy panel: date, brand, geo, interest, reach, n_campaigns
    """
    ads_daily = aggregate_ads_daily(ads_df)

    def to_period(df, value_cols, how):
        df = df.copy()
        df["period"] = df["date"].dt.to_period(freq).dt.start_time
        grouped = df.groupby(["period", "brand", "geo"], as_index=False)[value_cols].agg(how)
        grouped = grouped.rename(columns={"period": "date"})
        return grouped

    trends_agg = to_period(trends_df, ["interest"], "mean")
    ads_agg = to_period(ads_daily, ["reach", "n_campaigns"], "sum")

    merged = pd.merge(trends_agg, ads_agg, on=["date", "brand", "geo"], how="outer")
    merged["interest"] = merged["interest"].fillna(0.0)
    merged["reach"] = merged["reach"].fillna(0.0)
    merged["n_campaigns"] = merged["n_campaigns"].fillna(0).astype(int)
    merged = merged.sort_values(["brand", "geo", "date"]).reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# Synthetic demo data (used only if no real files are found)
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    brand: str = "DemoBrand",
    treated_geos=("US", "GB", "DE"),
    control_geos=("FR", "IT", "ES"),
    start: str = "2024-01-01",
    weeks: int = 52,
    campaign_start_week: int = 20,
    campaign_duration_weeks: int = 12,
    seed: int = 42,
):
    """
    Generate a plausible synthetic Google Trends + Meta Ads dataset:
      - Treated countries receive a Meta Ads campaign starting at
        `campaign_start_week` for `campaign_duration_weeks`.
      - Control countries never receive spend.
      - Interest responds to reach with a lag and noise, plus a baseline
        organic trend + seasonality, so correlation / lag / Granger / DiD
        analyses all have something real to find.

    Returns (trends_df, ads_df) in the *raw* schema described at top of file,
    so they exercise the same loaders/parsers as real exported CSVs would
    once written to disk.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=weeks * 7, freq="D")
    all_geos = list(treated_geos) + list(control_geos)
    country_names = {
        "US": "United States", "GB": "United Kingdom", "DE": "Germany",
        "FR": "France", "IT": "Italy", "ES": "Spain",
    }

    trend_rows, ads_rows = [], []
    campaign_start_date = pd.Timestamp(start) + pd.Timedelta(weeks=campaign_start_week)
    campaign_end_date = campaign_start_date + pd.Timedelta(weeks=campaign_duration_weeks)

    for geo in all_geos:
        is_treated = geo in treated_geos
        baseline = rng.uniform(15, 30)
        organic_drift = np.linspace(0, rng.uniform(-3, 5), len(dates))
        seasonality = 4 * np.sin(2 * np.pi * np.arange(len(dates)) / 60)
        noise = rng.normal(0, 2.0, len(dates))
        interest = baseline + organic_drift + seasonality + noise

        daily_reach = np.zeros(len(dates))
        if is_treated:
            in_campaign = (dates >= campaign_start_date) & (dates < campaign_end_date)
            ramp = rng.uniform(80_000, 150_000)
            campaign_noise = rng.normal(1.0, 0.15, len(dates))
            daily_reach[in_campaign] = ramp * campaign_noise[in_campaign]
            daily_reach[daily_reach < 0] = 0

            # Reach lifts interest with a ~5-8 day lag and diminishing effect
            lag_days = int(rng.integers(5, 9))
            effect = np.zeros(len(dates))
            uplift_coeff = rng.uniform(0.00004, 0.00007)
            for i in range(len(dates)):
                src = i - lag_days
                if src >= 0:
                    effect[i] = daily_reach[src] * uplift_coeff
            interest = interest + effect

        interest = np.clip(interest, 0, 100)

        for i, d in enumerate(dates):
            trend_rows.append({
                "brand": brand, "Country": country_names.get(geo, geo),
                "Date": d.strftime("%Y-%m-%d"), "ParsedDate": d,
                "interest": round(float(interest[i]), 2), "Geo": geo,
            })
            if daily_reach[i] > 0:
                ads_rows.append({
                    "campaign name": f"{brand}_{geo}_awareness_push",
                    "Date": d.strftime("%Y-%m-%d"), "Parsed date": d,
                    "Reach": int(daily_reach[i]), "brand": brand, "Country": geo,
                })

    trends_df = pd.DataFrame(trend_rows)
    ads_df = pd.DataFrame(ads_rows)
    return trends_df, ads_df
