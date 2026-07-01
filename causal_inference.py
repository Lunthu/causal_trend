"""
causal_inference.py
--------------------
Difference-in-Differences (DiD) estimation of the causal effect of a Meta
Ads campaign on Google Trends interest, comparing treated countries
(where a campaign ran) against control countries (where it didn't).

Model
=====
    interest_it = a + b*Treated_i + c*Post_t + d*(Treated_i * Post_t) + e_it

`d` (the interaction coefficient) is the DiD estimate: the average causal
effect of the campaign on interest in treated countries, net of (a) any
fixed baseline difference between treated/control countries and (b) any
time trend common to all countries.

Standard errors are clustered by country to account for within-country
autocorrelation, which is the standard practice for panel DiD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def build_did_panel(
    panel_df: pd.DataFrame,
    treated_geos: list[str],
    control_geos: list[str],
    treatment_start,
    brand: str | None = None,
) -> pd.DataFrame:
    """
    Subset the merged brand/geo/date panel down to a clean DiD dataset with
    `treated` and `post` indicator columns.

    panel_df must have columns: date, brand, geo, interest, reach
    treatment_start: str or Timestamp - date the campaign begins (the
        pre/post cutoff)
    """
    df = panel_df.copy()
    if brand is not None:
        df = df[df["brand"] == brand]

    geos = list(treated_geos) + list(control_geos)
    df = df[df["geo"].isin(geos)].copy()

    treatment_start = pd.Timestamp(treatment_start)
    df["treated"] = df["geo"].isin(treated_geos).astype(int)
    df["post"] = (df["date"] >= treatment_start).astype(int)
    df["treated_x_post"] = df["treated"] * df["post"]

    return df.sort_values(["geo", "date"]).reset_index(drop=True)


def run_did_regression(did_panel: pd.DataFrame, outcome: str = "interest"):
    """
    Fit the DiD OLS model with country-clustered standard errors.
    Returns the fitted statsmodels results object.
    """
    formula = f"{outcome} ~ treated + post + treated_x_post"
    model = smf.ols(formula, data=did_panel)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": did_panel["geo"]})
    return result


def did_summary(result) -> dict:
    """Pull the key DiD numbers out of a fitted statsmodels result."""
    coef = result.params.get("treated_x_post", np.nan)
    se = result.bse.get("treated_x_post", np.nan)
    p = result.pvalues.get("treated_x_post", np.nan)
    ci_low, ci_high = result.conf_int().loc["treated_x_post"] if "treated_x_post" in result.params else (np.nan, np.nan)
    return {
        "did_effect": coef,
        "std_error": se,
        "p_value": p,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant_at_0.05": bool(p < 0.05) if pd.notna(p) else False,
        "r_squared": result.rsquared,
    }


def parallel_trends_check(
    panel_df: pd.DataFrame,
    treated_geos: list[str],
    control_geos: list[str],
    treatment_start,
    outcome: str = "interest",
) -> pd.DataFrame:
    """
    Build a pre-period-only group-mean-by-date table so you can eyeball
    (or later formally test) whether treated and control groups moved in
    parallel before the campaign started — the key identifying assumption
    behind DiD.
    """
    treatment_start = pd.Timestamp(treatment_start)
    pre = panel_df[panel_df["date"] < treatment_start].copy()
    pre["group"] = np.where(pre["geo"].isin(treated_geos), "treated",
                      np.where(pre["geo"].isin(control_geos), "control", "other"))
    pre = pre[pre["group"] != "other"]

    out = (
        pre.groupby(["date", "group"], as_index=False)[outcome]
        .mean()
        .pivot(index="date", columns="group", values=outcome)
        .reset_index()
    )
    return out


def placebo_test(
    panel_df: pd.DataFrame,
    treated_geos: list[str],
    control_geos: list[str],
    fake_treatment_start,
    real_treatment_start,
    outcome: str = "interest",
    brand: str | None = None,
):
    """
    Re-run the DiD using a fake treatment date that falls *before* the real
    campaign start. A significant "effect" here would suggest the real DiD
    result may be confounded rather than causal; a null result strengthens
    confidence in the main estimate.
    """
    fake_treatment_start = pd.Timestamp(fake_treatment_start)
    real_treatment_start = pd.Timestamp(real_treatment_start)
    if fake_treatment_start >= real_treatment_start:
        raise ValueError("Placebo date must be before the real treatment start.")

    pre_period_only = panel_df[panel_df["date"] < real_treatment_start]
    did_panel = build_did_panel(
        pre_period_only, treated_geos, control_geos, fake_treatment_start, brand=brand
    )
    result = run_did_regression(did_panel, outcome=outcome)
    return did_summary(result)
