"""
viz.py
------
Plotly chart builders for the reach-vs-interest analysis. All functions
return a `go.Figure` so the notebook can display / further customize them.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


TEMPLATE = "plotly_white"


def plot_dual_axis_timeseries(df: pd.DataFrame, brand: str, geo: str) -> go.Figure:
    """Reach (bars) and interest (line) over time on twin y-axes."""
    sub = df[(df["brand"] == brand) & (df["geo"] == geo)].sort_values("date")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=sub["date"], y=sub["reach"], name="Meta Reach",
               marker_color="rgba(66, 133, 244, 0.55)"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=sub["date"], y=sub["interest"], name="Google Trends Interest",
                    mode="lines+markers", line=dict(color="#EA4335", width=2)),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Reach", secondary_y=False)
    fig.update_yaxes(title_text="Interest (0-100)", secondary_y=True)
    fig.update_layout(
        title=f"Reach vs. Google Trends Interest — {brand} / {geo}",
        template=TEMPLATE, hovermode="x unified", legend=dict(orientation="h", y=1.12),
    )
    return fig


def plot_cross_correlation(cc_df: pd.DataFrame, brand: str, geo: str, freq_label="period") -> go.Figure:
    """Bar chart of correlation by lag, highlighting the strongest lag."""
    best_idx = cc_df["correlation"].abs().idxmax() if cc_df["correlation"].notna().any() else None
    colors = [
        "#EA4335" if best_idx is not None and i == best_idx else "#4285F4"
        for i in cc_df.index
    ]

    fig = go.Figure(go.Bar(x=cc_df["lag"], y=cc_df["correlation"], marker_color=colors))
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=f"Cross-Correlation: Reach vs Interest — {brand} / {geo}<br>"
              f"<sup>Positive lag = reach leads interest (each {freq_label})</sup>",
        xaxis_title=f"Lag ({freq_label}s)", yaxis_title="Correlation coefficient",
        template=TEMPLATE,
    )
    return fig


def plot_granger_pvalues(p_values_by_lag: dict, brand: str, geo: str) -> go.Figure:
    lags = list(p_values_by_lag.keys())
    pvals = list(p_values_by_lag.values())
    fig = go.Figure(go.Bar(x=lags, y=pvals, marker_color="#34A853"))
    fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                  annotation_text="p = 0.05 significance threshold")
    fig.update_layout(
        title=f"Granger Causality p-values by Lag — {brand} / {geo}<br>"
              f"<sup>Lower is stronger evidence reach → interest</sup>",
        xaxis_title="Lag", yaxis_title="p-value", template=TEMPLATE,
    )
    return fig


def plot_did_trends(
    panel_df: pd.DataFrame,
    treated_geos: list[str],
    control_geos: list[str],
    treatment_start,
    outcome: str = "interest",
) -> go.Figure:
    """Average outcome over time for treated vs control groups, with a
    vertical line marking campaign start — the classic DiD visual."""
    df = panel_df.copy()
    df["group"] = df["geo"].apply(
        lambda g: "Treated" if g in treated_geos else ("Control" if g in control_geos else None)
    )
    df = df.dropna(subset=["group"])
    grouped = df.groupby(["date", "group"], as_index=False)[outcome].mean()

    fig = go.Figure()
    for grp, color in [("Treated", "#EA4335"), ("Control", "#4285F4")]:
        gdf = grouped[grouped["group"] == grp]
        fig.add_trace(go.Scatter(
            x=gdf["date"], y=gdf[outcome], mode="lines+markers", name=grp,
            line=dict(color=color, width=2),
        ))

    fig.add_vline(x=pd.Timestamp(treatment_start), line_dash="dash", line_color="black",
                  annotation_text="Campaign start", annotation_position="top")
    fig.update_layout(
        title=f"DiD: {outcome.title()} — Treated vs Control Countries",
        xaxis_title="Date", yaxis_title=outcome.title(), template=TEMPLATE,
        hovermode="x unified", legend=dict(orientation="h", y=1.12),
    )
    return fig


def plot_did_effect_bar(did_summary_dict: dict, label: str = "DiD Effect") -> go.Figure:
    """Single bar with 95% CI whiskers for the estimated treatment effect."""
    effect = did_summary_dict["did_effect"]
    ci_low, ci_high = did_summary_dict["ci_low"], did_summary_dict["ci_high"]
    err_plus = ci_high - effect
    err_minus = effect - ci_low
    color = "#34A853" if did_summary_dict["significant_at_0.05"] else "#9AA0A6"

    fig = go.Figure(go.Bar(
        x=[label], y=[effect], marker_color=color,
        error_y=dict(type="data", symmetric=False, array=[err_plus], arrayminus=[err_minus]),
    ))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title=f"Estimated Causal Effect (95% CI) — p = {did_summary_dict['p_value']:.3f}",
        yaxis_title="Effect on interest (points)", template=TEMPLATE, showlegend=False,
    )
    return fig


def plot_parallel_trends(pre_df: pd.DataFrame) -> go.Figure:
    """Pre-period treated vs control means — visual check of the DiD
    parallel-trends assumption."""
    fig = go.Figure()
    if "treated" in pre_df.columns:
        fig.add_trace(go.Scatter(x=pre_df["date"], y=pre_df["treated"],
                                   mode="lines+markers", name="Treated (pre-period)",
                                   line=dict(color="#EA4335")))
    if "control" in pre_df.columns:
        fig.add_trace(go.Scatter(x=pre_df["date"], y=pre_df["control"],
                                   mode="lines+markers", name="Control (pre-period)",
                                   line=dict(color="#4285F4")))
    fig.update_layout(
        title="Pre-Campaign Parallel Trends Check",
        xaxis_title="Date", yaxis_title="Mean interest", template=TEMPLATE,
        hovermode="x unified",
    )
    return fig
