"""
interpretation.py
-------------------
Turns raw statistical output (correlation dicts, lag tables, Granger
results, DiD summaries) into plain-English interpretation strings that
get printed / displayed in the notebook next to each analysis.

These are template-based, not LLM-generated — deterministic, auditable,
and don't require any API key to run.
"""

from __future__ import annotations


def _strength_label(r: float) -> str:
    r = abs(r)
    if r >= 0.7:
        return "strong"
    if r >= 0.4:
        return "moderate"
    if r >= 0.2:
        return "weak"
    return "negligible"


def interpret_correlation(corr: dict, brand: str, geo: str) -> str:
    if corr["n"] < 3 or corr["pearson_r"] != corr["pearson_r"]:  # NaN check
        return (f"[{brand} / {geo}] Not enough overlapping data points "
                f"({corr['n']}) to compute a reliable correlation.")

    r = corr["pearson_r"]
    p = corr["pearson_p"]
    direction = "positive" if r > 0 else "negative"
    strength = _strength_label(r)
    sig = "statistically significant (p < 0.05)" if p < 0.05 else \
          "not statistically significant (p >= 0.05)"

    return (
        f"[{brand} / {geo}] Reach and interest show a {strength} {direction} "
        f"correlation (Pearson r = {r:.2f}, p = {p:.3f}, n = {corr['n']}), "
        f"which is {sig}. Spearman rank correlation is {corr['spearman_r']:.2f} "
        f"(p = {corr['spearman_p']:.3f}), suggesting the relationship is "
        f"{'roughly monotonic and consistent with the linear estimate' if abs(corr['spearman_r'] - r) < 0.15 else 'somewhat non-linear — treat the Pearson estimate with caution'}."
    )


def interpret_lag(best_lag_result: dict, brand: str, geo: str, freq_label: str = "period") -> str:
    lag = best_lag_result["lag"]
    r = best_lag_result["correlation"]

    if lag is None:
        return f"[{brand} / {geo}] No valid lag correlation could be computed."

    strength = _strength_label(r)
    if lag > 0:
        direction = (f"reach leads interest by {lag} {freq_label}(s) — spend today is "
                     f"most associated with an interest change {lag} {freq_label}(s) later")
    elif lag < 0:
        direction = (f"interest leads reach by {abs(lag)} {freq_label}(s) — interest moves "
                     f"*before* reach changes, which is more consistent with reactive "
                     f"budget allocation (spend chasing existing demand) than with ads "
                     f"driving awareness")
    else:
        direction = "reach and interest move together with no detectable lag"

    return (f"[{brand} / {geo}] Strongest cross-correlation ({strength}, r = {r:.2f}) "
            f"occurs at lag = {lag}: {direction}.")


def interpret_granger(granger_result: dict, brand: str, geo: str) -> str:
    if not granger_result.get("ok"):
        return f"[{brand} / {geo}] Granger test could not be run: {granger_result.get('reason')}"

    best_lag = granger_result["best_lag"]
    p = granger_result["best_p_value"]
    sig = granger_result["significant_at_0.05"]

    verdict = (
        f"there is statistically significant evidence that past Meta Ads reach "
        f"helps predict future Google Trends interest beyond interest's own history "
        f"(Granger-causal at lag {best_lag}, p = {p:.3f})"
        if sig else
        f"there is not enough evidence to conclude reach Granger-causes interest "
        f"(best p = {p:.3f} at lag {best_lag}, above the 0.05 threshold)"
    )

    note = ""
    if granger_result.get("x_differenced") or granger_result.get("y_differenced"):
        note = " (series were differenced first to satisfy stationarity requirements)"

    return f"[{brand} / {geo}] Granger causality test: {verdict}{note}."


def interpret_did(did_result: dict, treated_geos: list, control_geos: list,
                   outcome: str = "interest") -> str:
    effect = did_result["did_effect"]
    p = did_result["p_value"]
    ci_low, ci_high = did_result["ci_low"], did_result["ci_high"]
    sig = did_result["significant_at_0.05"]

    direction = "increase" if effect > 0 else "decrease"
    treated_str = ", ".join(treated_geos)
    control_str = ", ".join(control_geos)

    if sig:
        verdict = (
            f"The Difference-in-Differences estimate suggests the campaign caused an "
            f"average {direction} of {abs(effect):.2f} points in {outcome} in the "
            f"treated countries ({treated_str}), relative to the counterfactual implied "
            f"by the control countries ({control_str}). This effect is statistically "
            f"significant (p = {p:.3f}, 95% CI [{ci_low:.2f}, {ci_high:.2f}])."
        )
    else:
        verdict = (
            f"The Difference-in-Differences estimate is {effect:+.2f} points in "
            f"{outcome} for treated countries ({treated_str}) vs. control "
            f"({control_str}), but this is not statistically significant "
            f"(p = {p:.3f}, 95% CI [{ci_low:.2f}, {ci_high:.2f}]). "
            f"On this data, we cannot confidently attribute a change in {outcome} "
            f"to the campaign."
        )

    return verdict


def interpret_placebo(placebo_result: dict) -> str:
    p = placebo_result["p_value"]
    effect = placebo_result["did_effect"]
    if placebo_result["significant_at_0.05"]:
        return (
            f"⚠️ Placebo check FAILED: using a fake pre-campaign treatment date still "
            f"produces a significant effect ({effect:+.2f}, p = {p:.3f}). This suggests "
            f"treated and control countries were already diverging before the real "
            f"campaign started, so the main DiD estimate may be confounded rather than "
            f"purely causal — inspect the parallel-trends chart closely."
        )
    return (
        f"✅ Placebo check passed: the fake pre-campaign treatment date shows no "
        f"significant effect ({effect:+.2f}, p = {p:.3f}), which is consistent with "
        f"the parallel-trends assumption and supports treating the main DiD estimate "
        f"as causal."
    )
