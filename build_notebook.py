import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ---------------------------------------------------------------------------
md(r"""# Paid Social Impact on Brand Awareness
### Meta Ads Reach vs. Google Trends Interest — Correlation, Lag, Granger Causality & Difference-in-Differences

This notebook estimates whether paid social activity (Meta Ads) moves brand awareness
(Google Trends interest), using four complementary lenses:

1. **Correlation** — Pearson/Spearman, same-period association between reach and interest.
2. **Lag / cross-correlation** — does reach move *before* interest (consistent with ads driving
   awareness) or after (consistent with budget chasing existing demand)?
3. **Granger causality** — a formal statistical test of whether past reach improves the
   prediction of future interest, beyond interest's own history.
4. **Difference-in-Differences (DiD)** — a causal-inference comparison of countries that ran a
   campaign ("treated") against countries that didn't ("control"), before vs. after the
   campaign started. This is the closest thing to a causal effect estimate this data allows,
   since we don't have a randomized experiment.

Every analysis includes an **auto-generated plain-English interpretation** printed alongside the
numbers/chart, so you don't have to translate p-values yourself.

> ⚠️ **Correlation/Granger causality are not proof of causation.** They tell you about
> predictive relationships in the *time series*. DiD is the section that gets closest to a causal
> claim, and even that rests on the *parallel trends* assumption — checked explicitly below.
""")

# ---------------------------------------------------------------------------
md("## 1. Setup")
code(r"""import sys
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
from IPython.display import display, Markdown

from src import data_loader, stats_analysis, causal_inference, interpretation, viz

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
""")

# ---------------------------------------------------------------------------
md(r"""## 2. Load data

Point this at your real exports. Expected raw columns (exactly as described in the project spec):

**Google Trends CSV** — `brand, Country, Date, ParsedDate, interest, Geo`
**Meta Ads CSV** — `campaign name, Date, Parsed date, Reach, brand, Country`

If the files below aren't found, the notebook falls back to **synthetic demo data** (clearly
labeled) so you can see the full pipeline run end-to-end before plugging in real exports.
""")
code(r"""GOOGLE_TRENDS_PATH = "data/google_trends.csv"
META_ADS_PATH = "data/meta_ads.csv"

USE_SYNTHETIC = False

try:
    trends_raw = data_loader.load_google_trends(GOOGLE_TRENDS_PATH)
    ads_raw = data_loader.load_meta_ads(META_ADS_PATH)
    print(f"Loaded real data: {len(trends_raw)} trends rows, {len(ads_raw)} ads rows")
except FileNotFoundError:
    USE_SYNTHETIC = True
    print("Real data files not found under data/ — generating synthetic demo data instead.\n"
          "Drop your real CSVs at the paths above and re-run this cell to use your own data.")
    trends_raw_synth, ads_raw_synth = data_loader.generate_synthetic_data()
    # Write to disk then reload through the *real* loaders, so the exact same
    # parsing/validation code path is exercised for demo and real data alike.
    import os
    os.makedirs("data", exist_ok=True)
    trends_raw_synth.to_csv(GOOGLE_TRENDS_PATH, index=False)
    ads_raw_synth.to_csv(META_ADS_PATH, index=False)
    trends_raw = data_loader.load_google_trends(GOOGLE_TRENDS_PATH)
    ads_raw = data_loader.load_meta_ads(META_ADS_PATH)
    print(f"Synthetic data generated: {len(trends_raw)} trends rows, {len(ads_raw)} ads rows")

trends_raw.head()
""")
code(r"""ads_raw.head()""")

# ---------------------------------------------------------------------------
md(r"""## 3. Merge into a tidy analysis panel

Both datasets are aggregated to a common weekly grain (Google Trends' native resolution) and
merged on `brand + geo + date`. Weeks with no ad spend get `reach = 0` rather than being dropped,
so pre-campaign baselines are preserved.
""")
code(r"""panel = data_loader.merge_datasets(trends_raw, ads_raw, freq="W")
print(f"Panel shape: {panel.shape}")
print(f"Brands: {panel['brand'].unique().tolist()}")
print(f"Geos: {panel['geo'].unique().tolist()}")
print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
panel.head(10)
""")

# ---------------------------------------------------------------------------
md(r"""## 4. Configure the analysis

Pick the brand and country/countries to analyze. For the DiD section you'll also need to specify
which countries were **treated** (ran a campaign) vs. **control** (didn't), and the campaign
start date.
""")
code(r"""BRAND = panel["brand"].unique()[0]          # <-- change to your brand
GEOS_TO_ANALYZE = sorted(panel[panel["brand"] == BRAND]["geo"].unique())

print(f"Analyzing brand: {BRAND}")
print(f"Available geos: {GEOS_TO_ANALYZE}")
""")

# ---------------------------------------------------------------------------
md(r"""## 5. Correlation, lag correlation & Granger causality (per country)

Run for every geo available for the selected brand. Adjust `MAX_LAG` to the number of
weeks you think a reasonable maximum delay between spend and awareness could be.
""")
code(r"""MAX_LAG = 6  # weeks

results = {}

for geo in GEOS_TO_ANALYZE:
    sub = panel[(panel["brand"] == BRAND) & (panel["geo"] == geo)].sort_values("date")
    if sub["reach"].sum() == 0:
        print(f"Skipping {geo}: no ad spend recorded for this brand/geo.")
        continue

    corr = stats_analysis.compute_correlation(sub["reach"], sub["interest"])
    cc = stats_analysis.cross_correlation(sub["reach"], sub["interest"], max_lag=MAX_LAG)
    bl = stats_analysis.best_lag(cc)
    granger = stats_analysis.granger_causality(sub["reach"], sub["interest"], max_lag=min(MAX_LAG, 4))

    results[geo] = {"panel": sub, "correlation": corr, "cross_corr": cc,
                     "best_lag": bl, "granger": granger}

    display(Markdown(f"### {BRAND} / {geo}"))
    fig = viz.plot_dual_axis_timeseries(panel, BRAND, geo)
    fig.show()

    display(Markdown(f"**Correlation:** {interpretation.interpret_correlation(corr, BRAND, geo)}"))

    fig = viz.plot_cross_correlation(cc, BRAND, geo, freq_label="week")
    fig.show()
    display(Markdown(f"**Lag:** {interpretation.interpret_lag(bl, BRAND, geo, freq_label='week')}"))

    if granger.get("ok"):
        fig = viz.plot_granger_pvalues(granger["p_values_by_lag"], BRAND, geo)
        fig.show()
    display(Markdown(f"**Granger causality:** {interpretation.interpret_granger(granger, BRAND, geo)}"))
    display(Markdown("---"))
""")

# ---------------------------------------------------------------------------
md(r"""### Summary table across countries""")
code(r"""summary_rows = []
for geo, r in results.items():
    summary_rows.append({
        "geo": geo,
        "pearson_r": round(r["correlation"]["pearson_r"], 3) if r["correlation"]["n"] >= 3 else np.nan,
        "pearson_p": round(r["correlation"]["pearson_p"], 4) if r["correlation"]["n"] >= 3 else np.nan,
        "best_lag_weeks": r["best_lag"]["lag"],
        "best_lag_corr": round(r["best_lag"]["correlation"], 3) if r["best_lag"]["correlation"] == r["best_lag"]["correlation"] else np.nan,
        "granger_significant": r["granger"].get("significant_at_0.05", False),
        "granger_best_p": r["granger"].get("best_p_value", np.nan),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("pearson_r", ascending=False)
summary_df
""")

# ---------------------------------------------------------------------------
md(r"""## 6. Causal inference — Difference-in-Differences across countries

Compare **treated** countries (ran a Meta Ads campaign) to **control** countries (didn't), before
vs. after the campaign start date. The DiD coefficient is the estimated causal lift in interest
attributable to the campaign, netting out (a) baseline differences between countries and
(b) any organic trend common to all of them.

**Set these three things based on your actual campaign:**
""")
code(r"""TREATED_GEOS = [g for g in GEOS_TO_ANALYZE if panel[(panel.brand==BRAND) & (panel.geo==g)]["reach"].sum() > 0]
CONTROL_GEOS = [g for g in GEOS_TO_ANALYZE if g not in TREATED_GEOS]

# Campaign start = first week with non-zero reach across treated geos
TREATMENT_START = panel[(panel["brand"] == BRAND) & (panel["geo"].isin(TREATED_GEOS)) & (panel["reach"] > 0)]["date"].min()

print(f"Treated geos: {TREATED_GEOS}")
print(f"Control geos: {CONTROL_GEOS}")
print(f"Treatment start (inferred): {TREATMENT_START}")
print("\nOverride TREATED_GEOS / CONTROL_GEOS / TREATMENT_START above if this inference is wrong.")
""")

# ---------------------------------------------------------------------------
md(r"""### 6a. Parallel trends check

DiD assumes treated and control countries would have moved *in parallel* had the campaign never
happened. We can't observe that counterfactual directly, but we can check whether they moved in
parallel **before** the campaign — if they didn't, the DiD estimate below should be treated with
caution.
""")
code(r"""if not CONTROL_GEOS:
    print("No control geos found — DiD requires at least one country with no campaign to serve "
          "as the counterfactual. Set CONTROL_GEOS manually above.")
else:
    pre_trends = causal_inference.parallel_trends_check(panel, TREATED_GEOS, CONTROL_GEOS, TREATMENT_START)
    fig = viz.plot_parallel_trends(pre_trends)
    fig.show()
""")

# ---------------------------------------------------------------------------
md("### 6b. DiD estimate")
code(r"""if CONTROL_GEOS:
    did_panel = causal_inference.build_did_panel(
        panel, TREATED_GEOS, CONTROL_GEOS, TREATMENT_START, brand=BRAND
    )
    did_result = causal_inference.run_did_regression(did_panel)
    did_sum = causal_inference.did_summary(did_result)

    fig = viz.plot_did_trends(panel[panel.brand == BRAND], TREATED_GEOS, CONTROL_GEOS, TREATMENT_START)
    fig.show()

    fig = viz.plot_did_effect_bar(did_sum)
    fig.show()

    display(Markdown(f"**Interpretation:** {interpretation.interpret_did(did_sum, TREATED_GEOS, CONTROL_GEOS)}"))

    print("\nFull regression output:")
    print(did_result.summary())
""")

# ---------------------------------------------------------------------------
md(r"""### 6c. Placebo test (robustness check)

Re-runs the DiD using a *fake* treatment date that falls before the real campaign, using only
pre-campaign data. If this "placebo" shows a significant effect too, treated and control groups
were likely already diverging for reasons unrelated to the campaign — a warning sign that the
main DiD estimate may not be purely causal.
""")
code(r"""if CONTROL_GEOS:
    pre_campaign_weeks = panel[(panel.brand == BRAND) & (panel.date < TREATMENT_START)]["date"].nunique()

    if pre_campaign_weeks >= 8:
        # Fake treatment date = halfway through the pre-campaign period
        pre_dates = sorted(panel[(panel.brand == BRAND) & (panel.date < TREATMENT_START)]["date"].unique())
        fake_start = pre_dates[len(pre_dates) // 2]

        placebo_result = causal_inference.placebo_test(
            panel, TREATED_GEOS, CONTROL_GEOS,
            fake_treatment_start=fake_start, real_treatment_start=TREATMENT_START, brand=BRAND,
        )
        display(Markdown(interpretation.interpret_placebo(placebo_result)))
    else:
        print(f"Only {pre_campaign_weeks} pre-campaign weeks available — need at least 8 for a "
              f"meaningful placebo test. Skipping.")
""")

# ---------------------------------------------------------------------------
md(r"""## 7. Executive summary

Auto-compiled from the results above — a quick narrative you can drop into a deck or share with
stakeholders directly.
""")
code(r"""lines = [f"## Executive Summary — {BRAND}\n"]

if not summary_df.empty:
    strongest = summary_df.iloc[0]
    lines.append(
        f"- Across {len(summary_df)} countries analyzed, the strongest same-period correlation "
        f"between reach and interest was in **{strongest['geo']}** (r = {strongest['pearson_r']}, "
        f"p = {strongest['pearson_p']})."
    )
    n_granger_sig = int(summary_df["granger_significant"].sum())
    lines.append(
        f"- Granger causality (reach → interest) was statistically significant in "
        f"**{n_granger_sig} of {len(summary_df)}** countries tested."
    )
    reach_leads = summary_df[summary_df["best_lag_weeks"] > 0]
    if not reach_leads.empty:
        lines.append(
            f"- In **{len(reach_leads)}** countries, the strongest lag correlation had reach "
            f"leading interest (consistent with ads driving awareness rather than the reverse)."
        )

if CONTROL_GEOS:
    lines.append(
        f"- **DiD causal estimate:** {interpretation.interpret_did(did_sum, TREATED_GEOS, CONTROL_GEOS)}"
    )
else:
    lines.append("- **DiD causal estimate:** not computed — no control countries were available.")

display(Markdown("\n".join(lines)))
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

with open("notebook.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written.")
