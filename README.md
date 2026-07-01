# Paid Social → Brand Awareness Impact Analysis

Estimates the impact of Meta Ads activity on brand awareness (Google Trends interest) using:

- **Correlation** (Pearson + Spearman)
- **Lag / cross-correlation** (does reach lead or lag interest?)
- **Granger causality** (does past reach help predict future interest?)
- **Difference-in-Differences (DiD)** across treated vs. control countries — the causal estimate

Every statistical result comes with an auto-generated plain-English interpretation.

## Project structure

```
.
├── notebook.ipynb          # Main analysis notebook — start here
├── src/
│   ├── data_loader.py       # CSV loading, cleaning, merging, synthetic demo data
│   ├── stats_analysis.py    # Correlation, cross-correlation, Granger causality
│   ├── causal_inference.py  # DiD regression, parallel-trends check, placebo test
│   ├── interpretation.py    # Stats results -> plain-English strings
│   └── viz.py                # Plotly chart builders
├── data/                    # Put your CSV exports here (see schema below)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Your data

Drop your exports into `data/` with these exact filenames (or edit the paths in the notebook's
"Load data" cell):

- `data/google_trends.csv`
- `data/meta_ads.csv`

**Google Trends CSV columns:** `brand, Country, Date, ParsedDate, interest, Geo`
**Meta Ads CSV columns:** `campaign name, Date, Parsed date, Reach, brand, Country`

`Geo` (Trends) and `Country` (Ads) should both be ISO country codes (e.g. `US`, `GB`, `DE`) so the
two datasets can be joined.

If no files are found, the notebook auto-generates synthetic demo data (clearly labeled) so you
can see the whole pipeline run before plugging in real data.

## How the DiD section works

1. **Treated countries** = countries with any Meta Ads spend for the brand.
   **Control countries** = countries with none. Override this in the notebook if your actual
   experiment design differs (e.g. you want to compare two *specific* markets rather than all
   spend vs. no spend).
2. **Treatment start date** = first date any treated country received spend. Override if your
   campaigns started at different times in different countries — you may want to run DiD
   separately per campaign wave.
3. The notebook runs a **parallel-trends check** (pre-period visual) and a **placebo test**
   (fake earlier treatment date) automatically, since DiD's causal claim depends entirely on
   treated/control countries having moved in parallel absent the campaign.

## Caveats worth keeping in mind

- **Granger causality tests predictive precedence in a time series, not true causation.**
  It's evidence, not proof — omitted variables that move both reach and interest (e.g. a PR
  event, competitor activity, seasonality) can create spurious Granger-significance.
- **DiD's validity depends on parallel trends.** If treated and control countries were already
  diverging before the campaign, the estimate is biased. Always check the parallel-trends chart
  and placebo test result before trusting the headline number.
- **Weekly aggregation** is the default (matching Google Trends' typical resolution). If your
  Meta Ads data has meaningful day-of-week patterns you want to preserve, you can change
  `freq="W"` to `freq="D"` in `merge_datasets()` — just note Google Trends daily data is often
  noisier and may need smoothing first.
- Country-level DiD assumes campaigns didn't have cross-border spillover (e.g. someone in a
  control country seeing an ad meant for a treated country via VPN or shared social feeds). Worth
  a gut-check for brands with highly overlapping regional audiences.
