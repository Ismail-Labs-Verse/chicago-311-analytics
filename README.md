# Chicago 311 Request Analytics

An end-to-end analysis of **13.07 million** Chicago 311 service requests (2019–2025),
asking two operational questions:

1. **Equity** — does how long a request takes to resolve depend on where it was made,
   after accounting for what was requested?
2. **Capacity** — can future demand be anticipated well enough to size crews and the
   contact centre in advance?

The answer to the first question turned out to depend on a data-quality decision that
most 311 analyses get wrong, and that is the most interesting part of this repository.

---

## The finding that changes everything else

**56% of records in this dataset are opened and closed within one minute.** They are
citizen information calls and auto-closed noise reports — filings that never involve
dispatching anyone.

Average them in and you get garbage:

| Scope | Median "resolution time" |
|---|---|
| All records | **0.00001 days** (~0.9 minutes) |
| Actionable service requests only | **1.26 days** |

Any analysis that reports a single citywide 311 response time for Chicago is
measuring the contact centre's auto-close behaviour, not service delivery. This
repository therefore carries **two explicit scopes** and never mixes them:

- **Demand scope** — every request. Used for volume, seasonality, and forecasting,
  because a phone call is still load on the system.
- **Service scope** (`is_actionable`) — real field work only. Used for **every**
  response-time, equity, and severity statistic.

The filings are *flagged, not deleted*, so the decision is auditable.

---

## Selected findings

| # | Finding | Evidence |
|---|---|---|
| 1 | 20% sample is unbiased in time | sampled share averages **20.002%** across 71 validated months against exact API counts |
| 2 | **58.4%** of records close in under one minute — the single decision that changes every other number | all-row median **0.000012 days** vs **5.75 days** for actionable requests |
| 3 | Tract racial composition is **associated with** longer waits, but only marginally | WLS weighted by resolved volume: `pct_minority` coef **+0.0254**, p = **0.064** (not significant at 5%); income p = 1.00 |
| 4 | Slow tracts **cluster spatially** rather than scattering | Moran's I = **0.587**, permutation p = 0.001; **122** tracts in significant high–high clusters — this is the strongest equity evidence in the project |
| 5 | **Null result:** the 1930s HOLC grade gradient does *not* appear in response time | grade D **5.53 d** vs grade A **6.30 d** (ratio 0.88×, one-sided p = 0.83 — not supported; grade A has only 6 tracts) |
| 6 | Seasonal naive is a genuinely strong forecasting baseline | 12-month hold-out + 24-fold rolling-origin backtest |
| 7 | Isolation Forest isolates 16 anomalous weeks | e.g. week of 2025-08-11: **45,513** requests vs 7,492 seasonal-naive expectation |

> **Findings 3 and 5 are stated as weaker results on purpose.** A redlining narrative
> is the expected headline, but the data does not support it: redlined tracts are much
> less white today, yet their 311 response times are marginally *faster*, and the
> difference is not statistically significant (grade A has only 6 tracts). The
> composition regression runs in the expected direction but is marginal (p = 0.064),
> and it is sensitive to the 13 tracts whose ACS income estimate is unavailable.
> What survives scrutiny is **spatial clustering** (Finding 4): slow tracts are slow
> together, and LISA localises 122 of them.
>
> One earlier version of this analysis reported `pct_minority` at p = 1.7e-10. That
> figure was an artefact of the Census `-666666666` sentinel for "estimate not
> available" flowing into the income control; correcting it moved the result to
> p = 0.064. The corrected number is the one reported.

Every one of these numbers is produced by an executed notebook and written to
`data/processed/findings.json`. The PDF report reads its statistics from that file —
nothing in the prose was transcribed by hand.

---

## Architecture

```
data.cityofchicago.org ─┐
NOAA NCEI ──────────────┤
US Census (keyless) ────┤   src/fetch.py        resumable, month-chunked,
HOLC / Mapping Ineq. ───┘                       retry with backoff
                                    │
                                    ▼
                        data/raw/            (never committed)
                                    │
                                    ▼
                        src/etl.py + sql/01_ingest.sql
                          • clean & type dates
                          • severity index from one ruleset
                          • point-in-polygon → 2020 census tracts
                          • join ACS, weather, HOLC, hardship
                          • flag is_actionable
                                    │
                                    ▼
                  data/processed/chicago311.duckdb
                    data/processed/data_quality_report.json
                           │                    │
             ┌─────────────┴──────┐             │
             ▼                    ▼             ▼
    notebooks/01–04        dashboards/     reports/case_study.pdf
    (executed)             .hyper / PBIP / HTML
```

### Why the pipeline is shaped this way

- **Tracts are computed, not assumed.** The dataset has *no* census tract field —
  only community area, ward, and coordinates. Tracts come from a real
  point-in-polygon test (DuckDB spatial, R-tree indexed) against 2020 boundaries.
- **The severity index is data, not code.** The ruleset lives in
  `src/config.py` and is loaded into DuckDB as a table, so the SQL and the Python
  cannot disagree about what "critical" means.
- **Sampling is validated, not asserted.** Sampled months are checked against exact
  full-population monthly counts pulled from the API.
- **The pipeline is resumable.** The 311 pull is chunked by month with a `.done`
  marker, so an interrupted 25-minute run resumes rather than restarts.

---

## Repository layout

```
├── sql/01_ingest.sql            # the whole relational + spatial pipeline
├── src/
│   ├── config.py                # every dataset ID, date boundary, and rule
│   ├── fetch.py                 # resumable collection
│   ├── etl.py                   # runs the SQL + quality report + sample export
│   ├── analysis_helpers.py      # shared, scope-aware metric helpers
│   ├── notebook_src/*.py        # notebook sources (readable, diffable)
│   ├── generate_notebooks.py    # builds + EXECUTES the notebooks
│   ├── build_hyper.py           # Tableau .hyper + tract GeoJSON
│   ├── build_powerbi.py         # Power BI PBIP (TMDL + report)
│   ├── build_html_dashboard.py  # self-contained HTML dashboard
│   ├── build_report.py          # the 10-page PDF case study
│   └── render_pdf.js            # headless Chrome → PDF with page numbers
├── notebooks/01–04              # executed notebooks with real outputs
├── dashboards/                  # .hyper, PBIP, HTML dashboard
├── reports/case_study.pdf       # the case study
└── data/samples/                # 1,000-row representative sample (committed)
```

**Data policy.** `data/raw/` and `data/processed/` are git-ignored — they are large
and regenerable. The only data committed is `data/samples/` (~1,000 cleaned rows,
stratified by year) so the schema and cleaning are reviewable without the bulk.

---

## Reproducing

```bash
conda create -n chicago311 -c conda-forge python=3.12 \
    duckdb lightgbm geopandas pyarrow jupyter statsmodels \
    scikit-learn matplotlib seaborn scipy
conda activate chicago311
pip install -r requirements.txt

python src/fetch.py --all           # collect every source (~25 min, resumable)
python src/etl.py                   # warehouse + data-quality report (~2 min)
python src/generate_notebooks.py    # build AND execute notebooks 01–04
python src/build_hyper.py           # Tableau extract + choropleth GeoJSON
python src/build_powerbi.py         # Power BI project
python src/build_html_dashboard.py  # self-contained dashboard
python src/build_report.py          # the PDF (requires node + Brave/Chrome)
```

Verified against Python **3.12.15**, duckdb 1.5.6, lightgbm 4.7.0, geopandas 1.2.0,
statsmodels 0.15.0, scikit-learn 1.9.1 on Apple Silicon macOS.

---

## Deliverables

| Deliverable | Path | Status |
|---|---|---|
| Analytical warehouse | `data/processed/chicago311.duckdb` | built from 13.07M source rows |
| Data quality report | `reports/data_quality_report.md` | **generated from the real run** |
| 4 analysis notebooks | `notebooks/01–04.ipynb` | **executed**, outputs stored |
| Tableau extract | `dashboards/chicago311_extract.hyper` | written and **read back verified** |
| Tableau choropleth layer | `dashboards/spatial/chicago311_tract_equity.geojson` | tract polygons + equity metrics |
| Power BI project | `dashboards/powerbi/chicago311.pbip` | schema-checked, **not opened in Power BI** |
| HTML dashboard | `dashboards/chicago311_dashboard.html` | self-contained, open in any browser |
| Case study | `reports/case_study.pdf` | 10 pages, figures inlined |

---

## Known limitations — read before quoting anything

**About the data**

- The analysis is **observational**. The demographic associations are correlates, not
  causes. Nothing here identifies a mechanism.
- **Crew routing, staffing, budgets, and intake channel are not observed.** No claim
  is made that any City employee behaves differently by neighbourhood.
- `days_to_resolution` records **closure in the system**, which is an administrative
  act — not necessarily the date the work was done.
- HOLC grades cover only the mapped portion of the city (~77% of requests); redlining
  results describe that portion only.
- ~0.4% of rows could not be assigned to a tract (no coordinates, or points outside
  city boundaries). They are left **unassigned rather than guessed**.
- A handful of records have timestamps inconsistent with reality (closed before
  created). They are **flagged and excluded**, not silently repaired.

**About the sampling**

- Months before 2024-12 are a **20% systematic sample**; 2024-12 onward is complete
  row-level detail. The sample was validated against exact API counts (mean 20.0009%),
  but it is still a sample: individual-level response-time distributions and spatial
  statistics have less statistical power than they would at full detail. No claim in
  this repository depends on a difference smaller than sampling noise can produce.

**About the dashboards — please read**

- **Tableau Desktop and Power BI Desktop are not installed on the machine that built
  this repository.** Both dashboard deliverables were therefore *generated and verified
  as files* (the `.hyper` is read back and its row counts and schema asserted; the PBIP
  is schema-checked) but they have **not been opened or rendered** in their target
  applications. Colours, filters, and layout will likely need adjusting on first open.
- The `.hyper` extract is large and therefore git-ignored — regenerate it with
  `python src/build_hyper.py`.
- `dashboards/chicago311_dashboard.html` exists precisely so that at least one
  dashboard in this repository is verifiable immediately, by opening it.

**About the models**

- Forecasting validation spans a window containing a structural break (2020), so
  results mix regimes that may not repeat. Intake-policy changes would invalidate the
  seasonal structure entirely.
- Isolation Forest contamination (5%) is a chosen assumption, not a discovered rate.
  The ranking is the trustworthy output; the count is a convenience.

---

## Data sources and licences

| Source | Licence / terms |
|---|---|
| City of Chicago Data Portal (311, tract boundaries, socioeconomic indicators) | Open data, public domain |
| NOAA NCEI daily summaries | US Government public domain |
| US Census ACS 5-year 2023 | US Government public domain |
| HOLC grades — University of Richmond Digital Scholarship Lab, *Mapping Inequality* | Public domain map scans; cite the project |

Note: the pipeline uses the **keyless** Census table-based Summary File, because
`api.census.gov` now requires a registered API key and no credential belongs in a
public repository.
