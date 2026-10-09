# %% [markdown]
# # 01 — Exploratory Data Analysis
# ## Chicago 311 Service Requests, 2019–2025
#
# **Purpose.** Establish what is actually in the dataset before any inference:
# volume, composition, temporal structure, and — most importantly — the
# boundary between *real field work* and *administratively closed filings*.
#
# **Two scopes are used throughout this project and are never mixed:**
#
# | Scope | Definition | Used for |
# |---|---|---|
# | **Demand** | every request, including information calls | volume, seasonality, forecasting |
# | **Service** | `is_actionable` rows only | every response-time statistic |
#
# The reason for the split is demonstrated in §3.

# %%
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
import analysis_helpers as H

H.setup_plot_style()
pd.set_option("display.width", 160, "display.max_columns", 60)
con = H.connect()

print(f"warehouse : {C.DB_PATH.relative_to(PROJECT_ROOT)}")
print(f"window    : {C.START_DATE} .. 2025-12-31")

# %% [markdown]
# ## 1. Scope and shape of the analytical table

# %%
overview = H.query("""
    SELECT
        count(*)                                              AS rows_total,
        count(DISTINCT month)                                 AS months,
        min(month)                                            AS first_month,
        max(month)                                            AS last_month,
        count(DISTINCT sr_type)                               AS request_types,
        count(DISTINCT tract_fips)                            AS tracts,
        count(DISTINCT community_area)                        AS community_areas
    FROM analytics.sr_enriched
""", con)
overview.T

# %%
# How the data was acquired matters for interpretation: part of the window is a
# 20% systematic sample, the most recent months are complete row-level detail.
H.query("""
    SELECT
        src_detail                                     AS sampling,
        min(month)                                     AS first_month,
        max(month)                                     AS last_month,
        count(DISTINCT month)                          AS months,
        count(*)                                       AS rows
    FROM analytics.sr_enriched
    GROUP BY 1 ORDER BY 2
""", con)

# %% [markdown]
# ## 2. Is the 20% sample trustworthy?
#
# The sampled months are a systematic sample on the last digit of `sr_number`.
# `sr_number` is assigned sequentially, so the sample should be even in time,
# but that is an assumption — so it is tested directly against exact
# full-population monthly counts pulled from the API (`monthly_true_counts`).

# %%
calib = H.query("""
    SELECT
        month,
        count(*)                                              AS sample_n,
        true_count,
        round(count(*) * 100.0 / true_count, 3)               AS sample_pct,
        round(true_count / 5.0)                               AS expected_n_rounded
    FROM analytics.sr_enriched s
    JOIN analytics.monthly_true_counts t USING (month)
    WHERE s.src_detail = 'sample20'
    GROUP BY month, true_count
    ORDER BY month
""", con)

if len(calib):
    pct = calib["sample_pct"]
    print(f"months validated : {len(calib)}")
    print(f"sample share     : mean {pct.mean():.3f}%  "
          f"min {pct.min():.3f}%  max {pct.max():.3f}%")
    print(f"spread           : {(pct.max() - pct.min()):.3f} percentage points")
    print(f"\nA systematic sample on a sequentially assigned ID draws ~20% of rows in "
          f"every month, so the sample is unbiased in time and 5x weighting "
          f"reproduces monthly totals.")
    H.record_finding(
        "sample_calibration_mean_pct", round(float(pct.mean()), 3),
        f"mean sample share across {len(calib)} validated months "
        f"(min {pct.min():.3f}, max {pct.max():.3f})",
        "notebooks/01_eda.ipynb §2")
else:
    print("no sampled months overlap the ground-truth counts")

# %%
# Visual check: sampled count x 5 against the exact count.
if len(calib):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(calib["month"], calib["true_count"], lw=1.6,
            color=H.PALETTE["ink"], label="Exact count (API)")
    ax.plot(calib["month"], calib["sample_n"] * 5, lw=1.2, ls="--",
            color=H.PALETTE["accent2"], label="20% sample × 5")
    ax.set_title("Sampled monthly volume reproduces exact volume (×5)")
    ax.set_ylabel("Requests per month")
    ax.legend(loc="upper left")
    H.save_figure(fig, "01_sample_calibration.png")
    plt.show()

# %% [markdown]
# ## 3. The actionability split — the single most important data-quality finding
#
# A large fraction of 311 records are **not field work**. Citizens call 311
# with questions, and those calls are logged and closed by the contact centre
# in seconds. Aircraft-noise reports are likewise logged and auto-closed
# without dispatching anyone.
#
# If these are left in, the citywide median "response time" becomes ~0 days and
# any equity comparison is destroyed. They are therefore flagged, not deleted.

# %%
H.query("""
    SELECT
        is_informational                                  AS administrative_filing,
        count(*)                                          AS rows,
        round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct_of_all,
        round(median(days_to_resolution), 6)              AS median_days,
        round(avg(days_to_resolution), 2)                 AS mean_days
    FROM analytics.sr_enriched
    GROUP BY 1 ORDER BY 2 DESC
""", con)

# %%
instants = H.query("""
    SELECT sr_type,
           count(*)                                        AS rows,
           round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct_of_instant,
           round(median(days_to_resolution) * 86400, 1)    AS median_seconds
    FROM analytics.sr_enriched
    WHERE days_to_resolution IS NOT NULL AND days_to_resolution < 1.0/1440
    GROUP BY 1 ORDER BY 2 DESC LIMIT 8
""", con)
print("Request types closed in under one minute:\n")
instants

# %%
scope = H.query("""
    SELECT
        CASE WHEN is_actionable THEN 'Service requests (actionable)'
             ELSE 'Administrative filings' END              AS scope,
        count(*)                                            AS rows,
        round(100.0 * count(*) / sum(count(*)) OVER (), 2)  AS pct_of_all,
        round(median(days_to_resolution), 3)                AS median_days,
        round(quantile_cont(days_to_resolution, 0.9), 2)    AS p90_days,
        round(avg(days_to_resolution), 2)                   AS mean_days
    FROM analytics.sr_enriched
    GROUP BY 1 ORDER BY 1
""", con)
scope

# %%
med_all = float(H.query(
    "SELECT median(days_to_resolution) d FROM analytics.sr_enriched", con)["d"][0])
med_svc = float(scope.loc[scope["scope"].str.startswith("Service"), "median_days"].iloc[0])
print(f"Citywide median response time, all rows            : {med_all:.5f} days "
      f"({med_all * 24 * 60:.1f} minutes)")
print(f"Citywide median response time, actionable requests : {med_svc:.3f} days "
      f"({med_svc * 24:.1f} hours)")
print(f"\nReporting the first number as 'the city's 311 response time' would be "
      f"a {med_svc / max(med_all, 1e-9):,.0f}x understatement caused entirely by "
      f"administrative closures.")

H.record_finding("median_days_all_rows", round(med_all, 6),
                 "citywide median including administrative filings",
                 "notebooks/01_eda.ipynb §3")
H.record_finding("median_days_actionable", round(med_svc, 3),
                 "citywide median over actionable service requests",
                 "notebooks/01_eda.ipynb §3")

# %% [markdown]
# ## 4. Composition — what Chicago actually reports

# %%
top_types = H.query("""
    SELECT sr_type,
           count(*)                                          AS rows,
           round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct_of_all,
           any_value(severity_band)                          AS severity,
           round(median(days_to_resolution)
                 FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL), 2)
                                                             AS median_days_actionable
    FROM analytics.sr_enriched
    GROUP BY 1 ORDER BY 2 DESC LIMIT 15
""", con)
top_types

# %%
bands = H.query("""
    SELECT severity_band,
           count(*)                                          AS rows,
           round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct_of_all,
           round(median(days_to_resolution)
                 FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL), 2)
                                                             AS median_days_actionable
    FROM analytics.sr_enriched
    GROUP BY 1
    ORDER BY CASE severity_band WHEN 'critical' THEN 1 WHEN 'high' THEN 2
                                WHEN 'medium' THEN 3 ELSE 4 END
""", con)
bands

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
b = bands.set_index("severity_band").reindex(["critical", "high", "medium", "low"])
axes[0].bar(b.index, b["rows"], color=H.PALETTE["bands"])
axes[0].set_title("Request volume by severity band")
axes[0].set_ylabel("Requests")
for i, v in enumerate(b["rows"]):
    axes[0].text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)

tt = top_types.head(10).iloc[::-1]
axes[1].barh(tt["sr_type"].str.slice(0, 40), tt["rows"], color=H.PALETTE["accent"])
axes[1].set_title("Ten most common request types")
axes[1].set_xlabel("Requests")
H.save_figure(fig, "01_composition.png")
plt.show()

# %% [markdown]
# ## 5. Temporal structure
#
# Monthly volume, seasonality by month-of-year, and the working-week rhythm —
# all on **demand** scope, because a citizen call is demand even when it is
# closed administratively.

# %%
monthly = H.query("""
    SELECT month,
           count(*)                                     AS n_requests,
           count(*) FILTER (WHERE is_actionable)        AS n_actionable,
           round(median(days_to_resolution)
                 FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL), 3)
                                                        AS median_days
    FROM analytics.sr_enriched
    GROUP BY 1 ORDER BY 1
""", con)

fig, ax = plt.subplots(figsize=(12, 3.8))
ax.plot(monthly["month"], monthly["n_requests"], lw=1.7,
        color=H.PALETTE["accent"], label="All requests (demand)")
ax.plot(monthly["month"], monthly["n_actionable"], lw=1.4,
        color=H.PALETTE["accent3"], label="Actionable service requests")
ax.set_title("Monthly 311 volume, 2019–2025")
ax.set_ylabel("Requests per month")
ax.legend(loc="upper left")
H.save_figure(fig, "01_monthly_volume.png")
plt.show()

# %%
seasonal = H.query("""
    SELECT created_month AS month_of_year,
           round(avg(n)) AS avg_requests,
           round(avg(n_actionable)) AS avg_actionable
    FROM (
        SELECT month, EXTRACT(month FROM month)::INT AS created_month,
               count(*) AS n, count(*) FILTER (WHERE is_actionable) AS n_actionable
        FROM analytics.sr_enriched GROUP BY month
    ) GROUP BY 1 ORDER BY 1
""", con)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
axes[0].bar(seasonal["month_of_year"], seasonal["avg_requests"],
            color=H.PALETTE["accent"])
axes[0].set_title("Seasonality — mean volume by calendar month")
axes[0].set_xlabel("Month")
axes[0].set_xticks(range(1, 13))
axes[0].set_ylabel("Mean requests")

rhythm = H.query("""
    SELECT created_dow, created_hour, count(*) AS n
    FROM analytics.sr_enriched GROUP BY 1, 2
""", con).pivot(index="created_dow", columns="created_hour", values="n")
im = axes[1].imshow(rhythm.values, aspect="auto", cmap="magma")
axes[1].set_title("Reporting rhythm — day of week × hour")
axes[1].set_xlabel("Hour of day")
axes[1].set_ylabel("ISO day of week (1=Mon)")
axes[1].grid(False)
fig.colorbar(im, ax=axes[1], label="Requests")
H.save_figure(fig, "01_temporal_patterns.png")
plt.show()

# %%
# Compare early vs late window: has demand, or workload, shifted?
H.query("""
    SELECT yr,
           count(*)                                      AS requests,
           round(median(days_to_resolution)
                 FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL), 2)
                                                         AS median_days_actionable,
           round(100.0 * count(*) FILTER (WHERE severity_band = 'critical')
                 / count(*), 2)                          AS pct_critical
    FROM analytics.sr_enriched
    GROUP BY 1 ORDER BY 1
""", con)

# %% [markdown]
# ## 6. Weather relationship (motivating the forecasting covariates)

# %%
daily = H.query("""
    SELECT CAST(created_ts AS DATE) AS d,
           count(*)                 AS n,
           avg(tmax_c)              AS tmax_c,
           sum(prcp_mm)             AS prcp_mm,
           sum(snow_mm)             AS snow_mm
    FROM analytics.sr_enriched
    GROUP BY 1 ORDER BY 1
""", con)

tmp = daily.dropna(subset=["tmax_c"])
corr = tmp[["n", "tmax_c", "prcp_mm", "snow_mm"]].corr()["n"].drop("n")
print("Correlation of daily request volume with weather:")
print(corr.round(3).to_string())

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.scatter(tmp["tmax_c"], tmp["n"], s=8, alpha=0.35, color=H.PALETTE["accent"])
z = np.polyfit(tmp["tmax_c"], tmp["n"], 1)
xs = np.linspace(tmp["tmax_c"].min(), tmp["tmax_c"].max(), 50)
ax.plot(xs, np.polyval(z, xs), color=H.PALETTE["accent2"], lw=2,
        label=f"OLS slope {z[0]:,.1f} requests/°C")
ax.set_title("Daily 311 volume vs. daily maximum temperature")
ax.set_xlabel("Tmax (°C)")
ax.set_ylabel("Requests that day")
ax.legend()
H.save_figure(fig, "01_weather_relationship.png")
plt.show()

# %% [markdown]
# ## 7. Conclusions carried into the rest of the project
#
# 1. **The sample is sound.** The 20% systematic draw lands within a fraction of
#    a percentage point of 20.000% in every validated month, so ×5 weighting
#    reproduces true monthly totals and the sample introduces no time bias.
# 2. **Actionability is the decisive analytical choice.** Administrative
#    filings collapse the citywide median response time to effectively zero.
#    Every response-time and equity statistic in this project is therefore
#    computed over actionable service requests, and the exclusion is documented
#    rather than silent.
# 3. **Volume is strongly seasonal and weather-linked**, which justifies the
#    temperature and precipitation covariates used in notebook 03.
# 4. **Demand is heavily concentrated in a few request types**, so response-time
#    comparisons across areas must control for request mix — the reason
#    notebook 02 compares like-for-like request types as well as raw medians.

# %%
H.record_finding("rows_total", int(overview["rows_total"][0]),
                 f"analytical rows across {int(overview['months'][0])} months",
                 "notebooks/01_eda.ipynb §1")
H.record_finding("n_request_types", int(overview["request_types"][0]),
                 "distinct request types", "notebooks/01_eda.ipynb §1")
print("findings written to", C.DATA_PROCESSED / "findings.json")
con.close()
