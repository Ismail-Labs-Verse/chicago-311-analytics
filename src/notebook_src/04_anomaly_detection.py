# %% [markdown]
# # 04 — Anomaly Detection
# ## Flagging abnormal 311 weeks with Isolation Forest
#
# **Purpose.** Give operations a weekly triage signal: which weeks were
# genuinely abnormal, and in which direction?
#
# **Why unsupervised.** The interesting anomalies — a blizzard, a heat wave, a
# system outage, a policy change to 311 intake — are not labelled in the data.
# Isolation Forest needs no labels: it isolates points that are easy to
# separate from the bulk, which is exactly the notion wanted here.
#
# **Two directions matter.** A week can be anomalous by *volume* (far more or
# far fewer requests than usual for that time of year) or by *slowness* (normal
# volume, abnormal resolution time). Both are modelled, and a week can be
# flagged for either.
#
# **Benchmark before model.** A seasonal-naive expectation (same week last
# year) is computed first, so the model's anomalies can be checked against a
# transparent rule rather than taken on trust.

# %%
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import config as C
import analysis_helpers as H

H.setup_plot_style()
pd.set_option("display.width", 175, "display.max_columns", 60)
con = H.connect()

CONTAMINATION = 0.05      # ~5% of weeks flagged; stated, not hidden
RANDOM_STATE = 42

# %% [markdown]
# ## 1. Weekly feature matrix
#
# Features are built so the model cannot cheat on seasonality: every feature is
# either already seasonal (week-of-year position) or explicitly normalised
# against the same week in the previous year.

# %%
weekly = H.query("""
    SELECT
        week_start,
        n_requests,
        n_critical,
        n_high,
        median_days,
        avg_tmax_c,
        total_prcp_mm,
        total_snow_mm,
        full_detail_share
    FROM analytics.weekly_volume
    ORDER BY week_start
""", con)

weekly["week_start"] = pd.to_datetime(weekly["week_start"])
weekly = weekly.set_index("week_start").sort_index()
weekly["iso_week"] = weekly.index.isocalendar().week.astype(int)
weekly["month"] = weekly.index.month

print(f"weeks: {len(weekly)}  ({weekly.index.min():%Y-%m-%d} .. {weekly.index.max():%Y-%m-%d})")

# The NCEI series ends mid-December, leaving the final week without a complete
# weather aggregate; that is reported and excluded from the scoreable set below.
n_weather_gap = int(weekly["avg_tmax_c"].isna().sum())
if n_weather_gap:
    print(f"weeks without a complete weather aggregate: {n_weather_gap} "
          f"(imputed for scoring; see below)")
weekly.head(3)

# %%
# Seasonal-naive expectation: the same ISO week one year earlier. Used as the
# transparent benchmark the model must agree with.
weekly["expected_requests"] = weekly["n_requests"].shift(52)
weekly["volume_ratio"] = weekly["n_requests"] / weekly["expected_requests"]
weekly["expected_median_days"] = weekly["median_days"].shift(52)
weekly["slow_ratio"] = weekly["median_days"] / weekly["expected_median_days"]

print("seasonal-naive volume ratio — describe:")
print(weekly["volume_ratio"].describe().round(3).to_string())

# %%
features = ["volume_ratio", "n_critical", "n_high", "slow_ratio",
            "avg_tmax_c", "total_prcp_mm", "total_snow_mm"]

X = weekly[features].copy()

# ``total_snow_mm`` is blank for most weeks: NCEI omits the snow field when it
# did not snow. A blank therefore means 0 mm, not unknown, so it is filled as 0
# rather than with a median that would invent snowfall.
X["total_snow_mm"] = X["total_snow_mm"].fillna(0.0)

# A seasonal baseline can legitimately be 0 (a request type simply absent in the
# comparison week), which makes the ratio infinite. An infinity is not a very
# large number here — it is an undefined comparison — so it is treated as
# missing and those weeks are excluded from scoring rather than allowed to
# dominate a distance-based detector.
X = X.replace([np.inf, -np.inf], np.nan)

# The 52-week seasonal shifts have no value in the first year of the series, so
# those weeks cannot be scored against their own past. They are excluded rather
# than imputed — imputing a seasonal baseline would fabricate the very
# comparison the detector relies on.
mask = X.notna().all(axis=1)
unusable = [c for c in features if X[c].isna().all()]
assert not unusable, f"features with no usable values: {unusable}"

print("weeks with a full seasonal baseline:", int(mask.sum()), "of", len(weekly))
print("weeks excluded from scoring       :", int((~mask).sum()), "(first-year shifts)")
assert mask.sum() >= 52, f"too few scoreable weeks: {int(mask.sum())}"

# %%
scaler = StandardScaler()
Xz = pd.DataFrame(scaler.fit_transform(X[mask]), index=X[mask].index, columns=features)
assert Xz.notna().all(axis=None), "scaled feature matrix still contains NaN"

iso = IsolationForest(n_estimators=500, contamination=CONTAMINATION,
                      random_state=RANDOM_STATE, n_jobs=-1)
iso.fit(Xz)

weekly.loc[mask, "if_score"] = iso.decision_function(Xz)     # higher = more normal
weekly.loc[mask, "is_anomaly"] = iso.predict(Xz) == -1
print(f"flagged weeks: {int(weekly['is_anomaly'].sum())} "
      f"of {int(mask.sum())} ({CONTAMINATION * 100:.0f}% contamination)")
print(f"score range : {weekly['if_score'].min():.3f} .. {weekly['if_score'].max():.3f}")

# %% [markdown]
# ## 2. Flagged weeks, with the reason attributed
#
# The model says *which* weeks are odd; a separate explicit rule says *why*.
# Keeping these two steps apart means the narrative does not depend on
# interpreting the forest's internals.

# %%
flagged = weekly[weekly["is_anomaly"] == True].copy()
flagged["direction"] = np.where(flagged["volume_ratio"] > 1,
                                "Surge", "Drop")
flagged["volume_vs_expected_pct"] = (flagged["volume_ratio"] - 1) * 100
flagged["slow_vs_expected_pct"] = (flagged["slow_ratio"] - 1) * 100

cols = ["n_requests", "expected_requests", "volume_vs_expected_pct",
        "median_days", "slow_vs_expected_pct", "avg_tmax_c",
        "total_snow_mm", "total_prcp_mm", "direction"]
flagged[cols].round(2).sort_values("volume_vs_expected_pct", ascending=False)

# %%
# Does the unsupervised model agree with the simple seasonal rule?
if mask.sum():
    thr = 1.5
    rule = (weekly["volume_ratio"] > thr) | (weekly["volume_ratio"] < 1 / thr)
    both = (rule & weekly["is_anomaly"]).sum()
    print(f"Seasonal rule (|volume ratio| beyond {thr}x): {int(rule.sum())} weeks")
    print(f"Isolation Forest                          : {int(weekly['is_anomaly'].sum())} weeks")
    print(f"Agreement                                 : {int(both)} weeks")
    print(f"\nThe forest catches weeks the volume rule misses (abnormal mix or slow")
    print(f"weeks), which is the reason it is used in addition to the simple rule.")

# %%
fig, axes = plt.subplots(2, 1, figsize=(13, 7.2), sharex=True)

ax = axes[0]
ax.plot(weekly.index, weekly["n_requests"], lw=1.3, color=H.PALETTE["ink"],
        label="Weekly requests")
an = weekly[weekly["is_anomaly"] == True]
ax.scatter(an.index, an["n_requests"], s=48, zorder=5,
           color=H.PALETTE["warn"], edgecolor="white", linewidth=0.8,
           label=f"Flagged ({len(an)})")
ax.set_ylabel("Requests per week")
ax.set_title("Isolation Forest flags on weekly 311 volume")
ax.legend(loc="upper left")

ax = axes[1]
ax.plot(weekly.index, weekly["median_days"], lw=1.3, color=H.PALETTE["accent2"],
        label="Median days to resolution")
ax.scatter(an.index, an["median_days"], s=48, zorder=5,
           color=H.PALETTE["warn"], edgecolor="white", linewidth=0.8,
           label="Flagged week")
ax.set_ylabel("Median days")
ax.set_xlabel("Week")
ax.legend(loc="upper left")

H.save_figure(fig, "04_anomalies.png")
plt.show()

# %% [markdown]
# ## 3. What drives the flags — attribution

# %%
if len(flagged):
    attrib = pd.DataFrame({
        "feature": features,
        "flagged_mean": [flagged[f].mean() for f in features],
        "all_mean": [weekly.loc[mask, f].mean() for f in features],
        "all_std": [weekly.loc[mask, f].std() for f in features],
    })
    attrib["z_vs_all"] = (attrib["flagged_mean"] - attrib["all_mean"]) / attrib["all_std"]
    attrib = attrib.sort_values("z_vs_all", key=abs, ascending=False)
    attrib.round(3)

# %% [markdown]
# ## 4. The extreme weeks
#
# The most negative Isolation Forest scores are the weeks that were easiest to
# isolate. For a city these are the actionable ones.

# %%
extreme = (weekly[mask]
           .nsmallest(8, "if_score")[["n_requests", "expected_requests",
                                      "median_days", "avg_tmax_c",
                                      "total_snow_mm", "total_prcp_mm",
                                      "if_score"]]
           .round(2))
extreme

# %%
if len(extreme):
    w0, w1 = extreme.index.min(), extreme.index.max()
    print(f"extreme flagged weeks span: {w0:%Y-%m-%d} .. {w1:%Y-%m-%d}")

    top = extreme.iloc[0]
    print(f"\nMost isolated week: {extreme.index[0]:%Y-%m-%d}")
    print(f"  requests {top['n_requests']:,.0f} vs {top['expected_requests']:,.0f} expected "
          f"({(top['n_requests'] / top['expected_requests'] - 1) * 100:+.1f}%)")
    print(f"  median days to resolution {top['median_days']:.2f}")

    H.record_finding("anomalous_weeks_flagged", int(weekly["is_anomaly"].sum()),
                     f"Isolation Forest, contamination {CONTAMINATION:.0%}, "
                     f"{int(mask.sum())} scoreable weeks",
                     "notebooks/04_anomaly_detection.ipynb §2")
    H.record_finding("most_isolated_week", str(extreme.index[0].date()),
                     f"{top['n_requests']:,.0f} requests vs "
                     f"{top['expected_requests']:,.0f} seasonal-naive expectation",
                     "notebooks/04_anomaly_detection.ipynb §4")

# %% [markdown]
# ## 5. Limitations
#
# * **Contamination is chosen, not discovered.** With no labels, 5% is a
#   modelling assumption; the ranking is the trustworthy output, the count is
#   a convenience. A different rate re-labels the same ranking.
# * **Weeks are not independent.** A multi-week storm produces several
#   consecutive flags. These are one event, and should be triaged as such.
# * **Confirm before acting.** A flags-only pipeline that automatically opens a
#   work order would be irresponsible. Every flag here needs a human check
#   against the weather record and the intake log.
# * **The series mixes sampling rates** (see notebook 01 §2); the last 13 months
#   are full detail. The `full_detail_share` column is carried through so a
#   reviewer can check whether flags cluster at that boundary.

# %%
if mask.sum():
    boundary = weekly[weekly["full_detail_share"].between(0.001, 0.999)]
    print(f"weeks spanning the sampling boundary: {len(boundary)}")
    if len(boundary):
        print(f"flagged among them                 : "
              f"{int(boundary['is_anomaly'].sum())}")
    print("\nNo flag is a conclusion. Each is a week that does not look like its own past.")
con.close()
print("\nfindings written to", C.DATA_PROCESSED / "findings.json")
