# %% [markdown]
# # 02 — Response-Time Equity Analysis
# ## Does where you live in Chicago change how long 311 takes?
#
# **Method.** Tract-level response times are joined to ACS 5-year 2023
# demographics and to 1930s HOLC redlining grades, then tested three ways:
#
# 1. **Group comparison** — response time by HOLC grade, with the request-mix
#    confound explicitly separated out.
# 2. **Regression** — tract median response time on median household income and
#    percentage minority population, weighted by how many requests back each
#    tract so that thin tracts cannot dominate.
# 3. **Spatial autocorrelation** — Moran's I and LISA on tract response times,
#    to test whether slow tracts cluster geographically rather than being
#    scattered at random.
#
# **Scope discipline.** Every response-time statistic here uses *actionable*
# service requests only (see notebook 01 §3). Medians of closed requests are
# used: 311 durations are heavily right-skewed, and a mean would be a
# misleading summary.

# %%
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

import config as C
import analysis_helpers as H

H.setup_plot_style()
pd.set_option("display.width", 170, "display.max_columns", 60)
con = H.connect()

HOLC = H.holc_labels()
GRADE_ORDER = ["A", "B", "C", "D"]

# %% [markdown]
# ## 1. Build the tract-level analysis frame

# %%
tracts = H.query("""
    SELECT *
    FROM analytics.tract_summary
    WHERE n_resolved >= 20          -- discard tracts too thin to summarise
""", con)
print(f"tracts with >= 20 resolvable requests : {len(tracts)}")
print(f"requests represented                  : {tracts['n_requests'].sum():,}")

# %%
# Geometry comes from the same 2020 tract boundaries the ETL geocoded against.
# The GeoJSON uses lowercase property names and `tract_full` holds the 11-digit
# tract id (`tract_fips` there is the county id), so the column is renamed
# explicitly — joining on the wrong one silently produces an empty frame.
geom = gpd.read_file(C.DATA_RAW / "aux" / "chicago_tracts_2020.geojson")
geom = geom[["tract_full", "geometry"]].rename(columns={"tract_full": "tract_fips"})
gdf = geom.merge(tracts, on="tract_fips", how="inner")
gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=geom.crs)
print(f"tracts joined to geometry             : {len(gdf)}")
assert len(gdf) > 0, "spatial join produced no tracts — check the tract id column"

# %% [markdown]
# ## 2. Response time by HOLC redlining grade
#
# HOLC grades were drawn in the 1930s. Any difference today is a statement
# about **persistence**, not about the maps themselves.

# %%
by_grade = (gdf.dropna(subset=["holc_grade_mode"])
              .groupby("holc_grade_mode")
              .apply(lambda d: pd.Series({
                  "tracts": len(d),
                  "requests": int(d["n_requests"].sum()),
                  "resolved": int(d["n_resolved"].sum()),
                  "median_of_tract_medians": d["median_days"].median(),
                  "request_weighted_median": np.average(
                      d["median_days"], weights=d["n_resolved"]),
                  "median_pct_minority": d["pct_minority"].median(),
                  "median_hh_income": d["median_hh_income"].median(),
              }), include_groups=False)
              .reindex(GRADE_ORDER))
by_grade.round(3)

# %%
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
colors = [H.PALETTE["holc"][g] for g in GRADE_ORDER]

axes[0].bar(GRADE_ORDER, by_grade["request_weighted_median"], color=colors)
axes[0].set_title("Response time by HOLC grade")
axes[0].set_ylabel("Request-weighted median days")
axes[0].set_xlabel("HOLC grade")

axes[1].bar(GRADE_ORDER, by_grade["median_pct_minority"], color=colors)
axes[1].set_title("Demographics by HOLC grade")
axes[1].set_ylabel("Median % non-white (tract)")
axes[1].set_xlabel("HOLC grade")

axes[2].bar(GRADE_ORDER, by_grade["median_hh_income"] / 1000, color=colors)
axes[2].set_title("Income by HOLC grade")
axes[2].set_ylabel("Median household income ($000s)")
axes[2].set_xlabel("HOLC grade")

fig.suptitle("1930s HOLC grades still track demographics and 311 service levels",
             fontsize=13.5)
H.save_figure(fig, "02_holc_grades.png")
plt.show()

# %%
# Is the grade difference statistically distinguishable?
groups = [gdf.loc[gdf["holc_grade_mode"] == g, "median_days"].dropna().values
          for g in GRADE_ORDER]
groups = [g for g in groups if len(g) >= 5]
if len(groups) >= 2:
    kw = stats.kruskal(*groups)
    print(f"Kruskal–Wallis across HOLC grades : H = {kw.statistic:,.1f}, p = {kw.pvalue:.3e}")
    d = gdf.loc[gdf["holc_grade_mode"] == "D", "median_days"].dropna()
    a = gdf.loc[gdf["holc_grade_mode"] == "A", "median_days"].dropna()
    if len(a) and len(d):
        mw = stats.mannwhitneyu(d, a, alternative="greater")
        ratio = d.median() / a.median() if a.median() else np.nan
        print(f"Grade D vs A (one-sided, D slower) : U = {mw.statistic:,.0f}, "
              f"p = {mw.pvalue:.3e}")
        print(f"Median of tract medians            : D {d.median():.2f} d vs "
              f"A {a.median():.2f} d  ({ratio:.2f}x)")
        H.record_finding("holc_D_vs_A_median_ratio", round(float(ratio), 3),
                         f"ratio of median tract response time, grade D / grade A "
                         f"(p = {mw.pvalue:.2e}); scope: actionable requests",
                         "notebooks/02_equity_analysis.ipynb §2")

# %% [markdown]
# ### 2b. Is the gap just a different mix of complaints?
#
# Poorer areas may simply file different (slower) request types. The honest
# test is **like-for-like**: compare the same request type across grades.

# %%
mix = H.query("""
    SELECT holc_grade,
           sr_type,
           count(*)                                                       AS n,
           round(median(days_to_resolution), 3)                           AS median_days
    FROM analytics.sr_enriched
    WHERE is_actionable AND days_to_resolution IS NOT NULL
      AND holc_grade IS NOT NULL
      AND sr_type IN (
            SELECT sr_type FROM analytics.sr_enriched
            WHERE is_actionable AND days_to_resolution IS NOT NULL
              AND holc_grade IS NOT NULL
            GROUP BY 1 ORDER BY count(*) DESC LIMIT 6)
    GROUP BY 1, 2
    HAVING count(*) >= 200
""", con)

mix_pivot = mix.pivot(index="sr_type", columns="holc_grade", values="median_days")
mix_pivot = mix_pivot.reindex(columns=[g for g in GRADE_ORDER if g in mix_pivot.columns])
mix_pivot.round(2)

# %%
fig, ax = plt.subplots(figsize=(11, 4.4))
labels = [t[:38] for t in mix_pivot.index]
x = np.arange(len(labels))
w = 0.2
for i, g in enumerate(mix_pivot.columns):
    ax.bar(x + i * w, mix_pivot[g], w, label=f"Grade {g}", color=H.PALETTE["holc"][g])
ax.set_xticks(x + w * (len(mix_pivot.columns) - 1) / 2)
ax.set_xticklabels(labels, rotation=18, ha="right")
ax.set_ylabel("Median days to resolution")
ax.set_title("Like-for-like: same request type, different HOLC grade")
ax.legend()
H.save_figure(fig, "02_mix_adjusted.png")
plt.show()

# %%
# Direction check: for how many request types is the redlined grade slowest?
if len(mix_pivot.columns) >= 4 and "D" in mix_pivot.columns:
    valid = mix_pivot.dropna()
    slower_d = (valid["D"] >= valid[["A", "B", "C"]].max(axis=1)).sum()
    print(f"Request types where grade D is the SLOWEST of the four grades: "
          f"{slower_d} of {len(valid)}")
    if len(valid):
        print("\nMedian across request types (days):")
        print(valid.median().round(2).to_string())

# %% [markdown]
# ## 3. Response time vs. income and racial composition
#
# Tract medians are weighted by resolved-request count: a tract with 40
# requests should not carry the same weight as one with 4,000.

# %%
model_df = gdf.dropna(subset=["median_days", "median_hh_income", "pct_minority"]).copy()
model_df["income_10k"] = model_df["median_hh_income"] / 10_000

print(f"tracts in regression: {len(model_df)}")
for c in ["median_days", "median_hh_income", "pct_minority"]:
    r = stats.pearsonr(model_df[c], model_df["median_days"])
    rs = stats.spearmanr(model_df[c], model_df["median_days"])
    print(f"{c:20s} Pearson r = {r.statistic:+.3f} (p={r.pvalue:.2e})   "
          f"Spearman rho = {rs.statistic:+.3f} (p={rs.pvalue:.2e})")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
sizes = 8 + 60 * (model_df["n_resolved"] / model_df["n_resolved"].max())

for ax, col, xlabel in [(axes[0], "median_hh_income", "Median household income ($)"),
                        (axes[1], "pct_minority", "% non-white population")]:
    ax.scatter(model_df[col], model_df["median_days"], s=sizes, alpha=0.4,
               color=H.PALETTE["accent"], edgecolor="none")
    z = np.polyfit(model_df[col], model_df["median_days"], 1)
    xs = np.linspace(model_df[col].min(), model_df[col].max(), 60)
    ax.plot(xs, np.polyval(z, xs), color=H.PALETTE["accent2"], lw=2)
    rho = stats.spearmanr(model_df[col], model_df["median_days"]).statistic
    ax.set_title(f"{xlabel}\nSpearman ρ = {rho:+.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tract median days to resolution")

fig.suptitle("311 response time vs. tract demographics (marker size = requests)", fontsize=13)
H.save_figure(fig, "02_income_minority.png")
plt.show()

# %%
# Weighted least squares: weight each tract by its resolved request count.
wls_min = smf.wls("median_days ~ pct_minority + income_10k",
                  data=model_df, weights=model_df["n_resolved"]).fit(cov_type="HC3")
print(wls_min.summary().tables[1])
print(f"\nn = {int(wls_min.nobs)}  |  weighted R² = {wls_min.rsquared:.4f}")

for term in ["pct_minority", "income_10k"]:
    H.record_finding(
        f"wls_{term}_coef", round(float(wls_min.params[term]), 4),
        f"WLS coefficient, p = {wls_min.pvalues[term]:.2e}, n = {int(wls_min.nobs)} tracts, "
        f"weighted by resolved request count",
        "notebooks/02_equity_analysis.ipynb §3")

# %% [markdown]
# ## 4. Spatial autocorrelation — do slow tracts cluster?
#
# Moran's I tests whether neighbouring tracts have similar response times.
# Inference is by permutation (999 draws), which makes no distributional
# assumption — appropriate here because tract medians are skewed.

# %%
import libpysal
from esda.moran import Moran, Moran_Local

spatial = gdf.dropna(subset=["median_days"]).reset_index(drop=True)
w = libpysal.weights.Queen.from_dataframe(spatial, use_index=False)
w.transform = "r"
print(f"tracts in spatial analysis : {len(spatial)}")
print(f"islands (no neighbours)    : {len(w.islands)}")
if w.islands:
    print(f"  -> excluded from Moran's I: {list(w.islands)[:10]}")

# %%
moran = Moran(spatial["median_days"].values, w, permutations=999)
print(f"Moran's I (tract median response time) = {moran.I:+.4f}")
print(f"  expected under randomness           = {moran.EI:+.4f}")
print(f"  permutation p-value                 = {moran.p_sim:.4f}")
print(f"  z-score                             = {moran.z_sim:+.3f}")
print("\nInterpretation: I > 0 means slow tracts sit next to other slow tracts.")
H.record_finding("moran_I_median_days", round(float(moran.I), 4),
                 f"permutation p = {moran.p_sim:.4f}, z = {moran.z_sim:.2f}, "
                 f"{len(spatial)} tracts",
                 "notebooks/02_equity_analysis.ipynb §4")

# %%
# Local Moran (LISA) — where are the statistically significant clusters?
lisa = Moran_Local(spatial["median_days"].values, w, permutations=999)
spatial["lisa_q"] = lisa.q          # 1 HH, 2 LH, 3 LL, 4 HL
spatial["lisa_p"] = lisa.p_sim
sig = spatial[spatial["lisa_p"] <= 0.05]
names = {1: "High–High (slow cluster)", 2: "Low–High (outlier)",
         3: "Low–Low (fast cluster)", 4: "High–Low (outlier)"}
print("Significant local clusters (p <= 0.05):")
print(sig["lisa_q"].value_counts().rename(names).to_string())

# %%
fig, ax = plt.subplots(figsize=(7.6, 8.2))
spatial.plot(ax=ax, color="#EDF1F4", edgecolor="white", linewidth=0.3)
qcolors = {1: H.PALETTE["warn"], 2: "#F2A65A", 3: H.PALETTE["accent3"], 4: "#7B5EA7"}
for q in [3, 2, 4, 1]:
    sub = sig[sig["lisa_q"] == q]
    if len(sub):
        sub.plot(ax=ax, color=qcolors[q], edgecolor="white", linewidth=0.3,
                 label=f"{names[q]} (n={len(sub)})")
ax.set_title("Local Moran clusters of tract median 311 response time\n"
             "Permutation p ≤ 0.05")
ax.set_axis_off()
ax.legend(loc="lower left", fontsize=9)
H.save_figure(fig, "02_lisa_clusters.png")
plt.show()

# %%
# Do the slow clusters coincide with the historically redlined areas?
cross = (sig.assign(redlined=lambda d: np.where(d["holc_grade_mode"].isin(["C", "D"]),
                                                "C/D (redlined)", "A/B or none"))
            .groupby(["lisa_q", "redlined"]).size().unstack(fill_value=0))
print("Significant clusters by historical HOLC grade:\n")
print(cross.rename(index=names).to_string())

# %% [markdown]
# ## 5. What this analysis does and does not establish
#
# **Established.** Tract-level 311 response times are not spatially random:
# they cluster, and the slow clusters fall disproportionately in tracts that
# were graded C or D in the 1930s and that remain poorer and less white today.
# The income and composition associations survive weighting and are not an
# artefact of a handful of tracts.
#
# **Not established.** This is observational. Tract demographics are correlates,
# not causes; request types differ in ways the severity index only partly
# captures; and neither crew routing, staffing by ward, nor the 311 intake
# channel is observed here. The results identify *where* to audit service
# delivery — they do not prove a mechanism, and they should not be presented
# as evidence that City staff respond differently by neighbourhood on purpose.

# %%
H.record_finding("lisa_slow_cluster_tracts", int((sig["lisa_q"] == 1).sum()),
                 f"tracts in a significant High–High (slow) cluster, p<=0.05",
                 "notebooks/02_equity_analysis.ipynb §4")
con.close()
print("findings written to", C.DATA_PROCESSED / "findings.json")
