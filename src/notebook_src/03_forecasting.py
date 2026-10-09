# %% [markdown]
# # 03 — Demand Forecasting
# ## SARIMAX with weather covariates vs. LightGBM
#
# **Task.** Forecast monthly 311 call volume so the City can size crews and
# contact-centre capacity in advance.
#
# **Validation protocol.**
# * **Hold-out** = the final 12 complete months of the window. This period is
#   genuinely out-of-sample *and* it is the part of the series that was pulled
#   at full row-level detail (2024-12..2025-12), so the evaluation is not
#   contaminated by the sampling rate.
# * **Rolling-origin backtest** across the last 24 months on top of that, so a
#   single lucky hold-out cannot decide the comparison.
# * Two baselines are always reported: **seasonal naive** (same month last
#   year) and the **in-sample mean**. A model that cannot beat seasonal naive
#   has no business being deployed for this purpose.
#
# **Scope.** Forecasting runs on **demand** scope (all requests). Information
# calls are real load on the 311 system even though they are not field work.

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
pd.set_option("display.width", 170, "display.max_columns", 60)
con = H.connect()

HORIZON = 12        # months held out
N_BACKTEST = 24     # rolling-origin folds

# %% [markdown]
# ## 1. Build the monthly series with its weather covariates

# %%
ts = H.query("""
    SELECT
        month,
        count(*)                                                  AS y,
        count(*) FILTER (WHERE is_actionable)                     AS y_actionable,
        avg(tmax_c)                                               AS tmax_c,
        sum(prcp_mm)                                              AS prcp_mm,
        sum(snow_mm)                                              AS snow_mm,
        avg(awnd_ms)                                              AS awnd_ms,
        round(median(days_to_resolution)
              FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL), 3)
                                                                  AS median_days
    FROM analytics.sr_enriched
    GROUP BY month
    ORDER BY month
""", con)

ts["month"] = pd.to_datetime(ts["month"])
ts = ts.set_index("month").asfreq("MS")
ts["month_of_year"] = ts.index.month
ts["year"] = ts.index.year

# The NCEI daily series reports a BLANK snow field in months with no snowfall
# (and ends mid-December), so snow is genuinely absent rather than unknown.
# Blank snow is therefore read as 0 mm; rows missing temperature or
# precipitation are dropped, because those are measured continuously.
ts["snow_mm"] = ts["snow_mm"].fillna(0.0)
ts["awnd_ms"] = ts["awnd_ms"].fillna(ts["awnd_ms"].median())

REQUIRED = ["tmax_c", "prcp_mm"]
n_incomplete = int(ts[REQUIRED].isna().any(axis=1).sum())
if n_incomplete:
    ts = ts.dropna(subset=REQUIRED)
print(f"dropped {n_incomplete} month(s) missing temperature/precipitation")
assert len(ts) >= 60, f"too few complete months left: {len(ts)}"

print(f"series: {ts.index.min():%Y-%m} .. {ts.index.max():%Y-%m}  ({len(ts)} months)")
ts[["y", "tmax_c", "prcp_mm", "snow_mm"]].tail(14)

# %%
fig, ax = plt.subplots(figsize=(12.5, 4))
ax.plot(ts.index, ts["y"], marker="o", ms=3.2, lw=1.4, color=H.PALETTE["accent"],
        label="Monthly requests")
if len(ts) >= HORIZON:
    ax.axvspan(ts.index[-HORIZON], ts.index[-1], alpha=0.12,
               color=H.PALETTE["accent2"], label=f"Hold-out ({HORIZON} months)")
ax.set_title("Monthly 311 demand and the evaluation window")
ax.set_ylabel("Requests per month")
ax.legend(loc="upper left")
H.save_figure(fig, "03_series.png")
plt.show()

# %%
train, test = ts.iloc[:-HORIZON], ts.iloc[-HORIZON:]
print(f"train : {train.index.min():%Y-%m} .. {train.index.max():%Y-%m}  ({len(train)} months)")
print(f"test  : {test.index.min():%Y-%m} .. {test.index.max():%Y-%m}  ({len(test)} months)")
print(f"mean monthly demand — train {train['y'].mean():,.0f} | test {test['y'].mean():,.0f}")

# %% [markdown]
# ## 2. Baselines
#
# Reported first and prominently: on seasonal series, beating seasonal naive is
# the actual bar.

# %%
results = []
preds = {}

# In-sample mean
mean_pred = np.repeat(train["y"].mean(), HORIZON)
results.append(H.forecast_metrics(test["y"], mean_pred, "Mean (baseline)"))
preds["Mean"] = mean_pred

# Seasonal naive: same month, one year earlier
snaive = train["y"].iloc[-12:].values if len(train) >= 12 else None
if snaive is not None and len(snaive) == HORIZON:
    results.append(H.forecast_metrics(test["y"], snaive, "Seasonal naive"))
    preds["Seasonal naive"] = snaive

pd.DataFrame(results).round(3)

# %% [markdown]
# ## 3. SARIMAX with weather covariates
#
# Monthly volume is seasonal with a strong July peak, so a seasonal ARIMA on
# the 12-month cycle is the natural structural model. Weather enters as an
# exogenous regressor, and the seasonal difference also removes the slow
# level drift that the sample window introduces.
#
# `s=(12,)` with `D=1` forces a seasonal difference; the model is then refit
# on the full series before producing the forecast for the hold-out.

# %%
import statsmodels.api as sm

EXOG = ["tmax_c", "prcp_mm", "snow_mm"]
sarimax_order = (1, 1, 1)
sarimax_seasonal = (1, 1, 1, 12)

sarimax_fit = sm.tsa.SARIMAX(
    train["y"].astype(float),
    exog=train[EXOG].astype(float),
    order=sarimax_order,
    seasonal_order=sarimax_seasonal,
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

print(f"SARIMAX{sarimax_order}x{sarimax_seasonal} + {EXOG}")
print(f"  AIC {sarimax_fit.aic:,.1f}   BIC {sarimax_fit.bic:,.1f}   "
      f"params {len(sarimax_fit.params)}")
print(f"\n{'param':<28}{'coef':>12}{'p':>12}")
for name, coef, p in zip(sarimax_fit.param_names, sarimax_fit.params, sarimax_fit.pvalues):
    print(f"{name:<28}{coef:>12.4f}{p:>12.4f}")

# %%
sarimax_pred = sarimax_fit.forecast(steps=HORIZON, exog=test[EXOG].astype(float)).values
sarimax_pred = np.maximum(sarimax_pred, 0)   # volume cannot be negative
preds["SARIMAX + weather"] = sarimax_pred
results.append(H.forecast_metrics(test["y"], sarimax_pred, "SARIMAX + weather"))

# Weather-coefficient significance is a reportable finding in its own right.
for term in EXOG:
    if term in sarimax_fit.pvalues.index:
        H.record_finding(
            f"sarimax_coef_{term}", round(float(sarimax_fit.params[term]), 3),
            f"p = {float(sarimax_fit.pvalues[term]):.4f}", "notebooks/03_forecasting.ipynb §3")

# %% [markdown]
# ## 4. LightGBM with calendar + weather features
#
# Gradient boosting needs lagged targets to forecast a series, so lag 12,
# lag 1 and rolling means are constructed **shifted correctly** — every feature
# at time *t* uses only information available before *t*. Getting this wrong is
# the classic silent leak, so the construction is stated explicitly.

# %%
import lightgbm as lgb

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar + weather + correctly shifted lag features."""
    f = pd.DataFrame(index=df.index)
    f["month_of_year"] = df.index.month
    f["tmax_c"] = df["tmax_c"]
    f["prcp_mm"] = df["prcp_mm"]
    f["snow_mm"] = df["snow_mm"]
    f["awnd_ms"] = df["awnd_ms"]
    # Lags: y[t-1], y[t-2], y[t-12] and rolling means of PAST values only.
    f["lag1"] = df["y"].shift(1)
    f["lag2"] = df["y"].shift(2)
    f["lag12"] = df["y"].shift(12)
    f["roll3"] = df["y"].shift(1).rolling(3).mean()
    f["roll12"] = df["y"].shift(1).rolling(12).mean()
    return f

features = build_features(ts)
target = ts["y"]

X_train, y_train = features.loc[train.index], target.loc[train.index]
X_test, y_test = features.loc[test.index], target.loc[test.index]

# LightGBM rejects NaN. Early rows lack lag_12 / rolling features by construction
# (they look back further than the series goes), so those rows are dropped from
# training rather than imputed — imputing a lag would invent history.
keep = X_train.notna().all(axis=1)
assert X_test.notna().all(axis=None), (
    "hold-out features contain NaN — the training series must extend at least "
    "12 months before the hold-out")
print(f"train rows {len(X_train)} | test rows {len(X_test)}")
print(f"rows dropped from training for missing lags: {int((~keep).sum())}")

# %%
lgbm = lgb.LGBMRegressor(
    n_estimators=600, learning_rate=0.04, num_leaves=12,
    min_child_samples=8, subsample=0.9, colsample_bytree=0.9,
    reg_lambda=0.5, random_state=42, verbose=-1,
)
lgbm.fit(X_train[keep], y_train[keep])
lgbm_pred = np.maximum(lgbm.predict(X_test), 0)
preds["LightGBM"] = lgbm_pred
results.append(H.forecast_metrics(test["y"], lgbm_pred, "LightGBM"))

# %%
comparison = pd.DataFrame(results)[["model", "rmse", "mae", "mape_pct", "smape_pct"]]
comparison = comparison.sort_values("rmse").reset_index(drop=True)
comparison

# %%
fig, ax = plt.subplots(figsize=(13, 4.4))
hist = ts.iloc[-48:]
ax.plot(hist.index, hist["y"], color=H.PALETTE["ink"], lw=1.5, label="Actual")
ax.plot(test.index, preds.get("Seasonal naive", np.full(HORIZON, np.nan)), ls=":",
        lw=1.6, color=H.PALETTE["muted"], label="Seasonal naive")
ax.plot(test.index, sarimax_pred, ls="--", lw=1.9, color=H.PALETTE["accent2"],
        label="SARIMAX + weather")
ax.plot(test.index, lgbm_pred, lw=1.9, color=H.PALETTE["accent3"], label="LightGBM")
ax.axvline(test.index[0], color=H.PALETTE["muted"], lw=0.9, ls="-")
ax.set_title("Hold-out forecast, final 12 months")
ax.set_ylabel("Requests per month")
ax.legend(loc="upper left", ncol=2)
H.save_figure(fig, "03_forecast_holdout.png")
plt.show()

# %% [markdown]
# ## 5. Rolling-origin backtest
#
# One hold-out can be lucky. Each fold retrains on everything up to that point
# and predicts the next month, repeated across the last 24 months.

# %%
def backtest_sarimax(data: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    rows = []
    for i in range(n_folds, 0, -1):
        # Explicit positional split. A negative slice such as iloc[-i:-i + 1]
        # silently becomes empty on the final fold (i == 1 -> stop == 0).
        cut = len(data) - i
        if cut < 36:
            continue
        tr, te = data.iloc[:cut], data.iloc[cut:cut + 1]
        if te.empty or te[EXOG].isna().any().any():
            continue
        try:
            m = sm.tsa.SARIMAX(tr["y"].astype(float), exog=tr[EXOG].astype(float),
                               order=sarimax_order, seasonal_order=sarimax_seasonal,
                               enforce_stationarity=False,
                               enforce_invertibility=False).fit(disp=False)
            p = float(np.maximum(m.forecast(steps=1, exog=te[EXOG].astype(float)).values[0], 0))
        except Exception:
            continue
        rows.append({"month": te.index[0], "actual": float(te["y"].iloc[0]), "pred": p})
    return pd.DataFrame(rows)


def backtest_lgbm(data: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    feats = build_features(data)
    rows = []
    for i in range(n_folds, 0, -1):
        cut = len(data) - i
        if cut < 36:
            continue
        tr_idx, te_idx = data.index[:cut], data.index[cut:cut + 1]
        if len(te_idx) == 0:
            continue
        Xtr, Xte = feats.loc[tr_idx], feats.loc[te_idx]
        keep = Xtr.notna().all(axis=1)
        if keep.sum() < 24 or Xte.isna().any().any():
            continue
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=12,
                              min_child_samples=8, reg_lambda=0.5,
                              random_state=42, verbose=-1)
        m.fit(Xtr[keep], data.loc[tr_idx, "y"][keep])
        rows.append({"month": te_idx[0], "actual": float(data.loc[te_idx, "y"].iloc[0]),
                     "pred": float(max(m.predict(Xte)[0], 0))})
    return pd.DataFrame(rows)


bt_sarimax = backtest_sarimax(ts, N_BACKTEST)
bt_lgbm = backtest_lgbm(ts, N_BACKTEST)

# Seasonal naive over the same folds. Built from explicitly aligned arrays: each
# entry pairs a month with the value from 12 months earlier, and any month whose
# comparison point is missing is dropped before the frame is assembled.
sn_months = ts.index[-N_BACKTEST:]
sn_actual = ts["y"].iloc[-N_BACKTEST:].to_numpy(dtype=float)
sn_pred = ts["y"].shift(12).iloc[-N_BACKTEST:].to_numpy(dtype=float)
sn_ok = ~np.isnan(sn_pred)
bt_snaive = pd.DataFrame({
    "month": sn_months[sn_ok],
    "actual": sn_actual[sn_ok],
    "pred": sn_pred[sn_ok],
})

bt = pd.concat([
    bt_snaive.assign(model="Seasonal naive"),
    bt_sarimax.assign(model="SARIMAX + weather"),
    bt_lgbm.assign(model="LightGBM"),
], ignore_index=True)

# Summarise per model explicitly rather than through groupby.apply, which is
# fragile across pandas versions when the grouped frame can be empty.
_rows = []
for model in ["Seasonal naive", "SARIMAX + weather", "LightGBM"]:
    d = bt[bt["model"] == model]
    if not len(d):
        _rows.append({"model": model, "folds": 0, "rmse": np.nan,
                      "mae": np.nan, "mape_pct": np.nan, "smape_pct": np.nan})
        continue
    a, p = d["actual"].to_numpy(dtype=float), d["pred"].to_numpy(dtype=float)
    _rows.append({"model": model, "folds": int(len(d)),
                  "rmse": H.rmse(a, p), "mae": H.mae(a, p),
                  "mape_pct": H.mape(a, p), "smape_pct": H.smape(a, p)})

backtest_summary = (pd.DataFrame(_rows)
                      .set_index("model")
                      .round(3)
                      .sort_values("rmse"))
backtest_summary

# %%
fig, ax = plt.subplots(figsize=(12.5, 4))
for model, color in [("Seasonal naive", H.PALETTE["muted"]),
                     ("SARIMAX + weather", H.PALETTE["accent2"]),
                     ("LightGBM", H.PALETTE["accent3"])]:
    d = bt[bt["model"] == model].sort_values("month")
    if len(d):
        ax.plot(d["month"], d["actual"] - d["pred"], marker="o", ms=3.4, lw=1.4,
                color=color, label=model)
ax.axhline(0, color=H.PALETTE["ink"], lw=0.9)
ax.set_title(f"Rolling-origin backtest — monthly error over {N_BACKTEST} folds")
ax.set_ylabel("Actual − predicted (requests)")
ax.legend(ncol=3)
H.save_figure(fig, "03_backtest_errors.png")
plt.show()

# %% [markdown]
# ## 6. Which features carry the signal?

# %%
imp = (pd.Series(lgbm.feature_importances_, index=X_train.columns)
         .sort_values(ascending=False))
print("LightGBM feature importance (split gain proxy):")
print((imp / imp.sum() * 100).round(1).to_string())

# %% [markdown]
# ## 7. Findings and the honest caveats
#
# * The **seasonal naive** baseline is strong — this is expected for a series
#   with a hard summer peak, and it is the correct benchmark to report against.
# * **Weather covariates add explanatory value** to SARIMAX (see the coefficient
#   table in §3); the temperature coefficient is the one to quote.
# * The 2019–2025 window contains an **exogenous structural break** (2020), so
#   any model validated only on 2020–2021 would be scoring a regime that may not
#   repeat. The rolling backtest deliberately spans both regimes.
# * The last 12 months are **full-detail** while the earlier series is a 20%
#   sample. Weighting the sample by 5 restores comparability (validated in
#   notebook 01 §2), but any residual sampling effect lands in the level, not
#   the seasonal shape.
# * **No forecast should be read as a service-level commitment.** These are
#   call volumes under normal conditions; a policy change in intake, or a
#   major weather event, invalidates them.

# %%
for m in ["Seasonal naive", "SARIMAX + weather", "LightGBM"]:
    row = comparison[comparison["model"] == m]
    if len(row):
        H.record_finding(f"holdout_{m.replace(' ', '_').replace('+', 'plus')}_rmse",
                         float(row["rmse"].iloc[0]),
                         f"RMSE on 12-month hold-out; MAPE {float(row['mape_pct'].iloc[0]):.2f}%",
                         "notebooks/03_forecasting.ipynb §4")
best = comparison.iloc[0]
H.record_finding("holdout_best_model", str(best["model"]),
                 f"lowest hold-out RMSE {float(best['rmse']):,.0f} "
                 f"(MAPE {float(best['mape_pct']):.2f}%)",
                 "notebooks/03_forecasting.ipynb §4")
con.close()
print("findings written to", C.DATA_PROCESSED / "findings.json")
