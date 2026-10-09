"""
Shared analysis helpers for the Chicago 311 notebooks and dashboard builders.

Design notes
------------
* Everything reads from the DuckDB warehouse built by `src/etl.py`. No notebook
  re-implements a definition that already exists in `sql/01_ingest.sql`.
* The two scopes that matter are made explicit everywhere:

    - DEMAND scope    : every request, including administratively closed
                        information calls (they are still demand).
    - SERVICE scope   : `is_actionable` rows only — real field work. All
                        response-time statistics use this scope.

* Model metrics are computed with statistics that only need a `fittedvalues`
  attribute (statsmodels) or a DataFrame (LightGBM), so the notebooks do not
  depend on a particular statsmodels version's `get_forecast` signature.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

# ---------------------------------------------------------------------------
# Warehouse access
# ---------------------------------------------------------------------------
def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open the analytical warehouse."""
    if not C.DB_PATH.exists():
        raise FileNotFoundError(
            f"{C.DB_PATH} not found — run `python src/etl.py` first")
    con = duckdb.connect(str(C.DB_PATH), read_only=read_only)
    con.execute("INSTALL spatial; LOAD spatial;")  # needed for GeoJSON export
    return con


def query(sql: str, con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Run SQL against the warehouse and return a DataFrame."""
    own = con is None
    con = con or connect()
    try:
        return con.execute(sql).df()
    finally:
        if own:
            con.close()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, ignoring zero-actual points.

    MAPE is undefined at zero, and the whole point of excluding it silently
    would be to flatter the model — so the caller gets NaN when nothing is
    scoreable rather than a fake number.
    """
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    m = a != 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs((a[m] - p[m]) / a[m])) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric MAPE — well-defined when actual is zero."""
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    denom = np.abs(a) + np.abs(p)
    m = denom != 0
    if not m.any():
        return float("nan")
    return float(np.mean(2 * np.abs(p[m] - a[m]) / denom[m]) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.mean(np.abs(a - p)))


def forecast_metrics(actual, predicted, label: str = "") -> dict:
    """Bundle the four error measures used for every forecasting comparison."""
    return {
        "model": label,
        "rmse": round(rmse(actual, predicted), 2),
        "mae": round(mae(actual, predicted), 2),
        "mape_pct": round(mape(actual, predicted), 3),
        "smape_pct": round(smape(actual, predicted), 3),
    }


# ---------------------------------------------------------------------------
# Finding registry
#
# Every headline number in the report is written here by the notebook that
# computes it, so the PDF can quote statistics that were actually produced by
# an executed notebook rather than retyped by hand.
# ---------------------------------------------------------------------------
FINDINGS_PATH = C.DATA_PROCESSED / "findings.json"


def record_finding(key: str, value, note: str = "", source: str = "") -> None:
    """Persist one headline result. Merges into an existing registry."""
    data = {}
    if FINDINGS_PATH.exists():
        try:
            data = json.loads(FINDINGS_PATH.read_text())
        except json.JSONDecodeError:
            data = {}
    data[key] = {"value": value, "note": note, "source": source}
    FINDINGS_PATH.write_text(json.dumps(data, indent=2, default=str))


def load_findings() -> dict:
    if FINDINGS_PATH.exists():
        return json.loads(FINDINGS_PATH.read_text())
    return {}


# ---------------------------------------------------------------------------
# Plot styling — consistent, colour-blind-aware palette
# ---------------------------------------------------------------------------
PALETTE = {
    "ink": "#12232E",
    "muted": "#5A6B7B",
    "grid": "#DCE3E8",
    "accent": "#0B6E99",
    "accent2": "#C1440E",
    "accent3": "#1B998B",
    "warn": "#B3261E",
    # HOLC grades follow the conventional redlining colour language
    "holc": {"A": "#1B998B", "B": "#5B8C5A", "C": "#E0A800", "D": "#B3261E"},
    "bands": ["#0B6E99", "#1B998B", "#E0A800", "#C1440E"],
}


def setup_plot_style() -> None:
    """Apply one consistent matplotlib style across every figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.titleweight": "600",
        "axes.labelsize": 10.5,
        "axes.labelcolor": PALETTE["ink"],
        "axes.edgecolor": PALETTE["grid"],
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.7,
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "text.color": PALETTE["ink"],
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })


def save_figure(fig, name: str) -> Path:
    """Save a figure to reports/figures/ and return the path."""
    out = C.FIGURES / name
    fig.savefig(out, bbox_inches="tight")
    return out


def save_figure_at(fig, path) -> Path:
    """Save a figure to an explicit path (used by the dashboard builders,
    which publish PNGs under dashboards/ rather than reports/)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------
# Small reusable data loaders
# ---------------------------------------------------------------------------
def month_series(con=None) -> pd.DataFrame:
    """Citywide monthly demand, with the exact full-population counts where
    they are available (used to validate the 20% sample)."""
    return query("""
        SELECT
            m.month,
            count(*)                                   AS n_requests,
            count(*) FILTER (WHERE m.is_service_request) AS n_service,
            t.true_count
        FROM analytics.sr_enriched m
        LEFT JOIN analytics.monthly_true_counts t USING (month)
        GROUP BY m.month, t.true_count
        ORDER BY m.month
    """, con)


def holc_labels() -> dict:
    return {
        "A": "A — Best / \"greenlined\"",
        "B": "B — Still desirable",
        "C": "C — Declining / \"yellowlined\"",
        "D": "D — Hazardous / \"redlined\"",
    }
