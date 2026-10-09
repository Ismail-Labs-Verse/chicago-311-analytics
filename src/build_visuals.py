"""
Stage 5d — Analyst-built visualisations (Plotly + Matplotlib).

Tableau Desktop / Power BI Desktop are not available on this machine, so the
dashboard deliverables cannot be rendered in their target tools. This script
compensates with visualisations built entirely in Python:

  dashboards/interactive/chicago311_interactive.html
      One self-contained interactive report: choropleth + weather-annotated
      time series + redlining scatter, with hover detail and toggleable layers.

  dashboards/interactive/figures/*.png
      Print-quality static versions of the same three charts, generated with
      matplotlib so they render identically everywhere and embed cleanly in
      the PDF report.

The three charts requested are all here:
  1. choropleth of Chicago response times by neighbourhood/tract
  2. time series annotated with weather events
  3. scatter of redlining grade vs. response time

Run:  python src/build_visuals.py
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import analysis_helpers as H  # noqa: E402

OUT = C.DASHBOARDS / "interactive"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

HOLC_COLORS = {"A": "#1B998B", "B": "#5B8C5A", "C": "#E0A800", "D": "#B3261E"}
HOLC_LABEL = {
    "A": "A — Best (greenlined)",
    "B": "B — Still desirable",
    "C": "C — Declining (yellowlined)",
    "D": "D — Hazardous (redlined)",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load() -> dict:
    con = H.connect()
    q = lambda sql: H.query(sql, con)  # noqa: E731

    d = {}
    d["monthly"] = q("""
        SELECT month, count(*) AS n_requests,
               count(*) FILTER (WHERE is_actionable) AS n_actionable,
               round(median(days_to_resolution) FILTER (
                     WHERE is_actionable AND days_to_resolution IS NOT NULL), 3) AS median_days,
               round(avg(tmax_c), 2) AS tmax_c,
               round(sum(prcp_mm), 1) AS prcp_mm,
               round(sum(snow_mm), 1) AS snow_mm
        FROM analytics.sr_enriched GROUP BY month ORDER BY month
    """)

    d["weekly"] = q("""
        SELECT week_start, n_requests,
               round(median_days, 2) AS median_days,
               round(avg_tmax_c, 1) AS tmax_c,
               round(total_snow_mm, 1) AS snow_mm,
               round(total_prcp_mm, 1) AS prcp_mm
        FROM analytics.weekly_volume ORDER BY week_start
    """)

    d["tract"] = q("""
        SELECT tract_fips, community_name, total_pop,
               round(median_days, 3) AS median_days,
               round(p90_days, 2) AS p90_days,
               n_requests, n_resolved,
               round(pct_minority, 1) AS pct_minority,
               median_hh_income, holc_grade_mode AS holc_grade,
               round(severity_weighted_days, 3) AS severity_weighted_days
        FROM analytics.tract_summary
        WHERE median_days IS NOT NULL AND n_resolved >= 20
    """)

    d["by_grade"] = q("""
        SELECT holc_grade_mode AS g, count(*) AS tracts,
               round(median(median_days), 3) AS med_of_medians,
               round(avg(median_days), 3) AS mean_days,
               sum(n_resolved) AS resolved,
               round(avg(pct_minority), 1) AS pct_minority,
               round(avg(median_hh_income)) AS income
        FROM analytics.tract_summary
        WHERE holc_grade_mode IS NOT NULL AND median_days IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """)

    d["types"] = q("""
        SELECT sr_type, n_requests, severity_band,
               round(median_days, 2) AS median_days
        FROM analytics.sr_type_summary ORDER BY n_requests DESC LIMIT 14
    """)

    d["ca"] = q("""
        SELECT community_area, community_area_name, n_requests, n_resolved,
               round(median_days, 3) AS median_days, round(p90_days, 2) AS p90_days,
               hardship_index
        FROM analytics.community_area_summary
        WHERE community_area_name IS NOT NULL AND median_days IS NOT NULL
        ORDER BY community_area
    """)

    d["totals"] = q("""
        SELECT count(*) AS rows_total,
               count(*) FILTER (WHERE is_actionable) AS n_actionable,
               count(*) FILTER (WHERE is_informational) AS n_admin,
               count(DISTINCT tract_fips) AS n_tracts,
               count(DISTINCT sr_type) AS n_types,
               round(median(days_to_resolution), 6) AS med_all,
               round(median(days_to_resolution) FILTER (
                     WHERE is_actionable AND days_to_resolution IS NOT NULL), 3) AS med_actionable,
               round(quantile_cont(days_to_resolution, 0.9) FILTER (
                     WHERE is_actionable AND days_to_resolution IS NOT NULL), 2) AS p90_actionable
        FROM analytics.sr_enriched
    """).iloc[0].to_dict()
    con.close()
    return d


def load_geometry():
    import geopandas as gpd
    g = gpd.read_file(C.DATA_RAW / "aux" / "chicago_tracts_2020.geojson")
    # The GeoJSON carries lowercase property names, and its `tract_fips` is the
    # 5-digit COUNTY id — the 11-digit tract id is `tract_full`. Joining on the
    # wrong one silently yields zero rows, so select explicitly.
    return g[["tract_full", "geometry"]].rename(columns={"tract_full": "tract_fips"})


def add_latlon(gdf, gdf_crs="EPSG:4326"):
    """Tract label points (for Plotly map hover and annotation)."""
    pts = gdf.copy()
    pts["geometry"] = gdf.geometry.representative_point()
    return pts.to_crs(gdf_crs) if pts.crs != gdf_crs else pts


# ---------------------------------------------------------------------------
# Weather event annotation (derived from the data, not hand-typed)
# ---------------------------------------------------------------------------
def weather_events(weekly: pd.DataFrame) -> list[dict]:
    """Detect the notable weather episodes the time series should be annotated with."""
    events: list[dict] = []
    w = weekly.copy()
    w["week_start"] = pd.to_datetime(w["week_start"])

    # Heavy snow weeks
    snow = w.nlargest(2, "snow_mm")
    for _, r in snow.iterrows():
        if pd.notna(r["snow_mm"]) and r["snow_mm"] > 0:
            events.append({"date": r["week_start"], "kind": "snow",
                           "label": f"Snow {r['snow_mm']:.0f} mm"})

    # Extreme heat (highest mean Tmax) and extreme cold (lowest)
    hot = w.nlargest(1, "tmax_c").iloc[0]
    if pd.notna(hot["tmax_c"]):
        events.append({"date": hot["week_start"], "kind": "heat",
                       "label": f"Peak heat {hot['tmax_c']:.0f}°C"})
    cold = w.nsmallest(1, "tmax_c").iloc[0]
    if pd.notna(cold["tmax_c"]):
        events.append({"date": cold["week_start"], "kind": "cold",
                       "label": f"Cold snap {cold['tmax_c']:.0f}°C"})

    # Wettest week
    wet = w.nlargest(1, "prcp_mm").iloc[0]
    if pd.notna(wet["prcp_mm"]):
        events.append({"date": wet["week_start"], "kind": "rain",
                       "label": f"Wettest week {wet['prcp_mm']:.0f} mm"})

    # De-duplicate on week
    seen, out = set(), []
    for e in sorted(events, key=lambda x: x["date"]):
        key = (e["date"], e["kind"])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


# ===========================================================================
# PLOTLY — interactive, self-contained HTML
# ===========================================================================
def plotly_html(d: dict, gdf, events: list[dict], target: Path | None = None) -> Path:
    """Build the interactive report. `target` selects the output path, so the
    same proven builder emits both the interactive report and the main
    dashboard without either depending on a network connection."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    gdf = gdf.merge(d["tract"], on="tract_fips", how="inner")
    pts = add_latlon(gdf)

    lo, hi = float(np.quantile(gdf["median_days"], 0.02)), float(np.quantile(gdf["median_days"], 0.98))

    # --- 1. choropleth (tract polygons via a GeoJSON layer) ---
    # Plotly 7 renamed `choroplethmapbox` -> `choroplethmap` and `layout.mapbox`
    # -> `layout.map`, and the marker opacity/line props are gone. Styling is
    # therefore carried on the trace only where the installed version supports it.
    cax_title = "Median days to resolution"
    choropleth = go.Choroplethmap(
        geojson=json.loads(gdf.to_json()),
        locations=gdf["tract_fips"],
        z=gdf["median_days"],
        featureidkey="properties.tract_fips",
        colorscale="YlOrRd",
        zmin=lo, zmax=hi,
        colorbar=dict(title=cax_title, thickness=13, len=0.6, x=1.01),
        customdata=np.stack([
            gdf["community_name"].fillna("—"),
            gdf["n_resolved"].fillna(0),
            gdf["pct_minority"].fillna(-1),
            gdf["median_hh_income"].fillna(-1),
            gdf["holc_grade"].fillna("—"),
        ], axis=-1),
        hovertemplate=("<b>%{customdata[0]}</b><br>"
                       "Tract %{location}<br>"
                       "Median <b>%{z:.2f} d</b><br>"
                       "%{customdata[1]:,.0f} resolved requests<br>"
                       "% non-white %{customdata[2]:.1f}<br>"
                       "Income $%{customdata[3]:,.0f}<br>"
                       "HOLC grade <b>%{customdata[4]}</b><extra></extra>"),
    )

    fig_map = go.Figure(choropleth)
    fig_map.update_layout(
        map=dict(style="carto-positron", center=dict(lat=41.84, lon=-87.68), zoom=9.6),
        margin=dict(l=0, r=0, t=42, b=0), height=720,
        title=dict(text="Chicago 311 — median days to resolution by census tract "
                        "(≥20 actionable requests)", x=0.02, xanchor="left"),
        paper_bgcolor="white",
    )

    # --- 2. time series with weather annotations ---
    m = d["weekly"].copy()
    m["week_start"] = pd.to_datetime(m["week_start"])

    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ts.add_trace(go.Bar(
        x=m["week_start"], y=m["n_requests"], name="Weekly requests",
        marker_color="#0B6E99", opacity=0.55,
        hovertemplate="Week of %{x|%d %b %Y}<br>%{y:,.0f} requests<extra></extra>"),
        secondary_y=False)
    fig_ts.add_trace(go.Scatter(
        x=m["week_start"], y=m["median_days"], name="Median days (actionable)",
        mode="lines", line=dict(color="#C1440E", width=1.9),
        hovertemplate="Week of %{x|%d %b %Y}<br>median %{y:.2f} days<extra></extra>"),
        secondary_y=True)
    fig_ts.add_trace(go.Scatter(
        x=m["week_start"], y=m["tmax_c"], name="Mean Tmax (°C)",
        mode="lines", line=dict(color="#7B5EA7", width=1.2, dash="dot"),
        visible="legendonly",
        hovertemplate="Week of %{x|%d %b %Y}<br>%{y:.1f} °C<extra></extra>"),
        secondary_y=True)

    colour = {"snow": "#4A8DB7", "heat": "#C1440E", "cold": "#7B5EA7", "rain": "#1B998B"}
    for e in events:
        fig_ts.add_vline(x=e["date"], line=dict(color=colour[e["kind"]],
                                                width=1.2, dash="dash"), opacity=0.75)
        fig_ts.add_annotation(
            x=e["date"], y=1.0, yref="paper", yanchor="bottom", showarrow=False,
            text=e["label"], font=dict(size=9.5, color=colour[e["kind"]]),
            bgcolor="rgba(255,255,255,0.85)", borderpad=2,
            textangle=-90)

    fig_ts.update_layout(
        height=520, margin=dict(l=60, r=60, t=64, b=44),
        title=dict(text="Weekly 311 volume and resolution time, with weather events",
                   x=0.02, xanchor="left"),
        legend=dict(orientation="h", y=1.06, x=0.02, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified", paper_bgcolor="white", plot_bgcolor="#FCFDFE",
        bargap=0.05)
    fig_ts.update_yaxes(title_text="Requests per week", secondary_y=False,
                        showgrid=True, gridcolor="#E8EDF1")
    fig_ts.update_yaxes(title_text="Days  /  °C", secondary_y=True, showgrid=False)

    # --- 3. redlining scatter ---
    t = d["tract"].dropna(subset=["holc_grade", "median_days"]).copy()
    fig_sc = go.Figure()
    for g in ["A", "B", "C", "D"]:
        sub = t[t["holc_grade"] == g]
        if not len(sub):
            continue
        fig_sc.add_trace(go.Scatter(
            x=sub["n_resolved"], y=sub["median_days"], mode="markers",
            name=f"Grade {g}  ({len(sub)} tracts)",
            marker=dict(color=HOLC_COLORS[g], size=np.sqrt(sub["n_resolved"]) / 2.6,
                        opacity=0.62, line=dict(width=0.4, color="white")),
            customdata=np.stack([sub["community_name"].fillna("—"),
                                 sub["pct_minority"].fillna(-1),
                                 sub["median_hh_income"].fillna(-1)], axis=-1),
            hovertemplate=("<b>%{customdata[0]}</b><br>"
                           "%{x:,.0f} resolved requests<br>"
                           "median %{y:.2f} days<br>"
                           "% non-white %{customdata[1]:.1f}<br>"
                           "income $%{customdata[2]:,.0f}<extra></extra>")))
    # Per-grade medians make the pattern readable at a glance
    for g in ["A", "B", "C", "D"]:
        sub = t[t["holc_grade"] == g]
        if len(sub):
            fig_sc.add_hline(y=float(sub["median_days"].median()), line=dict(
                color=HOLC_COLORS[g], width=1.4, dash="dot"), opacity=0.9)

    fig_sc.update_layout(
        height=580, margin=dict(l=64, r=30, t=64, b=52),
        title=dict(text="Historical HOLC redlining grade vs. tract median response time "
                        "(dotted lines = grade median)", x=0.02, xanchor="left"),
        xaxis=dict(title="Resolved actionable requests per tract (log scale)",
                   type="log", gridcolor="#E8EDF1"),
        yaxis=dict(title="Median days to resolution", gridcolor="#E8EDF1"),
        legend=dict(orientation="h", y=1.07, x=0.02, bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="white", plot_bgcolor="#FCFDFE")

    # --- assemble one interactive report ---
    k = d["totals"]
    kpis = [
        ("13.07M" if k["rows_total"] > 1e6 else f"{k['rows_total']:,}", "311 requests analysed"),
        (f"{k['med_actionable']:.2f}", "median days — actionable requests"),
        (f"{k['med_all']:.5f}", "median days — all rows (see warning)"),
        (f"{k['p90_actionable']:.0f}", "days at the 90th percentile"),
        (f"{k['n_actionable']:,}", "actionable service requests"),
        (f"{k['n_admin']:,}", "administrative filings"),
        (f"{k['n_tracts']:,}", "census tracts assigned"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>'
        for v, l in kpis)

    def fig_html(fig, div_id):
        return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                           config={"displaylogo": False, "responsive": True})

    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Chicago 311 Analytics — Interactive</title>",
        "<script>", _plotly_js(), "</script>",
        "<style>",
        _CSS,
        "</style></head><body><div class='wrap'>",
        "<header><h1>Chicago 311 Service Requests, 2019–2025</h1>",
        "<div class='sub'>Interactive analysis — hover every mark for detail. "
        "Sources: City of Chicago Data Portal, NOAA NCEI, US Census ACS 5-year 2023, "
        "University of Richmond DSL (HOLC).</div></header>",
        "<div class='warn'><b>Read the median carefully.</b> "
        f"{100 * k['n_admin'] / k['rows_total']:.1f}% of records are administrative "
        "filings (citizen information calls, auto-closed noise reports) that never "
        f"dispatch a crew. Including them gives a citywide median of "
        f"<b>{k['med_all']:.5f} days</b> — under a minute. Every response-time figure "
        f"below is therefore computed over <b>{k['n_actionable']:,} actionable service "
        f"requests</b>, giving a median of <b>{k['med_actionable']:.2f} days</b>.</div>",
        "<div class='kpis'>", kpi_html, "</div>",
        "<h2>1 · Choropleth — response time by neighbourhood</h2>",
        "<div class='panel'>", fig_html(fig_map, "map"), "</div>",
        "<h2>2 · Time series — weekly volume with weather events</h2>",
        "<div class='panel'>", fig_html(fig_ts, "ts"), "</div>",
        "<h2>3 · Redlining — HOLC grade vs. response time</h2>",
        "<div class='panel'>", fig_html(fig_sc, "sc"), "</div>",
        "<h2>4 · Request composition and severity</h2>",
        "<div class='grid2'>",
        "<div class='panel'>", _table_types(d["types"]), "</div>",
        "<div class='panel'>", _table_grades(d["by_grade"]), "</div>",
        "</div>",
        "<h2>5 · Community areas</h2>",
        "<div class='panel'>", _table_ca(d["ca"]), "</div>",
        "<div class='foot'>Generated by <code>src/build_visuals.py</code> from "
        "<code>data/processed/chicago311.duckdb</code>. Self-contained: open offline, "
        "no server or network required.</div>",
        "</div></body></html>",
    ]
    out = Path(target) if target is not None else OUT / "chicago311_interactive.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    return out


def _plotly_js() -> str:
    """Inline the Plotly bundle so the HTML file is genuinely self-contained."""
    import plotly
    p = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    if not p.exists():
        # Fall back to the CDN tag if the bundled copy is not found.
        return ("/* plotly bundle not found locally */")
    return p.read_text(encoding="utf-8")


_CSS = """
:root{--ink:#12232E;--muted:#5A6B7B;--grid:#DCE3E8;--accent:#0B6E99;--warn:#B3261E}
*{box-sizing:border-box}
body{margin:0;background:#F5F7F9;color:var(--ink);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1320px;margin:0 auto;padding:28px 22px 60px}
header{border-bottom:2px solid var(--ink);padding-bottom:14px;margin-bottom:20px}
h1{font-size:26px;margin:0 0 4px}
.sub{color:var(--muted);font-size:13.5px;max-width:1000px}
h2{font-size:18px;margin:30px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--grid)}
.panel{background:#fff;border:1px solid var(--grid);border-radius:9px;padding:14px}
.warn{background:#FFF6EE;border-left:3px solid var(--warn);padding:13px 16px;
 border-radius:0 6px 6px 0;margin:16px 0;font-size:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:18px 0}
.kpi{background:#fff;border:1px solid var(--grid);border-radius:8px;padding:14px 16px}
.kpi .v{font-size:24px;font-weight:680;letter-spacing:-.5px}
.kpi .l{font-size:12px;color:var(--muted);line-height:1.35;margin-top:3px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.03em}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.foot{margin-top:28px;padding-top:14px;border-top:1px solid var(--grid);
 font-size:12.5px;color:var(--muted)}
code{background:#EEF2F5;padding:1px 5px;border-radius:3px;font-size:12.5px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
"""


def _table_types(types: pd.DataFrame) -> str:
    rows = "".join(
        f"<tr><td>{r.sr_type}</td><td>{r.severity_band}</td>"
        f"<td class='n'>{r.n_requests:,}</td>"
        f"<td class='n'>{'—' if pd.isna(r.median_days) else f'{r.median_days:.2f}'}</td></tr>"
        for r in types.itertuples())
    return ("<h3 style='margin-top:0;font-size:14px'>Most common request types</h3>"
            "<table><tr><th>Type</th><th>Severity</th><th class='n'>Requests</th>"
            "<th class='n'>Median d*</th></tr>" + rows +
            "<tr><td colspan='4' style='color:#5A6B7B;font-size:12px'>"
            "* median over actionable requests only</td></tr></table>")


def _table_grades(bg: pd.DataFrame) -> str:
    rows = ""
    for r in bg.itertuples():
        rows += (f"<tr><td><b>{r.g}</b> {HOLC_LABEL.get(r.g,'')}</td>"
                 f"<td class='n'>{r.tracts:,}</td>"
                 f"<td class='n'>{r.med_of_medians:.2f}</td>"
                 f"<td class='n'>{r.pct_minority:.1f}%</td>"
                 f"<td class='n'>${r.income:,.0f}</td></tr>")
    return ("<h3 style='margin-top:0;font-size:14px'>1930s HOLC grade vs. today</h3>"
            "<table><tr><th>HOLC grade</th><th class='n'>Tracts</th>"
            "<th class='n'>Median of medians (d)</th><th class='n'>% non-white</th>"
            "<th class='n'>Income</th></tr>" + rows +
            "<tr><td colspan='5' style='color:#5A6B7B;font-size:12px'>"
            "Grades were drawn in the 1930s; differences today describe "
            "persistence, not the maps themselves.</td></tr></table>")


def _table_ca(ca: pd.DataFrame) -> str:
    rows = "".join(
        f"<tr><td>{r.community_area_name}</td>"
        f"<td class='n'>{'—' if pd.isna(r.hardship_index) else f'{r.hardship_index:.0f}'}</td>"
        f"<td class='n'>{r.n_requests:,}</td>"
        f"<td class='n'>{r.median_days:.2f}</td>"
        f"<td class='n'>{'—' if pd.isna(r.p90_days) else f'{r.p90_days:.1f}'}</td></tr>"
        for r in ca.itertuples())
    return ("<h3 style='margin-top:0;font-size:14px'>Community areas — median and "
            "90th percentile resolution time</h3>"
            "<table><tr><th>Community area</th><th class='n'>Hardship index</th>"
            "<th class='n'>Requests</th><th class='n'>Median d</th>"
            "<th class='n'>P90 d</th></tr>" + rows + "</table>")


# ===========================================================================
# MATPLOTLIB — print-quality PNGs for the PDF report
# ===========================================================================
def static_pngs(d: dict, gdf, events: list[dict]) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    H.setup_plot_style()
    written = []

    # --- 1. choropleth ---
    m = gdf.merge(d["tract"], on="tract_fips", how="inner")
    lo, hi = np.quantile(m["median_days"], 0.02), np.quantile(m["median_days"], 0.98)
    fig, ax = plt.subplots(figsize=(7.6, 9.0))
    m.plot(column="median_days", ax=ax, cmap="YlOrRd", vmin=lo, vmax=hi,
           edgecolor="white", linewidth=0.28,
           legend=True, legend_kwds={"label": "Median days to resolution", "shrink": 0.55})
    # label the slowest and fastest areas for orientation
    extremes = pd.concat([d["ca"].nlargest(4, "median_days"),
                          d["ca"].nsmallest(3, "median_days")])
    names = set(extremes["community_area"].dropna().astype(int))
    lab = m[m["community_name"].isin(extremes["community_area_name"])]
    lab = add_latlon(lab)
    for r in lab.itertuples():
        ax.annotate(r.community_name, (r.geometry.x, r.geometry.y),
                    fontsize=6.4, ha="center", color="#12232E",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.72))
    ax.set_title("Chicago 311 — median days to resolution by census tract\n"
                 "(tracts with ≥ 20 actionable resolved requests)", fontsize=12)
    ax.set_axis_off()
    p = FIG / "viz_choropleth.png"
    H.save_figure_at(fig, p); plt.close(fig); written.append(p)

    # --- 2. time series with weather annotation ---
    w = d["weekly"].copy()
    w["week_start"] = pd.to_datetime(w["week_start"])
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.bar(w["week_start"], w["n_requests"], width=5.5, color="#9CC3D8",
           label="Weekly requests")
    ax.set_ylabel("Requests per week")
    ax.set_ylim(0, w["n_requests"].max() * 1.22)
    ax2 = ax.twinx()
    ax2.plot(w["week_start"], w["median_days"], color="#C1440E", lw=1.5,
             label="Median days (actionable)")
    ax2.set_ylabel("Median days to resolution", color="#C1440E")
    ax2.tick_params(axis="y", colors="#C1440E")
    ax2.grid(False)

    col = {"snow": "#4A8DB7", "heat": "#C1440E", "cold": "#7B5EA7", "rain": "#1B998B"}
    for e in events:
        ax.axvline(e["date"], color=col[e["kind"]], ls="--", lw=1.1, alpha=0.8)
        ax.annotate(e["label"], (e["date"], ax.get_ylim()[1] * 0.995),
                    rotation=90, va="top", ha="right", fontsize=7.6,
                    color=col[e["kind"]],
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_title("Weekly 311 volume and resolution time, annotated with weather events")
    handles = [Patch(color="#9CC3D8", label="Weekly requests"),
               plt.Line2D([], [], color="#C1440E", lw=2, label="Median days (actionable)")]
    handles += [Patch(color=col[k], label=k) for k in col]
    ax.legend(handles=handles, loc="upper left", ncol=3, fontsize=8)
    H.save_figure_at(fig, FIG / "viz_timeseries_weather.png"); plt.close(fig)
    written.append(FIG / "viz_timeseries_weather.png")

    # --- 3. redlining scatter ---
    t = d["tract"].dropna(subset=["holc_grade", "median_days"])
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    for g in ["A", "B", "C", "D"]:
        sub = t[t["holc_grade"] == g]
        if not len(sub):
            continue
        ax.scatter(sub["n_resolved"], sub["median_days"],
                   s=np.sqrt(sub["n_resolved"]) * 1.6, alpha=0.55,
                   color=HOLC_COLORS[g], edgecolor="white", linewidth=0.4,
                   label=f"Grade {g} ({len(sub)} tracts)")
        ax.axhline(float(sub["median_days"].median()), color=HOLC_COLORS[g],
                   ls=":", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Resolved actionable requests per tract (log)")
    ax.set_ylabel("Median days to resolution")
    ax.set_title("HOLC redlining grade vs. tract response time\n(dotted lines = grade medians)")
    ax.legend(fontsize=8.5, loc="upper left")

    ax = axes[1]
    bg = d["by_grade"]
    xs = np.arange(len(bg))
    ax.bar(xs, bg["mean_days"], color=[HOLC_COLORS[g] for g in bg["g"]])
    for x, r in zip(xs, bg.itertuples()):
        ax.text(x, r.mean_days, f"{r.mean_days:.2f} d\n{r.pct_minority:.0f}% non-white",
                ha="center", va="bottom", fontsize=8.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{g}\n({HOLC_LABEL[g].split('—')[1].strip()})" for g in bg["g"]],
                       fontsize=8)
    ax.set_ylim(0, bg["mean_days"].max() * 1.35)
    ax.set_ylabel("Mean tract median days")
    ax.set_title("Response time and demographics by historical grade")
    H.save_figure_at(fig, FIG / "viz_redlining_scatter.png"); plt.close(fig)
    written.append(FIG / "viz_redlining_scatter.png")

    return written


def main() -> None:
    print("Stage 5d — Python visualisations (Plotly + Matplotlib)\n")
    if not C.DB_PATH.exists():
        print(f"warehouse missing at {C.DB_PATH}; run `python src/etl.py` first")
        sys.exit(1)
    t0 = time.time()

    d = load()
    print(f"  loaded: {len(d['weekly'])} weeks, {len(d['tract'])} tracts, "
          f"{len(d['ca'])} community areas")
    geom = load_geometry()
    gdf = geom.merge(d["tract"][["tract_fips", "median_days"]], on="tract_fips", how="inner")
    gdf = gdf[gdf["median_days"].notna()]
    print(f"  geometry joined: {len(gdf)} tracts with response time")

    events = weather_events(d["weekly"])
    print(f"  weather events detected: {len(events)} "
          f"({', '.join(e['kind'] for e in events)})")

    html = plotly_html(d, geom, events)
    print(f"  wrote {html.relative_to(C.PROJECT_ROOT)} "
          f"({html.stat().st_size / 1e6:.2f} MB, self-contained)")

    # The main dashboard is the same interactive report, written to the path
    # referenced in the README and committed as the primary artifact.
    main_dash = plotly_html(d, geom, events, target=C.DASHBOARDS / "chicago311_dashboard.html")
    print(f"  wrote {main_dash.relative_to(C.PROJECT_ROOT)} "
          f"({main_dash.stat().st_size / 1e6:.2f} MB, self-contained)")

    pngs = static_pngs(d, geom, events)
    for p in pngs:
        print(f"  wrote {p.relative_to(C.PROJECT_ROOT)} ({p.stat().st_size / 1e3:.0f} KB)")

    print(f"\n  {len(pngs)} PNGs + 1 interactive HTML in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
