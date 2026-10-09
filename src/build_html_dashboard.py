"""
Stage 5c — Self-contained HTML dashboard.

Tableau Desktop and Power BI Desktop cannot be run headlessly here, so neither
of those deliverables can be rendered by the machine that produced the data.
This script closes that gap: it emits ONE `.html` file with the real aggregates
and a real tract choropleth inlined — no server, no CDN, no external asset — so
the analysis is viewable by opening a file.

Run:  python src/build_html_dashboard.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import analysis_helpers as H  # noqa: E402

# NOTE: this builder emits a lightweight, dependency-free SVG/HTML dashboard.
# The PRIMARY interactive dashboard (Plotly, with the library inlined so it works
# offline) is produced by src/build_visuals.py, and that script overwrites
# `dashboards/chicago311_dashboard.html`. This one writes to a separate path so
# the two builders cannot clobber each other.
OUT_HTML = C.DASHBOARDS / "chicago311_dashboard_svg.html"
MAP_WIDTH, MAP_HEIGHT = 760, 900
SIMPLIFY_TOL = 0.0006


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def collect() -> dict:
    con = H.connect()
    q = lambda sql: H.query(sql, con)  # noqa: E731

    totals = q("""
        SELECT count(*) AS rows_total,
               count(DISTINCT sr_type) AS n_types,
               count(DISTINCT tract_fips) AS n_tracts,
               count(*) FILTER (WHERE is_informational) AS n_admin,
               count(*) FILTER (WHERE is_actionable) AS n_actionable
        FROM analytics.sr_enriched
    """).iloc[0]

    med = q("""
        SELECT round(median(days_to_resolution)
               FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL), 2) AS med
        FROM analytics.sr_enriched
    """).iloc[0]["med"]

    monthly = q("""
        SELECT month, count(*) AS n,
               count(*) FILTER (WHERE is_actionable) AS n_act,
               round(median(days_to_resolution)
                     FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL), 3) AS med_days
        FROM analytics.sr_enriched GROUP BY month ORDER BY month
    """)

    grades = q("""
        SELECT holc_grade_mode AS holc_grade,
               count(DISTINCT tract_fips) AS tracts,
               round(avg(median_days), 2) AS tract_avg_med,
               round(avg(pct_minority), 1) AS pct_minority,
               round(avg(median_hh_income)) AS income,
               sum(n_resolved) AS resolved
        FROM analytics.tract_summary
        WHERE holc_grade_mode IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """)

    types_ = q("""
        SELECT sr_type, n_requests, severity_band, round(median_days, 2) AS median_days
        FROM analytics.sr_type_summary
        ORDER BY n_requests DESC LIMIT 12
    """)

    ca = q("""
        SELECT community_area_name, round(hardship_index) AS hardship_index,
               n_requests, round(median_days, 2) AS median_days,
               round(p90_days, 1) AS p90_days
        FROM analytics.community_area_summary
        WHERE community_area_name IS NOT NULL AND median_days IS NOT NULL
        ORDER BY median_days DESC LIMIT 12
    """)

    weekly = q("""
        SELECT week_start, n_requests, round(median_days, 2) AS median_days
        FROM analytics.weekly_volume ORDER BY week_start
    """)

    quality = json.loads((C.DATA_PROCESSED / "data_quality_report.json").read_text()) \
        if (C.DATA_PROCESSED / "data_quality_report.json").exists() else {}

    findings = H.load_findings()
    con.close()

    return {
        "totals": {k: (int(v) if k != "med" else v) for k, v in totals.items()},
        "median_all": float(med) if med is not None else None,
        "monthly": monthly.to_dict("records"),
        "grades": grades.to_dict("records"),
        "types": types_.to_dict("records"),
        "communities": ca.to_dict("records"),
        "weekly": weekly.to_dict("records"),
        "quality": quality,
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# Choropleth
# ---------------------------------------------------------------------------
def build_map() -> tuple[list[dict], dict]:
    """Simplify tract polygons into SVG paths coloured by median response time."""
    import geopandas as gpd

    con = H.connect()
    metrics = H.query("""
        SELECT tract_fips, community_name, median_days, n_resolved,
               pct_minority, median_hh_income, holc_grade_mode
        FROM analytics.tract_summary
        WHERE median_days IS NOT NULL AND n_resolved >= 20
    """, con)
    con.close()

    geom = gpd.read_file(C.DATA_RAW / "aux" / "chicago_tracts_2020.geojson")
    geom = geom[["tract_full", "geometry"]].rename(columns={"tract_full": "tract_fips"})
    gdf = geom.merge(metrics, on="tract_fips", how="inner")

    # Equirectangular projection with a latitude-corrected x scale is adequate at
    # city scale and keeps the path arithmetic trivial.
    lat0 = gdf.geometry.centroid.y.mean()
    kx = np.cos(np.radians(lat0))
    minx, miny, maxx, maxy = gdf.total_bounds
    spanx, spany = (maxx - minx) * kx, (maxy - miny)

    def to_svg(x, y):
        sx = (x - minx) * kx / spanx * MAP_WIDTH
        sy = MAP_HEIGHT - (y - miny) / spany * MAP_HEIGHT
        return sx, sy

    def ring_path(coords):
        pts = [to_svg(x, y) for x, y in coords]
        if len(pts) < 3:
            return ""
        d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
        d += "".join(f"L{x:.1f},{y:.1f}" for x, y in pts[1:])
        return d + "Z"

    def polygon_path(poly):
        parts = [ring_path(poly.exterior.coords)]
        for hole in poly.interiors:
            parts.append(ring_path(hole.coords))
        return " ".join(p for p in parts if p)

    feats = []
    for row in gdf.itertuples():
        g = row.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
        if g.is_empty:
            continue
        polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        d = " ".join(polygon_path(p) for p in polys)
        if not d:
            continue
        feats.append({
            "fips": row.tract_fips,
            "name": row.community_name,
            "days": round(float(row.median_days), 2),
            "n": int(row.n_resolved),
            "minority": round(float(row.pct_minority), 1) if row.pct_minority == row.pct_minority else None,
            "income": int(row.median_hh_income) if row.median_hh_income == row.median_hh_income else None,
            "holc": row.holc_grade_mode if isinstance(row.holc_grade_mode, str) else None,
            "d": d,
        })

    vals = [f["days"] for f in feats]
    scale = {
        "min": float(np.min(vals)),
        "q20": float(np.quantile(vals, 0.20)),
        "q40": float(np.quantile(vals, 0.40)),
        "q60": float(np.quantile(vals, 0.60)),
        "q80": float(np.quantile(vals, 0.80)),
        "max": float(np.max(vals)),
    }
    return feats, scale


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chicago 311 Request Analytics — 2019–2025</title>
<style>
  :root{--ink:#12232E;--muted:#5A6B7B;--grid:#DCE3E8;--accent:#0B6E99;
        --accent2:#C1440E;--accent3:#1B998B;--warn:#B3261E;--card:#FFFFFF;
        --bg:#F5F7F9;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:1240px;margin:0 auto;padding:28px 22px 70px}
  header{border-bottom:2px solid var(--ink);padding-bottom:14px;margin-bottom:22px}
  h1{font-size:27px;margin:0 0 4px}
  .sub{color:var(--muted);font-size:14px}
  h2{font-size:19px;margin:34px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--grid)}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}
  .card{background:var(--card);border:1px solid var(--grid);border-radius:9px;padding:15px 17px}
  .card .v{font-size:26px;font-weight:650;letter-spacing:-.5px}
  .card .l{font-size:12.5px;color:var(--muted);margin-top:3px;line-height:1.35}
  .panel{background:var(--card);border:1px solid var(--grid);border-radius:9px;padding:18px;margin-top:14px}
  table{border-collapse:collapse;width:100%;font-size:13.5px}
  th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--grid)}
  th{color:var(--muted);font-weight:600;font-size:12.5px;text-transform:uppercase;letter-spacing:.03em}
  td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .mapwrap{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:20px;align-items:start}
  svg{max-width:100%;height:auto;display:block}
  .legend{font-size:13px}
  .legend div{display:flex;align-items:center;gap:8px;margin:5px 0}
  .sw{width:22px;height:12px;border-radius:2px;flex:0 0 auto}
  .note{font-size:13px;color:var(--muted);margin-top:10px;line-height:1.5}
  #tip{position:fixed;pointer-events:none;background:rgba(18,35,46,.95);color:#fff;
       padding:8px 11px;border-radius:6px;font-size:12.5px;opacity:0;transition:opacity .1s;
       max-width:270px;z-index:99;line-height:1.45}
  .flag{background:#FFF4E5;border-left:3px solid var(--accent2);padding:12px 15px;
        border-radius:5px;margin-top:14px;font-size:13.5px}
  .ok{background:#EAF6F3;border-left:3px solid var(--accent3);padding:12px 15px;
      border-radius:5px;margin-top:14px;font-size:13.5px}
  code{background:#EEF2F5;padding:1px 5px;border-radius:3px;font-size:12.5px}
  @media(max-width:900px){.grid2,.mapwrap{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Chicago 311 Service Requests, 2019&ndash;2025</h1>
  <div class="sub">__SUBTITLE__</div>
</header>

<h2>Headline numbers</h2>
<div class="cards" id="kpis"></div>

<div class="flag"><b>Read the median carefully.</b> __ADMIN_NOTE__</div>

<h2>Demand over time</h2>
<div class="panel">
  <div id="chart-monthly"></div>
  <div class="note">Monthly request volume and the median resolution time for
  actionable service requests. Volume includes administratively closed filings
  because they are still load on the 311 system; the timing line does not.</div>
</div>

<h2>Equity &mdash; response time by neighbourhood</h2>
<div class="panel">
  <div class="mapwrap">
    <div>
      <div id="choropleth"></div>
      <div class="note">Tract median days to resolution for actionable requests
      (tracts with at least 20 resolved requests). Hover a tract for its
      demographics and historical HOLC grade.</div>
    </div>
    <div class="legend" id="maplegend"></div>
  </div>
</div>

<div class="grid2">
  <div class="panel">
    <h3 style="margin-top:0;font-size:15px">1930s HOLC grade vs. today</h3>
    <table id="tbl-holc"></table>
    <div class="note">HOLC grades were drawn in the 1930s. Differences today
    describe persistence, not the maps themselves.</div>
  </div>
  <div class="panel">
    <h3 style="margin-top:0;font-size:15px">Slowest community areas</h3>
    <table id="tbl-ca"></table>
    <div class="note">Median days for actionable requests, with the 90th
    percentile to show how bad the tail is.</div>
  </div>
</div>

<h2>Composition &amp; severity</h2>
<div class="panel"><table id="tbl-types"></table></div>

<h2>Weekly volume (anomaly review)</h2>
<div class="panel">
  <div id="chart-weekly"></div>
  <div class="note">Weeks that depart sharply from the seasonal norm are the
  ones notebook 04 flags with Isolation Forest for manual triage.</div>
</div>

<h2>Data quality</h2>
<div class="panel" id="quality"></div>

<h2>Findings recorded by the notebooks</h2>
<div class="panel" id="findings"></div>

<div class="note" style="margin-top:30px;padding-top:14px;border-top:1px solid var(--grid)">
Generated by <code>src/build_html_dashboard.py</code> from
<code>data/processed/chicago311.duckdb</code>. Sources: City of Chicago Data
Portal (311, tract boundaries, socioeconomic indicators), NOAA NCEI (daily
weather), US Census ACS 5-year 2023 table-based Summary File, and the
University of Richmond Digital Scholarship Lab HOLC grades.
</div>
</div>
<div id="tip"></div>

<script>
const DATA = __DATA__;
const MAP = __MAP__;
const SCALE = __SCALE__;

const fmt = n => n===null||n===undefined ? "—" : Number(n).toLocaleString("en-US");
const fmt1 = n => n===null||n===undefined ? "—" : Number(n).toFixed(1);
const fmt2 = n => n===null||n===undefined ? "—" : Number(n).toFixed(2);

/* ---------- KPIs ---------- */
const t = DATA.totals;
document.getElementById("kpis").innerHTML = [
  [fmt(t.rows_total), "311 requests analysed"],
  [fmt1(DATA.median_all), "Median days — all rows (see note)"],
  [fmt(SCALE.median_days), "Median days — actionable requests"],
  [fmt(t.n_actionable), "Actionable service requests"],
  [fmt(t.n_admin), "Administratively closed filings"],
  [fmt(t.n_types), "Distinct request types"],
  [fmt(t.n_tracts), "Census tracts assigned"],
].map(([v,l]) => `<div class="card"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");

/* ---------- helpers ---------- */
function svgEl(tag, attrs={}) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
function lineChart(mount, rows, xKey, yKey, opts={}) {
  const W = 1080, H = opts.height || 260, P = {t:14,r:16,b:34,l:66};
  const svg = svgEl("svg", {viewBox:`0 0 ${W} ${H}`, width:"100%"});
  const xs = rows.map((_,i)=>i);
  const ys = rows.map(r=>Number(r[yKey])).filter(v=>v===v);
  const ymax = opts.ymax || Math.max(...ys)*1.08, ymin = opts.ymin ?? 0;
  const X = i => P.l + (i/(Math.max(rows.length-1,1)))*(W-P.l-P.r);
  const Y = v => H-P.b - ((v-ymin)/(ymax-ymin||1))*(H-P.t-P.b);
  // gridlines
  for (let g=0; g<=4; g++){
    const v = ymin + (ymax-ymin)*g/4;
    svg.appendChild(svgEl("line",{x1:P.l,x2:W-P.r,y1:Y(v),y2:Y(v),
      stroke:"#DCE3E8","stroke-width":1}));
    const lab = svgEl("text",{x:P.l-8,y:Y(v)+4,"text-anchor":"end",
      "font-size":11,fill:"#5A6B7B"});
    lab.textContent = opts.intFmt ? Math.round(v).toLocaleString("en-US") : v.toFixed(1);
    svg.appendChild(lab);
  }
  const d = rows.map((r,i)=>`${i?"L":"M"}${X(i).toFixed(1)},${Y(Number(r[yKey])).toFixed(1)}`).join("");
  svg.appendChild(svgEl("path",{d, fill:"none", stroke:opts.color||"#0B6E99","stroke-width":1.8}));
  // x labels ~6 ticks
  const step = Math.ceil(rows.length/6);
  rows.forEach((r,i)=>{
    if (i%step) return;
    const lab = svgEl("text",{x:X(i),y:H-12,"text-anchor":"middle","font-size":11,fill:"#5A6B7B"});
    lab.textContent = String(r[xKey]).slice(0,7);
    svg.appendChild(lab);
  });
  const mount_ = document.getElementById(mount);
  mount_.innerHTML = "";
  mount_.appendChild(svg);
}

/* ---------- monthly ---------- */
/* Mount targets must exist in the document: assigning innerHTML to a null
   element throws and would abort every chart defined after it. */
lineChart("chart-monthly", DATA.monthly, "month", "n", {color:"#0B6E99", intFmt:true});

/* ---------- weekly ---------- */
lineChart("chart-weekly", DATA.weekly, "week_start", "n_requests",
          {color:"#1B998B", intFmt:true, height:230});

/* ---------- choropleth ---------- */
const COLORS = ["#E8F1F5","#BBD8E6","#7FB6D1","#4A8DB7","#1F5F8B","#0B3350"];
function bucket(v){
  if (v<=SCALE.q20) return 0;
  if (v<=SCALE.q40) return 1;
  if (v<=SCALE.q60) return 2;
  if (v<=SCALE.q80) return 3;
  if (v<=SCALE.max) return 4;
  return 5;
}
(function(){
  const svg = svgEl("svg",{viewBox:`0 0 760 900`, width:"100%",
    style:"background:#FBFCFD;border:1px solid #DCE3E8;border-radius:7px"});
  MAP.forEach(f=>{
    const p = svgEl("path",{d:f.d, fill:COLORS[bucket(f.days)],
      stroke:"#FFFFFF","stroke-width":0.35});
    p.addEventListener("mousemove", e=>{
      const tip = document.getElementById("tip");
      tip.innerHTML = `<b>Tract ${f.fips}</b><br>${f.name||""}<br>`+
        `Median <b>${fmt2(f.days)}</b> days (${fmt(f.n)} requests)<br>`+
        `% non-white ${fmt1(f.minority)} · income $${fmt(f.income)}<br>`+
        `HOLC grade: <b>${f.holc||"—"}</b>`;
      tip.style.opacity = 1;
      tip.style.left = Math.min(e.clientX+14, window.innerWidth-285)+"px";
      tip.style.top  = (e.clientY+14)+"px";
    });
    p.addEventListener("mouseleave", ()=>{ document.getElementById("tip").style.opacity = 0; });
    svg.appendChild(p);
  });
  document.getElementById("choropleth").appendChild(svg);
  document.getElementById("maplegend").innerHTML =
    `<div style="font-weight:600;margin-bottom:8px">Median days to resolution</div>`+
    [0,1,2,3,4,5].map(i=>{
      const lo = [SCALE.min,SCALE.q20,SCALE.q40,SCALE.q60,SCALE.q80,SCALE.max][i];
      const hi = [SCALE.q20,SCALE.q40,SCALE.q60,SCALE.q80,SCALE.max,null][i];
      const band = i===5 ? `&gt; ${fmt1(SCALE.max)}`
                         : `${fmt1(lo)} – ${fmt1(hi)}`;
      return `<div><span class="sw" style="background:${COLORS[i]}"></span>${band}</div>`;
    }).join("") +
    `<div class="note" style="margin-top:12px">Quintile bands. Each tract is at
    least 20 resolved actionable requests.</div>`;
})();

/* ---------- tables ---------- */
document.getElementById("tbl-holc").innerHTML =
  `<tr><th>HOLC</th><th class="n">Tracts</th><th class="n">Avg tract median (d)</th>
   <th class="n">% non-white</th><th class="n">Income</th></tr>` +
  DATA.grades.map(r=>`<tr><td><b>${r.holc_grade}</b></td>
    <td class="n">${fmt(r.tracts)}</td><td class="n">${fmt2(r.tract_avg_med)}</td>
    <td class="n">${fmt1(r.pct_minority)}%</td>
    <td class="n">$${fmt(r.income)}</td></tr>`).join("");

document.getElementById("tbl-ca").innerHTML =
  `<tr><th>Community area</th><th class="n">Hardship</th><th class="n">Requests</th>
   <th class="n">Median d</th><th class="n">P90 d</th></tr>` +
  DATA.communities.map(r=>`<tr><td>${r.community_area_name}</td>
    <td class="n">${r.hardship_index===null?"—":fmt(r.hardship_index)}</td>
    <td class="n">${fmt(r.n_requests)}</td><td class="n">${fmt2(r.median_days)}</td>
    <td class="n">${fmt1(r.p90_days)}</td></tr>`).join("");

document.getElementById("tbl-types").innerHTML =
  `<tr><th>Request type</th><th>Severity</th><th class="n">Requests</th>
   <th class="n">Median days*</th></tr>` +
  DATA.types.map(r=>`<tr><td>${r.sr_type}</td><td>${r.severity_band}</td>
    <td class="n">${fmt(r.n_requests)}</td>
    <td class="n">${r.median_days===null?"—":fmt2(r.median_days)}</td></tr>`).join("") +
  `<tr><td colspan="4" class="note">* median over actionable requests only</td></tr>`;

/* ---------- quality ---------- */
(function(){
  const q = DATA.quality; if (!q.rows_total) { document.getElementById("quality").textContent="—"; return; }
  const cov = q.enrichment_coverage_pct||{}, di = q.date_integrity||{};
  const row = (a,b)=>`<tr><td>${a}</td><td class="n">${b}</td></tr>`;
  document.getElementById("quality").innerHTML =
    `<table>
      <tr><th>Check</th><th class="n">Value</th></tr>
      ${row("Analytical rows", fmt(q.rows_total))}
      ${row("Months covered", `${q.months_covered} (${q.month_min_max})`)}
      ${row("Assigned a census tract", fmt1(cov.assigned_census_tract)+"%")}
      ${row("Have ACS demographics", fmt1(cov.has_acs_demographics)+"%")}
      ${row("Inside a 1930s HOLC area", fmt1(cov.in_holc_area)+"%")}
      ${row("Matched to daily weather", fmt1(cov.has_weather)+"%")}
      ${row("Reversed timestamps (flagged)", fmt(di.reversed_timestamps))}
    </table>`;
})();

/* ---------- findings ---------- */
(function(){
  const f = DATA.findings||{}; const keys = Object.keys(f);
  if (!keys.length) { document.getElementById("findings").textContent =
    "Run the notebooks to populate this section."; return; }
  document.getElementById("findings").innerHTML = `<table>
    <tr><th>Finding</th><th>Value</th><th>Detail</th><th>Source</th></tr>` +
    keys.map(k=>{
      const v = f[k];
      return `<tr><td><code>${k}</code></td><td class="n">${
        typeof v.value==="number" ? Number(v.value).toLocaleString("en-US",{maximumFractionDigits:4}) : v.value
      }</td><td>${v.note||""}</td><td class="note">${v.source||""}</td></tr>`;
    }).join("") + `</table>`;
})();
</script>
</body>
</html>
"""


def main() -> None:
    print("Stage 5c — Self-contained HTML dashboard\n")
    if not C.DB_PATH.exists():
        print(f"warehouse missing at {C.DB_PATH}; run `python src/etl.py` first")
        sys.exit(1)

    t0 = time.time()
    data = collect()
    print(f"  collected aggregates: {len(data['monthly'])} months, "
          f"{len(data['types'])} request types, {len(data['communities'])} community areas")

    feats, scale = build_map()
    vals = [f["days"] for f in feats]
    scale["median_days"] = round(float(np.median(vals)), 2)
    print(f"  built choropleth: {len(feats)} tract polygons simplified")
    print(f"  tract median days: {scale['median_days']} "
          f"(range {scale['min']:.2f} – {scale['max']:.2f})")

    quality = data.get("quality", {})
    act = quality.get("actionability", {})
    admin_note = (
        f"{act.get('pct_closed_under_1_minute', 0):.1f}% of all records in this dataset were "
        f"opened and closed by the 311 contact centre in under a minute — citizen "
        f"information calls and auto-closed noise reports. They never involve dispatching a "
        f"crew, so counting them makes the citywide median resolution time "
        f"{data['median_all']:.5f} days (about {data['median_all']*1440:.1f} minutes). Every "
        f"response-time statistic in this dashboard is therefore computed over "
        f"{act.get('actionable', 0):,} <b>actionable</b> service requests, giving a median of "
        f"{scale['median_days']} days. The administrative filings are kept for volume and "
        f"seasonality analysis — they are still load on the system."
    )

    subtitle = (f"{quality.get('months_covered', 84)} months · "
                f"{data['totals']['rows_total']:,} requests · "
                f"{data['totals']['n_types']} request types · "
                f"{data['totals']['n_tracts']} census tracts · "
                "sources: City of Chicago Data Portal, NOAA NCEI, US Census ACS 5-year (2023), "
                "University of Richmond DSL (HOLC)")

    html = (HTML
            .replace("__DATA__", json.dumps(data, default=str))
            .replace("__MAP__", json.dumps(feats))
            .replace("__SCALE__", json.dumps(scale))
            .replace("__SUBTITLE__", subtitle)
            .replace("__ADMIN_NOTE__", admin_note))

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\n  {OUT_HTML.relative_to(C.PROJECT_ROOT)} "
          f"({OUT_HTML.stat().st_size / 1e6:.2f} MB) in {time.time() - t0:.1f}s")
    print("  self-contained: open directly in a browser, no server or network needed")


if __name__ == "__main__":
    main()
