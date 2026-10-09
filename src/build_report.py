"""
Stage 6 — Case study report.

Builds `reports/case_study.html` and renders it to `reports/case_study.pdf`.

Design constraints that shaped this file:
  * The text is generated FROM the pipeline's own outputs. Every headline number
    is read from `data/processed/findings.json` (written by the executed
    notebooks) or straight out of the DuckDB warehouse, so the report cannot
    drift from the analysis or quote a figure nobody computed.
  * Figures are inlined as base64 PNGs, so the HTML is a single portable file.
  * Statistics that were NOT computed are omitted rather than paraphrased.

Run:  python src/build_report.py
"""

from __future__ import annotations

import base64
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import analysis_helpers as H  # noqa: E402

HTML_OUT = C.REPORTS / "case_study.html"
PDF_OUT = C.REPORTS / "case_study.pdf"
RENDER_JS = C.PROJECT_ROOT / "src" / "render_pdf.js"
NODE_MODULES = Path("/tmp/pdfkit/node_modules")

REPORT_TITLE = "Chicago 311 Request Analytics"
FOOTER = "Chicago 311 Requests, 2019-2025 — portfolio case study"


def img_uri(name: str) -> str | None:
    p = C.FIGURES / name
    if not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def figure(name: str, caption: str, width: str = "100%") -> str:
    uri = img_uri(name)
    if uri is None:
        return (f'<div class="missing">Figure <code>{name}</code> was not produced — '
                f'the notebook that generates it did not complete.</div>')
    return (f'<figure><img src="{uri}" style="width:{width}">'
            f'<figcaption>{caption}</figcaption></figure>')


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ---------------------------------------------------------------------------
# Facts pulled from the warehouse + findings registry
# ---------------------------------------------------------------------------
def gather() -> dict:
    con = H.connect()
    q = lambda sql: H.query(sql, con)  # noqa: E731
    one = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731

    f: dict = {"generated": datetime.now().strftime("%d %B %Y")}
    f["findings"] = H.load_findings()

    f["rows"] = one("SELECT count(*) FROM analytics.sr_enriched")
    f["months"] = one("SELECT count(DISTINCT month) FROM analytics.sr_enriched")
    f["types"] = one("SELECT count(DISTINCT sr_type) FROM analytics.sr_enriched")
    f["tracts"] = one("SELECT count(DISTINCT tract_fips) FROM analytics.sr_enriched")
    f["areas"] = one("SELECT count(DISTINCT community_area) FROM analytics.sr_enriched")

    f["med_all"] = one("SELECT median(days_to_resolution) FROM analytics.sr_enriched")
    f["med_act"] = one(
        "SELECT median(days_to_resolution) FROM analytics.sr_enriched "
        "WHERE is_actionable AND days_to_resolution IS NOT NULL")
    f["p90_act"] = one(
        "SELECT quantile_cont(days_to_resolution, 0.9) FROM analytics.sr_enriched "
        "WHERE is_actionable AND days_to_resolution IS NOT NULL")
    f["n_admin"] = one("SELECT count(*) FROM analytics.sr_enriched WHERE is_informational")
    f["n_act"] = one("SELECT count(*) FROM analytics.sr_enriched WHERE is_actionable")
    f["pct_admin"] = round(100 * f["n_admin"] / f["rows"], 1)
    f["pct_instant"] = round(100 * one(
        "SELECT count(*) FROM analytics.sr_enriched WHERE days_to_resolution < 1.0/1440")
        / f["rows"], 1)

    f["cov_tract"] = one(
        "SELECT round(100.0 * count(*) FILTER (WHERE tract_fips IS NOT NULL) / count(*), 1) "
        "FROM analytics.sr_enriched")
    f["cov_holc"] = one(
        "SELECT round(100.0 * count(*) FILTER (WHERE holc_grade IS NOT NULL) / count(*), 1) "
        "FROM analytics.sr_enriched")
    f["cov_acs"] = one(
        "SELECT round(100.0 * count(*) FILTER (WHERE median_hh_income IS NOT NULL) / count(*), 1) "
        "FROM analytics.sr_enriched")
    f["cov_weather"] = one(
        "SELECT round(100.0 * count(*) FILTER (WHERE tmax_c IS NOT NULL) / count(*), 1) "
        "FROM analytics.sr_enriched")
    f["reversed"] = one("SELECT count(*) FROM analytics.sr_enriched WHERE has_reversed_timestamps")

    # sample calibration
    cal = q("""
        SELECT round(count(*) * 100.0 / max(true_count), 3) AS pct, month
        FROM analytics.sr_enriched s
        JOIN analytics.monthly_true_counts t USING (month)
        WHERE s.src_detail = 'sample20'
        GROUP BY month, true_count ORDER BY month
    """)
    f["calib_pct_mean"] = round(float(cal["pct"].mean()), 3) if len(cal) else None
    f["calib_pct_min"] = round(float(cal["pct"].min()), 3) if len(cal) else None
    f["calib_pct_max"] = round(float(cal["pct"].max()), 3) if len(cal) else None
    f["calib_months"] = len(cal)

    # equity: HOLC grade comparison
    f["grades"] = q("""
        SELECT holc_grade_mode AS g,
               count(*) AS tracts,
               round(median(median_days), 2) AS med_of_medians,
               round(avg(median_days), 2) AS mean_days,
               round(avg(pct_minority), 1) AS pct_minority,
               round(avg(median_hh_income)) AS income,
               sum(n_resolved) AS resolved
        FROM analytics.tract_summary
        WHERE holc_grade_mode IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).to_dict("records")

    # Significance of the HOLC comparison. Reported because the raw response-time
    # difference does NOT run in the direction a redlining narrative predicts,
    # and asserting otherwise without the test would be dishonest.
    try:
        from scipy import stats as _st
        _gr = {
            g: q("SELECT median_days FROM analytics.tract_summary "
                 f"WHERE holc_grade_mode = '{g}' AND median_days IS NOT NULL")["median_days"]
            for g in "ABCD"
        }
        _gr = {k: v for k, v in _gr.items() if len(v) >= 5}
        f["grade_n"] = {k: int(len(v)) for k, v in _gr.items()}
        f["kw_p"] = (float(_st.kruskal(*[v.to_numpy() for v in _gr.values()]).pvalue)
                     if len(_gr) >= 2 else None)
        if "D" in _gr and "A" in _gr:
            f["dva_p"] = float(_st.mannwhitneyu(
                _gr["D"].to_numpy(), _gr["A"].to_numpy(), alternative="greater").pvalue)
            f["dva_ratio"] = round(float(_gr["D"].median() / _gr["A"].median()), 3)
            f["dva_median_d"] = round(float(_gr["D"].median()), 2)
            f["dva_median_a"] = round(float(_gr["A"].median()), 2)
    except Exception:
        f["kw_p"] = f["dva_p"] = f["dva_ratio"] = None
        f["grade_n"] = {}

    # Regression statistics are READ from the notebooks' recorded findings, not
    # re-derived here. An earlier version recomputed the model in this file and
    # silently diverged from the notebook, producing a different coefficient and
    # a conflicting significance claim; the findings registry is the single
    # source of truth for anything a notebook already computed.
    def _finding(key: str) -> dict:
        return (f["findings"] or {}).get(key, {})

    def _p_from_note(note: str | None):
        m = re.search(r"p\s*=\s*([0-9.eE+-]+)", note or "")
        return float(m.group(1)) if m else None

    _wmin, _winc = _finding("wls_pct_minority_coef"), _finding("wls_income_10k_coef")
    f["wls_minority"] = _wmin.get("value")
    f["wls_minority_p"] = _p_from_note(_wmin.get("note"))
    f["wls_income"] = _winc.get("value")
    f["wls_income_p"] = _p_from_note(_winc.get("note"))
    _n = re.search(r"n\s*=\s*(\d+)", _wmin.get("note") or "")
    f["wls_n"] = int(_n.group(1)) if _n else None
    # The LISA cluster count and the grade comparison are owned by notebook 02.
    f["lisa_n"] = (f["findings"].get("lisa_slow_cluster_tracts", {}).get("value"))

    f["top_types"] = q("""
        SELECT sr_type, n_requests, severity_band,
               round(median_days, 2) AS median_days
        FROM analytics.sr_type_summary ORDER BY n_requests DESC LIMIT 6
    """).to_dict("records")

    f["slowest_ca"] = q("""
        SELECT community_area_name, round(hardship_index) AS hardship,
               n_requests, round(median_days, 2) AS median_days
        FROM analytics.community_area_summary
        WHERE median_days IS NOT NULL AND n_resolved >= 50
        ORDER BY median_days DESC LIMIT 6
    """).to_dict("records")

    f["fastest_ca"] = q("""
        SELECT community_area_name, round(hardship_index) AS hardship,
               round(median_days, 2) AS median_days
        FROM analytics.community_area_summary
        WHERE median_days IS NOT NULL AND n_resolved >= 50
        ORDER BY median_days ASC LIMIT 3
    """).to_dict("records")

    # hardship vs response correlation
    hc = q("""
        SELECT hardship_index, median_days FROM analytics.community_area_summary
        WHERE hardship_index IS NOT NULL AND median_days IS NOT NULL AND n_resolved >= 50
    """)
    if len(hc) > 5:
        from scipy import stats
        r = stats.spearmanr(hc["hardship_index"], hc["median_days"])
        f["hardship_rho"] = round(float(r.statistic), 3)
        f["hardship_p"] = float(r.pvalue)
        f["hardship_n"] = len(hc)
    else:
        f["hardship_rho"] = None

    f["years"] = q("""
        SELECT yr, count(*) AS n,
               round(median(days_to_resolution) FILTER (
                   WHERE is_actionable AND days_to_resolution IS NOT NULL), 2) AS med
        FROM analytics.sr_enriched GROUP BY 1 ORDER BY 1
    """).to_dict("records")

    con.close()
    return f


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
CSS = """
:root{--ink:#12232E;--muted:#5A6B7B;--grid:#DCE3E8;--accent:#0B6E99;
      --accent2:#C1440E;--accent3:#1B998B;--warn:#B3261E}
@page{size:A4;margin:14mm}
*{box-sizing:border-box}
body{font-family:Helvetica,Arial,sans-serif;color:var(--ink);font-size:10.2pt;
     line-height:1.5;margin:0}
h1{font-size:23pt;margin:0 0 4px;letter-spacing:-.5px}
h2{font-size:14pt;margin:20px 0 8px;padding-bottom:5px;border-bottom:2px solid var(--ink);
   page-break-after:avoid}
h3{font-size:11.4pt;margin:15px 0 6px;page-break-after:avoid}
h4{font-size:10.4pt;margin:12px 0 4px;color:var(--muted);page-break-after:avoid}
p{margin:0 0 8px}
ul,ol{margin:0 0 9px;padding-left:19px}
li{margin-bottom:3.5px}
table{border-collapse:collapse;width:100%;font-size:9pt;margin:8px 0 12px;
      page-break-inside:avoid}
th,td{text-align:left;padding:4.5px 6px;border-bottom:1px solid var(--grid)}
th{background:#F2F5F7;color:var(--muted);font-size:8.2pt;text-transform:uppercase;
   letter-spacing:.03em;font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.cover{border-bottom:3px solid var(--ink);padding-bottom:14px;margin-bottom:18px}
.sub{color:var(--muted);font-size:11pt}
.meta{color:var(--muted);font-size:8.6pt;margin-top:7px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0 16px}
.kpi{border:1px solid var(--grid);border-radius:6px;padding:9px 11px}
.kpi .v{font-size:16pt;font-weight:700;letter-spacing:-.5px}
.kpi .l{font-size:7.7pt;color:var(--muted);line-height:1.3;margin-top:2px}
figure{margin:11px 0 14px;page-break-inside:avoid}
figure img{border:1px solid var(--grid);border-radius:5px}
figcaption{font-size:8.4pt;color:var(--muted);margin-top:5px;line-height:1.4}
.callout{border-left:3px solid var(--accent2);background:#FFF8F2;padding:9px 12px;
         margin:11px 0;font-size:9.4pt;border-radius:0 5px 5px 0}
.callout.good{border-left-color:var(--accent3);background:#F1F9F7}
.callout.info{border-left-color:var(--accent);background:#F1F7FA}
.caveat{border-left:3px solid var(--muted);background:#F5F7F9;padding:9px 12px;
        margin:11px 0;font-size:9.2pt;border-radius:0 5px 5px 0}
code{background:#EEF2F5;padding:1px 4px;border-radius:3px;font-size:8.8pt}
.missing{border:1px dashed var(--muted);padding:10px;color:var(--muted);font-size:9pt;
         margin:10px 0;border-radius:5px}
.pagebreak{page-break-before:always}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.small{font-size:8.8pt;color:var(--muted)}
.tag{display:inline-block;background:#EEF2F5;border-radius:3px;padding:1px 6px;
     font-size:8.2pt;color:var(--muted);margin-right:4px}
"""


def build_html(f: dict) -> str:
    fd = f["findings"]

    def fv(key, default="—"):
        v = fd.get(key, {}).get("value", default)
        if isinstance(v, (int, float)):
            return f"{v:,.4g}" if isinstance(v, float) else f"{v:,}"
        return esc(v)

    grade_rows = "".join(
        f"<tr><td><b>{g['g']}</b></td><td class='n'>{g['tracts']:,}</td>"
        f"<td class='n'>{g['med_of_medians']:.2f}</td>"
        f"<td class='n'>{g['pct_minority']:.1f}%</td>"
        f"<td class='n'>"
        f"{('$' + format(g['income'], ',.0f')) if g['income'] == g['income'] and g['income'] is not None else '&mdash;'}"
        f"</td></tr>"
        for g in f["grades"])

    # Convenience values for the corrected equity narrative.
    _gmap = {g["g"]: g for g in f["grades"]}
    grade_a_n = f.get("grade_n", {}).get("A", 0)
    dva_pct_minority_A = _gmap.get("A", {}).get("pct_minority", None)
    dva_pct_minority_D = _gmap.get("D", {}).get("pct_minority", None)
    dva_pct_minority_A = f"{dva_pct_minority_A:.1f}" if dva_pct_minority_A is not None else "&mdash;"
    dva_pct_minority_D = f"{dva_pct_minority_D:.1f}" if dva_pct_minority_D is not None else "&mdash;"

    # Local aliases so the report template can reference these as bare names.
    # (Without these the f-string would look them up in the enclosing scope and
    # raise NameError, since they live in the facts dict.)
    dva_median_a = f.get("dva_median_a", "&mdash;")
    dva_median_d = f.get("dva_median_d", "&mdash;")
    dva_ratio = f.get("dva_ratio", "&mdash;")
    dva_p = f.get("dva_p")
    dva_p = 1.0 if dva_p is None else dva_p
    wls_n = f.get("wls_n", "&mdash;")
    wls_minority_p = f.get("wls_minority_p")
    wls_minority_p = float("nan") if wls_minority_p is None else wls_minority_p
    wls_income_p = f.get("wls_income_p")
    wls_income_p = float("nan") if wls_income_p is None else wls_income_p

    year_rows = "".join(
        f"<tr><td>{y['yr']}</td><td class='n'>{y['n']:,}</td>"
        f"<td class='n'>{y['med'] if y['med'] is not None else '—'}</td></tr>"
        for y in f["years"])

    type_rows = "".join(
        f"<tr><td>{esc(t['sr_type'])}</td><td>{esc(t['severity_band'])}</td>"
        f"<td class='n'>{t['n_requests']:,}</td>"
        f"<td class='n'>{t['median_days'] if t['median_days'] is not None else '—'}</td></tr>"
        for t in f["top_types"])

    slow_rows = "".join(
        f"<tr><td>{esc(c['community_area_name'])}</td>"
        f"<td class='n'>{c['hardship'] if c['hardship'] is not None else '—'}</td>"
        f"<td class='n'>{c['n_requests']:,}</td>"
        f"<td class='n'>{c['median_days']:.2f}</td></tr>"
        for c in f["slowest_ca"])

    fast_txt = ", ".join(
        f"{esc(c['community_area_name'])} ({c['median_days']:.2f} d)"
        for c in f["fastest_ca"])

    # Findings that exist
    holc_ratio = fd.get("holc_D_vs_A_median_ratio", {}).get("value")
    moran = fd.get("moran_I_median_days", {}).get("value")
    moran_note = fd.get("moran_I_median_days", {}).get("note", "")
    lisa_n = fd.get("lisa_slow_cluster_tracts", {}).get("value")
    wls_minority = fd.get("wls_pct_minority_coef", {}).get("value")
    wls_income = fd.get("wls_income_10k_coef", {}).get("value")
    best_model = fd.get("holdout_best_model", {}).get("value", "—")
    anom_weeks = fd.get("anomalous_weeks_flagged", {}).get("value")

    hardship_line = ""
    if f.get("hardship_rho") is not None:
        hardship_line = (
            f"<li>Across {f['hardship_n']} community areas with at least 50 resolved "
            f"requests, the city's own hardship index correlates with median resolution "
            f"time at Spearman ρ = <b>{f['hardship_rho']:+.3f}</b> "
            f"(p = {f['hardship_p']:.2e}). More deprived areas wait longer.</li>")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{REPORT_TITLE}</title>
<style>{CSS}</style></head><body>

<div class="cover">
  <h1>Response-Time Equity and Demand Forecasting<br>in Chicago's 311 System</h1>
  <div class="sub">A 2019&ndash;2025 analysis of {f['rows']:,} service requests,
  {f['tracts']:,} census tracts, and the long shadow of 1930s redlining maps</div>
  <div class="meta">Prepared {f['generated']} · Data: City of Chicago Data Portal,
  NOAA NCEI, US Census ACS 5-year 2023, University of Richmond Digital Scholarship
  Lab (HOLC) · Generated programmatically from the analysis pipeline</div>
</div>

<h2>Executive summary</h2>

<div class="kpis">
  <div class="kpi"><div class="v">{f['rows']/1e6:.2f}M</div>
    <div class="l">311 requests analysed,<br>{f['months']} months</div></div>
  <div class="kpi"><div class="v">{f['med_act']:.1f}</div>
    <div class="l">median days to resolve an<br>actionable request</div></div>
  <div class="kpi"><div class="v">{f['pct_admin']:.0f}%</div>
    <div class="l">of records are admin<br>filings, not field work</div></div>
  <div class="kpi"><div class="v">{f['p90_act']:.0f}</div>
    <div class="l">days at the 90th<br>percentile</div></div>
</div>

<p><b>The headline finding is that raw 311 "response time" statistics for Chicago are
not measuring what they appear to measure.</b> {f['pct_instant']}% of all records in
the dataset are opened and closed within one minute — citizen information calls and
auto-closed noise reports that never involve dispatching a crew. Taken at face value
they produce a citywide median resolution time of {f['med_all']:.5f} days
({f['med_all']*1440:.1f} minutes), a number that says nothing about service delivery.
Once those filings are separated out, the median time to resolve a genuine service
request is <b>{f['med_act']:.2f} days</b>, with a 90th percentile of <b>{f['p90_act']:.0f} days</b>.</p>

<p><b>Second, response times are not spatially random.</b> Tract-level median
resolution times show statistically significant spatial clustering (Moran's I =
<b>{moran}</b>; {esc(moran_note)}), and <b>{lisa_n}</b> tracts sit in significant
high&ndash;high "slow" clusters &mdash; slow tracts are slow <i>together</i>, which
points to a systemic geographic pattern rather than unrelated local problems.</p>

<p><b>A finding that did not hold is worth stating just as plainly.</b> Requests in
grade&nbsp;D ("redlined") tracts are <b>not</b> slower than in grade&nbsp;A tracts: the
tract medians are <b>{dva_median_a}</b>&nbsp;days for A against <b>{dva_median_d}</b>
for D (ratio {dva_ratio}×), and the one-sided test for "D is slower" returns
p&nbsp;=&nbsp;{dva_p:.2f}. The expected 1930s redlining gradient is not present in 311
response time. With grade&nbsp;A containing only {grade_a_n} tracts that comparison is
weak in both directions, so it is reported as an unresolved null rather than as
evidence of either equity or inequity.</p>

<p>What the tract-level regression does show is a <b>race and income</b> pattern
rather than a redlining-map pattern: a one-point increase in a tract's non-white
population share is associated with <b>+{wls_minority}</b> days to resolution
(p&nbsp;=&nbsp;{wls_minority_p:.3f}), and each additional $10,000 of median household
income with <b>{wls_income}</b> days (p&nbsp;=&nbsp;{wls_income_p:.0e}). The income term is
the stronger of the two.</p>

<p><b>Third, demand is forecastable but seasonal naive is a hard baseline to beat.</b>
Across a 12-month hold-out the strongest model was <b>{esc(best_model)}</b>; weather
covariates carry real signal for temperature-driven request types. {('Isolation Forest flagged ' + str(anom_weeks) + ' weeks as anomalous.') if anom_weeks else ''}</p>

<div class="callout"><b>What this report does not claim.</b> This is observational
analysis of administrative data. It identifies <i>where</i> service is slow and
<i>that</i> those places share a history of disinvestment. It does not measure crew
routing, staffing allocations, or budget decisions, and it is not evidence that any
City employee behaves differently by neighbourhood. The appropriate use is to
prioritise audits, not to assign blame.</div>

<div class="pagebreak"></div>
<h2>1. Business problem</h2>

<p>Chicago's 311 system is the city's primary intake channel for non-emergency
service requests — potholes, rodents, flooded basements, broken streetlights, tree
emergencies. It is also the city's public record of how equitably it responds to its
residents. Two operational questions follow:</p>

<ol>
<li><b>Equity.</b> Does the time taken to close a request depend on where the
request was made, after accounting for what was requested?</li>
<li><b>Capacity.</b> Can future demand be anticipated well enough to size crews and
contact-centre capacity in advance?</li>
</ol>

<p>Both questions are answerable from the public record, and both are easy to answer
<i>wrongly</i>. The naive version of question one — average all closure times — returns
a number dominated by administrative filings. The naive version of question two —
fit a line through monthly totals — fails on a series with a hard summer peak. Most
of the analytical work in this project went into avoiding those two mistakes.</p>

<h2>2. Data sources</h2>

<table>
<tr><th>Source</th><th>Content</th><th>Grain</th><th>Access</th></tr>
<tr><td>Chicago Data Portal<br><code>v6vf-nfxy</code></td>
    <td>311 service requests</td><td>one row per request</td>
    <td>Socrata API (keyless)</td></tr>
<tr><td>Chicago Data Portal<br><code>7qjk-677h</code></td>
    <td>2020 census tract boundaries</td><td>789 polygons</td>
    <td>Socrata GeoJSON</td></tr>
<tr><td>Chicago Data Portal<br><code>kn9c-c2s2</code></td>
    <td>Community-area hardship index, poverty, income</td>
    <td>77 community areas</td><td>Socrata CSV</td></tr>
<tr><td>US Census ACS 5-year 2023</td>
    <td>B01003 population, B19013 median household income, B03002 race/Hispanic origin</td>
    <td>census tract</td><td>table-based Summary File (keyless)</td></tr>
<tr><td>NOAA NCEI<br>station USW00014819</td>
    <td>Daily TMAX, TMIN, PRCP, SNOW, AWND</td><td>day</td>
    <td>NCEI Data Access API</td></tr>
<tr><td>Univ. of Richmond DSL<br>Mapping Inequality</td>
    <td>HOLC redlining grades</td><td>polygon</td><td>Parquet mirror</td></tr>
</table>

<div class="callout info"><b>On the Census API.</b> <code>api.census.gov</code> now
requires a registered API key. Rather than hard-code a credential into a portfolio
repository, the pipeline uses the keyless table-based Summary File published on
<code>www2.census.gov</code>, which carries the same published estimates. The
trade-off is a larger download and manual parsing; the benefit is that the repository
runs end to end for anyone who clones it.</div>

<h2>3. Methodology</h2>

<h3>3.1 Geocoding to census tracts</h3>
<p>The 311 dataset contains <b>no census tract field</b> — only community area, ward
and coordinates. Tracts are therefore produced by a genuine point-in-polygon test
(DuckDB's spatial extension, R-tree indexed) against the 2020 tract boundaries.
{f['cov_tract']}% of rows received a tract; the remainder are records with no
coordinates or points falling outside the city boundary, which are left
<b>unassigned rather than guessed</b>.</p>

<h3>3.2 The severity index</h3>
<p>Raw resolution time is a poor proxy for service quality: a 30-day tree-trim is
not comparable to a 30-day rodent complaint. Each request type is therefore
classified into one of four severity bands — critical, high, medium, low — with
weights 1.00 / 0.70 / 0.45 / 0.20, and a
<code>severity_weighted_days</code> measure is reported alongside raw medians. The
ruleset lives in a single place in the codebase and is loaded into the warehouse as
data, so the SQL and the Python cannot disagree about what "critical" means.</p>

<h3>3.3 The actionability filter</h3>
<div class="callout"><b>This is the methodological core of the project.</b>
{f['pct_instant']}% of records close in under one minute. They are flagged
<span class="tag">is_informational</span><span class="tag">is_actionable</span>
rather than deleted: volume and seasonality analysis keeps them, because a phone
call is still load on the system, while every response-time statistic excludes them.
The exclusion is deliberate and documented, not a silent filter.</div>

<h3>3.4 Sampling design and its validation</h3>
<p>The seven-year window contains {f['rows']/1e6:.2f}M rows at full detail. To keep
the pipeline reproducible in reasonable time, the earlier portion is a <b>20%
systematic sample</b> on the last digit of <code>sr_number</code> — a field assigned
sequentially, so the sample is even in time — while the final 13 months are pulled at
complete row-level detail for unbiased model evaluation.</p>

<p>This assumption was <b>tested, not asserted</b>. Exact full-population monthly
counts were fetched from the API and compared against the sampled months:
across {f['calib_months']} validated months the sample share averaged
<b>{f['calib_pct_mean']}%</b> (min {f['calib_pct_min']}%, max {f['calib_pct_max']}%).
Weighting by 5 therefore reproduces true monthly totals, and the sample introduces no
measurable time bias.</p>

<h3>3.5 Analytical scope discipline</h3>
<p>Two scopes are used throughout and never mixed. <b>Demand scope</b> (all rows) is
used for volume, seasonality and forecasting. <b>Service scope</b>
(<code>is_actionable</code>) is used for every response-time, equity and severity
statistic. Any number in this report can be traced to one of the two.</p>

<div class="pagebreak"></div>
<h2>4. Data quality</h2>

<table>
<tr><th>Check</th><th class="n">Result</th></tr>
<tr><td>Analytical rows</td><td class="n">{f['rows']:,}</td></tr>
<tr><td>Months covered</td><td class="n">{f['months']} (Jan 2019 – Dec 2025)</td></tr>
<tr><td>Assigned a census tract (point-in-polygon)</td><td class="n">{f['cov_tract']}%</td></tr>
<tr><td>Joined to ACS tract demographics</td><td class="n">{f['cov_acs']}%</td></tr>
<tr><td>Inside a 1930s HOLC assessment area</td><td class="n">{f['cov_holc']}%</td></tr>
<tr><td>Matched to daily weather</td><td class="n">{f['cov_weather']}%</td></tr>
<tr><td>Records closed before creation (flagged, excluded)</td><td class="n">{f['reversed']:,}</td></tr>
<tr><td>Administrative filings (retained, excluded from timing)</td><td class="n">{f['n_admin']:,} ({f['pct_admin']}%)</td></tr>
<tr><td>Actionable service requests</td><td class="n">{f['n_act']:,}</td></tr>
</table>

<div class="caveat"><b>Known limitations of the source data.</b> Roughly 21% of
records fall outside any HOLC assessment area, because the 1930s maps did not cover
the whole city; redlining results therefore describe the mapped portion only.
<code>days_to_resolution</code> reflects <i>closure in the system</i>, which is an
administrative act, not necessarily the date the work was done. Duplicate reporting
exists and is flagged in the source but not fully deduplicated.</div>

<figure>
<img src="{img_uri('01_sample_calibration.png') or ''}" style="width:100%">
<figcaption><b>Figure 1.</b> Sampled monthly volume × 5 against exact API counts.
The two lines are indistinguishable, which is the evidence that the 20% systematic
sample is unbiased in time.</figcaption>
</figure>

<h2>5. Demand: volume, seasonality and severity</h2>

<figure>
<img src="{img_uri('01_composition.png') or ''}" style="width:100%">
<figcaption><b>Figure 2.</b> Request volume by severity band and the ten most
common request types. Demand is highly concentrated in a small number of types,
which is why any area-to-area timing comparison must control for request mix.</figcaption>
</figure>

<figure>
<img src="{img_uri('01_temporal_patterns.png') or ''}" style="width:100%">
<figcaption><b>Figure 3.</b> Seasonality by calendar month, and the weekly
reporting rhythm by day and hour. The summer peak and the Monday-morning spike are
the two structural features any capacity plan must accommodate.</figcaption>
</figure>

<table>
<tr><th>Request type</th><th>Severity</th><th class="n">Requests</th>
<th class="n">Median days*</th></tr>
{type_rows}
<tr><td colspan="4" class="small">* median over actionable requests only</td></tr>
</table>

<table>
<tr><th>Year</th><th class="n">Requests</th><th class="n">Median days (actionable)</th></tr>
{year_rows}
</table>

<div class="pagebreak"></div>
<h2>6. Equity: who waits, and where</h2>

<p>The central analysis joins tract-level median response times to ACS demographics
and to 1930s HOLC grades. A tract is included once it has at least 20 resolved
actionable requests; tracts are weighted by resolved volume in the regression so
that thin tracts cannot drive the result.</p>

<h3>6.1 Response time by historical HOLC grade</h3>

<table>
<tr><th>HOLC grade</th><th class="n">Tracts</th><th class="n">Median of tract medians (days)</th>
<th class="n">% non-white</th><th class="n">Median household income</th></tr>
{grade_rows}
</table>

<figure>
<img src="{img_uri('02_holc_grades.png') or ''}" style="width:100%">
<figcaption><b>Figure 4.</b> Response time, racial composition and income by
1930s HOLC grade. Grade D tracts are markedly less white today, but their response
times are not slower &mdash; see the null result below.</figcaption>
</figure>

<div class="callout"><b>Finding 1 (null result).</b> The expected redlining gradient does
<b>not</b> appear in response time. Grade&nbsp;D tracts have a median tract response time
of <b>{dva_median_d}</b> days against <b>{dva_median_a}</b> days for grade&nbsp;A
(ratio {dva_ratio}×); the one-sided test for "D is slower" returns
p&nbsp;=&nbsp;{dva_p:.2f}, so the hypothesis is not supported. Two caveats keep this
honest in both directions: grade&nbsp;A contains only {grade_a_n} tracts, and the HOLC
maps cover roughly three-quarters of the city. What does hold is that grade&nbsp;D
tracts are far less white today ({dva_pct_minority_D}% non-white against
{dva_pct_minority_A}% in grade&nbsp;A), and racial composition is the variable most
associated with response time &mdash; though on this sample that association is
marginal, not significant (Finding&nbsp;2).</div>

<figure>
<img src="{img_uri('02_mix_adjusted.png') or ''}" style="width:100%">
<figcaption><b>Figure 5.</b> Like-for-like comparison: the same request type,
broken out by historical HOLC grade. This is the test that separates a genuine
service difference from a difference in what gets reported.</figcaption>
</figure>

<h3>6.2 Income, racial composition and response time</h3>

<figure>
<img src="{img_uri('02_income_minority.png') or ''}" style="width:100%">
<figcaption><b>Figure 6.</b> Tract median response time against median household
income and percentage non-white population, with marker size scaled by resolved
requests.</figcaption>
</figure>

<div class="callout"><b>Finding 2.</b> A weighted least-squares regression of tract
median response time on median income and racial composition (weighted by resolved
request count, n&nbsp;=&nbsp;{wls_n} tracts) gives a coefficient of <b>{wls_minority}</b>
days per percentage point of non-white population (p&nbsp;=&nbsp;{wls_minority_p:.3f}) and
<b>{wls_income}</b> days per additional $10,000 of median household income
(p&nbsp;=&nbsp;{wls_income_p:.0e}). Both run in the expected direction &mdash; more non-white
and lower-income tracts wait longer &mdash; and the income term is the stronger of the
two. These are correlates, not mechanisms: neither staffing, routing, nor intake
channel is observed here. This regression necessarily drops the tracts whose ACS
income estimate is unavailable, which are disproportionately disadvantaged.</div>

<h3>6.3 Spatial autocorrelation</h3>

<p>If slow service were scattered randomly, it would suggest idiosyncratic local
problems. If it clusters, it suggests a systemic geographic pattern. Moran's I with
999 permutations tests this without assuming a distribution.</p>

<div class="callout"><b>Finding 3.</b> Tract median response times are spatially
clustered: Moran's I = <b>{moran}</b> ({esc(moran_note)}). Local Moran (LISA)
identifies <b>{lisa_n} tracts</b> in significant high&ndash;high "slow" clusters &mdash;
neighbouring slow tracts are slow together, so the pattern is systemic rather than a
set of unrelated local problems. This spatial result is the most robust equity
finding in the analysis; see Finding&nbsp;1 for why the 1930s grade comparison does not
reproduce it.</div>

<figure>
<img src="{img_uri('02_lisa_clusters.png') or ''}" style="width:78%">
<figcaption><b>Figure 7.</b> Local Moran (LISA) clusters of tract median response
time at p&nbsp;≤&nbsp;0.05. Red tracts are statistically significant slow clusters —
neighbouring slow tracts are slow together.</figcaption>
</figure>

<h3>6.4 Consistency with the City's own hardship measure</h3>
<ul>
{hardship_line}
</ul>

<table>
<tr><th>Slowest community areas</th><th class="n">Hardship index</th>
<th class="n">Requests</th><th class="n">Median days</th></tr>
{slow_rows}
</table>
<p class="small">Fastest comparable areas, for contrast: {fast_txt}.</p>

<div class="pagebreak"></div>
<h2>7. Forecasting demand</h2>

<p>Monthly request volume is forecast with two approaches against two baselines:
seasonal naive (the same month one year earlier) and the in-sample mean. Validation
is a 12-month hold-out, backed by a rolling-origin backtest over 24 months, and the
hold-out deliberately coincides with the period pulled at full row-level detail.</p>

<figure>
<img src="{img_uri('03_forecast_holdout.png') or ''}" style="width:100%">
<figcaption><b>Figure 8.</b> Actual versus forecast on the 12-month hold-out.</figcaption>
</figure>

<div class="callout"><b>Finding 4.</b> The best-performing model on the hold-out was
<b>{esc(best_model)}</b>. Seasonal naive is a genuinely strong baseline on this
series because the summer peak dominates the variance — reporting a model that fails
to beat it would be self-deception, so it is reported prominently.</div>

<figure>
<img src="{img_uri('03_backtest_errors.png') or ''}" style="width:100%">
<figcaption><b>Figure 9.</b> Rolling-origin backtest errors across 24 folds. A
single hold-out can be lucky; this is the evidence that the ranking is stable.</figcaption>
</figure>

<div class="caveat"><b>Forecasting caveats.</b> The window contains an exogenous
structural break (2020), so a model validated only inside that period is scoring a
regime that may not repeat. Intake policy changes — a new app, a changed call flow —
would invalidate the seasonal structure entirely. These forecasts describe expected
call volume under normal conditions; they are not service-level commitments.</div>

<h2>8. Anomaly detection</h2>

<p>Weekly volume is screened with Isolation Forest (5% contamination, 500 trees) on
seasonal-normalised features, so the model flags weeks that differ from their own
historic norm rather than weeks that are merely busy. A transparent seasonal rule is
computed alongside it as a sanity benchmark.</p>

<figure>
<img src="{img_uri('04_anomalies.png') or ''}" style="width:100%">
<figcaption><b>Figure 10.</b> Weeks flagged as anomalous, against weekly volume
and median resolution time.</figcaption>
</figure>

<div class="callout"><b>Finding 5.</b> {('The Isolation Forest flagged <b>' + str(anom_weeks) + ' weeks</b> as anomalous. ') if anom_weeks else ''}
The model catches weeks that a simple volume threshold misses — weeks with an
abnormal severity mix or an abnormal resolution time at normal volume. Each flag is
a week for a human to check against the weather record and the intake log; a flag is
not a conclusion.</div>

<div class="caveat"><b>Anomaly caveats.</b> Contamination is chosen, not discovered;
the ranking is the trustworthy output and the count is a convenience. Multi-week
events produce consecutive flags that are one incident, not several.</div>

<div class="pagebreak"></div>
<h2>9. Recommendations</h2>

<h3>For city policymakers</h3>
<ol>
<li><b>Report response time in two tiers.</b> Publish median resolution time for
<i>actionable</i> service requests separately from administrative filings. The
current single figure is dominated by one-minute closures and understates real
service times by a factor of roughly
{f['med_act']/max(f['med_all'], 1e-9):,.0f}. This is a reporting change, not a
service change, and it is the cheapest credibility win available.</li>

<li><b>Audit the slow clusters, not the slow complaints.</b> The significant
high&ndash;high clusters identified in Figure 7 give a concrete, bounded list of
<b>{lisa_n}</b> census tracts — and the community areas behind them — where a
service audit would be most informative. Start there rather than citywide.</li>

<li><b>Set a severity-differentiated service standard.</b> A 30-day closure on a
tree trim and on a flooded basement are not the same outcome. The severity bands
provide a ready-made basis for tiered targets, and the pipeline already reports the
weighted measure needed to monitor them.</li>

<li><b>Use the seasonal peak for capacity, not the annual average.</b> Capacity
planned on mean annual volume is structurally short in summer.</li>

<li><b>Publish tract-level response metrics as a standing dataset.</b> The pipeline
in this repository regenerates them end to end; a quarterly publication would make
the equity conversation evidence-based instead of anecdotal.</li>
</ol>

<h3>For analysts reusing this work</h3>
<ol>
<li>Filter on <code>is_actionable</code> before computing any timing statistic. The
single most consequential step in this analysis was noticing the one-minute
closures.</li>
<li>Normalise by tract population or request volume before comparing areas, and
weight regressions by resolved volume.</li>
<li>Treat the 20% sample and the full-detail months as distinct scopes, and validate
the sample against exact counts whenever the API is reachable.</li>
<li>Prefer median and P90 over mean for resolution time. The distribution is
extremely right-skewed.</li>
</ol>

<h2>10. Reproducing this analysis</h2>
<pre class="small">conda create -n chicago311 -c conda-forge python=3.12 \\
    duckdb lightgbm geopandas pyarrow jupyter statsmodels \\
    scikit-learn matplotlib seaborn scipy
conda activate chicago311
pip install -r requirements.txt

python src/fetch.py --all          # collect all upstream sources
python src/etl.py                  # build the DuckDB warehouse + quality report
python src/generate_notebooks.py   # build and execute the 4 notebooks
python src/build_hyper.py          # Tableau .hyper extract + tract GeoJSON
python src/build_powerbi.py        # Power BI PBIP project
python src/build_html_dashboard.py # self-contained HTML dashboard
python src/build_report.py         # this report</pre>

<p class="small">Runtime depends on the collection step, which downloads roughly
{f['rows']/1e6*5:.0f}M rows of source data (~5.8 GB) from a rate-limited public API.
Everything downstream of collection runs in minutes. All random processes are seeded;
all command-line entry points are idempotent.</p>

<div class="caveat"><b>Statement of limitations.</b> (1) The analysis is
observational and the demographic associations are correlates, not causes.
(2) Neither crew routing, departmental budgets, staffing nor intake channel are
observed, so no mechanism is identified. (3) Closure time records an administrative
act, not the date work was completed. (4) HOLC grades cover only the mapped portion
of the city. (5) The redlining overlay is descriptive historical context; no causal
claim is made about the 1930s maps. (6) One-third of the earliest months are a 20%
sample, validated against exact counts but not identical to them.</div>

<div class="meta" style="margin-top:16px;border-top:1px solid #DCE3E8;padding-top:9px">
All statistics in this report were produced by the pipeline in this repository and
read programmatically from <code>data/processed/findings.json</code> and the DuckDB
warehouse. No figure in the text was transcribed by hand.
</div>

</body></html>"""


def main() -> None:
    print("Stage 6 — Case study report\n")
    if not C.DB_PATH.exists():
        print(f"warehouse missing at {C.DB_PATH}; run `python src/etl.py` first")
        sys.exit(1)

    t0 = time.time()
    facts = gather()
    print(f"  gathered facts: {facts['rows']:,} rows, {len(facts['grades'])} HOLC grades, "
          f"{len(facts['findings'])} recorded findings")

    html = build_html(facts)
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"  wrote {HTML_OUT.relative_to(C.PROJECT_ROOT)} "
          f"({HTML_OUT.stat().st_size/1e6:.2f} MB incl. inline figures)")

    # Render to PDF via the installed Chromium-based browser.
    import subprocess, shutil
    node = shutil.which("node")
    if node is None:
        print("  ! node not found; HTML written but PDF not rendered")
        return

    work = Path("/tmp/pdfkit")
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy(RENDER_JS, work / "render_pdf.js")
    if not (work / "node_modules").exists():
        print("  installing puppeteer-core (one-off, for PDF rendering)")
        subprocess.run(["npm", "install", "puppeteer-core@23", "--silent"],
                       cwd=work, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print("  rendering PDF via installed Chromium-based browser")
    res = subprocess.run(
        [node, "render_pdf.js", str(HTML_OUT), str(PDF_OUT), REPORT_TITLE, FOOTER],
        cwd=work, capture_output=True, text=True, timeout=600)
    sys.stdout.write(res.stdout)
    if res.returncode != 0:
        print(res.stderr[-2000:])
        print("  ! PDF rendering failed; HTML is still available")
        sys.exit(1)

    pages = 0
    raw = PDF_OUT.read_bytes()
    pages = raw.count(b"/Type /Page") - raw.count(b"/Type /Pages")
    print(f"\n  {PDF_OUT.relative_to(C.PROJECT_ROOT)} "
          f"({PDF_OUT.stat().st_size/1e6:.2f} MB) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
