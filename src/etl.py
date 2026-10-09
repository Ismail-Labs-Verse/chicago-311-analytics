"""
Stage 2 — Data engineering.

Runs the DuckDB pipeline in sql/01_ingest.sql and then does the things SQL
alone should not do:

  1. materialises the severity ruleset from src/config.py into DuckDB,
  2. pulls exact full-population monthly counts from the Socrata API so the
     20% sample can be validated against ground truth,
  3. writes a data-quality report (JSON + markdown) for review,
  4. exports the small representative sample that is committed to the repo.

Run:  python src/etl.py                 # full pipeline
      python src/etl.py --skip-truth    # skip the API calibration step
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import fetch  # noqa: E402

SQL_FILE = C.SQL_DIR / "01_ingest.sql"
QUALITY_JSON = C.DATA_PROCESSED / "data_quality_report.json"
QUALITY_MD = C.REPORTS / "data_quality_report.md"
SAMPLE_ROWS = 1000


def banner(msg: str) -> None:
    print(f"\n{'=' * 74}\n{msg}\n{'=' * 74}", flush=True)


# ---------------------------------------------------------------------------
# 1. Severity rules -> DuckDB
# ---------------------------------------------------------------------------
def load_severity_map(con: duckdb.DuckDBPyConnection) -> int:
    rows = C.severity_rows()
    con.execute("CREATE SCHEMA IF NOT EXISTS severity_map_stage")
    con.execute(
        "CREATE OR REPLACE TABLE severity_map_stage.r(keyword VARCHAR, band VARCHAR, "
        "weight DOUBLE, priority INTEGER)"
    )
    con.executemany("INSERT INTO severity_map_stage.r VALUES (?, ?, ?, ?)", rows)

    con.execute("CREATE SCHEMA IF NOT EXISTS patterns_stage")
    con.execute("CREATE OR REPLACE TABLE patterns_stage.p(pattern VARCHAR)")
    con.executemany("INSERT INTO patterns_stage.p VALUES (?)", C.informational_patterns())

    con.execute("CREATE SCHEMA IF NOT EXISTS params_stage")
    con.execute("CREATE OR REPLACE TABLE params_stage.k(key VARCHAR, value DOUBLE)")
    con.executemany("INSERT INTO params_stage.k VALUES (?, ?)", C.pipeline_params())
    return len(rows)


# ---------------------------------------------------------------------------
# 2. Exact monthly counts from the API (sample calibration ground truth)
# ---------------------------------------------------------------------------
def fetch_true_counts(con: duckdb.DuckDBPyConnection) -> int:
    """Full-population monthly counts, bypassing the 20% sample predicate."""
    url = (f"{C.CHI311_RESOURCE}.json?"
           + "&".join([
               "$select=date_trunc_ym(created_date)%20as%20m,count(1)%20as%20n",
               "$where=" + "created_date%3E%3D%272019-01-01T00%3A00%3A00%27"
                          "%20AND%20created_date%3C%272026-01-01T00%3A00%3A00%27",
               "$group=date_trunc_ym(created_date)",
               "$order=date_trunc_ym(created_date)",
               "$limit=200",
           ]))
    print(f"  querying {C.CHI311_RESOURCE} (aggregate, no row transfer)")
    body = fetch.http_get(url, timeout=300, label="true counts")
    data = json.loads(body)
    rows = [(r["m"][:10], int(r["n"])) for r in data]
    con.executemany("INSERT INTO analytics.monthly_true_counts VALUES (?, ?)", rows)
    print(f"  stored {len(rows)} monthly ground-truth counts")
    return len(rows)


# ---------------------------------------------------------------------------
# 3. Data-quality report
# ---------------------------------------------------------------------------
def _r(value, digits: int = 3):
    """Round tolerantly: aggregates over an empty/filtered set return None."""
    return None if value is None else round(value, digits)


def quality_report(con: duckdb.DuckDBPyConnection) -> dict:
    q = lambda sql: con.execute(sql).fetchall()  # noqa: E731
    one = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731

    total = one("SELECT count(*) FROM analytics.sr_enriched")
    report: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "rows_total": total}

    # --- ingest split ---
    report["rows_by_sampling"] = dict(
        q("SELECT src_detail, count(*) FROM analytics.sr_enriched GROUP BY 1 ORDER BY 1")
    )
    report["months_covered"] = one(
        "SELECT count(DISTINCT month) FROM analytics.sr_enriched")
    report["month_min_max"] = one(
        "SELECT min(month) || ' .. ' || max(month) FROM analytics.sr_enriched")

    # --- date integrity ---
    report["date_integrity"] = {
        "created_ts_null": one("SELECT count(*) FROM analytics.sr_enriched WHERE created_ts IS NULL"),
        "closed_ts_null": one("SELECT count(*) FROM analytics.sr_enriched WHERE closed_ts IS NULL"),
        "reversed_timestamps": one("SELECT count(*) FROM analytics.sr_enriched WHERE has_reversed_timestamps"),
        "negative_days": one("SELECT count(*) FROM analytics.sr_enriched WHERE days_to_resolution < 0"),
        "pct_unresolved": round(100.0 * one(
            "SELECT count(*) FROM analytics.sr_enriched WHERE days_to_resolution IS NULL") / total, 3),
    }

    # --- actionability: which rows can carry a response-time statistic ---
    report["actionability"] = {
        "rows_total": total,
        "informational_or_auto_closed": one(
            "SELECT count(*) FROM analytics.sr_enriched WHERE is_informational"),
        "service_requests": one(
            "SELECT count(*) FROM analytics.sr_enriched WHERE is_service_request"),
        "actionable": one(
            "SELECT count(*) FROM analytics.sr_enriched WHERE is_actionable"),
        "closed_under_1_minute": one(
            "SELECT count(*) FROM analytics.sr_enriched WHERE days_to_resolution < 0.0007"),
        "pct_closed_under_1_minute": round(100.0 * one(
            "SELECT count(*) FROM analytics.sr_enriched WHERE days_to_resolution < 0.0007") / total, 2),
        "note": ("Administratively closed filings (information calls, aircraft-noise "
                 "reports) are excluded from response-time statistics because they "
                 "never involve dispatching a crew; they are retained for demand "
                 "and seasonality analysis."),
    }

    # --- enrichment coverage ---
    report["enrichment_coverage_pct"] = {
        "has_coordinates": round(100.0 * one(
            "SELECT count(*) FROM analytics.sr_enriched WHERE longitude IS NOT NULL AND latitude IS NOT NULL") / total, 2),
        "assigned_census_tract": round(100.0 * one(
            "SELECT count(*) FROM analytics.sr_enriched WHERE tract_fips IS NOT NULL") / total, 2),
        "has_acs_demographics": round(100.0 * one(
            "SELECT count(*) FROM analytics.sr_enriched WHERE median_hh_income IS NOT NULL") / total, 2),
        "in_holc_area": round(100.0 * one(
            "SELECT count(*) FROM analytics.sr_enriched WHERE holc_grade IS NOT NULL") / total, 2),
        "has_weather": round(100.0 * one(
            "SELECT count(*) FROM analytics.sr_enriched WHERE tmax_c IS NOT NULL") / total, 2),
        "has_community_area_socioeconomic": round(100.0 * one(
            "SELECT count(*) FROM analytics.sr_enriched WHERE hardship_index IS NOT NULL") / total, 2),
    }

    # --- distributional profile ---
    report["severity_bands"] = dict(q(
        "SELECT severity_band, count(*) FROM analytics.sr_enriched GROUP BY 1 ORDER BY 2 DESC"))
    report["holc_grades"] = dict(q(
        "SELECT holc_grade, count(*) FROM analytics.sr_enriched WHERE holc_grade IS NOT NULL GROUP BY 1 ORDER BY 1"))
    report["top_request_types"] = [
        {"sr_type": t, "n": n} for t, n in q(
            "SELECT sr_type, count(*) FROM analytics.sr_enriched GROUP BY 1 ORDER BY 2 DESC LIMIT 12")
    ]
    report["response_time_days"] = {
        "scope": "actionable service requests only",
        "median": _r(one("SELECT median(days_to_resolution) FROM analytics.sr_enriched "
                         "WHERE is_actionable AND days_to_resolution IS NOT NULL"), 4),
        "mean": _r(one("SELECT avg(days_to_resolution) FROM analytics.sr_enriched "
                       "WHERE is_actionable AND days_to_resolution IS NOT NULL")),
        "p90": _r(one("SELECT quantile_cont(days_to_resolution, 0.9) FROM analytics.sr_enriched "
                      "WHERE is_actionable AND days_to_resolution IS NOT NULL")),
        "max": _r(one("SELECT max(days_to_resolution) FROM analytics.sr_enriched "
                      "WHERE is_actionable AND days_to_resolution IS NOT NULL"), 1),
        "median_all_rows_incl_admin": _r(one(
            "SELECT median(days_to_resolution) FROM analytics.sr_enriched"), 6),
    }

    # A guard: the whole equity analysis is meaningless if this collapses.
    n_act = report["actionability"]["actionable"]
    if total and n_act / total < 0.10:
        raise RuntimeError(
            f"only {n_act:,} of {total:,} rows are actionable — the actionability "
            f"rule is mis-configured (check that pipeline_params is populated "
            f"BEFORE analytics.sr_enriched is built)."
        )

    # --- sample calibration: 20% sample months x5 should match exact truth ---
    calib = q(f"""
        WITH s AS (
            SELECT month, count(*) AS sample_n
            FROM analytics.sr_enriched
            WHERE src_detail = 'sample20'
            GROUP BY month
        )
        SELECT s.month, s.sample_n, t.true_count,
               round(s.sample_n * 100.0 / t.true_count, 2) AS sample_pct,
               round(t.true_count / 5.0, 0)                AS expected_n
        FROM s JOIN analytics.monthly_true_counts t USING (month)
        ORDER BY s.month
    """)
    if calib:
        pcts = [r[3] for r in calib if r[3] is not None]
        report["sample_calibration"] = {
            "months_checked": len(calib),
            "mean_sample_pct": round(sum(pcts) / len(pcts), 3),
            "min_sample_pct": min(pcts),
            "max_sample_pct": max(pcts),
            "note": ("A 20% systematic sample of sr_number should land near 20.00% "
                     "for every month; deviation only from sampling noise."),
            "sample_months": [
                {"month": str(r[0]), "sample_n": r[1], "true_count": r[2],
                 "sample_pct": r[3], "expected_n": r[4]} for r in calib
            ],
        }

    report["aggregate_tables"] = {
        t: one(f"SELECT count(*) FROM analytics.{t}")
        for t in ["monthly_demand", "weekly_volume", "tract_summary",
                  "community_area_summary", "sr_type_summary"]
    }
    return report


def write_quality_markdown(rep: dict) -> None:
    def tbl(d: dict, k: str = "value") -> str:
        return "\n".join(f"| {a} | {b:,} |" if isinstance(b, (int, float))
                         else f"| {a} | {b} |" for a, b in d.items())

    lines = [
        "# Data Quality Report — Chicago 311 Analytics",
        "",
        f"Generated: `{rep['generated_at']}`  ",
        f"Analytical rows: **{rep['rows_total']:,}**  ",
        f"Months covered: **{rep['months_covered']}** ({rep['month_min_max']})",
        "",
        "## 1. Sampling design (as ingested)",
        "",
        "| Sampling mode | Rows |",
        "|---|---:|",
        tbl(rep["rows_by_sampling"]),
        "",
        "## 2. Date integrity",
        "",
        "| Check | Count |",
        "|---|---:|",
        tbl(rep["date_integrity"]),
        "",
        "`reversed_timestamps` are records closed before they were created. They are",
        "flagged, not silently repaired, and are excluded from `days_to_resolution`.",
        "",
        "## 2b. Actionability filter (response-time scope)",
        "",
        "A large share of rows are logged and closed administratively within seconds",
        "(citizen information calls, aircraft-noise reports). They never involve",
        "dispatching a crew, so leaving them in collapses the citywide median to ~0",
        "and destroys any equity signal. They are flagged, not deleted.",
        "",
        "| Measure | Rows |",
        "|---|---:|",
        "\n".join(f"| {k} | {v:,} |" if isinstance(v, (int, float)) else f"| {k} | {v} |"
                  for k, v in rep["actionability"].items() if k != "note"),
        "",
        "## 3. Enrichment coverage (% of all rows)",
        "",
        "| Enrichment | % |",
        "|---|---:|",
        tbl(rep["enrichment_coverage_pct"]),
        "",
        "## 4. Response time (days)",
        "",
        f"Scope: **{rep['response_time_days'].get('scope', 'all rows')}**",
        "",
        "| Statistic | Days |",
        "|---|---:|",
        tbl({k: v for k, v in rep["response_time_days"].items() if k != "scope"}),
        "",
        "## 5. Severity bands",
        "",
        "| Band | Requests |",
        "|---|---:|",
        tbl(rep["severity_bands"]),
        "",
        "## 6. HOLC redlining grades",
        "",
        "| Grade | Requests |",
        "|---|---:|",
        tbl(rep["holc_grades"]),
        "",
        "## 7. Sample calibration against exact full-population counts",
        "",
    ]
    cal = rep.get("sample_calibration")
    if cal:
        lines += [
            f"Checked **{cal['months_checked']}** sampled months against exact API counts.",
            f"Mean sample share **{cal['mean_sample_pct']}%** "
            f"(min {cal['min_sample_pct']}%, max {cal['max_sample_pct']}%).",
            "",
            "| Month | Sample n | True n | Sample % | Expected n |",
            "|---|---:|---:|---:|---:|",
        ]
        for m in cal["sample_months"]:
            lines.append(f"| {m['month']} | {m['sample_n']:,} | {m['true_count']:,} | "
                         f"{m['sample_pct']}% | {m['expected_n']:,.0f} |")
    else:
        lines.append("_Not run (`--skip-truth`)._")

    lines += ["", "## 8. Top request types", "", "| Request type | Requests |", "|---|---:|"]
    for t in rep["top_request_types"]:
        lines.append(f"| {t['sr_type']} | {t['n']:,} |")

    lines += ["", "## 9. Aggregate tables written", "", "| Table | Rows |", "|---|---:|",
              tbl(rep["aggregate_tables"]), ""]

    QUALITY_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 4. Committed representative sample
# ---------------------------------------------------------------------------
def export_sample(con: duckdb.DuckDBPyConnection) -> Path:
    """A small, representative slice of the CLEANED data for the public repo.

    Stratified by year so it is not accidentally a single-season snapshot.
    """
    out = C.DATA_SAMPLES / "sample_cleaned_311.csv"
    con.execute(f"""
        COPY (
            WITH ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY yr ORDER BY random()) AS rn
                FROM analytics.sr_enriched
                WHERE created_ts IS NOT NULL
            )
            SELECT * EXCLUDE (rn) FROM ranked
            WHERE rn <= {SAMPLE_ROWS}
            ORDER BY created_ts
        ) TO '{out}' (HEADER, DELIMITER ',')
    """)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def strip_sql_comments(sql: str) -> str:
    """Remove `--` line comments, respecting single-quoted string literals.

    Needed because the pipeline is split into statements on `;`, and a naive
    split would otherwise cut the file apart inside prose comments such as
    "... 2024-12..2025-12 are complete detail; ...".
    """
    lines = []
    for line in sql.split("\n"):
        keep, in_str = [], False
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "'":
                in_str = not in_str
                keep.append(ch)
            elif ch == "-" and not in_str and i + 1 < len(line) and line[i + 1] == "-":
                break
            else:
                keep.append(ch)
            i += 1
        lines.append("".join(keep))
    return "\n".join(lines)


def sql_statements(sql: str) -> list[str]:
    """Split a script into executable statements, rejecting a silent killer.

    In SQL a `--` comment ends at the newline. A comment continued onto the
    next line therefore leaks prose into the parser, and the resulting error
    ("syntax error at or near ...") points at a comment, which is confusing.
    This is checked explicitly so the failure names the real cause.
    """
    lines = sql.split("\n")
    for i, line in enumerate(lines[:-1]):
        s = line.strip()
        if not s.startswith("--"):
            continue
        nxt = lines[i + 1].strip()
        # A continuation of the comment, not a new comment and not real SQL.
        if nxt and not nxt.startswith("--") and not _looks_like_sql(nxt):
            raise ValueError(
                f"sql/01_ingest.sql line {i + 1}: '--' comment continues onto "
                f"line {i + 2} ({nxt[:60]!r}). SQL comments do not span lines — "
                f"prefix that line with '--' too."
            )
    return [s.strip() for s in strip_sql_comments(sql).split(";") if s.strip()]


_SQL_STARTS = ("select", "with", "create", "insert", "drop", "alter", "update",
               "delete", "install", "load", "pragma", "copy", "set", "call",
               "(", "--")


def _looks_like_sql(line: str) -> bool:
    """Heuristic: does this line plausibly begin a SQL statement/clause?"""
    low = line.lower()
    if low.startswith(_SQL_STARTS):
        return True
    # Continuation of a statement (indented clause, closing paren, etc.)
    return line[0] in (")", "]") or line.endswith((",", ")", ";"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Chicago 311 ETL")
    ap.add_argument("--skip-truth", action="store_true",
                    help="skip the API ground-truth calibration step")
    args = ap.parse_args()

    t0 = time.time()
    banner("Stage 2 — Data engineering (DuckDB)")

    n_files = len(list((C.DATA_RAW / "311").glob("*.csv")))
    print(f"input files: {n_files} monthly 311 CSVs")
    if n_files == 0:
        print("no 311 CSVs found — run `python src/fetch.py --req311` first")
        sys.exit(1)

    con = duckdb.connect(str(C.DB_PATH))
    con.execute("PRAGMA threads=8")
    if C.DB_PATH.exists():
        print(f"database: {C.DB_PATH}")

    banner("Loading severity ruleset from src/config.py")
    n = load_severity_map(con)
    print(f"  {n} keyword rules across {len(C.SEVERITY_BANDS)} bands")

    banner("Running sql/01_ingest.sql")
    sql = SQL_FILE.read_text(encoding="utf-8")
    # The file is pure DDL/DML; the parameter TABLES it declares are populated
    # from src/config.py the moment they exist. Ordering matters: `sr_enriched`
    # reads pipeline_params to decide actionability, so if the insert happened
    # after the whole script the comparison would be against an empty table.
    parameter_tables = {"severity_map": "severity_map_stage.r",
                        "informational_patterns": "patterns_stage.p",
                        "pipeline_params": "params_stage.k"}
    created: set[str] = set()
    loaded = False
    statements = sql_statements(sql)
    print(f"  {len(statements)} statements")
    for i, stmt in enumerate(statements, 1):
        head = " ".join(stmt.split())[:72]
        print(f"  [{i:02d}/{len(statements)}] {head}", flush=True)
        con.execute(stmt)
        for tbl in parameter_tables:
            if tbl in stmt and "CREATE" in stmt.upper():
                created.add(tbl)
        if not loaded and created == set(parameter_tables):
            con.execute("INSERT INTO severity_map SELECT keyword, band, weight, "
                        "priority FROM severity_map_stage.r")
            con.execute("INSERT INTO informational_patterns "
                        "SELECT pattern FROM patterns_stage.p")
            con.execute("INSERT INTO pipeline_params SELECT key, value "
                        "FROM params_stage.k")
            loaded = True
            print(f"  populated parameter tables from src/config.py: "
                  f"{con.execute('SELECT count(*) FROM severity_map').fetchone()[0]} "
                  f"severity rules, "
                  f"{con.execute('SELECT count(*) FROM informational_patterns').fetchone()[0]} "
                  f"informational patterns, "
                  f"{con.execute('SELECT count(*) FROM pipeline_params').fetchone()[0]} "
                  f"params", flush=True)
    if not loaded:
        raise RuntimeError("parameter tables were never populated — the severity / "
                           "actionability rules would silently not apply")

    if not args.skip_truth:
        banner("Exact full-population monthly counts (sample calibration)")
        try:
            fetch_true_counts(con)
        except Exception as exc:
            print(f"  ! ground-truth pull failed ({exc}); calibration will be skipped")

    banner("Data quality report")
    rep = quality_report(con)
    QUALITY_JSON.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    write_quality_markdown(rep)
    print(json.dumps({k: rep[k] for k in
                      ["rows_total", "months_covered", "month_min_max",
                       "rows_by_sampling", "date_integrity", "actionability",
                       "enrichment_coverage_pct", "response_time_days"]},
                     indent=2, default=str))
    print(f"\n  wrote {QUALITY_JSON.relative_to(C.PROJECT_ROOT)}")
    print(f"  wrote {QUALITY_MD.relative_to(C.PROJECT_ROOT)}")

    banner("Committed representative sample")
    p = export_sample(con)
    print(f"  wrote {p.relative_to(C.PROJECT_ROOT)} "
          f"({p.stat().st_size / 1e3:.0f} KB)")

    con.close()
    print(f"\nETL complete in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
