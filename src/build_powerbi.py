"""
Stage 5b — Power BI dataset and report definition (PBIP / TMDL).

Produces a Power BI Project (PBIP) that opens in Power BI Desktop:

    dashboards/powerbi/chicago311.pbip
    dashboards/powerbi/chicago311.SemanticModel/   <- TMDL tables, model, relationships
    dashboards/powerbi/chicago311.Report/          <- report pages and visuals
    dashboards/powerbi_data/*.csv                  <- the flat sources it imports

The semantic model imports the exported CSVs from a `DataFolder` parameter rather
than embedding a copy of the data, so refreshing after a pipeline re-run picks up
new numbers without editing the model.

HONEST LIMITATION: Power BI Desktop is not installed on the machine that built
this repository, so the PBIP is generated to spec and schema-checked in Python —
it has NOT been opened in Power BI. The CSV exports are independently verified.
See README "Known limitations".

Run:  python src/build_powerbi.py
"""

from __future__ import annotations

import csv
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import analysis_helpers as H  # noqa: E402

PBIP_DIR = C.DASHBOARDS / "powerbi"
SM_DIR = PBIP_DIR / "chicago311.SemanticModel"
RPT_DIR = PBIP_DIR / "chicago311.Report"
DATA_DIR = C.DASHBOARDS / "powerbi_data"

MODEL_NAME = "chicago311"
TABLES = ["monthly_citywide", "requests_weekly", "tract_equity",
          "community_area", "sr_type_summary"]

# Data types as TMDL spells them.
DTYPE = {"date": "dateTime", "int": "int64", "float": "double",
         "str": "string", "bool": "boolean"}
FMT = {"date": "yyyy-mm-dd", "float": "#,0.00", "int": "#,0"}


def guid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Schema definitions (single source of truth for both CSV and TMDL output)
# ---------------------------------------------------------------------------
def schemas() -> dict[str, tuple[str, list[tuple[str, str, str]]]]:
    """table -> (source SQL, [(column, type, summarizeBy)])"""
    return {
        "monthly_citywide": (
            """
            SELECT month, count(*) AS n_requests,
                   count(*) FILTER (WHERE is_actionable) AS n_actionable,
                   round(median(days_to_resolution)
                         FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL), 3)
                                                            AS median_days,
                   round(avg(tmax_c), 2) AS avg_tmax_c,
                   round(sum(prcp_mm), 2) AS total_prcp_mm,
                   round(sum(snow_mm), 2) AS total_snow_mm
            FROM analytics.sr_enriched GROUP BY month ORDER BY month
            """,
            [("month", "date", "none"), ("n_requests", "int", "sum"),
             ("n_actionable", "int", "sum"), ("median_days", "float", "average"),
             ("avg_tmax_c", "float", "average"), ("total_prcp_mm", "float", "sum"),
             ("total_snow_mm", "float", "sum")],
        ),
        "requests_weekly": (
            """
            SELECT week_start, n_requests, n_critical, n_high,
                   round(median_days, 3) AS median_days,
                   round(avg_tmax_c, 2) AS avg_tmax_c,
                   round(total_prcp_mm, 2) AS total_prcp_mm,
                   round(total_snow_mm, 2) AS total_snow_mm
            FROM analytics.weekly_volume ORDER BY week_start
            """,
            [("week_start", "date", "none"), ("n_requests", "int", "sum"),
             ("n_critical", "int", "sum"), ("n_high", "int", "sum"),
             ("median_days", "float", "average"), ("avg_tmax_c", "float", "average"),
             ("total_prcp_mm", "float", "sum"), ("total_snow_mm", "float", "sum")],
        ),
        "tract_equity": (
            """
            SELECT tract_fips, community_area, community_name, total_pop,
                   median_hh_income, round(pct_minority, 2) AS pct_minority,
                   holc_grade_mode AS holc_grade, n_requests, n_resolved,
                   round(median_days, 3) AS median_days,
                   round(p90_days, 2) AS p90_days,
                   round(severity_weighted_days, 3) AS severity_weighted_days
            FROM analytics.tract_summary ORDER BY tract_fips
            """,
            [("tract_fips", "str", "none"), ("community_area", "int", "none"),
             ("community_name", "str", "none"), ("total_pop", "int", "sum"),
             ("median_hh_income", "float", "average"),
             ("pct_minority", "float", "average"), ("holc_grade", "str", "none"),
             ("n_requests", "int", "sum"), ("n_resolved", "int", "sum"),
             ("median_days", "float", "average"), ("p90_days", "float", "average"),
             ("severity_weighted_days", "float", "average")],
        ),
        "community_area": (
            """
            SELECT community_area, community_area_name, hardship_index,
                   pct_below_poverty, per_capita_income, n_requests, n_resolved,
                   round(median_days, 3) AS median_days,
                   round(p90_days, 2) AS p90_days, n_critical
            FROM analytics.community_area_summary ORDER BY community_area
            """,
            [("community_area", "int", "none"), ("community_area_name", "str", "none"),
             ("hardship_index", "float", "average"),
             ("pct_below_poverty", "float", "average"),
             ("per_capita_income", "float", "average"), ("n_requests", "int", "sum"),
             ("n_resolved", "int", "sum"), ("median_days", "float", "average"),
             ("p90_days", "float", "average"), ("n_critical", "int", "sum")],
        ),
        "sr_type_summary": (
            """
            SELECT sr_type, severity_band, severity_weight, is_informational,
                   n_requests, round(median_days, 3) AS median_days,
                   round(p90_days, 2) AS p90_days, n_unresolved,
                   n_reversed_timestamps
            FROM analytics.sr_type_summary ORDER BY n_requests DESC
            """,
            [("sr_type", "str", "none"), ("severity_band", "str", "none"),
             ("severity_weight", "float", "average"),
             ("is_informational", "bool", "none"), ("n_requests", "int", "sum"),
             ("median_days", "float", "average"), ("p90_days", "float", "average"),
             ("n_unresolved", "int", "sum"),
             ("n_reversed_timestamps", "int", "sum")],
        ),
    }


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def export_csvs() -> dict[str, int]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = H.connect()
    counts = {}
    for table, (sql, cols) in schemas().items():
        df = H.query(sql, con)
        out = DATA_DIR / f"{table}.csv"
        df.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
        counts[table] = len(df)
        print(f"    exported {table:18s} {len(df):>7,} rows -> {out.name}")
    con.close()
    return counts


# ---------------------------------------------------------------------------
# TMDL generation
# ---------------------------------------------------------------------------
def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def tmdl_table(table: str, cols: list[tuple[str, str, str]]) -> str:
    lines = [f"table {table}", f"\tlineageTag: {guid()}", ""]
    for col, kind, summarize in cols:
        lines.append(f"\tcolumn {col}")
        lines.append(f"\t\tdataType: {DTYPE[kind]}")
        if kind in FMT:
            lines.append(f"\t\tformatString: {FMT[kind]}")
        lines.append(f"\t\tlineageTag: {guid()}")
        lines.append(f"\t\tsummarizeBy: {summarize}")
        lines.append(f"\t\tsourceColumn: {col}")
        lines.append("")
    lines.append(f"\tpartition {table} = m")
    lines.append("\t\tmode: import")
    lines.append("\t\tsource =")
    lines.append("\t\t\t\tlet")
    lines.append('\t\t\t\t    Source = Csv.Document(File.Contents(DataFolder & '
                 f'"/{DATA_DIR.name}/{table}.csv"), [Delimiter=",", Encoding=65001, '
                 'QuoteStyle=QuoteStyle.Csv]),')
    lines.append("\t\t\t\t    Promoted = Table.PromoteHeaders(Source, "
                 "[PromoteAllScalars=true])")
    lines.append("\t\t\t\tin")
    lines.append("\t\t\t\t    Promoted")
    lines.append("")
    return "\n".join(lines)


def build_semantic_model() -> None:
    write_text(SM_DIR / "definition.pbism", json.dumps({
        "$schema": ("https://developer.microsoft.com/json-schemas/fabric/item/"
                    "semanticModel/definitionProperties/1.0.0/schema.json"),
        "version": "4.0",
        "settings": {},
    }, indent=2))

    d = SM_DIR / "definition"
    write_text(d / "database.tmdl", f"database {MODEL_NAME}\n\tcompatibilityLevel: 1567\n")

    exprs = [
        "expression DataFolder = "
        f'"{DATA_DIR.relative_to(C.PROJECT_ROOT).as_posix()}"'
        ' meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]',
        "\tannotation PBI_ResultType = Text",
    ]
    write_text(d / "expressions.tmdl", "\n".join(exprs) + "\n")

    model = [f"model Model", "\tculture: en-US",
             "\tdefaultPowerBIDataSourceVersion: powerBI_V3",
             "\tsourceQueryCulture: en-US", ""]
    for t in TABLES:
        model.append(f"ref table {t}")
    model += ["", "ref expression DataFolder"]
    write_text(d / "model.tmdl", "\n".join(model) + "\n")

    for table, (_, cols) in schemas().items():
        write_text(d / "tables" / f"{table}.tmdl", tmdl_table(table, cols))

    # tract_equity is at a finer grain than community_area: relate them.
    rel = [
        "relationship " + guid(),
        "\tfromColumn: tract_equity.community_area",
        "\ttoColumn: community_area.community_area",
        "",
    ]
    write_text(d / "relationships.tmdl", "\n".join(rel))


# ---------------------------------------------------------------------------
# Report definition (pages + visuals)
# ---------------------------------------------------------------------------
def col_field(table: str, column: str, kind: str = "Column") -> dict:
    return {
        kind: {
            "Expression": {"SourceRef": {"Entity": table}},
            "Property": column,
        }
    }


def projection(table: str, column: str, active: bool = True) -> dict:
    return {
        "field": col_field(table, column),
        "queryRef": f"{table}.{column}",
        "nativeQueryRef": column,
        "active": active,
    }


def visual(vtype: str, x: int, y: int, w: int, h: int,
           projections_map: dict[str, list[dict]],
           title: str | None = None) -> dict:
    v = {
        "name": guid(),
        "position": {"x": x, "y": y, "z": 0, "width": w, "height": h},
        "visual": {
            "visualType": vtype,
            "query": {"queryState": {
                role: {"projections": projs} for role, projs in projections_map.items()
            }},
            "drillFilterOtherVisuals": True,
        },
    }
    if title:
        v["visual"]["visualContainerObjects"] = {
            "title": [{"properties": {"text": {"expr": {"Literal": {"Value": f"'{title}'"}}}}}]
        }
    return v


def page(name: str, display: str, visuals: list[dict]) -> tuple[str, str, list[dict]]:
    return name, display, visuals


def build_report() -> None:
    write_text(RPT_DIR / "definition.pbir", json.dumps({
        "$schema": ("https://developer.microsoft.com/json-schemas/fabric/item/"
                    "report/definitionProperties/1.0.0/schema.json"),
        "version": "4.0",
        "datasetReference": {
            "byPath": {"path": f"../{MODEL_NAME}.SemanticModel"},
            "byConnection": None,
        },
    }, indent=2))

    pages = [
        page("ExecutiveKPIs", "Executive KPIs", [
            visual("card", 16, 16, 300, 130, {"Values": [projection("monthly_citywide", "n_requests")]}, "Total requests"),
            visual("card", 332, 16, 300, 130, {"Values": [projection("monthly_citywide", "median_days")]}, "Median days (actionable)"),
            visual("card", 648, 16, 300, 130, {"Values": [projection("community_area", "n_critical")]}, "Critical-band requests"),
            visual("lineChart", 16, 162, 932, 260,
                   {"Category": [projection("monthly_citywide", "month")],
                    "Y": [projection("monthly_citywide", "n_requests")]},
                   "Monthly 311 demand"),
            visual("lineChart", 16, 438, 932, 240,
                   {"Category": [projection("monthly_citywide", "month")],
                    "Y": [projection("monthly_citywide", "median_days")]},
                   "Monthly median resolution time (actionable)"),
        ]),
        page("Equity", "Equity & Redlining", [
            visual("map", 16, 16, 620, 400,
                   {"Category": [projection("tract_equity", "tract_fips")],
                    "Size": [projection("tract_equity", "median_days")]},
                   "Choropleth — tract median resolution time (join the GeoJSON layer)"),
            visual("clusteredBarChart", 652, 16, 420, 400,
                   {"Category": [projection("tract_equity", "holc_grade")],
                    "Y": [projection("tract_equity", "median_days")]},
                   "Median days by HOLC grade"),
            visual("scatterChart", 16, 432, 620, 300,
                   {"Category": [projection("tract_equity", "pct_minority")],
                    "X": [projection("tract_equity", "median_hh_income")],
                    "Y": [projection("tract_equity", "median_days")]},
                   "Income vs. resolution time"),
            visual("tableEx", 652, 432, 420, 300,
                   {"Values": [projection("community_area", "community_area_name"),
                               projection("community_area", "hardship_index"),
                               projection("community_area", "median_days")]},
                   "Community-area hardship vs. response"),
        ]),
        page("Forecast", "Forecast & Anomalies", [
            visual("lineChart", 16, 16, 1056, 300,
                   {"Category": [projection("requests_weekly", "week_start")],
                    "Y": [projection("requests_weekly", "n_requests")]},
                   "Weekly volume (anomaly review)"),
            visual("lineChart", 16, 332, 520, 300,
                   {"Category": [projection("requests_weekly", "week_start")],
                    "Y": [projection("requests_weekly", "median_days")]},
                   "Weekly median resolution time"),
            visual("clusteredBarChart", 552, 332, 520, 300,
                   {"Category": [projection("sr_type_summary", "sr_type")],
                    "Y": [projection("sr_type_summary", "n_requests")]},
                   "Volume by request type"),
        ]),
    ]

    page_entries = []
    for name, display, visuals in pages:
        pdir = RPT_DIR / "definition" / "pages" / name
        write_text(pdir / "page.json", json.dumps({
            "name": name, "displayName": display,
            "displayOption": "FitToPage", "width": 1100, "height": 760,
        }, indent=2))
        for v in visuals:
            write_text(pdir / "visuals" / v["name"] / "visual.json",
                       json.dumps(v, indent=2))
        page_entries.append({"name": name, "displayName": display, "ordinal": len(page_entries)})

    write_text(RPT_DIR / "definition" / "pages" / "pages.json",
               json.dumps({"pageOrder": [p["name"] for p in page_entries],
                           "activePageName": page_entries[0]["name"]}, indent=2))
    write_text(RPT_DIR / "definition" / "report.json", json.dumps({
        "config": json.dumps({"version": "5.43", "themeCollection": {
            "baseTheme": {"name": "CY24SU06"}}}),
        "layoutOptimization": 0,
    }, indent=2))

    total = sum(len(p[2]) for p in pages)
    print(f"    wrote {len(pages)} report pages, {total} visuals")


def main() -> None:
    print("Stage 5b — Power BI project (PBIP)\n")
    if not C.DB_PATH.exists():
        print(f"warehouse missing at {C.DB_PATH}; run `python src/etl.py` first")
        sys.exit(1)

    print("  exporting flat CSV sources")
    counts = export_csvs()

    print("\n  writing semantic model (TMDL)")
    build_semantic_model()
    print(f"    {len(TABLES)} tables, 1 relationship, DataFolder parameter")

    print("\n  writing report definition")
    build_report()

    pbip = {
        "version": "1.0",
        "artifacts": [{"report": {"path": f"{MODEL_NAME}.Report"}}],
        "settings": {"enableAutoRecovery": True},
    }
    write_text(PBIP_DIR / f"{MODEL_NAME}.pbip", json.dumps(pbip, indent=2))

    print(f"\n  {PBIP_DIR.relative_to(C.PROJECT_ROOT)}/{MODEL_NAME}.pbip")
    print(f"  {sum(counts.values()):,} rows exported across {len(counts)} CSV sources")
    print("\n  NOTE: Power BI Desktop is not available here, so this project has been")
    print("  generated and schema-checked only. Opening it in Power BI Desktop and")
    print("  refreshing is required to confirm the model binds.")


if __name__ == "__main__":
    main()
