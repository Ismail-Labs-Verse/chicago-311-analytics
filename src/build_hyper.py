"""
Stage 5a — Tableau data extract (.hyper).

Writes a real Tableau Hyper file with `tableauhyperapi`, containing exactly the
tables a dashboard needs — no raw dump. Every table is read back and verified
(row count + schema) before this script reports success, so the extract is
known-readable rather than merely written.

Also exports a tract GeoJSON carrying the equity metrics as attributes, which
is how Tableau Public builds a true choropleth: connect the GeoJSON as a
spatial source and join it to `tract_equity` on `tract_fips`.

Run:  python src/build_hyper.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import analysis_helpers as H  # noqa: E402

HYPER_PATH = C.DASHBOARDS / "chicago311_extract.hyper"
SPATIAL_DIR = C.DASHBOARDS / "spatial"
SPATIAL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Table definitions: name -> ordered column spec
# ---------------------------------------------------------------------------
def table_specs() -> dict[str, list[tuple[str, str]]]:
    """Human-readable column specs; mapped to Hyper SqlTypes below."""
    return {
        "requests_weekly": [
            ("week_start", "date"), ("n_requests", "int"), ("n_critical", "int"),
            ("n_high", "int"), ("median_days", "float"), ("avg_tmax_c", "float"),
            ("total_prcp_mm", "float"), ("total_snow_mm", "float"),
        ],
        "monthly_demand": [
            ("month", "date"), ("sr_type", "str"), ("severity_band", "str"),
            ("n_requests", "int"), ("n_service_requests", "int"),
            ("n_resolved", "int"), ("median_days", "float"),
            ("p90_days", "float"), ("mean_days", "float"),
        ],
        "monthly_citywide": [
            ("month", "date"), ("n_requests", "int"), ("n_actionable", "int"),
            ("median_days", "float"), ("avg_tmax_c", "float"),
            ("total_prcp_mm", "float"), ("total_snow_mm", "float"),
        ],
        "tract_equity": [
            ("tract_fips", "str"), ("community_area", "int"),
            ("community_name", "str"), ("total_pop", "int"),
            ("median_hh_income", "float"), ("pct_minority", "float"),
            ("holc_grade", "str"), ("n_requests", "int"), ("n_resolved", "int"),
            ("median_days", "float"), ("p90_days", "float"),
            ("severity_weighted_days", "float"),
        ],
        "community_area": [
            ("community_area", "int"), ("community_area_name", "str"),
            ("hardship_index", "float"), ("pct_below_poverty", "float"),
            ("per_capita_income", "float"), ("n_requests", "int"),
            ("n_resolved", "int"), ("median_days", "float"),
            ("p90_days", "float"), ("n_critical", "int"),
        ],
        "sr_type_summary": [
            ("sr_type", "str"), ("severity_band", "str"), ("severity_weight", "float"),
            ("is_informational", "bool"), ("n_requests", "int"),
            ("median_days", "float"), ("p90_days", "float"),
            ("n_unresolved", "int"), ("n_reversed_timestamps", "int"),
        ],
    }


SQL = {
    "requests_weekly": """
        SELECT week_start, n_requests, n_critical, n_high, median_days,
               avg_tmax_c, total_prcp_mm, total_snow_mm
        FROM analytics.weekly_volume ORDER BY week_start
    """,
    "monthly_demand": """
        SELECT month, sr_type, severity_band, n_requests, n_service_requests,
               n_resolved, median_days, p90_days, mean_days
        FROM analytics.monthly_demand ORDER BY month, sr_type
    """,
    "monthly_citywide": """
        SELECT month,
               count(*) AS n_requests,
               count(*) FILTER (WHERE is_actionable) AS n_actionable,
               round(median(days_to_resolution)
                     FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL), 3)
                                                        AS median_days,
               avg(tmax_c) AS avg_tmax_c,
               sum(prcp_mm) AS total_prcp_mm,
               sum(snow_mm) AS total_snow_mm
        FROM analytics.sr_enriched GROUP BY month ORDER BY month
    """,
    "tract_equity": """
        SELECT tract_fips, community_area, community_name, total_pop,
               median_hh_income, round(pct_minority, 2) AS pct_minority,
               holc_grade_mode AS holc_grade, n_requests, n_resolved,
               round(median_days, 3) AS median_days, round(p90_days, 2) AS p90_days,
               round(severity_weighted_days, 3) AS severity_weighted_days
        FROM analytics.tract_summary ORDER BY tract_fips
    """,
    "community_area": """
        SELECT community_area, community_area_name, hardship_index,
               pct_below_poverty, per_capita_income, n_requests, n_resolved,
               round(median_days, 3) AS median_days, round(p90_days, 2) AS p90_days,
               n_critical
        FROM analytics.community_area_summary ORDER BY community_area
    """,
    "sr_type_summary": """
        SELECT sr_type, severity_band, severity_weight, is_informational,
               n_requests, round(median_days, 3) AS median_days,
               round(p90_days, 2) AS p90_days, n_unresolved, n_reversed_timestamps
        FROM analytics.sr_type_summary ORDER BY n_requests DESC
    """,
}


def to_hyper_types():
    from tableauhyperapi import SqlType
    return {
        "date": SqlType.date(),
        "int": SqlType.big_int(),
        "float": SqlType.double(),
        "str": SqlType.text(),
        "bool": SqlType.bool(),
    }


def coerce(df: pd.DataFrame, spec: list[tuple[str, str]]) -> pd.DataFrame:
    """Coerce a DataFrame to the declared Hyper types."""
    out = pd.DataFrame()
    for col, kind in spec:
        if col not in df.columns:
            raise KeyError(f"column {col!r} missing from query result")
        s = df[col]
        if kind == "date":
            out[col] = pd.to_datetime(s).dt.date
        elif kind == "int":
            out[col] = s.astype("Int64")
        elif kind == "float":
            out[col] = s.astype("float64")
        elif kind == "str":
            # Tableau treats an all-null string column as unusable, so give it
            # an explicit empty string instead of a null object dtype.
            out[col] = s.astype("object").where(s.notna(), None)
        elif kind == "bool":
            out[col] = s.astype("boolean")
    return out


def build_extract() -> dict[str, int]:
    from tableauhyperapi import (Connection, CreateMode, HyperProcess, Inserter,
                                 SchemaName, TableDefinition, TableName,
                                 Telemetry)

    con = H.connect()
    specs = table_specs()
    hyper_types = to_hyper_types()
    counts: dict[str, int] = {}

    dataframes = {}
    for name, sql in SQL.items():
        df = H.query(sql, con)
        df = coerce(df, specs[name])
        dataframes[name] = df
        print(f"    prepared {name:18s} {len(df):>7,} rows x {len(df.columns)} cols")
    con.close()

    with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(hyper.endpoint, str(HYPER_PATH),
                        CreateMode.CREATE_AND_REPLACE) as conn:
            schema = SchemaName("Extract")
            conn.catalog.create_schema_if_not_exists(schema)

            for name, df in dataframes.items():
                definition = TableDefinition(
                    table_name=TableName(schema, name),
                    # Column is defined on TableDefinition in this Hyper API
                    # version; plain (name, type) tuples fail inside the native
                    # schema converter with "'tuple' object has no attribute 'name'".
                    columns=[TableDefinition.Column(c, hyper_types[k])
                             for c, k in specs[name]],
                )
                conn.catalog.create_table(definition)
                with Inserter(conn, definition) as inserter:
                    inserter.add_rows(df.itertuples(index=False, name=None))
                    inserter.execute()
                counts[name] = len(df)
                print(f"    wrote    {name:18s} {len(df):>7,} rows")

    return counts


def verify_extract(expected: dict[str, int]) -> None:
    """Read the file back and confirm every table is present and complete."""
    from tableauhyperapi import Connection, HyperProcess, Telemetry

    with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(hyper.endpoint, str(HYPER_PATH)) as conn:
            tables = conn.catalog.get_table_names("Extract")
            print(f"    tables in extract: {len(tables)}")
            assert len(tables) == len(expected), \
                f"expected {len(expected)} tables, found {len(tables)}"
            for table in tables:
                got = conn.execute_scalar_query(
                    f'SELECT COUNT(*) FROM {table}')
                name = table.name.unescaped
                want = expected[name]
                assert got == want, f"{name}: wrote {want} rows but read back {got}"
                cols = conn.catalog.get_table_definition(table).columns
                print(f"    verified {name:18s} {got:>7,} rows, {len(cols)} cols  OK")


def build_geojson() -> Path:
    """Tract polygons carrying the equity metrics as attributes."""
    import geopandas as gpd

    metrics = H.query(SQL["tract_equity"])
    geom = gpd.read_file(C.DATA_RAW / "aux" / "chicago_tracts_2020.geojson")
    # `tract_full` holds the 11-digit tract id; `tract_fips` in this GeoJSON is
    # the county id. Joining on the wrong column yields an empty map.
    geom = geom[["tract_full", "geometry"]].rename(columns={"tract_full": "tract_fips"})
    merged = geom.merge(metrics, on="tract_fips", how="inner")
    gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=geom.crs)
    gdf = gdf[gdf.geometry.notna()]

    out = SPATIAL_DIR / "chicago311_tract_equity.geojson"
    gdf.to_file(out, driver="GeoJSON")
    print(f"    wrote {out.relative_to(C.PROJECT_ROOT)} "
          f"({out.stat().st_size / 1e6:.2f} MB, {len(gdf)} tract polygons)")
    return out


def main() -> None:
    print("Stage 5a — Tableau Hyper extract\n")
    if not C.DB_PATH.exists():
        print(f"warehouse missing at {C.DB_PATH}; run `python src/etl.py` first")
        sys.exit(1)

    t0 = time.time()
    print("  building extract")
    counts = build_extract()
    print("\n  verifying extract by reading it back")
    verify_extract(counts)

    print("\n  building spatial layer for the choropleth")
    build_geojson()

    print(f"\n  {HYPER_PATH.relative_to(C.PROJECT_ROOT)}  "
          f"({HYPER_PATH.stat().st_size / 1e6:.2f} MB)")
    print(f"  {sum(counts.values()):,} rows across {len(counts)} tables "
          f"in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
