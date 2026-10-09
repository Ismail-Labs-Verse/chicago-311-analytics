"""
Central configuration for the Chicago 311 analytics pipeline.

Keeping every dataset identifier, date boundary and sampling constant in one
place makes the pipeline auditable: a reviewer can see exactly which upstream
resource each number came from.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SAMPLES = PROJECT_ROOT / "data" / "samples"
SQL_DIR = PROJECT_ROOT / "sql"
DASHBOARDS = PROJECT_ROOT / "dashboards"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

for _d in (DATA_RAW, DATA_PROCESSED, DATA_SAMPLES, DASHBOARDS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Study window
# --------------------------------------------------------------------------
# The dataset itself starts 2018-07-01. The study window is calendar
# 2019-01-01 .. 2025-12-31 so that every year is complete and the window is
# fully covered by the weather series.
START_DATE = "2019-01-01"
END_DATE_EXCLUSIVE = "2026-01-01"
N_MONTHS = 84  # 7 years x 12

# Last N months are downloaded at FULL row-level detail (no sampling) so the
# forecasting notebooks are evaluated on a complete, unbiased hold-out.
FULL_DETAIL_MONTHS = 13

# --------------------------------------------------------------------------
# Upstream data sources
# --------------------------------------------------------------------------

# --- Chicago 311 Service Requests (Socrata, data.cityofchicago.org) --------
CHI311_DATASET_ID = "v6vf-nfxy"
CHI311_RESOURCE = f"https://data.cityofchicago.org/resource/{CHI311_DATASET_ID}"

# Only the columns the analysis actually uses. Pulling all 40 columns tripled
# the transfer volume for no analytical gain, and created_hour/day_of_week/month
# are trivially derivable from created_date in SQL.
CHI311_COLUMNS = [
    "sr_number",
    "sr_type",
    "created_date",
    "closed_date",
    "status",
    "owner_department",
    "origin",
    "community_area",
    "ward",
    "latitude",
    "longitude",
    "duplicate",
]

# 20% systematic sample: the last digit of sr_number is essentially a random
# uniform draw, and sr_number is assigned sequentially, so this samples evenly
# across time. Combined with a stable server-side sort it is ALSO deterministic:
# re-running the pipeline reproduces the identical row set.
SAMPLE_SQL = "(sr_number like '%0' OR sr_number like '%1')"

# Server-side ordering is mandatory: pagination without ORDER BY is undefined
# in Socrata and would silently duplicate/drop rows across pages.
ORDER_BY = "created_date,sr_number"

# --- NOAA NCEI daily weather (Chicago Midway, station USW00014819) ---------
NOAA_STATION = "USW00014819"
NOAA_URL = (
    "https://www.ncei.noaa.gov/access/services/data/v1"
    "?dataset=daily-summaries"
    f"&stations={NOAA_STATION}"
    f"&startDate={START_DATE}"
    "&endDate=2025-12-31"
    "&dataTypes=TMAX,TMIN,PRCP,SNOW,AWND"
    "&format=csv"
)

# --- Chicago community-area socioeconomic indicators (keyless, Socrata) ----
SOCIOECON_DATASET_ID = "kn9c-c2s2"
SOCIOECON_RESOURCE = (
    f"https://data.cityofchicago.org/resource/{SOCIOECON_DATASET_ID}.csv"
)

# --- Boundaries: 2020 census tracts (keyless, Socrata GeoJSON) -------------
TRACT_BOUNDARY_DATASET_ID = "7qjk-677h"
TRACT_BOUNDARY_GEOJSON = (
    f"https://data.cityofchicago.org/resource/{TRACT_BOUNDARY_DATASET_ID}.geojson"
)

# --- US Census: keyless table-based Summary File (ACS 5-year 2023) ---------
# The api.census.gov JSON API now requires a registered key. The table-based
# Summary File on www2.census.gov does not, so the pipeline uses it instead.
CENSUS_SF_BASE = (
    "https://www2.census.gov/programs-surveys/acs/"
    "summary_file/2023/table-based-SF/data/5YRData"
)
# Table / label / the variable we keep from it.
CENSUS_TABLES = {
    "b19013": {"label": "median_household_income", "variable": "B19013_001"},
    "b03002": {"label": "race_hispanic_origin", "variable": "B03002_001"},
    "b01003": {"label": "total_population", "variable": "B01003_001"},
}
CENSUS_GEOS_FILE = (
    "https://www2.census.gov/programs-surveys/acs/summary_file/2023/"
    "table-based-SF/documentation/Geos20235YR.txt"
)

# --- HOLC redlining grades (Mapping Inequality, Univ. of Richmond) ---------
HOLC_CHICAGO_GEOJSON = (
    "https://dsl.richmond.edu/panorama/redlining/static/downloads/geojson/ILChicago.geojson"
)

# --------------------------------------------------------------------------
# DuckDB
# --------------------------------------------------------------------------
DB_PATH = DATA_PROCESSED / "chicago311.duckdb"

# --------------------------------------------------------------------------
# Severity index
#
# severity_index classifies each request family by how disruptive it is to a
# resident's daily life and how urgently the city aims to respond. It is
# deliberately keyword-driven and lives in ONE place: etl.py loads this table
# into DuckDB (`severity_map`) so the SQL pipeline and any Python code agree
# by construction rather than by copied-and-pasted rules.
#
# Band priority is the order rules are evaluated: a request type mentioning
# both "water" (critical) and "street" (medium) resolves to critical.
# --------------------------------------------------------------------------
SEVERITY_BANDS = {
    "critical": 1.00,   # immediate risk to health/safety or loss of housing
    "high": 0.70,       # habitability, utilities, sanitation
    "medium": 0.45,     # infrastructure, street/sidewalk, environmental health
    "low": 0.20,        # information, administrative, cosmetic
}

# Lower band priority number wins when several keywords match.
BAND_PRIORITY = {"critical": 1, "high": 2, "medium": 3, "low": 4}

SEVERITY_KEYWORDS = {
    "critical": (
        "water", "sewer", "flooding", "gas", "heat", "no heat",
        "building violation", "electric", "safety", "hazard",
        "tree emergency", "collapse", "fire", "lead", "mold",
        "vacant", "abandoned", "dangerous",
    ),
    "high": (
        "rodent", "rat", "sanitation", "garbage", "refuse", "trash",
        "dumpster", "recycl", "snow", "ice", "pothole", "graffiti",
        "street light", "streetlight", "rodent baiting", "dead animal",
        "litter",
    ),
    "medium": (
        "pavement", "sidewalk", "traffic", "sign", "tree", "street",
        "alley", "curb", "bike", "parking", "vehicle", "abandoned vehicle",
        "weed", "grass", "forestry", "street cleaning",
    ),
    "low": (
        "information", "311 information", "compliment", "inquiry",
        "administrative", "signs", "beautification", "notice",
    ),
}


def severity_rows() -> list[tuple[str, str, float, int]]:
    """Flatten the ruleset to (keyword, band, weight, priority) for DuckDB."""
    rows: list[tuple[str, str, float, int]] = []
    for band, keywords in SEVERITY_KEYWORDS.items():
        for kw in keywords:
            rows.append((kw.lower(), band, SEVERITY_BANDS[band], BAND_PRIORITY[band]))
    return rows


def severity_band(sr_type: str | None) -> str:
    """Classify a 311 request type into a documented severity band.

    Mirrors the SQL resolution in sql/01_ingest.sql (lowest priority number
    wins) so Python and DuckDB cannot disagree.
    """
    t = (sr_type or "").lower()
    for band in sorted(BAND_PRIORITY, key=BAND_PRIORITY.get):
        for kw in SEVERITY_KEYWORDS[band]:
            if kw in t:
                return band
    return "medium"


# --------------------------------------------------------------------------
# Actionability filter
#
# A large share of rows in this dataset are NOT field-service work: citizens
# calling 311 with a question, and noise reports, are logged and closed
# administratively within seconds. Measured on the first 20 months of the
# study window, 56.3% of all rows close in under one minute, which collapses
# the citywide median response time to ~0 days and makes any equity comparison
# meaningless if they are left in.
#
# They are therefore FLAGGED rather than deleted: volume/seasonality analysis
# keeps them (a call is still demand on the contact centre), while every
# response-time statistic is computed over `is_actionable` rows only.
# --------------------------------------------------------------------------
INFORMATIONAL_PATTERNS = (
    "information only",   # "311 INFORMATION ONLY CALL"
    "311 information",
    "aircraft noise",     # logged and auto-closed without dispatch
)

# A resolution faster than this cannot have involved dispatching a crew.
ADMIN_CLOSURE_MINUTES = 5.0


def informational_patterns() -> list[tuple[str]]:
    """Patterns for DuckDB to match against the lowercased sr_type."""
    return [(p.lower(),) for p in INFORMATIONAL_PATTERNS]


def pipeline_params() -> list[tuple[str, float]]:
    """Numeric parameters shared between SQL and Python."""
    return [("admin_closure_minutes", ADMIN_CLOSURE_MINUTES)]


def is_informational(sr_type: str | None) -> bool:
    """True if the request type is informational / auto-closed."""
    t = (sr_type or "").lower()
    return any(p in t for p in INFORMATIONAL_PATTERNS)

