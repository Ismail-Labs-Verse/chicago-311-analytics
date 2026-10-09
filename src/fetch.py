"""
Stage 1 — Data collection.

Downloads every upstream source into data/raw/. The 311 pull is the expensive
part (millions of rows, rate-limited public API), so it is:

  * chunked by calendar month      -> a failure only costs one month,
  * paginated with a server-side   -> safe: Socrata pagination without ORDER BY
    ORDER BY                          is undefined and can duplicate/drop rows,
  * resumable                      -> a month that already finished is skipped,
  * retried with backoff           -> Socrata intermittently returns 5xx/429.

Run:  python src/fetch.py --req311       # the long pull
      python src/fetch.py --aux          # weather, census, boundaries, HOLC
      python src/fetch.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from calendar import monthrange
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

RAW_311 = C.DATA_RAW / "311"
RAW_AUX = C.DATA_RAW / "aux"
RAW_311.mkdir(parents=True, exist_ok=True)
RAW_AUX.mkdir(parents=True, exist_ok=True)

PAGE_SIZE = 50_000
USER_AGENT = "chicago-311-analytics/1.0 (portfolio case study; python-urllib)"


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only, so collection never depends on the ML stack)
# ---------------------------------------------------------------------------
def http_get(url: str, timeout: int = 600, retries: int = 6, label: str = "") -> bytes:
    """GET a URL with exponential backoff. Returns the raw response body."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            # 404 means the resource genuinely is not there: do not waste retries.
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                raise
            wait = min(2 ** attempt, 60)
            print(f"    ! {label or url[:70]} attempt {attempt + 1}/{retries} "
                  f"failed ({exc}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {label or url}: {last_err}")


def download_to_file(url: str, dest: Path, label: str = "", timeout: int = 900) -> Path:
    body = http_get(url, timeout=timeout, label=label or dest.name)
    dest.write_bytes(body)
    print(f"  saved {dest.name:44s} {len(body) / 1e6:9.2f} MB", flush=True)
    return dest


# ---------------------------------------------------------------------------
# Month enumeration
# ---------------------------------------------------------------------------
def month_list() -> list[str]:
    """['2019-01', ..., '2025-12'] covering the study window."""
    out = []
    y, m = int(C.START_DATE[:4]), int(C.START_DATE[5:7])
    for _ in range(C.N_MONTHS):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def month_bounds(ym: str) -> tuple[str, str]:
    """Inclusive start / exclusive end ISO timestamps for a 'YYYY-MM' month."""
    y, m = int(ym[:4]), int(ym[5:7])
    last = monthrange(y, m)[1]
    return (f"{y:04d}-{m:02d}-01T00:00:00",
            f"{y:04d}-{m:02d}-{last:02d}T23:59:59.999")


def full_detail_months() -> set[str]:
    """The most recent FULL_DETAIL_MONTHS months are pulled unfiltered."""
    return set(month_list()[-C.FULL_DETAIL_MONTHS:])


# ---------------------------------------------------------------------------
# 311 download
# ---------------------------------------------------------------------------
def _socrata_url(where: str, order: str, select: str, limit: int, offset: int) -> str:
    return (f"{C.CHI311_RESOURCE}.csv?"
            + urllib.parse.urlencode({
                "$select": select,
                "$where": where,
                "$order": order,
                "$limit": limit,
                "$offset": offset,
            }))


def download_311_month(ym: str) -> int:
    """Download one month of 311 data -> data/raw/311/<ym>.csv (resumable)."""
    start, end = month_bounds(ym)
    detail = "full" if ym in full_detail_months() else "sample20"
    dest = RAW_311 / f"{ym}.{detail}.csv"
    done = RAW_311 / f"{ym}.{detail}.done"

    if done.exists():
        n = int(done.read_text().strip() or 0)
        print(f"[{ym}] already complete ({n:,} rows) — skipped", flush=True)
        return n

    where = f"created_date>='{start}' AND created_date<='{end}'"
    if detail == "sample20":
        where += f" AND {C.SAMPLE_SQL}"

    select = ",".join(C.CHI311_COLUMNS)
    header: bytes | None = None
    rows = 0
    offset = 0
    tmp = dest.with_suffix(".csv.part")

    with open(tmp, "wb") as fh:
        while True:
            url = _socrata_url(where, C.ORDER_BY, select, PAGE_SIZE, offset)
            body = http_get(url, label=f"{ym} page@{offset}")
            lines = body.split(b"\n")
            # A paginated CSV repeats the header on every page.
            if not lines or not lines[0].strip():
                break
            page_header, page_rows = lines[0], [ln for ln in lines[1:] if ln.strip()]
            if header is None:
                header = page_header
                fh.write(header + b"\n")
            if not page_rows:
                break
            fh.write(b"\n".join(page_rows) + b"\n")
            rows += len(page_rows)
            print(f"    [{ym}] +{len(page_rows):,} rows (total {rows:,})", flush=True)
            if len(page_rows) < PAGE_SIZE:
                break
            offset += PAGE_SIZE

    tmp.replace(dest)
    done.write_text(str(rows))
    print(f"[{ym}] DONE  {rows:,} rows -> {dest.name}", flush=True)
    return rows


def fetch_311() -> None:
    months = month_list()
    detail_months = full_detail_months()
    print(f"311 pull: {len(months)} months "
          f"({len(months) - len(detail_months)} at 20% sample, "
          f"{len(detail_months)} at full detail)")
    print(f"  sample predicate : {C.SAMPLE_SQL}")
    print(f"  full-detail span : {min(detail_months)} .. {max(detail_months)}\n")
    t0 = time.time()
    total = 0
    for i, ym in enumerate(months, 1):
        print(f"--- [{i}/{len(months)}] {ym} ---", flush=True)
        total += download_311_month(ym)
    el = time.time() - t0
    print(f"\n311 pull complete: {total:,} rows in {el / 60:.1f} min")


# ---------------------------------------------------------------------------
# Auxiliary sources
# ---------------------------------------------------------------------------
def fetch_aux() -> None:
    print("NOAA daily weather (Chicago Midway, USW00014819)")
    download_to_file(C.NOAA_URL, RAW_AUX / "noaa_daily_weather.csv", "noaa")

    print("Community-area socioeconomic indicators (Chicago portal)")
    download_to_file(C.SOCIOECON_RESOURCE, RAW_AUX / "chicago_socioeconomic.csv", "socio")

    print("2020 census tract boundaries (Chicago portal, GeoJSON)")
    download_to_file(C.TRACT_BOUNDARY_GEOJSON, RAW_AUX / "chicago_tracts_2020.geojson",
                     "tracts")

    print("HOLC redlining grades (Mapping Inequality)")
    try:
        download_to_file(C.HOLC_CHICAGO_GEOJSON, RAW_AUX / "holc_chicago.geojson", "holc")
    except Exception as exc:  # redlining overlay is enrichment, not load-bearing
        print(f"  ! HOLC download failed ({exc}); "
              f"notebook 02 will report the overlay as unavailable")

    print("Census ACS 5-year 2023 (keyless table-based Summary File)")
    download_to_file(C.CENSUS_GEOS_FILE, RAW_AUX / "census_geos_2023_5yr.txt", "geos")
    for table in C.CENSUS_TABLES:
        url = f"{C.CENSUS_SF_BASE}/acsdt5y2023-{table}.dat"
        download_to_file(url, RAW_AUX / f"acsdt5y2023-{table}.dat", table)


def fetch_manifest() -> None:
    """Record exactly what was pulled, so the analysis is auditable."""
    months = month_list()
    detail = full_detail_months()
    manifest = {
        "sources": {
            "chicago_311": {
                "resource": C.CHI311_RESOURCE,
                "dataset_id": C.CHI311_DATASET_ID,
                "window": [C.START_DATE, C.END_DATE_EXCLUSIVE],
                "methods": {
                    "sample20": {
                        "predicate": C.SAMPLE_SQL,
                        "months": [m for m in months if m not in detail],
                        "note": ("20% systematic sample on the last digit of sr_number. "
                                 "sr_number is assigned sequentially, so the sample is "
                                 "even in time and exactly reproducible."),
                    },
                    "full": {
                        "months": sorted(detail),
                        "note": "unfiltered row-level detail for forecast hold-out",
                    },
                },
                "order_by": C.ORDER_BY,
                "columns": C.CHI311_COLUMNS,
                "license": "City of Chicago Data Portal — Open Data (public domain)",
            },
            "noaa_weather": {"url": C.NOAA_URL, "station": C.NOAA_STATION,
                             "variables": ["TMAX", "TMIN", "PRCP", "SNOW", "AWND"]},
            "chicago_socioeconomic": {"resource": C.SOCIOECON_RESOURCE},
            "census_tracts_2020": {"resource": C.TRACT_BOUNDARY_GEOJSON},
            "census_acs5_2023": {
                "base": C.CENSUS_SF_BASE,
                "tables": C.CENSUS_TABLES,
                "note": ("api.census.gov requires a registered API key; the keyless "
                         "table-based Summary File on www2.census.gov is used instead."),
            },
            "holc_redlining": {"url": C.HOLC_CHICAGO_GEOJSON},
        }
    }
    out = C.DATA_RAW / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out.relative_to(C.PROJECT_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Chicago 311 data collection")
    ap.add_argument("--req311", action="store_true",
                    help="download the 311 service request data (the long pull)")
    ap.add_argument("--aux", action="store_true",
                    help="weather, census, boundaries, socioeconomic, HOLC")
    ap.add_argument("--all", action="store_true", help="everything")
    args = ap.parse_args()

    did = False
    if args.all or args.req311:
        fetch_311()
        did = True
    if args.all or args.aux:
        fetch_aux()
        did = True
    if did:
        fetch_manifest()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
