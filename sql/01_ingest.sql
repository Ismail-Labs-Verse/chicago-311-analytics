-- ===========================================================================
-- 01_ingest.sql — Chicago 311 request analytics
--
-- Relational + spatial pipeline: raw files -> clean -> geocoded -> enriched
-- -> analytical aggregates. Executed by src/etl.py against DuckDB.
--
-- Inputs (all under data/raw/, never committed):
--   311/<YYYY-MM>.{sample20,full}.csv   Socrata 311 export, 12 projected columns
--   aux/noaa_daily_weather.csv          NCEI daily summaries, station USW00014819
--   aux/chicago_tracts_2020.geojson     2020 census tract polygons
--   aux/holc_chicago.geojson            HOLC redlining grades (A-D)
--   aux/acsdt5y2023-{b19013,b03002,b01003}.dat   ACS 5-yr 2023 tract estimates
--   aux/chicago_socioeconomic.csv       community-area hardship indicators
--
-- One table is materialized by the caller, not defined here:
--   severity_map(keyword, band, weight, priority)
--       Severity rules live in exactly one place — src/config.py — and etl.py
--       loads them into this table so the keyword logic is never duplicated.
--
-- Sampling calibre (see README): 2019-01..2024-11 are a 20% systematic sample
-- of all requests; 2024-12..2025-12 are complete row-level detail. Every
-- aggregate therefore carries the sample it came from, and notebook 01 checks
-- sample-derived monthly counts against exact full-population counts pulled
-- from the API (analytics.monthly_true_counts).
-- ===========================================================================

INSTALL spatial; LOAD spatial;
INSTALL json;    LOAD json;

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS analytics;

-- ---------------------------------------------------------------------------
-- 0. Parameter tables (materialized from src/config.py by etl.py)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE severity_map (
    keyword  VARCHAR,
    band     VARCHAR,
    weight   DOUBLE,
    priority INTEGER
);

-- Request types that are logged and closed administratively with no crew
-- dispatched. See src/config.py for why these must be excluded from
-- response-time statistics and kept for demand statistics.
CREATE OR REPLACE TABLE informational_patterns (pattern VARCHAR);

-- Shared numeric parameters (never hard-coded twice).
CREATE OR REPLACE TABLE pipeline_params (key VARCHAR, value DOUBLE);

-- ---------------------------------------------------------------------------
-- 1. Raw 311 ingest — every monthly file, tagged with month and sampling mode
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE raw.sr_311 AS
SELECT
    sr_number,
    sr_type,
    created_date,
    closed_date,
    status,
    owner_department,
    origin,
    community_area,
    ward,
    latitude,
    longitude,
    duplicate,
    regexp_extract(filename, '([0-9]{4}-[0-9]{2})', 1)                       AS src_month,
    CASE WHEN filename LIKE '%.full.csv' THEN 'full' ELSE 'sample20' END     AS src_detail,
    filename                                                                 AS src_file
FROM read_csv(
    'data/raw/311/*.csv',
    header = true,
    filename = true,
    union_by_name = true,
    all_varchar = true,
    ignore_errors = true
);

-- ---------------------------------------------------------------------------
-- 2. Clean + type + derive. Anything that cannot be typed is dropped or NULLed
--    rather than silently coerced.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE staging.sr_clean AS
SELECT
    upper(trim(sr_number))                                   AS sr_number,
    trim(sr_type)                                            AS sr_type,
    TRY_CAST(created_date AS TIMESTAMP)                      AS created_ts,
    TRY_CAST(closed_date  AS TIMESTAMP)                      AS closed_ts,
    trim(status)                                             AS status,
    trim(owner_department)                                   AS owner_department,
    trim(origin)                                             AS origin,
    TRY_CAST(community_area AS INTEGER)                      AS community_area,
    TRY_CAST(ward AS INTEGER)                                AS ward,
    TRY_CAST(latitude  AS DOUBLE)                            AS latitude,
    TRY_CAST(longitude AS DOUBLE)                            AS longitude,
    CASE lower(trim(duplicate))
        WHEN 'true'  THEN TRUE
        WHEN 'false' THEN FALSE
        ELSE NULL
    END                                                      AS is_duplicate,
    src_month,
    src_detail
FROM raw.sr_311
WHERE TRY_CAST(created_date AS TIMESTAMP) IS NOT NULL
  AND TRY_CAST(created_date AS TIMESTAMP) >= TIMESTAMP '2019-01-01 00:00:00'
  AND TRY_CAST(created_date AS TIMESTAMP) <  TIMESTAMP '2026-01-01 00:00:00';

-- Derived calendar fields and the response-time measure.
-- days_to_resolution is NULL when the request is still open, or when the two
-- timestamps are inconsistent (closed before created — a known data-entry
-- artefact in this dataset). Negative durations are not silently clamped.
CREATE OR REPLACE TABLE staging.sr_derived AS
SELECT
    *,
    CAST(date_trunc('month', created_ts) AS DATE)              AS month,
    CAST(date_trunc('week',  created_ts) AS DATE)              AS week_start,
    EXTRACT(year  FROM created_ts)::INTEGER                    AS yr,
    EXTRACT(month FROM created_ts)::INTEGER                    AS created_month,
    EXTRACT(hour  FROM created_ts)::INTEGER                    AS created_hour,
    EXTRACT(isodow FROM created_ts)::INTEGER                   AS created_dow,
    CASE
        WHEN closed_ts IS NOT NULL AND closed_ts >= created_ts
        THEN date_diff('second', created_ts, closed_ts) / 86400.0
    END                                                        AS days_to_resolution,
    CASE
        WHEN closed_ts IS NOT NULL AND closed_ts < created_ts THEN TRUE
        ELSE FALSE
    END                                                        AS has_reversed_timestamps
FROM staging.sr_clean;

-- ---------------------------------------------------------------------------
-- 3. Geography: 2020 census tract polygons, then point-in-polygon assignment
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE staging.tract_geom AS
SELECT
    TRACT_FULL          AS tract_fips,
    COMMAREA_N          AS tract_community_area,
    COMMUNITY_AREA      AS tract_community_name,
    geom                AS geom
FROM ST_Read('data/raw/aux/chicago_tracts_2020.geojson');

-- Give the spatial join an R-tree so it does not degenerate into a
-- 2.6M x 789 nested loop.
CREATE INDEX IF NOT EXISTS tract_geom_rtree ON staging.tract_geom USING RTREE (geom);

-- The dataset exposes coordinates, not tracts: the tract is produced by a real
-- point-in-polygon test. Requests outside every Chicago tract (bad geocodes,
-- or points in the lake) correctly receive a NULL tract instead of a
-- false-positive assignment.
CREATE OR REPLACE TABLE staging.sr_geo AS
SELECT
    s.*,
    t.tract_fips,
    t.tract_community_area,
    t.tract_community_name
FROM staging.sr_derived s
LEFT JOIN staging.tract_geom t
       ON s.longitude IS NOT NULL
      AND s.latitude  IS NOT NULL
      AND s.longitude BETWEEN -88.10 AND -87.50
      AND s.latitude  BETWEEN  41.60 AND  42.05
      AND ST_Contains(t.geom, ST_Point(s.longitude, s.latitude));

-- ---------------------------------------------------------------------------
-- 4. Severity mapping — resolved once per distinct request type, then joined
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE staging.sr_type_severity AS
WITH types AS (
    SELECT DISTINCT sr_type FROM staging.sr_clean WHERE sr_type IS NOT NULL
),
ranked AS (
    SELECT
        t.sr_type,
        sm.band,
        sm.weight,
        sm.priority,
        ROW_NUMBER() OVER (PARTITION BY t.sr_type ORDER BY sm.priority) AS rn
    FROM types t
    LEFT JOIN severity_map sm
           ON lower(t.sr_type) LIKE '%' || sm.keyword || '%'
)
SELECT
    sr_type,
    COALESCE(band, 'medium')    AS severity_band,
    COALESCE(weight, 0.45)      AS severity_weight
FROM ranked
WHERE rn = 1;

-- ---------------------------------------------------------------------------
-- 5. Census ACS 5-year 2023 at tract level (keyless table-based Summary File)
--    GEO_ID '1400000US17031010100' -> tract FIPS = trailing 11 characters
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE staging.acs_tract AS
WITH base AS (
    -- Population is never legitimately negative; the sentinel check guards
    -- against ACS"estimate not available" codes reaching the analytics.
    SELECT regexp_extract(geo_id, 'US([0-9]{11})$', 1) AS tract_fips,
           CASE WHEN TRY_CAST(B01003_E001 AS BIGINT) > 0
                THEN TRY_CAST(B01003_E001 AS BIGINT)
           END                                          AS total_pop
    FROM read_csv('data/raw/aux/acsdt5y2023-b01003.dat', delim = '|', header = true)
),
race AS (
    SELECT regexp_extract(geo_id, 'US([0-9]{11})$', 1) AS tract_fips,
           TRY_CAST(B03002_E001 AS BIGINT)             AS race_total,
           TRY_CAST(B03002_E003 AS BIGINT)             AS nh_white
    FROM read_csv('data/raw/aux/acsdt5y2023-b03002.dat', delim = '|', header = true)
),
income AS (
    -- ACS uses -666666666 as its "estimate not available" sentinel. Left as a
    -- number it poisons every average that touches it (it dragged the grade-C
    -- mean income to about -$8 million), so it is mapped to NULL here.
    SELECT regexp_extract(geo_id, 'US([0-9]{11})$', 1) AS tract_fips,
           CASE WHEN TRY_CAST(B19013_E001 AS DOUBLE) > 0
                THEN TRY_CAST(B19013_E001 AS DOUBLE)
           END                                          AS median_hh_income
    FROM read_csv('data/raw/aux/acsdt5y2023-b19013.dat', delim = '|', header = true)
)
SELECT
    b.tract_fips,
    b.total_pop,
    r.nh_white,
    i.median_hh_income,
    r.race_total,
    CASE WHEN r.race_total > 0
         THEN 100.0 * (r.race_total - r.nh_white) / r.race_total
    END                                                AS pct_minority,
    CASE WHEN b.total_pop > 0
         THEN 100.0 * r.nh_white / b.total_pop
    END                                                AS pct_nh_white
FROM base b
LEFT JOIN race   r USING (tract_fips)
LEFT JOIN income i USING (tract_fips)
WHERE b.tract_fips LIKE '17031%';   -- Cook County, Illinois

-- ---------------------------------------------------------------------------
-- 6. NOAA daily weather. NCEI units: temperature tenths degC, precip tenths mm,
--    snow mm, wind tenths m/s -> converted to degC / mm / m/s. Missing values
--    arrive as blanks and stay NULL.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE staging.weather_daily AS
SELECT
    TRY_CAST("DATE" AS DATE)      AS obs_date,
    TRY_CAST(TMAX AS DOUBLE) / 10 AS tmax_c,
    TRY_CAST(TMIN AS DOUBLE) / 10 AS tmin_c,
    TRY_CAST(PRCP AS DOUBLE) / 10 AS prcp_mm,
    TRY_CAST(SNOW AS DOUBLE)      AS snow_mm,
    TRY_CAST(AWND AS DOUBLE) / 10 AS awnd_ms
FROM read_csv('data/raw/aux/noaa_daily_weather.csv', header = true);

-- ---------------------------------------------------------------------------
-- 7. HOLC redlining grades (spatial). Only the four official grades are kept.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE staging.holc_geom AS
SELECT upper(trim(grade)) AS holc_grade, geom
FROM ST_Read('data/raw/aux/holc_chicago.geojson')
WHERE upper(trim(grade)) IN ('A', 'B', 'C', 'D');

CREATE INDEX IF NOT EXISTS holc_geom_rtree ON staging.holc_geom USING RTREE (geom);

CREATE OR REPLACE TABLE staging.sr_holc AS
SELECT
    s.sr_number,
    h.holc_grade
FROM staging.sr_derived s
LEFT JOIN staging.holc_geom h
       ON s.longitude IS NOT NULL
      AND s.latitude  IS NOT NULL
      AND s.longitude BETWEEN -88.10 AND -87.50
      AND s.latitude  BETWEEN  41.60 AND  42.05
      AND ST_Contains(h.geom, ST_Point(s.longitude, s.latitude));

-- ---------------------------------------------------------------------------
-- 8. Community-area socioeconomic indicators (hardship index and drivers)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE staging.socioeconomic AS
SELECT
    TRY_CAST(ca AS INTEGER)                                     AS community_area,
    trim(community_area_name)                                   AS community_area_name,
    TRY_CAST(percent_of_housing_crowded AS DOUBLE)              AS pct_housing_crowded,
    TRY_CAST(percent_households_below_poverty AS DOUBLE)        AS pct_below_poverty,
    TRY_CAST(percent_aged_16_unemployed AS DOUBLE)              AS pct_unemployed,
    TRY_CAST(percent_aged_25_without_high_school_diploma AS DOUBLE) AS pct_no_hs_diploma,
    TRY_CAST(percent_aged_under_18_or_over_64 AS DOUBLE)        AS pct_dependent_age,
    TRY_CAST(per_capita_income_ AS DOUBLE)                      AS per_capita_income,
    TRY_CAST(hardship_index AS DOUBLE)                          AS hardship_index
FROM read_csv('data/raw/aux/chicago_socioeconomic.csv', header = true);

-- ---------------------------------------------------------------------------
-- 9. The analytical table: one row per 311 request, fully enriched.
--
-- is_actionable separates real field-service work from administratively
-- closed filings. Response-time analysis uses is_actionable; demand and
-- seasonality analysis uses every row.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE analytics.sr_enriched AS
WITH enriched AS (
    SELECT
        g.sr_number,
        g.sr_type,
        g.created_ts,
        g.closed_ts,
        g.status,
        g.owner_department,
        g.origin,
        g.community_area,
        g.ward,
        g.latitude,
        g.longitude,
        g.is_duplicate,
        g.src_month,
        g.src_detail,
        g.month,
        g.week_start,
        g.yr,
        g.created_month,
        g.created_hour,
        g.created_dow,
        g.days_to_resolution,
        g.has_reversed_timestamps,
        g.tract_fips,
        g.tract_community_area,
        g.tract_community_name,
        sv.severity_band,
        sv.severity_weight,
        CASE WHEN g.days_to_resolution IS NOT NULL
             THEN g.days_to_resolution * sv.severity_weight END   AS severity_weighted_days,
        a.total_pop,
        a.median_hh_income,
        a.pct_minority,
        h.holc_grade,
        w.tmax_c,
        w.tmin_c,
        w.prcp_mm,
        w.snow_mm,
        w.awnd_ms,
        sc.hardship_index,
        sc.pct_below_poverty,
        sc.per_capita_income,
        sc.community_area_name,
        (EXISTS (SELECT 1 FROM informational_patterns p
                  WHERE lower(g.sr_type) LIKE '%' || p.pattern || '%')) AS is_informational
    FROM staging.sr_geo g
    LEFT JOIN staging.sr_type_severity sv ON g.sr_type = sv.sr_type
    LEFT JOIN staging.acs_tract        a  ON g.tract_fips = a.tract_fips
    LEFT JOIN staging.sr_holc          h  ON g.sr_number = h.sr_number
    LEFT JOIN staging.weather_daily    w  ON CAST(g.created_ts AS DATE) = w.obs_date
    LEFT JOIN staging.socioeconomic    sc ON g.community_area = sc.community_area
)
SELECT
    *,
    -- Resolvable by an actual work order: not an administrative filing, and
    -- either still open or genuinely held for longer than an instant.
    (NOT is_informational) AS is_service_request,
    CASE
        WHEN is_informational THEN FALSE
        WHEN days_to_resolution IS NULL THEN TRUE          -- open, still pending
        WHEN days_to_resolution >= (SELECT value / 1440.0 FROM pipeline_params
                                     WHERE key = 'admin_closure_minutes')
        THEN TRUE
        ELSE FALSE
    END AS is_actionable
FROM enriched;

-- ---------------------------------------------------------------------------
-- 10. Exact full-population reference counts (populated by etl.py from the
--     Socrata API) — lets the notebooks validate the 20% sample against truth.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE analytics.monthly_true_counts (
    month      DATE,
    true_count BIGINT
);

-- ---------------------------------------------------------------------------
-- 11. Analytical aggregates
-- ---------------------------------------------------------------------------

-- Monthly demand by request type (forecasting input).
-- Counts include informational calls (they are still demand); response-time
-- statistics are restricted to actionable service requests.
CREATE OR REPLACE TABLE analytics.monthly_demand AS
SELECT
    month,
    sr_type,
    severity_band,
    count(*)                                                   AS n_requests,
    count(*) FILTER (WHERE is_service_request)                 AS n_service_requests,
    count(*) FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL)
                                                               AS n_resolved,
    median(days_to_resolution) FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL)
                                                               AS median_days,
    quantile_cont(days_to_resolution, 0.90) FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL)
                                                               AS p90_days,
    avg(days_to_resolution) FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL)
                                                               AS mean_days
FROM analytics.sr_enriched
GROUP BY month, sr_type, severity_band;

-- Weekly citywide volume (anomaly detection input)
CREATE OR REPLACE TABLE analytics.weekly_volume AS
SELECT
    week_start,
    count(*)                                                   AS n_requests,
    count(*) FILTER (WHERE severity_band = 'critical')         AS n_critical,
    count(*) FILTER (WHERE severity_band = 'high')             AS n_high,
    median(days_to_resolution)                                 AS median_days,
    avg(tmax_c)                                                AS avg_tmax_c,
    sum(prcp_mm)                                               AS total_prcp_mm,
    sum(snow_mm)                                               AS total_snow_mm,
    sum(CASE WHEN src_detail = 'full' THEN 1 ELSE 0 END) * 1.0
        / count(*)                                             AS full_detail_share
FROM analytics.sr_enriched
GROUP BY week_start;

-- Tract-level equity table (response time vs. demographics + redlining).
-- Response-time statistics cover actionable service requests only.
CREATE OR REPLACE TABLE analytics.tract_summary AS
SELECT
    tract_fips,
    any_value(tract_community_area)                            AS community_area,
    any_value(tract_community_name)                            AS community_name,
    any_value(total_pop)                                       AS total_pop,
    any_value(median_hh_income)                                AS median_hh_income,
    any_value(pct_minority)                                    AS pct_minority,
    mode(holc_grade)                                           AS holc_grade_mode,
    count(*)                                                   AS n_requests,
    count(*) FILTER (WHERE is_service_request)                 AS n_service_requests,
    count(*) FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL)
                                                               AS n_resolved,
    median(days_to_resolution) FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL)
                                                               AS median_days,
    quantile_cont(days_to_resolution, 0.90) FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL)
                                                               AS p90_days,
    avg(days_to_resolution) FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL)
                                                               AS mean_days,
    avg(severity_weight) FILTER (WHERE is_actionable)          AS mean_severity,
    avg(severity_weighted_days) FILTER (WHERE is_actionable)   AS severity_weighted_days,
    count(*) FILTER (WHERE sr_type = 'Rodent Baiting/Rat Complaint') AS n_rodent,
    count(*) FILTER (WHERE severity_band = 'critical')         AS n_critical
FROM analytics.sr_enriched
WHERE tract_fips IS NOT NULL
GROUP BY tract_fips;

-- Community-area rollup (dashboard + policy recommendations)
CREATE OR REPLACE TABLE analytics.community_area_summary AS
SELECT
    e.community_area,
    any_value(e.community_area_name)                           AS community_area_name,
    any_value(sc.hardship_index)                               AS hardship_index,
    any_value(sc.pct_below_poverty)                            AS pct_below_poverty,
    any_value(sc.per_capita_income)                            AS per_capita_income,
    count(*)                                                   AS n_requests,
    count(*) FILTER (WHERE e.is_actionable AND e.days_to_resolution IS NOT NULL)
                                                               AS n_resolved,
    median(e.days_to_resolution) FILTER (WHERE e.is_actionable AND e.days_to_resolution IS NOT NULL)
                                                               AS median_days,
    quantile_cont(e.days_to_resolution, 0.90) FILTER (WHERE e.is_actionable AND e.days_to_resolution IS NOT NULL)
                                                               AS p90_days,
    avg(e.severity_weighted_days) FILTER (WHERE e.is_actionable)
                                                               AS severity_weighted_days,
    count(*) FILTER (WHERE e.severity_band = 'critical')       AS n_critical
FROM analytics.sr_enriched e
LEFT JOIN staging.socioeconomic sc ON e.community_area = sc.community_area
WHERE e.community_area IS NOT NULL
GROUP BY e.community_area;

-- Request-type breakdown (severity index documentation + EDA)
CREATE OR REPLACE TABLE analytics.sr_type_summary AS
SELECT
    sr_type,
    any_value(severity_band)                                   AS severity_band,
    any_value(severity_weight)                                 AS severity_weight,
    any_value(is_informational)                                AS is_informational,
    count(*)                                                   AS n_requests,
    median(days_to_resolution) FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL)
                                                               AS median_days,
    quantile_cont(days_to_resolution, 0.90) FILTER (WHERE is_actionable AND days_to_resolution IS NOT NULL)
                                                               AS p90_days,
    count(*) FILTER (WHERE days_to_resolution IS NULL)         AS n_unresolved,
    count(*) FILTER (WHERE has_reversed_timestamps)            AS n_reversed_timestamps
FROM analytics.sr_enriched
GROUP BY sr_type;
