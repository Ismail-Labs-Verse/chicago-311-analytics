#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# End-to-end pipeline for the Chicago 311 analytics case study.
#
# Every stage is idempotent and the collection stage is resumable, so this can
# be re-run safely at any point after an interruption.
#
# Usage:
#   ./run_pipeline.sh            # everything except the (long) data collection
#   ./run_pipeline.sh --all      # including collection
#   ./run_pipeline.sh --from etl # resume at a named stage
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-/opt/anaconda3/envs/chicago311/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3)"
  echo "warning: chicago311 env not found, falling back to $PY"
fi

STAGES=(fetch etl notebooks hyper powerbi html report)
START_AT="etl"
INCLUDE_FETCH=0

while [ $# -gt 0 ]; do
  case "$1" in
    --all)  INCLUDE_FETCH=1 ;;
    --from) START_AT="${2:-etl}"; shift ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
  shift
done

if [ "$INCLUDE_FETCH" = "1" ]; then START_AT="fetch"; fi

run() {
  local name="$1"; shift
  echo ""
  echo "############################################################"
  echo "# stage: $name"
  echo "############################################################"
  "$@"
}

skipping=1
for stage in "${STAGES[@]}"; do
  if [ "$stage" = "$START_AT" ]; then skipping=0; fi
  if [ "$skipping" = "1" ]; then
    [ "$stage" = "fetch" ] && echo "skipping collection (pass --all to include it)"
    continue
  fi
  case "$stage" in
    fetch)     run "collect data"     "$PY" src/fetch.py --all ;;
    etl)       run "build warehouse"  "$PY" src/etl.py ;;
    notebooks) run "run notebooks"    "$PY" src/generate_notebooks.py ;;
    hyper)     run "Tableau extract"  "$PY" src/build_hyper.py ;;
    powerbi)   run "Power BI project" "$PY" src/build_powerbi.py ;;
    html)      run "HTML dashboard"   "$PY" src/build_html_dashboard.py ;;
    report)    run "case study PDF"   "$PY" src/build_report.py ;;
  esac
done

echo ""
echo "############################################################"
echo "# pipeline complete"
echo "############################################################"
ls -lh data/processed/chicago311.duckdb 2>/dev/null || true
ls -lh notebooks/*.ipynb 2>/dev/null | awk '{print $5, $9}' || true
ls -lh dashboards/*.hyper dashboards/*.html 2>/dev/null | awk '{print $5, $9}' || true
ls -lh reports/case_study.pdf 2>/dev/null | awk '{print $5, $9}' || true
