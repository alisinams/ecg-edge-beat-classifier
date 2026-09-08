#!/usr/bin/env bash
# Everything after the hyperparameter search, in dependency order.
# Each stage writes its own log; the script stops at the first failure so a
# downstream table is never built from a half-finished upstream stage.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="${ECG_LOG_DIR:-.}"
mkdir -p "$LOG"

run () {
  local name="$1"; shift
  echo "=== $name ==="
  if ! python -u "$@" > "$LOG/$name.log" 2>&1; then
    echo "FAILED: $name  (see $LOG/$name.log)"
    tail -30 "$LOG/$name.log"
    exit 1
  fi
  tail -4 "$LOG/$name.log"
}

run teacher_ds2   code/04b_teacher_ds2.py
run train_ds1     code/06_train_arms.py --arms A,B,B2,C,D,E,F --tag ds1
run train_mixed   code/06_train_arms.py --arms A,B,B2,C,D,E,F --tag mixed --mixed
run predict       code/08_predict.py --what ds2,mixed,external
run deploy        code/09_deploy.py
run calibration   code/12_calibration.py
run ablations     code/11_ablations.py
run bandwidth     code/14_bandwidth.py
run optstats      code/13_optimiser_stats.py
run tables        code/10_tables.py
run figdata       code/18_figure_data.py
run manuscript    code/15_build_manuscript.py
run supplement    code/16_build_supplement.py
run persian       code/19_build_persian_report.py
run figguide      code/20_update_figures_guide.py
run docx          code/17_build_docx.py
run citations     code/21_check_citations.py
run styleaudit    code/22_style_audit.py
run consistency   code/23_consistency_check.py
echo "PIPELINE_COMPLETE"
