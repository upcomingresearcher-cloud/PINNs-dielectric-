#!/bin/sh
# Phase 12 driver: runs every robustness variation (resumable; skips completed run_ids). Usage: sh scripts/run_robustness.sh BUDGET_SECONDS
BUDGET=${1:-240}; START=$(date +%s)
run() {  # variant formulations override...
  V=$1; FS=$2; shift 2
  LEFT=$((BUDGET - ($(date +%s) - START))); [ $LEFT -le 5 ] && return
  OV=""; for o in "$@"; do OV="$OV --override $o"; done
  python scripts/run_sweep.py --run-type robustness --variant $V --formulations $FS --kappas 1,80,2000 --seeds 0,1,2,3,4 --max-seconds $LEFT $OV 2>/dev/null | grep -E "done:|max-seconds|failed" | sed "s/^/$V: /"
}
ALL=F1,F2,F3,F4,F5,F6
run width16 $ALL width=16
run depth2 $ALL depth=2
run n32 $ALL n_interior=32
run lr3e-3 $ALL lr_init=3e-3
run pm_width22 F2,F4,F6 width=22
run band16 $ALL n_extra_band=16
run depth5 $ALL depth=5
run n512 $ALL n_interior=512
run width64 $ALL width=64
run steps12000 $ALL steps=12000
