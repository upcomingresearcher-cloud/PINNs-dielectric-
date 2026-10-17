#!/bin/sh
# Phase 13 remedy driver (resumable). Usage: sh scripts/run_remedies.sh BUDGET_SECONDS
BUDGET=${1:-240}; START=$(date +%s)
run() {  # problem variant formulations kappas override...
  P=$1; V=$2; FS=$3; KS=$4; shift 4
  NF=$(echo $FS | tr ',' '\n' | wc -l); NK=$(echo $KS | tr ',' '\n' | wc -l); WANT=$((NF * NK * 5))
  HAVE=$(grep -c -- "^$P-.*-remedy-$V,remedy,complete," results/raw/runs_2d.csv); [ "$HAVE" -ge "$WANT" ] && return
  LEFT=$((BUDGET - ($(date +%s) - START))); [ $LEFT -le 40 ] && return
  OV=""; for o in "$@"; do OV="$OV --override $o"; done
  python scripts/run_sweep2d.py --problem $P --run-type remedy --variant $V --formulations $FS --kappas $KS --seeds 0,1,2,3,4 --max-seconds $((LEFT - 35)) $OV 2>/dev/null | grep -E "new runs|failed" | sed "s/^/$P $V: /"
}
run P2 R_lbfgs F2,F3,F6 1,80,2000 optimiser=lbfgs
run P5 R_lbfgs F2,F3,F6 1,80,2000 optimiser=lbfgs
run P2 R_adam20k F2,F6 80 steps=20000
run P5 R_adam20k F2,F6 80 steps=20000
