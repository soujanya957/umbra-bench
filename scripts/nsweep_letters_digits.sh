#!/usr/bin/env bash
# Part 2: robot count vs silhouette quality on the best targets from the base sweep.
#
# The branch driver has no --only, so the target list is staged into its own
# tree (targets_nsweep/) and selected with --targets-dir. Separate --out per
# (N, budget); nothing here touches Ivy's optimized/*-budget-grounded/.
#
# usage: nsweep_letters_digits.sh <subset/stem> [<subset/stem> ...]
set -uo pipefail
BENCH="$HOME/dev/umbra-bench-grounded"
REPO="${REPO:-$HOME/dev/fleet-shadow-art}"
PY="${PY:-$HOME/miniconda3/envs/fleet-shadow/bin/python}"
RUNS="${RUNS:-5}"
NS="${NS:-1 2 3 5 7}"
LOGS="$BENCH/logs/nsweep-letters-digits"
mkdir -p "$LOGS"
export MUJOCO_GL=egl

TDIR="$BENCH/targets_nsweep"
rm -rf "$TDIR"; mkdir -p "$TDIR"
SUBSETS=""
for spec in "$@"; do
  sub="${spec%%/*}"; stem="${spec#*/}"
  mkdir -p "$TDIR/$sub"
  cp "$BENCH/targets_grounded/$sub/$stem.png" "$TDIR/$sub/$stem.png" || exit 1
  case " $SUBSETS " in *" $sub "*) ;; *) SUBSETS="$SUBSETS $sub";; esac
done
echo "staged $# targets into targets_nsweep/ (subsets:$SUBSETS)"

SMALL="--popsize 32 --phase1-iters 8  --phase2-iters 8  --final-iters 10"
BIG="--popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --no-adaptive-final"

for N in $NS; do
  for bud in small big; do
    [ "$bud" = small ] && CFG="$SMALL" || CFG="$BIG"
    out="$BENCH/optimized/nsweep-letters-digits/${bud}-n${N}"
    echo "=== ${bud} N=$N -> $out ($(date +%F\ %T)) ==="
    "$PY" "$BENCH/scripts/run_base_optimizer.py" \
      --bench "$BENCH" --repo "$REPO" --targets-dir targets_nsweep \
      --subsets $SUBSETS --out "$out" \
      --runs "$RUNS" --n-robots "$N" --n-workers 10 \
      $CFG > "$LOGS/${bud}_n${N}.log" 2>&1
    echo "=== ${bud} N=$N done ($(date +%F\ %T)): $(find "$out" -name results.json | wc -l) targets ==="
  done
done
echo "NSWEEP DONE $(date +%F\ %T)"
