#!/usr/bin/env bash
# Base sweep for the paper charts: Ivy's grounded letters+digits,
# {small,big} budget x {3,5} robots, 5 runs each.
#
# Writes ONLY to optimized/letters-digits-grounded/ -- never into
# optimized/{small,big}-budget-grounded/, which are Ivy's runs. The driver
# resumes into an existing --out, so a shared folder would silently skip
# or overwrite her targets.
set -uo pipefail
BENCH="$HOME/dev/umbra-bench-grounded"
REPO="${REPO:-$HOME/dev/fleet-shadow-art}"
PY="${PY:-$HOME/miniconda3/envs/fleet-shadow/bin/python}"
SHARDS="${SHARDS:-6}"
WORKERS="${WORKERS:-2}"
RUNS="${RUNS:-5}"
SUBSETS="digits letters_upper letters_lower"
LOGS="$BENCH/logs/letters-digits"
mkdir -p "$LOGS"
export MUJOCO_GL=egl

SMALL="--popsize 32 --phase1-iters 8  --phase2-iters 8  --final-iters 10"
BIG="--popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --no-adaptive-final"

run_condition () {
  local name="$1"; local n="$2"; shift 2
  local out="$BENCH/optimized/letters-digits-grounded/$name"
  echo "=== $name (n=$n) -> $out  ($(date +%F\ %T)) ==="
  for s in $(seq 0 $((SHARDS - 1))); do
    "$PY" "$BENCH/scripts/run_base_optimizer.py" \
      --bench "$BENCH" --repo "$REPO" --targets-dir targets_grounded \
      --subsets $SUBSETS --out "$out" \
      --runs "$RUNS" --n-robots "$n" --n-workers "$WORKERS" \
      --shard "$s" --num-shards "$SHARDS" "$@" \
      > "$LOGS/${name}_shard${s}.log" 2>&1 &
  done
  wait
  echo "=== $name done ($(date +%F\ %T)): $(find "$out" -name results.json | wc -l)/186 targets ==="
}

run_condition small-n3 3 $SMALL
run_condition big-n3   3 $BIG
run_condition small-n5 5 $SMALL
run_condition big-n5   5 $BIG
echo "BASE SWEEP DONE $(date +%F\ %T)"
