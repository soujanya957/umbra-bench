#!/usr/bin/env bash
# Four conditions over the sequence axis: {small, big} budget x {no-distort, distort}.
#
# Conditions run one at a time; shards run in parallel *within* a condition. Two
# reasons, both learned the hard way: MuJoCo's EGL context does not survive being
# forked, so every shard is its own process with its own renderer, and running all
# four conditions at once oversubscribes the box and makes the per-solve timings
# incomparable to the static sweep they are supposed to be matched against.
#
# Sharding is by SEQUENCE, never by frame -- frames are serially dependent.
set -uo pipefail

BENCH="${BENCH:-$HOME/dev/umbra-bench}"
REPO="${REPO:-$HOME/dev/fleet-shadow-art}"
PY="${PY:-$HOME/miniconda3/envs/fleet-shadow/bin/python}"
SHARDS="${SHARDS:-6}"
WORKERS="${WORKERS:-2}"   # SHARDS * WORKERS should not exceed core count
RUNS="${RUNS:-5}"
LOGS="$BENCH/logs/sequences"
mkdir -p "$LOGS"

export MUJOCO_GL=egl

SMALL="--popsize 32 --phase1-iters 8  --phase2-iters 8  --final-iters 10"
BIG="--popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --no-adaptive-final"

run_condition () {
  local name="$1"; shift
  local out="$BENCH/optimized/anim-optimizer/$name"
  echo "=== $name -> $out  ($(date '+%F %T')) ==="
  for s in $(seq 0 $((SHARDS - 1))); do
    "$PY" "$BENCH/scripts/run_anim_optimizer.py" \
      --bench "$BENCH" --repo "$REPO" --out "$out" \
      --runs "$RUNS" --n-workers "$WORKERS" \
      --shard "$s" --num-shards "$SHARDS" "$@" \
      > "$LOGS/${name}_shard${s}.log" 2>&1 &
  done
  wait
  echo "=== $name done ($(date '+%F %T')): $(find "$out" -name results.json | wc -l)/15 sequences ==="
}

run_condition small-budget          $SMALL
run_condition big-budget            $BIG
run_condition small-budget-distort  $SMALL --distort
run_condition big-budget-distort    $BIG   --distort

echo "ALL DONE $(date '+%F %T')"
