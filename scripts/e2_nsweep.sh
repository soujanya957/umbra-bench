#!/usr/bin/env bash
# e2_nsweep.sh -- fleet size against STATIC benchmark quality.
#
# The paper's N-sweep (IoU 0.334/0.545/0.619/0.651 for N=1,2,3,5) rests on
# 6 targets x 3 seeds. This runs the same sweep on the stratified bench40 split
# and adds N=7, so the capacity-saturation claim rests on 40 targets spanning
# the full stroke-width range rather than six hand-picked ones.
#
# Matched to optimized/small-budget-fitted: same budget, same --fit-target, so
# the N=3 arm is directly comparable to the 546-target reference run.
#
# 40 targets x 5 N x 3 seeds = 600 solves. 3 shards per N, sequential over N.
set -u
BENCH="$HOME/dev/umbra-bench"
PY="$HOME/miniconda3/envs/fleet-shadow/bin/python"
export MUJOCO_GL=egl
LOG="$BENCH/logs/e2_nsweep_$(date +%Y%m%d_%H%M).log"
mkdir -p "$BENCH/logs"

for N in 1 2 3 5 7; do
  OUT="$BENCH/optimized/nsweep-fitted/n${N}"
  echo "=== N=$N  $(date) -> $OUT ===" | tee -a "$LOG"
  for S in 0 1 2; do
    $PY "$BENCH/scripts/run_base_optimizer.py" \
        --only-file "$BENCH/splits/bench40.txt" \
        --n-robots "$N" --runs 3 --fit-target \
        --popsize 32 --phase1-iters 8 --phase2-iters 8 --final-iters 10 \
        --n-workers 3 --shard "$S" --num-shards 3 \
        --out "$OUT" >> "$LOG" 2>&1 &
  done
  wait
  echo "=== N=$N done $(date) ===" | tee -a "$LOG"
done
echo "=== e2 nsweep complete $(date) ===" | tee -a "$LOG"
