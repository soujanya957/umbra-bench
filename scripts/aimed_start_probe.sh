#!/usr/bin/env bash
# Where does the zone-aimed start pay? It is a cold-start device (taken only
# when it beats the current start pose, forward sweep only), so its effect
# should grow as the budget shrinks and as the fleet grows. Full solver with
# and without it, ten probe glyphs, best-of-RUNS seeds, at three budgets and
# two fleet sizes.
#
#   bash scripts/aimed_start_probe.sh
#
# Output: optimized/aimed-start-probe10/<variant>-<budget>-n<N>/...
set -uo pipefail
BENCH="${BENCH:-$HOME/dev/umbra-bench-grounded}"
REPO="${REPO:-$HOME/dev/fleet-shadow-art}"
PY="${PY:-$HOME/miniconda3/envs/fleet-shadow/bin/python}"
RUNS="${RUNS:-5}"
OUT_SUB="aimed-start-probe10"
LOGS="$BENCH/logs/$OUT_SUB"; mkdir -p "$LOGS"
export MUJOCO_GL=egl

TDIR="$BENCH/targets_ablation-probe10"   # same ten glyphs as the ablation
[ -d "$TDIR" ] || { echo "missing $TDIR (run ablation_probe.sh first)"; exit 1; }

declare -A B
B[tiny]="--popsize 16 --phase1-iters 4 --phase2-iters 4 --final-iters 6 --no-adaptive-final"
B[small]="--popsize 32 --phase1-iters 8 --phase2-iters 8 --final-iters 10 --no-adaptive-final"
B[big]="--popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --no-adaptive-final"
declare -A V
V[full]=""
V[no_aimed]="--cfg use_voronoi_warmstart=false"

for N in 3 5; do
  for b in tiny small big; do
    for v in full no_aimed; do
      # big-n3 and big-n5 already exist in ablation-probe10 (full, full_no_voronoi at n3)
      [ "$b" = big ] && [ "$N" = 3 ] && continue
      out="$BENCH/optimized/$OUT_SUB/${v}-${b}-n${N}"
      if [ "$(find "$out" -name results.json 2>/dev/null | wc -l)" -ge 10 ]; then
        echo "=== $v $b N=$N already complete, skipping ==="; continue
      fi
      echo "=== $v $b N=$N -> $out ($(date +%F\ %T)) ==="
      "$PY" "$BENCH/scripts/run_base_optimizer.py" \
        --bench "$BENCH" --repo "$REPO" --targets-dir "targets_ablation-probe10" \
        --subsets letters_upper digits --out "$out" \
        --runs "$RUNS" --n-robots "$N" --n-workers 10 \
        ${B[$b]} ${V[$v]} > "$LOGS/${v}_${b}_n${N}.log" 2>&1
      echo "=== $v $b N=$N done ($(date +%F\ %T)): $(find "$out" -name results.json | wc -l)/10 ==="
    done
  done
done
echo "AIMED-START DONE $(date +%F\ %T)"
