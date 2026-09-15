#!/usr/bin/env bash
# Follow-up to aimed_start_probe.sh: the aimed start competes with the Sobol
# init samples for finding an arm's zone, so starve the init (4 samples instead
# of 16) and add the largest fleet (N=7). Small budget only.
set -uo pipefail
BENCH="${BENCH:-$HOME/dev/umbra-bench-grounded}"
REPO="${REPO:-$HOME/dev/fleet-shadow-art}"
PY="${PY:-$HOME/miniconda3/envs/fleet-shadow/bin/python}"
RUNS="${RUNS:-5}"
OUT_SUB="aimed-start-probe10"
LOGS="$BENCH/logs/$OUT_SUB"; mkdir -p "$LOGS"
export MUJOCO_GL=egl
SMALL="--popsize 32 --phase1-iters 8 --phase2-iters 8 --final-iters 10 --no-adaptive-final"
declare -A V
V[full]=""
V[no_aimed]="--cfg use_voronoi_warmstart=false"
run() {  # tag N extra
  local tag=$1 N=$2; shift 2
  for v in full no_aimed; do
    out="$BENCH/optimized/$OUT_SUB/${v}-${tag}-n${N}"
    if [ "$(find "$out" -name results.json 2>/dev/null | wc -l)" -ge 10 ]; then
      echo "=== $v $tag N=$N already complete, skipping ==="; continue
    fi
    echo "=== $v $tag N=$N -> $out ($(date +%F\ %T)) ==="
    "$PY" "$BENCH/scripts/run_base_optimizer.py" \
      --bench "$BENCH" --repo "$REPO" --targets-dir "targets_ablation-probe10" \
      --subsets letters_upper digits --out "$out" \
      --runs "$RUNS" --n-robots "$N" --n-workers 10 $SMALL "$@" ${V[$v]} \
      > "$LOGS/${v}_${tag}_n${N}.log" 2>&1
    echo "=== $v $tag N=$N done ($(date +%F\ %T)): $(find "$out" -name results.json | wc -l)/10 ==="
  done
}
run small-init4 3 --cfg init_samples=4
run small-init4 5 --cfg init_samples=4
run small 7
echo "AIMED-START-2 DONE $(date +%F\ %T)"
