#!/usr/bin/env bash
# Ablation with the zone assignment and aimed start removed from the system
# (use_hungarian=false, which also disables the aimed start). Same ten glyphs,
# budget and seeds as ablation_probe.sh; reuses its flat / fwd_joint /
# full_no_assign rows (identical configurations) and adds the rest.
#
#   bash scripts/ablation_probe_nz.sh
# Output: optimized/ablation-probe10/<variant>-n<N>/...   (nz_* variants)
set -uo pipefail
BENCH="${BENCH:-$HOME/dev/umbra-bench-grounded}"
REPO="${REPO:-$HOME/dev/fleet-shadow-art}"
PY="${PY:-$HOME/miniconda3/envs/fleet-shadow/bin/python}"
RUNS="${RUNS:-5}"
OUT_SUB="ablation-probe10"
LOGS="$BENCH/logs/$OUT_SUB"; mkdir -p "$LOGS"
export MUJOCO_GL=egl
TDIR="$BENCH/targets_ablation-probe10"; [ -d "$TDIR" ] || { echo "missing $TDIR"; exit 1; }

BIG="--popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --no-adaptive-final"
NZ="--cfg use_hungarian=false --cfg use_voronoi_warmstart=false"
declare -A V
V[nz_plus_backward]="$NZ --cfg use_icp_restarts=false --cfg fd_iters=0"
V[nz_plus_icp]="$NZ --cfg fd_iters=0"
V[nz_full]="$NZ"
V[nz_full_no_backward]="$NZ --cfg phase2_iters=0"
V[nz_full_no_icp]="$NZ --cfg use_icp_restarts=false"
V[nz_full_no_fd]="$NZ --cfg fd_iters=0"
V[nz_full_no_manifold]="$NZ --cfg use_manifold_search=false"

run() {  # variant N [extra]
  local v=$1 N=$2; shift 2
  local out="$BENCH/optimized/$OUT_SUB/${v}-n${N}"
  if [ "$(find "$out" -name results.json 2>/dev/null | wc -l)" -ge 10 ]; then
    echo "=== $v N=$N already complete ==="; return; fi
  echo "=== $v N=$N -> $out ($(date +%F\ %T)) ==="
  "$PY" "$BENCH/scripts/run_base_optimizer.py" --bench "$BENCH" --repo "$REPO" \
    --targets-dir targets_ablation-probe10 --subsets letters_upper digits --out "$out" \
    --runs "$RUNS" --n-robots "$N" --n-workers 10 "$@" ${V[$v]} > "$LOGS/${v}_n${N}.log" 2>&1
  echo "=== $v N=$N done ($(date +%F\ %T)): $(find "$out" -name results.json | wc -l)/10 ==="
}
for v in nz_plus_backward nz_plus_icp nz_full_no_backward nz_full_no_icp nz_full_no_fd; do run $v 3 $BIG; done
for v in nz_full nz_full_no_fd nz_full_no_icp nz_full_no_manifold; do run $v 5 $BIG; done
# sequence budget, 3 seeds: where restarts fire
SEQB="--popsize 128 --phase1-iters 64 --phase2-iters 48 --final-iters 32 --no-adaptive-final"
V[nz_full_long]="$NZ"; V[nz_full_long_no_icp]="$NZ --cfg use_icp_restarts=false"
RUNS=3 run nz_full_long 3 $SEQB
RUNS=3 run nz_full_long_no_icp 3 $SEQB
echo "NZ ABLATION DONE $(date +%F\ %T)"
