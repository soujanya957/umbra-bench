#!/usr/bin/env bash
# Component ablation on the ten probe glyphs (same set as nsweep-probe10, same
# large budget), best-of-RUNS seeds per glyph. Two ladders at N=3 — build-up
# from a monolithic CMA-ES, and leave-one-out from the full solver — plus a
# short list at N=5 where the manifold joint stage is active.
#
#   bash scripts/ablation_probe.sh              # everything
#   NS="3" bash scripts/ablation_probe.sh       # N=3 ladders only
#   RUNS=3 bash scripts/ablation_probe.sh       # quicker pilot
#
# Output: optimized/ablation-probe10/<variant>-n<N>/<subset>/<stem>/results.json
# Summarise with: python3 scripts/collect_ablation.py optimized/ablation-probe10
set -uo pipefail
BENCH="${BENCH:-$HOME/dev/umbra-bench-grounded}"
REPO="${REPO:-$HOME/dev/fleet-shadow-art}"
PY="${PY:-$HOME/miniconda3/envs/fleet-shadow/bin/python}"
RUNS="${RUNS:-5}"
NS="${NS:-3 5}"
OUT_SUB="${OUT_SUB:-ablation-probe10}"
LOGS="$BENCH/logs/$OUT_SUB"; mkdir -p "$LOGS"
export MUJOCO_GL=egl

# Same ten glyphs as nsweep-probe10.
GLYPHS="letters_upper/E_dejavusans-bold letters_upper/H_dejavusans-bold \
letters_upper/K_dejavusans-bold letters_upper/U_dejavusans-bold \
letters_upper/W_dejavusans-bold digits/2_dejavusans-bold digits/3_dejavusans-bold \
digits/4_dejavusans-bold digits/7_dejavusans-bold digits/9_dejavusans-bold"
TDIR="$BENCH/targets_${OUT_SUB}"
rm -rf "$TDIR"; mkdir -p "$TDIR/letters_upper" "$TDIR/digits"
for spec in $GLYPHS; do
  sub="${spec%%/*}"; stem="${spec#*/}"
  cp "$BENCH/targets_grounded/$sub/$stem.png" "$TDIR/$sub/$stem.png" || exit 1
done

# Large budget, as in nsweep_letters_digits.sh.
BIG="--popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --no-adaptive-final"

# variant name → extra flags. Build-up rows switch components ON one at a time;
# leave-one-out rows switch one OFF from the full solver.
declare -A V
V[flat]="--flat"
V[fwd_joint]="--cfg use_hungarian=false --cfg use_voronoi_warmstart=false --cfg phase2_iters=0 --cfg use_icp_restarts=false --cfg fd_iters=0"
V[plus_assign]="--cfg use_voronoi_warmstart=false --cfg phase2_iters=0 --cfg use_icp_restarts=false --cfg fd_iters=0"
V[plus_voronoi]="--cfg phase2_iters=0 --cfg use_icp_restarts=false --cfg fd_iters=0"
V[plus_backward]="--cfg use_icp_restarts=false --cfg fd_iters=0"
V[plus_icp]="--cfg fd_iters=0"
V[full]=""
V[full_no_assign]="--cfg use_hungarian=false"
V[full_no_voronoi]="--cfg use_voronoi_warmstart=false"
V[full_no_backward]="--cfg phase2_iters=0"
V[full_no_icp]="--cfg use_icp_restarts=false"
V[full_no_fd]="--cfg fd_iters=0"
V[full_no_manifold]="--cfg use_manifold_search=false"   # only meaningful at N>=5

LADDER_N3="flat fwd_joint plus_assign plus_voronoi plus_backward plus_icp full full_no_assign full_no_voronoi full_no_backward full_no_icp full_no_fd"
LADDER_N5="flat full full_no_manifold full_no_fd full_no_icp"

for N in $NS; do
  [ "$N" = 3 ] && LIST="$LADDER_N3" || LIST="$LADDER_N5"
  for v in $LIST; do
    out="$BENCH/optimized/$OUT_SUB/${v}-n${N}"
    if [ "$(find "$out" -name results.json 2>/dev/null | wc -l)" -ge 10 ]; then
      echo "=== $v N=$N already complete, skipping ==="; continue
    fi
    echo "=== $v N=$N -> $out ($(date +%F\ %T)) ==="
    "$PY" "$BENCH/scripts/run_base_optimizer.py" \
      --bench "$BENCH" --repo "$REPO" --targets-dir "targets_${OUT_SUB}" \
      --subsets letters_upper digits --out "$out" \
      --runs "$RUNS" --n-robots "$N" --n-workers 10 \
      $BIG ${V[$v]} > "$LOGS/${v}_n${N}.log" 2>&1
    echo "=== $v N=$N done ($(date +%F\ %T)): $(find "$out" -name results.json | wc -l)/10 targets ==="
  done
done
echo "ABLATION DONE $(date +%F\ %T)"
