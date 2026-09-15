# Budget — `small-budget-grounded`

Generated 2026-09-02 16:20:24 Eastern Daylight Time on `ivy`,
fleet-shadow-art @ `f2fe9af`.

## Optimizer

| setting | value |
|---|---|
| popsize | 32 |
| phase1_iters (per robot, 6-D) | 8 |
| phase2_iters (per robot, 6-D) | 8 |
| final_iters (joint, 18-D) | 10 |
| adaptive_final | True |
| floor / collision / self-collision penalty | 40.0 / 400.0 / 200.0 |
| n_workers per process | 0 requested / 1 effective |

## Renders per solve (derived)

| stage | renders |
|---|---|
| Hungarian pre-assignment | 72 |
| init sampling | 48 |
| phase 1 — forward greedy | 768 |
| phase 2 — backward pass | 768 |
| final — joint refinement (12 iters) | 384 |
| FD refinement | ~180 |
| **total** | **~2,220** |

## Sampling

- 10 independent solves per target, seeds `0…9`.
- Targets finishing below **IoU 0.5** get **0 extra** solves
  (seeds `10…9`), all of them, not stopping at the first
  to clear the bar. `results.json` marks these with `"extra": true`.
- Reported statistic is best-of-N. Within-target seed spread on this rig is
  σ ≈ 0.022 IoU, so a single solve is a sample, not a measurement.

## Rig

| setting | value |
|---|---|
| robots | 3 × SO-101 |
| arm gap | 0.2 m |
| light-to-front / back-to-wall | None / None (None = default) |
| render size | 128 px |
| target deformation (free-form warp) | False |
| target fit (similarity transform) | {'scale_range': [0.35, 1.6], 'max_shift_frac': 0.22, 'dy_range': 'symmetric', 'min_retained': 0.98, 'scale_penalty': 0.0, 'n_scales': 14, 'n_shifts': 15, 'reach_samples': 300} |

## Scale

60 targets × 10 runs ≈ **1.3M renders**
before extras.

Subsets: teleop

## Passes (append-only — the generator above overwrites itself per launch)

The header describes the LATEST launch; earlier passes into this directory:

1. main grounded sweep (2026-09-01, 542 targets), settings as above.
2. teleop follow-up (2026-09-01), the 29 v1 capture-targets.
3. teleop set2 (2026-09-02), the 31 new capture-targets — already-solved
   targets skipped, so "60 targets" above is the subset size, not this
   pass's work: 31 x 10 = 310 solves, zero targets under the extras bar.
