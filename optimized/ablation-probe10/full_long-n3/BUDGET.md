# Budget — `full_long-n3`

Generated 2026-09-14 21:56:17 EDT on `dutchman`,
fleet-shadow-art @ `7bdef82`.

## Optimizer

| setting | value |
|---|---|
| popsize | 128 |
| phase1_iters (per robot, 6-D) | 64 |
| phase2_iters (per robot, 6-D) | 48 |
| final_iters (joint, 18-D) | 32 |
| adaptive_final | False |
| floor / collision / self-collision penalty | 40.0 / 400.0 / 200.0 |
| n_workers per process | 10 requested / 10 effective |

## Renders per solve (derived)

| stage | renders |
|---|---|
| Hungarian pre-assignment | 72 |
| init sampling | 48 |
| phase 1 — forward greedy | 24,576 |
| phase 2 — backward pass | 18,432 |
| final — joint refinement (32 iters) | 4,096 |
| FD refinement | ~180 |
| **total** | **~47,404** |

## Sampling

- 3 independent solves per target, seeds `0…2`.
- Targets finishing below **IoU 0.5** get **0 extra** solves
  (seeds `3…2`), all of them, not stopping at the first
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
| target fit (similarity transform) | False |

## Scale

0 targets × 3 runs ≈ **0.0M renders**
before extras.

Subsets: letters_upper, digits
