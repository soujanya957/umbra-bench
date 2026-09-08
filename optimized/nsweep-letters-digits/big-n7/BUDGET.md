# Budget — `big-n7`

Generated 2026-09-04 15:25:26 EDT on `dutchman`,
fleet-shadow-art @ `7bdef82`.

## Optimizer

| setting | value |
|---|---|
| popsize | 48 |
| phase1_iters (per robot, 6-D) | 16 |
| phase2_iters (per robot, 6-D) | 16 |
| final_iters (joint, 42-D) | 30 |
| adaptive_final | False |
| floor / collision / self-collision penalty | 40.0 / 400.0 / 200.0 |
| n_workers per process | 10 requested / 10 effective |

## Renders per solve (derived)

| stage | renders |
|---|---|
| Hungarian pre-assignment | 392 |
| init sampling | 112 |
| phase 1 — forward greedy | 5,376 |
| phase 2 — backward pass | 5,376 |
| final — joint refinement (30 iters) | 1,440 |
| FD refinement | ~180 |
| **total** | **~12,876** |

## Sampling

- 5 independent solves per target, seeds `0…4`.
- Targets finishing below **IoU 0.5** get **0 extra** solves
  (seeds `5…4`), all of them, not stopping at the first
  to clear the bar. `results.json` marks these with `"extra": true`.
- Reported statistic is best-of-N. Within-target seed spread on this rig is
  σ ≈ 0.022 IoU, so a single solve is a sample, not a measurement.

## Rig

| setting | value |
|---|---|
| robots | 7 × SO-101 |
| arm gap | 0.2 m |
| light-to-front / back-to-wall | None / None (None = default) |
| render size | 128 px |
| target deformation (free-form warp) | False |
| target fit (similarity transform) | False |

## Scale

0 targets × 5 runs ≈ **0.0M renders**
before extras.

Subsets: letters_upper, letters_lower
