# Budget — `n1`

Generated 2026-08-28 16:43:57 EDT on `dutchman`,
fleet-shadow-art @ `b536755`.

## Optimizer

| setting | value |
|---|---|
| popsize | 32 |
| phase1_iters (per robot, 6-D) | 8 |
| phase2_iters (per robot, 6-D) | 8 |
| final_iters (joint, 6-D) | 10 |
| adaptive_final | True |
| floor / collision / self-collision penalty | 40.0 / 400.0 / 200.0 |
| n_workers per process | 3 |

## Renders per solve (derived)

| stage | renders |
|---|---|
| Hungarian pre-assignment | 8 |
| init sampling | 16 |
| phase 1 — forward greedy | 256 |
| phase 2 — backward pass | 0 |
| final — joint refinement (12 iters) | 384 |
| FD refinement | ~180 |
| **total** | **~844** |

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
| robots | 1 × SO-101 |
| arm gap | 0.2 m |
| light-to-front / back-to-wall | None / None (None = default) |
| render size | 128 px |
| target deformation (free-form warp) | False |
| target fit (similarity transform) | {'scale_range': [0.35, 1.6], 'max_shift_frac': 0.22, 'scale_penalty': 0.0, 'n_scales': 14, 'n_shifts': 15, 'reach_samples': 300} |

## Scale

546 targets × 3 runs ≈ **1.4M renders**
before extras.

Subsets: abstract, animals, digits, figures, hand_shadow, letters_lower, letters_upper, objects, vehicles
