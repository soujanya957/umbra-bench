# Budget — `big-budget`

Generated 2026-08-15 10:54:28 EDT on `dutchman`,
fleet-shadow-art @ `d964173`.

## Optimizer

| setting | value |
|---|---|
| popsize | 48 |
| phase1_iters (per robot, 6-D) | 16 |
| phase2_iters (per robot, 6-D) | 16 |
| final_iters (joint, 18-D) | 30 |
| adaptive_final | False |
| floor / collision / self-collision penalty | 40.0 / 400.0 / 200.0 |
| n_workers per process | 2 |

## Renders per solve (derived)

| stage | renders |
|---|---|
| Hungarian pre-assignment | 72 |
| init sampling | 48 |
| phase 1 — forward greedy | 2,304 |
| phase 2 — backward pass | 2,304 |
| final — joint refinement (30 iters) | 1,440 |
| FD refinement | ~180 |
| **total** | **~6,348** |

## Sampling

- 10 independent solves per target, seeds `0…9`.
- Targets finishing below **IoU 0.5** get **5 extra** solves
  (seeds `10…14`), all of them, not stopping at the first
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
| target deformation | False |

## Scale

546 targets × 10 runs ≈ **34.7M renders**
before extras.

Subsets: abstract, animals, digits, figures, hand_shadow, letters_lower, letters_upper, objects, vehicles
