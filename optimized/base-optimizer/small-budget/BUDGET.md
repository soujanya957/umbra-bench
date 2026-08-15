# Budget — `small-budget`

Generated 2026-08-15 10:17:17 EDT on `dutchman`,
fleet-shadow-art @ `7bb5276`.

> Backfilled after the fact from this sweep's `results.json` files —
> the configuration below is what the solves actually recorded.
>
> `n_workers` varied across this sweep: 2 (243 targets), 4 (303 targets). It sets how wide the
> population evaluation fans out, not what the search explores, so the
> budget below applies to every target regardless.

## Optimizer

| setting | value |
|---|---|
| popsize | 32 |
| phase1_iters (per robot, 6-D) | 8 |
| phase2_iters (per robot, 6-D) | 8 |
| final_iters (joint, 18-D) | 10 |
| adaptive_final | True |
| floor / collision / self-collision penalty | 40.0 / 400.0 / 200.0 |
| n_workers per process | 4 |

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
- Targets finishing below **IoU 0.0** get **0 extra** solves
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
| target deformation | False |

## Scale

546 targets × 10 runs ≈ **12.1M renders**
before extras.

Subsets: abstract, animals, digits, figures, hand_shadow, letters_lower, letters_upper, objects, vehicles
