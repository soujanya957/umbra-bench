# Budget — `small-budget-distort` (sequence axis)

Generated 2026-08-29 11:08:34 EDT on `dutchman`,
fleet-shadow-art @ `b536755`.

## Optimizer

| setting | value |
|---|---|
| popsize | 32 |
| phase1_iters (per robot, 6-D) | 8 |
| phase2_iters (per robot, 6-D) | 8 |
| final_iters (joint, 18-D) | 10 |
| adaptive_final | True |
| floor / collision / self-collision penalty | 40.0 / 400.0 / 200.0 |
| n_workers per process | 2 |

Identical to the static sweep of the same name, so a sequence frame and a
`targets/` sample cost the same search. The greedy phases are **not** skipped on
warm frames: the sequence solver in `run_sequence.py` does skip them once a
warm start looks good, which is right for production but would make these
numbers incomparable to the static ones.

## Chaining

| setting | value |
|---|---|
| solves per frame | 5 |
| prior for frame k | best run of frame k-1 (`x0` **and** `q_ref`) |
| winner selected by | `reachable` |
| temporal_weight (Ph1/Ph2) | 0.3 |
| final_temporal_weight | 0.0 (0 = curriculum: final pass free) |
| reachability_penalty | 100.0 (barrier at 1.2 rad/joint) |

Frame 00 has no prior and is solved cold — identical to a static target.

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

## Rig

| setting | value |
|---|---|
| robots | 3 × SO-101 |
| arm gap | 0.2 m |
| light-to-front / back-to-wall | None / None (None = default) |
| render size | 128 px |
| target fit (similarity transform) | False |
| target deformation (free-form warp) | {'grid': 4, 'max_disp_frac': 0.06, 'bending': 0.03, 'sequence_level': True} |

## Scale

15 sequences / 127 frames × 5 runs ≈ **1.41M renders**
