# Grounded run — state, comparisons, and the sequences track

Written 2026-09-02. This was planned as the overnight launch of
`small-budget-grounded`; on inspection **both grounded sweeps were already
complete** — 571/571 targets each, every `results.json` present and parseable,
all 24 shard logs ending in `DONE` (see §4). Nothing was launched. What follows
re-verifies the finished sweeps against the current 571-target tree, adds the
comparisons the plan called for, and records the sequences-track work
(`scripts/sequence_metrics.py`) done in the window the sweep was expected to
occupy.

Both `metrics_*-budget-grounded.csv` files were probed to confirm they were
scored with `--targets-dir targets_grounded` (recomputed IoU for a sample row
matches the CSV against `targets_grounded/` and not against `targets/`), so the
`ref=original` numbers below compare to the right tree.

## 1. `at_bound` — improved 4.4x, but not "near 0%"

| condition | at_bound | n |
| --- | --- | --- |
| grounded (all) | **13.7%** | 571 |
| grounded (paired) | 14.9% | 478 |
| centred, derived (paired) | 65.7% | 478 |

The plan predicted near-0%; the measured value is 13.7%. That is the same
result HANDOFF §4b recorded (14.2% on the then-542 targets) with the same open
question attached: the empirical reach-map sweep predicted 1.1%, so the
reach-map proxy is optimistic by ~10x and that gap is still unexplained. The
direction and magnitude of the improvement are real — 65.7% → 14.9% on the
paired 478 — but "near zero" should not be quoted.

Two definitional notes:

- The centred sweeps predate `adfda3a` (which records `fit.at_bound`), so their
  figure is derived here from the recorded `(scale, dx, dy)` using
  `target_fit.py`'s own rule: scale at either scale bound, or |dx| or |dy| at
  the ±28.16 px shift clamp. The often-quoted "60%" for the centred sweeps was
  the dy-clamp alone; the full rule gives 65.7%.
- The fit is computed deterministically before any solving, so the placement
  diagnostics are **identical across the two budgets** — the small- and
  big-budget grounded sweeps share the same 13.7% / clip stats below.

## 2. `clip_frac` — clipping is bounded and small

Over the 571 grounded fits: mean **0.90%**, median 0.89%, p90 1.81%,
max **2.00%** — the ceiling `--fit-min-retained 0.98` enforces — and **no
target clips more than 2%** of its area. 16.1% of fits clip nothing at all.

`touches_edge` reads 74.8% and is ignored, per HANDOFF §5: grounded targets
rest on the bottom row by construction, so touching the edge is the design,
not a defect. `clip_frac` is the number that carries information in this
condition.

## 3. IoU: shown vs original, grounded vs centred

Paired on the 478 target ids the conditions share (the 29 `teleop` and 64
`_v2` targets exist only in the grounded sweeps):

| budget | ref | grounded | centred | delta | better |
| --- | --- | --- | --- | --- | --- |
| small | `shown` | 0.7523 | 0.7345 | +0.0179 | 285/478 |
| small | `original` | **0.3504** | 0.1999 | **+0.1506** | **441/478** |
| big | `shown` | 0.7833 | 0.7677 | +0.0156 | 279/478 |
| big | `original` | **0.3483** | 0.2000 | **+0.1483** | 435/478 |

This reproduces HANDOFF §4b/§4c exactly. The gap between `shown` and
`original` is what placement costs: 0.535 centred vs 0.402 grounded (small
budget) — grounding narrows it by ~0.13 but does not close it; the fit still
moves the target (mean clip 0.9%, scale ≠ 1), so the two refs stay far apart
and neither should be quoted without the other. Means over all 571 grounded
targets: 0.7464 / 0.3519 (small), 0.7768 / 0.3500 (big).

Per-subset, small budget, ref=original, paired 478 — the effect tracks the
per-subset placement remainder (vehicles wanted to move up most and gains
most), which is what a placement effect should look like:

| subset | n | grounded | centred | delta | better |
| --- | --- | --- | --- | --- | --- |
| vehicles | 30 | 0.5261 | 0.1144 | +0.4117 | 30/30 |
| objects | 95 | 0.4076 | 0.1886 | +0.2190 | 89/95 |
| animals | 76 | 0.3914 | 0.1856 | +0.2058 | 69/76 |
| hand_shadow | 10 | 0.2825 | 0.1326 | +0.1499 | 10/10 |
| letters_lower | 77 | 0.2988 | 0.2010 | +0.0978 | 74/77 |
| abstract | 76 | 0.3391 | 0.2596 | +0.0795 | 65/76 |
| letters_upper | 74 | 0.2709 | 0.1918 | +0.0792 | 69/74 |
| digits | 30 | 0.2482 | 0.1916 | +0.0566 | 26/30 |
| figures | 10 | 0.4160 | 0.3604 | +0.0556 | 9/10 |

Not an IoU artefact: the topology metrics move the same way
(`betti_error` 0.391 grounded vs 0.485 centred, `cldice` 0.522 vs 0.299,
small budget, ref=original).

And the budget interaction replicates on the full 571 within the grounded
tree: 2.1x compute buys **+0.0304** on `ref=shown` (550/571 better) and
**−0.0019** on `ref=original` (279/571 — chance). Placement is worth ~5x a
budget doubling on fidelity to the authored shape; the budget is worth ~0 on it.

## 4. Failures and unfinished shards

None. All 12+12 shards ran to `DONE` (542 solves per budget in the shard
logs); the 29 `teleop` targets were solved in a follow-up pass, and both
sweeps verify at 571/571 `results.json`, all parseable, ids exactly matching
`targets_grounded/`. The only `.err` content across 24 shards is a benign
scipy k-means UserWarning (shard 4, both budgets). One stale zero-byte
`HEAD.lock` had to be cleared during this session — GitHub Desktop again, per
HANDOFF §3; no live git process held it.

## 5. The sequences track (done while the sweep was expected to run)

- `sequences/` committed: 119 PNGs, 13 sequences, all as LFS pointers
  (`git lfs ls-files` grew 15374 → 15493).
- `scripts/sequence_metrics.py` implements S1/S2/S3 from SEQUENCES.md §3:
  per-frame quality (every `metrics.py` metric, mean/min/max/std over frames),
  transitions (dq_max/dq_l2 degrees, infeasibility against
  `motion_planner.LARGE_Q_JUMP` read live from fleet-shadow-art — a real
  import when mujoco is present, an `ast` read of the same source when not),
  and sequence level (assignment_stability via Hungarian matching on the arms'
  joint vectors, loop_closure, total_path_length). The wrap transition is
  scored whenever `target_motion.loop` is true.
- Validated against the solved `spinning_star` clip, exactly:
  mean per-frame IoU **0.6679** (recorded *and* recomputed at native 256),
  per-transition dq_max **[160.9, 291.0, 116.6, 157.6]°**, worst step
  **291.0°**, `dq_infeasible_frac` **1.0** — now 5/5 rather than 4/4, because
  star_spin loops and the wrap (212.5°, also infeasible) is scored too.
  New finding from the same clip: `assignment_stability = 0.0` — the arms
  trade roles on **every** transition, which per-frame IoU cannot see.
- `sequences.jsonl` rebuilt in the eval env: `frame_attributes` now filled,
  33 attributes per frame.
