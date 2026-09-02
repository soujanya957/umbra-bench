# Overnight notes — 2026-09-02 (session umbra-bench-26, coordinating with umbra-bench-9f)

## Open question for the user (needs a decision, not urgent tonight)

**Is demo_01 going to be a rendered composite or a filmed rig?** It changes the
right `--fit-max-shift`:

- **Rendered**: take the larger fit and the better castability — the placement
  cost is recovered exactly in post (08_reassemble.py applies fit⁻¹; measured
  0.000→0.561, 0.187→0.825, within resampling of the fitted scores).
- **Filmed**: prefer the tight pass (0.22) even if it scores lower — the fit is
  one similarity per clip, i.e. one camera reframe (zoom+pan) per clip that
  someone has to physically make; scale 0.64 next to 1.0 is a visibly
  different camera distance.

The tight-pass A/B (queued) answers the score half either way.

## Decisions taken overnight (as project lead)

- Loop-wrap fix lands BEFORE the generated pass, single pass; the three
  archaeological runs (flower, stick_wave, star_spin) serve as the
  unconstrained-wrap baseline and will be preserved as source=optimizer_v0.
- Demo section quotes CLIP legibility (folded glyph set, authored ceiling);
  benchmark section keeps IoU primary. The two rank clips differently
  (scene_05_M: IoU 0.681, legibility ratio 0.000; scene_06_Y: IoU 0.747,
  ratio 1.500 — highest in the set).
- Sequence cards draw the SHOWN frames (static atlas's own rule); overlay
  agreement now matches measured vs-fitted IoU per card.
- The three I-clips have a zero CLIP ceiling by class design (I/l/1 fold →
  "digit 1" prompt misses a serif capital I): cards say "no ceiling", never 0.
  A folded-class prompt fix would change the benchmark's own numbers — left
  untouched, flagged here.

## Engineering notes for HANDOFF (end-of-night)

- Scripted replacements: assert count(old)==1 before replace (substring
  false-positive bit both sessions tonight: sagg.update contains agg.update;
  log-at-launch counted as finished).
- `| tail -1` masks exit codes; use pipefail in verification chains.
- run_sequence.py summaries: iou_original is declared but never populated —
  sequence_metrics.py's ref=original replay is the only vs-authored source.

## Wide pass complete (13/13, fit-max-shift 0.45, --prior, temporal chain)

| clip | vs fitted | vs authored | dq max | infeas | legib. ratio |
|---|---|---|---|---|---|
| scene_01_I | 0.667 | 0.260 | 36° | 0% | -- |
| scene_02_F | 0.580 | 0.023 | 34° | 0% | -- |
| scene_03_F | 0.669 | 0.341 | 57° | 0% | -- |
| scene_03_I | 0.608 | 0.243 | 33° | 0% | -- |
| scene_04_M | 0.719 | 0.091 | 32° | 0% | 0.21 |
| scene_05_I | 0.745 | 0.385 | 0° | 0% | -- |
| scene_05_M | 0.679 | 0.085 | 22° | 0% | 0.00 |
| scene_06_A | 0.825 | 0.187 | 36° | 0% | 1.00 |
| scene_06_F | 0.761 | 0.316 | 17° | 0% | 0.80 |
| scene_06_I | 0.428 | 0.000 | 68° | 0% | -- |
| scene_06_L | 0.544 | 0.000 | 65° | 0% | 0.00 |
| scene_06_M | 0.787 | 0.116 | 12° | 0% | 0.20 |
| scene_06_Y | 0.748 | 0.290 | 40° | 0% | 1.50 |

All 13 clips: every transition feasible, zero arm swaps. Legibility CSV predates
the last few solves, so some clips lack a row until stage 09 re-runs.

## The headline A/B (star_spin, new config vs archaeological independent solve)

| | independent (v0) | temporal chain + loop-aware |
|---|---|---|
| wrap dq | 212.5° | 63.8° (feasible) |
| infeasible transitions | 5/5 | 0/5 |
| arm swaps | 5/5 | 0/5 |
| mean frame IoU (vs fitted) | 0.668 | 0.646 |

Two IoU points buy the clip's entire playability. loop_anchor recorded
{attempted: false, "wrap already reachable"} — the guard checked and declined,
because the chain alone landed a feasible wrap. Both solves are on the board
(source=optimizer / optimizer_v0).
