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
