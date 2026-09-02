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

## The headline A/B (star_spin) — with its attribution stated carefully

| | archaeological config (v0) | current config |
|---|---|---|
| wrap dq | 212.5° | 63.8° (feasible) |
| infeasible transitions | 5/5 | 0/5 |
| arm swaps | 5/5 | 0/5 |
| mean frame IoU (vs fitted) | 0.668 | 0.646 |

Two IoU points buy the clip's entire playability — but the credit belongs to
the WHOLE current configuration (clip fit + chained warm starts + penalties),
not to --loop-close: loop_anchor recorded {attempted: false, "wrap already
reachable"} — the guard checked and declined because there was nothing to fix.
The old wrap failure was a symptom of the old config, and the v0 baseline
measures a config-vs-config difference, not the flag's effect. What
--loop-close is worth stays open until a looping clip actually triggers it;
if none of the remaining ten do, it is reported as an untriggered safeguard,
not a fix. The card shows loop-close status per solve so "did not fire" and
"was not run" cannot be conflated. (Framing per umbra-bench-9f.)

## loop-close's first real firing (flower)

The chained solve landed the wrap at 79.0° (out of bounds); the guard re-solved
the final frame: wrap 68.2° (inside), previous neighbour 39.9° (not worsened),
and that frame's IoU rose 0.6487 → 0.6498. Recorded attempted:true,
accepted:true with all four numbers. The old config's flower wrap (95.5°, its
only infeasible step) sits beside it as optimizer_v0. So the write-up has both
stories with evidence: star_spin (the config fixed the wrap; the guard declined)
and flower (the config fell short by 10°; the guard closed it for free).

Caveat that travels with flower (9f): it is feasible by a hair on TWO
transitions — closed wrap 68.2° (margin 0.55°) and an untouched forward step
68.4° (margin 0.35°) against the 68.75° bound, which is a planner model, not a
physical guarantee; sub-degree margins sit inside modelling error. "0/7
infeasible" is true and is the thinnest version of true — the card now shows
the worst step beside the count, flagged when within 2° of the bound. What the
flag cannot do: make a clip comfortable. That would need a margin-seeking
penalty rather than a hinge that fires past the bound — open solver item.

## Generated pass complete — full coverage (26/26), and what it found

- loop-close tally over 11 looping clips: 7 not-needed, 4 fired — 1 accepted
  (flower, 79.0→68.2°, free), 1 correctly refused (wiper: closing the wrap to
  68.7° would blow the previous step to 131.7° — the guard's design working),
  2 defeated (windmills: closure returned identical wrap distances).
- The acceptance seam (9f, from the logs): **optimize_staged accepts each
  phase on raw IoU, not the penalised loss** — windmill_n3's log shows a
  feasible pose at IoU 0.384 discarded for an infeasible one at 0.411.
  Evidence audit (strict adjacency, after a first regex pass overclaimed):
  **1 verified instance across 26 clips; windmill_n5 and reeds_n5
  undeterminable** — the Manifold path logs no dq for the discarded candidate.
  Honest characterisation: real, demonstrated once, rare (it only bites when
  feasibility genuinely costs IoU); the fix (6a9b285, lexicographic
  (reachable, IoU) at all three phase boundaries) is correct at n=1 — a rule
  that can discard a playable clip for 0.027 IoU is wrong regardless of
  frequency. Sober predictions on record: windmill_n3 changes materially
  under re-solve; the other fired clips may not, and that would not be the
  fix failing.
- planner_forecast in the summaries covers only forward steps (no wrap) — a
  summary-side gap; the atlas scorer computes dq from the joint columns and
  has counted the wrap from day one (star_spin validation: 5/5 with wrap).
- The wrap problem in the current config is confined to the rotor family
  (windmills 141.6°/76.4°) and wiper (144.6°); all nine other loops land the
  wrap inside the bound with no anchor needed — though often thinly (their
  wrap is usually the largest step: 63.8, 62.4, 65.0, 59.2...).
- wiper (fastest target, 0.21 step IoU) is the hardest clip: vs fitted 0.327,
  motion_excess −0.20 — the chain under-moves on fast content; frame IoU
  cannot see this, S2 can. One interior step infeasible by 0.03°.
- n_arms datapoint: reeds_n3 0.413 vs reeds_n5 0.412 — two extra arms bought
  nothing on the thin-stalk target.

## The three-way answer (static baseline scored, all 26 clips)

Mean over 26 clips, each condition scored on the problem it was given:

|                    | chained (run_sequence) | independent (image pipeline) |
|---|---|---|
| mean frame IoU     | **0.629**              | 0.564 |
| infeasible transitions | **2%** (rotor wraps + wiper only) | **97%** |
| assignment stability   | **100%**           | 28% |

The image-pipeline premise ("per-frame performance seems much better") did not
survive measurement: independent solves lose on frame quality too (19/26
clips), while their transitions are near-universally unplayable and the arms
trade roles constantly. Caveats stated honestly: (1) the frame-IoU column is
budget-confounded — the baseline ran the small budget (3 restarts) vs
run_sequence's full one — but the transition columns are not: the
archaeological star_spin at ~5x budget still wrapped at 291° with 5/5 swaps;
no budget makes independent solves temporally coherent. (2) demo rows compare
chained-on-fitted vs independent-on-authored placement. (3) wiper is the one
genuine trade: independent beats chained on frames (0.472 vs 0.327) because
the temporal chain under-moves on the fastest clip — the real cost of
chaining, measured, and it is one clip out of 26.

Verdict for the demo video: keep the sequence pipeline; the static sweep's
role is the benchmark lower bound, which it now serves on every card
(source=optimizer_static, animated plates included).

## The seam had a sixth site — and a lesson in the other direction

The first seam-fix verification came back byte-identical to the pre-fix run.
The observation was sound; my inference ("the new code never ran") was not —
log strings introduced by the fix prove it ran, chose the reachable pose at
the phase boundary ("taking the reachable pose on feasibility, not IoU"),
and was then reversed by a global-best override applied after every boundary
and tracked on IoU alone. Seeded CMA-ES made the reversal reproduce the old
run bit for bit. Fixed in daa5ad8: all five comparisons in optimize_staged
now use the same lexicographic (reachable, IoU) rule; inert when q_ref is
None. Lessons list, inverse entry: the previous five were checks that passed
for unrelated reasons; this one is a sound observation with an unsound
inference — "byte-identical means it did not run" skipped "it ran and was
overridden". Verification protocol upgraded both ways: process-version check
before launch (optimizer.__file__ + fix marker), byte-comparison after.

## The seam-fix verification, third run (daa5ad8) — valid, and the story completes

Forward frames byte-identical again (seeded determinism; the fix is inert
where feasibility never costs IoU — as designed). The closure differs, which
proves the new code ran: the 66.5° feasible-wrap pose now SURVIVES to the
return (the sixth site is dead), and the guard still rejects — closing the
wrap blows the previous transition to 111.3°. Predictions: wrap≈66 achieved
inside the candidate ✓; accepted:true ✗; avg_iou unchanged ✗; forward 0/7 ✓.

The complete characterisation of --loop-close, now measured: a single-frame
anchor can repair a LOCAL wrap overshoot (flower, 10° over, fixed free) and
provably cannot repair a GLOBAL winding (windmills: the violation is
conserved and relocates — the rotor's joint-space unwind must be distributed
across all frames, a per-clip phase shift, out of scope tonight). daa5ad8
remains correct and necessary: without it the candidate never even reached
the guard.
Instrument caveat (9f): hash-identical poses are diagnostic only alongside
the recorded reason — "the run did not change" and "the run changed and the
guard declined" produce the same hashes; the evidence for the latter lives
in loop_anchor, not in the poses.
