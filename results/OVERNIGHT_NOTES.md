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

## ⚠ Security-relevant event, for the user (both sessions raising it)

Around 04:48 a message appeared in umbra-bench-9f's transcript claiming to be
from this session and reporting seamfix_windmill_n5 results ("wrap 76.4 ->
61.7 accepted, 0/8 infeasible, avg_iou 0.4568"). Facts, all independently
verified by both sessions: (1) this session never sent it — the run had not
finished and no such CSV/score/board entry ever existed here; (2) the numbers
were fabricated — and matched 9f's publicly posted predictions exactly:
content shaped to be agreed with rather than checked; (3) the message was
structurally malformed (closed with `</parameter>` instead of the channel's
normal close — the only such message in 9f's transcript); (4) nothing from it
touched the repo, the board, or the report — both sessions' derive-from-file
discipline held, and 9f caught it by checking the run's own directory before
relaying. Treated as untrusted data on both sides; neither session acted on
it. We are not speculating about its origin beyond these observables — that
determination is yours. Cross-checking habit now formalised bilaterally:
any number in a cross-session message must be re-derived from the file it
claims to come from before use.

## loop-close, the final honest tally (after the n5 verification)

flower: confirmed (79.0→68.2°, free). windmill_n3: confirmed-as-correctly-
refused (global winding, one frame cannot absorb it). wiper: refused, same
conflict. windmill_n5: **inconclusive** — the whole solve diverged at frame
03 (daa5ad8 legitimately flipped an un-logged acceptance site there; f2fe9af
now instruments both silent sites), so its pre/post numbers are not an A/B,
and its closure no-op (incumbent returned, 3.7° over) is the n3 mechanism at
smaller scale, not new evidence. Its card shows a valid current-config solve
(iou_rep 0.4596) and nothing more.

Protocol upgrades from this exchange, both directions: byte-checks scan ALL
frames and report the FIRST differing index (a 3-frame sample proves a fork
exists but not where, and where is the only part that attributes it);
signature-matching alone is unsound once a fix can legitimately break
upstream determinism. Prediction scorecard kept honestly: 3 right, 5 wrong
(9f's own tally), registered before outcomes precisely so wanted results
could not pass unexamined — which they twice did not.

## Close-out (05:0x): the experimental program is complete

flower regression: clean — all 8 frames byte-identical, loop_anchor field-
for-field unchanged; the fix altered nothing where nothing needed altering.
Final prediction scorecard: 4 right, 5 wrong, all registered before outcomes.
A standing invariant found in passing: wrap_forecast.dq_max must equal
loop_anchor.dq_wrap_deg_closed (two independent computations of the same
quantity; 68.2° both ways on flower) — if they ever disagree, something
moved the pose after the closure.

Final --loop-close tally, 11 looping clips: 7 needed nothing, 1 closed and
accepted (flower), 2 correct refusals (windmill_n3, wiper), 1 inconclusive
(windmill_n5, forked). The acceptance-seam fix's one clean demonstration is
windmill_n3's wrap going from immovable to closable — that, not a closed
clip, is what daa5ad8 gets credited with.

Both repos: everything committed locally, nothing pushed. The anomalous
message(s) are documented above and with the user; neither session acted on
them and nothing from them touched either repo.

## User decision (morning): star_spin ships at scale 0.85

The bigger star is the quoted solve (source=optimizer, fit floor 0.85 —
at_bound deliberate). Frame IoU 0.724 vs the free fit's 0.646. The track's
declaration stays loop=true (the sequence IS a loop; single-pass is a usage
note about this cut, not a scoring override), so the board correctly counts
five transitions and reads **1/5 infeasible (20%)** — the wrap at 151.7°,
closure correctly refused. "Ships single-pass" means the declared wrap is
not performed in the video, stated beside the number, never instead of it.

The actual trade, in one sentence (9f's phrasing, adopted): the scale change
bought 0.078 IoU and spent most of the feasibility margin — worst forward
step 60.3° → 68.6° against the 68.75° bound, a 0.15° margin, the tightest of
the night and inside the planner model's error; fine for a rendered video,
worth physical testing before a filmed rig trusts it. Calibration fixture
for score_placement: its free optimum (0.638) LOST on frame IoU to a scale
it would never choose (0.6458 vs 0.7235) — the proxy picked the worse of
the two on the metric that matters, on the one clip measured both ways.
The 0.638 version is retired to git history (d9d90fd carries its numbers).

## User decision (morning): demo_01 is a RENDERED COMPOSITE — question closed

Size and position are adjustable in post, per the user. Consequences, all
already in place: the wide-fit (0.45) 13 demo solves are the ship set as
they stand; export goes through 08_reassemble.py (fit inverse at composite,
measured recovery); no per-clip camera constraints exist; the two
grazing-margin transitions (68.4°, 68.6° vs the 68.75° bound) need no
physical testing for the video. Asymmetric-dy fit tightening remains a
benchmark-side question only — nothing blocks the demo.

## Scale sweep epilogue: the verdict, the 4.3x example, and a hypothesis tested

Per-clip winners promoted (F all at scale 1.0; 04_M/06_M stay at the free
fit; 05_M takes 1.0 on a vs-fitted tie and wins 4.3x END TO END — wide
0.0847 vs s100 0.3640 vs-authored. Same clip, avg_iou and shipped result
pointing opposite ways: the sharpest single demonstration that vs-fitted
alone must never pick a shipped solve).

The glyph-orientation hypothesis, TESTED against on-disk data rather than
narrated (9f's demand, right one): proxy-chosen scale vs mean aspect_ratio
rho=-0.26 (p=0.19, n=26); scale-up gain vs aspect rho=-0.43 (p=0.34, n=7).
Direction suggestive, NOT significant, and star_spin (aspect 0.94, gain
+0.078) breaks the pure-aspect form outright. If vertical strokes eating the
reach band is the mechanism, its measurable carrier is not bounding-box
aspect — a vertical-ink-extent-vs-reach-band feature would be the next test,
and it needs the reach map, so it is future work, stated as such.

Also on record: lab_setup.json carries arm_gap 0.35 vs the 0.20 every solve
assumes — the section-C gate now has direct evidence of a geometry conflict,
with the user.

## Teleop v2 pipeline (user request, delivered while away) — c237e16

RealSense photo -> binary shadow in one command: scripts/teleop_pipeline.py.
AprilTag (36h11, ids 0-3) OUTER-frame rectification at pinned 560x468 — the
tags are IN the frame (nothing at the boundary lost) and excluded at
segmentation (statistics, background force, SAM2 negative prompts).
Hand seeds from points.json carried across frames via raw space, so v2
inherits every capture's human markup; SAM2-auto and Otsu as fallbacks.
Validated: 29/29 processed, 28/29 consistent with the shipped v1 masks, and
the single divergence is the redesign's purpose demonstrated —
letters_upperH's left stroke, amputated by v1's inner crop at the frame
edge, restored in v2 (+7%). Review sheet: Teleops/rectified2/_contact_sheet.png.
v1 outputs untouched; switching the atlas teleop payload to v2 frames is a
follow-up decision, not made unilaterally.

## Labeling loop for tag-rectified sets (post-restructure)

User wants to hand-label the new sets. Wired end to end and verified in the
browser: payload builder takes `--manifest` (v2 field set tolerated, `src`
recorded and shown in the view), clicks land in the 654x548 tag frame and are
used AS-IS via `<set>/points.json` (priority over the legacy v1-coordinate
transfer), the tally prints the exact per-set rerun command. Verified: a click
at (50%, 55%) exported as [327,300]; one-image rerun flipped candle_01 to
sam2-points.

Bug found by the test itself: a partial `--images` run rewrote the whole
set's labels.csv/manifest with only the processed subset (my test truncated
set2's labels.csv to one row -- restored from HEAD). Fixed with merge-by-stem
into the existing manifest; labels regenerate from the merged records.
Also discarded a jitter-only working-tree diff (a 14:53 rerun one minute after
dfed981: shape_frac +-0.0002, byte-different masks, nothing semantic).

Commits: 7299d2a (loop), d8a3b4b (atlas baked from set2 for the pass).

## set2 hand-relabel (user's pass)

User labeled all 31 in the dashboard and pasted the export; saved as
`Teleops/source/teleop_set2/points.json`, full rerun -> 31/31 sam2-points.
Reviewed every frac delta vs the auto masks: clear fixes at upperk_04
(0.070->0.171, fragments -> clean K), whale_01 (0.033->0.128, fin-only ->
full body), wrench_01 (spurious triangle gone), sneak_02 (left foot back).
Two went wrong and were corrected with extra points (marked as Claude's in
points.json "notes", scratch-tested first): upperp_01 leaked into the
penumbra (0.530 -> 0.123, six negatives), upperg_01 lost its upper strokes
to two low clicks (0.081 -> 0.138, three positives).

Second member of the truncation bug family, same day: with --images set,
output defaulting skipped the --teleop-root branch, so a two-capture
touch-up landed in legacy Teleops/rectified2 and labels.csv regenerated
from a fresh 2-record manifest. Fixed (defaults now follow the root
regardless of --images), stray dir removed, reran into the set tree.
No metrics rerun needed: set2 is still unlinked (31/31 unmatched), set1
untouched. fh_l1 is the CUDA env per user -- future SAM2 runs need not
crawl on CPU (saved to memory).

## Evening streams (user at dinner): studio, set1 reproduction, set2 import, stabilizer

Studio (committed 6fc3a1f): teleop view gains per-set chips synced to
Teleops/source/<set>/ (points namespaced per set -- set1 and v1 share stems
across DIFFERENT frames, so a shared namespace would have been a coordinate
bug); scripts/teleop_studio.py serves the dashboard plus /api/points (merge,
never delete, notes preserved -- verified byte-idempotent) and /api/rerun
(SAM2 -> merge -> payload -> atlas in one call; path traversal rejected;
verified live on candle_01). Tag footprints baked per item; browser
segmenter seeds them background, matching the pipeline's exclusion.

set1 "重新跑重新produce metric": full pipeline rerun -> 29/29 sam2-points,
ZERO frac drift vs committed masks (deterministic reproduction), then
compute_metrics --shadows --sources teleop --tag teleop-v2 -> 28 pairs,
iou 0.2995 / tc_iou 0.3315 / cldice 0.5142, byte-identical to the
committed CSV. Nothing changed and that is the finding: chain reproduces.

set2 import (v1 route, per Explore-agent archaeology): metadata backed up
(build_metadata writes none itself -- hazard H1), _teleop_index() now reads
all set manifests (was hardcoded to the retired v1 path -- would have
minted 31 rows with class "teleop"), masks copied prefix-stripped, padded
654x548 -> 512x429@(0,41), grounded (+50px mean shift). build_metadata
running. BUDGET.md snapshots taken before the sweeps (H4: the launcher
overwrites them). Big-budget sweep for the 31 launched pid 17880 with the
FULL explicit flag set from the snapshot; watcher pid 31140 chains the
small sweep on exit. ~4 solves/min observed.

Stabilizer (committed 5ca3c8c): scene_06_L's "bad" solve was geometry, not
solving -- 307px translation drift made the union fit 2.31x too small.
stabilize_sequence.py splits shape from trajectory exactly (verified
reconstruction), records trajectory_px for the compositor. The clip-safety
gate cleanly separates video-edited element moves (L, Y, triangle) from
base-anchored articulation (cheer, wiper, reeds -- correctly refused).
My earlier quick numbers for the two I clips were computed WITHOUT the
clipping check and are retracted: recentring them clips the glyph.

CLIP after import: needs a full rerun (class list for the teleop subset
changes -- ratios shift for the existing 29 too). fh_l1 has CUDA torch
(RTX 5070 Ti) but no open_clip; installing a package into the user's env
without them present is not my call. Queued CPU rerun after the sweeps
unless the user okays pip install open_clip_torch into fh_l1.

## Data run complete: set2 is fully in the benchmark

Both sweeps done (big 310 solves, small 310; zero extras triggered).
Big-budget best-of-10 on the 31 new capture-targets: mean 0.759, range
0.565 (upperh) to 0.897 (pencil) -- human-posed shadows are
in-distribution by construction and the solver confirms it. Metrics
recomputed for both grounded sweeps (1204 rows each, teleop shown = 60);
master_table rebuilt with the four-sweep whitelist (teleop-v2 excluded
from optimizer statistics -- hazard H3); CLIP full rerun (1806 images,
teleop now 60 samples over 50 classes, chance 0.02 -- every teleop ratio
shifted with the class list, including the original 29's, as predicted).
Atlas rebuilt and browser-verified: 602 cards, candle card carries the
full metric panel. BUDGET.md files got an append-only pass ledger.
Monitor false-complete en route (nested-quote PowerShell probe returned
empty == "process gone") -- re-armed on the plain PID; same bug family
as "check passed for a reason unrelated to what it was checking".

## L_stab: the decomposition wins on every axis

Same loss/budget as the tight L run (alpha 1.0 beta 0.3 gamma 0 delta 0,
128/128/256, pop 192, sigma0 0.4; fit bounds left at defaults and the
delta reported -- the original's non-default floor was the pathology).
Fit: scale 0.831 AT BOUND, uncastable 43.7%  ->  scale 1.312 free,
uncastable 2.5%. Solve: avg_iou 0.433 -> 0.719. Sequence metrics: the
tracked L solve moved dq_max 64.7 deg (a hair under the 68.75 bound,
arms chasing the traveling letter) with 245 deg total path; L_stab does
20.9 deg worst step, 59.7 deg total, stability 1.0, zero infeasible.
Because stab frames + trajectory_px == original frames exactly (verified
at derivation), per-frame IoU carries over to the composite unchanged --
the composited demo shows the SAME motion with a 1.6x bigger, 66%
better-matched letter. Compositor-side trajectory playback in
08_reassemble/10_compose_video is the remaining wiring.

## Y_stab and triangle_stab (user approved "可以重解"); GPU CLIP validated

Same config as each original (identical across the family). triangle:
0.649 -> 0.827 avg_iou, scale 0.735 -> 1.119. Y: 0.712 -> 0.812 at the
SAME fit scale (0.638) -- its original was bound-constrained and the win
is placement/castability, not size; scale-gain is sufficient but not
necessary for the decomposition to pay. All transitions comfortable
(worst 40.3 deg vs the 68.75 bound), stability 1.0, zero infeasible.
Three for three.

open_clip installed into fh_l1 by the user; GPU rerun of the full CLIP
eval reproduces the CPU summaries with zero delta across all 30
(subset, condition) rows -- future reruns cost seconds, no artifact
churn needed.

## CORRECTION: triangle_stab solve was 5/10 frames -- retracted and re-run

A head-truncated directory listing made me pass only f00-f04 to the
solver; the committed 0.649 -> 0.827 triangle comparison was 5 stab
frames against the original's 10 and is void. L (4/4) and Y (3/3) were
complete and stand. Full 10-frame triangle_stab re-launched; the ledger
keeps this entry so the wrong number cannot be quoted from the earlier
one. Lesson filed next to the 3-frame byte-sampling one: a listing
piped through head is a sample, not an inventory.

## Trajectory playback wired; demo video overwritten (user approved)

08_reassemble re-applies trajectory_px after the fit inverse (scaled
side/size into the pad frame); 10_compose lets a _stab reassembly
supersede its parent so the letter is not cast twice. The first wiring
had the SIGN inverted (trajectory_px is original->stabilised; restoring
means subtracting) -- caught by measuring canvas IoU vs the moving
authored letter BEFORE composing: 0.0/0.15/0.01/0.01 instead of a gain.
Fixed and re-measured: L mean 0.560 -> 0.729 on canvas (entering frame
0.291 -> 0.686), Y 0.741 -> 0.797. scene_06 and the whole cut
re-composed; FAMILY reads with L and Y at authored size and position.

## Full triangle_stab lands; stab variants live in the atlas

10-frame triangle_stab: 0.649 -> 0.825 (coincidentally near the retracted
5-frame number, but earned on the full clip), scale 0.735 -> 1.119.
Caveat stated on the card: two transitions run 68.5/67.6 deg against the
68.75 bound -- feasible, but with no margin. Sequences track now 29 rows
(26 + 3 _stab, loop declarations inherited from parents); payload joins
all three solves via --sequence-pinned ids; atlas rebuilt and the L_stab
card browser-verified (declared-loop provenance, full solve block).

## The demo folder consolidation + motion routing + the library framing

fleet-shadow-demo/ -> demo/, with outputs inside it (demo/out/{video,
reassembled, solve_logs, clip_legibility.csv, motion_routing.json});
08/09/10 defaults now anchor to the script's own directory, the payload
builder reads clip legibility from demo/out/, reassembled+solve_logs are
gitignored, video and the CSV ship. Full-chain smoke passed from the new
paths (08 -> 10 6/6 scenes -> payload with legibility joined -> atlas).

route_motion.py sorts elements static/translation/dynamic from measured
step IoU + an actual stabilisation derivation (edge-hugging clips fall
through to dynamic): the ad routes 4/2/7, the static lane alone replaces
46 per-frame solves with 4 holds; 08 gained --hold. Per the user, the
benchmark IS the static library (pull existing solves, add new shapes
via the set2 route) and assembly stays manual -- the README's assembly
spec documents exactly what each element hands you; 10 remains the
one-command automatic composite.

## Packaging + the zero-change deploy hand-off

demo/pack.py: one command -> demo/packages/<name>/ with frames, joints.csv,
meta.json (incl. solver config), video/, and choreo/<element>.json in
Shadow_robot_ui's unified clip envelope -- format read out of the UI's own
normalize_clip/list_choreographies (Explore recon), not guessed. Deploy =
copy choreo/*.json into fleet-shadow-art/choreographies/; NO code change
on that side needed, so no second session was spawned (user allowed it;
the cheaper path won and the skeleton stays untouched). Recon also
settled two long-standing wrong beliefs: lab_setup.json (0.35) is dead
code, and the solves' true default rig is gap 0.2 / light 1.0 / wall 2.4
(README gate rewritten; SR10x mapping is positional, calibration lives in
the driver, ports are operator settings). Scene block mapped into the UI
frame via optimize.py's documented transform so Play previews match.

## Labeling closes the library loop (user's design, implemented)

03_label_keypoints: `/` prompts a full label with prefix completion
against the library vocabulary (157 classes; free text allowed -- the
library suggests, the person decides), `.` records a per-(scene,label)
REUSE decision into keypoints.json; 04 skips reused objects entirely.
route_motion's static lane now points at demo/add_to_library.py (medoid
frame -> targets/demo/ -> normalize/ground/build_metadata with backup)
so every static shape becomes a named library row the atlas shows and
pack.py can pull; the router also prints existing same-class library
candidates, exact-case first. README opens with the ten-step
demo-to-real runbook. Heredoc escape lesson: \n through JSON->bash->
python triple-quote loses a backslash layer -- three broken string
literals repaired; single-line anchors + per-tag FAIL prints are now
the house style for scripted patches.

## demo_01 -> family_ad, the project rename

Mechanical key rename, no value touched: 15 sequence dirs (+ source.json
id/project/stabilized_from), 30 top-level run dirs across four families
(plain/s085/s100/tight), the static-sweep's per-sequence SUBdirs (the
top-level glob missed them -- caught because the payload shrank 130KB and
warned about an orphan row), 29 metrics CSVs renamed with in-file id/path
substitution, the static-baseline CSV (filename carries no project, only
contents -- second near-miss), reassembled dirs, video mp4s, legibility
CSV, docstring examples. sequences.jsonl rebuilt (15/0), payload back to
byte-parity with pre-rename size, atlas shows 15 family_ad cards and zero
demo_01. Historical solve logs keep the old name on purpose: they are
records of runs made under it. User directive noted: prioritize CUDA
(fh_l1) for heavy compute.

## demo design studio (user's ask: a UI for every step, per-video projects)

demo/studio.py + studio.html on :8478 -- pick a video (stem = project,
naming un-fumbleable), buttons for every stage running the exact
run_demo argument sets under the right interpreter per stage: eval env
for cv2/scipy stages, fh_l1 (CUDA, user's standing priority) for SAM2
segment and CLIP score, fleet-shadow for solves; ffmpeg borrowed from
fh_l1's Library/bin (user installed it there). Per-element solve buttons
are lane-aware (translation auto-runs the stabilizer then solves _stab;
static solves one frame) with the full explicit night-config flag set.
Library panel: 602 shapes, live search, thumbnails from
targets_grounded, click-to-copy id for the labeller's reuse prompt or
pack --library. One job at a time, live log tail. The labeller opens as
a native window on the machine (a matplotlib tool cannot live in a web
page, but the server can launch it). While building this the user was
already using the folder: pixar.mp4 arrived and was white-frame-trimmed
into pixar_01..04 with 00_trim's naming -- the flow is being used.

## Studio runs GPU steps as if conda-activated; legibility regenerated

env_for() builds the activation-equivalent PATH (Library/bin, Scripts,
CONDA_PREFIX) per interpreter -- the user's exact words: "you need to
activate fh_l1". Verified: torch cuda True + cv2 + open_clip resolve
under the constructed env. ffmpeg fact-check: the BINARY is in the
lerobot env; fh_l1 carries only the ffmpeg-python wrapper (my earlier
BOTH-OK misread corrected) -- FFMPEG_DIR points at lerobot and both
envs resolve ffmpeg/ffprobe through it. First real studio job: score
(09 on fh_l1 GPU) regenerated clip_legibility -- 340/340 rows, 71 value
changes confined to the F/M clips whose solves the scale sweep replaced
after the original scoring; L/Y unchanged because 09 scores the
non-stab reassembled dirs. The regenerated numbers are the current
truth; payload and atlas rebuilt on them.

## Labelling moves into the studio; the ffmpeg that never was

The user's 01 run died at ffprobe with empty stderr: exit 0xC0000139,
ENTRYPOINT_NOT_FOUND -- the lerobot conda ffmpeg has mismatched DLLs and
dies before main(), which is why which() passing meant nothing. Rather
than chase another install, 01_split_scenes gained a full OpenCV
backend (probe/luma/extract/scene-diffs/contact sheets) selected by
ffmpeg_works() -- an ffprobe that cannot RUN counts as absent. pixar.mp4
(real footage, 2 MB, distinct hash) split cleanly: 4 scenes, 54 frames
at sample 5, sheets included.

Root-caused a data bug the failure exposed: the studio set the active
project BEFORE 01 ran, so the failed pixar attempt left family_ad masks
in the workspace under a pixar state, and clicking "sequences" minted 13
pixar_scene_* rows whose frames are family_ad's to within 1-2 px of
06's smoothing jitter (byte-diff 81/157, pixel-diff ~4e-6 -- measured
before deleting). Removed, index rebuilt to 29. Fixes: activation now
happens only when scenes SUCCEEDS; numbered trims (_01, _02) strip to
one project (user's rule) and 01 gained --scene-start so a later trim
continues its predecessor's scene numbers; switching footage archives
keypoints.json to out/ and clears the single-project workspace.

Labelling now lives in the studio page itself: frame strip, canvas
clicks (left include / right exclude, undo, copy-prev, next-unlabelled),
per-scene reuse decisions, saving after every click into the SAME
keypoints.json schema 03 and 04 read -- verified by clicking through the
API and reading the file back. Clicking a library shape copies its id
AND sets it as the current label. Family labels archived at
out/keypoints_family_ad.json before the workspace swap.

## Video segmentation lands (SAM2 now, SAM3 one nod away)

The user asked for SAM3 + video segmentation. Two facts found by probing:
transformers 5.15.1 in fh_l1 already ships the FULL Sam3 suite
(Sam3TrackerVideoModel for click-tracking, Sam3VideoModel for
text-concept segmentation) -- only the facebook/sam3 checkpoint download
stands between us and it; and SAM2's own video predictor was already
installed and needs nothing. 04_video_segment.py: seeds from the same
keypoints.json (earliest labelled frame per (scene, object); later
labelled frames act as drift corrections; reuse-marked labels skipped),
propagates forward+backward, writes 04-compatible outputs so 06 onward
run unchanged. Smoke test on real pixar footage: ONE seed point on the
R tracked through 17/17 frames at 1.00 -- including the frames where
Luxo Jr lands on it -- in 1.3 s on the 5070 Ti. Labelling cost drops
from every-frame to one-frame-per-scene-per-object. Studio's segment
button now runs the video path; library thumbnails switched to the
SOLVED SHADOW (what the rig can cast -- the user's point) with target
fallback for unsolved rows, and a library click sets the label (class)
AND copies the id (reuse). --backend sam3 exits with a clear
not-yet-approved message until the download is okayed.

## The studio becomes per-project, and the board becomes assignments

Folder structure settled (user's design): raw videos stay flat in demo/,
everything derived lives in demo/projects/<name>/ -- scenes, keypoints,
masks, videos.json (which video produced which scene/frame numbers, so a
RE-split of the same video replaces its own scenes with identical
numbering and labels keyed on frame ids survive; a new trim continues).
Migration preserved every click: pixar's workspace moved whole, the
scene_05-08 duplicates (a re-split that predated replace semantics) were
byte-verified identical and their 4 labelled frames remapped by the
frame-id offset back onto scenes 01-04; family_ad's 92 labelled frames
restored from the archive into its own folder. The remaining 6
empty-object frames in pixar are the user's own undos, verified live.

Board redesign to the user's model: "clean" is gone as a concept
(segment chains 06 automatically); a mask GALLERY shows everything step
3 produced (clean beats raw, click jumps the labeller there); the
sequences table is now an ASSIGNMENT board -- per element choose solve
(launches the lane-aware solve) or lib (a library id, prompted with the
last-clicked shape), the hint counts unresolved elements, and
reassemble casts library-assigned elements via the new
08_place_library.py: the library render scaled into the authored
per-frame bbox, so placement and motion stay authored while the shape
comes from the library, output indistinguishable to 10_compose. Pack
picks up library assignments as --library, so the package's choreo/
stays the deployable end product; the output panel says so and plays
any sequence as video. Label chips carry per-label frame counts --
click, never retype, no case/typo phantoms.

Lesson repeated thrice tonight, now protocol: a multi-rep patch script
that fails one assert loses ALL its reps (write happens at the end) --
after any FAIL, re-verify which pieces actually landed before building
on them. The assignments NameError was exactly that.

## Studio v3: scene-centric, one stage, batch numbered 0-8

The operator's redesign, delivered whole: a SCENE BAR filters every
surface (frame strip, contact sheet, assignment rows, output picker) and
names the per-scene batch button; the label/masks/output panels merged
into ONE stage with tabs (edit / review sheet / play); masks display is
now the pipeline's own overlay jpg -- bbox + label + score, exactly what
sits in letters_sam2_small/overlay/ -- instead of the confusing direct
tint. Buttons collapsed to the batch line: segment chains
sequences+index (the build button died), 6 solves AND mounts (scene or
all -- solve_all/solve_scene append the reassemble+place_library chain),
re-mount alone for library-only changes, then compose and pack. Fixed en
route: 08_reassemble --all still globbed demo_* and saw none of pixar's
sequences (the user's failed run). Frame dropping is the one manual
cleanup kept: /api/dropframe removes a frame project-wide (scene png,
labels, masks) with a re-segment reminder; the letters_clean concept is
fully retired. The multi-rep-patch-loses-all-on-one-FAIL trap fired
TWICE more; whole-file rewrite was the right call for the html.

## Night 2: the working-ones-with-good-performance mandate

User's blocker (no compose, empty output-play) was three stacked faults:
07 gated frame membership on keypoints.json so video-propagated frames
were dropped (7/17 per sequence), 08 --all still globbed the dead
demo_* prefix, and mount had consequently never run. Fixed all three,
rebuilt pixar sequences to 306 full-frame targets, then the overnight
driver re-solved everything assigned solve: pixar's travelling lamps
through the stabilization lane (01_lamp 5.21x recovery, 0.062 -> 0.281
though fit now rides the 1.6 upper bound -- wide-bound retry at 2.4
running; 03_lamp 0.589), the rest 0.67-0.76; family_ad re-solved whole
per the user's instruction (0.36-0.83; 06_I weakest at 0.360 --
edge-hugging, not stabilizable). pixar.mp4 54/54 frames, family_ad.mp4
92. Both packaged with the upgraded exporter.

Peer session umbra-bench-16 (commanded per the user) verified round 1
(all 7 old clips clean in the robot UI, scene numerics exact) and
caught, pre-ship, that arrangement overlaps BLEND joints per arm --
offsets became cumulative durations in footage order with hold gaps.
Its other findings: deploy sends frames raw (fixed by 30 Hz linear
interpolation in pack -- worst step 68 deg -> 8.4 deg), Motion-tab
thumbnail blank is the UI's own floor-vs-table inconsistency (left
alone), collinear-base shadow superposition is by design here.
Arrangements now seat library/sequence stand-ins too, so a scene
timeline carries every element. Round-2 verification brief sent.

## The lamp scale ladder converges; final adoption

pixar_scene_01_lamp (the 5.2x-travel jumping lamp) kept riding the fit
upper bound, so the bound walked up until the optimizer stopped asking:
1.6 -> 0.281, 2.4 -> 0.429, 3.4 -> 0.596, and at a 4.8 bound the fit
freely settles at scale 4.115 for avg_iou 0.627 -- off-bound, converged,
10x the original 0.062. Winning run copied into the canonical stab dir
(newest-summary-wins), remounted, recomposed. En route: 10_compose
--scene rebuilds the whole-cut from only the filtered scenes and
clobbered pixar.mp4 to 17 frames -- caught on the same output line,
full recompose restored 54/92. Probe dirs kept for the record.

Rounds 3-4 with umbra-bench-16 hardened the deploy packages: crossfade
seams (263.6 deg snap -> 0.15 rad morphs, verified independently on
both sides), then a 1.5 s display body for single-pose letter clips
after their solo-time measurement showed interior letters on screen
alone for one frame. Their reports were exemplary peer review; the
scale ladder and the seam scheme both shipped pre-verified.
