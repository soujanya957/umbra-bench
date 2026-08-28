# need-to-collect

What is missing from this benchmark, why each item matters to the ICRA paper, and
the command that produces it. Ordered by value per unit of effort.

Status as of 2026-08-25. Rig is 3 × SO-101, arm gap 0.2 m, 128 px render.
Compute reference: `dutchman` (Linux, 4090) runs ~115 s per big-budget solve,
~65 s per small-budget solve, ~6× faster than the Mac. All timings below assume
that box.

______________________________________________________________________

## 0. Free wins — no new compute

### 0.1 Link the optimizer results that already exist

`metadata.jsonl` has **0 of 546** samples with any `shadows.*.path` populated,
including `optimizer`, even though `optimized/big-budget-fitted/<subset>/<name>/`
already holds the solved PNGs and `summary.csv` holds the configurations. The
dataset currently advertises a three-source design and ships zero linked sources.

Backfill `shadows.optimizer` (path, run_id, config) plus `rig` from
`optimized/big-budget-fitted/BUDGET.md`. Until this is done, anyone loading the
benchmark by its documented interface sees an empty dataset.

**Acceptance:** 546/546 optimizer paths resolve to a file on disk.

### 0.2 Record the achievability ceiling honestly

`summary.csv`'s `uncastable_before` / `uncastable_after` do **not** bound
achieved IoU: 480 of 546 targets (87.9%) score above the bound those columns
imply. Either the columns mean something other than a ceiling, or they are
mis-derived. Document what they actually measure, and stop treating them as
$\mathrm{IoU}^\star(T)$. The ceiling stays unmeasured until §3 lands.

______________________________________________________________________

## 1. Human data — the highest-value thing you can provide

You offered to collect human data. Spend it here, in this order.

### 1.1 Recognizability study (do this one first)

**This is the single most valuable dataset item and it is not more IoU.**

The paper reports IoU throughout and never establishes what an IoU means. A
reviewer will ask what 0.75 buys, and there is currently no answer. The medium's
actual goal is a shadow a person *recognizes*, not one that maximizes pixel
overlap.

- **Protocol:** show each rendered shadow (no target alongside) and ask for a
  free-response name, or a 4-way forced choice against distractors from the same
  subset. ~150 shadows, stratified across the achieved-IoU range (0.34–0.94) in
  ~6 bins, ≥15 per bin. 5 raters each; report inter-rater agreement.
- **Sample from:** `optimized/big-budget-fitted/<subset>/<name>/<name>_best.png`.
- **What it produces:** the IoU→recognition curve, and the threshold above which
  shadows are reliably named. If recognition saturates around IoU ≈ 0.7, then a
  large part of this paper's remaining IoU headroom does not matter, and saying
  so is a stronger result than another point of IoU.
- **Cost:** a few hours of rater time. No robot, no compute.
- **Figure:** recognition rate vs IoU, with the saturation knee marked. Put this
  early in Results; it licenses every IoU number that follows.

### 1.2 Teleoperation on a stratified subset — 40 targets

Unblocks $\Delta_{\mathrm{alg}} = \rho_{\mathrm{opt}} - \rho_{\mathrm{tel}}$
(Eq. 11), which the paper names as how every result should be read and then never
reports.

- **Sample:** 40 targets stratified by `median_stroke_width_rel` tercile ×
  subset, including all 10 `hand_shadow` and all 10 `figures`.
- **Also record, per frame: wall-clock time the operator took.** This is the
  comparison that lands for a robotics audience — if the optimizer matches a
  trained teleoperator's IoU in a fraction of the time, that is a systems result.
  Without timing, teleop is only a quality baseline; with it, it is a
  cost-benefit argument.
- **Acceptance:** 40 samples with `shadows.teleop.path`, `operator`, `n_arms`,
  and a `seconds` field added to the teleop record.

### 1.3 Hand shadows — 30 targets

Unblocks $\Delta_{\mathrm{emb}}$: what the robot embodiment costs relative to a
human hand. All 10 `hand_shadow` (where a hand should win outright) plus 20 drawn
from `animals` and `figures`.

Photograph against the same wall, same light, and binarize with the same
threshold path used for the optimizer shadows. **Check polarity on the way in** —
this repo has been bitten before by masks stored inverted.

______________________________________________________________________

## 2. Robot experiments — the ICRA content

The paper's problem formulation is robotics; its experiments currently are not.
These four close that gap. E1 is the one that decides whether the paper's central
claim survives.

### E1 — Reachability-penalty sweep (the paper's actual contribution)

**Why:** the intro claims consecutive frames are "connectable by construction"
via a motion-planning-aware objective. That claim is **currently unsupported** —
`PAPER_NUMBERS.md` lists the MP-barrier effect under *Claims CUT*, having
measured no effect on %STOMP+ at 32 transitions. Meanwhile we now know
independently-solved similar-target pairs are only 59.4% cheaply connectable and
5.1% unsafe, so there is a real problem for μ to solve. Either μ fixes it and
this is the headline, or it does not and the claim must come out. Nothing else on
this list matters as much.

```bash
# 8 sequences x 8 frames x mu in {0, 0.25, 0.5, 1.0, 2.0} x 3 seeds, N=3
for MU in 0 0.25 0.5 1.0 2.0; do for SEED in 0 1 2; do
  python scripts/run_sequence.py --targets <seq>.json --n-robots 3 \
      --reachability-penalty $MU --seed $SEED \
      --outdir results/mu/mu${MU}_s${SEED}
  python scripts/planner_level_report.py --scene-dir results/mu/mu${MU}_s${SEED}
done; done
```

- **Measure:** mean IoU, % cheap transitions, % unsafe, jerk RMS, path length,
  wall-clock. Seed-matched — same targets, same seeds, μ the only variable.
- **The plot:** Pareto. x = mean IoU, y = % cheaply connectable, one point per μ,
  seed error bars. A knee showing large connectability gains at negligible IoU
  cost is the paper's money figure.
- **Scope:** ~960 frame-solves ≈ 16–30 h. Two nights. Run μ=0 and μ=1.0 at one
  seed first (~2 h) to confirm the effect exists before committing the sweep.
- **Report it either way.** If μ does nothing, that is publishable as a negative
  and the claim gets cut — far better found here than in review.

### E2 — Robots × budget at matched renders

**Why:** answers a resource-allocation question a robotics audience actually
cares about: *given a fixed compute budget, add arms or add search?* We know from
the current N-sweep that IoU goes 0.334→0.545→0.619→0.651 for N=1,2,3,5 while
redundancy goes 0.00→0.41 — arms saturate once capacity κ exceeds ~1. Whether a
5-arm fleet at a small budget beats a 3-arm fleet at a large one is unmeasured
and non-obvious.

```bash
# 40 stratified targets x N in {1,2,3,5} x 3 budgets x 2 seeds
python scripts/budget_sweep.py --targets bench40.json \
    --n-robots {1,2,3,5} --budgets 2000,6000,18000 --seeds 0,1
```

- **x-axis must be renders or wall-clock, never iterations.** An iteration at N=5
  costs ~4.5× one at N=1; iteration-indexed curves flatter high N and a reviewer
  will catch it.
- **Measure:** IoU, redundancy ρ, spill, κ, collision count, wall-clock.
- **The plot:** IoU vs renders, one curve per N, iso-quality line at the
  budget needed to reach a fixed IoU. Subsumes the current fig2/fig3.
- **Scope:** 960 solves ≈ 16–30 h.

### E3 — Coordination cost vs fleet size (the most "multi-robot" result available)

**Why:** this is the measurement that makes the paper multi-robot rather than
single-robot-repeated, and none of it exists yet. As N grows, arms must avoid
each other in transit, not just at keyframes. The planner already exposes exactly
the right signal: the **`staggered`** level means arms had to be given temporal
offsets to avoid collision. How often staggering becomes *necessary* as N grows
is a direct, legible measure of coordination cost.

```bash
# same sequence, N in {2,3,5,7}, 3 seeds, full planner
for N in 2 3 5 7; do for SEED in 0 1 2; do
  python scripts/run_sequence.py --targets <seq>.json --n-robots $N --seed $SEED \
      --outdir results/coord/n${N}_s${SEED}
  python scripts/planner_level_report.py --scene-dir results/coord/n${N}_s${SEED}
done; done
```

- **Measure per transition:** planner level distribution, inter-robot collision
  checks failed *during transit*, jerk RMS, peak joint velocity, and **makespan**
  — wall-clock time to actually execute the transition under joint-velocity
  limits. Staggering trades makespan for safety; quantify the exchange rate.
- **The plot:** stacked planner-level bars vs N (share of transitions needing
  each fallback), with makespan overlaid on a second axis. The story to look for:
  *cheap transitions get rarer and execution gets slower as the fleet grows* —
  coordination cost, measured.
- **Scope:** ~350 frame-solves ≈ 6–11 h. One night. **Best value per hour on this
  list after E1.**

### E5 — Closed-loop wall correction (the system result)

**Why:** the `im-redistort` AprilTag rectifier is implemented and unit-tested
(72 tests, synthetic oblique views to 55°) but has **never been run against the
rig**. It is the piece that turns UMBRA from an open-loop renderer into a
closed-loop robotic system, which is the framing an ICRA reviewer rewards.
Paper section: `sec:method:closedloop`, results slot `sec:results:physical`.

Order of operations:

1. **Rig calibration.** Tape a 4-tag board beside the screen, outside the cast
   region. Photograph the wall for ~10 known configurations, rectify, binarize,
   then fit rig parameters $\theta = (\ell, \{b_i\}, \Pi)$ by maximising mean
   IoU(sim, wall) — same CMA-ES used for the frame solve, since $S$ is no more
   differentiable in $\theta$ than in $q$.
   **Acceptance:** post-calibration sim-to-real IoU per frame, and the recovered
   light position vs the tape-measure estimate. If the light moves >2 cm, that
   alone is a result — it is the amplification argument made concrete.
2. **Open-loop vs closed-loop ablation.** Same 8-frame sequence, deployed twice:
   open-loop, and with the per-frame residual correction. **Measure IoU on the
   wall, not in sim.** One paired plot; ~1 h of rig time once (1) works.
3. Log reprojection RMS per frame and drop anything above 2 px.

**Scope:** no optimizer compute; this is rig time and camera work.

______________________________________________________________________

### E4 — Distortion as a substitute for arms

**Why:** prior work in this repo established that bounded target distortion buys
*achievability*, not fidelity. The interesting question is therefore not "does
distortion help" but *what does it substitute for*. If N=3 with distortion
matches N=5 without, distortion is worth an arm, and that is a far more
interesting framing than an IoU delta.

```bash
python run.py --targets bench40.json --n-robots {1,3,5} --distort --distort-max 0.15
# and the matched no-distort arm of the comparison
```

- **Measure:** IoU vs *original* target (not the deformed one — say which, always),
  redundancy, κ, and topology retention.
- **Scope:** 240 solves ≈ 4–8 h.

______________________________________________________________________

## 3. What is still out of reach

$\mathrm{IoU}^\star(T)$, the true per-target achievability ceiling, is not
computable and is not measured by the `uncastable` columns (§0.2). The practical
substitute is the best-of-all-sources envelope: $\max$ over optimizer, teleop and
hand for each target. That requires §1.2 and §1.3, which is another reason to
collect them.

______________________________________________________________________

## Experiment → figure map

| item | feeds |
| --- | --- |
| §1.1 recognizability | **Fig. R** — recognition vs IoU; licenses every IoU in the paper |
| E1 μ-sweep | **Fig. 3** — connectability Pareto; the contribution figure |
| E2 robots × budget | **Fig. 4** — efficiency frontier; replaces current fig2 + fig3 |
| E3 coordination cost | **Fig. 5** — planner level + makespan vs N |
| §1.2 + §1.3 human | **Fig. 6** — three-source comparison, Eq. 11 |
| E4 distortion | table row, not a figure |
| E5 closed loop | **Fig. 7** — open vs closed loop IoU measured on the wall; the systems result |

If only one thing gets run: **E1**. If only one thing gets *built*: **E5** —
it is the cheapest way to make the paper read as robotics rather than graphics. If only one thing gets collected:
**§1.1 recognizability**.
