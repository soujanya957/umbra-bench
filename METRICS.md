# umbra-bench — attributes & evaluation metrics

Complete reference for every shape attribute the dataset records and every metric
proposed for scoring a shadow against its target: what each one is, and what
question it exists to answer.

Implementation: [`scripts/shape_attributes.py`](scripts/shape_attributes.py) (Part A),
[`scripts/metrics.py`](scripts/metrics.py) (Part B),
[`scripts/semantic_metrics.py`](scripts/semantic_metrics.py) (Part C),
driver [`scripts/compute_metrics.py`](scripts/compute_metrics.py).

---

## 0. Why more than IoU

IoU is the benchmark's only metric so far. On this data it substantially measures
**how fat the target is**. Spearman correlation of `best_iou` against target
attributes, over all 546 targets of the big-budget sweep:

| target attribute | ρ vs `best_iou` |
| --- | --- |
| `median_stroke_width_rel` | **+0.64** |
| `compactness` (lower = more disc-like) | **−0.54** |
| `area_frac` | +0.51 |
| `solidity` | +0.44 |
| `n_holes` | −0.15 |

Consequences visible in the results:

- The top-25 targets by IoU are almost entirely `hcircle`, `jar`, `bell`, `comma`,
  `face` — convex blobs. `abstract/hcircle_mpeg7-03` scores 0.851, and it is the
  subset defined by *having no semantic content*.
- `vehicles` peaks at 0.686 with a subset mean of 0.613, because wheels, window
  frames and axles are thin.
- Hole count barely moves IoU (ρ = −0.15) even though filling both eyes of an `8`
  destroys the character completely.

So IoU ranks a featureless disc above a recognisable car, and cannot see the
failure mode that matters most to a viewer. The fix is not to replace it — it is
still the right "how much area agrees" number — but to report it alongside metrics
that answer the questions it cannot.

Four questions, four metric families:

| question | family | Part |
| --- | --- | --- |
| How much area agrees? | overlap | B.1 |
| Is the *outline* in the right place, independent of interior fill? | boundary | B.2 |
| Did the thin parts survive at all? | thin-structure | B.3 |
| Is it still the same *kind* of shape? | topology | B.4 |
| Did the parts land in the right places? | placement | B.5 |
| Would anyone know what it is? | recognizability | C |
| Does it survive contact with a real rig? | physical | D |

---

# Part A — Target shape attributes

Computed by `compute_attributes(mask)` on any binary mask, stored per target in
`metadata.jsonl`. All lengths are normalised by the **image diagonal**, so every
`_rel` field is resolution-independent and comparable across subsets.

The same function runs on shadow masks — that is the point. See B.7.

## A.1 Scale & global geometry

| field | definition | range | purpose |
| --- | --- | --- | --- |
| `area_frac` | shape area / image area | 0–1 | Scale proxy. Bigger shapes are easier to hit; the third-strongest IoU predictor, so it must be controlled for before claiming an optimizer improvement. |
| `aspect_ratio` | min/max side of the min-area rectangle | 0–1 | Elongated targets constrain arm layout: a wide shape wants arms spread along the light axis, a tall one wants them stacked. |
| `solidity` | area / convex-hull area | 0–1 | Concavity. A 3-arm rig casts a union of convex-ish blobs, so deeply concave targets require the arms to *frame* negative space rather than fill positive space — a qualitatively different solve. |
| `compactness` | perimeter² / (4π·area) | ≥1 (1 = disc) | Perimeter cost per unit area. Strongly anti-correlated with IoU (ρ = −0.54): the cleanest single "this target is intrinsically hard" number. |
| `convexity_defect_depth_rel` | deepest single convexity defect | 0–1 | The one bite out of the outline that will be hardest to reproduce. Complements `solidity`, which averages concavity away. |

## A.2 Thinness — the dominant difficulty axis

| field | definition | range | purpose |
| --- | --- | --- | --- |
| `min_stroke_width_rel` | 2 × 5th-percentile distance-transform value on the skeleton | 0–1 | The thinnest part of the shape. Physically a hard floor: nothing can be thinner than the arm or prop casting it. |
| `median_stroke_width_rel` | 2 × median skeleton distance value | 0–1 | Typical limb thickness. **The strongest positive IoU predictor (ρ = +0.64)** — i.e. mostly a measure of how much IoU the target is giving away for free. |
| `stroke_width_ratio` | p10 / p90 of local width | 0–1 | Width *uniformity*. Low = fat torso plus thin legs — exactly the case where IoU is bought by the torso while the legs, which carry the identity, are free to vanish. New. |
| `thin_mass_frac` | fraction of area with local half-width < 3% of the diagonal | 0–1 | How much of the shape is thin. **The strongest predictor found so far (ρ ≈ −0.77 on a stratified subsample)**, beating `median_stroke_width_rel`. Saturates at 1.0 for uniformly thin shapes (e.g. `fork`); pair with `elongation` in that regime. New. |
| `elongation` | skeleton length / median stroke width | ≥0 | Dimensionless "wireiness". Does not saturate where `thin_mass_frac` does (ρ ≈ −0.69). New. |
| `skel_len_rel` | skeleton length / diagonal | ≥0 | Total structure to be traced. Raw version of `elongation`. New. |
| `neck_width_rel` | width of the narrowest bridge whose removal splits the shape; `null` if none | 0–1 or null | **Topology fragility.** A shape whose parts hang together by a 2%-of-diagonal neck comes apart in a real capture — blur, thresholding, a millimetre of arm droop — even when the rendered IoU looks fine. The single best predictor of "this will not survive the physical rig". New. |
| `closed_region` | median stroke width > 50% of the inscribed radius | bool | Blob-like vs stroke-like. Letters and digits are strokes; `jar` and `apple` are blobs. |

## A.3 Topology

Raw counts come from the **unmodified** mask; `*_signif` and the persistence fields
apply thresholds. The gap between the two is a data-quality signal in its own
right — the MPEG-7 `horse` target has 18 raw holes, the largest covering 0.06% of
its area, all of it binarization litter.

| field | definition | range | purpose |
| --- | --- | --- | --- |
| `n_components` | connected components | ≥1 | Whether the shape is one object or several. A 3-arm rig can only produce so many separated pieces. |
| `n_holes` | enclosed background regions | ≥0 | Raw genus. Kept for backward compatibility and as a noise indicator. |
| `n_holes_signif` | holes ≥ 0.5% of shape area | ≥0 | The usable hole count. `0` has 1, `8` has 2, `horse` has 0. New. |
| `euler_number` | `n_components − n_holes` (χ = β₀ − β₁) | ℤ | Single-number topological signature. Cheap invariant for grouping targets by topological type. New. |
| `hole_area_frac` | total hole area / shape area | 0–1 | How much of the shape's bounding structure is negative space. New. |
| `hole_area_frac_max` | largest hole area / shape area | 0–1 | Distinguishes the eye of a `0` (0.35) from binarization litter (0.0006). New. |
| `ph_n_holes_robust` | persistent H₁ classes born at ≤0 with death ≥1% of diagonal | ≥0 | **Size-graded hole count.** Immune to one-pixel flips in a way integer counting is not. New. |
| `ph_hole_max_size` | largest hole inradius / diagonal | 0–1 | Continuous version of "is the big hole still there". New. |
| `ph_holes_total_size` | Σ hole inradii / diagonal | ≥0 | Total loop content. New. |
| `ph_n_pockets_robust` | H₁ classes born at >0 with mouth ≥2% of diagonal | ≥0 | **Pockets**: concavities that only become loops once the shape is dilated. A fork has 0 holes and 53 pockets. Measures "how much of this shape is defined by the gaps in it". New. |
| `ph_pocket_max_mouth` | dilation radius needed to seal the widest concavity | ≥0 | Graded, multi-instance version of `convexity_defect_depth_rel`. New. |
| `ph_h0_total` | Σ finite H₀ lifetimes (part inradius + gap to nearest part) | ≥0 | How much of the shape is loosely-attached parts rather than one mass. New. |
| `ph_n_parts_robust` | H₀ classes with lifetime ≥1% of diagonal | ≥0 | Count of near-separable parts, including ones joined by thin necks that `n_components` calls a single object. New. |
| `ph_entropy` | Shannon entropy of the normalised persistence multiset | ≥0 | Structural complexity in one number — many comparable features scores high, one dominant feature scores low. New. |

**On persistent homology.** The filtration is the sublevel set of the signed
distance transform, so level *t* is the shape dilated by *t* (t>0) or eroded by |t|
(t<0). Two consequences shape the definitions above and are easy to get wrong:

- H₁ **lifetime** is *not* used. Lifetime = hole inradius + surrounding ring
  thickness, so a one-pixel hole inside a thick body would score as persistent as
  the eye of an `8`. Death alone gives hole size.
- H₁ features must be **split by birth sign**. Birth ≤ 0 = a real hole; birth > 0 =
  a concavity that dilation sealed shut. Without the split a fork reports 53
  "holes" — the gaps between its tines.

## A.4 Structure & perceptual

| field | definition | range | purpose |
| --- | --- | --- | --- |
| `n_limbs` | endpoints of the skeleton after pruning branches shorter than 8% of the diagonal | ≥0 | **Protrusion count.** The physically-motivated attribute for a 3-arm rig: you cannot cast more distinct limbs than you have arms and props. Horse → 6 (4 legs + head + tail), fork → 5 (4 tines + handle), `0` → 0. New. |
| `n_junctions` | skeleton branch points, counted by Rutovitz crossing number and clustered | ≥0 | Branching complexity. New. |
| `n_concave_extrema` | deep negative-curvature minima on the smoothed outline | ≥0 | **Perceptual part count.** Hoffman & Richards' minima rule: human vision segments a silhouette into parts at curvature minima. A different question from the skeleton limb count and closer to what a recognizability study measures. New. |
| `sym_h`, `sym_v` | self-IoU under reflection about the centroid | 0–1 | Symmetric targets are markedly easier on a bilateral 3-arm rig (ρ ≈ +0.46 for `sym_v`) — a rig-specific effect with a clean physical explanation, and a good sanity check that the attributes are measuring something real. New. |
| `contour_hf_energy` | share of Fourier-descriptor energy above harmonic 8 | 0–1 | Boundary detail level. Predicts which targets lose their character when a shadow is softened by penumbra. New. |

### Attribute tunables

Documented constants at the top of `shape_attributes.py`; changing one changes the
metric definition, so they belong in any results table:

`THIN_HALFWIDTH_FRAC` 0.03 · `LIMB_PRUNE_FRAC` 0.08 · `HOLE_SIGNIF_FRAC` 0.005 ·
`COMP_SIGNIF_FRAC` 0.005 · `CURV_SMOOTH_FRAC` 0.02 · `CURV_MIN_DEPTH` 0.15 ·
`PH_ROBUST_FRAC` 0.01 · `PH_NOISE_FRAC` 0.002 · `PH_POCKET_FRAC` 0.02

---

# Part B — Geometric & topological evaluation metrics

Target vs shadow, same frame, no alignment unless stated. All in `metrics.py`;
`all_metrics(target, shadow)` returns every one of them flat.

## B.1 Overlap

| metric | definition | range | purpose |
| --- | --- | --- | --- |
| `iou` | \|T∩S\| / \|T∪S\| | 0–1 ↑ | The incumbent. Keep it as the headline "how much area agrees" number; §0 is about what to report *next to* it. |
| `dice` | 2\|T∩S\| / (\|T\|+\|S\|) | 0–1 ↑ | Monotone in IoU, so it adds no ranking information. Included only because segmentation readers look for it. |

## B.2 Boundary

| metric | definition | range | purpose |
| --- | --- | --- | --- |
| `boundary_iou` | IoU restricted to a band of 2% of the diagonal along each outline (Cheng et al., CVPR 2021) | 0–1 ↑ | **The highest-value single addition.** Removes the interior's vote, and with it the stroke-width bias of §0. Empirically it reorders results: `truck_mpeg7-01` has IoU 0.633 but boundary IoU 0.198, while `fork_mpeg7-05` has IoU 0.278 and boundary IoU 0.256 — plain IoU rates the truck twice as good, boundary IoU rates it worse. |
| `nsd` | fraction of outline within a tolerance τ (default 1% of diagonal) of the other outline (Nikolov et al. 2018) | 0–1 ↑ | Boundary agreement with an **explicit physical tolerance**: on a 1 m screen, τ=0.01 reads "what fraction of the outline lands within a centimetre". The natural metric to state a physical-world pass/fail criterion in. |
| `chamfer` | mean symmetric outline-to-outline distance / diagonal | ≥0 ↓ | Continuous: keeps grading after the shapes stop overlapping, where IoU flatlines at 0. Use when comparing methods that are all bad. |
| `hd95` | 95th-percentile symmetric Hausdorff distance / diagonal | ≥0 ↓ | Worst-case outline error, trimmed so one stray pixel cannot set the score. Corresponds to what a viewer notices first. |

## B.3 Thin structure

| metric | definition | range | purpose |
| --- | --- | --- | --- |
| `cldice` | harmonic mean of (shadow skeleton inside target) and (target skeleton inside shadow) (Shit et al., CVPR 2021) | 0–1 ↑ | A limb counts as much as a torso regardless of area, and it **cannot be gamed by thickening** — a fattened prediction's skeleton drifts out of the thin target. Provably homotopy-preserving at 1.0 in 2D, and differentiable, which makes it the leading candidate for an extra term in the *optimizer objective*, not just its report. |

## B.4 Topology

| metric | definition | range | purpose |
| --- | --- | --- | --- |
| `d_betti0`, `d_betti1` | signed component / hole count difference (significance-thresholded) | ℤ | Direction of the error: does the optimizer *fill* holes or *break* the shape? Signed, so systematic bias is visible where an absolute error would average it away. |
| `betti_error` | \|Δβ₀\| + \|Δβ₁\| (Hu et al., NeurIPS 2019) | ≥0 ↓ | **The "is an 8 still an 8" test.** Filling both eyes of an `8` costs a few points of IoU and destroys the character; nothing else in B.1–B.3 detects it. Standard in topology-aware segmentation. |
| `pw_h0`, `pw_h1` | Wasserstein-1 distance between persistence diagrams, per dimension | ≥0 ↓ | **Graded topology error.** Betti error is a step function — a 99%-closed hole and a wide-open one score identically, and one pixel can flip the count. Wasserstein pays the cost of the *edit*, so a barely-lost hole is cheap and a missing one is expensive. Essential for physical captures, where integer counts jitter frame to frame. |

## B.5 Placement

| metric | definition | range | purpose |
| --- | --- | --- | --- |
| `n_limbs_target`, `n_limbs_shadow`, `limbs_unmatched` | pruned-skeleton endpoint counts and their difference | ≥0 | Did the right *number* of protrusions come out. |
| `limb_offset_rel` | mean Hungarian-matched endpoint distance / diagonal | ≥0 ↓ | Did they come out in the right *places*. A four-legged shadow whose legs point the wrong way scores well on every count-based metric and reads as wrong instantly. |

## B.6 Descriptor / retrieval

| metric | definition | range | purpose |
| --- | --- | --- | --- |
| `hu_distance` | log-scale L1 between the first 4 Hu moment invariants | ≥0 ↓ | Translation/scale/rotation invariant, so it separates "wrong shape" from "right shape, wrong pose or size" — a real distinction for a rig with uncalibrated throw distance. Truncated at 4 moments because the high-order invariants sit near zero and their log-scale noise dominates otherwise. |
| `fourier_distance` | L2 between normalised Fourier descriptors (32 harmonics) | ≥0 ↓ | The classical contour-retrieval distance. Its value here is **comparability**: MPEG-7 is a retrieval benchmark with a published protocol, so shadow-as-query results can be quoted against forty years of shape-matching literature rather than only against ourselves. |

## B.7 Attribute deltas

`attribute_delta(target_attrs, shadow_attrs)` → `d_<attribute>` for all of Part A.

The cheapest interpretable metric in the benchmark: no new machinery, and it turns
"IoU 0.64" into statements a reader can act on — *the optimizer systematically
thickens strokes by 18%, drops 0.7 holes per target, and loses 1.4 limbs*. Signed,
so systematic bias separates from random error. `d_n_holes_signif` and
`d_n_limbs` are the two to lead with.

## B.8 Alignment

`aligned_iou(target, shadow)` → `aligned_iou`, `align_gain`, `align_scale`,
`align_shift_rel`. Best IoU over a scale + translation search.

Shadow size is coupled to throw distance, so an uncalibrated rig loses IoU for
reasons unrelated to the solve. **Report the gap, not the aligned number alone**: a
large `align_gain` means the pose is right and the calibration is off; a small one
means the shape itself is wrong. Aligned IoU on its own quietly forgives real scale
errors. Note the sweeps already have a related quantity — `--fit-target` runs record
`best_iou` vs `best_iou_vs_original`, and that difference is the same measurement
taken from the other side.

## B.9 Considered and not implemented

Worth knowing about; each was left out for a stated reason rather than an oversight.

| candidate | what it is | why not (yet) |
| --- | --- | --- |
| **IDSC** (Ling & Jacobs, PAMI 2007) | Inner-distance shape context; articulation-invariant, the strongest classical MPEG-7 descriptor | Real value for the `animals` subset, where articulation is the dominant nuisance variable. Costly (O(n²) per pair) and needs a careful reimplementation. **Highest-priority addition after the current set.** |
| **Bullseye score** | MPEG-7 CE-Shape-1's own retrieval protocol: for each query, how many of its class appear in the top-2N | Needs the full MPEG-7 class structure, not our curated ≤5-per-class subset. Would let "shadow as query" be quoted directly against the shape-retrieval literature — a strong framing if the curation is extended. |
| **Shock graphs / bone graphs** (Siddiqi et al., IJCV 1999) | Skeleton as an attributed graph; compare by graph edit distance | The principled version of B.5 — it would say *"the shadow lost a leg and grew a spurious tail"* rather than reporting an average offset. Skeleton graph extraction is brittle on noisy captures; `limb_match` is the 90%-of-the-value approximation. |
| **Warping error** | Minimum pixel disagreement over all topology-preserving warps of the target (connectomics segmentation literature) | Exactly the right idea — a pixel metric that refuses to count topology-breaking errors as small. Expensive, and `betti_error` + `pw_h1` cover the same failure at a fraction of the cost. |
| **Topological loss** (Hu et al., NeurIPS 2019) | Differentiable persistence-based loss | Not a metric but an **optimizer objective**. The strongest follow-up experiment in this document: if IoU-only optimization loses holes, adding a persistence or clDice term should recover them at negligible IoU cost. |
| **Curvature Scale Space** | MPEG-7's standardised contour descriptor | Superseded in practice by IDSC; would be included mainly for standards compliance. |
| **LPIPS / DINOv2 / DreamSim** | Learned perceptual image distances | Trained on natural images; on 1-bit silhouettes they mostly measure out-of-distribution behaviour. Worth revisiting on the `render_shadow()` output rather than on masks, where DINOv2 features in particular may beat CLIP. |

---

# Part C — Recognizability

Everything in Part B asks whether the shadow matches the target's geometry. None of
it asks whether it reads as a horse. On this benchmark those come apart: the
highest-IoU targets are `hcircle` and `jar`, and `abstract` exists precisely as a
subset with no name to get right.

Two rules apply to all of C and are the easy things to get wrong:

1. **Always score the targets too.** CLIP is weak on 1-bit silhouettes — far outside
   its training distribution. Without the target as a ceiling, a low score is
   unattributable: bad shadow, or blind judge? Report
   `recognizability(shadow) / recognizability(target)`. Treat `abstract` as the null
   control: its ceiling should sit near chance, and if it doesn't, the class list is
   leaking information.
2. **Render, don't feed masks.** `render_shadow()` produces a soft-edged dark shape
   on a warm wall — closer to the models' training distribution *and* to what human
   raters will be shown, which keeps the automatic metrics and the human ceiling on
   the same footing.

## C.1 CLIP retrieval — `clip_retrieval()`

**Definition.** Encode the rendered shadow; encode `"a shadow of a {class}"` for
every class in the benchmark; report the rank of the true class → top-1, top-5, MRR.

**Purpose.** Zero-marginal-cost recognizability over all 546 targets, with a known
chance level, directly comparable to the human N-AFC task.

**Not raw cosine similarity.** A bare CLIP score has no interpretable scale — 0.24
is neither good nor bad — and drifts with prompt wording, so it cannot be compared
across subsets. Ranking against a fixed class list fixes both.

## C.2 Domain-matched classifier

**Definition.** Train a small CNN on binary target masks (this dataset, or the full
MPEG-7 set), classify shadow masks with it.

**Purpose.** CLIP's weakness on silhouettes is a confound that a ratio only
partially removes. A classifier trained *on* binary masks has no domain gap, so
disagreement between it and CLIP localises the problem: if the CNN reads the shadow
and CLIP doesn't, the shadow is fine and CLIP is the wrong instrument.

## C.3 VLM naming — `vlm_naming()`

**Definition.** Ask a VLM open-ended "what does this shadow look like, two words
max, or 'nothing'". Sample 5×, take the modal answer, grade against the class with a
**versioned synonym table** (`match_answer()` — the table is part of the metric
definition, not an implementation detail).

**Purpose.** Closer to the human task than CLIP: naming must be *produced*, not
merely ranked. It degrades gracefully on `abstract`, where a forced-choice model
must pick something but a VLM can correctly answer "nothing". Vote spread across the
5 samples is a free confidence estimate. ~$0.001/image, so it can run on all 546 and
be used to choose the ~100 items worth paying humans for.

## C.4 Human study — `build_human_study()`

The gold standard, and the only one that grounds the rest.

**Two task types, split by subset — this is not cosmetic:**

| subset | task | why |
| --- | --- | --- |
| `animals`, `objects`, `vehicles`, `figures`, `hand_shadow`, `digits`, `letters_*` | **open naming** (free response, coded against the synonym table) + **4-AFC** (true class + 3 foils) | Naming is the honest measure; 4-AFC gives clean numbers with a 25% chance floor. Run both — the gap between them is itself informative. |
| `abstract` | **match-to-target 2-AFC** (this shadow, two candidate targets — which was it trying to be?) | These shapes have no name, so naming is undefined. But `abstract` is the control group separating geometric matching from recognizability, and dropping it wastes the design. Matching a shadow back to its target is well-defined without semantics and measures exactly the fidelity half. |

**Conditions.** Include the **targets themselves** as an upper bound and randomly
paired shadows as a lower bound. Without the upper bound no shadow number is
interpretable; it is the piece people skip.

**Also record reaction time.** Continuous, and more sensitive than accuracy once
accuracy saturates on the easy subsets.

**Design.** ≥15–20 raters per item, randomised order, attention checks; report
Krippendorff's α for inter-rater agreement. Prolific over MTurk for data quality.
≈100 items × 20 raters ≈ $150–300.

## C.5 The actual contribution

The most valuable output is **not the human numbers themselves — it is the
correlation between each automatic metric and human recognition accuracy.**

If the result is "IoU ρ ≈ 0.3, boundary IoU + `betti_error` + `cldice` ρ ≈ 0.7",
then umbra-bench is a benchmark *of metrics* as well as a dataset, which is a
stronger contribution than one more dataset. Fit both `human_accuracy ~ attributes`
and `iou ~ attributes` and put the coefficients side by side; the two being
different **is** the finding.

---

# Part D — Physical-world metrics

Only measurable once captures exist; these are what a robotics venue will ask for,
and none of them are computable in simulation.

| metric | definition | purpose |
| --- | --- | --- |
| **sim2real gap** | IoU(rendered shadow, captured shadow) for the same joint pose | Rendering and calibration fidelity, isolated from solve quality. The number that says whether simulation results transfer at all. |
| **light robustness** | IoU as the light source moves ±10° | A pose that only works from one exact light position is a fragile demo. **The main filter for choosing demo items.** |
| **view robustness** | IoU across camera positions | Shadow art is view-dependent by construction; quantifying the tolerance is part of characterising the medium. |
| **repeatability** | std of IoU over repeated executions of the same joint command | Separates optimizer variance (σ ≈ 0.022 across seeds, per `BUDGET.md`) from hardware variance. |
| **feasibility rate** | fraction of targets with any collision-free solution | Reachability, independent of solve quality — the `hand` vs `teleop` vs `optimizer` comparison the dataset is built for. |
| **execution time / energy** | wall-clock and joint effort per shadow | Practical cost, and the axis along which `small-budget` vs `big-budget` is a real trade-off rather than a sweep parameter. |

## D.1 Demo / physical-testing subset selection

Do **not** take the top-N by IoU — that yields a set of `hcircle`s. Score:

```
demo_score = geometric_quality × recognizability × robustness

  geometric_quality = norm(best_iou) × (1 − norm(std_iou))     # already available
  recognizability   = C.1 / C.3 / C.4 score                    # Part C
  robustness        = IoU under ±10° light perturbation        # Part D
```

Then apply:

- **semantic de-duplication** — at most one item per class, so the demo isn't five
  variants of `hcircle`;
- **difficulty stratification** — easy (`figures/face`, high-scoring letters),
  medium (`animals` with limbs), and one honest **failure case** (`vehicles`, thin
  lowercase). A demo containing a real failure is more credible than one without;
- **attribute-space coverage** — stratify over `thin_mass_frac` × `n_holes_signif`
  × `sym_v` rather than over IoU, so the demo spans the capability envelope.

Existing high-scoring candidates with semantic content: `figures/face` (0.80–0.82,
very low σ), `objects/jar` and `objects/bell` (0.78–0.83), `hand_shadow/chicken`
(0.786, and the strongest narrative — it is what human shadowgraphists actually
cast), high-scoring letters spelling a word (`make_phrase_gif.py` already exists).
`vehicles` tops out at 0.686 and belongs in the hard-case slot, not the demo reel.

---

# Part E — Plan of attack

Metrics first, exploratory analysis last, so that the analysis sees every axis at
once instead of being run repeatedly against a growing feature set.

| stage | work | cost | status |
| --- | --- | --- | --- |
| 1 | Extend `shape_attributes.py` — Part A | free, ~7 min for 546 | **done** |
| 2 | `metrics.py` + `compute_metrics.py` — Part B | free, ~0.07 s/pair | **done** |
| 3 | Rebuild `metadata.jsonl`; run `compute_metrics.py` over both sweeps | ~15 min | next |
| 4 | CLIP retrieval + targets ceiling — C.1 | free (local model) | next |
| 5 | **EDA** — Part F below | — | after 3–4 |
| 6 | VLM naming on all 546 — C.3 | ~$1 | after 5 |
| 7 | Human study on the ~100 items EDA selects — C.4 | ~$150–300 | after 6 |
| 8 | Metric-validation: correlate every metric with human accuracy — C.5 | free | after 7 |
| 9 | Optimizer ablation: add a clDice or persistence term to the objective | compute | stretch |

## Part F — Exploratory analysis, once the metrics exist

Questions the wide CSV from `compute_metrics.py` is designed to answer:

1. **Metric redundancy.** Correlation matrix + PCA over all metrics. How many
   genuinely independent axes are there? If `boundary_iou` and `nsd` correlate at
   0.97, report one.
2. **Is IoU still measuring fatness?** Re-run the §0 correlations for every metric.
   The expected headline: `boundary_iou` and `cldice` have a much weaker relationship
   to `median_stroke_width_rel` than IoU does.
3. **Difficulty regression.** Fit `metric ~ attributes` per metric. The **residuals**
   are the interesting output — large negative residuals are targets the optimizer
   failed on unexpectedly, large positive ones are unexpected successes. Both are
   better demo and failure-analysis candidates than an IoU ranking.
4. **Systematic bias.** Distribution of every `d_*` delta. Does the optimizer
   thicken? Fill holes? Lose limbs? Signed means answer this directly.
5. **Subset structure.** Does `abstract` separate from the semantic subsets in
   attribute space, as its design intends?
6. **Budget effect.** `small` vs `big` per metric. Does 10× compute buy boundary
   accuracy and topology, or only interior fill? A plausible and reportable outcome
   is that extra budget buys IoU without buying recognizability.
7. **Demo selection.** Apply D.1 and emit the ranked shortlist.

---

## References

- Cheng, Girshick, Dollár, Berg, Kirillov. *Boundary IoU: Improving object-centric image segmentation evaluation.* CVPR 2021.
- Shit, Paetzold, et al. *clDice — a novel topology-preserving loss function for tubular structure segmentation.* CVPR 2021.
- Nikolov et al. *Deep learning to achieve clinically applicable segmentation of head and neck anatomy for radiotherapy.* 2018. (Normalised Surface Dice)
- Hu, Fuxin, Samaras, Chen. *Topology-preserving deep image segmentation.* NeurIPS 2019.
- Cohen-Steiner, Edelsbrunner, Harer. *Stability of persistence diagrams.* 2007.
- Ling, Jacobs. *Shape classification using the inner-distance.* PAMI 2007.
- Belongie, Malik, Puzicha. *Shape matching and object recognition using shape contexts.* PAMI 2002.
- Siddiqi, Shokoufandeh, Dickinson, Zucker. *Shock graphs and shape matching.* IJCV 1999.
- Hoffman, Richards. *Parts of recognition.* Cognition 1984. (minima rule)
- Radford et al. *Learning transferable visual models from natural language supervision.* ICML 2021. (CLIP)
- Latecki, Lakämper, Eckhardt. *Shape descriptors for non-rigid shapes with a single closed contour.* CVPR 2000. (MPEG-7 CE-Shape-1 / bullseye protocol)
