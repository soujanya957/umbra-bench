# umbra-bench — dataset layout & metadata schema

Metadata-first: every sample exists in `metadata.jsonl` from the moment its *target*
exists; shadow fields start as `null` and are filled in as captures happen. This lets
collection progress be tracked (`jq 'select(.shadows.hand.path == null)'`) and keeps
the dataset loadable at every stage.

## Directory layout

```
umbra-bench/
├── metadata.jsonl              # one JSON object per sample (the index)
├── CITATIONS.md                # BibTeX + credits for third-party sources
├── METRICS.md                  # every attribute + metric: definition and purpose
├── targets/                    # canonical binary masks, committed (LFS)
│   ├── digits/                 # generated  — e.g. 7_dejavusans-bold.png
│   ├── letters_upper/          # generated  — A_dejavuserif.png  (separate dir from
│   ├── letters_lower/          # generated  — a_dejavuserif.png   lowercase: macOS FS is case-insensitive!)
│   ├── animals/                # ┐ curated from MPEG-7 (curate_mpeg7.py + _sources.json)
│   ├── objects/                # │ semantic subsets: animals / man-made objects /
│   ├── vehicles/               # │ vehicles / human figures
│   ├── figures/                # ┘
│   ├── abstract/               # MPEG-7 device0-9 etc. — NO semantic prior; control group
│   │                           #   separating pure geometric matching from recognizability
│   └── hand_shadow/            # curated from HaSPeR (curate_hasper.py + _sources.json)
├── shadows/
│   └── <sample_id>/
│       ├── hand.png            # binarized shadow mask (+ hand_raw.jpg optional)
│       ├── teleop.png
│       └── optimizer.png
├── external/                   # gitignored: raw third-party downloads (MPEG7/, HaSPeR/)
│   └── README.md               # download instructions (the only committed file here)
├── scripts/
│   ├── make_targets_glyphs.py  # digits + upper/lower letters (deterministic)
│   ├── curate_mpeg7.py         # external/MPEG7 → targets/animals
│   ├── curate_hasper.py        # external/HaSPeR → targets/hand_shadow (exemplars only)
│   ├── shape_attributes.py     # attribute computation (shared, targets + shadows)
│   ├── metrics.py              # target↔shadow metrics: boundary, topology, thin-structure
│   ├── semantic_metrics.py     # recognizability: CLIP retrieval, VLM naming, human study
│   ├── compute_metrics.py      # driver: a sweep or shadows/ → results/metrics_*.csv
│   └── build_metadata.py       # scan targets/ + shadows/ → metadata.jsonl
└── results/                    # computed metrics (IoU etc.) — derived, gitignored
```

Third-party policy: raw external datasets live under `external/` (gitignored, never
committed); only small curated, re-rendered derivatives enter `targets/`, each with
provenance in `_sources.json` → `target_source` (e.g. `hasper:train/swan/xxx.jpg`).
HaSPeR in particular is 15k fair-use photos — keep ~2 exemplars/class, reference the
rest via Hugging Face. Cite per `CITATIONS.md`.

Conventions: targets and shadow masks are 1-bit binary PNGs, **black shape on
white background** (a shadow on a lit screen), 512×512, shape centered with ~10%
margin. 1-bit PNG is the right container for binary masks: smaller than 8-bit
grayscale, lossless, and still previewable everywhere (important for visual
filtering); raw formats like .npy would be larger and unviewable. Raw capture photos may sit next to the
mask (`hand_raw.jpg`) but the mask is the canonical field. `metadata.jsonl` is the
HuggingFace-`datasets`-friendly format (one JSON per line, relative paths), so a
later HF release is just `datasets.load_dataset` away.

## Sample schema

```json
{
  "id": "digits_7_dejavusans-bold",
  "subset": "digits",
  "class": "7",
  "prompt": "cast a shadow of the digit 7",
  "target": "targets/digits/7_dejavusans-bold.png",
  "target_source": "generated:dejavusans-bold",
  "attributes": {
    "area_frac": 0.14, "aspect_ratio": 0.63, "solidity": 0.62,
    "compactness": 3.4, "convexity_defect_depth_rel": 0.18,
    "min_stroke_width_rel": 0.055, "median_stroke_width_rel": 0.11,
    "stroke_width_ratio": 0.41, "thin_mass_frac": 0.52, "elongation": 14.2,
    "skel_len_rel": 1.31, "neck_width_rel": 0.049, "closed_region": true,
    "n_components": 1, "n_holes": 0, "n_holes_signif": 0, "euler_number": 1,
    "hole_area_frac": 0.0, "hole_area_frac_max": 0.0,
    "n_limbs": 3, "n_junctions": 1, "n_concave_extrema": 4,
    "sym_h": 0.41, "sym_v": 0.55, "contour_hf_energy": 0.12,
    "ph_h0_total": 0.011, "ph_n_parts_robust": 0,
    "ph_holes_total_size": 0.0, "ph_hole_max_size": 0.0, "ph_n_holes_robust": 0,
    "ph_n_pockets_robust": 2, "ph_pocket_max_mouth": 0.031, "ph_entropy": 0.53
  },
  "shadows": {
    "hand":      {"path": null, "captured_at": null, "operator": null, "notes": null},
    "teleop":    {"path": null, "captured_at": null, "operator": null, "n_arms": null, "notes": null},
    "optimizer": {"path": null, "captured_at": null, "run_id": null, "config": null, "notes": null}
  },
  "rig": {"light": null, "screen_distance_m": null, "camera": null}
}
```

Field notes: `class` is the semantic label (digit, animal name) — needed for
recognizability studies independent of rendering variant. `attributes` are
auto-computed by `shape_attributes.py`; never hand-edit (they're overwritten on
rebuild). Capture fields (`shadows.*` except `path`, and `rig`) are hand-edited or
filled by capture tooling; `build_metadata.py` preserves them across rebuilds and
auto-fills `path` when the file appears on disk.

## Attribute definitions

33 attributes in four groups, all computed on the binary target mask by
`shape_attributes.py`. `_rel` = normalised by the image diagonal, so every length
is resolution-independent.

**Full definitions, ranges, and what each attribute is diagnostically for:
[METRICS.md](METRICS.md), Part A.** Summarised here:

*Scale & geometry* — `area_frac` (scale proxy), `aspect_ratio` (min/max side of the
min-area rect), `solidity` (area / convex-hull area), `compactness`
(perimeter²/4π·area, 1.0 = disc), `convexity_defect_depth_rel` (deepest concavity).

*Thinness* — the dominant difficulty axis. `min_`/`median_stroke_width_rel` (2x the
5th-percentile / median distance-transform value on the skeleton),
`stroke_width_ratio` (p10/p90 width uniformity), `thin_mass_frac` (share of area
thinner than 3% of the diagonal — the strongest single IoU predictor),
`elongation` (skeleton length / median width), `skel_len_rel`, `neck_width_rel`
(narrowest bridge whose removal splits the shape; `null` if none — the
topology-fragility number), `closed_region` (blob-like vs stroke-like).

*Topology* — `n_components`, `n_holes` (raw), `n_holes_signif` (holes >=0.5% of shape
area), `euler_number` (chi = beta0 - beta1), `hole_area_frac`, `hole_area_frac_max`,
plus persistent-homology summaries of the signed-distance filtration:
`ph_n_holes_robust` / `ph_hole_max_size` / `ph_holes_total_size` (real holes, graded
by size), `ph_n_pockets_robust` / `ph_pocket_max_mouth` (concavities that only close
under dilation), `ph_h0_total` / `ph_n_parts_robust` (near-separable parts),
`ph_entropy` (structural complexity).

*Structure* — `n_limbs` (skeleton endpoints after pruning branches shorter than 8%
of the diagonal: horse -> 6, fork -> 5, `0` -> 0), `n_junctions` (branch points by
Rutovitz crossing number), `n_concave_extrema` (perceptual part count via the
curvature minima rule), `sym_h` / `sym_v` (self-IoU under reflection),
`contour_hf_energy` (Fourier-descriptor energy above harmonic 8).

**Raw vs thresholded counts.** Curated third-party silhouettes carry binarization
litter: the MPEG-7 `horse` target has 18 raw holes, the largest covering 0.06% of
its area. Raw `n_holes` / `n_components` are reported from the unmodified mask,
while every skeleton-derived and persistence attribute is computed after filling
sub-threshold holes and dropping sub-threshold components. The gap between the raw
and `*_signif` counts is a per-target data-quality signal, so both are kept.

**Persistent homology** requires `gudhi` (`pip install gudhi`). Without it those
seven fields come back `null` and everything else still computes. Metric-side
persistence distances additionally need `POT`.

**Attributes on shadows.** `compute_attributes()` takes any binary mask, so it runs
on `shadows/<id>/*.png` too. `attribute_delta(target, shadow)` returns `d_<name>`
for every field — the cheapest interpretable error metric in the benchmark. These
are written to `results/`, never into `metadata.jsonl`, which holds target
attributes only.

## Workflow

1. Add/generate targets under `targets/<subset>/`
2. `python scripts/build_metadata.py` → creates/updates `metadata.jsonl`
3. Capture shadows into `shadows/<id>/{hand,teleop,optimizer}.png`
4. Re-run `build_metadata.py` — paths get linked, capture metadata preserved
5. Metric computation reads `metadata.jsonl`, writes to `results/` (never into the dataset):
   `python scripts/compute_metrics.py --shadows` -> `results/metrics_shadows.csv`,
   one row per (sample, source) with target attributes, shadow attributes, their
   deltas, and every pairwise metric. See [METRICS.md](METRICS.md).
