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
│   ├── shape_attributes.py     # attribute computation (shared)
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
    "n_components": 1,
    "n_holes": 0,
    "solidity": 0.62,
    "convexity_defect_depth_rel": 0.18,
    "min_stroke_width_rel": 0.055,
    "median_stroke_width_rel": 0.11,
    "compactness": 3.4,
    "aspect_ratio": 0.63,
    "area_frac": 0.14,
    "closed_region": true
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

All computed on the binary target mask; `_rel` = normalized by bounding-box diagonal.

- `n_components` — connected components of the shape
- `n_holes` — enclosed background regions (topology: 0 for C/7, 1 for A/0/6, 2 for 8)
- `solidity` — area / convex-hull area (1.0 = convex; low = concave, e.g. star)
- `convexity_defect_depth_rel` — deepest concavity
- `min_stroke_width_rel`, `median_stroke_width_rel` — 2× distance-transform ridge value;
  min is taken at the 5th percentile of skeleton values for noise robustness.
  Expected to be the strongest IoU predictor (arm/prop thickness floor).
- `compactness` — perimeter² / (4π · area), 1.0 = disk
- `aspect_ratio` — min/max side of the min-area bounding rect
- `area_frac` — shape area / image area (scale proxy)
- `closed_region` — heuristic: median stroke width > 25% of the mask's inscribed
  radius (blob-like) vs stroke-like (letters/digits are strokes)

## Workflow

1. Add/generate targets under `targets/<subset>/`
2. `python scripts/build_metadata.py` → creates/updates `metadata.jsonl`
3. Capture shadows into `shadows/<id>/{hand,teleop,optimizer}.png`
4. Re-run `build_metadata.py` — paths get linked, capture metadata preserved
5. Metric computation reads `metadata.jsonl`, writes to `results/` (never into the dataset)
