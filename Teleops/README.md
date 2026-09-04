# Teleop captures

Shadows a person casts by hand-posing the SO-101 arms, photographed off the wall
and turned into binary masks. They are the reference the optimizer is measured
against, and they are reference for a structural reason rather than a numerical
one: a human physically produced each of these on this rig, so castability holds
by construction. When the optimizer falls short of one, the failure is search and
nothing else.

## The rig

Four **AprilTag 36h11** markers, ids **0–3**, one per corner of the projection
area, placed **clockwise from the top-left**. A RealSense colour stream at
1280×720 faces the wall. Three SO-101 arms and the lamp sit on the near side.

Marker order is the one thing a person can get wrong and not notice, so the
capture tool draws the detected quad live and numbers each corner. If the numbers
do not run clockwise from the top-left, a tag is in the wrong place.

## 1 · Capture

```bash
python scripts/capture_teleop.py --name letters_upperK_A3_topology
```

`SPACE` captures, `N` renames, `Q` quits. **A frame with fewer than four tags
cannot be saved** — a capture that cannot be rectified is not a capture, and
finding that out later means the pose is gone.

Each shot writes three files into `Teleops/`:

| file | what |
| --- | --- |
| `<name>.png` | the raw 1280×720 frame |
| `<name>_rectified.png` | 500×383, fronto-parallel |
| `<name>_capture.json` | tag quad, timestamp, output size, joint angles if recorded |

Add `--record-pose` to read the arms' joint angles at the moment of the shot.
Without it the shadow is recorded but the configuration that made it is not, and
the capture cannot be replayed.

To re-rectify photos taken some other way, or to reprocess everything after a
change:

```bash
python scripts/capture_teleop.py --from-images 'Teleops/*.png'
```

### How the rectification is defined

The projection quad is each tag's **inner corner** — the one nearest the centroid
of the four — warped onto a fixed 500×383. These parameters were not chosen; they
were recovered by fitting against the first 29 captures, whose own capture script
never reached the repo. Inner corners reproduce the stored rectified images at a
correlation of **0.983**; tag centres score 0.588 and outer corners 0.328, so the
convention is unambiguous. Output is pinned at 500×383 because the originals came
out 500×383 or 500×384 depending on where the quad's aspect rounded, and a
one-pixel difference that means nothing should not outlive the run that caused it.

## 2 · Select what to capture

Which targets are worth a person's time is decided in advance and recorded in
`part_abcd_selection.csv` and `part_abcd_letters_digits.csv` — 68 targets across
parts A–D, each with the attribute that earned it a slot (`deepest convexity
defect 0.54`, `betti_error=4, invisible to IoU`, `std_iou=0.038 across seeds`).
Capture names encode `<subset><class>_<partcode>_<words>`, which is what lets a
capture be matched back to its `sample_id`.

## 3 · Binarise

```bash
python scripts/auto_segment_teleop.py --write
```

Writes a mask per capture, a manifest, and `Teleops/masks/_contact_sheet.png` so
all of them can be judged at once. No clicking, no GPU, no model.

The pipeline is: **divide out the illumination, stretch the contrast, then Otsu.**

Contrast is the whole game here. The shadow sits 30–60 grey levels below the lit
wall and the lamp's falloff across the frame is the same magnitude, so a global
threshold on the raw frame classifies dim corners as shadow — on four captures it
returned 0.72–0.80 of the frame. Dividing by a heavily blurred copy removes the
falloff; CLAHE at clipLimit 5 then takes the working range from a std of 6–17 grey
levels to 29–45. **Otsu on that lands all 29 captures between 0.10 and 0.22 with
no outlier at all.** Otsu was never the problem.

Two settings are worth knowing before changing anything:

- **clipLimit 5, not higher.** A global stretch in front of CLAHE scores higher
  still (std 42–48) and is worse: it prints the wall's own texture, and any
  edge-aware method reads texture as edges. Measured the other way too — higher
  clip leaves *fewer* holes, not more (2.9% of shape area at 5, 9.8% at 1).
- Cleanup is the benchmark's own `_denoise`: sub-0.5% components dropped,
  sub-0.5% holes filled. **Never a blanket hole fill** — that erases the counter
  of an `a` and both bowls of a `B`, which is the topology `pw_h1` exists to
  measure.

### When a capture needs a hand

For a frame the automatic path gets wrong, the dashboard's Teleop view takes
point prompts — click the shadow, shift-click to exclude — and exports them:

```bash
python scripts/segment_teleop.py --write                  # geodesic, matches the preview
python scripts/segment_teleop.py --backend sam2 --write   # if torch is around
```

Positive and negative points compete for every pixel rather than one deleting the
other's region, so an exclude refines the boundary instead of destroying the mask.

## 4 · What is still open

**Scoring needs an alignment decision.** No target was displayed on the wall while
the arms were posed, so where it was meant to sit and how big it was meant to be
are unknown (`METRICS.md` §3). Mask and target can be shown side by side honestly;
overlaying them to compute IoU requires inventing that alignment. Three ways out,
none yet chosen: fit a shared similarity transform per capture and report against
both the placed and the authored target, as the `*-fitted` sweeps already do;
report only `hu_distance`, which is invariant to placement and is in the panel for
exactly this reason; or project the target onto the wall during the next session
and make the alignment measurable instead of inferred.

## Files

```
Teleops/
  <name>.png                     raw frame
  <name>_rectified.png           500x383 fronto-parallel
  <name>_capture.json            quad, timestamp, joint angles
  part_abcd_*.csv                the selection, 68 targets across parts A-D
  masks/
    <name>_mask.png              binary, dark = shadow
    manifest.json                per capture: backend, shape fraction, holes, seeds
    _contact_sheet.png           all captures at a glance
```

Scripts: `capture_teleop.py` (capture + rectify), `auto_segment_teleop.py`
(binarise, no interaction), `segment_teleop.py` (point-prompted, for the hard ones).

## How the captures attach to the benchmark

A capture is not a target. It is a shadow somebody produced *for* a target, so it
fills the `shadows.teleop` slot on that target's row in `metadata.jsonl` — the slot
was scaffolded when the dataset was built and had been null ever since — rather
than becoming a row of its own. Both halves are then addressable by one
`sample_id`: the shape a solver was asked to cast, and the shadow a person
actually cast for it.

```
python3 scripts/binarize_teleop.py --rematch --write   # resolve sample_id only
python3 scripts/link_teleop.py                         # dry run
python3 scripts/link_teleop.py --write                 # fill the slot, add the tag
```

Every filled row also gains the tag **`SO101_fleet_teleop`**, so the reference set
can be selected as a group instead of being inferred from a path. 29 captures
attach to 28 targets — `letters_lower_b_dejavusans-bold` was posed twice, and the
repeat hangs off the primary as `extra_captures` rather than replacing it.

Two conventions worth knowing before adding captures:

- **Capture stems are matched to targets by name**, `<subset><class>_<part>_<words>`
  against the selection CSVs. Class names come from the source dataset and are not
  always the English spelling — MPEG-7 ships `glas`, not `glass`. `CLASS_ALIAS` in
  `binarize_teleop.py` holds the exceptions; add to it rather than renaming a
  capture, so the photograph keeps the name it was taken under.
- **`--rematch` re-resolves `sample_id` on the existing manifest and touches
  nothing else.** Re-running the full binarise would overwrite hand-seeded masks
  with an automatic threshold, which is a much larger action than fixing a name.

Captures posed against a target the v1→v2 replacement has since retired are
re-pointed to the v2 id and carry a `notes` field saying so; the view labels their
thumbnail `target (v1)` rather than quietly showing the replacement.
