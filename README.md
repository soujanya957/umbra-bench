# umbra-bench

A shadow-art dataset where the same target is solved three ways.

Each sample pairs a target with the shadows produced for it:

| field | what it is |
| --- | --- |
| `prompt` | text description of the target |
| `target` | input image (the silhouette to cast) |
| `shadow_hand` | shadow made by a human, by hand |
| `shadow_teleop` | shadow made by a human teleoperating the robot fleet |
| `shadow_optimizer` | shadow produced by the optimizer |

## Why three sources

Comparing them separates two things that are usually confounded:

- **hand vs teleop** — what the robot's embodiment costs, since a human is driving in both cases
- **teleop vs optimizer** — what the algorithm costs, since the hardware is the same in both cases

A target the optimizer misses but a teleoperating human hits is an algorithm problem.
One that both miss is a reachability problem.

## Target categories

![target overview — one row per subset](docs/overview.png)

| subset | count | what it is | what it probes |
| --- | --- | --- | --- |
| `digits` | 30 | 0–9 × 3 bold fonts | controlled glyph benchmark; topology varies (0/6/9 have holes, 8 has two) |
| `letters_upper` | 78 | A–Z × 3 bold fonts | wider topology/stroke variety than digits |
| `letters_lower` | 78 | a–z × 3 bold fonts | thinner strokes, ascenders/descenders |
| `animals` | 110 | MPEG-7 animal silhouettes | organic outlines, thin limbs; 8 classes overlap `hand_shadow` for comparison |
| `objects` | 115 | MPEG-7 man-made objects | handles, holes, thin protrusions (fork, key, cup) |
| `vehicles` | 30 | MPEG-7 cars/trucks/... | boxy outlines + wheels, distinct attribute profile |
| `figures` | 10 | MPEG-7 human silhouettes | human forms — closest to shadow-theatre storyboards |
| `abstract` | 85 | MPEG-7 device0–9, heart, ... | **no semantic prior** — control group separating geometric matching from recognizability |
| `hand_shadow` | 10 | binarized HaSPeR exemplars | what human shadowgraphists actually cast; human-expert reference |

## Data structure

```
metadata.jsonl        the index — one JSON line per sample; start here
targets/              ← the final targets to cast (546 masks)
  digits/               0-9 × 3 bold fonts            (generated)
  letters_upper/        A-Z × 3 bold fonts            (generated)
  letters_lower/        a-z × 3 bold fonts            (generated)
  animals/              22 classes × ≤5               (curated from MPEG-7)
  objects/              23 classes × ≤5               (curated from MPEG-7)
  vehicles/             6 classes × ≤5                (curated from MPEG-7)
  figures/              human silhouettes             (curated from MPEG-7)
  abstract/             device0-9, heart, ... — no semantic prior (MPEG-7)
  hand_shadow/          real hand-shadow exemplars    (curated from HaSPeR)
shadows/<sample_id>/  captured results: hand.png, teleop.png, optimizer.png
scripts/              target generation, curation, metadata build
external/             raw third-party downloads (gitignored; see its README)
```

All targets are 1-bit PNGs, **black shape on white background**, 512×512,
centered with 10% margin. Every target has a record in `metadata.jsonl` with its
prompt, auto-computed shape attributes (holes, stroke width, solidity, ...), and
slots for the three shadow captures — `null` until captured. Full schema and
workflow: [DATASET.md](DATASET.md). Credits for third-party data: [CITATIONS.md](CITATIONS.md).

To find a specific target: filenames are `<class>_<variant>.png` under
`targets/<subset>/`, and each metadata record's `target` field holds the
repo-relative path.

## The sequence axis

The 546 targets above are static: each asks *can the fleet cast this shape*. The
sequences ask the question underneath it — *can the fleet cast this shape given
where it already is*. A frame is solved from the previous frame's pose, so a
sequence scores continuity, not just silhouette matching.

```
sequences/<group>/<name>/frame_00.png … frame_NN.png
sequences.jsonl                        one record per sequence
```

| sequence | frames | adj IoU | loop IoU | cyclic | prompt |
| --- | --- | --- | --- | --- | --- |
| `captured/plant_sway` | 20 | 0.91 | 0.70 | no | a potted plant swaying |
| `gesture/stick_wave` | 8 | 0.85 | 0.85 | yes | a stick figure waving one arm |
| `gesture/two_arm_wave` | 8 | 0.76 | 0.77 | yes | a stick figure waving both arms |
| `gesture/cheer` | 8 | 0.69 | 0.70 | yes | a figure raising both arms in a cheer |
| `gesture/flower` | 8 | 0.59 | 0.67 | yes | a flower opening its petals |
| `gesture/bird` | 8 | 0.51 | 0.58 | yes | a bird flapping its wings |
| `gesture/windmill` | 8 | 0.42 | 0.41 | no | a windmill turning |
| `gesture/reeds` | 8 | 0.40 | 0.43 | yes | reeds bending in wind |
| `gesture/wiper` | 12 | 0.21 | 0.22 | yes | a bar sweeping like a windscreen wiper |
| `storyboard/rig_sailboat` | 6 | 0.82 | 0.39 | no | a sailboat crossing, six-frame scene |
| `storyboard/rig_figure` | 6 | 0.75 | 0.74 | yes | a walking human figure, six-frame scene |
| `storyboard/rig_tree` | 6 | 0.38 | 0.04 | no | a tree growing, six-frame scene |
| `storyboard/rig_bird` | 6 | 0.29 | 0.15 | no | a bird in flight, six-frame scene |
| `synthetic/triangle` | 10 | 0.58 | 0.00 | no | a triangle translating across the frame |
| `synthetic/star_spin` | 5 | 0.56 | 0.56 | yes | a five-pointed star rotating in place |

`adj IoU` is the mean IoU between consecutive frames — the difficulty axis. It
runs from 0.21 (`wiper`, consecutive frames barely overlap) to 0.91
(`plant_sway`, barely moves). `loop IoU` compares the last frame to the first; a
sequence is **cyclic** when closing the loop is no harder than its worst interior
step, and only those can be scored on wrap-around continuity.

Two conventions differ from `targets/` on purpose:

- **Frames are not re-centered.** Static targets are centred with a 10% margin.
  Per-frame centring here would delete the signal — in `synthetic/triangle` the
  shape only translates, so it would collapse to ten copies of one image.
- **Upsampled, not re-rendered.** Sources are 128×128, stored at 512 like every
  other target. NEAREST upsampling is exact and reversible; `source_size` records
  what was authored.

### Sequence results

`optimized/anim-optimizer/<condition>/<group>/<name>/` — same layout and the same
optimizer budgets as `optimized/base-optimizer/`, one thing changed:

> Frame *k* is solved from frame *k−1*'s **best** pose. Each frame gets 5
> independent solves, all seeded from and anchored to the previous frame's
> winner; the winner of those becomes the prior for frame *k+1*.

Chaining run *i* to run *i* is the obvious mistake and measures something else —
five independent mediocre trajectories instead of one good one. The winner is
picked planner-first (prefer runs within 1.2 rad/joint of the prior, then by IoU),
because a run that lands outside the direct-connect radius is not just a worse
frame, it is a worse *prior*, and every later frame inherits it.

Four conditions: `{small,big}-budget` × `{baseline, -distort}`. The distort
conditions fit **one** bounded free-form deformation per clip (never per frame —
that makes the character breathe) and apply it unchanged to every frame.

**Reading the distort numbers.** A distort run is scored against a target it
moved. Only `mean_iou_vs_original` — IoU against the frames as authored — is
comparable across conditions; `mean_iou` is not. `scripts/summarize_sequences.py`
computes the comparison from the former for exactly this reason.

## Status

Early. Targets + metadata done (546 samples). Collecting shadow captures.
