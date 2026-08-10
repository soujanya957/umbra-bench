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

## Status

Early. Targets + metadata done (546 samples). Collecting shadow captures.
