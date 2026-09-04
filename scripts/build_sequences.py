#!/usr/bin/env python3
"""build_sequences.py — import the animation targets as an ordered-frame axis.

The static benchmark asks "can the fleet cast this shape". A sequence asks the
harder question underneath it: can the fleet cast this shape *given where it
already is*. Frame k+1 is solved from frame k's pose, so a sequence scores the
solver's ability to stay near a prior, not just to hit a silhouette.

Layout, parallel to targets/:

    sequences/<group>/<name>/frame_00.png … frame_NN.png
    sequences.jsonl

Two conventions differ deliberately from `targets/`:

  * **Frames are not re-centered.** Every static target is centred with a 10%
    margin. Doing that per frame here would delete the signal: in `triangle` the
    shape only translates, so a per-frame recentre turns the whole sequence into
    ten copies of one image. Framing is fixed per sequence, never per frame.

  * **Upsampled, not re-rendered.** Sources are 128x128; the bench stores 512.
    NEAREST upsampling is exact (4x4 blocks) and reversible, so nothing is
    invented, and `source_size` in the metadata records what was actually
    authored. The solver renders at 128 anyway.

Motion statistics go into the metadata because they are what makes a sequence
hard, and they are not visible from any single frame. `mean_adj_iou` spans
0.21 (wiper: near-disjoint consecutive frames) to 0.91 (plant_sway: barely
moves), which is the difficulty axis a sequence benchmark needs.
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)

# group, name, glob relative to motion-aware-shadow/targets, prompt
SOURCES = [
    # ── gesture: the eight authored loops the paper's sequence runs use ───────
    ("gesture", "bird",          "anim_n5/bird_[0-9][0-9].png",
     "a bird flapping its wings"),
    ("gesture", "cheer",         "anim_n5/cheer_[0-9][0-9].png",
     "a figure raising both arms in a cheer"),
    ("gesture", "flower",        "anim_n5/flower_[0-9][0-9].png",
     "a flower opening its petals"),
    ("gesture", "reeds",         "anim_n5/reeds_[0-9][0-9].png",
     "reeds bending in wind"),
    ("gesture", "stick_wave",    "anim_n5/stick_wave_[0-9][0-9].png",
     "a stick figure waving one arm"),
    ("gesture", "two_arm_wave",  "anim_n5/two_arm_wave_[0-9][0-9].png",
     "a stick figure waving both arms"),
    ("gesture", "windmill",      "anim_n5/windmill_[0-9][0-9].png",
     "a windmill turning"),
    ("gesture", "wiper",         "anim_n5/wiper_[0-9][0-9].png",
     "a bar sweeping like a windscreen wiper"),
    # ── storyboard: six-frame scenes authored against the rig ─────────────────
    ("storyboard", "rig_bird",     "rig_bird_[0-9][0-9].png",
     "a bird in flight, six-frame scene"),
    ("storyboard", "rig_figure",   "rig_figure_[0-9][0-9].png",
     "a walking human figure, six-frame scene"),
    ("storyboard", "rig_sailboat", "rig_sailboat_[0-9][0-9].png",
     "a sailboat crossing, six-frame scene"),
    ("storyboard", "rig_tree",     "rig_tree_[0-9][0-9].png",
     "a tree growing, six-frame scene"),
    # ── synthetic: geometric controls with known ground-truth motion ──────────
    ("synthetic", "triangle",  "triangle_[0-9][0-9].png",
     "a triangle translating across the frame"),
    ("synthetic", "star_spin", "star_spin_[0-9][0-9].png",
     "a five-pointed star rotating in place"),
    # ── captured: segmented from video, not authored ──────────────────────────
    ("captured", "plant_sway", "plant_run20_sway/plant_[0-9][0-9][0-9][0-9].png",
     "a potted plant swaying"),
]

OUT_SIZE = 512


def load_mask(path, size=128):
    """Read a source frame as a 0/1 mask, dark = shape.

    The polarity check is not paranoia: `targets/william_whole/` in the same tree
    is stored inverted (83% fill, all four corners dark), and silently importing
    an inverted frame would give the solver the background to cast.
    """
    a = np.array(Image.open(path).convert("L").resize((size, size), Image.NEAREST))
    m = (a < 128).astype(np.float32)
    corners = m[0, 0] + m[0, -1] + m[-1, 0] + m[-1, -1]
    if corners >= 3.0 and m.mean() > 0.5:
        raise ValueError(f"{path} looks stored inverted (fill {100 * m.mean():.0f}%)")
    return m


def iou(a, b):
    u = ((a + b) > 0).sum()
    return float((a * b).sum() / u) if u else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=os.path.expanduser(
        "~/Downloads/Projects/2026/fleet-shadow-art/motion-aware-shadow/targets"))
    p.add_argument("--bench", default=_BENCH)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    records, n_frames_total = [], 0
    for group, name, pattern, prompt in SOURCES:
        files = sorted(glob.glob(os.path.join(a.src, pattern)))
        if not files:
            print(f"[!] no frames for {group}/{name} ({pattern})")
            continue

        masks = [load_mask(f) for f in files]
        adj = [iou(masks[i], masks[i + 1]) for i in range(len(masks) - 1)]
        loop = iou(masks[-1], masks[0])
        # Cyclic when closing the loop is no harder than the worst interior step.
        # This matters for the solver, not just for description: a sequence that
        # does not close cannot be scored on wrap-around continuity, and chaining
        # a prior across a non-cyclic seam is measuring nothing.
        cyclic = bool(loop >= min(adj)) if adj else False

        odir = os.path.join(a.bench, "sequences", group, name)
        frame_paths = []
        for i, f in enumerate(files):
            rel = os.path.join("sequences", group, name, f"frame_{i:02d}.png")
            frame_paths.append(rel)
            if a.dry_run:
                continue
            os.makedirs(odir, exist_ok=True)
            src = Image.open(f).convert("L")
            img = src.resize((OUT_SIZE, OUT_SIZE), Image.NEAREST)
            # Re-threshold after the resize, then store 1-bit like every other
            # target. NEAREST on a 4x upsample cannot introduce grey, but the
            # sources are not all guaranteed clean 1-bit to begin with.
            arr = (np.array(img) >= 128).astype(np.uint8) * 255
            Image.fromarray(arr, "L").convert("1").save(
                os.path.join(a.bench, rel))

        src_size = Image.open(files[0]).size[0]
        records.append({
            "sequence_id": f"{group}/{name}",
            "group": group,
            "name": name,
            "prompt": prompt,
            "n_frames": len(files),
            "frames": frame_paths,
            "cyclic": cyclic,
            "source": os.path.relpath(os.path.dirname(files[0]), a.src) + "/",
            "source_size": src_size,
            "recentered": False,
            "motion": {
                "mean_adj_iou": round(float(np.mean(adj)), 4) if adj else None,
                "min_adj_iou": round(float(np.min(adj)), 4) if adj else None,
                "loop_iou": round(loop, 4),
                "span_iou": round(min(iou(masks[0], m) for m in masks), 4),
                "mean_fill": round(float(np.mean([m.mean() for m in masks])), 4),
            },
            "shadows": {"optimizer": None},
        })
        n_frames_total += len(files)
        print(f"{group}/{name:14s} {len(files):3d} frames  "
              f"adj={np.mean(adj):.3f} loop={loop:.3f} "
              f"{'cyclic' if cyclic else 'open'}")

    if not a.dry_run:
        out = os.path.join(a.bench, "sequences.jsonl")
        with open(out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {out}")
    print(f"{len(records)} sequences, {n_frames_total} frames")


if __name__ == "__main__":
    main()
