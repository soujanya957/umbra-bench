#!/usr/bin/env python3
"""Split a sequence into shape and trajectory, so the solver only sees shape.

The one-fit-per-clip rule in run_sequence.py fits the UNION footprint of all
frames into the rig's reachable support. For a clip whose element translates a
long way (family_ad_scene_06_L drifts 307 px), the union is the trajectory, not
the shape: the fit shrinks the letter by the whole sweep of its path and the
shadow comes out tiny. Measured on the current track, stabilising L recovers
2.3x scale and lifts its target step IoU from 0.39 to 0.84 -- the "motion" was
almost pure translation.

This derives `<id>_stab/` next to the source sequence:

    frames        each frame translated so its centroid sits at the mean
                  centroid of the clip (integer shift, no resampling, and the
                  script errors if any shape pixel would leave the canvas)
    source.json   stabilized_from, and `trajectory_px` -- the shift that was
                  REMOVED per frame, i.e. what a compositor must re-apply to
                  the rendered shadow to restore the authored motion

The split is exact for translation and only translation: rotation and
deformation stay in the frames, where they belong -- they are shape motion the
solver should be scored on. Re-applying `trajectory_px` to the stabilised
frames reconstructs the originals pixel-for-pixel (the script verifies this
before writing anything).

The variant is a solving condition, not a new benchmark row: it is NOT added
to sequences.jsonl here. Solve it with run_sequence.py pointed at the _stab
directory; at composite time (demo/08 + demo/10) shift each rendered
frame by trajectory_px scaled by the render's px-per-canvas-px ratio. For a
filmed rig the same numbers become a per-frame camera reframe.

    python scripts/stabilize_sequence.py --ids family_ad_scene_06_L
    python scripts/stabilize_sequence.py --min-gain 1.1      # scan the track
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)


def load(p):
    return np.array(Image.open(p).convert("L")) < 128


def bbox_side(m):
    ys, xs = np.where(m)
    return int(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)) if len(ys) else 0


def shift(m, dy, dx):
    out = np.zeros_like(m)
    ys, xs = np.where(m)
    ys2, xs2 = ys + dy, xs + dx
    H, W = m.shape
    ok = (ys2 >= 0) & (ys2 < H) & (xs2 >= 0) & (xs2 < W)
    out[ys2[ok], xs2[ok]] = True
    return out, int((~ok).sum())


def stabilize(rec, seq_dir, write):
    frames = [os.path.join(_BENCH, f) for f in rec["frames"]]
    masks = [load(f) for f in frames]
    cents = [np.array(np.where(m)).mean(axis=1) for m in masks]
    ref = np.mean(cents, axis=0)
    shifts = [tuple(int(v) for v in np.round(ref - c)) for c in cents]

    stab, lost = [], 0
    for m, (dy, dx) in zip(masks, shifts):
        s, n_lost = shift(m, dy, dx)
        lost += n_lost
        stab.append(s)
    if lost:
        # A frame whose content already hugs the canvas edge cannot be recentred
        # without clipping (cheer_n5 does this with raised arms). Not an error
        # in a scan -- just not a stabilisable clip.
        raise ValueError(f"{lost} shape px would leave the canvas")
    # The whole point is exactness: undoing the shift must reproduce the
    # original, or the trajectory the compositor replays is a lie.
    for m, s, (dy, dx) in zip(masks, stab, shifts):
        back, n_lost = shift(s, -dy, -dx)
        if n_lost or not np.array_equal(back, m):
            raise SystemExit(f"[!] {rec['id']}: reconstruction mismatch")

    union = np.zeros_like(masks[0])
    for m in masks:
        union |= m
    sunion = np.zeros_like(masks[0])
    for m in stab:
        sunion |= m
    gain = bbox_side(union) / max(bbox_side(sunion), 1)

    if write:
        out = os.path.join(seq_dir, rec["id"] + "_stab")
        os.makedirs(out, exist_ok=True)
        for f, s in zip(frames, stab):
            img = Image.fromarray(np.where(s, 0, 255).astype(np.uint8))
            img.convert("1").save(os.path.join(out, os.path.basename(f)))
        src = {}
        pj = os.path.join(seq_dir, rec["id"], "source.json")
        if os.path.exists(pj):
            with open(pj, encoding="utf-8") as f:
                src = json.load(f)
        src.update({
            # The parent's EFFECTIVE loop becomes a declaration: stabilising
            # nearly freezes the frames, so the wrap test on the _stab clip
            # would call any travelling shape a loop -- the exact mislabel
            # declarations exist to prevent.
            "loop": bool(rec["target_motion"]["loop"]),
            "stabilized_from": rec["id"],
            # per frame: the (dy, dx) ADDED to the original to centre it.
            # Restoring the motion means SUBTRACTING these from the
            # stabilised placement (see 08_reassemble.py).
            "trajectory_px": [[dy, dx] for dy, dx in shifts],
            "stabilize_scale_gain": round(gain, 3),
        })
        with open(os.path.join(out, "source.json"), "w", encoding="utf-8") as f:
            json.dump(src, f, indent=1)
    return gain, max(abs(v) for s in shifts for v in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=_BENCH)
    ap.add_argument("--seq-dir", default="sequences")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--min-gain", type=float, default=None,
                    help="scan the track and stabilise every clip whose "
                         "union-vs-stabilised bbox ratio is at least this")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.ids and a.min_gain is None:
        ap.error("give --ids or --min-gain")

    seq_dir = os.path.join(a.bench, a.seq_dir)
    recs = [json.loads(l) for l in
            open(os.path.join(a.bench, "sequences.jsonl"), encoding="utf-8")]
    done = 0
    for rec in recs:
        if rec["id"].endswith("_stab"):
            continue
        if a.ids and rec["id"] not in a.ids:
            continue
        try:
            gain, dmax = stabilize(rec, seq_dir, write=False)
        except ValueError as e:
            if a.ids:
                raise SystemExit(f"[!] {rec['id']}: {e}")
            print(f"[stab] {rec['id']:<24} skipped: {e}")
            continue
        if a.min_gain is not None and gain < a.min_gain:
            continue
        tag = "would write" if a.dry_run else "wrote"
        if not a.dry_run:
            stabilize(rec, seq_dir, write=True)
        print(f"[stab] {rec['id']:<24} gain {gain:.2f}x  max shift {dmax:>3}px"
              f"  -> {tag} {rec['id']}_stab/")
        done += 1
    if not done:
        print("[stab] nothing selected")


if __name__ == "__main__":
    main()
