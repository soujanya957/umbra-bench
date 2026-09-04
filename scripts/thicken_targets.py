#!/usr/bin/env python3
"""Thicken targets the rig physically cannot cast, by the least amount that works.

An arm link has a width, and the throw magnifies it: below some stroke width the
rig cannot draw a line at all, at any pose. `_rescue_common.rig_floor()` measures
that floor from the sweep's own shadows. A target finer than the floor is not a
hard target, it is an impossible one, and its score says nothing about the solver.

This dilates such a target until it just clears the floor, and then stops. The
dilation runs on the 512 px master; the stopping test runs on the 128 px downsample,
because that is where the floor was measured and where the solve happens.

Thickening is not free, so nothing is accepted silently:

  * **IoU(original, thickened) >= --min-iou.** This is the whole safeguard. A letter
    that needed 1 px scores 0.97 and is obviously still that letter; a fork that
    needed 10 px scores 0.44 and is a paddle. The default 0.90 keeps the first and
    refuses the second, and the manifest records the number either way.
  * **Topology must survive.** Dilation closes holes, and holes are exactly what
    `pw_h1` exists to measure. A thickened `8` that lost an eye is a different task,
    not a repaired one, so it is rejected unless --allow-topology-loss.

Rejected targets are NOT deleted. They stay below the floor and are the direct
evidence that the achievability ceiling is real -- report them as their own group.

    python scripts/thicken_targets.py                       # dry run + report
    python scripts/thicken_targets.py --write               # emit rescued/thickened/
    python scripts/thicken_targets.py --write --min-iou .95 # stricter
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

import _rescue_common as rc


def minimal_dilation(m: np.ndarray, floor: float, max_px: int = 40):
    """Smallest elliptical dilation whose 128 px downsample clears the floor."""
    for r in range(1, max_px + 1):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        d = cv2.dilate(m, k)
        s = rc.stroke_at_solve_res(d)
        if s >= floor:
            return r, d, s
    return None, None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floor", default="auto",
                    help="stroke floor, or 'auto' to measure it from the sweep")
    ap.add_argument("--min-iou", type=float, default=0.90,
                    help="reject if the thickened target keeps less than this much "
                         "of the original (IoU at 512 px). Default 0.90.")
    ap.add_argument("--allow-topology-loss", action="store_true",
                    help="accept even when dilation closes a hole. Off by default: "
                         "pw_h1 measures exactly what this destroys.")
    ap.add_argument("--out", default="rescued/thickened")
    ap.add_argument("--write", action="store_true", help="actually write the PNGs")
    a = ap.parse_args()

    floor, prov = rc.rig_floor(a.floor)
    cands = rc.below_floor(floor)
    print(f"rig floor {floor:.5f}  ({prov})")
    print(f"{len(cands)} targets below it at {rc.SOLVE_SIZE} px\n")

    out_dir = os.path.join(rc.BENCH, a.out)
    recs, kept = [], 0
    for d in sorted(cands, key=lambda x: x["id"]):
        src = os.path.join(rc.BENCH, d["target"])
        m = rc.load_mask(src)
        r, dm, s1 = minimal_dilation(m, floor)
        rec = {"id": d["id"], "subset": d["subset"], "class": d.get("class"),
               "source_target": d["target"], "stroke_before": d["stroke_128"]}
        if r is None:
            rec.update(accepted=False, reason="no dilation up to 40 px clears the floor")
            recs.append(rec); continue
        keep_iou = rc.iou(m, dm)
        h0, h1 = rc.n_holes_signif(m), rc.n_holes_signif(dm)
        rec.update(dilate_px=r, stroke_after=round(s1, 5),
                   area_gain=round(float(dm.sum() / m.sum()), 3),
                   iou_vs_original=round(keep_iou, 3),
                   holes_signif_before=h0, holes_signif_after=h1)
        if keep_iou < a.min_iou:
            rec.update(accepted=False,
                       reason=f"IoU {keep_iou:.3f} < --min-iou {a.min_iou}")
        elif h1 < h0 and not a.allow_topology_loss:
            rec.update(accepted=False,
                       reason=f"dilation closed {h0 - h1} significant hole(s)")
        else:
            rel = os.path.join(d["subset"], os.path.basename(d["target"]))
            rec.update(accepted=True, reason="", output=os.path.join(a.out, rel))
            if a.write:
                rc.save_mask(dm, os.path.join(out_dir, rel))
            kept += 1
        recs.append(rec)

    mpath = os.path.join(out_dir, "manifest.json")
    if a.write:
        rc.write_manifest(mpath, "thicken", floor, prov,
                          {"min_iou": a.min_iou,
                           "allow_topology_loss": a.allow_topology_loss}, recs)

    acc = [r for r in recs if r["accepted"]]
    print(f"accepted {kept} / {len(recs)}   rejected {len(recs) - kept}")
    if acc:
        px = sorted(r["dilate_px"] for r in acc)
        print(f"  dilation px: median {px[len(px) // 2]}, max {px[-1]}")
        print(f"  IoU kept   : min {min(r['iou_vs_original'] for r in acc):.3f}")
    from collections import Counter
    def bucket(why: str) -> str:
        if "min-iou" in why: return "kept too little of the original"
        if "hole" in why:    return "dilation closed a significant hole"
        return why
    for why, n in Counter(bucket(r["reason"]) for r in recs if not r["accepted"]).most_common():
        print(f"  rejected — {why}: {n}")
    print(("\nwrote " + os.path.relpath(out_dir, rc.BENCH) + "/ and manifest.json")
          if a.write else "\ndry run — pass --write to emit files")


if __name__ == "__main__":
    main()
