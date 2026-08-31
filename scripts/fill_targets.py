#!/usr/bin/env python3
"""Fill the interior of outline-style targets so the rig can cast them.

Some sub-floor targets are thin for a reason dilation handles badly: they are
*outlines* of solid objects, not thin objects. A drawn ring, a hollow letter, a
pocket rendered as its own contour -- the strokes are hairline but the thing they
enclose is a perfectly castable blob. Filling the interior recovers the intended
silhouette, keeps the outer boundary EXACTLY where it was, and needs no dilation.

Where thicken_targets.py grows a shape outward, this changes only what is inside
it, so the outline never moves. That is the whole reason to prefer it when it
applies -- and the reason it applies to fewer targets.

**Filling destroys topology by construction.** That is the operation, not a side
effect: after this there are no holes. For an outline of a solid object that is
correct; for an `8` or a butterfly it silently replaces the task with an easier
one, and `pw_h1` will never tell you, because the target it is scored against no
longer has the holes. Two things follow, and neither is optional:

  * `--max-area-gain` bounds how much of the result was hole. A shape that more
    than doubles when filled was mostly negative space, and filling it is not
    recovering a silhouette, it is drawing a new one. Default 2.0.
  * Every accepted item is written with `needs_review: true` and its destroyed
    significant-hole count. **Look at these before promoting them.** No threshold
    can tell an outlined pocket from a butterfly; a person can, at a glance.

    python scripts/fill_targets.py                 # dry run + report
    python scripts/fill_targets.py --write         # emit rescued/filled/
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy import ndimage

import _rescue_common as rc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floor", default="auto",
                    help="stroke floor, or 'auto' to measure it from the sweep")
    ap.add_argument("--max-area-gain", type=float, default=2.0,
                    help="reject if filling multiplies the area by more than this — "
                         "the shape was mostly hole. Default 2.0.")
    ap.add_argument("--out", default="rescued/filled")
    ap.add_argument("--write", action="store_true", help="actually write the PNGs")
    a = ap.parse_args()

    floor, prov = rc.rig_floor(a.floor)
    cands = rc.below_floor(floor)
    print(f"rig floor {floor:.5f}  ({prov})")
    print(f"{len(cands)} targets below it at {rc.SOLVE_SIZE} px\n")

    out_dir = os.path.join(rc.BENCH, a.out)
    recs, kept = [], 0
    for d in sorted(cands, key=lambda x: x["id"]):
        m = rc.load_mask(os.path.join(rc.BENCH, d["target"]))
        f = ndimage.binary_fill_holes(m).astype(np.uint8)
        s1 = rc.stroke_at_solve_res(f)
        gain = float(f.sum() / m.sum())
        h0, h1 = rc.n_holes_signif(m), rc.n_holes_signif(f)
        rec = {"id": d["id"], "subset": d["subset"], "class": d.get("class"),
               "source_target": d["target"], "stroke_before": d["stroke_128"],
               "stroke_after": round(s1, 5), "area_gain": round(gain, 3),
               "iou_vs_original": round(rc.iou(m, f), 3),
               "holes_signif_before": h0, "holes_signif_after": h1,
               "holes_signif_destroyed": h0 - h1}
        if gain < 1.02:
            rec.update(accepted=False, reason="nothing to fill (no enclosed interior)")
        elif s1 < floor:
            rec.update(accepted=False,
                       reason=f"still below the floor after filling ({s1:.4f})")
        elif gain > a.max_area_gain:
            rec.update(accepted=False,
                       reason=f"area x{gain:.2f} > --max-area-gain {a.max_area_gain}")
        else:
            rel = os.path.join(d["subset"], os.path.basename(d["target"]))
            rec.update(accepted=True, reason="", needs_review=True,
                       output=os.path.join(a.out, rel))
            if a.write:
                rc.save_mask(f, os.path.join(out_dir, rel))
            kept += 1
        recs.append(rec)

    if a.write:
        rc.write_manifest(os.path.join(out_dir, "manifest.json"), "fill", floor, prov,
                          {"max_area_gain": a.max_area_gain}, recs)

    acc = [r for r in recs if r["accepted"]]
    print(f"accepted {kept} / {len(recs)}   rejected {len(recs) - kept}")
    if acc:
        print(f"  area gain  : median x{sorted(r['area_gain'] for r in acc)[len(acc) // 2]:.2f}"
              f", max x{max(r['area_gain'] for r in acc):.2f}")
        print(f"  significant holes destroyed: "
              f"{sum(r['holes_signif_destroyed'] for r in acc)} across {len(acc)} targets")
        print("  ALL accepted items are flagged needs_review — eyeball them before promoting.")
    from collections import Counter
    def bucket(w): return ("area gain over the cap" if "area x" in w
                           else "still below the floor" if "still below" in w else w)
    for why, n in Counter(bucket(r["reason"]) for r in recs if not r["accepted"]).most_common():
        print(f"  rejected — {why}: {n}")
    print(("\nwrote " + os.path.relpath(out_dir, rc.BENCH) + "/ and manifest.json")
          if a.write else "\ndry run — pass --write to emit files")


if __name__ == "__main__":
    main()
