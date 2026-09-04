#!/usr/bin/env python3
"""Apply reviewed rescue decisions to the benchmark.

Reads `rescued/decisions.json` (exported from the dashboard's Rescue view) and,
for every target the reviewer chose to rescue, writes the rescued PNG beside its
original and appends a metadata row for it.

**Originals are never modified, moved or deleted.** Both versions live in the
benchmark: the original keeps its id and gains `below_floor: true`, the rescue
gets a suffixed id and a `rescue` block naming what was done and what it cost.
That is not caution for its own sake -- it is what makes the pair useful. A
target the rig cannot cast and the same shape thickened until it can are a
matched pair differing in exactly one thing, the rig's physical limit, which is
a far better demonstration that an achievability ceiling exists than a bag of
targets that merely score badly.

Decisions map to operations as:

    thicken(r)    dilate by r px          fill   fill enclosed interior
    closefill(r)  close by r, then fill   keep   stays indexed, flagged below_floor
    drop          leaves metadata.jsonl for dropped.jsonl; the PNG is never deleted

`keep` and `drop` differ only in the index. A kept target still counts as part of
the benchmark and is the ceiling evidence; a dropped one is excluded from every
sweep and table but its file stays on disk, because "we removed this" and "we
deleted this" are different claims and only the first is being made.

Everything is measured after the fact with the benchmark's own
`compute_attributes`, and the run refuses to finish if any rescued target still
sits under the floor -- the decisions were made against a table, and this is the
check that the table did not lie.

    python scripts/promote_targets.py                 # dry run + report
    python scripts/promote_targets.py --write         # apply
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import cv2
import numpy as np
from scipy import ndimage

import _rescue_common as rc
from shape_attributes import compute_attributes

SUFFIX = {"thicken": lambda r: f"__thicken{r}",
          "fill": lambda r: "__fill",
          "closefill": lambda r: f"__closefill{r}"}


def _se(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def apply_op(m: np.ndarray, op: str, r: int) -> np.ndarray:
    if op == "thicken":
        return cv2.dilate(m, _se(r))
    if op == "fill":
        return ndimage.binary_fill_holes(m).astype(np.uint8)
    if op == "closefill":
        k = _se(r)
        return ndimage.binary_fill_holes(cv2.erode(cv2.dilate(m, k), k)).astype(np.uint8)
    raise ValueError(op)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", default="rescued/decisions.json")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite rescued PNGs that already exist")
    a = ap.parse_args()

    dec = json.load(open(os.path.join(rc.BENCH, a.decisions)))
    floor = float(dec["floor"])
    by_id = {d["id"]: d for d in dec["decisions"]}

    meta = [json.loads(l) for l in open(os.path.join(rc.BENCH, "metadata.jsonl"))]
    index = {d["id"]: d for d in meta}
    missing = [i for i in by_id if i not in index]
    if missing:
        raise SystemExit(f"{len(missing)} decisions name unknown ids, e.g. {missing[:3]}")

    # Idempotent: a decision whose rescued row is already indexed is a no-op, so a
    # re-run after adding decisions applies only the new ones instead of duplicating
    # every row it applied last time.
    new_rows, flagged, failures, report, skipped = [], 0, [], [], 0
    for d in dec["decisions"]:
        src = index[d["id"]]
        src["below_floor"] = True                     # true of every id in this file
        flagged += 1
        op, r = d["op"], int(d["radius_px"])
        if op in SUFFIX and (src["id"] + SUFFIX[op](int(d["radius_px"]))) in index:
            skipped += 1
            continue
        if op in ("keep", "drop"):
            if op == "drop":
                src["dropped"] = True          # out of the index, file untouched
            report.append({"id": d["id"], "op": op, "kept_original_only": True})
            continue

        m = rc.load_mask(os.path.join(rc.BENCH, src["target"]))
        out = apply_op(m, op, r)
        stem, ext = os.path.splitext(os.path.basename(src["target"]))
        rel = os.path.join(os.path.dirname(src["target"]), stem + SUFFIX[op](r) + ext)
        dst = os.path.join(rc.BENCH, rel)
        if os.path.exists(dst) and not a.write and not a.force:
            pass
        if a.write:
            if os.path.exists(dst) and not a.force:
                raise SystemExit(f"{rel} exists — pass --force to overwrite")
            rc.save_mask(out, dst)

        s1 = rc.stroke_at_solve_res(out)
        if s1 < floor:
            failures.append((d["id"], round(s1, 5)))
        row = dict(src)
        row["id"] = src["id"] + SUFFIX[op](r)
        row["target"] = rel
        row["below_floor"] = False
        row["attributes"] = compute_attributes(out * 255)
        row["rescue"] = {
            "derived_from": src["id"], "op": op, "radius_px": r,
            "rig_floor": round(floor, 5),
            "stroke_before": round(rc.stroke_at_solve_res(m), 5),
            "stroke_after": round(s1, 5),
            "iou_vs_original": round(rc.iou(m, out), 4),
            "area_gain": round(float(out.sum() / m.sum()), 3),
            "holes_signif_before": rc.n_holes_signif(m),
            "holes_signif_after": rc.n_holes_signif(out),
            "reviewed": bool(d.get("reviewed")),
        }
        row["shadows"] = {k: {kk: None for kk in v} for k, v in src["shadows"].items()}
        new_rows.append(row)
        report.append({"id": row["id"], **row["rescue"]})

    if failures:
        raise SystemExit("rescued targets still under the floor: " + repr(failures))

    dropped = [r for r in meta if r.get("dropped")]
    kept = [r for r in meta if r.get("below_floor") and not r.get("dropped")]
    print(f"decisions {len(dec['decisions'])} | originals flagged below_floor {flagged}")
    print(f"  of those: {len(kept)} kept in the index, {len(dropped)} moved to dropped.jsonl")
    print(f"rescued targets to add: {len(new_rows)}"
          + (f"  ({skipped} already applied, skipped)" if skipped else ""))
    print(f"benchmark size: {len(meta)} -> {len(meta) + len(new_rows)}")
    from collections import Counter
    print("  by op:", dict(Counter(d["op"] for d in dec["decisions"])))
    ious = [r["iou_vs_original"] for r in report if "iou_vs_original" in r]
    if ious:
        ious.sort()
        print(f"  IoU vs original: median {ious[len(ious)//2]:.3f}  min {ious[0]:.3f}")

    if not new_rows and not dropped:
        print("\nnothing to do — every decision is already applied")
        return
    if not a.write:
        print("\ndry run — pass --write to apply")
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    bak = os.path.join(rc.BENCH, f"metadata.jsonl.bak_{stamp}")
    shutil.copy2(os.path.join(rc.BENCH, "metadata.jsonl"), bak)
    keep_rows = [r for r in meta if not r.get("dropped")] + new_rows
    with open(os.path.join(rc.BENCH, "metadata.jsonl"), "w") as f:
        for row in keep_rows:
            f.write(json.dumps(row) + "\n")
    # Dropped rows leave the index but keep their provenance and their PNG.
    dpath = os.path.join(rc.BENCH, "dropped.jsonl")
    prev = [json.loads(l) for l in open(dpath)] if os.path.exists(dpath) else []
    seen = {r["id"] for r in prev}
    with open(dpath, "a") as f:
        for row in dropped:
            if row["id"] not in seen:
                f.write(json.dumps(row) + "\n")
    with open(os.path.join(rc.BENCH, "rescued", f"promote_report_{stamp}.json"), "w") as f:
        json.dump({"rig_floor": floor, "records": report}, f, indent=1)
    print(f"\nwrote {len(new_rows)} PNGs + metadata.jsonl "
          f"(backup: {os.path.basename(bak)})")
    if dropped:
        print(f"moved {len(dropped)} rows to dropped.jsonl — their PNGs are still on disk")


if __name__ == "__main__":
    main()
