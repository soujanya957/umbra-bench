#!/usr/bin/env python3
"""Collapse the sequence sweep into one CSV and print the two comparisons it exists to make.

    budget:      small vs big          — does more search buy a better clip
    deformation: distort vs baseline   — does bending the figure buy a better clip

The second comparison has one trap, and it is the whole reason `iou_vs_original`
is carried through every run: a distort run is scored against a target *it moved*.
Its headline IoU is not comparable to the baseline's, and reading it as if it were
is how "deformation helps" gets claimed. Only `iou_vs_original` — IoU against the
frames as authored — is comparable across conditions, so that is what the
distort-vs-baseline delta is computed from.
"""

import argparse
import csv
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)

CONDITIONS = ["small-budget", "big-budget",
              "small-budget-distort", "big-budget-distort"]

COLS = ["condition", "sequence_id", "group", "name", "n_frames", "cyclic",
        "mean_iou", "mean_iou_vs_original", "min_iou", "first_frame_iou",
        "mean_dq", "max_dq", "cheap_frac", "loop_dq", "loop_cheap",
        "warp_rms_px", "warp_peak_px", "seconds"]


def load(root):
    rows = {}
    for cond in CONDITIONS:
        d = os.path.join(root, cond)
        if not os.path.isdir(d):
            continue
        for dirpath, _, files in os.walk(d):
            if "results.json" not in files:
                continue
            r = json.load(open(os.path.join(dirpath, "results.json")))
            w = r.get("warp") or {}
            rows[(cond, r["sequence_id"])] = {
                "condition": cond, "sequence_id": r["sequence_id"],
                "group": r["group"], "name": r["name"],
                "n_frames": r["n_frames"], "cyclic": r["cyclic"],
                "mean_iou": r["mean_iou"],
                "mean_iou_vs_original": r["mean_iou_vs_original"],
                "min_iou": r["min_iou"], "first_frame_iou": r["first_frame_iou"],
                "mean_dq": r["mean_dq"], "max_dq": r["max_dq"],
                "cheap_frac": r["cheap_frac"],
                "loop_dq": r["loop_dq"], "loop_cheap": r["loop_cheap"],
                "warp_rms_px": w.get("rms_disp_px"),
                "warp_peak_px": w.get("peak_disp_px"),
                "seconds": r["seconds"],
            }
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def compare(rows, a, b, label, field="mean_iou_vs_original"):
    ids = sorted({s for (c, s) in rows if c == a} & {s for (c, s) in rows if c == b})
    if not ids:
        print(f"\n{label}: no paired sequences yet")
        return
    print(f"\n{label}   ({field}, {len(ids)} paired sequences)")
    print(f"  {'sequence':22s} {a[:16]:>9s} {b[:16]:>9s} {'delta':>8s}")
    deltas = []
    for s in ids:
        va, vb = rows[(a, s)][field], rows[(b, s)][field]
        if va is None or vb is None:
            continue
        deltas.append(vb - va)
        print(f"  {s:22s} {va:9.3f} {vb:9.3f} {vb - va:+8.3f}")
    if deltas:
        wins = sum(1 for d in deltas if d > 0)
        print(f"  {'MEAN':22s} {mean([rows[(a, s)][field] for s in ids]):9.3f} "
              f"{mean([rows[(b, s)][field] for s in ids]):9.3f} "
              f"{mean(deltas):+8.3f}   ({wins}/{len(deltas)} sequences improved)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.join(_BENCH, "optimized", "anim-optimizer"))
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = a.out or os.path.join(a.root, "sequences_summary.csv")

    rows = load(a.root)
    if not rows:
        print(f"no results under {a.root}")
        return

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    print(f"wrote {out}  ({len(rows)} rows)")

    for cond in CONDITIONS:
        rs = [v for (c, _), v in rows.items() if c == cond]
        if not rs:
            continue
        print(f"\n[{cond}]  {len(rs)}/15 sequences")
        print(f"  mean IoU (vs shown)    {mean([r['mean_iou'] for r in rs]):.4f}")
        print(f"  mean IoU (vs authored) {mean([r['mean_iou_vs_original'] for r in rs]):.4f}")
        print(f"  interior cheap         {100 * mean([r['cheap_frac'] for r in rs]):.1f}%")
        loops = [r for r in rs if r["loop_cheap"] is not None]
        if loops:
            print(f"  loop-closing cheap     "
                  f"{100 * mean([float(r['loop_cheap']) for r in loops]):.1f}%  "
                  f"(n={len(loops)} cyclic)")

    compare(rows, "small-budget", "big-budget", "BUDGET: small -> big")
    compare(rows, "small-budget", "small-budget-distort",
            "DEFORMATION: baseline -> distort (small budget)")
    compare(rows, "big-budget", "big-budget-distort",
            "DEFORMATION: baseline -> distort (big budget)")


if __name__ == "__main__":
    main()
