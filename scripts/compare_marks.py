"""CMA-ES vs Adam, from the good-marks on the picker page.

Same 62 glyphs, same 13 objectives, seed 0 of each optimizer. Paired by
(glyph, objective): was the CMA-ES shadow marked good, was the Adam one? Also
compares machine metrics on exactly the pairs, so a human preference can be
checked against what changed in the shadows.

    python scripts/compare_marks.py <picks.json>
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def graded(d):
    out = {}
    for l in open(os.path.join(BENCH, "results", "fastshadow", d, "graded.jsonl"), encoding="utf-8"):
        g = json.loads(l)
        if g["seed"] == 0:
            out[(g["id"].replace("/", "__"), g["objective"])] = g
    return out


def sign_test(b, c):
    """two-sided exact binomial test on discordant pairs (McNemar exact)"""
    from math import comb

    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def main():
    doc = json.load(open(sys.argv[1], encoding="utf-8"))
    doc = doc.get("data", doc)
    C, A = doc.get("cma", {}), doc.get("adam", {})
    jc = {k for k, v in C.items() if v.get("good") or v.get("none")}
    ja = {k for k, v in A.items() if v.get("good") or v.get("none")}
    both = sorted(jc & ja)
    GC, GA = graded("objective_study"), graded("objective_study_adam")
    objs = sorted({o for (_, o) in GC})
    print(f"judged: CMA-ES {len(jc)}, Adam {len(ja)}, both {len(both)}")
    nc = [len(C[k].get("good", [])) for k in both]
    na = [len(A[k].get("good", [])) for k in both]
    print(f"good shadows per glyph: CMA-ES {np.mean(nc):.2f}, Adam {np.mean(na):.2f}  "
          f"(Adam more on {sum(a > c for a, c in zip(na, nc))}, fewer on {sum(a < c for a, c in zip(na, nc))}, "
          f"same on {sum(a == c for a, c in zip(na, nc))} glyphs)")
    print(f"'none look like it': CMA-ES {sum(bool(C[k].get('none')) for k in both)}, Adam {sum(bool(A[k].get('none')) for k in both)}")
    print(f"glyphs with at least one good: CMA-ES {sum(x > 0 for x in nc)}, Adam {sum(x > 0 for x in na)}")
    # paired (glyph, objective)
    b = c = 0  # b: Adam good & CMA not, c: CMA good & Adam not
    per = defaultdict(lambda: [0, 0, 0, 0])  # cma_good, adam_good, b, c
    for k in both:
        gc, ga = set(C[k].get("good", [])), set(A[k].get("good", []))
        for o in objs:
            x, y = o in gc, o in ga
            per[o][0] += x
            per[o][1] += y
            if y and not x:
                b += 1
                per[o][2] += 1
            if x and not y:
                c += 1
                per[o][3] += 1
    n = len(both) * len(objs)
    print(f"\npaired shadows ({n}): good under CMA-ES {sum(v[0] for v in per.values())}, under Adam {sum(v[1] for v in per.values())}; "
          f"Adam-only {b}, CMA-only {c}, exact McNemar p = {sign_test(b, c):.2g}")
    print("\nby objective: CMA-ES good-rate -> Adam good-rate (Adam-only / CMA-only)")
    rows = []
    for o in objs:
        cg, ag, bb, cc = per[o]
        rows.append((ag / len(both) - cg / len(both), o, cg, ag, bb, cc))
    for d, o, cg, ag, bb, cc in sorted(rows, reverse=True):
        print(f"  {o:11s} {100 * cg / len(both):5.1f}% -> {100 * ag / len(both):5.1f}%  ({d * 100:+5.1f})  {bb}/{cc}  p={sign_test(bb, cc):.2g}")
    for sub in ("letters_upper", "letters_lower", "digits"):
        ks = [k for k in both if k.startswith(sub)]
        print(f"  {sub:14s} good per glyph CMA-ES {np.mean([len(C[k].get('good', [])) for k in ks]):.2f} -> "
              f"Adam {np.mean([len(A[k].get('good', [])) for k in ks]):.2f}  (n={len(ks)})")
    # what changed in the shadows, on the same pairs
    print("\nmachine metrics on the same pairs (CMA-ES -> Adam):")
    for m, hb in [("iou", 1), ("tc_iou", 1), ("cldice", 1), ("betti_error", 0), ("cls_top1", 1), ("cls_p", 1),
                  ("clip_top1", 1), ("holes_filled", 0)]:
        xs = [(GC[(k, o)].get(m), GA[(k, o)].get(m)) for k in both for o in objs
              if (k, o) in GC and (k, o) in GA]
        xs = [(a, b) for a, b in xs if a is not None and b is not None]
        d = np.array([b - a for a, b in xs])
        print(f"  {m:12s} {np.mean([a for a, _ in xs]):.3f} -> {np.mean([b for _, b in xs]):.3f}  "
              f"(Δ {d.mean():+.3f} ± {d.std(ddof=1) / np.sqrt(len(d)):.3f})")
    # does the human preference follow a metric? per pair, sign of Δmark vs sign of Δmetric
    print("\nwhen your mark flips between the two, which metric moved the same way?")
    for m, hb in [("iou", 1), ("cldice", 1), ("cls_p", 1), ("clip_p", 1), ("betti_error", -1), ("tc_iou", 1)]:
        agree = tot = 0
        for k in both:
            gc, ga = set(C[k].get("good", [])), set(A[k].get("good", []))
            for o in objs:
                x, y = o in gc, o in ga
                if x == y or (k, o) not in GC or (k, o) not in GA:
                    continue
                va, vb = GC[(k, o)].get(m), GA[(k, o)].get(m)
                if va is None or vb is None or va == vb:
                    continue
                tot += 1
                agree += ((vb - va) * hb > 0) == y
        print(f"  {m:12s} {100 * agree / max(tot, 1):5.1f}% of {tot} flips")
    gs = [g for g in doc.get("guesses", []) if g[1].startswith("adam:")]
    print(f"\nguesses on Adam shadows so far: {len(gs)}")


if __name__ == "__main__":
    main()
