"""Blind A/B results (CMA-ES vs Adam) from the picker page.

    python scripts/analyze_ab.py <picks.json>
"""

import json
import os
import sys
from collections import defaultdict
from math import comb

import numpy as np

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sign_p(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def graded(d):
    out = {}
    for l in open(os.path.join(BENCH, "results", "fastshadow", d, "graded.jsonl"), encoding="utf-8"):
        g = json.loads(l)
        if g["seed"] == 0:
            out[(g["id"].replace("/", "__"), g["objective"])] = g
    return out


def main():
    doc = json.load(open(sys.argv[1], encoding="utf-8"))
    doc = doc.get("data", doc)
    R = [r for r in doc.get("ab", []) if r[2] != "identical"]
    GC, GA = graded("objective_study"), graded("objective_study_adam")
    n = len(R)
    a, c = sum(r[2] == "adam" for r in R), sum(r[2] == "cma" for r in R)
    t, b = sum(r[2] == "tie" for r in R), sum(r[2] == "bad" for r in R)
    print(f"{n} pairs: Adam {a} ({100 * a / n:.0f}%), CMA-ES {c} ({100 * c / n:.0f}%), same {t}, neither {b} ({100 * b / n:.0f}%)")
    print(f"decided {a + c}: Adam wins {100 * a / max(a + c, 1):.1f}%  sign test p = {sign_p(a, c):.3g}")
    side = sum(1 for r in R if r[2] in ("adam", "cma") and ((r[2] == "cma") == (r[3] == "L")))
    print(f"left chosen in {side}/{a + c} decided pairs (side bias check, ~50% expected)")
    print(f"median response {np.median([r[4] for r in R]) / 1000:.1f}s")
    print("\nby objective (Adam/CMA-ES/same/neither, Adam share of decided):")
    by = defaultdict(lambda: [0, 0, 0, 0])
    for r in R:
        by[r[1]][["adam", "cma", "tie", "bad"].index(r[2])] += 1
    for o, v in sorted(by.items(), key=lambda kv: -(kv[1][0] / max(kv[1][0] + kv[1][1], 1))):
        dec = v[0] + v[1]
        print(f"  {o:11s} {v[0]:2d}/{v[1]:2d}/{v[2]:2d}/{v[3]:2d}  {100 * v[0] / max(dec, 1):5.1f}%  p={sign_p(v[0], v[1]):.2g}")
    print("\nby set:")
    for sub in ("letters_upper", "letters_lower", "digits"):
        rs = [r for r in R if r[0].startswith(sub)]
        aa, cc = sum(r[2] == "adam" for r in rs), sum(r[2] == "cma" for r in rs)
        print(f"  {sub:14s} n={len(rs):3d} Adam {aa} CMA-ES {cc} neither {sum(r[2] == 'bad' for r in rs)}  p={sign_p(aa, cc):.2g}")
    print("\ndoes a metric's difference predict your choice? (decided pairs: metric favours the one you chose)")
    for m, hb in [("iou", 1), ("cldice", 1), ("cls_p", 1), ("clip_p", 1), ("betti_error", -1), ("tc_iou", 1), ("boundary_iou", 1)]:
        agree = tot = 0
        for r in R:
            if r[2] not in ("adam", "cma"):
                continue
            va, vc = GA[(r[0], r[1])].get(m), GC[(r[0], r[1])].get(m)
            if va is None or vc is None or va == vc:
                continue
            tot += 1
            agree += ((va - vc) * hb > 0) == (r[2] == "adam")
        print(f"  {m:12s} {100 * agree / max(tot, 1):5.1f}% of {tot}")
    # which glyphs are "neither" for both: still unsolved
    nb = defaultdict(int)
    for r in R:
        if r[2] == "bad":
            nb[r[0].split("__")[1].split("_")[0]] += 1
    print("\n'neither' most often for:", ", ".join(f"{k}×{v}" for k, v in sorted(nb.items(), key=lambda x: -x[1])[:12]))


if __name__ == "__main__":
    main()
