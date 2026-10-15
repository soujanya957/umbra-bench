"""Guess-the-glyph results from the picker page: which objective's shadows can a
person read, and which machine judge or metric predicts that?

    python scripts/analyze_guesses.py <picks.json>
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = {"cma": "objective_study", "adam": "objective_study_adam"}
METRICS = [("iou", True), ("tc_iou", True), ("boundary_iou", True), ("nsd", True), ("chamfer", False),
           ("cldice", True), ("betti_error", False), ("cls_p", True), ("clip_p", True)]


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def auc(pos, neg):
    if not pos or not neg:
        return None
    return float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg]))


def main():
    doc = json.load(open(sys.argv[1], encoding="utf-8"))
    doc = doc.get("data", doc)
    G = doc.get("guesses", [])
    graded = {}
    for tk, d in STUDY.items():
        p = os.path.join(BENCH, "results", "fastshadow", d, "graded.jsonl")
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                g = json.loads(l)
                if g["seed"] == 0:
                    graded[(tk, g["objective"], g["id"].replace("/", "__"))] = g
    rows = []
    for gk, cond, ans, ok, ms in G:
        tk, o = cond.split(":")
        m = graded.get((tk, o, gk))
        if m:
            rows.append({"g": gk, "tk": tk, "o": o, "ans": ans, "ok": ok, "ms": ms, "m": m,
                         "sub": gk.split("__")[0]})
    n = len(rows)
    k = sum(r["ok"] for r in rows)
    print(f"{n} guesses, {100 * k / n:.1f}% right; can't-tell {sum(r['ans'] == '?' for r in rows)}; "
          f"distinct items {len({(r['tk'], r['o'], r['g']) for r in rows})}")
    print(f"same shadows: classifier top-1 {100 * np.mean([r['m']['cls_top1'] for r in rows]):.1f}%, "
          f"CLIP top-1 {100 * np.mean([r['m']['clip_top1'] for r in rows]):.1f}%")
    print("\nby set:")
    for sub in ("letters_upper", "letters_lower", "digits"):
        rs = [r for r in rows if r["sub"] == sub]
        if rs:
            lo, hi = wilson(sum(r["ok"] for r in rs), len(rs))
            print(f"  {sub:14s} n={len(rs):3d}  you {100 * np.mean([r['ok'] for r in rs]):5.1f}% ({100 * lo:.0f}-{100 * hi:.0f})"
                  f"  classifier {100 * np.mean([r['m']['cls_top1'] for r in rs]):5.1f}%  CLIP {100 * np.mean([r['m']['clip_top1'] for r in rs]):5.1f}%")
    print("\nby objective (you | classifier | CLIP on the same shadows):")
    by = defaultdict(list)
    for r in rows:
        by[(r["tk"], r["o"])].append(r)
    out = []
    for (tk, o), rs in by.items():
        kk = sum(r["ok"] for r in rs)
        lo, hi = wilson(kk, len(rs))
        out.append((kk / len(rs), tk, o, len(rs), lo, hi, np.mean([r["m"]["cls_top1"] for r in rs]),
                    np.mean([r["m"]["clip_top1"] for r in rs]), np.median([r["ms"] for r in rs]) / 1000))
    for acc, tk, o, nn, lo, hi, c, cl, ms in sorted(out, reverse=True):
        print(f"  {tk}:{o:11s} n={nn:3d}  you {100 * acc:5.1f}% ({100 * lo:.0f}-{100 * hi:.0f})  "
              f"classifier {100 * c:5.1f}%  CLIP {100 * cl:5.1f}%  median {ms:.1f}s")
    print("\nwhich metric predicts that you read a shadow (AUC over guesses; 0.5 = chance):")
    res = []
    for mk, hb in METRICS:
        pos = [(r["m"][mk] if hb else -r["m"][mk]) for r in rows if r["ok"] and r["m"].get(mk) is not None]
        neg = [(r["m"][mk] if hb else -r["m"][mk]) for r in rows if not r["ok"] and r["m"].get(mk) is not None]
        res.append((auc(pos, neg), mk))
    for a, mk in sorted(res, reverse=True):
        print(f"  {mk:13s} AUC {a:.3f}")
    agree_c = np.mean([r["ok"] == r["m"]["cls_top1"] for r in rows])
    agree_l = np.mean([r["ok"] == r["m"]["clip_top1"] for r in rows])
    print(f"\nper-shadow agreement with you: classifier {100 * agree_c:.1f}%, CLIP {100 * agree_l:.1f}%")
    # wrong answers: which characters get confused
    conf = defaultdict(int)
    for r in rows:
        if not r["ok"] and r["ans"] != "?":
            conf[(r["g"].split("__")[1].split("_")[0], r["ans"])] += 1
    print("most common confusions (target -> your answer):",
          ", ".join(f"{a}->{b} ×{c}" for (a, b), c in sorted(conf.items(), key=lambda x: -x[1])[:12]))
    # marks vs guesses: are shadows you marked good also read correctly?
    marks = doc.get("cma", {})
    mk_rows = [r for r in rows if r["tk"] == "cma" and r["g"] in marks]
    if mk_rows:
        good = [r["ok"] for r in mk_rows if r["o"] in set(marks[r["g"]].get("good", []))]
        bad = [r["ok"] for r in mk_rows if r["o"] not in set(marks[r["g"]].get("good", []))]
        print(f"\nshadows you marked good: read correctly {100 * np.mean(good):.1f}% (n={len(good)}); "
              f"not marked: {100 * np.mean(bad):.1f}% (n={len(bad)})")


if __name__ == "__main__":
    main()
