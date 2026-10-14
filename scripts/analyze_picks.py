"""How well does each metric agree with human picks from the picker page?

Reads a picks JSON (an ArtifactData export of picks/<viewer>) and a graded study.
v1 docs ({picks: {glyph: {best, second}}}) count best and second as good; v2 docs
({cma|adam: {glyph: {good: [...]}}}) use the marks directly.

    python scripts/analyze_picks.py <picks.json> [--tab cma] [--study objective_study]
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = [("iou", True), ("tc_iou", True), ("boundary_iou", True), ("nsd", True), ("chamfer", False),
           ("cldice", True), ("betti_error", False), ("holes_filled", False), ("cls_p", True), ("clip_p", True)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("picks")
    p.add_argument("--tab", default="cma")
    p.add_argument("--study", default=None)
    a = p.parse_args()
    doc = json.load(open(a.picks, encoding="utf-8"))
    doc = doc.get("data", doc)
    marks = {}
    if a.tab in doc:
        marks = {k: v for k, v in doc[a.tab].items() if v.get("good") or v.get("none")}
    elif "picks" in doc and a.tab == "cma":
        marks = {k: ({"none": True} if v.get("none") else {"good": [x for x in (v.get("best"), v.get("second")) if x]})
                 for k, v in doc["picks"].items()}
    study = a.study or ("objective_study" if a.tab == "cma" else "objective_study_adam")
    s0 = defaultdict(dict)
    for l in open(os.path.join(BENCH, "results", "fastshadow", study, "graded.jsonl"), encoding="utf-8"):
        g = json.loads(l)
        if g["seed"] == 0:
            s0[g["id"].replace("/", "__")][g["objective"]] = g
    judged = [k for k in marks if k in s0]
    good = {k: set(marks[k].get("good", [])) for k in judged}
    objs = sorted({o for k in judged for o in s0[k]})
    print(f"{len(judged)} judged, {sum(1 for k in judged if not good[k])} with none good, "
          f"{np.mean([len(good[k]) for k in judged]):.2f} good per glyph")
    print("\ngood-rate by objective")
    for o, r in sorted(((o, np.mean([o in good[k] for k in judged])) for o in objs), key=lambda x: -x[1]):
        print(f"  {o:12s} {100 * r:5.1f}%")
    print("\nmetric agreement (P@1 = metric's top shadow is good; AUC = good ranked above not-good)")
    rows = []
    for m, hb in METRICS:
        p1, auc, n1, na = 0.0, 0.0, 0, 0
        for k in judged:
            if not good[k]:
                continue
            sc = {o: s0[k][o].get(m) for o in objs}
            if any(v is None for v in sc.values()):
                continue
            sc = {o: (v if hb else -v) for o, v in sc.items()}
            top = max(sc.values())
            tops = [o for o, v in sc.items() if v >= top - 1e-9]
            p1 += sum(o in good[k] for o in tops) / len(tops)
            n1 += 1
            pos = [sc[o] for o in objs if o in good[k]]
            neg = [sc[o] for o in objs if o not in good[k]]
            if pos and neg:
                auc += np.mean([1.0 if x > y + 1e-9 else 0.5 if abs(x - y) <= 1e-9 else 0.0 for x in pos for y in neg])
                na += 1
        rows.append((m, p1 / max(n1, 1), auc / max(na, 1), n1))
    for m, p1, auc, n1 in sorted(rows, key=lambda r: -r[2]):
        print(f"  {m:13s} P@1 {100 * p1:5.1f}%   AUC {auc:.3f}   (n={n1})")


if __name__ == "__main__":
    main()
