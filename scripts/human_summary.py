"""Summarise the picker page's human data for the objective report.

Writes results/fastshadow/human_summary.json from an ArtifactData export of
picks/<viewer>: guess-the-glyph accuracy per objective (with the classifier's and
CLIP's top-1 on the same shadows), which metric predicts a correct read, and the
good-marks (rate per objective, read rate of marked vs unmarked shadows).

    python scripts/human_summary.py <picks.json>
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BENCH, "scripts"))
from analyze_guesses import METRICS, STUDY, auc, wilson  # noqa: E402

LABEL = {"iou": "IoU", "tc_iou": "tc_iou", "boundary_iou": "boundary IoU", "nsd": "NSD", "chamfer": "chamfer",
         "cldice": "clDice", "betti_error": "Betti error", "cls_p": "glyph classifier", "clip_p": "CLIP (ViT-B-32)"}


def main():
    doc = json.load(open(sys.argv[1], encoding="utf-8"))
    doc = doc.get("data", doc)
    graded = {}
    for tk, d in STUDY.items():
        p = os.path.join(BENCH, "results", "fastshadow", d, "graded.jsonl")
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                g = json.loads(l)
                if g["seed"] == 0:
                    graded[(tk, g["objective"], g["id"].replace("/", "__"))] = g
    rows = []
    for gk, cond, ans, ok, ms in doc.get("guesses", []):
        tk, o = cond.split(":")
        m = graded.get((tk, o, gk))
        if m:
            rows.append({"g": gk, "tk": tk, "o": o, "ans": ans, "ok": ok, "m": m})
    by = defaultdict(list)
    for r in rows:
        by[(r["tk"], r["o"])].append(r)
    per = []
    for (tk, o), rs in by.items():
        k = sum(r["ok"] for r in rs)
        lo, hi = wilson(k, len(rs))
        per.append({"tk": tk, "o": o, "n": len(rs), "you": k / len(rs), "lo": lo, "hi": hi,
                    "cls": float(np.mean([r["m"]["cls_top1"] for r in rs])),
                    "clip": float(np.mean([r["m"]["clip_top1"] for r in rs]))})
    per.sort(key=lambda x: -x["you"])
    pred = []
    for mk, hb in METRICS:
        pos = [(r["m"][mk] if hb else -r["m"][mk]) for r in rows if r["ok"] and r["m"].get(mk) is not None]
        neg = [(r["m"][mk] if hb else -r["m"][mk]) for r in rows if not r["ok"] and r["m"].get(mk) is not None]
        a = auc(pos, neg)
        if a is not None:
            pred.append({"k": mk, "label": LABEL[mk], "auc": a})
    pred.sort(key=lambda x: -x["auc"])
    marks = doc.get("cma", {})
    mk_rows = [r for r in rows if r["tk"] == "cma" and r["g"] in marks]
    good = [r["ok"] for r in mk_rows if r["o"] in set(marks[r["g"]].get("good", []))]
    bad = [r["ok"] for r in mk_rows if r["o"] not in set(marks[r["g"]].get("good", []))]
    judged = [k for k, v in marks.items() if v.get("good") or v.get("none")]
    objs = sorted({r["o"] for r in rows})
    rate = sorted(({"o": o, "rate": float(np.mean([o in set(marks[k].get("good", [])) for k in judged]))}
                   for o in objs), key=lambda x: -x["rate"]) if judged else []
    conf = defaultdict(int)
    for r in rows:
        if not r["ok"] and r["ans"] != "?":
            conf[(r["g"].split("__")[1].split("_")[0], r["ans"])] += 1
    from math import comb

    def sign_p(b, c):
        n, k = b + c, min(b, c)
        return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)

    by_opt = {}
    for tk in sorted({r["tk"] for r in rows}):
        rs = [r for r in rows if r["tk"] == tk]
        by_opt[tk] = {"n": len(rs), "you": float(np.mean([r["ok"] for r in rs])),
                      "cls": float(np.mean([r["m"]["cls_top1"] for r in rs])),
                      "clip": float(np.mean([r["m"]["clip_top1"] for r in rs]))}
    paired = None
    pc = {(r["g"], r["o"]): r["ok"] for r in rows if r["tk"] == "cma"}
    pa = {(r["g"], r["o"]): r["ok"] for r in rows if r["tk"] == "adam"}
    both = [k for k in pc if k in pa]
    if both:
        b = sum(1 for k in both if pa[k] and not pc[k])
        c = sum(1 for k in both if pc[k] and not pa[k])
        paired = {"n": len(both), "adam_only": b, "cma_only": c, "p": sign_p(b, c),
                  "cma": float(np.mean([pc[k] for k in both])), "adam": float(np.mean([pa[k] for k in both]))}
    abr = [r for r in doc.get("ab", []) if r[2] != "identical"]
    ab = None
    if abr:
        a_, c_ = sum(r[2] == "adam" for r in abr), sum(r[2] == "cma" for r in abr)
        ab = {"n": len(abr), "adam": a_, "cma": c_, "tie": sum(r[2] == "tie" for r in abr),
              "bad": sum(r[2] == "bad" for r in abr), "p": sign_p(a_, c_), "by": []}
        for o in sorted({r[1] for r in abr}):
            rs = [r for r in abr if r[1] == o]
            ao, co = sum(r[2] == "adam" for r in rs), sum(r[2] == "cma" for r in rs)
            ab["by"].append({"o": o, "adam": ao, "cma": co, "tie": sum(r[2] == "tie" for r in rs),
                             "bad": sum(r[2] == "bad" for r in rs), "p": sign_p(ao, co)})
    out = {
        "by_opt": by_opt, "paired": paired, "ab": ab,
        "n": len(rows), "you": float(np.mean([r["ok"] for r in rows])),
        "cls": float(np.mean([r["m"]["cls_top1"] for r in rows])),
        "clip": float(np.mean([r["m"]["clip_top1"] for r in rows])),
        "cant": sum(r["ans"] == "?" for r in rows),
        "agree_cls": float(np.mean([r["ok"] == r["m"]["cls_top1"] for r in rows])),
        "agree_clip": float(np.mean([r["ok"] == r["m"]["clip_top1"] for r in rows])),
        "per": per, "pred": pred,
        "marked_read": float(np.mean(good)) if good else None, "n_marked": len(good),
        "unmarked_read": float(np.mean(bad)) if bad else None, "n_unmarked": len(bad),
        "n_judged": len(judged), "good_per_glyph": float(np.mean([len(marks[k].get("good", [])) for k in judged])) if judged else 0,
        "mark_rate": rate,
        "confusions": [{"t": a, "a": b, "n": c} for (a, b), c in sorted(conf.items(), key=lambda x: -x[1])[:10]],
        "optimizers": sorted({r["tk"] for r in rows}),
    }
    p = os.path.join(BENCH, "results", "fastshadow", "human_summary.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=2)
    print("wrote", p, f"{out['n']} guesses, optimizers {out['optimizers']}")


if __name__ == "__main__":
    main()
