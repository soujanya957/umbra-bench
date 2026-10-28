"""Analyse the blind shadow reading test (make_target_test_page.py) for the target designer.

Input: results/fastshadow/target_designer/human/tests/<uid>.json (an ArtifactData export of
tests/<uid>) plus report.json. Output: human/summary.json, printed too.

Pre-registered (protocol_review.json, before the run):
  check 1    chosen vs today's typed read rate, paired by character (both blocks; exact McNemar).
  check 2    chosen-font shadows with IoU >= 0.75 must read at >= about 75%, lower 95% Wilson CI
             above 0.65: does own-target IoU still certify reading on the new fonts?
Scoring: case-insensitive (the page names the category: capital / lowercase / digit); "can't tell"
counts as wrong. Everything else is descriptive or sensitivity, chosen after the answers came in:
  block 1    each character's first read, 31/31 per rater (no carry-over from the other arm);
  scoring    strict case, and look-alike lenient (O/o typed 0, I typed l);
  attempted  accuracy among answered items ("can't tell" removed);
  pre-seen   without W l Q f u 7, whose test shadows were on the overview/work-log pages;
  A/B        per character where both showings agree (a split pair is often a side habit);
  predictors within-arm AUC of own IoU, classifier p, blob floor, and IoU above the blob
             (= IoU of the shown solve minus the target's convex-hull IoU).

    python scripts/analyze_target_test.py
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, fisher_exact

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastshadow  # noqa: E402

TD = os.path.join(fastshadow.BENCH, "results", "fastshadow", "target_designer")
OUT = os.path.join(TD, "human", "summary.json")


LOOKALIKE = {("O", "0"), ("o", "0"), ("I", "l")}  # (character, typed): the same shape on the wall
PRESEEN = set("WlQfu7")


def ok(rec, strict=False, lenient=False):
    a = (rec["answer"] or "").strip()
    if lenient and (rec["ch"], a) in LOOKALIKE:
        return True
    return bool(a) and (a == rec["ch"] if strict else a.lower() == rec["ch"].lower())


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def boot_diff(a, b, g, n=10000):
    """Bootstrap CI of mean(a) - mean(b), resampling each group (independent characters)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [a[g.integers(0, len(a), len(a))].mean() - b[g.integers(0, len(b), len(b))].mean() for _ in range(n)]
    return [float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def mcnemar(b, c):
    return float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0


def auc(score, y):
    s, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def rater(doc, rep, g):
    reads, ab = doc["reads"], doc["ab"]
    per = {x["ch"]: x for x in rep["per_char"]}
    sec = rep["summary"]["secondary"]["per_char"]
    R = [dict(r, correct=ok(r), correct_strict=ok(r, True), correct_lenient=ok(r, lenient=True),
              cant=not (r["answer"] or "").strip()) for r in reads.values()]
    b1 = [r for r in R if r["block"] == 1]
    out = {"n_reads": len(R), "n_ab": len(ab)}

    def paired(key="correct", skip=()):
        by_ = defaultdict(dict)
        for r in R:
            if r["ch"] not in skip:
                by_[r["ch"]][r["arm"]] = r
        pp = [(v["chosen"][key], v["bench"][key]) for v in by_.values() if len(v) == 2]
        co, bo = sum(c and not b for c, b in pp), sum(b and not c for c, b in pp)
        d = np.array([float(c) - float(b) for c, b in pp])
        bs = [d[g.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        return {"n_chars": len(pp), "chosen_read": sum(c for c, _ in pp), "bench_read": sum(b for _, b in pp),
                "chosen_only": co, "bench_only": bo, "mcnemar_p": mcnemar(co, bo),
                "diff_ci": [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    # pre-registered checks
    hi = [r for r in R if r["arm"] == "chosen" and per[r["ch"]]["chosen"] >= 0.75]
    k2, n2 = sum(r["correct"] for r in hi), len(hi)
    lo2, hi2 = wilson(k2, n2)
    bh = [r for r in R if r["arm"] == "bench" and per[r["ch"]]["bench"] >= 0.75]
    out["preregistered"] = {
        "check1_paired": paired(),
        "check2_chosen_iou_ge_075": {"read": [k2, n2], "rate": k2 / n2 if n2 else None, "wilson95": [lo2, hi2],
                                     "p_vs_075": float(binomtest(k2, n2, 0.75, alternative="less").pvalue) if n2 else None,
                                     "pass": bool(n2 and k2 / n2 >= 0.75 and lo2 > 0.65),
                                     "bench_same_rule": [sum(r["correct"] for r in bh), len(bh)]}}
    out["sensitivity"] = {
        "strict_case": paired("correct_strict"), "lookalike_lenient": paired("correct_lenient"),
        "without_preseen": paired(skip=PRESEEN),
        "attempted_accuracy": {arm: [sum(r["correct"] for r in R if r["arm"] == arm and not r["cant"]),
                                     sum(r["arm"] == arm and not r["cant"] for r in R)] for arm in ("chosen", "bench")},
        "discordant_by_subset": {sub: {k: v for k, v in paired(skip={r["ch"] for r in R if r["sub"] != sub}).items()
                                       if k in ("chosen_only", "bench_only")}
                                 for sub in ("letters_upper", "letters_lower", "digits")}}

    # primary: block 1
    c1 = [r["correct"] for r in b1 if r["arm"] == "chosen"]
    t1 = [r["correct"] for r in b1 if r["arm"] == "bench"]
    tab = [[sum(c1), len(c1) - sum(c1)], [sum(t1), len(t1) - sum(t1)]]
    out["primary_block1"] = {"chosen": [sum(c1), len(c1)], "bench": [sum(t1), len(t1)],
                             "rate": {"chosen": float(np.mean(c1)), "bench": float(np.mean(t1))},
                             "diff_ci": boot_diff(c1, t1, g), "fisher_p": float(fisher_exact(tab)[1])}

    # secondary: both blocks, paired by character
    by = defaultdict(dict)
    for r in R:
        by[r["ch"]][r["arm"]] = r
    pairs = [(v["chosen"], v["bench"]) for v in by.values() if "chosen" in v and "bench" in v]
    co = sum(c["correct"] and not b["correct"] for c, b in pairs)
    bo = sum(b["correct"] and not c["correct"] for c, b in pairs)
    out["paired_both_blocks"] = {"n_chars": len(pairs), "chosen_read": sum(c["correct"] for c, _ in pairs),
                                 "bench_read": sum(b["correct"] for _, b in pairs), "chosen_only": co, "bench_only": bo,
                                 "mcnemar_p": mcnemar(co, bo)}
    for blk in (1, 2):
        rr = [r for r in R if r["block"] == blk]
        out[f"block{blk}"] = {arm: [sum(r["correct"] for r in rr if r["arm"] == arm), sum(r["arm"] == arm for r in rr)]
                              for arm in ("chosen", "bench")}
    out["strict_case"] = {arm: [sum(r["correct_strict"] for r in R if r["arm"] == arm), sum(r["arm"] == arm for r in R)]
                          for arm in ("chosen", "bench")}
    out["cant_tell"] = {arm: sum(r["cant"] for r in R if r["arm"] == arm) for arm in ("chosen", "bench")}
    out["by_subset"] = {sub: {arm: [sum(r["correct"] for r in R if r["arm"] == arm and r["sub"] == sub),
                                    sum(r["arm"] == arm and r["sub"] == sub for r in R)] for arm in ("chosen", "bench")}
                        for sub in ("letters_upper", "letters_lower", "digits")}
    holed = set(rep["summary"]["secondary"]["small_counter_chars"])
    out["small_counter_chars"] = {arm: [sum(r["correct"] for r in R if r["arm"] == arm and r["ch"] in holed),
                                        sum(r["arm"] == arm and r["ch"] in holed for r in R)] for arm in ("chosen", "bench")}
    font = {x["ch"]: x["chosen_font"] for x in rep["per_char"]}
    top = rep["summary"]["secondary"]["top_font"][0]
    out["top_font_chars"] = {arm: [sum(r["correct"] for r in R if r["arm"] == arm and font[r["ch"]] == top),
                                   sum(r["arm"] == arm and font[r["ch"]] == top for r in R)] for arm in ("chosen", "bench")}
    out["median_ms"] = {arm: float(np.median([r["ms"] for r in R if r["arm"] == arm])) for arm in ("chosen", "bench")}
    out["per_char"] = {ch: {arm: {"answer": v[arm]["answer"], "correct": v[arm]["correct"], "block": v[arm]["block"]}
                            for arm in ("chosen", "bench") if arm in v} for ch, v in sorted(by.items())}

    # what predicts a read, by arm (pooled over blocks)
    pred = {}
    for arm in ("chosen", "bench"):
        rr = [r for r in R if r["arm"] == arm]
        y = [r["correct"] for r in rr]
        feats = {"own-target IoU": [per[r["ch"]][arm] for r in rr],
                 "classifier p": [per[r["ch"]][f"cls_p_{arm}"] for r in rr],
                 "blob floor (target solidity)": [sec[r["ch"]][arm]["hull_iou"] for r in rr],
                 "IoU above the blob": [per[r["ch"]][arm] - sec[r["ch"]][arm]["hull_iou"] for r in rr]}
        pred[arm] = {k: auc(v, y) for k, v in feats.items()}
        iou = np.array(feats["own-target IoU"])
        pred[arm]["read_rate_iou_ge_075"] = [int(np.sum(np.array(y)[iou >= 0.75])), int(np.sum(iou >= 0.75))]
        pred[arm]["read_rate_iou_lt_075"] = [int(np.sum(np.array(y)[iou < 0.75])), int(np.sum(iou < 0.75))]
    out["predictors_auc"] = pred

    # A/B
    A = list(ab.values())
    dec = [a for a in A if a["pick"] in ("chosen", "bench")]
    nc = sum(a["pick"] == "chosen" for a in dec)
    out["ab_showings"] = {"chosen": nc, "bench": len(dec) - nc, "tie": len(A) - len(dec),
                          "chosen_share_decided": float(nc / len(dec)) if dec else None,
                          "binom_p": float(binomtest(nc, len(dec), 0.5).pvalue) if dec else None,
                          "left_share_decided": float(np.mean([a["side"] == "left" for a in dec])) if dec else None}
    pc = defaultdict(list)
    for a in A:
        pc[a["ch"]].append(a["pick"])
    agree = {ch: v[0] for ch, v in pc.items() if len(v) == 2 and v[0] == v[1]}
    cc = sum(v == "chosen" for v in agree.values())
    bb = sum(v == "bench" for v in agree.values())
    out["ab_chars"] = {"both_chosen": cc, "both_bench": bb, "both_tie": sum(v == "tie" for v in agree.values()),
                       "split": len(pc) - len(agree), "consistency": float(len(agree) / len(pc)),
                       "sign_p": float(binomtest(cc, cc + bb, 0.5).pvalue) if cc + bb else None,
                       "per_char": {ch: v for ch, v in sorted(pc.items())}}
    # does the A/B preference follow the read? (char-level, both showings agree)
    rd = {ch: (v.get("chosen", {}).get("correct"), v.get("bench", {}).get("correct")) for ch, v in by.items()}
    out["ab_vs_read"] = {pick: {"chosen_read_only": sum(1 for ch, p in agree.items() if p == pick and rd[ch] == (True, False)),
                                "bench_read_only": sum(1 for ch, p in agree.items() if p == pick and rd[ch] == (False, True))}
                         for pick in ("chosen", "bench")}
    return out


def main():
    rep = json.load(open(os.path.join(TD, "report.json"), encoding="utf-8"))
    g = np.random.default_rng(0)
    res = {}
    for f in sorted(glob.glob(os.path.join(TD, "human", "tests", "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        d = d.get("data", d)
        if len(d.get("reads", {})) + len(d.get("ab", {})) < 248:
            print("skip incomplete", os.path.basename(f), len(d.get("reads", {})), len(d.get("ab", {})))
            continue
        res[os.path.splitext(os.path.basename(f))[0]] = rater(d, rep, g)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    for uid, r in res.items():
        show = {k: v for k, v in r.items() if k not in ("per_char",)}
        show["ab_chars"] = {k: v for k, v in r["ab_chars"].items() if k != "per_char"}
        print(uid, json.dumps(show, indent=1))


if __name__ == "__main__":
    main()
