#!/usr/bin/env python3
"""CLIP recognition on the ablation and aimed-start probe outputs.

IoU says how much of the silhouette is filled; CLIP says whether the shadow
still reads as the glyph. Scores every variant's *_best.png for the ten probe
glyphs, ranked inside its subset vocabulary (26 uppercase letters, 10 digits)
with the paper's glyph prompts, and reports per variant: top-1, MRR, and the
softmax probability CLIP assigns the true class (continuous, so ten images can
show a difference that rank alone hides). Paired against the same reference
rows as collect_ablation.py.

    python scripts/clip_eval_ablation.py            # on a box with open_clip + GPU
    python scripts/clip_eval_ablation.py --out results/clip_ablation.json
"""
from __future__ import annotations
import argparse, glob, json, os, statistics as st, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tests"))
from semantic_metrics import clip_retrieval  # noqa: E402
from clip_eval import glyph_prompt  # noqa: E402

VOCAB = {"letters_upper": [chr(c) for c in range(65, 91)], "digits": [str(i) for i in range(10)]}

# variant -> reference, in table order (ablation-probe10, N=3; then aimed-start-probe10)
PAIRS = [
    ("ablation-probe10", "flat", None), ("ablation-probe10", "fwd_joint", "flat"),
    ("ablation-probe10", "plus_assign", "fwd_joint"), ("ablation-probe10", "plus_voronoi", "plus_assign"),
    ("ablation-probe10", "plus_backward", "plus_voronoi"), ("ablation-probe10", "plus_icp", "plus_backward"),
    ("ablation-probe10", "full", "plus_icp"),
    ("ablation-probe10", "full_no_assign", "full"), ("ablation-probe10", "full_no_voronoi", "full"),
    ("ablation-probe10", "full_no_backward", "full"), ("ablation-probe10", "full_no_icp", "full"),
    ("ablation-probe10", "full_no_fd", "full"),
    ("ablation-probe10", "full_long_no_icp", "full_long"),
    ("aimed-start-probe10", "no_aimed-tiny", "full-tiny"), ("aimed-start-probe10", "no_aimed-small", "full-small"),
]


def best_pngs(d: Path):
    out = {}
    for f in glob.glob(str(d / "*" / "*" / "*_best.png")):
        p = Path(f); out[(p.parts[-3], p.parts[-2])] = p   # (subset, stem)
    return out


def score(paths_by_key, model, pretrained):
    """-> {key: (rank, p_true)} over the subset vocabularies."""
    res = {}
    for subset, vocab in VOCAB.items():
        keys = sorted(k for k in paths_by_key if k[0] == subset)
        if not keys:
            continue
        ims = [Image.open(paths_by_key[k]).convert("RGB") for k in keys]
        true = [vocab.index(k[1].split("_")[0]) for k in keys]
        r = clip_retrieval(images=ims, class_names=vocab, true_idx=true, model_name=model,
                           pretrained=pretrained, prompt_tmpl=glyph_prompt, top_k=len(vocab))
        for k, rank, preds, t in zip(keys, r["rank"], r["top_predictions"], true):
            sims = np.array([p["similarity"] for p in sorted(preds, key=lambda p: p["index"])])
            p = np.exp(100 * sims); p /= p.sum()           # CLIP logit scale
            res[k] = (rank, float(p[t]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(ROOT))
    ap.add_argument("--model-name", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    bench = Path(a.bench)

    # targets = ceiling
    tdir = bench / "targets_ablation-probe10"
    tpaths = {(p.parts[-2], p.stem): p for p in tdir.glob("*/*.png")}
    scores = {"target": score(tpaths, a.model_name, a.pretrained)}
    variants = sorted({(s, v) for s, v, _ in PAIRS} | {(s, r) for s, _, r in PAIRS if r})
    for sweep, v in variants:
        d = bench / "optimized" / sweep / f"{v}-n3"
        if d.exists():
            scores[f"{sweep}/{v}"] = score(best_pngs(d), a.model_name, a.pretrained)
    keys = sorted(scores["target"])

    def summ(name):
        s = scores[name]
        return (st.mean(s[k][0] == 1 for k in keys), st.mean(1 / s[k][0] for k in keys),
                st.mean(s[k][1] for k in keys))
    print(f"{'variant':38s} {'top1':>5s} {'MRR':>6s} {'p_true':>7s}   {'d_top1':>6s} {'d_MRR':>6s} {'d_p':>7s}")
    t1, mrr, pt = summ("target"); print(f"{'target (ceiling)':38s} {t1:5.2f} {mrr:6.3f} {pt:7.3f}")
    rows = []
    for sweep, v, ref in PAIRS:
        name = f"{sweep}/{v}"
        if name not in scores:
            continue
        t1, mrr, pt = summ(name)
        line = f"{name:38s} {t1:5.2f} {mrr:6.3f} {pt:7.3f}"
        row = {"variant": name, "ref": ref, "top1": t1, "mrr": mrr, "p_true": pt}
        if ref and f"{sweep}/{ref}" in scores:
            r = scores[f"{sweep}/{ref}"]; s = scores[name]
            dt = st.mean((s[k][0] == 1) - (r[k][0] == 1) for k in keys)
            dm = st.mean(1 / s[k][0] - 1 / r[k][0] for k in keys)
            dp = [s[k][1] - r[k][1] for k in keys]
            line += f"   {dt:+6.2f} {dm:+6.3f} {st.mean(dp):+7.3f} ± {st.stdev(dp)/len(dp)**.5:.3f}  >0: {sum(x > 0 for x in dp)}"
            row.update(d_top1=dt, d_mrr=dm, d_p=st.mean(dp), d_p_sem=st.stdev(dp) / len(dp) ** .5)
        print(line); rows.append(row)
    if a.out:
        json.dump({"per_image": {n: {f"{k[0]}/{k[1]}": v for k, v in s.items()} for n, s in scores.items()},
                   "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
