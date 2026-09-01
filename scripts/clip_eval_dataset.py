#!/usr/bin/env python3
"""CLIP retrieval over the whole benchmark, not just the teleop captures.

`tests/clip_eval.py` scores 25 images: the letter and digit captures in
Teleops/masks. That is the human reference arm and it is 4% of the dataset. This
runs the same metric over every target in metadata.jsonl, in three conditions --
the target itself, and the shadow each sweep solved for it -- so the number that
matters, shadow score over target score, is computable.

The method is Simin's and is imported, not reimplemented: `clip_retrieval` does
the ranking, and the glyph folding rules come from `tests/clip_eval.py` so that
class design has one home.

    python scripts/clip_eval_dataset.py
    python scripts/clip_eval_dataset.py --render        # through render_shadow()
    python scripts/clip_eval_dataset.py --subsets digits letters_upper

Two things this has to get right that a single global run does not:

**Per-subset class lists.** `figures` has 2 classes and `letters_upper` has 26,
so chance top-1 runs from 0.500 to 0.038. Ranking every subset against one pooled
vocabulary would make the easy subsets look hard and hide the real differences,
so each subset is ranked inside its own class list and the report carries
`top1_over_chance` alongside `top1` -- that is the column to compare across
subsets.

**Targets as the ceiling.** CLIP is weak on 1-bit silhouettes, so an absolute
shadow accuracy is unattributable: a low number could be a bad shadow or a blind
judge. Scoring the target through the identical path gives the ceiling, and
`ratio = shadow / target` is what METRICS.md Part C asks to be reported.

`abstract` is the null control. Its classes are `device0`..`device9`, `bone`,
`comma` -- names that describe nothing about the shape -- so its ceiling should
sit near chance. If it does not, the class list is leaking information and every
other subset's number is suspect.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from semantic_metrics import (clip_retrieval, recognizability_ratio,  # noqa: E402
                              render_shadow)

# Simin's glyph class design: low-confidence lowercase folded to uppercase,
# visual equivalents collapsed (I/l/1, O/o/0, q/9). Imported rather than copied.
try:
    from clip_eval import ALL_LABELS as GLYPH_LABELS, canonicalize_label, glyph_prompt
except Exception:                                   # tests/ not on the branch
    GLYPH_LABELS, canonicalize_label, glyph_prompt = None, None, None

GLYPH_SUBSETS = ("digits", "letters_upper", "letters_lower")

# Wording follows build_metadata.py's PROMPTS, which is what the dataset says the
# task is. `abstract` deliberately has no semantic prompt -- naming is undefined
# for a shape with no name, and pretending otherwise is how a null control leaks.
PROMPT = {
    "digits": "a shadow of the digit {}",
    "letters_upper": "a shadow of the uppercase letter {}",
    "letters_lower": "a shadow of the lowercase letter {}",
    "animals": "a shadow of a {}",
    "objects": "a shadow of a {}",
    "vehicles": "a shadow of a {}",
    "figures": "a shadow of a {}",
    "hand_shadow": "a hand shadow of a {}",
    "abstract": "a shadow of a {}",
    "teleop": "a shadow of a {}",
}


def load_image(path: Path, render: bool):
    """-> PIL RGB. `render` puts the mask through render_shadow() first.

    Simin's runner feeds the 1-bit mask straight in, which is what this matches by
    default so the two are comparable. METRICS.md argues the rendered version is
    the fairer instrument; --render is that experiment rather than an assertion.
    """
    if not render:
        with Image.open(path) as im:
            return im.convert("RGB")
    with Image.open(path) as im:
        mask = np.asarray(im.convert("L")) < 128
    return render_shadow(mask).convert("RGB")


def collect(bench: Path, targets_dir: str, sweeps: dict[str, str], subsets) -> list[dict]:
    items = []
    with open(bench / "metadata.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if subsets and d["subset"] not in subsets:
                continue
            stem = Path(d["target"].replace("\\", "/")).stem
            tp = bench / targets_dir / d["subset"] / f"{stem}.png"
            if not tp.exists():
                continue
            rec = {"id": d["id"], "subset": d["subset"], "cls": d["class"],
                   "stem": stem, "paths": {"target": tp}}
            for slot, sweep in sweeps.items():
                sp = bench / "optimized" / sweep / d["subset"] / stem / f"{stem}_best.png"
                if sp.exists():
                    rec["paths"][slot] = sp
            items.append(rec)
    return items


def score_group(group: list[dict], condition: str, class_names: list[str],
                prompt_tmpl, render: bool, model: str, pretrained: str,
                batch_size: int, top_k: int):
    """Run one (subset, condition) through clip_retrieval. -> (result, used items)."""
    used = [it for it in group if condition in it["paths"]]
    if not used:
        return None, []
    idx = {c: i for i, c in enumerate(class_names)}
    keep, images, true_idx = [], [], []
    for it in used:
        c = it["scoring_cls"]
        if c not in idx:
            continue
        keep.append(it)
        images.append(load_image(it["paths"][condition], render))
        true_idx.append(idx[c])
    if not keep:
        return None, []
    res = clip_retrieval(images=images, class_names=class_names, true_idx=true_idx,
                         model_name=model, pretrained=pretrained,
                         prompt_tmpl=prompt_tmpl, batch_size=batch_size, top_k=top_k)
    return res, keep


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=str(ROOT))
    ap.add_argument("--targets-dir", default="targets_grounded")
    ap.add_argument("--big", default="big-budget-grounded")
    ap.add_argument("--small", default="small-budget-grounded")
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--glyph-mode", choices=("subset", "folded"), default="subset",
                    help="`subset` ranks each glyph subset inside its own alphabet. "
                         "`folded` uses the pooled 49-class set from tests/clip_eval.py, "
                         "which is Simin's design and folds visual equivalents.")
    ap.add_argument("--render", action="store_true",
                    help="score render_shadow() output instead of the raw 1-bit mask")
    ap.add_argument("--model-name", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    bench = Path(a.bench)
    out_dir = Path(a.out) if a.out else bench / "results" / (
        "clip_dataset_render" if a.render else "clip_dataset")
    sweeps = {"big": a.big, "small": a.small}

    items = collect(bench, a.targets_dir, sweeps, set(a.subsets) if a.subsets else None)
    if not items:
        raise SystemExit("no targets found -- check --targets-dir and metadata.jsonl")

    folded = a.glyph_mode == "folded" and GLYPH_LABELS is not None
    for it in items:
        it["scoring_cls"] = it["cls"]
        if folded and it["subset"] in GLYPH_SUBSETS:
            it["scoring_cls"] = canonicalize_label(it["cls"])

    by_subset = defaultdict(list)
    for it in items:
        if it["scoring_cls"] is not None:
            by_subset[it["subset"]].append(it)

    conditions = ["target"] + [s for s in sweeps if any(s in i["paths"] for i in items)]
    print(f"items={len(items)}  subsets={len(by_subset)}  conditions={conditions}  "
          f"render={a.render}  glyph_mode={a.glyph_mode}")

    rows, summary = [], []
    for subset in sorted(by_subset):
        group = by_subset[subset]
        if folded and subset in GLYPH_SUBSETS:
            class_names, tmpl = GLYPH_LABELS, glyph_prompt
        else:
            class_names = sorted({i["scoring_cls"] for i in group})
            tmpl = PROMPT.get(subset, "a shadow of a {}")
        per_cond = {}
        for cond in conditions:
            res, used = score_group(group, cond, class_names, tmpl, a.render,
                                    a.model_name, a.pretrained, a.batch_size, a.top_k)
            if res is None:
                continue
            per_cond[cond] = res
            for it, rank, preds in zip(used, res["rank"], res["top_predictions"]):
                rows.append({
                    "id": it["id"], "subset": subset, "condition": cond,
                    "true_class": it["scoring_cls"], "rank": rank,
                    "top1": int(rank == 1),
                    "clip_top1_label": preds[0]["label"],
                    "clip_top1_similarity": round(preds[0]["similarity"], 4),
                    # what CLIP actually guessed, for the card detail view. A
                    # rank of 8 says the shadow failed; these say what it failed
                    # *as*, which is the part a reader can act on.
                    "clip_top3": "|".join(
                        f"{q['label']}:{q['similarity']:.4f}" for q in preds[:3]),
                    "n_classes": len(class_names),
                    "chance_top1": round(1 / len(class_names), 4),
                })
            print(f"  {subset:<14} {cond:<7} n={len(used):>4}  "
                  f"top1={res['top1']:.3f}  mrr={res['mrr']:.3f}  "
                  f"chance={res['chance_top1']:.3f}")
        base = per_cond.get("target")
        for cond, res in per_cond.items():
            ch = res["chance_top1"]
            summary.append({
                "subset": subset, "condition": cond, "n": len(res["rank"]),
                "n_classes": len(class_names), "chance_top1": round(ch, 4),
                "top1": round(res["top1"], 4), "top5": round(res["top5"], 4),
                "mrr": round(res["mrr"], 4),
                "top1_over_chance": round(res["top1"] / ch, 3) if ch else None,
                # Through recognizability_ratio() rather than reimplemented: it
                # is the repo's definition of this number and predates all of
                # this. `top1` is its default and the strict reading -- how often
                # the shadow wins outright. `mrr` is the graded one, and it is
                # the aggregate counterpart of the per-item clip_rr ratio the
                # atlas card shows, since mean(1/rank) is MRR. Reporting both is
                # what stops "the ratio" meaning two things: hand_shadow is 0.000
                # by top1 and 0.576 by rank, and each is true.
                "ratio_vs_target": (round(recognizability_ratio(res, base, "top1"), 4)
                                    if base and base["top1"] else None),
                "mrr_ratio_vs_target": (round(recognizability_ratio(res, base, "mrr"), 4)
                                        if base and base["mrr"] else None),
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "clip_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "clip_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(out_dir / "clip_summary.json", "w", encoding="utf-8") as f:
        json.dump({"model": a.model_name, "pretrained": a.pretrained,
                   "render": a.render, "glyph_mode": a.glyph_mode,
                   "targets_dir": a.targets_dir, "sweeps": sweeps,
                   "summary": summary}, f, indent=1)
    print(f"\n{len(rows)} scored images -> {out_dir}")


if __name__ == "__main__":
    main()
