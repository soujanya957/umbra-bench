#!/usr/bin/env python3
"""CLIP recognizability for the grounded letters+digits sweep.

Does the cast shadow *read* as the letter, not just overlap it. Follows
METRICS.md Part C: retrieval rank over the benchmark's own class set (top-1 and
MRR, with chance printed beside them), raw masks rather than renders, and the
authored target scored through the identical path as the ceiling -- CLIP is
half-blind to 1-bit silhouettes, so an absolute score cannot separate "the
shadow is poor" from "the judge cannot see either". Only the ratio can.

The class set, alias map and prompt wording are imported from tests/clip_eval.py
rather than restated: they define the benchmark and I/l/1 collapsing to one class
is a deliberate call recorded there.
"""
import argparse, csv, os, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(os.path.expanduser("~/dev/umbra-bench-grounded"))
sys.path.insert(0, str(ROOT / "tests"))
from clip_eval import ALL_LABELS, canonicalize_label, glyph_prompt  # noqa: E402

import open_clip  # noqa: E402


def stem_label(stem: str) -> str | None:
    """'L_dejavusans-bold' -> 'L';  '0_dejavusansmono-bold' -> '0'."""
    head = stem.split("_", 1)[0]
    return head if len(head) == 1 else None


def collect(conditions):
    """(condition, id, subset, label, path) for every scorable image."""
    items = []
    base = ROOT / "optimized" / "letters-digits-grounded"
    for cond in conditions:
        for subset_dir in sorted((base / cond).iterdir()):
            if not subset_dir.is_dir():
                continue
            for tgt in sorted(subset_dir.iterdir()):
                png = tgt / f"{tgt.name}_best.png"
                if not png.exists():
                    continue
                raw = stem_label(tgt.name)
                if raw is None or canonicalize_label(raw) is None:
                    continue
                items.append((cond, f"{subset_dir.name}_{tgt.name}",
                              subset_dir.name, raw, png))
    # the authored targets, through the identical path, as the ceiling
    for subset in ("digits", "letters_upper", "letters_lower"):
        for png in sorted((ROOT / "targets_grounded" / subset).glob("*.png")):
            raw = stem_label(png.stem)
            if raw is None or canonicalize_label(raw) is None:
                continue
            items.append(("target", f"{subset}_{png.stem}", subset, raw, png))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+",
                    default=["small-n3", "big-n3", "small-n5", "big-n5"])
    ap.add_argument("--model-name", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "optimized" /
                                         "letters-digits-grounded" / "clip_scores.csv"))
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        a.model_name, pretrained=a.pretrained, device=dev)
    model.eval()
    tok = open_clip.get_tokenizer(a.model_name)

    prompts = [glyph_prompt(l) for l in ALL_LABELS]
    with torch.no_grad():
        tfeat = model.encode_text(tok(prompts).to(dev))
        tfeat /= tfeat.norm(dim=-1, keepdim=True)

    items = collect(a.conditions)
    print(f"scoring {len(items)} images over {len(ALL_LABELS)} classes "
          f"(chance top-1 = {1/len(ALL_LABELS):.4f}) on {dev}")

    rows = []
    for i in range(0, len(items), a.batch_size):
        batch = items[i:i + a.batch_size]
        px = torch.stack([preprocess(Image.open(p).convert("RGB"))
                          for *_, p in batch]).to(dev)
        with torch.no_grad():
            f = model.encode_image(px)
            f /= f.norm(dim=-1, keepdim=True)
            sims = (f @ tfeat.T).cpu().numpy()
        for (cond, tid, subset, raw, _), sim in zip(batch, sims):
            gold = ALL_LABELS.index(canonicalize_label(raw))
            order = np.argsort(-sim)
            rank = int(np.where(order == gold)[0][0]) + 1
            rows.append({"condition": cond, "id": tid, "subset": subset,
                         "label": raw, "gold_class": canonicalize_label(raw),
                         "rank": rank, "top1": int(rank == 1),
                         "rr": 1.0 / rank,
                         "pred": ALL_LABELS[int(order[0])]})
        print(f"  {min(i+a.batch_size, len(items))}/{len(items)}", flush=True)

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out}")

    chance = 1.0 / len(ALL_LABELS)
    ceil = [r for r in rows if r["condition"] == "target"]
    c_top1 = np.mean([r["top1"] for r in ceil])
    c_mrr = np.mean([r["rr"] for r in ceil])
    print(f"\n{'condition':<12}{'top-1':>9}{'MRR':>9}{'ratio_top1':>12}{'ratio_MRR':>11}")
    print(f"{'target':<12}{c_top1:>9.4f}{c_mrr:>9.4f}{'-- ceiling':>12}{'':>11}")
    for cond in a.conditions:
        sub = [r for r in rows if r["condition"] == cond]
        t1, mr = np.mean([r["top1"] for r in sub]), np.mean([r["rr"] for r in sub])
        print(f"{cond:<12}{t1:>9.4f}{mr:>9.4f}"
              f"{t1/c_top1 if c_top1 else float('nan'):>12.4f}"
              f"{mr/c_mrr if c_mrr else float('nan'):>11.4f}")
    print(f"\nchance top-1 = {chance:.4f} over {len(ALL_LABELS)} classes")


if __name__ == "__main__":
    main()
