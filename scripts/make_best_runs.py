#!/usr/bin/env python3
"""Collect every target's winning run into one flat folder for visual triage.

`run_base_optimizer.py` writes one folder per target, which is right for archiving and
wrong for the actual job of deciding which shadows are usable — that job is scrolling
a few hundred images and drawing a line. This flattens the sweep into:

    optimized/base-optimizer/best-runs/
        iou0790__abstract__comma_mpeg7-04.png       target | shadow | overlay, IoU burnt in
        iou0771__letters_upper__L_dejavusans-bold.png
        ...
        shadows/abstract__comma_mpeg7-04_best.png   the raw 1-bit shadow, unannotated

Comparison sheets are named IoU-first and zero-padded, so a plain filename sort in any
file browser is a quality sort: scroll until the shapes stop reading, and everything
below that point is the reject pile. The IoU is drawn *into* the image rather than kept
alongside in a CSV, so a sheet stays interpretable once it has been dragged into a
slide or a message.

Raw shadows live in `shadows/` under stable names — no IoU in the filename — because
those get referenced by deploy configs and downstream scripts, which should not have to
care that a re-run nudged the score.

Runs on partial output, and is safe to re-run: existing files are overwritten in place.
"""

import argparse
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)


def load_mask(path: str, size: int | None = None) -> np.ndarray:
    """Benchmark and output PNGs are both 1-bit, dark = shape.

    NEAREST on resize for the same reason the runner uses it: a smooth filter turns a
    1-bit edge into a grey ramp that then gets re-thresholded, quietly eroding the thin
    strokes this benchmark deliberately includes.
    """
    img = Image.open(path).convert("L")
    if size:
        img = img.resize((size, size), Image.NEAREST)
    return (np.array(img) < 128).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default=_BENCH)
    p.add_argument("--results", default=None, help="default <bench>/optimized/base-optimizer")
    p.add_argument("--out", default=None, help="default <results>/best-runs")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--min-iou", type=float, default=None, help="only emit at or above this")
    a = p.parse_args()

    root = a.results or os.path.join(a.bench, "optimized", "base-optimizer")
    out = a.out or os.path.join(root, "best-runs")
    shadows = os.path.join(out, "shadows")
    os.makedirs(shadows, exist_ok=True)

    records = []
    for sub in sorted(os.listdir(root)):
        sd = os.path.join(root, sub)
        if not os.path.isdir(sd) or sub in ("gallery", "sheets", "best-runs"):
            continue
        for stem in sorted(os.listdir(sd)):
            rj = os.path.join(sd, stem, "results.json")
            if os.path.exists(rj):
                with open(rj) as f:
                    records.append((sub, stem, json.load(f)))
    if not records:
        print(f"[!] no results.json under {root}")
        return

    records.sort(key=lambda t: -t[2]["best_iou"])
    n_written = 0
    for sub, stem, r in records:
        iou = r["best_iou"]
        if a.min_iou is not None and iou < a.min_iou:
            continue
        src = os.path.join(root, sub, stem, f"{stem}_best.png")
        # A fitted sweep solves for a placed copy of the target — same proportions,
        # moved and scaled into the rig's reach — and scores against that copy. The
        # runner writes it as <stem>_shown.png, and it is the only honest thing to put
        # a shadow next to: pairing the shadow with the unplaced original would show a
        # mismatch the optimizer was never asked to close, under a score that measured
        # something else.
        shown = os.path.join(root, sub, stem, f"{stem}_shown.png")
        fitted = os.path.exists(shown)
        tgt_p = shown if fitted else os.path.join(a.bench, r["target"])
        if not (os.path.exists(src) and os.path.exists(tgt_p)):
            print(f"[skip] {sub}/{stem}: missing shadow or target")
            continue

        shutil.copyfile(src, os.path.join(shadows, f"{sub}__{stem}_best.png"))

        T, S = load_mask(tgt_p, a.size), load_mask(src, a.size)
        fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.6))
        ax[0].imshow(T, cmap="gray_r")
        f = r.get("fit") or {}
        ax[0].set_title(
            f"target — fitted ×{f['scale']:.2f}" if fitted else "target", fontsize=8
        )
        ax[1].imshow(S, cmap="gray_r")
        ax[1].set_title("robot silhouette", fontsize=8)
        ov = np.zeros((*T.shape, 3), np.float32)
        ov[..., 0], ov[..., 1] = T, S
        ax[2].imshow(1.0 - ov)
        ax[2].set_title("overlay — cyan missed, magenta spill", fontsize=7)
        for x in ax:
            x.set_xticks([])
            x.set_yticks([])
        # On a fitted sweep the headline IoU is against the placed target, so the
        # unplaced score rides along in the title: without it the two conditions look
        # directly comparable in a folder listing, and they are not.
        title = (
            f"{sub}/{stem}      IoU {iou:.3f}"
            f"      ({r['n_runs']} runs: mean {r['mean_iou']:.3f} ± {r['std_iou']:.3f},"
            f" min {r['min_iou']:.3f})"
        )
        # A second line rather than a longer one: the fitted note runs the single-line
        # title past the figure width, and a clipped title loses the sample size at one
        # end and the caveat at the other.
        if fitted and r.get("best_iou_vs_original") is not None:
            title += (f"\nplaced ×{f['scale']:.2f}, shift ({f['dx']:+.0f}, {f['dy']:+.0f}) px"
                      f"  —  IoU against the unplaced target {r['best_iou_vs_original']:.3f}")
        fig.suptitle(title, fontsize=9 if not fitted else 8)
        fig.tight_layout(rect=[0, 0, 1, 0.93 if not fitted else 0.88])
        # IoU-first, zero-padded: filename sort == quality sort.
        name = f"iou{int(round(iou * 1000)):04d}__{sub}__{stem}.png"
        fig.savefig(os.path.join(out, name), dpi=a.dpi)
        plt.close(fig)
        n_written += 1

    print(f"[best-runs] {n_written} comparisons → {out}/")
    print(f"[best-runs] {n_written} raw shadows → {shadows}/")
    if n_written:
        top = records[0]
        print(f"[best-runs] best: {top[0]}/{top[1]}  IoU {top[2]['best_iou']:.3f}")


if __name__ == "__main__":
    main()
