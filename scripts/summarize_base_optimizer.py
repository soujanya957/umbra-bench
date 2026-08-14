#!/usr/bin/env python3
"""Aggregate a base-optimizer sweep into a CSV plus a flat gallery for visual triage.

`run_base_optimizer.py` writes one folder per target, which is the right archival
layout and the wrong layout for deciding which results are usable. This produces the
two things you actually filter with:

  summary.csv   one row per target — name, IoU (best / mean / std / min), and the
                joint pose of the best run, both as flat per-joint degree columns
                (sortable in a spreadsheet) and as a JSON blob (pasteable into a
                deploy file).

  gallery/      one flat triptych per target — target | best shadow | overlay — named
                <subset>__<stem>.png so the whole benchmark sorts into one scrollable
                folder. Sorted-by-IoU contact sheets go alongside, which is usually
                the faster way to spot the cutoff where results stop being usable.

Runs on partial output: targets still in flight are simply absent, so this can be
called while the sweep is going.
"""

import argparse
import csv
import json
import os

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)

JOINT_SUFFIX = ["rot", "pitch", "elbow", "wristpitch", "wristroll", "grip"]


def load_mask(path: str, size: int | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size:
        img = img.resize((size, size), Image.NEAREST)
    return (np.array(img) < 128).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default=_BENCH)
    p.add_argument("--results", default=None, help="default <bench>/optimized/base-optimizer")
    p.add_argument("--out", default=None, help="default = results dir")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--no-gallery", action="store_true")
    p.add_argument("--sheet-cols", type=int, default=6)
    a = p.parse_args()

    root = a.results or os.path.join(a.bench, "optimized", "base-optimizer")
    out = a.out or root
    os.makedirs(out, exist_ok=True)

    records = []
    for sub in sorted(os.listdir(root)):
        sd = os.path.join(root, sub)
        if not os.path.isdir(sd) or sub in ("gallery", "sheets"):
            continue
        for stem in sorted(os.listdir(sd)):
            rj = os.path.join(sd, stem, "results.json")
            if os.path.exists(rj):
                with open(rj) as f:
                    records.append(json.load(f))
    if not records:
        print(f"[!] no results.json under {root}")
        return
    print(f"[summary] {len(records)} targets")

    n_arm_dof = len(JOINT_SUFFIX)
    max_rob = max(len(r["runs"][0]["q_rad"]) // n_arm_dof for r in records)
    jcols = [
        f"{chr(ord('a') + i)}_{JOINT_SUFFIX[j]}_deg"
        for i in range(max_rob)
        for j in range(n_arm_dof)
    ]

    rows = []
    for r in records:
        best = r["runs"][r["best_run"]]
        q = np.array(best["q_rad"], dtype=float)
        deg = np.degrees(q)
        stem = os.path.splitext(os.path.basename(r["target"]))[0]
        row = {
            "shadow_name": f"{r['subset']}/{stem}",
            "subset": r["subset"],
            "target": r["target"],
            "best_iou": r["best_iou"],
            "mean_iou": r["mean_iou"],
            "std_iou": r["std_iou"],
            "min_iou": r["min_iou"],
            "n_runs": r["n_runs"],
            "best_run": r["best_run"],
            "n_robots": r["rig"]["n_robots"],
            "arm_gap_m": r["rig"]["arm_gap_m"],
            "seconds": r.get("seconds"),
            "best_shadow_png": os.path.join(r["subset"], stem, stem + "_best.png"),
            "q_rad_json": json.dumps([round(float(v), 6) for v in q]),
        }
        for i, c in enumerate(jcols):
            row[c] = round(float(deg[i]), 2) if i < len(deg) else ""
        rows.append(row)

    rows.sort(key=lambda r: (-r["best_iou"],))
    csv_path = os.path.join(out, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[summary] → {csv_path}")

    ious = np.array([r["best_iou"] for r in rows])
    print(f"\n  best_iou  mean {ious.mean():.3f}  median {np.median(ious):.3f}  "
          f"min {ious.min():.3f}  max {ious.max():.3f}")
    print(f"  spread within a target (std over 10 runs): "
          f"mean {np.mean([r['std_iou'] for r in rows]):.4f}")
    print("\n  per subset:")
    for s in sorted({r["subset"] for r in rows}):
        v = np.array([r["best_iou"] for r in rows if r["subset"] == s])
        print(f"    {s:<16} n={len(v):>4}  best_iou mean {v.mean():.3f}  "
              f"min {v.min():.3f}  max {v.max():.3f}")

    if a.no_gallery:
        return

    gdir = os.path.join(out, "gallery")
    os.makedirs(gdir, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    for r in rows:
        tgt_p = os.path.join(a.bench, r["target"])
        sh_p = os.path.join(root, r["best_shadow_png"])
        if not (os.path.exists(tgt_p) and os.path.exists(sh_p)):
            continue
        T = load_mask(tgt_p, a.size)
        S = load_mask(sh_p, a.size)
        fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.5))
        ax[0].imshow(T, cmap="gray_r")
        ax[0].set_title("target", fontsize=8)
        ax[1].imshow(S, cmap="gray_r")
        ax[1].set_title("best shadow", fontsize=8)
        ov = np.zeros((*T.shape, 3), np.float32)
        ov[..., 0], ov[..., 1] = T, S
        ax[2].imshow(1.0 - ov)
        ax[2].set_title("cyan=missed magenta=spill", fontsize=7)
        for x in ax:
            x.set_xticks([])
            x.set_yticks([])
        fig.suptitle(
            f"{r['shadow_name']}   IoU best {r['best_iou']:.3f}  "
            f"mean {r['mean_iou']:.3f}±{r['std_iou']:.3f}",
            fontsize=9,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        gp = os.path.join(gdir, f"{r['subset']}__{os.path.basename(r['shadow_name'])}.png")
        fig.savefig(gp, dpi=100)
        plt.close(fig)
        made.append((r, gp))
    print(f"\n[gallery] {len(made)} triptychs → {gdir}/")

    # Contact sheets, sorted by IoU: the fast way to find where quality falls off.
    sdir = os.path.join(out, "sheets")
    os.makedirs(sdir, exist_ok=True)
    for s in sorted({r["subset"] for r, _ in made}):
        items = [(r, g) for r, g in made if r["subset"] == s]
        cols = a.sheet_cols
        n = len(items)
        rws = (n + cols - 1) // cols
        fig, axes = plt.subplots(rws, cols, figsize=(2.1 * cols, 2.45 * rws), squeeze=False)
        for k in range(rws * cols):
            ax = axes[k // cols][k % cols]
            ax.set_xticks([])
            ax.set_yticks([])
            if k >= n:
                ax.axis("off")
                continue
            r, _ = items[k]
            T = load_mask(os.path.join(a.bench, r["target"]), a.size)
            S = load_mask(os.path.join(root, r["best_shadow_png"]), a.size)
            ov = np.zeros((*T.shape, 3), np.float32)
            ov[..., 0], ov[..., 1] = T, S
            ax.imshow(1.0 - ov)
            ax.set_title(
                f"{os.path.basename(r['shadow_name'])[:18]}\n{r['best_iou']:.3f}",
                fontsize=6,
            )
        fig.suptitle(f"{s} — best shadow vs target, sorted by IoU", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.975])
        sp = os.path.join(sdir, f"sheet_{s}.png")
        fig.savefig(sp, dpi=95)
        plt.close(fig)
        print(f"[sheet] {s}: {len(items)} → {sp}")


if __name__ == "__main__":
    main()
