#!/usr/bin/env python3
"""Figures 3-5: legibility, and what the results actually look like.

Fig 3  CLIP recognizability by condition and subset, against the authored
       target as ceiling and retrieval chance as floor.
Fig 4  the best results, target above cast shadow.
Fig 5  one glyph across fleet sizes -- the picture behind Fig 2's reversal.
"""
import csv, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.expanduser("~/dev/umbra-bench-grounded")
FIGS = os.path.join(ROOT, "figures")
BASE = os.path.join(ROOT, "optimized", "letters-digits-grounded")
NS = os.path.join(ROOT, "optimized", "nsweep-letters-digits")
os.makedirs(FIGS, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#dcdbd4"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})
CONDS = ["small-n3", "big-n3", "small-n5", "big-n5"]
NICE = {"small-n3": "small\nn=3", "big-n3": "big\nn=3",
        "small-n5": "small\nn=5", "big-n5": "big\nn=5"}


def load(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def fig_clip():
    clip = load(os.path.join(BASE, "clip_scores.csv"))
    n_classes = 49
    chance = 1.0 / n_classes
    ceil = st.mean(int(r["top1"]) for r in clip if r["condition"] == "target")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # A: by condition
    vals = [st.mean(int(r["top1"]) for r in clip if r["condition"] == c) for c in CONDS]
    axA.bar(range(4), vals, width=0.6, color=[BLUE, ORANGE, BLUE, ORANGE], zorder=3)
    for i, v in enumerate(vals):
        axA.text(i, v + 0.018, f"{v:.3f}", ha="center", fontsize=7, color=INK, zorder=5)
    axA.axhline(ceil, color=INK, lw=1.2, ls="--", zorder=4)
    axA.text(3.45, ceil, f" authored target  {ceil:.3f}", va="bottom", ha="right",
             fontsize=7.5, color=INK)
    axA.axhline(chance, color=MUTED, lw=1, ls=":", zorder=4)
    axA.text(3.45, chance, f" chance  {chance:.3f}", va="bottom", ha="right",
             fontsize=7.5, color=MUTED)
    style(axA)
    axA.set_xticks(range(4))
    axA.set_xticklabels([NICE[c] for c in CONDS])
    axA.set_ylabel("CLIP top-1 retrieval")
    axA.set_ylim(0, 0.88)
    axA.set_title("Legibility vs the authored ceiling")

    # B: by subset, ceiling vs best condition
    subs = ["letters_upper", "letters_lower", "digits"]
    lab = {"letters_upper": "uppercase", "letters_lower": "lowercase", "digits": "digits"}
    tgt = [st.mean(int(r["top1"]) for r in clip
                   if r["condition"] == "target" and r["subset"] == s) for s in subs]
    sha = [st.mean(int(r["top1"]) for r in clip
                   if r["condition"] == "big-n5" and r["subset"] == s) for s in subs]
    x = range(3)
    axB.bar([i - 0.19 for i in x], tgt, width=0.36, color=MUTED, zorder=3,
            label="authored target")
    axB.bar([i + 0.19 for i in x], sha, width=0.36, color=ORANGE, zorder=3,
            label="cast shadow (big, n=5)")
    for i, (t, s_) in enumerate(zip(tgt, sha)):
        axB.text(i - 0.19, t + 0.018, f"{t:.2f}", ha="center", fontsize=7, color=INK)
        axB.text(i + 0.19, s_ + 0.018, f"{s_:.2f}", ha="center", fontsize=7, color=INK)
    axB.axhline(chance, color=MUTED, lw=1, ls=":", zorder=4)
    style(axB)
    axB.set_xticks(list(x))
    axB.set_xticklabels([lab[s] for s in subs])
    axB.set_ylim(0, 0.98)
    axB.set_title("Digits barely read at all")
    axB.legend(frameon=False, loc="upper right")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig3_clip_recognizability.{ext}"))
    plt.close(fig)
    print("wrote fig3_clip_recognizability")


def _img(path):
    return Image.open(path).convert("L")


def fig_gallery(n=8, cond="big-n5"):
    rows = [r for r in load(os.path.join(BASE, "letters_digits_summary.csv"))
            if r["condition"] == cond]
    clip = {r["id"]: r for r in load(os.path.join(BASE, "clip_scores.csv"))
            if r["condition"] == cond}
    rows.sort(key=lambda r: -float(r["best_iou"]))
    # One entry per distinct glyph class: the raw top-8 is three L's and two r's,
    # and v/V are the same class under the alias map, which shows off the fonts
    # rather than the range of shapes.
    seen, top = set(), []
    for r in rows:
        stem = r["id"].replace(r["subset"] + "_", "", 1)
        g = stem.split("_")[0]
        key = {"v": "V", "l": "1", "I": "1", "o": "0", "O": "0"}.get(g, g)
        if key in seen:
            continue
        seen.add(key)
        top.append(r)
        if len(top) == n:
            break

    fig, axes = plt.subplots(2, n, figsize=(7.2, 2.35))
    for j, r in enumerate(top):
        stem = r["id"].replace(r["subset"] + "_", "", 1)
        tpath = os.path.join(ROOT, "targets_grounded", r["subset"], stem + ".png")
        spath = os.path.join(BASE, cond, r["subset"], stem, stem + "_best.png")
        for i, p in enumerate((tpath, spath)):
            ax = axes[i][j]
            ax.imshow(_img(p), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID)
        c = clip.get(r["id"])
        mark = "reads" if c and c["top1"] == "1" else "misread"
        axes[0][j].set_title(stem.split("_")[0], fontsize=9, pad=3)
        axes[1][j].set_xlabel(f"{float(r['best_iou']):.3f}\n{mark}", fontsize=6.5,
                              color=INK if mark == "reads" else MUTED)
    axes[0][0].set_ylabel("target", fontsize=8)
    axes[1][0].set_ylabel("shadow", fontsize=8)
    fig.suptitle(f"Best {n} results by IoU (big budget, n=5), with CLIP verdict",
                 y=1.03, fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig4_gallery_best.{ext}"))
    plt.close(fig)
    print("wrote fig4_gallery_best")


def fig_progression(subset="letters_upper", stem="J_dejavusans-bold", bud="small"):
    ns = [1, 2, 3, 5, 7]
    rows = {(r["n_robots"]): r for r in load(os.path.join(NS, "nsweep_summary.csv"))
            if r["budget"] == bud and r["id"] == f"{subset}_{stem}"}
    ious = {n: float(rows[str(n)]["best_iou"]) for n in ns if str(n) in rows}
    best_n = max(ious, key=ious.get)
    fig, axes = plt.subplots(1, len(ns) + 1, figsize=(7.2, 1.65))
    tpath = os.path.join(ROOT, "targets_grounded", subset, stem + ".png")
    axes[0].imshow(_img(tpath), cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("target", fontsize=8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
    for k, n in enumerate(ns):
        p = os.path.join(NS, f"{bud}-n{n}", subset, stem, stem + "_best.png")
        ax = axes[k + 1]
        if os.path.exists(p):
            ax.imshow(_img(p), cmap="gray", vmin=0, vmax=255)
        r = rows.get(str(n))
        iou = float(r["best_iou"]) if r else float("nan")
        peak = (n == best_n)
        ax.set_title(f"n={n}", fontsize=8, color=INK)
        ax.set_xlabel(f"{iou:.3f}", fontsize=7,
                      color=ORANGE if peak else INK,
                      fontweight="bold" if peak else "normal")
    drop = ious[best_n] - ious[max(ns)]
    fig.suptitle(f"'{stem.split('_')[0]}' across fleet sizes ({bud} budget): "
                 f"peaks at n={best_n}, then loses {drop:.3f} IoU by n={max(ns)}",
                 y=1.10, fontsize=9.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig5_fleet_progression.{ext}"))
    plt.close(fig)
    print("wrote fig5_fleet_progression")


if __name__ == "__main__":
    fig_clip()
    fig_gallery()
    fig_progression()
