#!/usr/bin/env python3
"""fig_ablation: paired delta-IoU per component, ten probe glyphs, N=3.

Left: build-up (each row vs the row before). Right: leave-one-out (each row vs
the full solver). Bars are the mean paired delta; dots are one glyph each; the
grey band is the within-target seed spread (+-0.02), below which a change is noise.
Data: optimized/ablation-probe10 via collect_ablation.load().
"""
import os, statistics as st, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_ablation import SPREAD, load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "figures")
BLUE, INK, MUTED, GRID = "#2a78d6", "#1a1a19", "#5c5b55", "#dcdbd4"
plt.rcParams.update({
    "font.size": 6.8, "axes.labelsize": 6.8, "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.6, "axes.titlesize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})

data = load(os.path.join(ROOT, "optimized", "ablation-probe10"), 3)
glyphs = sorted(set.intersection(*(set(c) for c in data.values())))
COMPONENTS = [  # label, build-up (variant, ref), leave-one-out (variant, ref)
    ("per-arm sweep + joint stage", ("fwd_joint", "flat"),             None),
    ("image-space assignment",      ("plus_assign", "fwd_joint"),      ("full_no_assign", "full")),
    ("zone-aimed start",            ("plus_voronoi", "plus_assign"),   ("full_no_voronoi", "full")),
    ("backward sweep",              ("plus_backward", "plus_voronoi"), ("full_no_backward", "full")),
    ("ICP restarts",                ("plus_icp", "plus_backward"),     ("full_no_icp", "full")),
    ("FD polish",                   ("full", "plus_icp"),              ("full_no_fd", "full")),
]

def deltas(v, ref):
    return [data[v][g][0] - data[ref][g][0] for g in glyphs]

ORANGE = "#eb6834"
sem = lambda v: st.stdev(v) / len(v) ** 0.5

fig, ax = plt.subplots(figsize=(3.45, 1.4))
ys = list(range(len(COMPONENTS)))[::-1]
h = 0.36
ax.axvspan(-SPREAD, SPREAD, color=GRID, alpha=0.55, lw=0, zorder=0)
ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
for y, (lab, bu, lo) in zip(ys, COMPONENTS):
    d = deltas(*bu)
    ax.barh(y + h / 2, st.mean(d), height=h, color=BLUE, lw=0, zorder=2,
            xerr=sem(d), error_kw=dict(elinewidth=0.7, capsize=1.5, ecolor=INK),
            label="gain when added" if y == ys[0] else None)
    if lo is None:
        continue
    d = [-x for x in deltas(*lo)]  # loss when removed, sign flipped: right = helps
    ax.barh(y - h / 2, st.mean(d), height=h, color=ORANGE, lw=0, zorder=2,
            xerr=sem(d), error_kw=dict(elinewidth=0.7, capsize=1.5, ecolor=INK),
            label="loss when removed" if y == ys[1] else None)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in COMPONENTS])
ax.set_ylim(-0.6, len(COMPONENTS) - 0.4)
ax.set_xlim(-0.025, 0.03)
ax.set_xticks([-0.02, 0, 0.02]); ax.set_xticklabels(["$-$0.02", "0", "+0.02"])
ax.set_xlabel("$\\Delta$ IoU")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.subplots_adjust(left=0.33, right=0.99, top=0.98, bottom=0.2)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_ablation.{ext}"), bbox_inches="tight", pad_inches=0.02)
for lab, bu, lo in COMPONENTS:
    for tag, pair in (("+", bu), ("-", lo)):
        if pair is None: continue
        d = deltas(*pair)
        print(f"{tag} {lab:28s} mean {st.mean(d):+.3f}  >+{SPREAD}: {sum(x > SPREAD for x in d)}  <-{SPREAD}: {sum(x < -SPREAD for x in d)}  >0: {sum(x > 0 for x in d)}")
