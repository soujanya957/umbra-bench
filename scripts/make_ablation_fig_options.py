#!/usr/bin/env python3
"""Three candidate forms for the ablation figure, one file each:

  fig_ablation_A  waterfall: IoU built up component by component (cumulative)
  fig_ablation_B  leave-one-out on the IoU axis: UMBRA without each component,
                  against the UMBRA and monolithic reference lines
  fig_ablation_C  two panels of paired deltas, added | removed, as lollipops

Data: optimized/ablation-probe10 via collect_ablation.load(), N=3, ten glyphs.
"""
import os, statistics as st, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_ablation import SPREAD, load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "figures")
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#5c5b55", "#dcdbd4"
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})

data = load(os.path.join(ROOT, "optimized", "ablation-probe10"), 3)
G = sorted(set.intersection(*(set(c) for c in data.values() if c)))
mean = lambda v: st.mean(data[v][g][0] for g in G)
def paired(v, ref):
    d = [data[v][g][0] - data[ref][g][0] for g in G]
    return st.mean(d), st.stdev(d) / len(d) ** 0.5

# pipeline number, label, build-up variant, its predecessor, leave-one-out variant
STEPS = [
    ("3, 6", "arm-by-arm sweep + joint stage", "fwd_joint",     "flat",          None),
    ("1",    "zone assignment",                "plus_assign",   "fwd_joint",     "full_no_assign"),
    ("2",    "aimed start",                    "plus_voronoi",  "plus_assign",   "full_no_voronoi"),
    ("4",    "backward sweep",                 "plus_backward", "plus_voronoi",  "full_no_backward"),
    ("5",    "ICP-guided restart",             "plus_icp",      "plus_backward", "full_no_icp"),
    ("7",    "FD polish",                      "full",          "plus_icp",      "full_no_fd"),
]
LAB = [f"{n}  {l}" for n, l, *_ in STEPS]
FLAT, FULL = mean("flat"), mean("full")

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# ── A: waterfall ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.45, 1.55))
rows = ["monolithic CMA-ES"] + LAB + ["UMBRA"]
ys = list(range(len(rows)))[::-1]
ax.barh(ys[0], FLAT, color=MUTED, height=0.55, lw=0)
cur = FLAT
for y, (_, _, v, ref, _) in zip(ys[1:-1], STEPS):
    d, _ = paired(v, ref)
    ax.barh(y, d, left=cur, height=0.55, lw=0, color=BLUE if d >= 0 else ORANGE)
    if abs(d) >= 0.0005:
        ax.text(max(cur, cur + d) + 0.0012, y, f"{d:+.3f}", va="center", ha="left", fontsize=6)
    else:
        ax.text(cur + 0.0012, y, "0", va="center", ha="left", fontsize=6)
    ax.plot([cur + d, cur + d], [y - 0.28, y - 0.72], color=MUTED, lw=0.5)
    cur += d
ax.barh(ys[-1], FULL, color=INK, height=0.55, lw=0)
ax.text(FLAT + 0.0012, ys[0], f"{FLAT:.3f}", va="center", fontsize=6, ha="left")
ax.text(FULL + 0.0012, ys[-1], f"{FULL:.3f}", va="center", fontsize=6, ha="left")
ax.set_yticks(ys); ax.set_yticklabels(rows)
ax.set_xlim(0.685, 0.73); ax.set_xticks([0.69, 0.70, 0.71, 0.72, 0.73])
ax.set_xlabel("IoU, ten probe glyphs, $N{=}3$")
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
save(fig, "fig_ablation_A")

# ── B: leave-one-out on the IoU axis ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.45, 1.35))
loo = [(lab, s) for lab, s in zip(LAB, STEPS) if s[4]]
ys = list(range(len(loo)))[::-1]
ax.axvspan(FULL - SPREAD, FULL + SPREAD, color=GRID, alpha=0.55, lw=0, zorder=0)
ax.plot([FULL, FULL], [-0.6, len(loo) - 0.45], color=INK, lw=0.8, zorder=1)
ax.plot([FLAT, FLAT], [-0.6, len(loo) - 0.45], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
for y, (lab, s) in zip(ys, loo):
    v = mean(s[4]); d, sem = paired(s[4], "full")
    ax.plot([FULL, v], [y, y], color=ORANGE if d < -0.0005 else BLUE, lw=1.2, zorder=2)
    ax.errorbar(v, y, xerr=sem, fmt="o", ms=4, color=ORANGE if d < -0.0005 else BLUE,
                mec="white", mew=0.6, elinewidth=0.6, capsize=1.5, ecolor=INK, zorder=3)
ax.text(FULL, len(loo) - 0.4, "UMBRA", ha="center", va="bottom", fontsize=6.5, color=INK)
ax.text(FLAT, len(loo) - 0.4, "monolithic", ha="center", va="bottom", fontsize=6.5, color=MUTED)
ax.set_yticks(ys); ax.set_yticklabels([lab for lab, _ in loo])
ax.set_title("UMBRA without:", fontsize=6.8, loc="left", pad=2, x=-0.5)
ax.set_ylim(-0.6, len(loo) + 0.3)
ax.set_xlim(0.685, 0.74); ax.set_xticks([0.69, 0.70, 0.71, 0.72, 0.73, 0.74])
ax.set_xlabel("IoU, ten probe glyphs, $N{=}3$")
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
save(fig, "fig_ablation_B")

# ── C: two panels of paired deltas, lollipops ───────────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.45, 1.4), sharey=True,
                             gridspec_kw=dict(wspace=0.12, width_ratios=[1, 1]))
ys = list(range(len(STEPS)))[::-1]
for ax, title, col in ((a1, "added, cumulative", 2), (a2, "removed from UMBRA", 4)):
    ax.axvspan(-SPREAD, SPREAD, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    for y, s in zip(ys, STEPS):
        v, ref = (s[2], s[3]) if col == 2 else (s[4], "full")
        if v is None:
            ax.text(0.0015, y, "n/a", va="center", fontsize=6, color=MUTED); continue
        d, sem = paired(v, ref)
        c = BLUE if d >= -0.0005 else ORANGE
        ax.plot([0, d], [y, y], color=c, lw=1.2, zorder=2)
        ax.errorbar(d, y, xerr=sem, fmt="o", ms=4, color=c, mec="white", mew=0.6,
                    elinewidth=0.6, capsize=1.5, ecolor=INK, zorder=3)
    ax.set_title(title, fontsize=6.8, loc="left", pad=3)
    ax.set_xlim(-0.028, 0.028); ax.set_xticks([-0.02, 0, 0.02]); ax.set_xticklabels(["$-$0.02", "0", "+0.02"])
    ax.set_xlabel("$\\Delta$ IoU")
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
a1.set_yticks(ys); a1.set_yticklabels(LAB); a1.set_ylim(-0.6, len(STEPS) - 0.4)
save(fig, "fig_ablation_C")
print("wrote A, B, C to", OUT)
