#!/usr/bin/env python3
"""Paper figures for the grounded letters+digits sweep.

Fig 1  budget x fleet size over the 186 grounded letters+digits.
Fig 2  fleet size vs silhouette quality, five distinct glyphs, N in {1,2,3,5,7}.

Both read the CSVs written by collect_letters_digits.py / collect_nsweep.py and
write PDF (for LaTeX) plus PNG (for looking at) into figures/.
"""
import csv, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/dev/umbra-bench-grounded")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)

# Categorical slots 1-5, validated for CVD separation against a light surface.
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


def num(v):
    return float(v) if v not in ("", "None", None) else None


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- figure 1
def fig_base():
    rows = load(os.path.join(ROOT, "optimized", "letters-digits-grounded",
                             "letters_digits_summary.csv"))
    subsets = ["digits", "letters_lower", "letters_upper"]
    panels = subsets + ["all"]
    label = {"digits": "digits (30)", "letters_lower": "lowercase (78)",
             "letters_upper": "uppercase (78)", "all": "all 186"}

    # Dots, not bars: the effect is ~0.03 IoU, and a zero-baseline bar chart
    # (the only honest kind) compresses it to nothing. Points carry no area, so
    # a zoomed axis is legitimate.
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.7), sharey=True)
    handles = []
    for ax, panel in zip(axes, panels):
        sel = rows if panel == "all" else [r for r in rows if r["subset"] == panel]
        for bud, colour in [("small", BLUE), ("big", ORANGE)]:
            xs, ys, es = [], [], []
            for j, n in enumerate([3, 5]):
                vals = [num(r["best_iou"]) for r in sel
                        if r["budget"] == bud and int(r["n_robots"]) == n]
                if not vals:
                    continue
                xs.append(j)
                ys.append(st.mean(vals))
                es.append(st.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0)
            line = ax.errorbar(xs, ys, yerr=es, color=colour, lw=2, marker="o",
                               ms=7, capsize=3, ecolor=colour, zorder=3,
                               label=f"{bud} budget", clip_on=False)
            if panel == "digits":
                handles.append(line)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 9),
                            textcoords="offset points", ha="center",
                            fontsize=6.5, color=INK, zorder=5)
        style(ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["3", "5"])
        ax.set_title(label[panel])
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(0.66, 0.80)
    axes[0].set_ylabel("best IoU (5 runs)")
    fig.supxlabel("robots in the fleet", y=-0.06, fontsize=9)
    fig.legend(handles=handles, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.26))
    fig.suptitle("Fleet size and search budget on grounded letters and digits",
                 y=1.02, fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig1_budget_fleet.{ext}"))
    plt.close(fig)
    print("wrote fig1_budget_fleet.pdf/.png")


# ---------------------------------------------------------------- figure 2
def fig_nsweep():
    path = os.path.join(ROOT, "optimized", "nsweep-letters-digits", "nsweep_summary.csv")
    if not os.path.exists(path):
        print("no nsweep summary yet, skipping fig 2")
        return
    rows = load(path)
    glyphs = sorted({r["id"] for r in rows})
    colours = dict(zip(glyphs, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]))
    budgets = sorted({r["budget"] for r in rows}, reverse=True)  # small, big

    fig, axes = plt.subplots(1, len(budgets), figsize=(7.2, 3.0), sharey=True)
    if len(budgets) == 1:
        axes = [axes]
    for ax, bud in zip(axes, budgets):
        sub = [r for r in rows if r["budget"] == bud]
        ns = sorted({int(r["n_robots"]) for r in sub})
        for g in glyphs:
            ys = []
            for n in ns:
                v = [num(r["best_iou"]) for r in sub
                     if r["id"] == g and int(r["n_robots"]) == n]
                ys.append(st.mean(v) if v else float("nan"))
            ax.plot(ns, ys, color=colours[g], lw=2, marker="o", ms=5,
                    zorder=3, clip_on=False)
            ax.annotate(g.split("_")[-2] if g.count("_") > 1 else g,
                        (ns[-1], ys[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center",
                        fontsize=7.5, color=colours[g])
        mean_ys = [st.mean([num(r["best_iou"]) for r in sub
                            if int(r["n_robots"]) == n]) for n in ns]
        ax.plot(ns, mean_ys, color=INK, lw=1.4, ls="--", zorder=4,
                marker="s", ms=4, label="mean of 5")
        style(ax)
        ax.set_xticks(ns)
        ax.set_xlabel("robots in the fleet")
        ax.set_title(f"{bud} budget")
        ax.legend(frameon=False, loc="lower right")
        ax.margins(x=0.14)
    axes[0].set_ylabel("best IoU (5 runs)")
    fig.suptitle("Fleet size vs silhouette quality, five glyphs",
                 y=1.02, fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig2_fleet_size.{ext}"))
    plt.close(fig)
    print("wrote fig2_fleet_size.pdf/.png")


if __name__ == "__main__":
    fig_base()
    fig_nsweep()
