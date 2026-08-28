#!/usr/bin/env python3
"""
analyze_nsweep.py -- fleet size vs static benchmark quality.

Reads optimized/nsweep-fitted/n<N>/<subset>/<stem>/results.json and answers the
question the paper currently answers with six targets: what does an extra arm
buy on a benchmark that spans the difficulty range?

Writes nsweep.csv and, with --fig, fig_nsweep.{pdf,png}.
"""
import argparse, collections, csv, glob, json, os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sem(v):
    v = np.asarray(v, float)
    return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def load(runroot):
    rows = []
    for p in sorted(glob.glob(os.path.join(runroot, "n*", "*", "*", "results.json"))):
        d = json.load(open(p))
        n = int(os.path.basename(os.path.dirname(os.path.dirname(
            os.path.dirname(p)))).lstrip("n"))
        rows.append(dict(
            n=n, id=d.get("id"), subset=d.get("subset"),
            best_iou=d.get("best_iou"), mean_iou=d.get("mean_iou"),
            std_iou=d.get("std_iou"),
            best_vs_original=d.get("best_iou_vs_original"),
            seconds=d.get("seconds"),
            uncastable_after=(d.get("fit") or {}).get("uncastable_after"),
            fit_scale=(d.get("fit") or {}).get("scale"),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runroot", default=os.path.join(ROOT, "optimized", "nsweep-fitted"))
    ap.add_argument("--out", default=os.path.join(ROOT, "optimized", "nsweep-fitted", "nsweep.csv"))
    ap.add_argument("--fig", default=None, help="write fig_nsweep.{pdf,png} here")
    a = ap.parse_args()

    rows = load(a.runroot)
    if not rows:
        raise SystemExit(f"no results under {a.runroot}")
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}  ({len(rows)} target-solves)")

    by = collections.defaultdict(list)
    for r in rows:
        by[r["n"]].append(r)
    ns = sorted(by)

    print(f"\n{'N':>3} {'targets':>8} {'best IoU':>16} {'mean IoU':>16} "
          f"{'vs original':>12} {'sec/target':>11}")
    print("-" * 72)
    prev = None
    for n in ns:
        g = by[n]
        b = [r["best_iou"] for r in g if r["best_iou"] is not None]
        m = [r["mean_iou"] for r in g if r["mean_iou"] is not None]
        o = [r["best_vs_original"] for r in g if r["best_vs_original"] is not None]
        s = [r["seconds"] for r in g if r["seconds"] is not None]
        delta = f"  ({np.mean(b) - prev:+.3f})" if prev is not None else ""
        print(f"{n:>3} {len(g):>8} {np.mean(b):>8.4f}±{sem(b):<7.4f} "
              f"{np.mean(m):>8.4f}±{sem(m):<7.4f} {np.mean(o):>12.4f} "
              f"{np.mean(s):>11.1f}{delta}")
        prev = np.mean(b)

    # per-target marginal value of each added arm -- the mean hides sign flips
    print("\nPer-target change from the previous N (best IoU):")
    ids = sorted({r["id"] for r in rows})
    for i in range(1, len(ns)):
        lo, hi = ns[i - 1], ns[i]
        dl = {r["id"]: r["best_iou"] for r in by[lo]}
        dh = {r["id"]: r["best_iou"] for r in by[hi]}
        d = [dh[k] - dl[k] for k in ids if k in dl and k in dh]
        worse = sum(1 for x in d if x < -0.005)
        print(f"  N={lo}->{hi}: mean {np.mean(d):+.4f}  improved "
              f"{sum(1 for x in d if x > 0.005)}/{len(d)}, WORSE {worse}/{len(d)}")

    if a.fig:
        make_fig(by, ns, a.fig)


def make_fig(by, ns, outdir):
    """Three panels, because the headline is a disagreement between two metrics
    and the disagreement has a mechanism worth showing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.linewidth": 0.6, "font.family": "sans-serif",
    })
    C_FIT, C_ORIG, C_SC = "#0072B2", "#009E73", "#D55E00"

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.05),
                             gridspec_kw=dict(width_ratios=[1.05, 0.85, 1.05],
                                              wspace=0.62))
    x = np.arange(len(ns))

    # (a) the two metrics disagree about where arms stop paying
    ax = axes[0]
    b = [np.mean([r["best_iou"] for r in by[n]]) for n in ns]
    be = [sem([r["best_iou"] for r in by[n]]) for n in ns]
    ax.errorbar(x, b, yerr=be, fmt="o-", color=C_FIT, lw=1.3, ms=4, capsize=2.5,
                label="vs fitted target")
    o = [np.mean([r["best_vs_original"] for r in by[n]]) for n in ns]
    oe = [sem([r["best_vs_original"] for r in by[n]]) for n in ns]
    ax.errorbar(x, o, yerr=oe, fmt="s--", color=C_ORIG, lw=1.2, ms=3.5, capsize=2.5,
                label="vs target as authored")
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("arms $N$"); ax.set_ylabel("best IoU")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.4, fontsize=6.0)
    ax.set_title("(a) saturation is a property\nof the metric, not the fleet",
                 loc="left")

    # (b) mechanism: a bigger fleet is allowed to keep the target bigger
    ax = axes[1]
    sc = [np.mean([r["fit_scale"] for r in by[n] if r["fit_scale"]]) for n in ns]
    sce = [sem([r["fit_scale"] for r in by[n] if r["fit_scale"]]) for n in ns]
    ax.errorbar(x, sc, yerr=sce, fmt="D-", color=C_SC, lw=1.3, ms=3.5, capsize=2.5)
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.text(0.05, 1.01, "target kept at\nauthored size", fontsize=5.6, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("arms $N$"); ax.set_ylabel("fit scale")
    ax.set_title("(b) small fleets shrink\nthe target to fit", loc="left")

    # (c) per-target marginal value, both metrics
    ax = axes[2]
    ids = sorted({r["id"] for n in ns for r in by[n]})
    w = 0.36
    for j, (key, col, lab) in enumerate((("best_iou", C_FIT, "vs fitted"),
                                         ("best_vs_original", C_ORIG, "vs authored"))):
        mu, er = [], []
        for i in range(1, len(ns)):
            dl = {r["id"]: r[key] for r in by[ns[i - 1]]}
            dh = {r["id"]: r[key] for r in by[ns[i]]}
            d = [dh[k] - dl[k] for k in ids if k in dl and k in dh]
            mu.append(np.mean(d)); er.append(sem(d))
        ax.bar(np.arange(len(mu)) + (j - 0.5) * w, mu, yerr=er, width=w, color=col,
               edgecolor="black", linewidth=0.4, capsize=2, label=lab)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(np.arange(len(ns) - 1))
    ax.set_xticklabels([f"{ns[i-1]}$\\to${ns[i]}" for i in range(1, len(ns))],
                       fontsize=6.4)
    ax.set_xlabel("arms added"); ax.set_ylabel(r"$\Delta$ best IoU")
    ax.legend(frameon=False, loc="upper right", fontsize=6.0, handlelength=1.0)
    ax.set_title("(c) the 4th arm pays only\non the authored target", loc="left")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_nsweep.{ext}"))
    print(f"[fig] {outdir}/fig_nsweep.pdf / .png")


if __name__ == "__main__":
    main()
