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
    C_IOU, C_SEC, C_ORIG = "#0072B2", "#D55E00", "#009E73"

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.0),
                             gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.55))
    x = np.arange(len(ns))

    ax = axes[0]
    b = [np.mean([r["best_iou"] for r in by[n]]) for n in ns]
    be = [sem([r["best_iou"] for r in by[n]]) for n in ns]
    ax.errorbar(x, b, yerr=be, fmt="o-", color=C_IOU, lw=1.3, ms=4, capsize=2.5,
                label="vs fitted target")
    o = [np.mean([r["best_vs_original"] for r in by[n]]) for n in ns]
    oe = [sem([r["best_vs_original"] for r in by[n]]) for n in ns]
    ax.errorbar(x, o, yerr=oe, fmt="s--", color=C_ORIG, lw=1.2, ms=3.5, capsize=2.5,
                label="vs original target")
    ax.set_xticks(x); ax.set_xticklabels([f"{n}" for n in ns])
    ax.set_xlabel("arms $N$"); ax.set_ylabel("best IoU")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="center right", handletextpad=0.4)
    ax.set_title("(a) quality saturates by $N{=}3$", loc="left")

    ax = axes[1]
    ids = sorted({r["id"] for n in ns for r in by[n]})
    for i in range(1, len(ns)):
        lo, hi = ns[i - 1], ns[i]
        dl = {r["id"]: r["best_iou"] for r in by[lo]}
        dh = {r["id"]: r["best_iou"] for r in by[hi]}
        d = [dh[k] - dl[k] for k in ids if k in dl and k in dh]
        ax.scatter(np.full(len(d), i - 1) + np.random.default_rng(i).normal(0, .06, len(d)),
                   d, s=6, color="0.3", alpha=0.55, lw=0)
        ax.errorbar(i - 1, np.mean(d), yerr=sem(d), fmt="o", color=C_SEC, ms=5,
                    capsize=3, zorder=5)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xticks(np.arange(len(ns) - 1))
    ax.set_xticklabels([f"{ns[i-1]}$\\to${ns[i]}" for i in range(1, len(ns))])
    ax.set_xlabel("arms added"); ax.set_ylabel(r"$\Delta$ best IoU per target")
    ax.set_title("(b) past $N{=}3$ an added arm\nhelps and hurts about equally", loc="left")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_nsweep.{ext}"))
    print(f"[fig] {outdir}/fig_nsweep.pdf / .png")


if __name__ == "__main__":
    main()
