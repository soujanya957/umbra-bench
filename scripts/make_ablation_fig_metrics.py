#!/usr/bin/env python3
"""fig_ablation_iou / fig_ablation_shape: per-component paired deltas on IoU, NSD and Hu distance, N=3.

    python scripts/ablation_metrics.py optimized/ablation-bench562 --n 3 --k 3 --no-clip    # 1. per-cell CSV
    python scripts/make_ablation_fig_metrics.py optimized/ablation-bench562 --k 3          # 2. figures
    python scripts/make_ablation_fig_metrics.py optimized/ablation-glyph36 --k 3 --subsets digits --out DIR
    python scripts/make_ablation_fig_metrics.py optimized/ablation-bench562 --k 3 --removed-sign action  # table signs

Writes, under --out (default <bench>/figures), with tag = <root basename>[_k<k>][_action] unless --tag:
  fig_ablation_iou_<tag>.{pdf,png}    delta IoU per component, when added and when removed,
                                      mean +- s.e.m. over targets
  fig_ablation_shape_<tag>.{pdf,png}  (a) delta NSD, mean +- s.e.m.; (b) delta Hu distance, the
                                      median of per-target differences with a bootstrap 95% CI
  fig_ablation_metrics_<tag>.json     every plotted number, the target set, the caveats
and prints the same numbers as a plain-text table to paste into a caption.

Input. The per-cell CSV of ablation_metrics.py, results/ablation_metrics/<root basename>[_k<k>].csv
(--csv overrides). Nothing is scored here: if the CSV is missing, the script prints the
ablation_metrics.py command and stops. Rows kept: this root, N=3, the 12 variants below, a mask
that was found, a finite IoU; --subsets narrows them. The target set is the (subset, stem) pairs
present in all 12 variants; its size, its per-subset split and the variants that limit it are
printed. A pair with a non-finite NSD (a shadow with no outline) drops out of that bar only, so n
is reported per bar. When the root is on this machine, the CSV is checked against its results.json
files (cells not scored yet, cells re-solved since scoring), which matters while a root is still
solving. ablation-bench562 holds the 36 glyph cells of ablation-glyph36 with 5 runs and every other
cell with 3, so it must be scored and plotted with --k 3: a CSV whose k column disagrees with --k
is refused, and a k that varies across the target set is flagged.

Components, numbered as in make_ablation_table.py's compact rows:
  #  component                        added (vs its predecessor)       removed (vs full)
  1  arm-by-arm sweep + joint stage   fwd_joint     vs flat            --
  2  zone assignment                  plus_assign   vs fwd_joint       full_no_assign
  3  aimed start                      plus_voronoi  vs plus_assign     full_no_voronoi
  4  backward sweep                   plus_backward vs plus_voronoi    full_no_backward
  5  ICP-guided restart               plus_icp      vs plus_backward   full_no_icp
  6  FD polish                        full          vs plus_icp        full_no_fd

Sign, as in make_ablation_fig.py. A bar is s * (metric with the component - metric without it),
per target, with s = +1 for IoU and NSD and s = -1 for Hu distance, which is lower-is-better.
Added (blue, "gain when added"): with = the variant, without = its predecessor. Removed (orange,
"loss when removed"): with = full, without = full_no_<x>, i.e. make_ablation_fig.py's
sign-flipped leave-one-out delta. So right of zero means the component helps, in every panel, and
each x label says so. Orange is the removal series, as in fig_ablation, not "the component hurts"
as in make_ablation_fig_options.py B/C.

--removed-sign action (default: helps) uses the paper table's convention instead: every bar is the
change in quality caused by the action, with quality = IoU, NSD or -Hu distance (higher is
better). Added is unchanged: quality(variant) - quality(predecessor). Removed becomes
quality(full_no_<x>) - quality(full), i.e. the helps bar negated, its CI (or +- s.e.m.) endpoints
swapped and negated. A component that helps is then + when added and - when removed. The legend
("change when added" / "change when removed"), the x labels ("+ = better"), the printed table and
the JSON ("removed_sign": "action", "sign", "series") say so, and the default tag gains "_action".
s.e.m., stars and n_helps / n_hurts (targets where the component helps / hurts) do not depend on
the sign. The default, helps, writes the same bytes as before the option existed.

Statistics. IoU and NSD: mean over targets, error bar one s.e.m. (sample s.d. / sqrt n), and a
star when |mean| > 2 s.e.m., the rule that sets bold in make_ablation_table.py. Hu distance is
heavy-tailed (a few targets dominate its mean), so its bar is the median of the per-target
differences, its error bar a percentile-bootstrap 95% CI (targets resampled with replacement,
--boot resamples, default 5000), and it gets a star when the CI excludes 0. Each Hu bar draws from
its own RNG stream, seeded with (--seed, component, series), so a bar is reproducible no matter
what else is plotted. fig_ablation's grey +-0.02 band is left out: it is one glyph's seed spread,
while these bars average over many targets, and their noise is the s.e.m. (or CI) shown. A bar
whose paired difference is exactly 0 on every target is labelled, "identical" for a row where
both series are, so an absent bar cannot be mistaken for missing data. With --removed-sign action a
bar whose value and CI are exactly 0 is labelled "0" as well, as the table prints it (a Hu median
of 0 with a bootstrap CI of [0, 0], while a few targets differ), and "both 0" for a row where both
series are; helps keeps the every-target rule, so such a Hu bar stays blank there.

Caveats, printed with the numbers and stored in the JSON:
  - full_no_assign is --cfg use_hungarian=false. That leaves the zone masks None, and the aimed
    start only runs with a zone mask, so "zone assignment removed" removes components 2 AND 3.
    Component 2's label carries a dagger, with a footnote in both figures.
  - At the big budget, ICP restarts never change a result on the 36 glyphs of ablation-glyph36:
    component 5's deltas are exactly 0 there. They can elsewhere (plus_icp differed from
    plus_backward on 2 lower-case letters in a partial bench562 scoring, n_dejavusans-bold and
    z_dejavusansmono-bold, run 0), so for each subset the script counts the targets whose IoU, NSD
    and Hu are identical with and without component 5, and prints the counts.

Layout. Both figures are 3.45 in (one column) wide, with make_ablation_fig.py's fonts and
colours. The plot area is refitted to that width after drawing. Ticks are the densest nice
signed steps whose labels do not collide, within a panel or across panels. Legend and footnotes
are placed from measured extents. A note ("n/a", "0") is taller than a bar, so it sits on the side
of zero that the row's other bar leaves free, and a star past the end of its own bar, whichever
way the bar points. Any text overlap or overflow that is left (labels, legend, footnotes, and stars
or notes running into a bar or another text) is printed as a
warning, so a root with other numbers does not silently yield a broken figure.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from collect_ablation import ORDER  # noqa: E402  (stdlib only)

N = 3
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#5c5b55", "#dcdbd4"  # make_ablation_fig.py
RC = {  # make_ablation_fig.py
    "font.size": 6.8, "axes.labelsize": 6.8, "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.6, "axes.titlesize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
}
WIDTH_IN, PAD_IN = 3.45, 0.03
BAR_H = 0.36
ERR_KW = dict(elinewidth=0.7, capsize=1.5, ecolor=INK)
NOTE = dict(fontsize=5.6, color=MUTED)
SERIES = (("added", BAR_H / 2, BLUE), ("removed", -BAR_H / 2, ORANGE))

COMPONENTS = [  # number, label, (added variant, its predecessor), removed variant (vs full) or None
    ("1", "arm-by-arm sweep + joint stage", ("fwd_joint", "flat"), None),
    ("2", "zone assignment", ("plus_assign", "fwd_joint"), "full_no_assign"),
    ("3", "aimed start", ("plus_voronoi", "plus_assign"), "full_no_voronoi"),
    ("4", "backward sweep", ("plus_backward", "plus_voronoi"), "full_no_backward"),
    ("5", "ICP-guided restart", ("plus_icp", "plus_backward"), "full_no_icp"),
    ("6", "FD polish", ("full", "plus_icp"), "full_no_fd"),
]
TICK_LABEL = {"1": "arm-by-arm sweep\n+ joint stage"}  # wrapped: the label column sets the panels' width
VARIANTS = ["flat", "fwd_joint", "plus_assign", "plus_voronoi", "plus_backward", "plus_icp", "full",
            "full_no_assign", "full_no_voronoi", "full_no_backward", "full_no_icp", "full_no_fd"]
ZONES, ICP = "2", "5"
METRICS = [  # CSV column, name, s (so that + = the component helps), statistic, decimals in the table
    ("iou", "IoU", +1, "mean", 4),
    ("nsd", "NSD", +1, "mean", 4),
    ("hu_distance", "Hu distance", -1, "median", 3),
]
MIN_BOOT = 2000
CAVEAT_ZONES = ("zone assignment removed = full_no_assign (--cfg use_hungarian=false): the zone masks stay None "
                "and the aimed start only runs with a zone mask, so this removes components 2 AND 3.")
FOOT_ZONES = "$^{\\dagger}$removing zone assignment also removes 3: the aimed start needs zone masks"
REMOVED_SIGNS = ("helps", "action")
CONVENTION = {  # --removed-sign -> legend and x labels; helps = the strings from before the option existed
    "helps": dict(added="gain when added", removed="loss when removed",
                  iou="$\\Delta$ IoU, mean $\\pm$ s.e.m.; right = component helps",
                  nsd="$\\Delta$ NSD\nright = helps", hu="$-\\Delta$ Hu distance\nright = helps (lower Hu)"),
    "action": dict(added="change when added", removed="change when removed",
                   iou="$\\Delta$ IoU (+ = better), mean $\\pm$ s.e.m.",
                   nsd="$\\Delta$ NSD\n(+ = better)", hu="$-\\Delta$ Hu distance\n(+ = better, lower Hu)"),
}

_REF = {v: ref for v, _, ref in ORDER}
assert all(_REF.get(v) == ref for _, _, (v, ref), _ in COMPONENTS), "COMPONENTS disagree with collect_ablation.ORDER"
assert all(rm is None or _REF.get(rm) == "full" for *_, rm in COMPONENTS), "removal rows must pair with full"
assert sorted({x for _, _, pair, rm in COMPONENTS for x in (*pair, rm) if x}) == sorted(VARIANTS)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def num(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def finite(x):
    return x is not None and math.isfinite(x)


def rel(bench, p):
    """Bench-relative when inside the checkout, absolute otherwise."""
    try:
        r = os.path.relpath(p, bench)
    except ValueError:  # other drive
        r = os.path.abspath(p)
    if r == ".." or r.startswith(".." + os.sep):
        r = os.path.abspath(p)
    return r.replace(os.sep, "/")


def symbols():
    try:
        "Δ±†".encode(sys.stdout.encoding or "ascii")
        return dict(d="Δ", pm="±", dag="†")
    except (UnicodeEncodeError, LookupError):
        return dict(d="d", pm="+-", dag="^")


def clean(x):
    """NaN -> None, so the JSON stays standard."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


# --- input ---------------------------------------------------------------------------

def read_cells(path, root_rel, subsets):
    """-> ({variant: {(subset, stem): {iou, nsd, hu_distance, k}}}, info) for this root at N=3."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        lacking = {"root", "variant", "N", "subset", "stem", "k", "mask_source", "iou", "nsd",
                   "hu_distance"} - set(reader.fieldnames or ())
        if lacking:
            sys.exit(f"{path}: not an ablation_metrics.py CSV (no column {', '.join(sorted(lacking))})")
        rows = list(reader)
    roots = Counter(r["root"] for r in rows if r["variant"] != "target")
    if roots and root_rel not in roots:
        sys.exit(f"{path} holds root {', '.join(sorted(roots))}, not {root_rel}")
    cells = {v: {} for v in VARIANTS}
    info = dict(seen={}, unscored=Counter(), subsets=Counter())
    for r in rows:
        if r["root"] != root_rel or r["variant"] not in cells or num(r["N"]) != N:
            continue
        info["subsets"][r["subset"]] += 1
        if subsets and r["subset"] not in subsets:
            continue
        key = (r["subset"], r["stem"])
        info["seen"][(r["variant"], key)] = num(r.get("rj_mtime"))
        iou = num(r["iou"])
        if r["mask_source"] == "missing" or not finite(iou):
            info["unscored"][r["variant"]] += 1
            continue
        k = num(r["k"])
        cells[r["variant"]][key] = dict(iou=iou, nsd=num(r["nsd"]), hu_distance=num(r["hu_distance"]),
                                        k=None if k is None else int(k))
    return cells, info


def freshness(root, seen, subsets):
    """Cells on disk that the CSV lacks, and cells whose results.json changed after scoring."""
    disk = {}
    for v in VARIANTS:
        for rj in glob.glob(os.path.join(root, f"{v}-n{N}", "*", "*", "results.json")):
            cdir = os.path.dirname(rj)
            key = (os.path.basename(os.path.dirname(cdir)), os.path.basename(cdir))
            if subsets and key[0] not in subsets:
                continue
            try:
                disk[(v, key)] = os.path.getmtime(rj)
            except OSError:  # being rewritten by a running solve
                continue
    unscored = sorted(c for c in disk if c not in seen)
    stale = sorted(c for c, mt in disk.items() if c in seen and seen[c] is not None and seen[c] != mt)
    return len(disk), unscored, stale


# --- statistics ------------------------------------------------------------------------

def paired(cells, key, sign, with_v, without_v, targets):
    """s * (with - without) per target, skipping targets where either value is not finite."""
    out = []
    for t in targets:
        a, b = cells[with_v][t][key], cells[without_v][t][key]
        if finite(a) and finite(b):
            out.append(sign * (a - b))
    return np.asarray(out, dtype=np.float64)


def counts(d):
    return dict(n=int(d.size), n_helps=int((d > 0).sum()), n_hurts=int((d < 0).sum()), n_zero=int((d == 0).sum()))


def mean_sem(d):
    n = d.size
    m = float(d.mean()) if n else float("nan")
    sem = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return dict(stat="mean", value=m, sem=sem, lo=m - sem, hi=m + sem,
                marked=bool(n > 1 and abs(m) > 2 * sem), **counts(d))


def median_ci(d, boot, seed, stream, level=0.95):
    n = d.size
    if n == 0:
        return dict(stat="median", value=float("nan"), lo=float("nan"), hi=float("nan"), marked=False,
                    level=level, boot=boot, seed=[seed, *stream], **counts(d))
    rng = np.random.default_rng([seed, *stream])
    meds = np.empty(boot)
    step = max(1, 4_000_000 // n)  # bounded index array for large target sets
    for i in range(0, boot, step):
        b = min(step, boot - i)
        meds[i:i + b] = np.median(d[rng.integers(0, n, size=(b, n))], axis=1)
    lo, hi = (float(x) for x in np.percentile(meds, [50 * (1 - level), 50 * (1 + level)]))
    return dict(stat="median", value=float(np.median(d)), lo=lo, hi=hi, marked=bool(lo > 0 or hi < 0),
                level=level, boot=boot, seed=[seed, *stream], **counts(d))


def compute(cells, targets, boot, seed):
    res = {}
    for key, name, sign, stat, dec in METRICS:
        rows = []
        for ci, (number, label, (v, ref), rm) in enumerate(COMPONENTS):
            pairs = {"added": (v, ref), "removed": ("full", rm) if rm else None}
            row = dict(component=int(number), label=label, added_pair=dict(with_=v, without=ref),
                       removed_pair=dict(with_="full", without=rm) if rm else None)
            for si, series in enumerate(("added", "removed")):
                if pairs[series] is None:
                    row[series] = None
                    continue
                d = paired(cells, key, sign, *pairs[series], targets)
                row[series] = mean_sem(d) if stat == "mean" else median_ci(d, boot, seed, (ci, si))
            rows.append(row)
        res[key] = dict(name=name, sign=sign, statistic=stat, rows=rows)
    return res


def removed_as_action(res):
    """In place, for --removed-sign action: each removed stat, s * (full - full_no_<x>), becomes the change
    caused by removing, s * (full_no_<x> - full): value negated, (lo, hi) -> (-hi, -lo). sem, marked and the
    n_helps / n_hurts counts (targets where the component helps / hurts) stay. 0.0 - x, not -x, so that an
    exact 0 stays +0.0 in the table and the JSON."""
    for m in res.values():
        for row in m["rows"]:
            s = row["removed"]
            if s is not None:
                row["removed"] = dict(s, value=0.0 - s["value"], lo=0.0 - s["hi"], hi=0.0 - s["lo"])


def identical(cells, a, b, targets):
    """{subset: (targets, targets whose IoU, NSD and Hu are equal in variants a and b)}."""
    def same(x, y):
        return x == y or (not finite(x) and not finite(y))
    by = {}
    for t in targets:
        eq = all(same(cells[a][t][k], cells[b][t][k]) for k, *_ in METRICS)
        tot, n_eq = by.get(t[0], (0, 0))
        by[t[0]] = (tot + 1, n_eq + int(eq))
    return by


# --- text --------------------------------------------------------------------------------

def fmt(s, dec, sym):
    if s is None:
        return "n/a"
    if not finite(s["value"]):
        return "no data"
    star = " *" if s["marked"] else ""
    if s["stat"] == "mean":
        sem = f"{s['sem']:.{dec}f}" if finite(s["sem"]) else "nan"
        return f"{s['value']:+.{dec}f} {sym['pm']} {sem}{star}"
    return f"{s['value']:+.{dec}f} [{s['lo']:+.{dec}f}, {s['hi']:+.{dec}f}]{star}"


def table(res, sym, boot, seed, removed_sign="helps"):
    action = removed_sign == "action"
    removed_head = "removed (full_no_x vs full)" if action else "removed (full vs full_no_x)"
    L = []
    for key, name, sign, stat, dec in METRICS:
        if stat == "mean":
            how = f"mean paired {sym['d']} {sym['pm']} s.e.m.; * = |mean| > 2 s.e.m."
        else:
            how = f"median paired {sym['d']} [percentile bootstrap 95% CI, {boot} resamples, seed {seed}]; * = CI excludes 0"
        flip = (", sign flipped (lower Hu is better; + = Hu drops with the " + ("action)" if action else "component)")
                if sign < 0 else "")
        L.append(f"{name}{flip}: {how}")
        L.append(f"  #  {'component':<31s} {'added (vs predecessor)':<28s} {removed_head:<30s} n")
        for r in res[key]["rows"]:
            rem = fmt(r["removed"], dec, sym)
            if r["removed"] is not None and str(r["component"]) == ZONES:
                rem += f" {sym['dag']}"
            ns = [str(r[s]["n"]) for s in ("added", "removed") if r[s] is not None]
            n = ns[0] if len(set(ns)) == 1 else "/".join(ns)
            L.append(f"  {r['component']}  {r['label']:<31s} {fmt(r['added'], dec, sym):<28s} {rem:<30s} {n}")
        L.append("")
    return L


# --- figures -----------------------------------------------------------------------------

def signed(x, _pos=None):
    if abs(x) < 1e-12:
        return "0"
    return ("+" if x > 0 else "$-$") + np.format_float_positional(round(abs(x), 10), trim="-")


def tick_labels(rows):
    return [f"{r['component']}  {TICK_LABEL.get(str(r['component']), r['label'])}"
            + ("$^{\\dagger}$" if str(r["component"]) == ZONES else "") for r in rows]


def all_zero(s):
    return s is not None and s["n"] > 0 and s["n_zero"] == s["n"]


def flat(s, zero_ci):
    """No bar to see: 0 on every target, or (zero_ci) a value and CI that are exactly 0, which the table prints "0"."""
    return all_zero(s) or (zero_ci and s is not None and s["n"] > 0 and s["value"] == 0 and s["lo"] == 0 and s["hi"] == 0)


def draw_panel(ax, rows, zero_ci=False):
    """Grouped horizontal bars, component 1 on top: blue = added, orange = removed. zero_ci (--removed-sign
    action) also labels bars that flat() finds only through their CI."""
    ys = list(range(len(rows)))[::-1]
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    ext, stars, notes = [0.0], [], []
    for y, r in zip(ys, rows):
        both_zero = r["added"] is not None and r["removed"] is not None and all_zero(r["added"]) and all_zero(r["removed"])
        if both_zero:
            notes.append((y, "identical", 1))
        # two "0" notes would run into each other in one row: one row note instead
        both_flat = (zero_ci and not both_zero and r["added"] is not None and r["removed"] is not None
                     and flat(r["added"], True) and flat(r["removed"], True))
        if both_flat:
            notes.append((y, "both 0", 1))
        for series, dy, color in SERIES:
            s = r[series]
            # A note is taller than a bar, so it would run into the other series' bar in this row: put it on
            # the side of zero that bar does not cover.
            other = r["removed" if series == "added" else "added"]
            side = -1 if other is not None and finite(other["value"]) and other["value"] > 0 else 1
            if s is None or not finite(s["value"]):
                notes.append((y + dy, "n/a", side))
                continue
            v, lo, hi = s["value"], s["lo"], s["hi"]
            err = None
            if finite(lo) and finite(hi) and hi > lo:
                err = [[max(v - lo, 0.0)], [max(hi - v, 0.0)]]
                ext += [lo, hi]
            ext.append(v)
            ax.barh(y + dy, v, height=BAR_H, color=color, lw=0, zorder=2, xerr=err, error_kw=ERR_KW)
            if flat(s, zero_ci) and not both_zero and not both_flat:
                notes.append((y + dy, "0", side))
            if s["marked"]:
                side = 1 if (v > 0 or (v == 0 and lo > 0)) else -1
                end = (hi if side > 0 else lo) if err is not None else v
                stars.append((y + dy, end, side))
    lo, hi = min(ext), max(ext)
    span = (hi - lo) or 0.02
    ax.set_xlim(lo - 0.17 * span, hi + 0.17 * span)
    ax._ablation_marks = []  # (name, text): layout_problems checks these against the bars
    for y, x, side in stars:
        ax._ablation_marks.append((f"star at y={y:+.2f}", ax.annotate(
            "$\\ast$", (x, y), xytext=(1.3 * side, -0.4), textcoords="offset points",
            ha="left" if side > 0 else "right", va="center", fontsize=7, color=INK, zorder=4)))
    for y, text, side in notes:
        ax._ablation_marks.append((f"note {text!r}", ax.annotate(
            text, (0, y), xytext=(2 * side, -0.2), textcoords="offset points",
            ha="left" if side > 0 else "right", va="center", **NOTE)))
    ax.set_xticks([0.0])
    ax.xaxis.set_major_formatter(FuncFormatter(signed))
    ax.set_yticks(ys)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def _renderer(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _hit(a, b, gap=0.0):
    return a.x0 < b.x1 + gap and b.x0 < a.x1 + gap and a.y0 < b.y1 and b.y0 < a.y1


def fit_width(fig):
    """Move the plot area until everything drawn sits PAD_IN inside the figure's left and right edges."""
    W = fig.get_figwidth()
    for _ in range(5):
        bb = fig.get_tightbbox(_renderer(fig))  # inches
        dl, dr = PAD_IN - bb.x0, bb.x1 - (W - PAD_IN)
        if abs(dl) < 0.004 and abs(dr) < 0.004:
            return
        sp = fig.subplotpars
        fig.subplots_adjust(left=sp.left + dl / W, right=sp.right - dr / W)


def fit_ticks(fig, axes, max_ticks):
    """Per panel, the densest nice signed ticks (1, 2, 2.5, 5 x 10^p) whose labels stay clear of each
    other and of the other panels' labels, and inside the figure."""
    gap = 3 / 72 * fig.dpi
    for ax in axes:
        lo, hi = ax.get_xlim()

        def clear(ticks):
            ax.set_xticks(ticks)
            ax.set_xlim(lo, hi)
            r = _renderer(fig)
            mine = [t.get_window_extent(r) for t in ax.get_xticklabels() if t.get_text()]
            others = [t.get_window_extent(r) for o in axes if o is not ax for t in o.get_xticklabels() if t.get_text()]
            return (all(not _hit(a, b, gap) for i, a in enumerate(mine) for b in mine[i + 1:] + others)
                    and all(b.x0 >= 0 and b.x1 <= fig.bbox.width for b in mine))

        base = 10 ** math.floor(math.log10((hi - lo) / max_ticks))
        choice, done = None, False
        for step in (m * base * 10 ** p for p in range(3) for m in (1, 2, 2.5, 5)):
            full = [i * step for i in range(math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9) + 1)]
            if len(full) > max_ticks:
                continue
            if len(full) < 2:
                break
            trio = [x for x in (-step, 0.0, step) if lo <= x <= hi]  # both sides labelled, before a coarser step
            for ticks in [full] + ([trio] if len(trio) == 3 and len(full) > 3 else []):
                choice = ticks
                if clear(ticks):
                    done = True
                    break
            if done:
                break
        ax.set_xticks(choice or [0.0])
        ax.set_xlim(lo, hi)


def layout_problems(fig, named):
    """Pairs of text boxes that overlap, and boxes outside the figure width."""
    r = _renderer(fig)
    items = list(named)
    for i, ax in enumerate(fig.axes):
        p = f"panel {'ab'[i]} " if len(fig.axes) > 1 else ""
        items += [(f"{p}x tick {t.get_text()!r}", t) for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
        items += [(f"y label {t.get_text().splitlines()[0]!r}", t) for t in ax.get_yticklabels()
                  if t.get_visible() and t.get_text()]
        if ax.xaxis.label.get_text():
            items.append((f"{p}x label", ax.xaxis.label))
    boxes = [(name, a.get_window_extent(r)) for name, a in items]
    out = [f"{a} overlaps {b}" for i, (a, ba) in enumerate(boxes) for b, bb in boxes[i + 1:] if _hit(ba, bb, -0.5)]
    marks = []
    for i, ax in enumerate(fig.axes):  # stars and notes against the bars
        p = f"panel {'ab'[i]} " if len(fig.axes) > 1 else ""
        bars = [b.get_window_extent(r) for b in ax.patches if abs(b.get_width()) > 0]
        for name, t in getattr(ax, "_ablation_marks", []):
            tb = t.get_window_extent(r)
            marks.append((f"{p}{name}", tb))
            if any(_hit(tb, bb, -0.5) for bb in bars):
                out.append(f"{p}{name} overlaps a bar")
    # notes against the other marks, and all marks against every other text (warnings only: nothing drawn
    # changes). The two stars of one row touch by design, so star against star is not checked.
    out += [f"{a} overlaps {b}" for i, (a, ba) in enumerate(marks) for b, bb in marks[i + 1:]
            if not ("star at" in a and "star at" in b) and _hit(ba, bb, -0.5)]
    out += [f"{a} overlaps {b}" for a, ba in marks for b, bb in boxes if _hit(ba, bb, -0.5)]
    out += [f"{a} is outside the {WIDTH_IN} in width" for a, b in boxes if b.x0 < -0.5 or b.x1 > fig.bbox.width + 0.5]
    return out


def finish(fig, axes, max_ticks, star_label, footnotes, titles, out, name, conv=CONVENTION["helps"]):
    fit_width(fig)
    fit_ticks(fig, axes, max_ticks)
    fit_width(fig)
    fit_ticks(fig, axes, max_ticks)
    fit_width(fig)
    bb = fig.get_tightbbox(_renderer(fig))  # inches from the canvas origin
    W, H = fig.get_figwidth(), fig.get_figheight()
    handles = [Patch(color=BLUE, lw=0, label=conv["added"]), Patch(color=ORANGE, lw=0, label=conv["removed"])]
    if star_label:
        # mew=0: a mathtext marker is stroked by default, which draws a heavy asterisk unlike the plot's
        handles.append(Line2D([], [], ls="none", marker="$\\ast$", ms=4.6, mew=0, color=INK, label=star_label))
    # figure fractions, not inches: savefig(bbox_inches="tight") shifts transFigure but not dpi_scale_trans
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, (bb.y1 + 0.035) / H),
                     bbox_transform=fig.transFigure, ncol=len(handles), frameon=False, fontsize=6.2,
                     handlelength=1.0, handleheight=0.8, handletextpad=0.4, columnspacing=1.1,
                     borderaxespad=0, borderpad=0)
    foot = fig.text(bb.x0 / W, (bb.y0 - 0.025) / H, "\n".join(footnotes), ha="left", va="top",
                    linespacing=1.35, **NOTE)
    for p in layout_problems(fig, [("legend", leg), ("footnote", foot)] + titles):
        log(f"[!] {name}: {p}")
    return save(fig, out, name)


def save(fig, out, name):
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
        paths.append(p)
    plt.close(fig)
    return paths


def fig_iou(res, out, tag, removed_sign="helps"):
    conv = CONVENTION[removed_sign]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(WIDTH_IN, 1.2))
        fig.subplots_adjust(left=0.33, right=0.98, top=0.98, bottom=0.02)
        rows = res["iou"]["rows"]
        draw_panel(ax, rows, zero_ci=removed_sign == "action")
        ax.set_yticklabels(tick_labels(rows))
        ax.set_xlabel(conv["iou"])
        return finish(fig, [ax], 5, "$|\\Delta| > 2$ s.e.m.", [FOOT_ZONES], [], out, f"fig_ablation_iou_{tag}", conv)


def fig_shape(res, out, tag, removed_sign="helps"):
    conv = CONVENTION[removed_sign]
    with plt.rc_context(RC):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH_IN, 1.25), sharey=True, gridspec_kw=dict(wspace=0.14))
        fig.subplots_adjust(left=0.33, right=0.98, top=0.98, bottom=0.02)
        draw_panel(a1, res["nsd"]["rows"], zero_ci=removed_sign == "action")
        draw_panel(a2, res["hu_distance"]["rows"], zero_ci=removed_sign == "action")
        a1.set_yticklabels(tick_labels(res["nsd"]["rows"]))
        t1 = a1.set_title("(a) NSD, mean $\\pm$ s.e.m.", loc="left", fontsize=6.8, pad=2.5)
        t2 = a2.set_title("(b) Hu, median, 95% CI", loc="left", fontsize=6.8, pad=2.5)
        a1.set_xlabel(conv["nsd"])
        a2.set_xlabel(conv["hu"])
        foot = ["$\\ast$ (a) $|\\Delta| > 2$ s.e.m.; (b) the 95% CI excludes 0", FOOT_ZONES]
        return finish(fig, [a1, a2], 4, None, foot, [("panel a title", t1), ("panel b title", t2)],
                      out, f"fig_ablation_shape_{tag}", conv)


# --- main --------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog="See the module docstring for "
                                "the pairing, sign convention, statistics and caveats.")
    p.add_argument("root", help="ablation root, e.g. optimized/ablation-bench562 (cwd- or --bench-relative)")
    p.add_argument("--k", type=int, default=None,
                   help="best-of-k the CSV was scored with: reads <root>_k<k>.csv (default: <root>.csv)")
    p.add_argument("--csv", default=None,
                   help="per-cell CSV to read instead of results/ablation_metrics/<root>[_k<k>].csv")
    p.add_argument("--out", default=None, help="output directory (default: <bench>/figures)")
    p.add_argument("--tag", default=None, help="file-name tag (default: <root basename>[_k<k>][_action])")
    p.add_argument("--removed-sign", choices=REMOVED_SIGNS, default="helps",
                   help="removed series: helps = quality(full) - quality(full_no_<x>), + = the component helps "
                        "(orange 'loss when removed'); action = quality(full_no_<x>) - quality(full), the change "
                        "caused by removing, as in the paper's table, so a helpful component is - there; adds "
                        "_action to the default tag (default: %(default)s)")
    p.add_argument("--subsets", nargs="+", default=None, metavar="SUBSET",
                   help="only these subsets, space- or comma-separated (default: all)")
    p.add_argument("--boot", type=int, default=5000,
                   help=f"bootstrap resamples for the Hu CI, >= {MIN_BOOT} (default: %(default)s)")
    p.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed (default: %(default)s)")
    p.add_argument("--bench", default=_BENCH, help="umbra-bench checkout (default: this script's)")
    a = p.parse_args()
    if a.k is not None and a.k < 1:
        p.error("--k must be >= 1")
    if a.boot < MIN_BOOT:
        p.error(f"--boot must be >= {MIN_BOOT}")
    sym = symbols()

    bench = os.path.abspath(a.bench)
    root = os.path.abspath(a.root if os.path.isdir(a.root) else os.path.join(bench, a.root))
    root_rel = rel(bench, root)
    base = os.path.basename(root)
    csv_path = (os.path.abspath(a.csv) if a.csv else
                os.path.join(bench, "results", "ablation_metrics", base + (f"_k{a.k}" if a.k else "") + ".csv"))
    if not os.path.exists(csv_path):
        if a.csv:
            sys.exit(f"no such CSV: {a.csv}")
        cmd = f"python scripts/ablation_metrics.py {root_rel} --n {N}" + (f" --k {a.k}" if a.k else "") + " --no-clip"
        sys.exit(f"no metrics CSV at {rel(bench, csv_path)}. Score the root first, from {bench} in the "
                 f"umbra-bench env (these figures use no CLIP column):\n    {cmd}")
    subsets = {s for x in a.subsets for s in x.split(",") if s} if a.subsets else None
    action = a.removed_sign == "action"
    tag = a.tag or base + (f"_k{a.k}" if a.k else "") + ("_action" if action else "")
    out = os.path.abspath(a.out or os.path.join(bench, "figures"))

    cells, info = read_cells(csv_path, root_rel, subsets)
    if subsets and subsets - set(info["subsets"]):
        sys.exit(f"--subsets {', '.join(sorted(subsets - set(info['subsets'])))}: not in {rel(bench, csv_path)} "
                 f"for {root_rel} N={N} (it has {', '.join(sorted(info['subsets'])) or 'nothing'})")
    empty = [v for v in VARIANTS if not cells[v]]
    if empty:
        sys.exit(f"{rel(bench, csv_path)}: no scored N={N} cells for {', '.join(empty)} under {root_rel}")
    sets = [set(cells[v]) for v in VARIANTS]
    targets = sorted(set.intersection(*sets))
    union = set.union(*sets)
    if not targets:
        sys.exit(f"{rel(bench, csv_path)}: no target is common to all {len(VARIANTS)} variants")
    limited = {v: len(cells[v]) for v in VARIANTS if len(cells[v]) < len(union)}

    ks = Counter(cells[v][t]["k"] for v in VARIANTS for t in targets)
    if a.k is not None and set(ks) != {a.k}:
        sys.exit(f"{rel(bench, csv_path)}: its cells were scored best-of-k with k = "
                 f"{', '.join(str(k) for k in sorted(ks, key=str))}, not --k {a.k}")
    if len(ks) > 1:
        log(f"[!] best-of-k varies across the target set ({', '.join(f'k={k}: {c} cells' for k, c in sorted(ks.items(), key=str))}); "
            f"score with a fixed --k, e.g. python scripts/ablation_metrics.py {root_rel} --n {N} --k 3 --no-clip")
    if info["unscored"]:
        log("[!] cells in the CSV with no mask or IoU, left out: "
            + ", ".join(f"{v} {c}" for v, c in sorted(info["unscored"].items())))
    fresh = None
    if os.path.isdir(root):
        n_disk, unscored, stale = freshness(root, info["seen"], subsets)
        fresh = dict(results_json_on_disk=n_disk, not_in_csv=len(unscored), resolved_since_scoring=len(stale))
        if unscored or stale:
            ex = ", ".join(f"{v}/{s}/{t}" for v, (s, t) in (unscored + stale)[:3])
            log(f"[!] the CSV is behind {root_rel}: {len(unscored)} cell(s) on disk not scored, {len(stale)} "
                f"re-solved since scoring (e.g. {ex}). Rerun: python scripts/ablation_metrics.py {root_rel} "
                f"--n {N}" + (f" --k {a.k}" if a.k else "") + " --no-clip"
                + (f' --out-dir "{os.path.dirname(csv_path)}"' if a.csv else ""))
    else:
        log(f"[i] {root_rel} is not on this machine; the CSV's freshness was not checked")

    res = compute(cells, targets, a.boot, a.seed)
    if action:
        removed_as_action(res)
    by_subset = dict(sorted(Counter(t[0] for t in targets).items()))
    icp = {series: identical(cells, w, wo, targets)
           for series, (w, wo) in (("added", ("plus_icp", "plus_backward")), ("removed", ("full", "full_no_icp")))}

    kdesc = ", ".join(str(k) for k in sorted(ks, key=str))
    head = [f"# {root_rel}  N={N}, best-of-k k={kdesc}: {len(targets)} targets common to the {len(VARIANTS)} variants "
            f"({', '.join(f'{s} {c}' for s, c in by_subset.items())})",
            f"# csv {rel(bench, csv_path)}" + (f"; subsets {', '.join(sorted(subsets))}" if subsets else "")]
    if limited:
        head.append(f"#   of {len(union)} seen; limited by " + ", ".join(f"{v} ({c})" for v, c in limited.items()))
    if action:
        head += ["# bar = quality after the action - quality before it, per target; quality = IoU, NSD, -Hu (+ = better)",
                 "#   added = variant - its predecessor (blue, 'change when added'); removed = full_no_<x> - full",
                 "#   (orange, 'change when removed'). --removed-sign action, the paper table's signs: a component",
                 "#   that helps is + when added and - when removed", ""]
    else:
        head += ["# bar = s * (metric with the component - metric without it), per target; s = +1 IoU, NSD; s = -1 Hu",
                 "#   so + = the component helps. added = variant vs its predecessor (blue, 'gain when added');",
                 "#   removed = full vs full_no_<x> (orange, 'loss when removed', make_ablation_fig.py's sign)", ""]
    parts = []
    for series in ("added", "removed"):
        tot = sum(t for t, _ in icp[series].values())
        eq = sum(e for _, e in icp[series].values())
        parts.append((series, tot, eq, ", ".join(f"{s} {e}/{t}" for s, (t, e) in sorted(icp[series].items()))))
    if all(tot == eq for _, tot, eq, _ in parts):
        verdict = "the restarts never changed a result on these targets (component 5 deltas are exactly 0)"
    else:
        verdict = "the restarts changed the result on the targets not counted identical"
    caveat_icp = ("5 ICP-guided restart: IoU, NSD and Hu identical with and without it on "
                  + "; ".join(f"{series} {eq}/{tot} ({desc})" for series, tot, eq, desc in parts) + f"; {verdict}.")
    lines = (head + table(res, sym, a.boot, a.seed, a.removed_sign)
             + ["Caveats", f"  {sym['dag']} {CAVEAT_ZONES}", f"  {caveat_icp}"])
    print("\n".join(lines))

    os.makedirs(out, exist_ok=True)
    figs = fig_iou(res, out, tag, a.removed_sign) + fig_shape(res, out, tag, a.removed_sign)
    js = os.path.join(out, f"fig_ablation_metrics_{tag}.json")
    if action:  # no removed_sign key under helps: its JSON keeps the bytes it had before the option existed
        convention = dict(
            removed_sign="action",
            sign=("bar = quality after the action - quality before it, per target, with quality = iou, nsd or "
                  "-hu_distance (lower Hu is better), i.e. s * (metric after - metric before) with s = +1 for iou "
                  "and nsd, -1 for hu_distance; positive = better, so a component that helps is positive when added "
                  "and negative when removed (the paper table's signs). sem, marked and n_helps / n_hurts "
                  "(targets where the component helps / hurts) do not depend on the sign"),
            series={"added": ("s * (variant - its predecessor): with = the variant, without = its predecessor "
                              "(blue, 'change when added')"),
                    "removed": ("s * (full_no_<x> - full), the change caused by removing: removed_pair with_ = full, "
                                "without = full_no_<x>; value = minus the --removed-sign helps value, lo / hi = minus "
                                "its hi / lo (orange, 'change when removed')")})
    else:
        convention = dict(
            sign=("bar = s * (metric with the component - metric without it), per target; s = +1 for iou and nsd, "
                  "-1 for hu_distance; positive = the component helps"),
            series={"added": "with = the variant, without = its predecessor (blue, 'gain when added')",
                    "removed": "with = full, without = full_no_<x> (orange, 'loss when removed')"})
    doc = dict(
        root=root_rel, csv=rel(bench, csv_path), N=N, k=a.k, k_in_csv=sorted(ks, key=str), tag=tag,
        subsets=sorted(subsets) if subsets else None, generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        n_targets=len(targets), n_by_subset=by_subset, n_seen=len(union), limited_by=limited, freshness=fresh,
        **convention,
        statistics={"iou": "mean over targets; sem = sample sd / sqrt(n); marked = |mean| > 2 sem",
                    "nsd": "mean over targets; sem = sample sd / sqrt(n); marked = |mean| > 2 sem",
                    "hu_distance": (f"median of per-target differences; percentile bootstrap 95% CI over targets, "
                                    f"{a.boot} resamples, rng seeded with [seed, component index, series index] "
                                    f"(seed {a.seed}); marked = CI excludes 0")},
        caveats=[CAVEAT_ZONES, caveat_icp],
        icp_identical={s: {k: dict(targets=t, identical=e) for k, (t, e) in v.items()} for s, v in icp.items()},
        metrics=res, figures=[rel(bench, f) for f in figs], targets=[f"{s}/{t}" for s, t in targets], table=lines)
    with open(js, "w", encoding="utf-8") as f:
        json.dump(clean(doc), f, indent=1)
    log(f"[make_ablation_fig_metrics] -> {', '.join(figs)}, {js}")


if __name__ == "__main__":
    main()
