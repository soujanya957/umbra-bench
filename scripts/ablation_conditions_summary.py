#!/usr/bin/env python3
"""Aggregate ablation_metrics.py's per-cell CSV into one per-condition summary, broken down by category.

    python scripts/ablation_conditions_summary.py
    python scripts/ablation_conditions_summary.py --print all glyphs animals

Every number here is a plain aggregate of the per-cell CSV that
scripts/ablation_metrics.py writes (one row per (variant, subset, stem): the
best-of-k run and its iou / nsd / hu_distance / boundary_iou, plus CLIP
rank / top1 / rr / rr_cc / p_true on the glyph subsets). That CSV, not this
summary, is the source of truth. Any other grouping -- a different mix of
categories, per-font splits, medians instead of means -- is a groupby on it,
and conditions.md ends with the three lines that do it.

Scopes: `all`, the two groups (glyphs = letters_upper + letters_lower + digits,
shapes = the other five) and each of the 8 categories, so 11 scopes x 12
conditions. CLIP ranks a shadow inside its subset's own vocabulary, so it
exists only where every subset in the scope has one: the glyphs group and the
three glyph categories. Those scopes also get a `target` row -- the authored
PNG through the same scorer, the ceiling the solver is working towards.

renders_per_solve is measured, not configured: per cell the mean n_evals of the
runs best-of-k ranged over (runs < k, extras excluded), read from each
results.json, then averaged over the scope's cells.

Deltas are paired within the scope: the mean over targets of
(condition - flat) and (condition - full), never a difference of column means.
s.e.m. is the sample sd of those per-target differences over sqrt(n).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics as st
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)

# (condition, label, block). The build-up ladder adds one component at a time to monolithic CMA-ES;
# the leave-one-out block takes one back out of UMBRA. Order is the order the tables print in.
ORDER = [
    ("flat",             "monolithic CMA-ES (baseline)",   "build-up"),
    ("fwd_joint",        "+1 arm-by-arm sweep + joint stage", "build-up"),
    ("plus_assign",      "+2 zone assignment",             "build-up"),
    ("plus_voronoi",     "+3 aimed start",                 "build-up"),
    ("plus_backward",    "+4 backward sweep",              "build-up"),
    ("plus_icp",         "+5 ICP-guided restart",          "build-up"),
    ("full",             "+6 FD polish (= UMBRA)",         "build-up"),
    ("full_no_assign",   "UMBRA - 2 zone assignment",      "leave-one-out"),
    ("full_no_voronoi",  "UMBRA - 3 aimed start",          "leave-one-out"),
    ("full_no_backward", "UMBRA - 4 backward sweep",       "leave-one-out"),
    ("full_no_icp",      "UMBRA - 5 ICP-guided restart",   "leave-one-out"),
    ("full_no_fd",       "UMBRA - 6 FD polish",            "leave-one-out"),
]
LABEL = {c: lab for c, lab, _ in ORDER}
BLOCK = {c: b for c, _, b in ORDER}
BASE, UMBRA = "flat", "full"
DAGGER = {"full_no_assign"}  # removing zone assignment also removes the aimed start (it needs the zone masks)

GLYPH_SUBSETS = ["letters_upper", "letters_lower", "digits"]          # have a CLIP vocabulary
SHAPE_SUBSETS = ["abstract", "animals", "hand_shadow", "objects", "teleop"]
CATEGORIES = GLYPH_SUBSETS + SHAPE_SUBSETS                            # table order for the per-category tables
CHANCE = {"letters_upper": (26, 1 / 26), "letters_lower": (26, 1 / 26), "digits": (10, 1 / 10)}
CHANCE_MRR = {s: sum(1 / i for i in range(1, K + 1)) / K for s, (K, _) in CHANCE.items()}

WIDE_COLS = ["scope", "scope_kind", "condition", "label", "block", "n_targets",
             "iou_mean", "iou_sem", "iou_median", "nsd_mean", "nsd_sem",
             "hu_median", "hu_iqr_lo", "hu_iqr_hi", "hu_mean", "boundary_iou_mean",
             "renders_per_solve", "clip_top1", "clip_mrr", "clip_mrr_cc", "clip_p_true",
             "d_iou_vs_monolithic", "d_iou_vs_umbra"]
# (wide column, metric, statistic) for the long form; n is filled per row.
TIDY_MAP = [("iou_mean", "iou", "mean"), ("iou_sem", "iou", "sem"), ("iou_median", "iou", "median"),
            ("nsd_mean", "nsd", "mean"), ("nsd_sem", "nsd", "sem"),
            ("hu_median", "hu_distance", "median"), ("hu_iqr_lo", "hu_distance", "iqr_lo"),
            ("hu_iqr_hi", "hu_distance", "iqr_hi"), ("hu_mean", "hu_distance", "mean"),
            ("boundary_iou_mean", "boundary_iou", "mean"),
            ("renders_per_solve", "renders_per_solve", "mean"),
            ("clip_top1", "clip_top1", "mean"), ("clip_mrr", "clip_mrr", "mean"),
            ("clip_mrr_cc", "clip_mrr_cc", "mean"), ("clip_p_true", "clip_p_true", "mean"),
            ("d_iou_vs_monolithic", "d_iou_vs_monolithic", "mean"),
            ("d_iou_vs_umbra", "d_iou_vs_umbra", "mean")]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def num(x):
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def mean_sem(vals):
    """-> (mean, sem, n). sem = sample sd / sqrt(n), undefined for n < 2."""
    if not vals:
        return None, None, 0
    m = st.mean(vals)
    sem = st.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else None
    return m, sem, len(vals)


# --- inputs ---------------------------------------------------------------------

def load_cells(path, n):
    """-> (cells {condition: {(subset, stem): row}}, targets {(subset, stem): row}, problems [str])."""
    cells, targets, problems = {}, {}, []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["variant"] == "target":
                targets[(r["subset"], r["stem"])] = r
                continue
            if str(r["N"]) != str(n):
                continue
            key = (r["subset"], r["stem"])
            if r["mask_source"] == "missing" or num(r.get("iou")) is None:
                problems.append(f"{r['variant']}/{r['subset']}/{r['stem']}: "
                                f"mask_source={r['mask_source']} note={r.get('note', '')}")
                continue
            cells.setdefault(r["variant"], {})[key] = r
    return cells, targets, problems


def load_renders(root, n, k):
    """-> {(variant, subset, stem): mean n_evals over the runs best-of-k ranged over}."""
    out = {}
    for rj in glob.glob(os.path.join(root, f"*-n{n}", "*", "*", "results.json")):
        parts = os.path.normpath(rj).split(os.sep)
        variant = parts[-4][: -len(f"-n{n}")]
        try:
            with open(rj, encoding="utf-8") as f:
                r = json.load(f)
        except (OSError, ValueError):
            continue
        kk = k or int(r.get("n_base_runs") or len(r["runs"]))
        ev = [x["n_evals"] for x in r["runs"]
              if not x.get("extra") and int(x["run"]) < kk and x.get("n_evals")]
        if ev:
            out[(variant, parts[-3], parts[-2])] = st.mean(ev)
    return out


# --- aggregation ----------------------------------------------------------------

def scopes():
    out = [("all", "all", CATEGORIES), ("glyphs", "group", GLYPH_SUBSETS), ("shapes", "group", SHAPE_SUBSETS)]
    out += [(c, "category", [c]) for c in CATEGORIES]
    return out


def has_clip(subs):
    """CLIP ranks within a subset's own vocabulary, so a scope only has it if every subset has one."""
    return all(s in CHANCE for s in subs)


def aggregate(cells, targets, renders, k):
    """-> list of wide rows (dicts over WIDE_COLS) + per-row extras used only by the renderers."""
    rows = []
    for scope, kind, subs in scopes():
        subs = set(subs)
        keys = sorted({key for c in cells.values() for key in c if key[0] in subs})
        clip_ok = has_clip(subs)
        if clip_ok and targets:
            tk = [targets[key] for key in keys if key in targets and num(targets[key].get("clip_rr")) is not None]
            if tk:
                row = {c: "" for c in WIDE_COLS}
                row.update(scope=scope, scope_kind=kind, condition="target", label="target (authored image)",
                           block="ceiling", n_targets=len(tk),
                           clip_top1=st.mean(num(t["clip_top1"]) for t in tk),
                           clip_mrr=st.mean(num(t["clip_rr"]) for t in tk),
                           clip_mrr_cc=st.mean(num(t["clip_rr_cc"]) for t in tk),
                           clip_p_true=st.mean(num(t["clip_p_true"]) for t in tk))
                row["_n_clip"] = len(tk)
                rows.append(row)
        for cond, label, block in ORDER:
            if cond not in cells:
                continue
            got = [(key, cells[cond][key]) for key in keys if key in cells[cond]]
            if not got:
                continue
            iou = [num(r["iou"]) for _, r in got]
            nsd = [num(r["nsd"]) for _, r in got]
            hu = [num(r["hu_distance"]) for _, r in got]
            biou = [num(r["boundary_iou"]) for _, r in got]
            q25, q50, q75 = (float(x) for x in np.percentile(hu, [25, 50, 75]))
            iou_m, iou_s, _ = mean_sem(iou)
            nsd_m, nsd_s, _ = mean_sem(nsd)
            rend = [renders[(cond,) + key] for key, _ in got if (cond,) + key in renders]
            row = {c: "" for c in WIDE_COLS}
            row.update(scope=scope, scope_kind=kind, condition=cond, label=label, block=block,
                       n_targets=len(got), iou_mean=iou_m, iou_sem=iou_s, iou_median=float(np.median(iou)),
                       nsd_mean=nsd_m, nsd_sem=nsd_s, hu_median=q50, hu_iqr_lo=q25, hu_iqr_hi=q75,
                       hu_mean=st.mean(hu), boundary_iou_mean=st.mean(biou),
                       renders_per_solve=st.mean(rend) if rend else "")
            row["_n_renders"] = len(rend)
            if clip_ok:
                cl = [r for _, r in got if num(r.get("clip_rr")) is not None]
                if cl:
                    row.update(clip_top1=st.mean(num(r["clip_top1"]) for r in cl),
                               clip_mrr=st.mean(num(r["clip_rr"]) for r in cl),
                               clip_mrr_cc=st.mean(num(r["clip_rr_cc"]) for r in cl),
                               clip_p_true=st.mean(num(r["clip_p_true"]) for r in cl))
                row["_n_clip"] = len(cl)
            for col, ref in (("d_iou_vs_monolithic", BASE), ("d_iou_vs_umbra", UMBRA)):
                d = [num(r["iou"]) - num(cells[ref][key]["iou"])
                     for key, r in got if ref in cells and key in cells[ref]]
                m, s, dn = mean_sem(d)
                row[col] = m if m is not None else ""
                row["_" + col] = (m, s, dn)
            rows.append(row)
    return rows


# --- formatting -----------------------------------------------------------------

def f3(v, dec=3):
    return "-" if v == "" or v is None else f"{v:.{dec}f}"


def fpm(m, s, dec=3):
    if m == "" or m is None:
        return "-"
    return f"{m:.{dec}f}" if s is None else f"{m:.{dec}f} ± {s:.{dec}f}"


def fren(v):
    return "-" if v == "" or v is None else f"{v / 1000:.1f}k"


def fhu(row):
    if row["hu_median"] == "":
        return "-"
    return f"{row['hu_median']:.3f} [{row['hu_iqr_lo']:.3f}, {row['hu_iqr_hi']:.3f}]"


def fdelta(row, col, md=True):
    """Paired delta as mean ± s.e.m.; bold when |mean| is beyond 2 s.e.m. 'ref' for the row itself."""
    m, s, n = row.get("_" + col, ("", None, 0))
    if m == "" or m is None or n == 0:
        return "-"
    if (col == "d_iou_vs_monolithic" and row["condition"] == BASE) or \
       (col == "d_iou_vs_umbra" and row["condition"] == UMBRA):
        return "ref"
    txt = f"{m:+.3f}" + (f" ± {s:.3f}" if s is not None else "")
    strong = s is not None and abs(m) > 2 * s
    return f"**{txt}**" if (md and strong) else txt


def name(row, md=True):
    """The row's label. The dagger is markdown's; LaTeX appends its own $^\\dagger$."""
    lab = row["label"]
    if md and row["condition"] in DAGGER:
        lab += " †"
    return lab


# --- markdown -------------------------------------------------------------------

CORE_HDR = ["condition", "n", "IoU", "IoU med", "NSD", "Hu med [IQR]", "Hu mean", "bIoU",
            "renders", "Δ IoU vs flat", "Δ IoU vs UMBRA"]
CLIP_HDR = ["CLIP top-1", "MRR", "MRR_cc", "p_true"]
SMALL_HDR = ["condition", "n", "IoU", "NSD", "Hu med", "bIoU", "Δ IoU vs flat", "Δ IoU vs UMBRA"]


def _core_cells(row):
    return [name(row), str(row["n_targets"]), fpm(row["iou_mean"], row["iou_sem"]), f3(row["iou_median"]),
            fpm(row["nsd_mean"], row["nsd_sem"]), fhu(row), f3(row["hu_mean"]), f3(row["boundary_iou_mean"]),
            fren(row["renders_per_solve"]), fdelta(row, "d_iou_vs_monolithic"), fdelta(row, "d_iou_vs_umbra")]


def _small_cells(row):
    return [name(row), str(row["n_targets"]), fpm(row["iou_mean"], row["iou_sem"]), fpm(row["nsd_mean"], row["nsd_sem"]),
            f3(row["hu_median"]), f3(row["boundary_iou_mean"]),
            fdelta(row, "d_iou_vs_monolithic"), fdelta(row, "d_iou_vs_umbra")]


def _clip_cells(row):
    return [f3(row["clip_top1"]), f3(row["clip_mrr"]), f3(row["clip_mrr_cc"]), f3(row["clip_p_true"])]


def md_table(rows, small=False):
    clip = any(r["clip_mrr"] != "" for r in rows)
    hdr = (SMALL_HDR if small else CORE_HDR) + (CLIP_HDR if clip else [])
    ncol = len(hdr)
    out = ["| " + " | ".join(hdr) + " |",
           "|" + "|".join(["---"] * ncol) + "|"]
    ceiling = [r for r in rows if r["condition"] == "target"]
    body = [r for r in rows if r["condition"] != "target"]
    prev = None
    for r in body:
        if prev == "build-up" and r["block"] == "leave-one-out":
            out.append("| *leave-one-out from UMBRA* |" + " |" * (ncol - 1))
        cells = (_small_cells(r) if small else _core_cells(r)) + (_clip_cells(r) if clip else [])
        out.append("| " + " | ".join(cells) + " |")
        prev = r["block"]
    for r in ceiling:
        blank = ["-"] * ((len(SMALL_HDR) if small else len(CORE_HDR)) - 2)
        out.append("| " + " | ".join([name(r), str(r["n_targets"])] + blank + _clip_cells(r)) + " |")
    return "\n".join(out)


def scope_n(rows):
    """Targets in a scope: the ceiling row is one per glyph, so read it off a real condition."""
    for r in rows:
        if r["condition"] != "target":
            return r["n_targets"]
    return 0


def write_md(path, rows, meta):
    by = {}
    for r in rows:
        by.setdefault(r["scope"], []).append(r)
    L = []
    L.append(f"# Ablation conditions, per category — {meta['root']}")
    L.append("")
    L.append(f"{meta['n_cells']} cells: {len(ORDER)} conditions x {meta['n_targets_all']} targets, "
             f"N={meta['N']} arms, best of {meta['k']} seeds (k={meta['k']}: the run with the highest per-run IoU "
             f"among runs < {meta['k']}, extras excluded), all solved unfitted against the authored target.")
    L.append("")
    L.append("Columns. IoU / NSD / boundary IoU are means over the scope's targets with the s.e.m. of that mean "
             "(sample sd / sqrt(n)); higher is better. Hu distance is lower-is-better and outlier-heavy, so it is "
             "reported as median [interquartile range] and, separately, as a mean. renders = measured renders per "
             "solve (mean n_evals of the runs best-of-k ranged over). **Δ IoU is paired within the scope**: the mean "
             "over targets of (this condition − flat) and (this condition − UMBRA), with the s.e.m. of those "
             "per-target differences; bold = beyond 2 s.e.m.; `ref` marks the row a column is measured against. "
             "A component that helps reads + when added and − when removed.")
    L.append("")
    L.append("† `full_no_assign` (`--cfg use_hungarian=false`) removes components 2 **and** 3: the aimed start needs "
             "the zone masks that assignment produces.")
    L.append("")
    L.append("CLIP (ViT-B-32 laion2b_s34b_b79k, glyph prompts) ranks each shadow inside its own subset's vocabulary, "
             "so it exists **only for the three glyph categories** — the 376 shape targets have no vocabulary and no "
             "CLIP, and the `all` and `shapes` scopes therefore leave the CLIP columns empty. Chance: letters_upper "
             f"and letters_lower K=26, top-1 {1/26:.3f}, MRR {CHANCE_MRR['letters_upper']:.6f}; digits K=10, top-1 "
             f"{1/10:.3f}, MRR {CHANCE_MRR['digits']:.6f}. MRR_cc = (RR − chance)/(1 − chance) per glyph then "
             "averaged (0 = chance, 1 = always rank 1) and is the only CLIP column comparable across the two K, so "
             "prefer it when reading the pooled glyphs row. `target (authored image)` is the ceiling: the authored "
             "PNG through the same scorer.")
    L.append("")
    L.append(f"**Source of truth: `{meta['cells_csv']}`** — one row per (variant, subset, stem) with the chosen run "
             "and its per-cell metrics. Every number below is a groupby on that file; regroup it any way you like "
             "(see the end of this document).")
    L.append("")
    L.append(f"## Table A — all {scope_n(by['all'])} targets")
    L.append("")
    L.append(md_table(by["all"]))
    L.append("")
    L.append(f"## Table B — glyphs {scope_n(by['glyphs'])} (letters_upper + letters_lower + digits, with CLIP)")
    L.append("")
    L.append(md_table(by["glyphs"]))
    L.append("")
    L.append(f"## Table C — shapes {scope_n(by['shapes'])} (abstract, animals, hand_shadow, objects, teleop)")
    L.append("")
    L.append(md_table(by["shapes"]))
    L.append("")
    L.append("## By category")
    L.append("")
    for cat in CATEGORIES:
        if cat not in by:
            continue
        n = [r for r in by[cat] if r["condition"] != "target"][0]["n_targets"]
        tag = " (with CLIP)" if has_clip({cat}) else ""
        L.append(f"### {cat} — {n} targets{tag}")
        L.append("")
        L.append(md_table(by[cat], small=True))
        L.append("")
    L.append("## How to regroup")
    L.append("")
    L.append("The per-cell CSV keeps every target separately, so any other grouping is three lines — swap the "
             "`isin(...)` for whatever selection you want (one category, a hand-picked list of stems, a font, "
             "`~cells.subset.isin(GLYPHS)`), or swap `.iou` for `nsd`, `hu_distance`, `boundary_iou`, `clip_rr`:")
    L.append("")
    L.append("```python")
    L.append("import pandas as pd")
    L.append(f'cells = pd.read_csv(r"{meta["cells_csv"]}").query("variant != \'target\'")')
    L.append('pick  = cells[cells.subset.isin(["animals", "objects", "hand_shadow"])]   # <- any selection')
    L.append(f'print(pick.groupby("variant").iou.mean().reindex({[c for c, _, _ in ORDER]!r}))')
    L.append("```")
    L.append("")
    L.append("Hu distance wants a median (`.hu_distance.median()`), and a paired delta wants a pivot first: "
             "`p = pick.pivot(index=['subset','stem'], columns='variant', values='iou')`, then "
             "`(p['full'] - p['flat']).mean()`.")
    L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# --- latex ----------------------------------------------------------------------

def _tex(s):
    return s.replace("_", r"\_").replace("UMBRA", r"\textsc{umbra}")


def _tex_delta(row, col):
    txt = fdelta(row, col, md=False)
    if txt in ("-", "ref"):
        return r"\textit{ref}" if txt == "ref" else "---"
    m, s, _ = row["_" + col]
    body = f"{m:+.3f}" + (rf" \pm {s:.3f}" if s is not None else "")
    return rf"$\mathbf{{{body}}}$" if (s is not None and abs(m) > 2 * s) else f"${body}$"


def _tex_pm(m, s):
    if m == "" or m is None:
        return "---"
    return f"${m:.3f}$" if s is None else rf"${m:.3f} \pm {s:.3f}$"


def tex_table(rows, caption, clip):
    ncol = 8 + (4 if clip else 0)
    L = [r"% " + caption,
         r"\begin{tabular}{@{}l" + "r" * (ncol - 1) + r"@{}}", r"\toprule"]
    hdr = [r"condition", r"$n$", r"IoU $\uparrow$", r"NSD $\uparrow$", r"Hu (med) $\downarrow$",
           r"bIoU $\uparrow$", r"$\Delta$IoU vs.\ flat", r"$\Delta$IoU vs.\ \textsc{umbra}"]
    if clip:
        hdr += [r"top-1 $\uparrow$", r"MRR $\uparrow$", r"MRR$_\mathrm{cc}$ $\uparrow$", r"$p_\mathrm{true}$ $\uparrow$"]
    L.append(" & ".join(hdr) + r" \\")
    L.append(r"\midrule")
    body = [r for r in rows if r["condition"] != "target"]
    prev = None
    for r in body:
        if prev == "build-up" and r["block"] == "leave-one-out":
            L.append(r"\midrule")
        cells = [_tex(name(r, md=False)) + (r"$^\dagger$" if r["condition"] in DAGGER else ""),
                 str(r["n_targets"]), _tex_pm(r["iou_mean"], r["iou_sem"]), _tex_pm(r["nsd_mean"], r["nsd_sem"]),
                 f"${r['hu_median']:.3f}$", f"${r['boundary_iou_mean']:.3f}$",
                 _tex_delta(r, "d_iou_vs_monolithic"), _tex_delta(r, "d_iou_vs_umbra")]
        if clip:
            cells += [f"${r['clip_top1']:.3f}$", f"${r['clip_mrr']:.3f}$",
                      f"${r['clip_mrr_cc']:.3f}$", f"${r['clip_p_true']:.3f}$"]
        L.append(" & ".join(cells) + r" \\")
        prev = r["block"]
    for r in [r for r in rows if r["condition"] == "target"]:
        L.append(r"\midrule")
        cells = ["target (ceiling)", str(r["n_targets"])] + ["---"] * 6
        cells += [f"${r['clip_top1']:.3f}$", f"${r['clip_mrr']:.3f}$",
                  f"${r['clip_mrr_cc']:.3f}$", f"${r['clip_p_true']:.3f}$"]
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def write_tex(path, rows, meta):
    by = {}
    for r in rows:
        by.setdefault(r["scope"], []).append(r)
    L = [f"% {meta['root']}, N={meta['N']}, best of {meta['k']} seeds (k={meta['k']}).",
         "% Deltas are paired within the scope; bold = beyond 2 s.e.m. of the per-target difference.",
         r"% $^\dagger$ removing zone assignment also removes the aimed start.",
         "", tex_table(by["all"], f"Table A: all {scope_n(by['all'])} targets", clip=False), "",
         tex_table(by["glyphs"], f"Table B: glyphs {scope_n(by['glyphs'])}, with CLIP", clip=True), ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# --- csv out --------------------------------------------------------------------

def write_wide(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=WIDE_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {}
            for c in WIDE_COLS:
                v = r.get(c, "")
                if isinstance(v, float):
                    v = f"{v:.6f}" if c != "renders_per_solve" else f"{v:.1f}"
                out[c] = v
            w.writerow(out)


def write_tidy(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "scope_kind", "condition", "label", "block", "metric", "statistic", "value", "n"])
        for r in rows:
            for col, metric, statistic in TIDY_MAP:
                v = r.get(col, "")
                if v == "" or v is None:
                    continue
                if metric.startswith("clip"):
                    n = r.get("_n_clip", r["n_targets"])
                elif metric == "renders_per_solve":
                    n = r.get("_n_renders", r["n_targets"])
                elif metric.startswith("d_iou"):
                    n = r["_" + col][2]
                else:
                    n = r["n_targets"]
                w.writerow([r["scope"], r["scope_kind"], r["condition"], r["label"], r["block"],
                            metric, statistic, f"{v:.6f}", n])
            # the s.e.m. of each paired delta, which the wide table has no column for
            for col in ("d_iou_vs_monolithic", "d_iou_vs_umbra"):
                m, s, n = r.get("_" + col, ("", None, 0))
                if s is not None:
                    w.writerow([r["scope"], r["scope_kind"], r["condition"], r["label"], r["block"],
                                col, "sem", f"{s:.6f}", n])


# --- main -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cells", default=None,
                   help="per-cell CSV from ablation_metrics.py (default: <out-dir>/ablation-bench562_k3.csv)")
    p.add_argument("--root", default="optimized/ablation-bench562", help="solve root, for measured renders per solve")
    p.add_argument("--bench", default=_BENCH)
    p.add_argument("--n", type=int, default=3, help="fleet size to summarise (default 3)")
    p.add_argument("--k", type=int, default=3, help="best-of-k the CSV was scored with (default 3)")
    p.add_argument("--out-dir", default=None, help="default: <bench>/results/ablation_summary")
    p.add_argument("--print", dest="show", nargs="*", default=["all", "glyphs"],
                   help="scopes to print to stdout (default: all glyphs); 'none' to print nothing")
    a = p.parse_args()

    a.bench = os.path.abspath(a.bench)
    out_dir = os.path.abspath(a.out_dir or os.path.join(a.bench, "results", "ablation_summary"))
    root = a.root if os.path.isdir(a.root) else os.path.join(a.bench, a.root)
    cells_csv = a.cells or os.path.join(out_dir, os.path.basename(os.path.normpath(root)) + f"_k{a.k}.csv")
    if not os.path.exists(cells_csv):
        sys.exit(f"no per-cell CSV at {cells_csv}: run scripts/ablation_metrics.py first")

    cells, targets, problems = load_cells(cells_csv, a.n)
    if not cells:
        sys.exit(f"no cells at N={a.n} in {cells_csv}")
    renders = load_renders(root, a.n, a.k)
    n_cells = sum(len(c) for c in cells.values())
    sizes = {v: len(c) for v, c in cells.items()}
    log(f"[conditions] {cells_csv}: {len(cells)} conditions, {n_cells} cells, "
        f"{len(targets)} target rows, {len(renders)} renders read from {root}")
    if problems:
        log(f"[!] {len(problems)} cell(s) dropped (missing mask or no iou):")
        for x in problems[:5]:
            log(f"      {x}")
    if len(set(sizes.values())) != 1:
        log(f"[!] conditions hold different numbers of targets: {sizes}")
    missing_r = [k for c in cells for k in ((c,) + key for key in cells[c]) if k not in renders]
    if missing_r:
        log(f"[!] {len(missing_r)} cell(s) without measured renders, e.g. {missing_r[:3]}")

    rows = aggregate(cells, targets, renders, a.k)
    meta = dict(root=os.path.relpath(root, a.bench).replace(os.sep, "/"), N=a.n, k=a.k, n_cells=n_cells,
                n_targets_all=max(sizes.values()),
                cells_csv=os.path.relpath(cells_csv, a.bench).replace(os.sep, "/"))
    os.makedirs(out_dir, exist_ok=True)
    write_wide(os.path.join(out_dir, "conditions.csv"), rows)
    write_tidy(os.path.join(out_dir, "conditions_tidy.csv"), rows)
    write_md(os.path.join(out_dir, "conditions.md"), rows, meta)
    write_tex(os.path.join(out_dir, "conditions.tex"), rows, meta)
    log(f"[conditions] -> {out_dir}/conditions.csv / conditions_tidy.csv / conditions.md / conditions.tex")

    show = [s for s in (a.show or []) if s != "none"]
    if show:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
        by = {}
        for r in rows:
            by.setdefault(r["scope"], []).append(r)
        for s in show:
            if s not in by:
                log(f"[!] no scope {s!r}")
                continue
            n = [r for r in by[s] if r["condition"] != "target"][0]["n_targets"]
            print(f"### {s} — {n} targets\n")
            print(md_table(by[s], small=(by[s][0]["scope_kind"] == "category")))
            print()


if __name__ == "__main__":
    main()
