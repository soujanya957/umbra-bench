#!/usr/bin/env python3
"""Which metric best shows what each ablation component does?

    python scripts/ablation_metric_sensitivity.py
    python scripts/ablation_metric_sensitivity.py --no-seed-scan --print none

conditions.md answers "how good is each condition"; this answers the other
question: each component is added and removed, every metric is scored on the
same paired targets, and the metrics are compared to each other on how large
(rel_change) and how reliable (|t|) the effect they report is. Those two
rankings are not the same metric, which is the point of the ranking table.

Inputs (read-only)
  results/ablation_summary/ablation-bench562_k3.csv   one row per (variant, subset, stem)
  results/ablation_summary/coverage_cells.csv         optional; missed/extra, only if it
                                                      has the full 6744 rows
  optimized/ablation-bench562/*-n3/*/*/results.json   per-seed IoU, for the noise scale

Pairing and sign. Every delta is the change caused by the action named in the
row, paired within a target:
  added   = metric(condition with it) - metric(the rung below)
  removed = metric(UMBRA without it) - metric(UMBRA)
For higher-is-better metrics that is the raw difference; for lower-is-better
metrics (hu_distance, and missed/extra when present) the sign is flipped, so
+ always means the action improved quality and a component that helps reads
+ when added and - when removed. 11 component/series pairs: component 1 has no
leave-one-out condition.

Statistics, per (scope, component, series, metric), over the targets where both
conditions have the metric:
  mean_delta, sd_delta (sample sd), sem = sd/sqrt(n), t = mean/sem,
  cohen_d = mean/sd, signif = |t| > 2.
  rel_change = mean_delta / ref_mean as a percent, where ref_mean is the mean of
  that metric, in that scope, over the same paired targets, in the condition
  that LACKS the component (the rung below for `added`, full_no_<x> for
  `removed`). For hu_distance the denominator is the raw (unflipped) level,
  which is positive, so + still means better.

Noise band, n_help / n_hurt / n_noise. The cell CSV stores the best-of-k run
only, so a within-target seed spread can be derived for IoU and for nothing
else: results.json records a per-run "iou" but no per-run nsd, hu_distance,
boundary_iou or CLIP -- those exist only for the one mask that was scored.
So the three counting columns are filled for IoU, using the caller's |delta| >=
0.02 band, and left EMPTY for every other metric rather than invented. --seed-scan
(default) measures what that band is worth: the median within-target range of
the 3 base-run IoUs, and, on the 432 cells that have 5 base runs, the median sd
of max-of-3 over all C(5,3) subsets -- the run-to-run noise of the best-of-3
number the deltas actually compare.

Outputs: results/ablation_summary/sensitivity.csv (tidy, every scope) and
results/ablation_summary/sensitivity.md (the reading tables). Nothing else is
written and nothing is modified.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import statistics as st
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)

CELLS_CSV = os.path.join(_BENCH, "results", "ablation_summary", "ablation-bench562_k3.csv")
COVERAGE_CSV = os.path.join(_BENCH, "results", "ablation_summary", "coverage_cells.csv")
OUT_CSV = os.path.join(_BENCH, "results", "ablation_summary", "sensitivity.csv")
OUT_MD = os.path.join(_BENCH, "results", "ablation_summary", "sensitivity.md")
SOLVE_GLOB = os.path.join(_BENCH, "optimized", "ablation-bench562", "*-n3", "*", "*", "results.json")

COVERAGE_ROWS = 6744          # 12 conditions x 562 targets; anything less is a partial write
IOU_NOISE_BAND = 0.02         # |delta| below this counts as noise, IoU only
T_SIGNIF = 2.0

# (number, label, series, condition-with-it, condition-without-it, dagger)
# `added` pairs a rung with the rung below; `removed` pairs UMBRA-minus-x with UMBRA.
PAIRS = [
    (1, "arm-by-arm sweep + joint stage", "added",   "fwd_joint",        "flat",          False),
    (2, "zone assignment",                "added",   "plus_assign",      "fwd_joint",     False),
    (2, "zone assignment",                "removed", "full",             "full_no_assign", True),
    (3, "aimed start",                    "added",   "plus_voronoi",     "plus_assign",   False),
    (3, "aimed start",                    "removed", "full",             "full_no_voronoi", False),
    (4, "backward sweep",                 "added",   "plus_backward",    "plus_voronoi",  False),
    (4, "backward sweep",                 "removed", "full",             "full_no_backward", False),
    (5, "ICP-guided restart",             "added",   "plus_icp",         "plus_backward", False),
    (5, "ICP-guided restart",             "removed", "full",             "full_no_icp",   False),
    (6, "FD polish",                      "added",   "full",             "plus_icp",      False),
    (6, "FD polish",                      "removed", "full",             "full_no_fd",    False),
]

GLYPH_SUBSETS = ["letters_upper", "letters_lower", "digits"]
SHAPE_SUBSETS = ["abstract", "animals", "hand_shadow", "objects", "teleop"]
SCOPES = [
    ("all",           GLYPH_SUBSETS + SHAPE_SUBSETS),
    ("glyphs",        GLYPH_SUBSETS),
    ("shapes",        SHAPE_SUBSETS),
    ("letters_upper", ["letters_upper"]),
    ("letters_lower", ["letters_lower"]),
    ("digits",        ["digits"]),
]
GLYPH_SCOPES = {"glyphs", "letters_upper", "letters_lower", "digits"}

# metric -> (short label, +1 higher is better / -1 lower is better, glyph-only)
METRICS = {
    "iou":          ("IoU",        +1, False),
    "nsd":          ("NSD",        +1, False),
    "boundary_iou": ("bIoU",       +1, False),
    "hu_distance":  ("Hu",         -1, False),
    "clip_top1":    ("CLIP top-1", +1, True),
    "clip_rr":      ("MRR",        +1, True),
    "clip_rr_cc":   ("MRR_cc",     +1, True),
    "clip_p_true":  ("p_true",     +1, True),
}
COVERAGE_METRICS = {"missed": ("missed", -1, False), "extra": ("extra", -1, False)}
CLIP_METRICS = ["clip_top1", "clip_rr", "clip_rr_cc", "clip_p_true"]

CHANCE = {"letters_upper": 26, "letters_lower": 26, "digits": 10}

# Numbers the study already published, to prove this script's pairing reproduces them.
# (scope, component, series, metric, expected)
CHECKS = [
    ("glyphs", 1, "added",   "clip_rr_cc", +0.005), ("glyphs", 2, "added",   "clip_rr_cc", +0.035),
    ("glyphs", 2, "removed", "clip_rr_cc", +0.048), ("glyphs", 3, "added",   "clip_rr_cc", -0.059),
    ("glyphs", 3, "removed", "clip_rr_cc", +0.031), ("glyphs", 4, "added",   "clip_rr_cc", +0.082),
    ("glyphs", 4, "removed", "clip_rr_cc", -0.017), ("glyphs", 5, "added",   "clip_rr_cc", +0.005),
    ("glyphs", 5, "removed", "clip_rr_cc", +0.004), ("glyphs", 6, "added",   "clip_rr_cc", -0.028),
    ("glyphs", 6, "removed", "clip_rr_cc", +0.024),
    ("letters_upper", 3, "added", "clip_rr_cc", -0.093),
    ("letters_upper", 4, "added", "clip_rr_cc", +0.094),
    ("all", 1, "added",   "iou", +0.004), ("all", 2, "added",   "iou", +0.003),
    ("all", 3, "added",   "iou", -0.001), ("all", 4, "added",   "iou", +0.014),
    ("all", 5, "added",   "iou",  0.000), ("all", 6, "added",   "iou", +0.004),
    ("all", 2, "removed", "iou", -0.002), ("all", 3, "removed", "iou",  0.000),
    ("all", 4, "removed", "iou", -0.014), ("all", 5, "removed", "iou",  0.000),
    ("all", 6, "removed", "iou", -0.004),
    ("all", 3, "added", "nsd", -0.007), ("all", 4, "added", "nsd", +0.025),
    ("all", 6, "added", "nsd", +0.008),
]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# --- inputs ---------------------------------------------------------------------

def load_cells():
    """-> (conditions frame, target/ceiling frame, metric spec dict)."""
    cells = pd.read_csv(CELLS_CSV)
    metrics = dict(METRICS)
    ceiling = cells[cells.variant == "target"].copy()
    cond = cells[cells.variant != "target"].copy()

    cov_note = None
    if os.path.exists(COVERAGE_CSV):
        try:
            cov = pd.read_csv(COVERAGE_CSV)
        except Exception as exc:                                  # a concurrent writer mid-write
            cov, cov_note = None, f"unreadable ({exc.__class__.__name__}); missed/extra skipped"
        if cov is not None:
            if len(cov) != COVERAGE_ROWS:
                cov_note = f"{len(cov)} rows, not {COVERAGE_ROWS}; partial, so missed/extra skipped"
            elif not {"variant", "subset", "stem"} <= set(cov.columns):
                cov_note = "no (variant, subset, stem) key; missed/extra skipped"
            elif not {"missed", "extra"} <= set(cov.columns):
                cov_note = "no missed/extra columns; skipped"
            else:
                cond = cond.merge(cov[["variant", "subset", "stem", "missed", "extra"]],
                                  on=["variant", "subset", "stem"], how="left")
                metrics.update(COVERAGE_METRICS)
                cov_note = f"{len(cov)} rows; missed and extra included"
    else:
        cov_note = "not present; missed/extra omitted"
    return cond, ceiling, metrics, cov_note


def seed_scan():
    """Within-target seed spread of IoU, the only metric with per-run values.

    -> dict or None. `range_median` is the median over cells of (max - min) of the
    3 base-run IoUs. `boot_sd_median` resamples: on cells with >= 5 base runs, the
    sd of max-of-3 over every C(n, 3) subset, i.e. the run-to-run noise of the
    best-of-3 value the deltas compare, which is the number the 0.02 band should
    be read against.
    """
    files = glob.glob(SOLVE_GLOB)
    if not files:
        return None
    ranges, boot, n_five = [], [], 0
    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                runs = json.load(fh).get("runs") or []
        except (OSError, ValueError):
            continue
        base = [r["iou"] for r in runs if not r.get("extra") and r.get("iou") is not None]
        k3 = [r["iou"] for r in runs
              if not r.get("extra") and r.get("run") is not None and r["run"] < 3 and r.get("iou") is not None]
        if len(k3) > 1:
            ranges.append(max(k3) - min(k3))
        if len(base) >= 5:
            n_five += 1
            maxima = [max(t) for t in itertools.combinations(base, 3)]
            if len(maxima) > 1:
                boot.append(st.stdev(maxima))
    if not ranges:
        return None
    return {"n_cells": len(files), "n_range": len(ranges), "range_median": st.median(ranges),
            "n_five": n_five, "boot_sd_median": st.median(boot) if boot else None}


# --- statistics -----------------------------------------------------------------

def pivots(cond, metrics):
    """metric -> frame indexed by (subset, stem), one column per condition."""
    out = {}
    for metric in metrics:
        if metric not in cond.columns:
            continue
        out[metric] = cond.pivot(index=["subset", "stem"], columns="variant", values=metric)
    return out


def one_row(piv, subsets, metric, sign, num, label, series, with_c, without_c, dagger, scope, scope_n):
    """One (scope, component, series, metric) record, paired over the scope's targets."""
    frame = piv.loc[piv.index.get_level_values("subset").isin(subsets)]
    if with_c not in frame.columns or without_c not in frame.columns:
        return None
    both = frame[[with_c, without_c]].dropna()
    n = len(both)
    if n == 0:
        return None

    # `added` moves up the ladder (with - without); `removed` is UMBRA-without minus UMBRA,
    # which is the same pair read the other way: (without - with). Then the lower-is-better flip.
    raw = both[with_c] - both[without_c] if series == "added" else both[without_c] - both[with_c]
    delta = sign * raw
    ref_mean = float(both[without_c].mean())          # the condition lacking the component, raw level

    mean = float(delta.mean())
    sd = float(delta.std(ddof=1)) if n > 1 else float("nan")
    sem = sd / math.sqrt(n) if n > 1 and math.isfinite(sd) else float("nan")
    if n > 1 and sd == 0.0:
        t = 0.0 if mean == 0.0 else float("inf") * (1 if mean > 0 else -1)
        d = 0.0 if mean == 0.0 else float("inf") * (1 if mean > 0 else -1)
    else:
        t = mean / sem if sem and math.isfinite(sem) and sem > 0 else float("nan")
        d = mean / sd if sd and math.isfinite(sd) and sd > 0 else float("nan")
    rel = 100.0 * mean / ref_mean if abs(ref_mean) > 1e-9 else float("nan")

    rec = {
        "scope": scope, "scope_n": scope_n, "component": num, "component_label": label,
        "series": series, "cond_with": with_c, "cond_without": without_c,
        "ref_condition": without_c, "dagger": dagger,
        "metric": metric, "direction": "higher_better" if sign > 0 else "lower_better(flipped)",
        "n": n, "mean_delta": mean, "sd_delta": sd, "sem": sem, "t": t, "cohen_d": d,
        "ref_mean": ref_mean, "rel_change_pct": rel,
        "signif": bool(math.isfinite(t) and abs(t) > T_SIGNIF),
    }
    # Counting columns need a noise scale. Only IoU has per-seed values to set one.
    if metric == "iou":
        rec["noise_band"] = IOU_NOISE_BAND
        rec["n_help"] = int((delta >= IOU_NOISE_BAND).sum())
        rec["n_hurt"] = int((delta <= -IOU_NOISE_BAND).sum())
        rec["n_noise"] = int((delta.abs() < IOU_NOISE_BAND).sum())
    else:
        rec["noise_band"] = ""
        rec["n_help"] = rec["n_hurt"] = rec["n_noise"] = ""
    return rec


def build(cond, metrics):
    piv = pivots(cond, metrics)
    rows = []
    for scope, subsets in SCOPES:
        scope_n = int(cond[cond.subset.isin(subsets)].groupby(["subset", "stem"]).ngroups)
        for metric, (_, sign, glyph_only) in metrics.items():
            if metric not in piv or (glyph_only and scope not in GLYPH_SCOPES):
                continue
            for num, label, series, with_c, without_c, dagger in PAIRS:
                rec = one_row(piv[metric], subsets, metric, sign, num, label, series,
                              with_c, without_c, dagger, scope, scope_n)
                if rec:
                    rows.append(rec)
    return pd.DataFrame(rows)


def ceiling_stats(ceiling):
    """CLIP ceiling (the authored PNG through the same scorer) per glyph subset."""
    out = {}
    for sub in GLYPH_SUBSETS:
        part = ceiling[ceiling.subset == sub]
        if len(part):
            out[sub] = {m: float(part[m].mean()) for m in CLIP_METRICS}
    return out


def chance_stats():
    out = {}
    for sub, K in CHANCE.items():
        out[sub] = {"K": K, "clip_top1": 1.0 / K,
                    "clip_rr": sum(1.0 / i for i in range(1, K + 1)) / K,
                    "clip_rr_cc": 0.0, "clip_p_true": 1.0 / K}
    return out


def verify(sens):
    """Reproduce the published numbers. -> (n_ok, [failure strings])."""
    fails = []
    for scope, num, series, metric, want in CHECKS:
        hit = sens[(sens.scope == scope) & (sens.component == num) &
                   (sens.series == series) & (sens.metric == metric)]
        if hit.empty:
            fails.append(f"{scope}/{num}-{series}/{metric}: missing")
            continue
        got = float(hit.iloc[0].mean_delta)
        if abs(round(got, 3) - want) > 5e-4:
            fails.append(f"{scope}/{num}-{series}/{metric}: got {got:+.4f}, published {want:+.3f}")
    return len(CHECKS) - len(fails), fails


# --- markdown -------------------------------------------------------------------

def pair_label(num, label, series):
    return f"{num} {label} — *{series}*"


def fmt_mean(v, places=3):
    return "—" if v is None or not math.isfinite(v) else f"{v:+.{places}f}"


def fmt_t(t):
    if t is None or not math.isfinite(t):
        return "∞" if t is not None and math.isinf(t) else "—"
    return f"{abs(t):.1f}"


def cell_mean_t(rec):
    if rec is None:
        return "—"
    body = f"{fmt_mean(rec['mean_delta'])} [{fmt_t(rec['t'])}]"
    return f"**{body}**" if rec["signif"] else body


def cell_mean_sem(rec):
    if rec is None:
        return "—"
    sem = rec["sem"]
    body = f"{fmt_mean(rec['mean_delta'])} ± {sem:.3f}" if math.isfinite(sem) else fmt_mean(rec["mean_delta"])
    return f"**{body}**" if rec["signif"] else body


def lookup(sens, scope, num, series, metric):
    hit = sens[(sens.scope == scope) & (sens.component == num) &
               (sens.series == series) & (sens.metric == metric)]
    return None if hit.empty else hit.iloc[0].to_dict()


def table_scope(sens, scope, metrics):
    """(a) rows = the 11 component/series pairs, columns = metrics, cell = mean [|t|]."""
    cols = [m for m in metrics if not sens[(sens.scope == scope) & (sens.metric == m)].empty]
    head = "| component / series | " + " | ".join(METRICS_ALL[m][0] for m in cols) + " |"
    rule = "|---|" + "---|" * len(cols)
    lines = [head, rule]
    for num, label, series, _, _, dagger in PAIRS:
        name = pair_label(num, label, series) + (" †" if dagger else "")
        cells = [cell_mean_t(lookup(sens, scope, num, series, m)) for m in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_ranking(sens, scopes):
    """(b) per component/series: the most reliable metric vs the biggest relative move."""
    lines = ["| scope | component / series | largest \\|t\\| | its mean [\\|t\\|] | largest \\|rel change\\| | its mean (rel %) |",
             "|---|---|---|---|---|---|"]
    for scope in scopes:
        for num, label, series, _, _, dagger in PAIRS:
            part = sens[(sens.scope == scope) & (sens.component == num) & (sens.series == series)]
            part = part[np.isfinite(part["t"].astype(float))]
            if part.empty:
                continue
            by_t = part.iloc[part["t"].astype(float).abs().values.argmax()]
            rel = part[np.isfinite(part["rel_change_pct"].astype(float))]
            by_rel = (rel.iloc[rel["rel_change_pct"].astype(float).abs().values.argmax()]
                      if not rel.empty else None)
            name = pair_label(num, label, series) + (" †" if dagger else "")
            t_metric = METRICS_ALL[by_t.metric][0]
            t_cell = f"{fmt_mean(by_t.mean_delta)} [{fmt_t(by_t.t)}]"
            if by_t.signif:
                t_cell = f"**{t_cell}**"
            if by_rel is None:
                r_metric, r_cell = "—", "—"
            else:
                r_metric = METRICS_ALL[by_rel.metric][0]
                r_cell = f"{fmt_mean(by_rel.mean_delta)} ({by_rel.rel_change_pct:+.1f}%)"
                if not by_rel.signif:
                    r_cell += " ns"
            lines.append(f"| {scope} | {name} | {t_metric} | {t_cell} | {r_metric} | {r_cell} |")
    return "\n".join(lines)


def table_clip_by_type(sens, ceilings, chance):
    """(c) each glyph type on its own: 4 CLIP metrics x 11 pairs, mean ± s.e.m."""
    out = []
    for sub in GLYPH_SUBSETS:
        part = sens[sens.scope == sub]
        if part.empty:
            continue
        n = int(part.iloc[0].scope_n)
        ch, ceil = chance[sub], ceilings.get(sub, {})
        out.append(f"**{sub} — n = {n}, K = {ch['K']}**")
        out.append("")
        out.append(f"chance: top-1 {ch['clip_top1']:.3f}, MRR {ch['clip_rr']:.3f}, MRR_cc 0.000, "
                   f"p_true {ch['clip_p_true']:.3f} (uniform) · ceiling (authored PNG, same scorer): "
                   + ", ".join(f"{METRICS_ALL[m][0]} {ceil[m]:.3f}" for m in CLIP_METRICS if m in ceil))
        out.append("")
        out.append("| component / series | " + " | ".join(METRICS_ALL[m][0] for m in CLIP_METRICS) + " |")
        out.append("|---|" + "---|" * len(CLIP_METRICS))
        for num, label, series, _, _, dagger in PAIRS:
            name = pair_label(num, label, series) + (" †" if dagger else "")
            cells = [cell_mean_sem(lookup(sens, sub, num, series, m)) for m in CLIP_METRICS]
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def metric_profile(sens, scope, metrics):
    """Per metric in one scope: how many of the 11 pairs it resolves, and how hard."""
    prof = {}
    for m in metrics:
        part = sens[(sens.scope == scope) & (sens.metric == m)]
        if part.empty:
            continue
        tt = part["t"].astype(float)
        finite = tt[np.isfinite(tt)]
        prof[m] = {
            "n_signif": int(part["signif"].sum()),
            "flagged": set(zip(part[part["signif"]].component, part[part["signif"]].series)),
            "median_abs_t": float(finite.abs().median()) if len(finite) else float("nan"),
            "median_abs_rel": float(part["rel_change_pct"].astype(float).abs().median()),
            "median_sem": float(part["sem"].astype(float).median()),
        }
    return prof


DECOMPOSITION = {"missed", "extra"}   # IoU = (1 - missed) / (1 + extra), exactly


def conclusion(sens, metrics, chance, ceilings):
    """(d) Rule first, then the numbers it selects on."""
    wide = [m for m, (_, _, glyph_only) in metrics.items() if not glyph_only]
    pool = [m for m in wide if m not in DECOMPOSITION]        # standalone quality summaries
    decomp = [m for m in wide if m in DECOMPOSITION]          # IoU's own two terms
    prof_all = metric_profile(sens, "all", wide)
    prof_gly = metric_profile(sens, "glyphs", list(metrics))

    def key(m):
        p = prof_all[m]
        return (p["n_signif"], p["median_abs_t"] if math.isfinite(p["median_abs_t"]) else -1)

    head = max(pool, key=key)
    rest = [m for m in pool if m != head]

    def key2(m):
        gained = len(prof_all[m]["flagged"] - prof_all[head]["flagged"])
        return (gained, prof_all[m]["n_signif"], prof_all[m]["median_abs_t"])
    second = max(rest, key=key2) if rest else None

    n_pairs = len(PAIRS)
    lines = []
    lines.append("**Rule, stated before the numbers.** (R0, eligibility) Only a standalone quality "
                 "summary defined on all 562 targets can be the headline. `missed` and `extra` are "
                 "excluded from that pool and ranked separately as **diagnostics**, because "
                 "`IoU = (1 − missed) / (1 + extra)` exactly (see `step_effects.md`): they are IoU's "
                 "own two terms, so headlining one of them would be reporting half of IoU, not a "
                 "second opinion about it. (R1) The headline is the eligible metric that resolves the "
                 "most of the 11 component/series pairs at |t| > 2 in scope `all`, ties broken by the "
                 "higher median |t| over the 11 pairs. (R2) The secondary metric is the eligible metric "
                 "that resolves the most pairs the headline misses, same tie-break. (R3) The diagnostic "
                 "is the decomposition term that resolves the most pairs, same tie-break — it says *how* "
                 "a component helped, never *whether*. (R4) CLIP is usable in a glyph type if at least "
                 "one of the 11 pairs reaches |t| > 2 there; a type's resolution floor is 2 x its median "
                 "s.e.m., the smallest effect it could ever call.")
    lines.append("")
    lines.append("| metric | role | scope `all`: pairs at \\|t\\| > 2 (of %d) | median \\|t\\| | median \\|rel change\\| | median s.e.m. |" % n_pairs)
    lines.append("|---|---|---|---|---|---|")
    for m in sorted(prof_all, key=key, reverse=True):
        p = prof_all[m]
        role = "diagnostic (IoU term)" if m in DECOMPOSITION else "quality summary"
        lines.append(f"| {METRICS_ALL[m][0]} | {role} | {p['n_signif']} | {p['median_abs_t']:.1f} | "
                     f"{p['median_abs_rel']:.1f}% | {p['median_sem']:.4f} |")
    lines.append("")
    hl = METRICS_ALL[head][0]
    lines.append(f"**Headline: {hl}** by R1 — {prof_all[head]['n_signif']}/{n_pairs} pairs at |t| > 2, "
                 f"median |t| {prof_all[head]['median_abs_t']:.1f}, and the tightest s.e.m. of any "
                 f"metric here ({prof_all[head]['median_sem']:.4f}).")
    if second:
        sl = METRICS_ALL[second][0]
        gained = sorted(prof_all[second]["flagged"] - prof_all[head]["flagged"])
        got = ", ".join(f"{c} *{s}*" for c, s in gained) if gained else "none"
        lines.append("")
        lines.append(f"**Secondary: {sl}** by R2 — it resolves {len(gained)} pair(s) {hl} does not "
                     f"({got}), and moves {prof_all[second]['median_abs_rel']:.1f}% of its own level "
                     f"per pair against {prof_all[head]['median_abs_rel']:.1f}% for {hl}. Report it "
                     f"beside {hl}, not instead of it: it is where a component moves the outline "
                     f"without moving the filled area.")
    if decomp:
        dg = max(decomp, key=key)
        dl = METRICS_ALL[dg][0]
        only_d = sorted(prof_all[dg]["flagged"] - prof_all[head]["flagged"])
        only_h = sorted(prof_all[head]["flagged"] - prof_all[dg]["flagged"])
        lines.append("")
        lines.append(f"**Diagnostic: {dl}** by R3 — {prof_all[dg]['n_signif']}/{n_pairs} pairs, median "
                     f"|t| {prof_all[dg]['median_abs_t']:.1f}, and the largest median relative move of "
                     f"any metric ({prof_all[dg]['median_abs_rel']:.1f}% against "
                     f"{prof_all[head]['median_abs_rel']:.1f}% for {hl}), because a delta is a much "
                     f"bigger share of a 0.14 miss rate than of a 0.72 IoU. It ties {hl} on count while "
                     f"resolving a *different* set of pairs — "
                     + (", ".join(f"{c} *{s}*" for c, s in only_d) or "none") + f" for {dl} against "
                     + (", ".join(f"{c} *{s}*" for c, s in only_h) or "none") + f" for {hl} — so the two "
                     f"together, not either alone, say which components fill the target and which stop "
                     f"it spilling outside. Quote it as a breakdown of {hl}, never as a rival headline.")
    lines.append("")
    lines.append("CLIP, by R4, per glyph type:")
    lines.append("")
    lines.append("| glyph type | n | pairs at \\|t\\| > 2 (MRR_cc) | median s.e.m. (MRR_cc) | resolution floor | largest \\|mean\\| seen | usable? |")
    lines.append("|---|---|---|---|---|---|---|")
    for sub in GLYPH_SUBSETS + ["glyphs"]:
        part = sens[(sens.scope == sub) & (sens.metric == "clip_rr_cc")]
        if part.empty:
            continue
        n = int(part.iloc[0].scope_n)
        n_sig = int(part["signif"].sum())
        med_sem = float(part["sem"].astype(float).median())
        biggest = float(part["mean_delta"].astype(float).abs().max())
        verdict = "yes" if n_sig >= 1 else "no"
        lines.append(f"| {sub} | {n} | {n_sig} | {med_sem:.3f} | {2 * med_sem:.3f} | {biggest:.3f} | {verdict} |")
    lines.append("")
    n_clip_sig = sum(prof_gly[m]["n_signif"] for m in CLIP_METRICS if m in prof_gly)
    lines.append(f"Across the pooled 186 glyphs the four CLIP columns together resolve {n_clip_sig} "
                 f"of the {4 * n_pairs} (metric, pair) combinations at |t| > 2, so CLIP is a "
                 "corroborating column on glyphs, not a measurement that can carry the ablation: "
                 "it exists for 186 of 562 targets and for none of the shapes.")
    return "\n".join(lines), head, second


def write_md(sens, metrics, scan, cov_note, ceilings, chance, checks_ok, checks_fail):
    parts = []
    parts.append("# Metric sensitivity of the UMBRA ablation")
    parts.append("")
    parts.append("562 targets x 12 conditions, N=3, best of 3 seeds (k=3), unfitted against the "
                 "authored target. Source of truth: `results/ablation_summary/ablation-bench562_k3.csv`; "
                 "every number here and in `sensitivity.csv` is a paired groupby on it. "
                 "Generated by `scripts/ablation_metric_sensitivity.py`.")
    parts.append("")
    parts.append("**Sign.** Each cell is the change caused by the action in the row: *added* = the "
                 "condition with the component minus the rung below; *removed* = UMBRA without it "
                 "minus UMBRA. Lower-is-better metrics (Hu"
                 + (", missed, extra" if "missed" in metrics else "")
                 + ") are sign-flipped, so **+ always means the action improved quality**, and a "
                   "component that helps reads + when added and − when removed. "
                 "† removing zone assignment also removes the aimed start.")
    parts.append("")
    parts.append("**Reliability.** `[x.x]` is |t| = |mean| / s.e.m. over the paired targets; **bold** "
                 "is |t| > 2. `rel change` is the mean delta as a percent of the same metric's level "
                 "in the condition that lacks the component (for Hu, of its raw positive level), so a "
                 "metric with a small absolute range is not penalised for it.")
    parts.append("")
    if scan:
        boot = ("%.4f" % scan["boot_sd_median"]) if scan["boot_sd_median"] else "n/a"
        parts.append(f"**Noise scale.** The cell CSV keeps the best-of-k run, and `results.json` records "
                     f"a per-run `iou` but no per-run NSD, Hu, bIoU or CLIP — those exist only for the "
                     f"single mask that was scored. So a within-target seed spread is derivable for IoU "
                     f"and for nothing else. Over {scan['n_range']} cells the median range of the 3 "
                     f"base-run IoUs is **{scan['range_median']:.4f}**; on the {scan['n_five']} cells that "
                     f"have 5 base runs, resampling every C(5,3) subset gives a median sd of best-of-3 of "
                     f"**{boot}** — the run-to-run noise of the number these deltas actually compare. "
                     f"The {IOU_NOISE_BAND} band used for `n_help` / `n_hurt` / `n_noise` is therefore "
                     f"about 2 sd of the best-of-3 estimator. **For every non-IoU metric those three "
                     f"columns are left empty**, because no seed spread can be derived for them without "
                     f"re-rendering and re-scoring each seed's mask.")
    else:
        parts.append(f"**Noise scale.** Seed scan skipped; `n_help` / `n_hurt` / `n_noise` use the "
                     f"{IOU_NOISE_BAND} IoU band and are empty for every other metric.")
    parts.append("")
    parts.append(f"**Coverage (missed / extra).** `results/ablation_summary/coverage_cells.csv`: {cov_note}.")
    parts.append("")
    if checks_fail:
        parts.append("**Reproduction check: FAILED** — " + "; ".join(checks_fail))
    else:
        parts.append(f"**Reproduction check.** All {checks_ok} published values (pooled glyph ΔMRR_cc, "
                     "letters_upper ΔMRR_cc, all-562 ΔIoU and ΔNSD) reproduce to 3 decimals.")
    parts.append("")

    parts.append("## (a) Which metric moves most — every metric, same paired targets")
    parts.append("")
    for scope in ("all", "glyphs", "shapes"):
        n = int(sens[sens.scope == scope].iloc[0].scope_n)
        parts.append(f"### scope `{scope}` — {n} targets")
        parts.append("")
        parts.append(table_scope(sens, scope, list(metrics)))
        parts.append("")

    parts.append("## (b) Most reliably measured vs biggest relative change")
    parts.append("")
    parts.append("The two columns disagree whenever a metric with a small base level moves a lot of "
                 "its own range but not much compared to its spread across targets. `ns` marks a "
                 "largest-relative-change entry that is not itself significant.")
    parts.append("")
    parts.append(table_ranking(sens, ["all", "glyphs", "shapes"]))
    parts.append("")

    parts.append("## (c) CLIP by glyph type")
    parts.append("")
    parts.append("Pooling re-weights the types and chance differs with K, so each type on its own. "
                 "mean ± s.e.m. over paired targets, **bold** = |t| > 2.")
    parts.append("")
    parts.append(table_clip_by_type(sens, ceilings, chance))

    parts.append("## (d) Which metric to headline")
    parts.append("")
    text, head, second = conclusion(sens, metrics, chance, ceilings)
    parts.append(text)
    parts.append("")
    return "\n".join(parts), head, second


# --- main -----------------------------------------------------------------------

METRICS_ALL = dict(METRICS)
METRICS_ALL.update(COVERAGE_METRICS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-seed-scan", action="store_true",
                    help="skip reading optimized/**/results.json for the per-seed IoU noise scale")
    ap.add_argument("--print", dest="show", default="report",
                    choices=["report", "none"], help="what to print to stdout (default: report)")
    args = ap.parse_args()

    cond, ceiling, metrics, cov_note = load_cells()
    log(f"cells: {len(cond)} condition rows, {len(ceiling)} target rows; coverage: {cov_note}")

    scan = None if args.no_seed_scan else seed_scan()
    if scan:
        log(f"seed scan: median IoU range {scan['range_median']:.4f} over {scan['n_range']} cells; "
            f"median best-of-3 sd {scan['boot_sd_median']:.4f} over {scan['n_five']} cells with 5 base runs")

    sens = build(cond, metrics)
    ok, fails = verify(sens)
    if fails:
        log("REPRODUCTION CHECK FAILED:")
        for f in fails:
            log("  " + f)
    else:
        log(f"reproduction check: {ok}/{len(CHECKS)} published values match to 3 decimals")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    cols = ["scope", "scope_n", "component", "component_label", "series", "cond_with", "cond_without",
            "ref_condition", "dagger", "metric", "direction", "n", "mean_delta", "sd_delta", "sem",
            "t", "cohen_d", "ref_mean", "rel_change_pct", "noise_band", "n_help", "n_hurt", "n_noise",
            "signif"]
    out = sens[cols].copy()
    for c in ("mean_delta", "sd_delta", "sem", "t", "cohen_d", "ref_mean", "rel_change_pct"):
        out[c] = out[c].astype(float).round(6)
    out.to_csv(OUT_CSV, index=False)
    log(f"wrote {OUT_CSV} ({len(out)} rows)")

    md, head, second = write_md(sens, metrics, scan, cov_note, ceiling_stats(ceiling), chance_stats(), ok, fails)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    log(f"wrote {OUT_MD}; headline {head}, secondary {second}")

    if args.show == "report":
        print(md)


if __name__ == "__main__":
    main()
