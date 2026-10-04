#!/usr/bin/env python3
"""Decompose every ablation cell into coverage terms and turn them into per-step effects.

    python scripts/ablation_coverage_breakdown.py
    python scripts/ablation_coverage_breakdown.py --print glyphs shapes all

The question this answers is "what does each step of the build-up ladder
actually improve?", which plain IoU cannot answer on its own: IoU moves when the
shadow covers more of the target and when it stops spilling outside it, and the
two are different fixes. So every cell is re-measured into the two terms that
IoU is built from.

With target T and shadow S (both 1-bit at 128 px, dark = shape):

    missed      m = |T \\ S| / |T|      target area left uncovered
    extra       e = |S \\ T| / |T|      shadow outside the target, per TARGET area
    area ratio  a = |S| / |T|          = 1 - m + e
    iot           = |T & S| / |T|      = 1 - m
    IoU           = |T & S| / |T | S|  = (1 - m) / (1 + e)      [exact identity]

The identity is what makes the decomposition honest: the last column of
coverage_cells.csv recomputes IoU from m and e alone and must equal the measured
IoU to machine precision, and the flat -> full summary splits the total IoU gain
into the part explained by m and the part explained by e by holding the other
term at its flat value.

Inputs (read-only, none of them modified):
  results/ablation_summary/ablation-bench562_k3.csv  one row per (variant, subset,
      stem): the best-of-3 run already chosen for the tables the user has, its
      mask_path / target_path, and iou / nsd / hu_distance / boundary_iou
      (+ CLIP rr_cc on the glyph subsets). The chosen run is REUSED, never
      re-selected, so every number here lines up with the existing tables.
  results/ablation_summary/conditions.csv            per-condition aggregates,
      joined in so one row carries both the coverage terms and the shape metrics.
  results/ablation_cases/cases.csv                   cross-check only: its
      missed_frac / extra_frac (UMBRA condition) must reproduce.

Outputs:
  results/ablation_summary/coverage_cells.csv        6744 rows, one per cell
  results/ablation_summary/coverage_conditions.csv   11 scopes x 12 conditions
  results/ablation_summary/step_effects.md           the headline deliverable

Every delta is PAIRED within the scope: the mean over targets of (B - A), never a
difference of column means. s.e.m. is the sample sd of those per-target
differences over sqrt(n); for Hu, which is summarised by a median, the s.e.m. is
a 2000-resample bootstrap of the median of the paired differences (seed 0).
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys

import numpy as np
from PIL import Image

NEAREST = getattr(Image, "Resampling", Image).NEAREST
SIZE = 128
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CELLS_CSV = "results/ablation_summary/ablation-bench562_k3.csv"
COND_CSV = "results/ablation_summary/conditions.csv"
CASES_CSV = "results/ablation_cases/cases.csv"
OUT_CELLS = "results/ablation_summary/coverage_cells.csv"
OUT_COND = "results/ablation_summary/coverage_conditions.csv"
OUT_MD = "results/ablation_summary/step_effects.md"

GLYPH_CATS = ("letters_upper", "letters_lower", "digits")
SHAPE_CATS = ("abstract", "animals", "hand_shadow", "objects", "teleop")

ORDER = ["flat", "fwd_joint", "plus_assign", "plus_voronoi", "plus_backward",
         "plus_icp", "full", "full_no_assign", "full_no_voronoi",
         "full_no_backward", "full_no_icp", "full_no_fd"]

# (A, B, label) -- the delta is B - A, i.e. the effect of taking this step.
STEPS = [
    ("flat", "fwd_joint", "+1 arm-by-arm sweep + joint stage"),
    ("fwd_joint", "plus_assign", "+2 zone assignment"),
    ("plus_assign", "plus_voronoi", "+3 aimed start"),
    ("plus_voronoi", "plus_backward", "+4 backward sweep"),
    ("plus_backward", "plus_icp", "+5 ICP-guided restart"),
    ("plus_icp", "full", "+6 FD polish (= UMBRA)"),
    ("full", "full_no_assign", "UMBRA - 2 zone assignment (also drops 3)"),
    ("full", "full_no_voronoi", "UMBRA - 3 aimed start"),
    ("full", "full_no_backward", "UMBRA - 4 backward sweep"),
    ("full", "full_no_icp", "UMBRA - 5 ICP-guided restart"),
    ("full", "full_no_fd", "UMBRA - 6 FD polish"),
]
BUILD_STEPS = STEPS[:6]

BOOT_N, BOOT_SEED = 2000, 0


# ---------------------------------------------------------------- small helpers
def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_mask(path, size=SIZE):
    """run_base_optimizer.load_target convention: NEAREST to size, dark = shape."""
    im = Image.open(path).convert("L")
    if im.size != (size, size):
        im = im.resize((size, size), NEAREST)
    return np.array(im) < 128


def mean_sem(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = st.fmean(xs)
    if n < 2:
        return m, 0.0
    return m, st.stdev(xs) / (n ** 0.5)


def median_sem(xs):
    """Median of paired differences + bootstrap s.e.m. of that median."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    a = np.asarray(xs, dtype=np.float64)
    med = float(np.median(a))
    if n < 2:
        return med, 0.0
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.median(a[rng.integers(0, n, size=(BOOT_N, n))], axis=1)
    return med, float(boots.std(ddof=1))


def fmt(val, sem, dec):
    """value +- s.e.m., bolded when the effect clears 2 s.e.m."""
    s = f"{val:+.{dec}f} ± {sem:.{dec}f}"
    return f"**{s}**" if abs(val) > 2 * sem else s


def direction(val, sem, lower_is_better=True):
    """'better' / 'worse' / 'flat' at the 2 s.e.m. bar."""
    if abs(val) <= 2 * sem:
        return "flat"
    good = val < 0 if lower_is_better else val > 0
    return "better" if good else "worse"


# ------------------------------------------------------------ the verdict rule
VERDICT_RULE = (
    "`what it changes` is filled by an explicit rule, never by eye. A term counts as moved "
    "only when its paired mean clears 2 s.e.m.; `m` is missed, `e` is extra, and *reshape* "
    "means Δ boundary IoU or Δ Hu cleared 2 s.e.m. while both coverage terms stayed flat.\n\n"
    "| m \\ e | e better | e flat | e worse |\n"
    "|---|---|---|---|\n"
    "| **m better** | both: fills target + trims extra | fills target | trade: fills target, casts more extra |\n"
    "| **m flat** | trims extra shadow | reshapes only *(if reshape)* / no measurable change | casts more extra shadow |\n"
    "| **m worse** | trade: trims extra, uncovers target | leaves more target uncovered | worse on both |\n"
)


def verdict(dm, de, reshaped):
    table = {
        ("better", "better"): "both: fills target + trims extra",
        ("better", "flat"): "fills target",
        ("better", "worse"): "trade: fills target, casts more extra",
        ("flat", "better"): "trims extra shadow",
        ("flat", "worse"): "casts more extra shadow",
        ("worse", "better"): "trade: trims extra, uncovers target",
        ("worse", "flat"): "leaves more target uncovered",
        ("worse", "worse"): "worse on both",
    }
    if (dm, de) == ("flat", "flat"):
        return "reshapes only (boundary/Hu move, coverage flat)" if reshaped else "no measurable change"
    return table[(dm, de)]


# ------------------------------------------------------------------- step 1
def measure_cells(rows, verbose=True):
    """Re-measure every cell's chosen-run mask against its target."""
    targets, out = {}, []
    warn_iou, worst = [], 0.0
    for i, r in enumerate(rows):
        tp = r["target_path"]
        if tp not in targets:
            targets[tp] = load_mask(tp)
        T = targets[tp]
        S = load_mask(r["mask_path"])

        ta = int(T.sum())
        inter = int((T & S).sum())
        union = int((T | S).sum())
        sa = int(S.sum())
        if ta == 0 or union == 0:
            continue
        m = (ta - inter) / ta
        e = (sa - inter) / ta
        iou = inter / union
        iou_id = (1.0 - m) / (1.0 + e)

        rec = dict(
            variant=r["variant"], subset=r["subset"], stem=r["stem"], run=int(r["run"]),
            missed=m, extra=e, area_ratio=sa / ta, iot=inter / ta,
            iou=iou, iou_from_identity=iou_id, iou_recorded=num(r["iou"]),
            target_px=ta, shadow_px=sa,
            nsd=num(r["nsd"]), hu=num(r["hu_distance"]), biou=num(r["boundary_iou"]),
            clip_cc=num(r.get("clip_rr_cc")),
        )
        out.append(rec)

        d = abs(iou - rec["iou_recorded"])
        worst = max(worst, d)
        if d > 1e-4:
            warn_iou.append((r["variant"], r["subset"], r["stem"], iou, rec["iou_recorded"]))
        if verbose and (i + 1) % 1000 == 0:
            print(f"  ... {i + 1}/{len(rows)} cells", file=sys.stderr)

    if verbose:
        ident = max(abs(r["iou"] - r["iou_from_identity"]) for r in out)
        print(f"  sanity: max |IoU_measured - IoU_recorded| = {worst:.2e} "
              f"({len(warn_iou)} cells over 1e-4)", file=sys.stderr)
        print(f"  sanity: max |IoU - (1-m)/(1+e)|          = {ident:.2e}", file=sys.stderr)
        for w in warn_iou[:5]:
            print(f"    WARN iou drift {w[0]}/{w[1]}/{w[2]}: {w[3]:.6f} vs {w[4]:.6f}", file=sys.stderr)
    return out


def write_cells(recs, path):
    cols = ["variant", "subset", "stem", "run", "missed", "extra", "area_ratio",
            "iou", "iou_from_identity", "iot", "iou_recorded", "target_px", "shadow_px"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in sorted(recs, key=lambda x: (ORDER.index(x["variant"]), x["subset"], x["stem"])):
            w.writerow([r["variant"], r["subset"], r["stem"], r["run"]]
                       + [f"{r[c]:.6f}" for c in
                          ("missed", "extra", "area_ratio", "iou", "iou_from_identity", "iot")]
                       + [f"{r['iou_recorded']:.6f}", r["target_px"], r["shadow_px"]])


# ------------------------------------------------------------------- step 2
def scopes_of(subset):
    yield "all"
    yield "glyphs" if subset in GLYPH_CATS else "shapes"
    yield subset


SCOPE_ORDER = ["all", "glyphs", "shapes"] + list(GLYPH_CATS) + list(SHAPE_CATS)
SCOPE_KIND = {"all": "all", "glyphs": "group", "shapes": "group"}


def index_cells(recs):
    """(scope, condition) -> {stem_key: rec}, stem_key = (subset, stem)."""
    idx = {}
    for r in recs:
        for sc in scopes_of(r["subset"]):
            idx.setdefault((sc, r["variant"]), {})[(r["subset"], r["stem"])] = r
    return idx


def write_conditions(idx, cond_rows, path):
    joined = {(r["scope"], r["condition"]): r for r in cond_rows}
    join_cols = ["label", "block", "iou_mean", "iou_sem", "iou_median", "nsd_mean",
                 "nsd_sem", "hu_median", "hu_mean", "boundary_iou_mean",
                 "renders_per_solve", "clip_top1", "clip_mrr", "clip_mrr_cc",
                 "clip_p_true", "d_iou_vs_monolithic", "d_iou_vs_umbra"]
    cols = (["scope", "scope_kind", "condition", "label", "block", "n_targets",
             "missed_mean", "missed_median", "missed_sem",
             "extra_mean", "extra_median", "extra_sem",
             "area_ratio_mean", "area_ratio_median", "iot_mean",
             "iou_mean_recomputed"]
            + [c for c in join_cols if c not in ("label", "block")])

    mismatch = []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for sc in SCOPE_ORDER:
            for cond in ORDER:
                cells = idx.get((sc, cond))
                if not cells:
                    continue
                v = list(cells.values())
                mm, msem = mean_sem([x["missed"] for x in v])
                em, esem = mean_sem([x["extra"] for x in v])
                j = joined.get((sc, cond), {})
                iou_re = st.fmean([x["iou"] for x in v])
                if j.get("iou_mean") and abs(iou_re - float(j["iou_mean"])) > 5e-6:
                    mismatch.append((sc, cond, iou_re, float(j["iou_mean"])))
                row = [sc, SCOPE_KIND.get(sc, "category"), cond,
                       j.get("label", ""), j.get("block", ""), len(v),
                       f"{mm:.6f}", f"{st.median([x['missed'] for x in v]):.6f}", f"{msem:.6f}",
                       f"{em:.6f}", f"{st.median([x['extra'] for x in v]):.6f}", f"{esem:.6f}",
                       f"{st.fmean([x['area_ratio'] for x in v]):.6f}",
                       f"{st.median([x['area_ratio'] for x in v]):.6f}",
                       f"{st.fmean([x['iot'] for x in v]):.6f}",
                       f"{iou_re:.6f}"]
                row += [j.get(c, "") for c in join_cols if c not in ("label", "block")]
                w.writerow(row)
    return mismatch


# ------------------------------------------------------------------- step 3
def paired(idx, scope, a, b, field):
    """Per-target (B - A) for one field, over targets present in both conditions."""
    A, B = idx.get((scope, a), {}), idx.get((scope, b), {})
    keys = sorted(set(A) & set(B))
    out = []
    for k in keys:
        va, vb = A[k][field], B[k][field]
        if va is None or vb is None:
            continue
        out.append(vb - va)
    return out


def step_row(idx, scope, a, b, label, with_clip):
    A, B = idx.get((scope, a), {}), idx.get((scope, b), {})
    keys = sorted(set(A) & set(B))
    n = len(keys)
    # A step that leaves the solve bit-identical never fired at all; that is a very
    # different claim from "it fired and achieved nothing", and a table of means
    # blurs the two into the same flat row.
    unchanged = sum(1 for k in keys if abs(B[k]["iou"] - A[k]["iou"]) < 1e-9)
    pct_unchanged = (100.0 * unchanged / n) if n else 0.0
    d_iou, s_iou = mean_sem(paired(idx, scope, a, b, "iou"))
    d_m, s_m = mean_sem([100 * x for x in paired(idx, scope, a, b, "missed")])
    d_e, s_e = mean_sem([100 * x for x in paired(idx, scope, a, b, "extra")])
    d_a, s_a = mean_sem(paired(idx, scope, a, b, "area_ratio"))
    d_n, s_n = mean_sem(paired(idx, scope, a, b, "nsd"))
    # Hu: lower is better, so negate the paired difference to make + = better.
    d_h, s_h = median_sem([-x for x in paired(idx, scope, a, b, "hu")])
    d_b, s_b = mean_sem(paired(idx, scope, a, b, "biou"))

    cells = [fmt(d_iou, s_iou, 4), fmt(d_m, s_m, 3), fmt(d_e, s_e, 3),
             fmt(d_a, s_a, 4), fmt(d_n, s_n, 4), fmt(d_h, s_h, 3), fmt(d_b, s_b, 4)]
    if with_clip:
        d_c, s_c = mean_sem(paired(idx, scope, a, b, "clip_cc"))
        cells.append(fmt(d_c, s_c, 4))

    reshaped = abs(d_b) > 2 * s_b or abs(d_h) > 2 * s_h
    v = verdict(direction(d_m, s_m, True), direction(d_e, s_e, True), reshaped)
    return [label, str(n), f"{pct_unchanged:.0f}%"] + cells + [v], d_iou


def step_table(idx, scope):
    with_clip = scope in ("glyphs",) + GLYPH_CATS
    head = ["step", "n", "unchanged", "Δ IoU", "Δ missed (pp)", "Δ extra (pp)",
            "Δ area ratio", "Δ NSD", "Δ Hu (med)", "Δ bIoU"]
    if with_clip:
        head.append("Δ CLIP MRR_cc")
    head.append("what it changes")

    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    gains = {}
    for a, b, label in STEPS:
        row, d_iou = step_row(idx, scope, a, b, label, with_clip)
        gains[(a, b)] = d_iou
        if (a, b) == STEPS[6][:2]:  # first leave-one-out row: separate the blocks
            lines.append("|" + "|".join([" "] * len(head)) + "|")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines), gains


# ------------------------------------------------------------------- step 4
def summarise(idx, scope, gains):
    """Which step carries the flat -> full gain, and is it coverage or spill?"""
    A, B = idx[(scope, "flat")], idx[(scope, "full")]
    keys = sorted(set(A) & set(B))
    tot, tot_sem = mean_sem([B[k]["iou"] - A[k]["iou"] for k in keys])
    dm, dm_sem = mean_sem([100 * (B[k]["missed"] - A[k]["missed"]) for k in keys])
    de, de_sem = mean_sem([100 * (B[k]["extra"] - A[k]["extra"]) for k in keys])

    # Hold one term at its flat value and move the other to full: the identity
    # IoU = (1-m)/(1+e) turns the total into a coverage part and a spill part.
    cov, spill = [], []
    for k in keys:
        mf, ef = A[k]["missed"], A[k]["extra"]
        mF, eF = B[k]["missed"], B[k]["extra"]
        base = (1 - mf) / (1 + ef)
        cov.append((1 - mF) / (1 + ef) - base)
        spill.append((1 - mf) / (1 + eF) - base)
    cov_m, cov_s = mean_sem(cov)
    spill_m, spill_s = mean_sem(spill)

    build = [(lab, gains[(a, b)]) for a, b, lab in BUILD_STEPS]
    top_lab, top_val = max(build, key=lambda x: x[1])
    share = (top_val / tot * 100) if tot else float("nan")

    if abs(cov_m) > abs(spill_m) * 1.5:
        verdict_txt = "almost entirely from covering more of the target"
    elif abs(spill_m) > abs(cov_m) * 1.5:
        verdict_txt = "almost entirely from casting less extra shadow"
    else:
        verdict_txt = "from both terms in comparable measure"

    lines = [
        f"**{scope}** (n = {len(keys)}) — monolithic `flat` → `full` (UMBRA):",
        "",
        f"- total Δ IoU **{tot:+.4f} ± {tot_sem:.4f}**",
        f"- biggest single build-up step: **{top_lab}** at {top_val:+.4f} "
        f"({share:.0f}% of the total)",
        f"- total Δ missed **{dm:+.3f} ± {dm_sem:.3f} pp**, "
        f"total Δ extra **{de:+.3f} ± {de_sem:.3f} pp**",
        f"- decomposition of the IoU gain: coverage term {cov_m:+.4f} ± {cov_s:.4f}, "
        f"spill term {spill_m:+.4f} ± {spill_s:.4f} → {verdict_txt}",
    ]
    ladder = " ".join(f"`{lab.split(' ')[0]}` {g:+.4f}" for lab, g in build)
    lines += ["", f"- build-up ladder Δ IoU per step: {ladder}"]
    return "\n".join(lines), dict(total=tot, cov=cov_m, spill=spill_m,
                                  dm=dm, de=de, top=top_lab, top_val=top_val)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", dest="show", nargs="*",
                    default=["glyphs", "shapes", "all"],
                    help="scopes to print to stdout (default: glyphs shapes all)")
    args = ap.parse_args()
    os.chdir(ROOT)
    # The tables are full of "Δ"; a cp1252 console would die on the first row.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    rows = [r for r in csv.DictReader(open(CELLS_CSV, newline="", encoding="utf-8"))
            if r["variant"] != "target"]
    print(f"[1/4] measuring {len(rows)} cells ...", file=sys.stderr)
    recs = measure_cells(rows)
    write_cells(recs, OUT_CELLS)
    print(f"      wrote {OUT_CELLS} ({len(recs)} rows)", file=sys.stderr)

    idx = index_cells(recs)
    cond_rows = list(csv.DictReader(open(COND_CSV, newline="", encoding="utf-8")))
    print("[2/4] aggregating per scope x condition ...", file=sys.stderr)
    mismatch = write_conditions(idx, cond_rows, OUT_COND)
    print(f"      wrote {OUT_COND}", file=sys.stderr)
    if mismatch:
        print(f"      WARN {len(mismatch)} scope/condition IoU means differ from "
              f"conditions.csv:", file=sys.stderr)
        for m in mismatch[:5]:
            print(f"        {m[0]}/{m[1]}: {m[2]:.6f} vs {m[3]:.6f}", file=sys.stderr)
    else:
        print("      sanity: every scope x condition mean IoU matches conditions.csv "
              "to 5e-6", file=sys.stderr)

    # cross-check missed/extra against ablation_cases/cases.csv (UMBRA condition)
    try:
        cases = list(csv.DictReader(open(CASES_CSV, newline="", encoding="utf-8")))
        full = {(r["subset"], r["stem"]): r for r in recs if r["variant"] == "full"}
        bad = 0
        for c in cases:
            k = (c["subset"], c["stem"])
            if k in full:
                if (abs(full[k]["missed"] - float(c["missed_frac"])) > 1e-4
                        or abs(full[k]["extra"] - float(c["extra_frac"])) > 1e-4):
                    bad += 1
        print(f"      sanity: missed/extra vs cases.csv (UMBRA, {len(cases)} targets): "
              f"{bad} mismatches", file=sys.stderr)
    except OSError:
        print("      note: cases.csv not readable, skipped cross-check", file=sys.stderr)

    print("[3/4] building step effects ...", file=sys.stderr)
    md = [
        "# What each ablation step actually improves",
        "",
        "Generated by `scripts/ablation_coverage_breakdown.py` from "
        "`results/ablation_summary/ablation-bench562_k3.csv` (best-of-3 run per cell, "
        "reused as recorded) and `coverage_cells.csv`.",
        "",
        "## How to read these tables",
        "",
        "With target T and shadow S: **missed** `m = |T\\S|/|T|` is the fraction of the "
        "target left uncovered, **extra** `e = |S\\T|/|T|` is shadow cast outside the "
        "target as a fraction of target area, and **area ratio** `a = |S|/|T|`. These are "
        "not three independent numbers next to IoU — they *are* IoU, exactly:",
        "",
        "```",
        "IoU = (1 - m) / (1 + e)",
        "```",
        "",
        "so a step can raise IoU in exactly two ways: lower `m` (fill in more of the "
        "target) or lower `e` (stop spilling outside it). That is the question each row "
        "answers.",
        "",
        "Each row is a **step**, and the delta is always `B - A` for the step as written: "
        "the six build-up rows add a component, the five leave-one-out rows **remove** one "
        "from UMBRA. So in the build-up block a positive Δ IoU means the component helped; "
        "in the leave-one-out block a *negative* Δ IoU means the component was helping "
        "(matching `removed_delta_*` in `results/ablation_cases/cases.csv`).",
        "",
        "Deltas are **paired** per target and reported as mean ± s.e.m. of the per-target "
        "difference; **bold** marks an effect beyond 2 s.e.m. `Δ missed` and `Δ extra` are "
        "in percentage points and **negative is better** (less target missed, less spurious "
        "shadow). `Δ Hu` is the median of the paired differences, negated so **+ is better** "
        "(Hu distance fell), with a 2000-resample bootstrap s.e.m. `Δ NSD`, `Δ bIoU` and "
        "`Δ CLIP MRR_cc` are means where + is better. `Δ area ratio` has no preferred "
        "direction — it says whether the shadow grew or shrank.",
        "",
        "`unchanged` is the share of the scope's targets whose solve the step left "
        "bit-identical (|Δ IoU| < 1e-9). It separates *fired and achieved nothing* from "
        "*never fired*: a step that reads flat only because it rarely triggers is a "
        "statement about how often the bench hits that code path, not evidence that the "
        "component is useless.",
        "",
        VERDICT_RULE,
    ]

    printed, summaries, gains_by_scope = [], [], {}
    for scope in ("glyphs", "shapes", "all"):
        tbl, gains = step_table(idx, scope)
        gains_by_scope[scope] = gains
        n = len(idx[(scope, "full")])
        block = [f"## {scope} (n = {n})", "", tbl, ""]
        md += block
        printed.append((scope, "\n".join(block)))

    md += ["## Summary: where the monolithic → UMBRA gain comes from", ""]
    for scope in ("glyphs", "shapes", "all"):
        s, info = summarise(idx, scope, gains_by_scope[scope])
        md += [s, ""]
        summaries.append((scope, s, info))

    # appendix: Δ IoU per step for the 8 categories
    md += ["## Appendix — Δ IoU per step, by category", "",
           "Same paired Δ IoU, per category; bold beyond 2 s.e.m.", ""]
    cats = list(GLYPH_CATS) + list(SHAPE_CATS)
    md.append("| step | " + " | ".join(cats) + " |")
    md.append("|" + "|".join(["---"] * (len(cats) + 1)) + "|")
    for a, b, label in STEPS:
        cells = []
        for c in cats:
            d, s = mean_sem(paired(idx, c, a, b, "iou"))
            cells.append(fmt(d, s, 4))
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    md.append("")

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print(f"      wrote {OUT_MD}", file=sys.stderr)

    print("[4/4] done\n", file=sys.stderr)
    for scope, block in printed:
        if scope in args.show:
            print(block)
    print("## Summary: where the monolithic → UMBRA gain comes from\n")
    for scope, s, _ in summaries:
        if scope in args.show:
            print(s + "\n")


if __name__ == "__main__":
    main()
