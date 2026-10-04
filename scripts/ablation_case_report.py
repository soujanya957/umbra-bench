#!/usr/bin/env python3
"""Per-target case report for the bench562 component ablation: figures, CSV, HTML index, failures digest.

    python scripts/ablation_case_report.py                     # draw missing or stale figures (--only-missing)
    python scripts/ablation_case_report.py --force             # redraw every figure
    python scripts/ablation_case_report.py --check-csv results/ablation_metrics/ablation-bench562_k3.csv

The ablation table says how far a component moves mean IoU. This says, target
by target, where it moved and what the shadow gets wrong, so a mean can be
traced back to the cases behind it.

For every FINISHED target (all 12 variants have a readable results.json with at
least 3 base runs and the chosen run's mask) it writes, under results/ablation_cases/:
  <subset>/<stem>.png  one figure: a build-up row (target, monolithic CMA-ES,
                       +1 ... +6 = UMBRA), a leave-one-out row (target, UMBRA,
                       -2 ... -6, legend) and a plain-English findings block;
  cases.csv            one row per target;
  index.html           self-contained sortable, filterable table; click a row for its figure;
  failures.md          notable effects per component, worst and best targets.
cases.csv, index.html and failures.md are rebuilt from all finished targets on
every run. A figure is redrawn only when it is missing or older than one of its
12 results.json (--only-missing, the default), or always with --force.

Cells. Best of 3 seeds per cell, picked exactly as scripts/ablation_metrics.py
--k 3 picks it (choose): among non-extra runs with run < 3, the highest per-run
"iou"; a tie at the stored 4 decimals goes to best_run when best_run is among
the tied runs, otherwise to the first tied run in results.json order. The 36
DejaVu Sans Bold letters_upper/digits cells hold 5 runs; runs 3-4 are ignored.
The shadow is that run's <stem>_runKK.png (<stem>_best.png only when the chosen
run is best_run and has no run PNG, as ablation_metrics.py). The target is
targets_grounded/<subset>/<stem>.png read as run_base_optimizer.load_target
reads it (NEAREST to 128 px, dark = shape). IoU is recomputed from the two masks
and checked against the run's recorded iou_vs_original (tolerance 1e-4, as
ablation_metrics.py). These few helpers are copied rather than imported because
importing ablation_metrics / run_base_optimizer pulls in torch.

Sign convention (the paper's tables). Build-up row: delta = IoU(cell) -
IoU(cell to its left). Leave-one-out row: delta = IoU(UMBRA without the
component) - IoU(UMBRA). A component that helps is + when added and - when
removed. Notable means |delta| >= SPREAD = 0.02 (collect_ablation.py), the
within-target seed spread; a smaller change is seed noise and is never called a
failure or a gain. Removing zone assignment also removes the aimed start, which
needs the zones (dagger).

Coverage of the UMBRA shadow: missed = target pixels with no shadow, extra =
shadow pixels off the target, both as a fraction of the target area;
under-coverage when missed > extra, over-coverage when extra > missed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import importlib.util
import json
import math
import os
import statistics as st
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from collect_ablation import SPREAD  # noqa: E402  (stdlib-only module)

NEAREST = getattr(Image, "Resampling", Image).NEAREST
N_ARMS, K, SIZE = 3, 3, 128
IOU_TOL = 1e-4   # ablation_metrics.IOU_TOL: recorded IoU is rounded to 4 dp
POOR = 0.5       # "poor solve" below this UMBRA IoU
ZERO = 0.0005    # |delta| below this prints as +-0.000, in grey

BUILD_UP = ["flat", "fwd_joint", "plus_assign", "plus_voronoi", "plus_backward", "plus_icp", "full"]
LEAVE_ONE_OUT = ["full_no_assign", "full_no_voronoi", "full_no_backward", "full_no_icp", "full_no_fd"]
VARIANTS = BUILD_UP + LEAVE_ONE_OUT
SUBSETS = ["abstract", "animals", "digits", "hand_shadow", "letters_lower", "letters_upper", "objects", "teleop"]
GLYPH_SUBSETS = {"digits", "letters_lower", "letters_upper"}
FONT_NAMES = {"dejavusans-bold": "DejaVu Sans Bold", "dejavusansmono-bold": "DejaVu Sans Mono Bold",
              "dejavuserif-bold": "DejaVu Serif Bold"}

# (index, name, variant with it added, the build-up cell before it, UMBRA without it or None)
COMPONENTS = [
    (1, "arm-by-arm sweep", "fwd_joint", "flat", None),
    (2, "zone assignment", "plus_assign", "fwd_joint", "full_no_assign"),
    (3, "aimed start", "plus_voronoi", "plus_assign", "full_no_voronoi"),
    (4, "backward sweep", "plus_backward", "plus_voronoi", "full_no_backward"),
    (5, "ICP restart", "plus_icp", "plus_backward", "full_no_icp"),
    (6, "FD polish", "full", "plus_icp", "full_no_fd"),
]
CNAME = {i: name for i, name, *_ in COMPONENTS}
DAGGER = {2: "\u2020"}
MINUS = "\u2212"

# figure colours
INK = (34, 34, 34)
SHADE = (30, 30, 30)        # overlap, and the target itself
EXTRA = (240, 132, 28)      # shadow only
MISSED = (150, 201, 242)    # target only
WHITE = (255, 255, 255)
BLUE = (28, 98, 180)
ORANGE = (212, 102, 0)
GREY = (138, 138, 138)
RULE = (208, 208, 208)


# --- selection and masks (copies; see docstring) -------------------------------------

def choose(r, k=K):
    """ablation_metrics.choose with k fixed -> the chosen run dict, or None when fewer than k base runs."""
    base = [x for x in r["runs"] if not x.get("extra")]
    cand = [x for x in base if int(x["run"]) < k]
    if len(cand) < k:
        return None
    top = max(x["iou"] for x in cand)
    tied = [x for x in cand if x["iou"] == top]
    return next((x for x in tied if x["run"] == r.get("best_run")), tied[0])


def load_target(path, size=SIZE):
    """run_base_optimizer.load_target: NEAREST to size, dark = shape."""
    return np.array(Image.open(path).convert("L").resize((size, size), NEAREST)) < 128


def load_mask(path, size=SIZE):
    """metrics.load_mask: dark = shadow."""
    return np.array(Image.open(path).convert("L").resize((size, size), NEAREST)) < 128


def mask_iou(t, s):
    u = int((t | s).sum())
    return float((t & s).sum() / u) if u else 1.0


# --- judgement ---------------------------------------------------------------------------

def status(d, removed=False):
    """'fail', 'gain' or 'noise'. A helpful component is + when added and - when removed."""
    h = -d if removed else d
    if h >= SPREAD:
        return "gain"
    if h <= -SPREAD:
        return "fail"
    return "noise"


def tone(d, removed=False):
    if abs(d) < ZERO:
        return GREY
    return BLUE if (d < 0 if removed else d > 0) else ORANGE


def fd(x, dec=3, minus=MINUS):
    """Signed delta: +0.031, -0.031 (U+2212 by default), +-0.000 below display precision."""
    if abs(x) < 0.5 * 10 ** -dec:
        return "\u00b1" + f"{0:.{dec}f}"
    return ("+" if x > 0 else minus) + f"{abs(x):.{dec}f}"


def describe(subset, stem):
    """-> (glyph or name, font or '')."""
    if subset in GLYPH_SUBSETS and "_" in stem:
        glyph, rest = stem.split("_", 1)
        v2 = rest.endswith("_v2")
        key = rest[:-3] if v2 else rest
        return glyph, FONT_NAMES.get(key, key) + (", v2" if v2 else "")
    return stem, ""


def findings(rec):
    """-> [(kind, prefix, text)], most severe first; kinds: fail, warn, gain, quiet, info."""
    iou = rec["iou"]
    bad, good = [], []
    for i, name, add_v, prev_v, loo_v in COMPONENTS:
        dag = DAGGER.get(i, "")
        da, sa = rec["added"][i], status(rec["added"][i])
        dr = rec["removed"].get(i)
        sr = status(dr, removed=True) if dr is not None else "noise"
        at = f"({iou[prev_v]:.3f} \u2192 {iou[add_v]:.3f})"
        rt = f"({iou['full']:.3f} \u2192 {iou[loo_v]:.3f})" if loo_v else ""
        if sa == "fail" and sr == "fail":
            bad.append((max(-da, dr), "fail", "FAILURE:",
                        f"{name} hurts both ways here \u2014 adding it lowered IoU by {-da:.3f} {at}; "
                        f"removing it from UMBRA raised IoU by {dr:.3f} {rt}{dag}"))
        else:
            if sa == "fail":
                bad.append((-da, "fail", "FAILURE:", f"adding {name} lowered IoU by {-da:.3f} {at}"))
            if sr == "fail":
                bad.append((dr, "fail", "FAILURE:",
                            f"{name} hurts here \u2014 removing it raised IoU by {dr:.3f} {rt}{dag}"))
        if sa == "gain" and sr == "gain":
            good.append((max(da, -dr), "gain", "GAIN:",
                         f"{name} +{da:.3f} when added, {MINUS}{-dr:.3f} when removed{dag}"))
        else:
            if sa == "gain":
                good.append((da, "gain", "GAIN:", f"adding {name} raised IoU by {da:.3f} {at}"))
            if sr == "gain":
                good.append((-dr, "gain", "GAIN:",
                             f"{name} helps here \u2014 removing it lowered IoU by {-dr:.3f} {rt}{dag}"))
    g = rec["gain"]
    if g <= -SPREAD:
        bad.append((-g, "warn", "", f"UMBRA is worse than monolithic CMA-ES by {-g:.3f} "
                                    f"({iou['flat']:.3f} \u2192 {iou['full']:.3f})"))
    bad.sort(key=lambda x: -x[0])
    good.sort(key=lambda x: -x[0])
    out = [x[1:] for x in bad]
    if iou["full"] < POOR:
        out.append(("warn", "Poor solve:", f"UMBRA IoU {iou['full']:.3f} < {POOR}"))  # 3 dp: 0.499 must not print as 0.50
    out += [x[1:] for x in good]
    if rec["n_failures"] + rec["n_gains"] == 0:
        out.append(("quiet", "", "No notable effects (all |\u0394| < 0.02, within seed noise)" if not out else
                    "No notable component effects (all component |\u0394| < 0.02, within seed noise)"))
    out.append(("info", "", f"UMBRA misses {100 * rec['missed']:.0f}% of the target (blue) and casts "
                            f"{100 * rec['extra']:.0f}% extra shadow (orange), both relative to target area: "
                            f"{rec['coverage']}"))
    return out


def line_text(f):
    kind, prefix, text = f
    return f"{prefix} {text}" if prefix else text


# --- figure ------------------------------------------------------------------------------

class Fonts:
    def __init__(self):
        self.paths = None
        cands = []
        spec = importlib.util.find_spec("matplotlib")  # its bundled DejaVu, without importing matplotlib
        if spec and spec.origin:
            d = os.path.join(os.path.dirname(spec.origin), "mpl-data", "fonts", "ttf")
            cands.append((os.path.join(d, "DejaVuSans.ttf"), os.path.join(d, "DejaVuSans-Bold.ttf")))
        win = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        cands += [("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
                  (os.path.join(win, "arial.ttf"), os.path.join(win, "arialbd.ttf"))]
        for reg, bold in cands:
            try:
                ImageFont.truetype(reg, 12), ImageFont.truetype(bold, 12)
                self.paths = (reg, bold)
                break
            except OSError:
                continue
        self.cache = {}

    def __call__(self, size, bold=False):
        key = (size, bold)
        if key not in self.cache:
            self.cache[key] = (ImageFont.truetype(self.paths[bold], size) if self.paths
                               else ImageFont.load_default())
        return self.cache[key]


F = None
CELL, FR, GAP, MARGIN, NCOL = 256, 8, 10, 28, 8
COLW = CELL + 2 * FR
W = 2 * MARGIN + NCOL * COLW + (NCOL - 1) * GAP
LH = 22                      # line height under cells
CELL_H = 2 * FR + CELL + 8 + 3 * LH + 6
FIND_SIZE, FIND_LH = 19, 27


def wrap(draw, s, font, width):
    lines, cur = [], ""
    for w in s.split(" "):
        t = w if not cur else f"{cur} {w}"
        if not cur or draw.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def overlay_tile(t, s):
    rgb = np.empty(t.shape + (3,), np.uint8)
    rgb[:] = WHITE
    rgb[t & s] = SHADE
    rgb[s & ~t] = EXTRA
    rgb[t & ~s] = MISSED
    return Image.fromarray(rgb, "RGB").resize((CELL, CELL), NEAREST)


def target_tile(t):
    rgb = np.empty(t.shape + (3,), np.uint8)
    rgb[:] = WHITE
    rgb[t] = SHADE
    return Image.fromarray(rgb, "RGB").resize((CELL, CELL), NEAREST)


def col_x(c):
    return MARGIN + c * (COLW + GAP)


def draw_cell(im, dr, c, y, tile, title, lines, state=None):
    """lines: [(text, colour, bold)] under the title; state 'fail' | 'gain' | None frames the tile."""
    x = col_x(c)
    ix, iy = x + FR, y + FR
    im.paste(tile, (ix, iy))
    dr.rectangle([ix - 1, iy - 1, ix + CELL, iy + CELL], outline=RULE, width=1)
    if state == "fail":
        dr.rectangle([ix - 7, iy - 7, ix + CELL + 6, iy + CELL + 6], outline=ORANGE, width=7)
        cx, cy, rr = ix + CELL - 21, iy + 21, 16
        dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=ORANGE, outline=WHITE, width=2)
        dr.text((cx, cy + 1), "\u2717", font=F(21, True), fill=WHITE, anchor="mm")
    elif state == "gain":
        dr.rectangle([ix - 3, iy - 3, ix + CELL + 2, iy + CELL + 2], outline=BLUE, width=3)
    ty = iy + CELL + FR + 8
    font = F(17, True)
    for ln in wrap(dr, title, font, COLW - 4)[:2]:
        dr.text((x + COLW / 2, ty), ln, font=font, fill=INK, anchor="ma")
        ty += LH
    for text, colour, bold in lines:
        dr.text((x + COLW / 2, ty), text, font=F(17, bold), fill=colour, anchor="ma")
        ty += LH


def draw_legend(im, dr, c, y):
    x = col_x(c) + FR
    w = CELL
    dr.rectangle([x - 1, y + FR - 1, x + w, y + CELL_H - 10], outline=RULE, width=1, fill=(250, 250, 250))
    ty = y + FR + 10
    dr.text((x + 12, ty), "Legend", font=F(17, True), fill=INK)
    ty += 28
    f15 = F(15)
    sw = 18
    for colour, label in ((SHADE, "shadow on target"), (EXTRA, "extra shadow (orange)"),
                          (MISSED, "missed target (blue)")):
        dr.rectangle([x + 12, ty + 1, x + 12 + sw, ty + 1 + sw], fill=colour, outline=RULE)
        dr.text((x + 40, ty), label, font=f15, fill=INK)
        ty += 25
    ty += 4
    dr.rectangle([x + 12, ty, x + 12 + sw + 4, ty + sw + 4], outline=ORANGE, width=5)
    dr.text((x + 44, ty + 1), "\u2717 notable failure", font=f15, fill=INK)
    ty += 28
    dr.rectangle([x + 12, ty, x + 12 + sw + 4, ty + sw + 4], outline=BLUE, width=2)
    dr.text((x + 44, ty + 1), "notable gain", font=f15, fill=INK)
    ty += 32
    for segs in ((("\u0394 ", INK), ("blue", BLUE), (" = component helped", INK)),
                 (("\u0394 ", INK), ("orange", ORANGE), (" = it hurt", INK)),
                 (("\u0394 ", INK), ("grey", GREY), (" = |\u0394| < 0.0005", INK))):
        tx = x + 12
        for s, colour in segs:
            dr.text((tx, ty), s, font=f15, fill=colour)
            tx += dr.textlength(s, font=f15)
        ty += 21
    ty += 6
    for ln in (f"notable: |\u0394| \u2265 {SPREAD:.2f} IoU,", "the within-target seed spread",
               "\u2020 removing zone assignment", "   also removes the aimed start"):
        dr.text((x + 12, ty), ln, font=f15, fill=INK)
        ty += 21


def draw_figure(rec, t, masks, path):
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    avail = W - 2 * MARGIN - 22
    ff, ffb = F(FIND_SIZE), F(FIND_SIZE, True)
    wrapped = [(f, wrap(probe, line_text(f), ffb if f[1] else ff, avail)) for f in rec["findings"]]
    n_find = sum(len(w) for _, w in wrapped)

    head_h = 40 + 34 + 16
    row_head = 32
    h = MARGIN + head_h + 2 * (row_head + CELL_H) + 18 + 38 + n_find * FIND_LH + MARGIN
    im = Image.new("RGB", (W, h), WHITE)
    dr = ImageDraw.Draw(im)
    iou, s = rec["iou"], rec["subset"]

    # title
    y = MARGIN
    name, font = rec["glyph"] or rec["stem"], rec["font"]
    t1 = f"{s} / {name}"
    dr.text((MARGIN, y), t1, font=F(30, True), fill=INK)
    if font:
        dr.text((MARGIN + dr.textlength(t1, font=F(30, True)) + 14, y + 4), f"({font})", font=F(26), fill=GREY)
    y += 42
    tx = MARGIN
    for text, colour, bold in ((f"monolithic IoU {iou['flat']:.3f} \u2192 UMBRA IoU {iou['full']:.3f} (\u0394 ", INK, False),
                               (fd(rec["gain"]), tone(rec["gain"]), True), (")", INK, False),
                               ("  \u00b7  N=3, best of 3 seeds", GREY, False)):
        dr.text((tx, y), text, font=F(22, bold), fill=colour)
        tx += dr.textlength(text, font=F(22, bold))
    if rec["glyph"]:  # the file name is not in the title for glyphs
        dr.text((W - MARGIN, y + 4), f"{rec['subset']}/{rec['stem']}", font=F(17), fill=GREY, anchor="ra")
    y += head_h - 42

    ttile = target_tile(t)
    area = int(t.sum())
    tlines = [(f"{area:,} px ({100 * area / t.size:.0f}% of frame)", GREY, False)]

    def row_header(y, bold, rest):
        dr.text((MARGIN, y), bold, font=F(20, True), fill=INK)
        dr.text((MARGIN + dr.textlength(bold, font=F(20, True)) + 10, y + 2), rest, font=F(17), fill=GREY)

    # row 1: adding
    row_header(y, "Adding components",
               f"\u0394 = IoU \u2212 IoU of the cell to its left;  + means the component helped")
    y += row_head
    draw_cell(im, dr, 0, y, ttile, "target", tlines)
    draw_cell(im, dr, 1, y, overlay_tile(t, masks["flat"]), "monolithic CMA-ES",
              [(f"IoU {iou['flat']:.3f}", INK, False), ("baseline", GREY, False)])
    for i, name_, add_v, prev_v, _ in COMPONENTS:
        d = rec["added"][i]
        st_ = status(d)
        draw_cell(im, dr, 1 + i, y, overlay_tile(t, masks[add_v]),
                  f"+{i} {name_}" + (" (= UMBRA)" if add_v == "full" else ""),
                  [(f"IoU {iou[add_v]:.3f}", INK, False), (f"\u0394 {fd(d)}", tone(d), True)],
                  st_ if st_ != "noise" else None)
    y += CELL_H + 18
    dr.line([MARGIN, y - 10, W - MARGIN, y - 10], fill=RULE, width=1)

    # row 2: removing
    row_header(y, "Removing from UMBRA",
               f"\u0394 = IoU without the component \u2212 IoU of UMBRA;  \u2212 means the component helped")
    y += row_head
    draw_cell(im, dr, 0, y, ttile, "target", tlines)
    draw_cell(im, dr, 1, y, overlay_tile(t, masks["full"]), "UMBRA",
              [(f"IoU {iou['full']:.3f}", INK, False), ("reference", GREY, False)])
    for i, name_, _, _, loo_v in COMPONENTS:
        if not loo_v:
            continue
        d = rec["removed"][i]
        st_ = status(d, removed=True)
        draw_cell(im, dr, i, y, overlay_tile(t, masks[loo_v]), f"{MINUS}{i} {name_}{DAGGER.get(i, '')}",
                  [(f"IoU {iou[loo_v]:.3f}", INK, False), (f"\u0394 {fd(d)}", tone(d, removed=True), True)],
                  st_ if st_ != "noise" else None)
    draw_legend(im, dr, 7, y)
    y += CELL_H + 8

    # findings
    dr.line([MARGIN, y, W - MARGIN, y], fill=RULE, width=1)
    y += 10
    dr.text((MARGIN, y), "Findings", font=F(21, True), fill=INK)
    dr.text((MARGIN + dr.textlength("Findings", font=F(21, True)) + 12, y + 3),
            f"most severe first;  notable = |\u0394| \u2265 {SPREAD:.2f} IoU (within-target seed spread), "
            f"anything smaller is seed noise", font=F(17), fill=GREY)
    y += 38
    colours = {"fail": ORANGE, "gain": BLUE, "warn": ORANGE, "quiet": GREY, "info": INK}
    for (kind, prefix, text), lines in wrapped:
        x0 = MARGIN + 22
        dr.text((MARGIN + 4, y), "\u2022", font=ff, fill=colours[kind] if kind != "info" else GREY)
        for j, ln in enumerate(lines):
            if j == 0 and prefix and ln.startswith(prefix):
                dr.text((x0, y), prefix, font=ffb, fill=colours[kind])
                rest_x = x0 + dr.textlength(prefix + " ", font=ffb)
                dr.text((rest_x, y), ln[len(prefix) + 1:], font=ff, fill=INK)
            else:
                dr.text((x0, y), ln, font=ff, fill=INK if kind in ("fail", "gain", "info") else colours[kind])
            y += FIND_LH

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.png"
    im.quantize(colors=96, method=Image.Quantize.MEDIANCUT).save(tmp, optimize=True)
    os.replace(tmp, path)


# --- scan and score ----------------------------------------------------------------------

def scan(root, targets_dir, warn):
    """-> (targets [(subset, stem)], finished {key: {variant: cell}}, unfinished {key: [reasons]})."""
    targets = [(s, os.path.splitext(os.path.basename(p))[0])
               for s in SUBSETS for p in sorted(glob.glob(os.path.join(targets_dir, s, "*.png")))]
    finished, unfinished = {}, {}
    for s, stem in targets:
        cells, missing = {}, []
        for v in VARIANTS:
            d = os.path.join(root, f"{v}-n{N_ARMS}", s, stem)
            rj = os.path.join(d, "results.json")
            if not os.path.exists(rj):
                missing.append(v)
                continue
            try:
                with open(rj, encoding="utf-8") as fh:
                    r = json.load(fh)
            except (OSError, ValueError):
                missing.append(f"{v} (results.json unreadable, being written?)")
                continue
            run = choose(r)
            if run is None:
                missing.append(f"{v} (< {K} base runs)")
                continue
            kk = int(run["run"])
            mp, src = os.path.join(d, f"{stem}_run{kk:02d}.png"), "run_png"
            if not os.path.exists(mp):
                bp = os.path.join(d, f"{stem}_best.png")
                if kk == r.get("best_run") and os.path.exists(bp):
                    mp, src = bp, "best_png"
                    warn("chosen run has no run PNG, used _best.png", f"{v}/{s}/{stem} run {kk}")
                else:
                    missing.append(f"{v} (no {stem}_run{kk:02d}.png)")
                    continue
            if (r.get("rig") or {}).get("target_fit"):
                warn("fitted solve (IoU here is vs the authored target)", f"{v}/{s}/{stem}")
            base = [x for x in r["runs"] if not x.get("extra") and int(x["run"]) < K]
            top = max(x["iou"] for x in base)
            n_tied = sum(x["iou"] == top for x in base)
            cells[v] = dict(run=kk, rj=rj, rj_mtime=os.path.getmtime(rj), mask_path=mp, src=src,
                            iou_reported=run.get("iou_vs_original", run.get("iou")),
                            n_runs=len(r["runs"]), tied=n_tied, best_run=r.get("best_run"))
        if missing:
            unfinished[(s, stem)] = missing
        else:
            finished[(s, stem)] = cells
    known = set(targets)
    for rj in glob.glob(os.path.join(root, f"*-n{N_ARMS}", "*", "*", "results.json")):
        parts = os.path.normpath(rj).split(os.sep)
        if (parts[-3], parts[-2]) not in known:
            warn(f"results.json for a target not in {os.path.basename(targets_dir)}", f"{parts[-4]}/{parts[-3]}/{parts[-2]}")
    return targets, finished, unfinished


def score(key, cells, targets_dir, out_dir, warn):
    s, stem = key
    t = load_target(os.path.join(targets_dir, s, f"{stem}.png"))
    masks, iou = {}, {}
    for v in VARIANTS:
        c = cells[v]
        m = load_mask(c["mask_path"])
        masks[v] = m
        iou[v] = mask_iou(t, m)
        c["iou"] = iou[v]
        rep = c["iou_reported"]
        if rep is not None and abs(iou[v] - float(rep)) > IOU_TOL:
            warn(f"recomputed IoU differs from results.json by > {IOU_TOL}",
                 f"{v}/{s}/{stem} run {c['run']}: {iou[v]:.5f} vs {float(rep):.4f}")
    glyph, font = describe(s, stem)
    rec = dict(subset=s, stem=stem, glyph=glyph if s in GLYPH_SUBSETS else "", font=font, iou=iou,
               gain=iou["full"] - iou["flat"],
               added={i: iou[a] - iou[p] for i, _, a, p, _ in COMPONENTS},
               removed={i: iou[lo] - iou["full"] for i, _, _, _, lo in COMPONENTS if lo},
               runs={v: cells[v]["run"] for v in VARIANTS},
               image=f"{s}/{stem}.png", rj_mtime=max(c["rj_mtime"] for c in cells.values()))
    rec["n_failures"] = (sum(status(d) == "fail" for d in rec["added"].values())
                         + sum(status(d, True) == "fail" for d in rec["removed"].values()))
    rec["n_gains"] = (sum(status(d) == "gain" for d in rec["added"].values())
                      + sum(status(d, True) == "gain" for d in rec["removed"].values()))
    area = int(t.sum())
    S = masks["full"]
    rec["missed"] = int((t & ~S).sum()) / area if area else 0.0
    rec["extra"] = int((S & ~t).sum()) / area if area else 0.0
    rec["coverage"] = ("under-coverage" if rec["missed"] > rec["extra"] else
                       "over-coverage" if rec["extra"] > rec["missed"] else "balanced")
    rec["findings"] = findings(rec)
    return rec, t, masks


# --- tables --------------------------------------------------------------------------------

CSV_COLS = (["subset", "stem", "glyph", "font"] + [f"iou_{v}" for v in VARIANTS] + ["gain_umbra_vs_mono"]
            + [f"added_delta_{i}" for i in range(1, 7)] + [f"removed_delta_{i}" for i in range(2, 7)]
            + ["n_failures", "n_gains", "missed_frac", "extra_frac", "coverage_type", "findings", "image",
               "best_runs"])


def write_csv(recs, path, out_rel):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for r in recs:
            w.writerow([r["subset"], r["stem"], r["glyph"], r["font"]]
                       + [f"{r['iou'][v]:.6f}" for v in VARIANTS] + [f"{r['gain']:.6f}"]
                       + [f"{r['added'][i]:.6f}" for i in range(1, 7)]
                       + [f"{r['removed'][i]:.6f}" for i in range(2, 7)]
                       + [r["n_failures"], r["n_gains"], f"{r['missed']:.4f}", f"{r['extra']:.4f}", r["coverage"],
                          " | ".join(line_text(x) for x in r["findings"]), f"{out_rel}/{r['image']}",
                          " ".join(f"{v}:{r['runs'][v]}" for v in VARIANTS)])


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ablation cases bench562</title>
<style>
:root{--ink:#222;--grey:#8a8a8a;--blue:#1c62b4;--orange:#d46600;--bgb:#e2edf9;--bgo:#fde8d3;--rule:#dcdcdc}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{font:13px/1.35 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:#fff;display:flex;flex-direction:column}
header{padding:10px 16px 8px;border-bottom:1px solid var(--rule)}
h1{font-size:16px;margin:0 0 3px}
.note{color:#555;font-size:12px;max-width:1400px}
.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin-top:8px}
.controls input[type=search]{width:260px;padding:3px 6px}
#count{color:#555}
#wrap{flex:1 1 44%;overflow:auto;min-height:140px}
table{border-collapse:collapse;width:100%}
th,td{padding:3px 6px;border-bottom:1px solid #eee;white-space:nowrap}
th{position:sticky;top:0;z-index:1;background:#f4f4f4;border-bottom:1px solid #c8c8c8;cursor:pointer;user-select:none;text-align:right;font-weight:600}
th:hover{background:#e9eef5}
th.l,td.l{text-align:left}
td{text-align:right;font-variant-numeric:tabular-nums}
td.find{max-width:560px;overflow:hidden;text-overflow:ellipsis;text-align:left;color:#444}
tbody tr{cursor:pointer}
tbody tr:hover td{background-color:#f2f6fb}
tr.sel td{box-shadow:inset 0 2px 0 var(--blue),inset 0 -2px 0 var(--blue)}
.help{color:var(--blue)}.hurt{color:var(--orange)}.zero{color:var(--grey)}
td.ngain{color:var(--blue);font-weight:600;background:var(--bgb)}
td.nfail{color:var(--orange);font-weight:700;background:var(--bgo)}
td.poor{color:var(--orange);font-weight:600}
td.fails{color:var(--orange);font-weight:700}
#panel{flex:1 1 56%;overflow:auto;border-top:3px solid var(--rule);padding:8px 16px 16px}
#ptitle{font-weight:600;font-size:14px;margin:2px 0 4px}
#pfind{margin:0 0 8px;padding-left:18px}
#pfind li{margin:1px 0}
#pfind .fail b,#pfind .warn{color:var(--orange)}
#pfind .gain b{color:var(--blue)}
#pfind .quiet{color:var(--grey)}
#panel img{width:100%;max-width:2302px;display:block;border:1px solid #eee}
kbd{border:1px solid #ccc;border-radius:3px;padding:0 3px;font-size:11px}
</style></head><body>
<header>
<h1>Ablation cases &middot; __ROOT__ &middot; N=3, best of 3 seeds</h1>
<div class="note">__NOTE__</div>
<div class="controls">
<label>subset <select id="subset"><option value="">all</option>__OPTIONS__</select></label>
<label><input type="checkbox" id="failonly"> failures only</label>
<label>search <input type="search" id="q" placeholder="stem, glyph, font, finding"></label>
<span id="count"></span>
<span class="note">click a header to sort; click a row (or <kbd>&uarr;</kbd>/<kbd>&darr;</kbd>) to show its figure</span>
</div>
</header>
<div id="wrap"><table><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table></div>
<div id="panel"><div id="ptitle">select a row</div><ul id="pfind"></ul><a id="plink" target="_blank"><img id="pimg" alt=""></a></div>
<script>
const D = __DATA__;
const SPREAD = __SPREAD__;
const COMP = ["", "arm-by-arm sweep", "zone assignment", "aimed start", "backward sweep", "ICP restart", "FD polish"];
const COLS = [
  {k:"s", h:"subset", t:"s"}, {k:"t", h:"target", t:"s"},
  {k:"mono", h:"mono IoU", t:"n", tip:"monolithic CMA-ES (flat), best of 3"},
  {k:"umbra", h:"UMBRA IoU", t:"n", tip:"UMBRA (full), best of 3"},
  {k:"gain", h:"gain", t:"d", dir:1, tip:"UMBRA IoU \u2212 monolithic IoU"}];
for (let i = 1; i <= 6; i++) COLS.push({k:"a"+i, h:"+\u0394"+i, t:"d", dir:1,
  tip:"adding "+COMP[i]+": IoU \u2212 IoU of the cell before it (+ = helped)"});
for (let i = 2; i <= 6; i++) COLS.push({k:"r"+i, h:"\u2212\u0394"+i+(i==2?"\u2020":""), t:"d", dir:-1,
  tip:"removing "+COMP[i]+" from UMBRA: IoU without it \u2212 IoU of UMBRA (\u2212 = it helped)"+(i==2?"; \u2020 also removes the aimed start":"")});
COLS.push({k:"nf", h:"failures", t:"n", tip:"notable failures: added \u0394 \u2264 \u22120.02 or removed \u0394 \u2265 +0.02"});
COLS.push({k:"fs", h:"findings", t:"s"});
let sortKey = "umbra", sortDir = 1, selKey = null, view = [];
const $ = id => document.getElementById(id);
function fmtD(v) { if (v == null) return ""; if (Math.abs(v) < 0.0005) return "\u00b10.000"; return (v > 0 ? "+" : "\u2212") + Math.abs(v).toFixed(3); }
function dcls(v, dir) { if (v == null) return ""; const h = v * dir;
  if (h >= SPREAD) return "ngain"; if (h <= -SPREAD) return "nfail"; if (Math.abs(v) < 0.0005) return "zero"; return h > 0 ? "help" : "hurt"; }
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"})[c]); }
function head() {
  $("head").innerHTML = COLS.map(c => `<th class="${c.t=="s"?"l":""}" data-k="${c.k}" title="${esc(c.tip||"")}">${esc(c.h)}${c.k==sortKey?(sortDir>0?" \u25b2":" \u25bc"):""}</th>`).join("");
  for (const th of $("head").children) th.onclick = () => {
    const k = th.dataset.k; if (k == sortKey) sortDir = -sortDir; else { sortKey = k; sortDir = 1; }
    render(); };
}
function render() {
  head();
  const sub = $("subset").value, fo = $("failonly").checked, q = $("q").value.trim().toLowerCase();
  view = D.filter(r => (!sub || r.s == sub) && (!fo || r.nf > 0) &&
    (!q || (r.s+" "+r.t+" "+r.g+" "+r.f+" "+r.fs).toLowerCase().includes(q)));
  const col = COLS.find(c => c.k == sortKey);
  view.sort((a, b) => { const x = a[sortKey], y = b[sortKey];
    if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
    const c = col.t == "s" ? String(x).localeCompare(String(y)) : x - y;
    return c * sortDir || a.s.localeCompare(b.s) || a.t.localeCompare(b.t); });
  $("body").innerHTML = view.map((r, i) => "<tr data-i=\"" + i + "\"" + (r.s+"/"+r.t == selKey ? " class=\"sel\"" : "") + ">" + COLS.map(c => {
    const v = r[c.k];
    if (c.t == "d") return `<td class="${dcls(v, c.dir)}">${fmtD(v)}</td>`;
    if (c.k == "umbra") return `<td class="${v < 0.5 ? "poor" : ""}">${v.toFixed(3)}</td>`;
    if (c.k == "mono") return `<td>${v.toFixed(3)}</td>`;
    if (c.k == "nf") return `<td class="${v > 0 ? "fails" : "zero"}">${v}</td>`;
    if (c.k == "fs") return `<td class="find" title="${esc(v)}">${esc(v)}</td>`;
    if (c.k == "t") return `<td class="l" title="${esc(r.g ? r.g + " \u00b7 " + r.f : r.t)}">${esc(v)}</td>`;
    return `<td class="l">${esc(v)}</td>`; }).join("") + "</tr>").join("");
  for (const tr of $("body").children) tr.onclick = () => select(+tr.dataset.i, false);
  $("count").textContent = view.length + " of " + D.length + " targets";
}
function select(i, scroll) {
  if (i < 0 || i >= view.length) return;
  const r = view[i]; selKey = r.s + "/" + r.t;
  for (const tr of $("body").children) tr.classList.toggle("sel", +tr.dataset.i == i);
  if (scroll) $("body").children[i].scrollIntoView({block: "nearest"});
  $("ptitle").textContent = r.s + " / " + (r.g ? r.g + " (" + r.f + ")" : r.t) + "  \u00b7  mono " + r.mono.toFixed(3) + " \u2192 UMBRA " + r.umbra.toFixed(3) + " (\u0394 " + fmtD(r.gain) + ")";
  $("pfind").innerHTML = r.find.map(f => `<li class="${f[0]}">${f[1] ? "<b>" + esc(f[1]) + "</b> " : ""}${esc(f[2])}</li>`).join("");
  $("pimg").src = r.img; $("plink").href = r.img;
}
document.addEventListener("keydown", e => {
  if (e.target.tagName == "INPUT" || e.target.tagName == "SELECT") return;
  if (e.key != "ArrowDown" && e.key != "ArrowUp") return;
  e.preventDefault();
  let i = view.findIndex(r => r.s + "/" + r.t == selKey);
  select(i < 0 ? 0 : Math.max(0, Math.min(view.length - 1, i + (e.key == "ArrowDown" ? 1 : -1))), true);
});
for (const id of ["subset", "failonly", "q"]) $(id).addEventListener("input", render);
render();
if (view.length) select(0, false);
</script></body></html>
"""


def write_html(recs, path, root_rel, n_total):
    data = []
    for r in recs:
        row = dict(s=r["subset"], t=r["stem"], g=r["glyph"], f=r["font"], mono=round(r["iou"]["flat"], 6),
                   umbra=round(r["iou"]["full"], 6), gain=round(r["gain"], 6), nf=r["n_failures"], ng=r["n_gains"],
                   miss=round(r["missed"], 4), extra=round(r["extra"], 4), cov=r["coverage"], img=r["image"],
                   fs="; ".join(line_text(x) for x in r["findings"] if x[0] != "info"),
                   find=[list(x) for x in r["findings"]])
        row.update({f"a{i}": round(r["added"][i], 6) for i in range(1, 7)})
        row.update({f"r{i}": round(r["removed"][i], 6) for i in range(2, 7)})
        data.append(row)
    subsets = [s for s in SUBSETS if any(r["subset"] == s for r in recs)]
    note = (f"{len(recs)} of {n_total} targets finished (all 12 variants). "
            "Adding &Delta; = IoU(cell) &minus; IoU(previous build-up cell); removing &Delta; = IoU(UMBRA without it) "
            "&minus; IoU(UMBRA). A helpful component is + when added, &minus; when removed: "
            "<span class=help>blue</span> = helped, <span class=hurt>orange</span> = hurt, "
            f"shaded = notable (|&Delta;| &ge; {SPREAD:.2f}, the within-target seed spread). "
            "&dagger; removing zone assignment also removes the aimed start.")
    page = (HTML.replace("__ROOT__", root_rel).replace("__NOTE__", note)
            .replace("__OPTIONS__", "".join(f'<option value="{s}">{s}</option>' for s in subsets))
            .replace("__SPREAD__", repr(SPREAD))
            .replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")))
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


# --- failures.md -------------------------------------------------------------------------

def _name(r):
    return f"{r['subset']}/{r['stem']}"


def _img(r):
    return f"[{r['image']}]({r['image']})"


def _dmd(x):
    return fd(x, 3, minus="-")


def counts_table(recs):
    out = ["| component | added: helped | added: hurt | added: noise | mean added \u0394 "
           "| removed: IoU fell (helps) | removed: IoU rose (hurts) | removed: noise | mean removed \u0394 |",
           "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for i, name, _, _, lo in COMPONENTS:
        a = [r["added"][i] for r in recs]
        ca = collections.Counter(status(d) for d in a)
        line = f"| {i} {name}{DAGGER.get(i, '')} | {ca['gain']} | {ca['fail']} | {ca['noise']} | {st.mean(a):+.4f} |"
        if lo:
            rm = [r["removed"][i] for r in recs]
            cr = collections.Counter(status(d, True) for d in rm)
            line += f" {cr['gain']} | {cr['fail']} | {cr['noise']} | {st.mean(rm):+.4f} |"
        else:
            line += " \u2014 | \u2014 | \u2014 | \u2014 |"
        out.append(line)
    return out


def target_table(rows):
    out = ["| # | target | UMBRA IoU | mono IoU | UMBRA \u2212 mono | missed | extra | notable fail / gain | image |",
           "|--:|---|--:|--:|--:|--:|--:|--:|---|"]
    for n, r in enumerate(rows, 1):
        out.append(f"| {n} | {_name(r)} | {r['iou']['full']:.3f} | {r['iou']['flat']:.3f} | {_dmd(r['gain'])} "
                   f"| {100 * r['missed']:.0f}% | {100 * r['extra']:.0f}% | {r['n_failures']} / {r['n_gains']} | {_img(r)} |")
    return out if rows else ["_none_"]


def effect_table(recs, i, removed, want):
    """want 'hurt' or 'help': top 10 entries in that direction, most extreme first."""
    comp = [x for x in COMPONENTS if x[0] == i][0]
    _, _, add_v, prev_v, loo_v = comp
    if removed:
        vals = [(r["removed"][i], r) for r in recs]
        sign = 1 if want == "hurt" else -1       # removal raising IoU = component hurts
    else:
        vals = [(r["added"][i], r) for r in recs]
        sign = -1 if want == "hurt" else 1       # adding lowering IoU = component hurts
    vals = sorted([x for x in vals if sign * x[0] > 0], key=lambda x: -sign * x[0])[:10]
    if not vals:
        return ["_none_"]
    which = "removed" if removed else "added"
    span = "UMBRA \u2192 without" if removed else "before \u2192 after"
    out = [f"| # | target | {which} \u0394 | IoU {span} | notable | image |",
           "|--:|---|--:|---|---|---|"]
    for n, (d, r) in enumerate(vals, 1):
        before, after = (r["iou"]["full"], r["iou"][loo_v]) if removed else (r["iou"][prev_v], r["iou"][add_v])
        noted = status(d, removed) != "noise"
        out.append(f"| {n} | {_name(r)} | {_dmd(d)} | {before:.3f} \u2192 {after:.3f} "
                   f"| {'yes' if noted else 'no (seed noise)'} | {_img(r)} |")
    return out


def write_md(recs, path, root_rel, n_total, checks):
    by_sub = collections.OrderedDict((s, [r for r in recs if r["subset"] == s]) for s in SUBSETS)
    by_sub = collections.OrderedDict((s, v) for s, v in by_sub.items() if v)
    per = ", ".join(f"{s} {len(v)}" for s, v in by_sub.items())
    worst_iou = sorted(recs, key=lambda r: (r["iou"]["full"], _name(r)))
    worst_gain = sorted(recs, key=lambda r: (r["gain"], _name(r)))
    best_gain = sorted(recs, key=lambda r: (-r["gain"], _name(r)))
    n_fail_t = sum(r["n_failures"] > 0 for r in recs)
    n_poor = sum(r["iou"]["full"] < POOR for r in recs)
    n_worse = sum(r["gain"] <= -SPREAD for r in recs)
    L = ["# Ablation cases: notable effects per target", "",
         f"Root `{root_rel}`, N=3, best of 3 seeds per cell. {len(recs)} of {n_total} targets finished "
         f"(all 12 variants): {per}. Generated {time.strftime('%Y-%m-%d %H:%M')} by scripts/ablation_case_report.py.", "",
         f"Adding \u0394 = IoU(cell) \u2212 IoU(previous build-up cell); removing \u0394 = IoU(UMBRA without the "
         f"component) \u2212 IoU(UMBRA). A component that helps is + when added and \u2212 when removed. Notable = "
         f"|\u0394| \u2265 {SPREAD} IoU, the within-target seed spread; smaller changes are seed noise and are "
         "counted as noise, not as failures or gains. \u2020 removing zone assignment also removes the aimed start. "
         "Image paths are relative to results/ablation_cases/.", "",
         "## Summary", "",
         f"{n_fail_t} of {len(recs)} targets {'has' if n_fail_t == 1 else 'have'} at least one notable failure cell; "
         f"{n_poor} {'has' if n_poor == 1 else 'have'} UMBRA IoU < {POOR}; "
         f"{n_worse} {'has' if n_worse == 1 else 'have'} UMBRA notably worse than monolithic CMA-ES "
         f"(by ≥ {SPREAD}).", "",
         "### Notable effects per component (all finished targets)", "", *counts_table(recs), "",
         "### Top 5 worst UMBRA IoU", "", *target_table(worst_iou[:5]), "",
         "### Top 5 most negative UMBRA \u2212 monolithic", "", *target_table(worst_gain[:5]), "",
         "### Top 5 best UMBRA \u2212 monolithic", "", *target_table(best_gain[:5]), "",
         "## (a) Notable effects per component", "", "### Overall", "", *counts_table(recs), ""]
    for s, rs in by_sub.items():
        L += [f"### {s} ({len(rs)} targets)", "", *counts_table(rs), ""]
    L += ["## (b) Failures", "", "### Top 20 worst by UMBRA IoU", "", *target_table(worst_iou[:20]), "",
          "### Top 20 most negative UMBRA \u2212 monolithic", "", *target_table(worst_gain[:20]), ""]
    for i, name, _, _, lo in COMPONENTS:
        L += [f"### {i} {name}: adding hurt most", "", *effect_table(recs, i, False, "hurt"), ""]
        if lo:
            L += [f"### {i} {name}: removing raised IoU most{DAGGER.get(i, '')}", "",
                  *effect_table(recs, i, True, "hurt"), ""]
    L += ["## (c) Gains", "", "### Top 20 best by UMBRA \u2212 monolithic", "", *target_table(best_gain[:20]), ""]
    for i, name, _, _, lo in COMPONENTS:
        L += [f"### {i} {name}: largest gains when added", "", *effect_table(recs, i, False, "help"), ""]
        if lo:
            L += [f"### {i} {name}: largest drops when removed (it helps){DAGGER.get(i, '')}", "",
                  *effect_table(recs, i, True, "help"), ""]
    L += ["## Checks", ""] + [f"- {c}" for c in checks] + [""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# --- main --------------------------------------------------------------------------------

def lower_priority():
    """Run below normal priority: a solve may be using every core."""
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(10)
    except Exception:
        pass


def cross_check(path, finished, root_rel):
    """Compare chosen run and IoU with an ablation_metrics.py --k 3 CSV. -> (n compared, mismatches)."""
    n, bad = 0, []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("variant_dir") == "target" or row.get("root", root_rel) != root_rel:
                continue
            v = row["variant_dir"][: -len(f"-n{N_ARMS}")] if row["variant_dir"].endswith(f"-n{N_ARMS}") else None
            cells = finished.get((row["subset"], row["stem"]))
            if v not in VARIANTS or cells is None or str(row.get("k")) != str(K):
                continue
            c = cells[v]
            n += 1
            ok_run = row["run"] != "" and int(float(row["run"])) == c["run"]
            ok_iou = row["iou"] != "" and abs(float(row["iou"]) - round(c["iou"], 6)) <= 1.5e-6
            if not (ok_run and ok_iou):
                bad.append(f"{row['variant_dir']}/{row['subset']}/{row['stem']}: csv run {row['run']} iou {row['iou']}"
                           f" vs run {c['run']} iou {c['iou']:.6f}")
    return n, bad


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="optimized/ablation-bench562", help="study root (bench-relative)")
    p.add_argument("--targets", default="targets_grounded", help="target PNGs (bench-relative)")
    p.add_argument("--out", default="results/ablation_cases", help="output folder (bench-relative)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--only-missing", action="store_true", default=True,
                   help="default: draw a figure only if it is missing or older than one of its results.json")
    g.add_argument("--force", action="store_true", help="redraw every figure")
    p.add_argument("--only", nargs="+", metavar="SUBSET/STEM", default=None,
                   help="draw only these figures (with --force: redraw them); tables still cover every finished target")
    p.add_argument("--check-csv", default=None,
                   help="ablation_metrics.py --k 3 CSV to cross-check chosen runs and IoU against")
    p.add_argument("--bench", default=BENCH)
    a = p.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    lower_priority()
    t0 = time.time()
    bench = os.path.abspath(a.bench)
    root = os.path.join(bench, a.root)
    targets_dir = os.path.join(bench, a.targets)
    out_dir = os.path.join(bench, a.out)
    root_rel, out_rel = a.root.replace("\\", "/").rstrip("/"), a.out.replace("\\", "/").rstrip("/")
    if not os.path.isdir(root):
        p.error(f"no such root {root}")

    warned = collections.OrderedDict()

    def warn(kind, msg):
        warned.setdefault(kind, []).append(msg)

    targets, finished, unfinished = scan(root, targets_dir, warn)
    global F
    F = Fonts()
    if F.paths is None:
        warn("no TrueType font found, using PIL's bitmap default (figures will look rough)", "")
    only = set(x.replace("\\", "/") for x in a.only) if a.only else None

    recs, drawn, skipped, unselected = [], collections.Counter(), collections.Counter(), collections.Counter()
    for key in sorted(finished, key=lambda k: (SUBSETS.index(k[0]), k[1])):
        rec, t, masks = score(key, finished[key], targets_dir, out_dir, warn)
        recs.append(rec)
        png = os.path.join(out_dir, rec["subset"], f"{rec['stem']}.png")
        if only is not None and f"{rec['subset']}/{rec['stem']}" not in only:
            unselected[rec["subset"]] += 1
            continue
        fresh = os.path.exists(png) and os.path.getmtime(png) > rec["rj_mtime"]
        if fresh and not a.force:
            skipped[rec["subset"]] += 1
            continue
        draw_figure(rec, t, masks, png)
        drawn[rec["subset"]] += 1
        if sum(drawn.values()) % 50 == 0:
            print(f"  ... {sum(drawn.values())} figures drawn ({time.time() - t0:.0f}s)", flush=True)

    # checks
    cells = [c for cs in finished.values() for c in cs.values()]
    diffs = [abs(c["iou"] - float(c["iou_reported"])) for c in cells if c.get("iou_reported") is not None]
    checks = [f"{len(cells)} cells in {len(finished)} finished targets; mask source: "
              + ", ".join(f"{k} {v}" for k, v in collections.Counter(c["src"] for c in cells).items())
              + f"; {sum(c['n_runs'] == 5 for c in cells)} cells hold 5 runs (runs 3-4 ignored).",
              f"best-of-3 ties at 4 dp: {sum(c['tied'] > 1 for c in cells)} cells, resolved to best_run when tied "
              f"({sum(c['tied'] > 1 and c['run'] == c['best_run'] for c in cells)}) else the first tied run.",
              f"recomputed IoU vs recorded iou_vs_original: max |diff| {max(diffs) if diffs else float('nan'):.2e}, "
              f"{sum(d > IOU_TOL for d in diffs)} cells beyond {IOU_TOL}."]
    deltas = [d for r in recs for d in list(r["added"].values()) + list(r["removed"].values())]
    checks.append(f"{len(deltas)} deltas; {sum(abs(d) < SPREAD and round(abs(d), 3) >= SPREAD for d in deltas)} "
                  f"print as {SPREAD:.3f} but are below the threshold (notable is decided on the unrounded value).")
    if only:
        for name in sorted(only - {f"{r['subset']}/{r['stem']}" for r in recs}):
            warn("--only names a target that is not finished (or does not exist)", name)
    xbad = []
    if a.check_csv:
        n_x, xbad = cross_check(a.check_csv, finished, root_rel)
        checks.append(f"cross-check vs {os.path.basename(a.check_csv)}: {n_x} cells compared, "
                      f"{n_x - len(xbad)} identical run and IoU (6 dp), {len(xbad)} differ"
                      + (": " + "; ".join(xbad[:5]) if xbad else "."))
    checks.append(f"unfinished targets: {len(unfinished)} of {len(targets)} (not in the tables)"
                  + (": " + ", ".join(f"{s}/{t}" for s, t in sorted(unfinished)) if unfinished else "."))

    os.makedirs(out_dir, exist_ok=True)
    write_csv(recs, os.path.join(out_dir, "cases.csv"), out_rel)
    write_html(recs, os.path.join(out_dir, "index.html"), root_rel, len(targets))
    write_md(recs, os.path.join(out_dir, "failures.md"), root_rel, len(targets), checks)

    # console
    total = collections.Counter(s for s, _ in targets)
    fin = collections.Counter(r["subset"] for r in recs)
    fails = collections.Counter(r["subset"] for r in recs if r["n_failures"] > 0)
    sel = only is not None
    print(f"{'subset':<14s} {'finished':>9s} {'with failure':>13s} {'drawn':>6s} {'up to date':>11s}"
          + (f" {'not in --only':>14s}" if sel else ""))
    for s in SUBSETS + ["total"]:
        c = (lambda ctr: sum(ctr.values())) if s == "total" else (lambda ctr, s=s: ctr[s])
        nf, nt = (len(recs), len(targets)) if s == "total" else (fin[s], total[s])
        print(f"{s:<14s} {nf:>4d}/{nt:<4d} {c(fails):>13d} {c(drawn):>6d} {c(skipped):>11d}"
              + (f" {c(unselected):>14d}" if sel else ""))
    print(f"\nunfinished targets: {len(unfinished)}")
    for (s, stem), miss in sorted(unfinished.items(), key=lambda kv: (SUBSETS.index(kv[0][0]), kv[0][1])):
        print(f"  {s}/{stem}: missing {', '.join(miss)}")
    print("\nchecks:")
    for c in checks[:-1]:
        print("  " + c)
    for kind, msgs in warned.items():
        print(f"[!] {kind}: {len(msgs)}")
        for m in msgs[:4]:
            print(f"      {m}")
        if len(msgs) > 4:
            print(f"      ... {len(msgs) - 4} more")
    print(f"\n-> {out_rel}/cases.csv, {out_rel}/index.html, {out_rel}/failures.md, "
          f"{out_rel}/<subset>/<stem>.png  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
