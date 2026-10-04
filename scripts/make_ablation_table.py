#!/usr/bin/env python3
"""Emit the paper's ablation table from an ablation root (default optimized/ablation-probe10).

    python3 scripts/make_ablation_table.py > figures/tab_ablation.tex           # compact
    python3 scripts/make_ablation_table.py --check > figures/tab_ablation_check.tex   # checkmark form
    python3 scripts/make_ablation_table.py --root optimized/<study> --aimed-root optimized/<aimed-study>

One row per solver configuration; a check in a column means that component is
on, so each row's composition is explicit. Columns are in pipeline order
(Alg. 1). IoU is the mean, over whatever glyphs the root holds (the ten probe
glyphs in the default root), of best-of-seeds IoU vs the authored target; delta
is paired, vs the row above in the build-up block and vs UMBRA in the
leave-one-out blocks; renders are measured per solve.

Glyph counts go to stderr, so stdout is only the .tex. The main block (the
compact table's seven rows, blocks 1-2 of --check) is one glyph set: those
common to its 12 variants. Every other row or block is computed over that set
when its runs cover it, and otherwise over the part they cover, with the glyph
count written into its label or heading: a study still solving, or one reused
from probe10 (ablation_probe.sh does not solve full_long), then cannot pass for
the main block's glyphs in the PDF. Only a row whose variants are absent, or
that shares no glyph with the main block, is left out (noted on stderr).
"""
import argparse, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_ablation import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK, RM = r"\ck", r"\rmv"  # on / removed-from-UMBRA

# column order = pipeline order
COLS = ["assign", "aimed", "fwd", "bwd", "icp", "joint", "fd"]
HEAD = {"assign": "zone assign.", "aimed": "aimed start", "fwd": "arm-by-arm",
        "bwd": "backward", "icp": "ICP-guided restart", "joint": "joint stage", "fd": "FD polish"}

# variant -> set of components on. The manifold subspace of the joint stage
# (Nd >= 30 only) is not a column: full_no_manifold is reported in the text.
FULL3 = {"assign", "aimed", "fwd", "bwd", "icp", "joint", "fd"}
FULL5 = FULL3
VAR = {
    "flat":             set(),
    "fwd_joint":        {"fwd", "joint"},
    "plus_assign":      {"fwd", "joint", "assign"},
    "plus_voronoi":     {"fwd", "joint", "assign", "aimed"},
    "plus_backward":    {"fwd", "joint", "assign", "aimed", "bwd"},
    "plus_icp":         {"fwd", "joint", "assign", "aimed", "bwd", "icp"},
    "full":             FULL3,
    "full_no_assign":   FULL3 - {"assign"},
    "full_no_voronoi":  FULL3 - {"aimed"},
    "full_no_backward": FULL3 - {"bwd"},
    "full_no_icp":      FULL3 - {"icp"},
    "full_no_fd":       FULL3 - {"fd"},
    "full_long":        FULL3,
    "full_long_no_icp": FULL3 - {"icp"},
}
VAR5 = {"flat": set(), "full": FULL5, "full_no_icp": FULL5 - {"icp"}, "full_no_fd": FULL5 - {"fd"}}

BLOCKS = [  # (heading, n, [(variant, ref, label)])
    (r"Adding to a monolithic CMA-ES, $N{=}3$ (each row vs.\ the row above)", 3, [
        ("flat", None, "monolithic"), ("fwd_joint", "flat", ""), ("plus_assign", "fwd_joint", ""),
        ("plus_voronoi", "plus_assign", ""), ("plus_backward", "plus_voronoi", ""),
        ("plus_icp", "plus_backward", ""), ("full", "plus_icp", r"\textsc{umbra}")]),
    (r"Removing one component from \textsc{umbra}, $N{=}3$ (vs.\ \textsc{umbra})", 3, [
        ("full_no_assign", "full", ""), ("full_no_voronoi", "full", ""),
        ("full_no_backward", "full", ""), ("full_no_icp", "full", ""), ("full_no_fd", "full", "")]),
    (r"Same at $N{=}5$", 5, [
        ("flat", None, "monolithic"), ("full", "flat", r"\textsc{umbra}"),
        ("full_no_icp", "full", ""), ("full_no_fd", "full", "")]),
    (r"Sequence budget, $N{=}3$: restarts fire; $\rmv$ = random direction", 3, [
        ("full_long", None, r"\textsc{umbra}"), ("full_long_no_icp", "full_long", "")]),
]

# the main block: the compact table's seven rows are blocks 1 and 2 of --check
MAIN = sorted({x for _, _, rows in BLOCKS[:2] for v, ref, _ in rows for x in (v, ref) if x})

def note(msg):
    """Diagnostics go to stderr: stdout is redirected straight into the .tex."""
    print(msg, file=sys.stderr)

def common(data, variants):
    """Glyphs solved by every one of `variants` (all of which must be in `data`)."""
    return sorted(set.intersection(*(set(data[v]) for v in variants)))

def row(data, n, v, ref, label, glyphs):
    on = (VAR5 if n == 5 else VAR)[v]
    iou = st.mean(data[v][g][0] for g in glyphs)
    rend = st.mean(data[v][g][1] for g in glyphs) / 1000
    if ref:
        d = [data[v][g][0] - data[ref][g][0] for g in glyphs]
        m = st.mean(d)
        dl = "$0$" if abs(m) < 5e-4 else f"${m:+.3f}$"
    else:
        dl = ""
    # a removal row marks the one component it drops, so "UMBRA minus X" reads at a glance
    base = (VAR5 if n == 5 else VAR)[ref] if ref else set()
    marks = [CK if c in on else (RM if c in base else "") for c in COLS]
    return f"{label} & " + " & ".join(marks) + f" & ${iou:.3f}$ & {dl} & {rend:.1f}k \\\\"

<<<<<<< Updated upstream
def compact(d3, d5):
    """Four rows, two delta columns, for the system WITHOUT zone assignment or
    aimed start (nz_* variants; flat / fwd_joint / full_no_assign are the same
    configurations under their original names). `added`: each component on top
    of the rows above it, from a monolithic CMA-ES; `removed`: that component
    alone taken out of UMBRA. Rows whose runs are missing print as --."""
    g3 = sorted(set.intersection(*(set(c) for c in d3.values() if c)))
    g5 = sorted(set.intersection(*(set(c) for c in d5.values() if c))) if d5 else []
    def mean(d, v, gs): return st.mean(d[v][g][0] for g in gs)
    def rend(d, v, gs): return st.mean(d[v][g][1] for g in gs) / 1000
    def delta(d, v, ref, gs):
        x = [d[v][g][0] - d[ref][g][0] for g in gs]
        return st.mean(x), st.stdev(x) / len(x) ** 0.5
    def fmt(ms):  # bold when the paired mean is more than two s.e.m. from zero
        m, sem = ms
        if abs(m) < 5e-4: return "$0$"
        return f"$\\mathbf{{{m:+.3f}}}$" if abs(m) > 2 * sem else f"${m:+.3f}$"
    def have(d, *vs): return all(v in d and len(set(d[v]) & set(g3 if d is d3 else g5)) == 10 for v in vs)
    UMBRA = "full_no_assign"   # == nz_full at N=3
    ROWS = [  # number, label, (added variant, its predecessor), removed variant
=======
def compact(d3, d5, da, root):
    """Seven rows, two delta columns. `added`: each component on top of the rows
    above it, starting from a monolithic CMA-ES; `removed`: that component alone
    taken out of UMBRA. The N=5 and sequence-budget results are one footer row each.
    `da` is the aimed-start study at N=3, for the quarter-budget footer rows."""
    ROWS = [  # number, label, (added variant, its predecessor), (removed variant) or None
>>>>>>> Stashed changes
        ("1", "arm-by-arm sweep, joint stage", ("fwd_joint", "flat"), None),
        ("2", "backward sweep", ("nz_plus_backward", "fwd_joint"), "nz_full_no_backward"),
        ("3", "ICP-guided restart", ("nz_plus_icp", "nz_plus_backward"), "nz_full_no_icp"),
        ("4", "FD polish", (UMBRA, "nz_plus_icp"), "nz_full_no_fd"),
    ]
    # Glyphs common to the variants the main block shows, not to every N=3 variant
    # in the root: the sequence-budget pair comes from another script and may hold
    # fewer glyphs, which must not silently shrink the whole table.
    main = {"flat", "full"} | {x for _, _, pair, rm in ROWS for x in (*pair, rm) if x}
    assert main == set(MAIN), "ROWS and BLOCKS[:2] must name the same main block"
    missing = sorted(main - set(d3))
    if missing:
        sys.exit(f"{root}: no N=3 results for {', '.join(missing)}")
    g3 = common(d3, main)
    every = set.union(*(set(d3[v]) for v in main))
    short = [f"{v} {len(d3[v])}" for v in sorted(main) if set(d3[v]) != every]
    note(f"N=3 main block: {len(g3)} glyphs common to its {len(main)} variants"
         + (f" (of {len(every)} seen; short: {', '.join(short)})" if short else ""))
    if not g3:
        sys.exit(f"{root}: no glyph is solved by every N=3 variant of the main block")
    mean = lambda d, v, gs: st.mean(d[v][g][0] for g in gs)
    rend = lambda d, v, gs: st.mean(d[v][g][1] for g in gs) / 1000
    def delta(d, v, ref, gs):
        x = [d[v][g][0] - d[ref][g][0] for g in gs]
        # a single glyph has no spread, so it is never bold
        return st.mean(x), (st.stdev(x) / len(x) ** 0.5 if len(x) > 1 else float("inf"))
    def fmt(ms):  # bold when the paired mean is more than two s.e.m. from zero
        m, sem = ms
        if abs(m) < 5e-4: return "$0$"
        return f"$\\mathbf{{{m:+.3f}}}$" if abs(m) > 2 * sem else f"${m:+.3f}$"
    def cover(d, pair, what):
        """Glyphs a footer row is computed over, or None to leave the row out.
        Never a glyph outside the main block, so a footer cannot average over a
        different set of the same size; fewer glyphs than the main block show in
        the row's label via tag()."""
        absent = [v for v in pair if v not in d]
        if absent:
            note(f"{what}: left out, no results for {', '.join(absent)}")
            return None
        gs = sorted(set(common(d, pair)) & set(g3))
        if len(gs) == len(g3):
            note(f"{what}: {len(g3)} glyphs")
            return g3
        if not gs:
            note(f"{what}: left out, shares no glyph with the main block's {len(g3)}")
            return None
        note(f"WARNING {what}: over {len(gs)} of the main block's {len(g3)} glyphs (count in its label)")
        return gs
    tag = lambda gs: f" ({len(gs)} glyphs)" if len(gs) < len(g3) else ""
    print(r"\begin{tabular}{@{}rlrrr@{}}")
    print(r"\toprule")
    print(r" & & \multicolumn{2}{c}{$\Delta$ IoU when} & \\")
    print(r"\cmidrule(lr){3-4}")
    print(r" & component & added & removed & renders \\")
    print(r"\midrule")
    print(f" & monolithic CMA-ES, IoU ${mean(d3,'flat',g3):.3f}$ & & & {rend(d3,'flat',g3):.1f}k \\\\")
    for num, lab, (v, ref), rm in ROWS:
<<<<<<< Updated upstream
        a = fmt(delta(d3, v, ref, g3)) if have(d3, v, ref) else "--"
        r = (fmt(delta(d3, rm, UMBRA, g3)) if have(d3, rm, UMBRA) else "--") if rm else ""
        rd = f"{rend(d3,v,g3):.1f}k" if have(d3, v) else "--"
        print(f"{num} & {lab} & {a} & {r} & {rd} \\\\")
    print(f" & \\textsc{{umbra}}, IoU ${mean(d3,UMBRA,g3):.3f}$ & & & {rend(d3,UMBRA,g3):.1f}k \\\\")
    print(r"\midrule")
    if have(d5, "flat", "nz_full"):
        print(f" & $N{{=}}5$: monolithic ${mean(d5,'flat',g5):.3f}$, \\textsc{{umbra}} ${mean(d5,'nz_full',g5):.3f}$ & {fmt(delta(d5,'nz_full','flat',g5))} & & {rend(d5,'nz_full',g5):.1f}k \\\\")
    else:
        print(f" & $N{{=}}5$: monolithic ${mean(d5,'flat',g5):.3f}$, \\textsc{{umbra}} -- & -- & & -- \\\\")
    if have(d3, "nz_full_long", "nz_full_long_no_icp"):
        gl = sorted(set(d3["nz_full_long"]) & set(d3["nz_full_long_no_icp"]))
        print(f"3 & restart direction random, sequence budget & & {fmt(delta(d3,'nz_full_long_no_icp','nz_full_long',gl))} & {rend(d3,'nz_full_long',gl):.0f}k \\\\")
    else:
        print(r"3 & restart direction random, sequence budget & & -- & -- \\")
=======
        a = fmt(delta(d3, v, ref, g3))
        r = fmt(delta(d3, rm, "full", g3)) if rm else ""
        print(f"{num} & {lab} & {a} & {r} & {rend(d3,v,g3):.1f}k \\\\")
    print(f" & \\textsc{{umbra}}, IoU ${mean(d3,'full',g3):.3f}$ & & & {rend(d3,'full',g3):.1f}k \\\\")
    foot = []
    gs = cover(d5, ("flat", "full"), "N=5 row")
    if gs:
        foot.append(f" & $N{{=}}5${tag(gs)}: monolithic ${mean(d5,'flat',gs):.3f}$, \\textsc{{umbra}} ${mean(d5,'full',gs):.3f}$ & {fmt(delta(d5,'full','flat',gs))} & & {rend(d5,'full',gs):.1f}k \\\\")
    # aimed start at a quarter of the budget (aimed_start_probe.sh, tiny, N=3)
    # no zones means nothing to aim at, so no_zones-tiny removes 2 and 3 together
    for v, num, lab in (("no_zones-tiny", "2, 3", "zone assignment and aimed start removed"),
                        ("no_aimed-tiny", "3", "aimed start removed")):
        gs = cover(da, ("full-tiny", v), f"{v} row")
        if gs:
            foot.append(f"{num} & {lab}, quarter budget{tag(gs)} & & {fmt(delta(da,v,'full-tiny',gs))} & {rend(da,'full-tiny',gs):.1f}k \\\\")
    gs = cover(d3, ("full_long", "full_long_no_icp"), "sequence-budget row")
    if gs:
        foot.append(f"5 & restart direction random, sequence budget{tag(gs)} & & {fmt(delta(d3,'full_long_no_icp','full_long',gs))} & {rend(d3,'full_long',gs):.0f}k \\\\")
    if foot:
        print(r"\midrule")
        print("\n".join(foot))
>>>>>>> Stashed changes
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="checkmark form instead of the compact table")
    ap.add_argument("--root", default=os.path.join(ROOT, "optimized", "ablation-probe10"),
                    help="ablation study, <variant>-n<N>/<subset>/<stem>/results.json (default: %(default)s)")
    ap.add_argument("--aimed-root", default=os.path.join(ROOT, "optimized", "aimed-start-probe10"),
                    help="aimed-start study, for the compact table's quarter-budget rows (default: %(default)s)")
    a = ap.parse_args()
    d3 = load(a.root, 3)
    d5 = load(a.root, 5)
    if not d3 and not d5:  # else --check would write an empty tabular into the .tex
        sys.exit(f"no results under {a.root}")
    if not a.check:
        return compact(d3, d5, load(a.aimed_root, 3), a.root)
    ncol = 1 + len(COLS) + 3
    hdr = " & ".join(r"\rot{" + f"{i+1}. {HEAD[c]}" + "}" for i, c in enumerate(COLS))
    print(r"\begin{tabular}{@{}l" + "c" * len(COLS) + r"rrr@{}}")
    print(r"\toprule")
    print(f" & {hdr} & IoU & $\\Delta$ & renders \\\\")
    print(r"\midrule")
    # One glyph set per block, so a block's IoU column compares its rows. Blocks 1
    # and 2 share `full`, so both use the main block's set; later blocks use the
    # part of it they cover, and name the count in their heading when it is less.
    g3 = common(d3, MAIN) if all(v in d3 for v in MAIN) else []
    shown = 0
    for bi, (head, n, rows) in enumerate(BLOCKS):
        data = d5 if n == 5 else d3
        variants = sorted({x for v, ref, _ in rows for x in (v, ref) if x})
        absent = [v for v in variants if v not in data]
        if absent:
            note(f"block {bi + 1} (N={n}): left out, no results for {', '.join(absent)}")
            continue
        # without a complete main block (a block-2 variant still to solve) block 1
        # falls back to its own glyphs, and its heading then carries the count
        gs = g3 if (bi < 2 and g3) else common(data, variants)
        if bi >= 2 and g3:
            gs = sorted(set(gs) & set(g3))
        if not gs:
            note(f"block {bi + 1} (N={n}): left out, no glyph is solved by all its variants"
                 + (" and the main block" if g3 else ""))
            continue
        if len(gs) < len(g3) or not g3:
            head += f" ({len(gs)} glyphs)"
            note(f"WARNING block {bi + 1} (N={n}): over {len(gs)} glyphs, not the main block's {len(g3)} (count in its heading)")
        else:
            note(f"block {bi + 1} (N={n}): {len(gs)} glyphs")
        if shown:
            print(r"\addlinespace[2pt]")
        shown += 1
        # zero-width box: a long heading overhangs instead of stretching the last column
        print(r"\multicolumn{" + str(ncol) + r"}{@{}l@{}}{\makebox[0pt][l]{\itshape " + head + r"}} \\")
        for v, ref, label in rows:
            print(row(data, n, v, ref, label, gs))
    print(r"\bottomrule")
    print(r"\end{tabular}")

if __name__ == "__main__":
    main()
