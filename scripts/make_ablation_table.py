#!/usr/bin/env python3
"""Emit the paper's ablation table (checkmark form) from optimized/ablation-probe10.

    python3 scripts/make_ablation_table.py > figures/tab_ablation.tex           # compact
    python3 scripts/make_ablation_table.py --check > figures/tab_ablation_check.tex   # checkmark form

One row per solver configuration; a check in a column means that component is
on, so each row's composition is explicit. Columns are in pipeline order
(Alg. 1). IoU is the mean over the ten probe glyphs of best-of-seeds IoU vs
the authored target; delta is paired, vs the row above in the build-up block
and vs UMBRA in the leave-one-out blocks; renders are measured per solve.
"""
import os, statistics as st, sys
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

def row(data, n, v, ref, label):
    on = (VAR5 if n == 5 else VAR)[v]
    glyphs = sorted(set(data[v]) & (set(data[ref]) if ref else set(data[v])))
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

def compact(d3, d5):
    """Seven rows, two delta columns. `added`: each component on top of the rows
    above it, starting from a monolithic CMA-ES; `removed`: that component alone
    taken out of UMBRA. The N=5 and sequence-budget results are one footer row each."""
    g3 = sorted(set.intersection(*(set(c) for c in d3.values() if c)))
    g5 = sorted(set.intersection(*(set(c) for c in d5.values() if c)))
    mean = lambda d, v, gs: st.mean(d[v][g][0] for g in gs)
    rend = lambda d, v, gs: st.mean(d[v][g][1] for g in gs) / 1000
    delta = lambda d, v, ref, gs: st.mean(d[v][g][0] - d[ref][g][0] for g in gs)
    fmt = lambda x: "$0$" if abs(x) < 5e-4 else f"${x:+.3f}$"
    ROWS = [  # number, label, (added variant, its predecessor), (removed variant) or None
        ("3, 6", "arm-by-arm sweep, joint stage", ("fwd_joint", "flat"), None),
        ("1", "zone assignment", ("plus_assign", "fwd_joint"), "full_no_assign"),
        ("2", "aimed start", ("plus_voronoi", "plus_assign"), "full_no_voronoi"),
        ("4", "backward sweep", ("plus_backward", "plus_voronoi"), "full_no_backward"),
        ("5", "ICP-guided restart", ("plus_icp", "plus_backward"), "full_no_icp"),
        ("7", "FD polish", ("full", "plus_icp"), "full_no_fd"),
    ]
    print(r"\begin{tabular}{@{}rlrrr@{}}")
    print(r"\toprule")
    print(r" & & \multicolumn{2}{c}{$\Delta$ IoU when} & \\")
    print(r"\cmidrule(lr){3-4}")
    print(r" & component & added & removed & renders \\")
    print(r"\midrule")
    print(f" & monolithic CMA-ES, IoU ${mean(d3,'flat',g3):.3f}$ & & & {rend(d3,'flat',g3):.1f}k \\\\")
    for num, lab, (v, ref), rm in ROWS:
        a = fmt(delta(d3, v, ref, g3))
        r = fmt(delta(d3, rm, "full", g3)) if rm else ""
        print(f"{num} & {lab} & {a} & {r} & {rend(d3,v,g3):.1f}k \\\\")
    print(f" & \\textsc{{umbra}}, IoU ${mean(d3,'full',g3):.3f}$ & & & {rend(d3,'full',g3):.1f}k \\\\")
    print(r"\midrule")
    print(f" & $N{{=}}5$: monolithic ${mean(d5,'flat',g5):.3f}$, \\textsc{{umbra}} ${mean(d5,'full',g5):.3f}$ & & & {rend(d5,'full',g5):.1f}k \\\\")
    gl = sorted(set(d3["full_long"]) & set(d3["full_long_no_icp"]))
    print(f"5 & sequence budget, restart direction random & & {fmt(delta(d3,'full_long_no_icp','full_long',gl))} & {rend(d3,'full_long',gl):.0f}k \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    d3 = load(os.path.join(ROOT, "optimized", "ablation-probe10"), 3)
    d5 = load(os.path.join(ROOT, "optimized", "ablation-probe10"), 5)
    if "--check" not in sys.argv:
        return compact(d3, d5)
    ncol = 1 + len(COLS) + 3
    hdr = " & ".join(r"\rot{" + f"{i+1}. {HEAD[c]}" + "}" for i, c in enumerate(COLS))
    print(r"\begin{tabular}{@{}l" + "c" * len(COLS) + r"rrr@{}}")
    print(r"\toprule")
    print(f" & {hdr} & IoU & $\\Delta$ & renders \\\\")
    print(r"\midrule")
    for bi, (head, n, rows) in enumerate(BLOCKS):
        if bi:
            print(r"\addlinespace[2pt]")
        # zero-width box: a long heading overhangs instead of stretching the last column
        print(r"\multicolumn{" + str(ncol) + r"}{@{}l@{}}{\makebox[0pt][l]{\itshape " + head + r"}} \\")
        data = d5 if n == 5 else d3
        for v, ref, label in rows:
            print(row(data, n, v, ref, label))
    print(r"\bottomrule")
    print(r"\end{tabular}")

if __name__ == "__main__":
    main()
