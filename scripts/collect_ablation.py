#!/usr/bin/env python3
"""Summarise optimized/ablation-probe10 into the paper's ablation table.

    python3 scripts/collect_ablation.py optimized/ablation-probe10 [--n 3] [--tex]

Per variant: mean over glyphs of best-of-seeds IoU vs the authored target,
paired delta against a reference row, how many glyphs the change helped/hurt by
more than SPREAD, and mean measured renders per solve. Build-up rows are
referenced to the row above; leave-one-out rows to `full`.
"""
import glob, json, os, statistics as st, sys

SPREAD = 0.02  # within-target seed spread on the grounded glyphs (~0.023)

ORDER = [  # (variant, label, reference)
    ("flat",             "monolithic CMA-ES, all $Nd$ joints",     None),
    ("fwd_joint",        "+ per-arm forward sweep + joint stage",   "flat"),
    ("plus_assign",      "+ image-space assignment",                "fwd_joint"),
    ("plus_voronoi",     "+ Voronoi warm start",                    "plus_assign"),
    ("plus_backward",    "+ backward sweep",                        "plus_voronoi"),
    ("plus_icp",         "+ ICP restarts",                          "plus_backward"),
    ("full",             r"+ FD polish (= \textsc{umbra})",         "plus_icp"),
    ("full_no_assign",   r"\textsc{umbra} $-$ assignment",          "full"),
    ("full_no_voronoi",  r"\textsc{umbra} $-$ Voronoi warm start",  "full"),
    ("full_no_backward", r"\textsc{umbra} $-$ backward sweep",      "full"),
    ("full_no_icp",      r"\textsc{umbra} $-$ ICP restarts",        "full"),
    ("full_no_fd",       r"\textsc{umbra} $-$ FD polish",           "full"),
    ("full_no_manifold", r"\textsc{umbra} $-$ manifold joint stage","full"),
]


def load(root, n):
    """variant -> {glyph_id: (best_iou_vs_original, mean_renders)}"""
    out = {}
    for d in sorted(glob.glob(os.path.join(root, f"*-n{n}"))):
        v = os.path.basename(d)[: -len(f"-n{n}")]
        cells = {}
        for f in glob.glob(os.path.join(d, "*", "*", "results.json")):
            r = json.load(open(f))
            ev = [x["n_evals"] for x in r["runs"] if x.get("n_evals")]
            cells[r["id"]] = (float(r["best_iou_vs_original"]), st.mean(ev) if ev else float("nan"))
        if cells:
            out[v] = cells
    return out


def main():
    root = sys.argv[1]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 3
    tex = "--tex" in sys.argv
    data = load(root, n)
    if not data:
        sys.exit(f"no results under {root} for N={n}")
    glyphs = sorted(set.intersection(*(set(c) for c in data.values())))
    print(f"# N={n}: {len(glyphs)} glyphs common to all {len(data)} variants\n")
    hdr = ("variant", "IoU", "delta", "helped", "hurt", "renders")
    if not tex:
        print("{:<38s} {:>6s} {:>7s} {:>6s} {:>5s} {:>8s}".format(*hdr))
    for v, label, ref in ORDER:
        if v not in data:
            continue
        ious = [data[v][g][0] for g in glyphs]
        rend = st.mean(data[v][g][1] for g in glyphs)
        mean = st.mean(ious)
        if ref and ref in data:
            d = [data[v][g][0] - data[ref][g][0] for g in glyphs]
            delta, helped, hurt = st.mean(d), sum(x > SPREAD for x in d), sum(x < -SPREAD for x in d)
            per = "  ".join(f"{g.split('_')[-2]}:{x:+.3f}" for g, x in zip(glyphs, d))
        else:
            delta = helped = hurt = None; per = ""
        if tex:
            dl = "---" if delta is None else f"${delta:+.3f}$"
            hl = "---" if helped is None else f"{helped}/{len(glyphs)}"
            print(f" & {label} & ${mean:.3f}$ & {dl} & {hl} & {rend/1000:.1f}k \\\\")
        else:
            dl = "" if delta is None else f"{delta:+.3f}"
            hl = "" if helped is None else str(helped)
            ht = "" if hurt is None else str(hurt)
            print(f"{label:<38s} {mean:6.3f} {dl:>7s} {hl:>6s} {ht:>5s} {rend:8.0f}")
            if per:
                print(f"{'':<38s} {per}")


if __name__ == "__main__":
    main()
