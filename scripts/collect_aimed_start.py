#!/usr/bin/env python3
"""Summarise optimized/aimed-start-probe10: full vs no_aimed per (budget, N)."""
import os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_ablation import SPREAD, load
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.join(ROOT, "optimized", "aimed-start-probe10")
for n in (3, 5, 7):
    d = load(root, n)
    for b in ("tiny", "small", "small-init4", "big"):
        f, g = f"full-{b}", f"no_aimed-{b}"
        if f not in d or g not in d:
            continue
        gs = sorted(set(d[f]) & set(d[g]))
        if len(gs) < 10:
            print(f"N={n} {b}: partial ({len(gs)} glyphs)"); continue
        dl = [d[f][x][0] - d[g][x][0] for x in gs]
        print(f"N={n} {b:12s} with {st.mean(d[f][x][0] for x in gs):.3f}  without {st.mean(d[g][x][0] for x in gs):.3f}"
              f"  delta {st.mean(dl):+.3f} ± {st.stdev(dl)/len(dl)**.5:.3f}"
              f"  >+{SPREAD}: {sum(x > SPREAD for x in dl)}  <-{SPREAD}: {sum(x < -SPREAD for x in dl)}  >0: {sum(x > 0 for x in dl)}"
              f"  renders {st.mean(d[f][x][1] for x in gs):.0f} vs {st.mean(d[g][x][1] for x in gs):.0f}")
        print("   " + "  ".join(f"{x.split('_')[-2]}:{v:+.3f}" for x, v in zip(gs, dl)))
