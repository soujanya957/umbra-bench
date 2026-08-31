#!/usr/bin/env python3
"""Move dropped targets and their stale results out of the scanned trees.

Removing a row from `metadata.jsonl` is not enough to remove a target from the
benchmark. Neither of the two scripts that matter reads that file:
`run_base_optimizer.py` enumerates `targets/<subset>/*.png` with os.listdir, and
`compute_metrics.py` walks the sweep's own result directories. A dropped target
whose PNG is still on disk therefore keeps getting solved, and its old results
keep landing in `master_table.csv` — silently, and looking exactly like data.

So the drop has to happen on disk. Files are MOVED, never deleted: everything
lands under `dropped/` with its structure intact, so the v1-vs-v2 comparison
stays reconstructable and nothing about the decision is irreversible.

    dropped/targets/<subset>/<stem>.png
    dropped/optimized/<sweep>/<subset>/<stem>/…

    python scripts/quarantine_dropped.py           # dry run
    python scripts/quarantine_dropped.py --write
"""
from __future__ import annotations
import argparse, json, os, shutil
import _rescue_common as rc

SWEEPS = ["base-optimizer/big-budget", "base-optimizer/small-budget",
          "big-budget-fitted", "small-budget-fitted"]
NON_SUBSET = {"sheets", "reels", "phrases", "best-runs", "gallery"}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--write", action="store_true")
a = ap.parse_args()

B = rc.BENCH
rows = [json.loads(l) for l in open(os.path.join(B, "dropped.jsonl"))]
live = {json.loads(l)["target"] for l in open(os.path.join(B, "metadata.jsonl"))}

moves = []
for r in rows:
    stem = os.path.splitext(os.path.basename(r["target"]))[0]
    src = os.path.join(B, r["target"])
    if r["target"] in live:
        raise SystemExit(f"{r['target']} is still referenced by metadata.jsonl — refusing")
    if os.path.exists(src):
        moves.append((src, os.path.join(B, "dropped", r["target"]), r["id"], "target"))
    for sw in SWEEPS:
        d = os.path.join(B, "optimized", sw, r["subset"], stem)
        if os.path.isdir(d):
            moves.append((d, os.path.join(B, "dropped", "optimized", sw, r["subset"], stem),
                          r["id"], sw))

from collections import Counter
print(f"{len(rows)} dropped rows -> {len(moves)} paths to move")
for k, n in Counter(m[3] for m in moves).most_common():
    print(f"  {k:34} {n}")
before = len([p for p in os.listdir(os.path.join(B, "targets")) if not p.startswith("_")])
n_png = sum(len([f for f in os.listdir(os.path.join(B, "targets", s)) if f.endswith(".png")])
            for s in os.listdir(os.path.join(B, "targets"))
            if os.path.isdir(os.path.join(B, "targets", s)))
print(f"\ntargets/*.png now {n_png} -> {n_png - sum(1 for m in moves if m[3] == 'target')} "
      f"(metadata.jsonl has {len(live)})")

if not a.write:
    print("\ndry run — pass --write to move")
    raise SystemExit

for src, dst, _id, _k in moves:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
for r in rows:
    r["quarantined_target"] = os.path.join("dropped", r["target"])
with open(os.path.join(B, "dropped.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"\nmoved {len(moves)} paths under dropped/ — nothing deleted")
