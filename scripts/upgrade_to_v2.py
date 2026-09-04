#!/usr/bin/env python3
"""Replace sub-floor targets with their rescued version instead of keeping both.

`promote_targets.py` writes a rescued sibling and keeps the original indexed, so
each pair can be compared. This collapses that: the rescued image becomes the
target, and the original leaves the index.

Two choices worth stating, because both are easy to get wrong:

**The v2 target gets a NEW id (`<id>_v2`), not the original's.** Reusing the id
would be tidier to read and quietly wrong: the sweep tables join on `sample_id`,
so 3,276 existing rows would attach themselves to an image that is not the one
they were measured on. A new id makes the old rows fail to join, which is the
correct outcome — they describe a target that is no longer in the benchmark.

**Nothing is deleted.** v1 rows move to `dropped.jsonl` and v1 PNGs stay where
they are. Dropping a target from a benchmark and destroying the evidence are
different acts; only the first is intended here, and the matched-pair comparison
stays reconstructable from what is left on disk.

    python scripts/upgrade_to_v2.py            # dry run
    python scripts/upgrade_to_v2.py --write
"""
from __future__ import annotations
import argparse, json, os, shutil, time
import _rescue_common as rc

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--write", action="store_true")
a = ap.parse_args()

MP = os.path.join(rc.BENCH, "metadata.jsonl")
rows = [json.loads(l) for l in open(MP)]
resc = [r for r in rows if "rescue" in r]
by_src = {r["rescue"]["derived_from"]: r for r in resc}
subfloor = [r for r in rows if r.get("below_floor") and "rescue" not in r]
no_rescue = [r for r in subfloor if r["id"] not in by_src]

keep, dropped, renames = [], [], []
for r in rows:
    if "rescue" in r:
        old_rel = r["target"]
        src_row = next(x for x in rows if x["id"] == r["rescue"]["derived_from"])
        base, ext = os.path.splitext(os.path.basename(src_row["target"]))   # ext from the FILE
        assert ext, f"no extension on {src_row['target']}"
        new_rel = os.path.join(os.path.dirname(src_row["target"]), base + "_v2" + ext)
        r = dict(r)
        r["id"] = src_row["id"] + "_v2"
        r["target"] = new_rel
        r["version"] = 2
        r["rescue"] = dict(r["rescue"], v1_target=src_row["target"])
        r.pop("below_floor", None)
        renames.append((old_rel, new_rel))
        keep.append(r)
    elif r.get("below_floor"):
        dropped.append(dict(r, dropped_reason="below the rig stroke floor; "
                            + ("replaced by " + by_src[r["id"]]["rescue"]["derived_from"] + "_v2"
                               if r["id"] in by_src else "no rescue applied")))
    else:
        keep.append(r)

print(f"metadata {len(rows)} -> {len(keep)}   (dropped {len(dropped)}: "
      f"{len(by_src)} replaced by v2, {len(no_rescue)} unrescuable)")
print(f"files to rename: {len(renames)}   nothing deleted")
if not a.write:
    print("\ndry run — pass --write to apply")
    raise SystemExit

stamp = time.strftime("%Y%m%d_%H%M%S")
shutil.copy2(MP, MP + f".bak_{stamp}")
for old, new in renames:
    o, n = os.path.join(rc.BENCH, old), os.path.join(rc.BENCH, new)
    if os.path.exists(o):
        os.rename(o, n)
with open(MP, "w") as f:
    for r in keep:
        f.write(json.dumps(r) + "\n")
dp = os.path.join(rc.BENCH, "dropped.jsonl")
prev = {json.loads(l)["id"] for l in open(dp)} if os.path.exists(dp) else set()
with open(dp, "a") as f:
    for r in dropped:
        if r["id"] not in prev:
            f.write(json.dumps(r) + "\n")
print(f"\nwrote metadata.jsonl ({len(keep)} rows, backup .bak_{stamp}), "
      f"dropped.jsonl (+{len(dropped)}), renamed {len(renames)} PNGs")
