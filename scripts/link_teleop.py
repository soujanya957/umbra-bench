#!/usr/bin/env python3
"""Attach the teleop captures to the benchmark as a tagged reference set.

`metadata.jsonl` already carries a `shadows.teleop` slot per target — it was
scaffolded for exactly this and has been null since the dataset was built. A
capture is not a target; it is a shadow somebody produced FOR a target, so it
belongs in that slot rather than as a row of its own, and filling it is what makes
the pair addressable: same `sample_id`, one target and one human-made shadow.

Every filled row also gains the tag **SO101_fleet_teleop**, so the set can be
selected as a group without inferring it from a path.

    python scripts/link_teleop.py            # dry run
    python scripts/link_teleop.py --write
"""
from __future__ import annotations

import argparse, json, os, shutil, time

import _rescue_common as rc

TAG = "SO101_fleet_teleop"

# The shape the slot has on an unlinked row, and the shape it is reset to before
# being refilled -- so a re-run reproduces the previous result exactly.
BLANK = {"path": None, "captured_at": None, "operator": None, "n_arms": None,
         "notes": None}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--write", action="store_true")
a = ap.parse_args()

MP = os.path.join(rc.BENCH, "metadata.jsonl")
rows = [json.loads(l) for l in open(MP)]
index = {r["id"]: r for r in rows}
v2_of = {r["rescue"]["derived_from"]: r["id"] for r in rows if r.get("version") == 2}

man = json.load(open(os.path.join(rc.BENCH, "Teleops", "masks", "manifest.json")))
linked, retargeted, orphan, repeats = [], [], [], []
touched = set()
for rec in man["records"]:
    sid = rec.get("sample_id")
    note = None
    if sid and sid not in index and sid in v2_of:
        note = f"posed against {sid}, which v2 replaced"
        sid = v2_of[sid]                       # the capture still describes this shape
        retargeted.append(rec["capture"])
    if not sid or sid not in index:
        orphan.append((rec["capture"], rec.get("sample_id")))
        continue
    row = index[sid]
    cap = os.path.join("Teleops", rec["capture"])
    # The slot is rebuilt from the manifest on every run, never added to. Appending
    # would make the script's output depend on how many times it had been run
    # before, and a second --write would silently duplicate all 29 captures.
    if id(row) not in touched:
        touched.add(id(row))
        row.setdefault("shadows", {})["teleop"] = dict(BLANK)
    entry = {
        "path": rec.get("mask", cap + "_mask.png"),
        "photo": cap + ".png",
        "rectified": rec["rectified"],
        "captured_at": None,
        "operator": None,
        "n_arms": 3,
        "capture": rec["capture"],
        "part": rec.get("part"),
        "selected_for": rec.get("reason"),
        "binarised_by": rec.get("mask_backend", "auto-otsu"),
        "shape_frac": rec.get("shape_frac"),
        "notes": note,
    }
    # A target can be posed more than once. The slot is a single object, so the
    # first capture in the manifest stays primary and the rest hang off it --
    # assigning straight into the slot would have dropped every repeat silently.
    prev = row["shadows"]["teleop"]
    if prev.get("capture"):
        prev.setdefault("extra_captures", []).append(entry)
        repeats.append((rec["capture"], sid))
    else:
        row["shadows"]["teleop"] = entry
    row["tags"] = sorted(set(row.get("tags", [])) | {TAG})
    linked.append((rec["capture"], sid))

print(f"captures {len(man['records'])}  ->  linked {len(linked)}, orphan {len(orphan)}")
if retargeted:
    print(f"  re-pointed to v2 (posed against the retired v1): {len(retargeted)}")
    for c in retargeted:
        print(f"    {c}")
for c, sid in orphan:
    print(f"  orphan {c[:46]:46} sample_id {sid}")
if repeats:
    print(f"  repeat captures of an already-posed target: {len(repeats)}")
    for c, sid in repeats:
        print(f"    {c}  ->  {sid} (kept as extra_captures)")
print(f"\ncaptures linked: {len(linked)}   rows tagged {TAG}: "
      f"{len(linked) - len(repeats)}")

if not a.write:
    print("\ndry run — pass --write to update metadata.jsonl")
    raise SystemExit

stamp = time.strftime("%Y%m%d_%H%M%S")
shutil.copy2(MP, MP + f".bak_{stamp}")
with open(MP, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"\nwrote metadata.jsonl (backup .bak_{stamp})")
