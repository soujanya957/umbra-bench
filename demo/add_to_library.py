#!/usr/bin/env python3
"""add_to_library.py — a static element becomes a named library shape.

    python demo/add_to_library.py --sequence demo_01_scene_06_A
    python demo/add_to_library.py --from-routing        # every static-lane clip

The static lane's end state is not "a clip solved cheaply", it is "a shape the
library knows by name": once it sits in `targets/demo/` and `metadata.jsonl`,
every future show pulls it with `pack.py --library <id>` instead of touching
the footage again, and the atlas shows it like any other target.

What one import does:

  1. pick the clip's MEDOID frame -- the frame with the highest mean IoU to
     the others, i.e. the most representative pose, not just f00
  2. write it to `targets/demo/<class>_<source>.png` -- class first, because
     `build_metadata.py` reads the class from the stem's first token
  3. re-run the library chain: normalize (pads to 512 if needed) -> ground ->
     build_metadata (metadata.jsonl backed up first; the script writes no
     backup of its own)

Solving the new rows is a sweep launch and CLIP is a full re-run (the class
list changes), so both are printed as next commands rather than run here.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent


def load(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L")) < 128


def medoid(masks: list[np.ndarray]) -> int:
    def iou(a, b):
        u = (a | b).sum()
        return (a & b).sum() / u if u else 1.0
    scores = [np.mean([iou(m, o) for j, o in enumerate(masks) if j != i])
              for i, m in enumerate(masks)]
    return int(np.argmax(scores))


def import_one(rec: dict, force: bool) -> str | None:
    sid = rec["id"]
    masks = [load(BENCH / f) for f in rec["frames"]]
    i = medoid(masks)
    cls = str(rec.get("class") or sid.split("_")[0])
    src = re.sub(r"[^A-Za-z0-9]", "", sid)
    name = f"{cls}_{src}"
    dst = BENCH / "targets" / "demo" / f"{name}.png"
    if dst.exists() and not force:
        print(f"  {sid}: targets/demo/{name}.png already exists, skipped")
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BENCH / rec["frames"][i], dst)
    print(f"  {sid}: frame f{i:02d} (medoid) -> targets/demo/{name}.png"
          f"  class {cls}")
    return f"demo_{name}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", action="append", default=[])
    ap.add_argument("--from-routing", action="store_true",
                    help="import every clip demo/out/motion_routing.json "
                         "put in the static lane")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-metadata", action="store_true",
                    help="stage the PNGs only; skip normalize/ground/metadata")
    a = ap.parse_args()

    recs = {json.loads(l)["id"]: json.loads(l)
            for l in open(BENCH / "sequences.jsonl", encoding="utf-8")}
    ids = list(a.sequence)
    if a.from_routing:
        routing = json.loads((ROOT / "out" / "motion_routing.json")
                             .read_text(encoding="utf-8"))["routing"]
        ids += [k for k, v in routing.items() if v["lane"] == "static"]
    ids = sorted(set(ids))
    if not ids:
        ap.error("give --sequence or --from-routing")

    new = []
    for sid in ids:
        if sid not in recs:
            sys.exit(f"[!] {sid} not in sequences.jsonl")
        n = import_one(recs[sid], a.force)
        if n:
            new.append(n)
    if not new:
        print("nothing new to import")
        return
    if a.no_metadata:
        print("\nstaged only; the library chain still needs running")
        return

    py = sys.executable
    bak = BENCH / f"metadata.jsonl.bak_library_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copyfile(BENCH / "metadata.jsonl", bak)
    print(f"\nmetadata backed up -> {bak.name}")
    for cmd in (
        [py, "scripts/normalize_targets.py", "--subsets", "demo"],
        [py, "scripts/ground_targets.py", "--subsets", "demo"],
        [py, "scripts/build_metadata.py"],
    ):
        print("$", " ".join(cmd[1:]))
        r = subprocess.run(cmd, cwd=BENCH)
        if r.returncode:
            sys.exit(f"[!] {cmd[1]} failed; metadata backup is {bak.name}")

    print(f"\n{len(new)} shape(s) in the library: {', '.join(new)}")
    print("next (not run here -- a sweep launch and a full CLIP re-run):")
    print("  python scripts/run_base_optimizer.py --targets-dir targets_grounded"
          " --subsets demo ...   # settings per optimized/<sweep>/BUDGET.md")
    print("  python scripts/clip_eval_dataset.py   # class list changed")
    print("  python scripts/_build_browser_payload.py && python atlas/build_atlas.py")


if __name__ == "__main__":
    main()
