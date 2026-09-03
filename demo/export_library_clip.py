#!/usr/bin/env python3
"""export_library_clip.py — one library shadow, straight onto the robot.

    python demo/export_library_clip.py --library-id letters_upper_K_dejavusans-bold
    python demo/export_library_clip.py --library-id id1 --library-id id2

The packages only carry the letters a project actually cast, but every solved
target in the benchmark is a deployable single pose. This writes the same
single-pose choreography pack.py would (30 Hz envelope, held head/tail, the
display body, the step and joint-stop audits) DIRECTLY into the robot UI's
choreographies/ folder, so the shadow shows up in Play's library on the next
refresh — no package build required.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent
DEFAULT_DEST = BENCH.parent / "fleet-shadow-art" / "choreographies"

_spec = importlib.util.spec_from_file_location("pack", ROOT / "pack.py")
pack = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, [sys.argv[0]]      # pack.py parses argv on import guard only
_spec.loader.exec_module(pack)
sys.argv = _argv


def solve_pose(library_id: str, sweep: str) -> np.ndarray:
    rows = {json.loads(l)["id"]: json.loads(l)
            for l in open(BENCH / "metadata.jsonl", encoding="utf-8")}
    if library_id not in rows:
        sys.exit(f"[!] {library_id}: not in metadata.jsonl")
    rec = rows[library_id]
    tdir = (BENCH / "optimized" / sweep / rec["subset"]
            / Path(rec["target"]).stem)
    rj = tdir / "results.json"
    if not rj.exists():
        sys.exit(f"[!] {library_id}: no solve in {sweep}")
    res = json.loads(rj.read_text(encoding="utf-8"))
    return np.asarray(res["runs"][res["best_run"]]["q_rad"], dtype=float).ravel()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library-id", action="append", required=True)
    ap.add_argument("--sweep", default="big-budget-grounded")
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    a = ap.parse_args()

    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for lid in a.library_id:
        q = solve_pose(lid, a.sweep)
        pack.write_choreo(dest, lid, 5.0, [q])
        print(f"  {lid} -> {dest / (lid + '.json')}")
    print(f"{len(a.library_id)} clip(s); refresh Play's library to deploy")


if __name__ == "__main__":
    main()
