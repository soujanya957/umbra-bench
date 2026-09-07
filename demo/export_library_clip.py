#!/usr/bin/env python3
"""export_library_clip.py — a library shadow, straight onto the robot.

    python demo/export_library_clip.py --list              # every class
    python demo/export_library_clip.py --list bird         # ids in one class
    python demo/export_library_clip.py --class bird        # export them all
    python demo/export_library_clip.py --library-id letters_upper_K_dejavusans-bold
    python demo/export_library_clip.py --sequence ICRA_scene_01_bird
    python demo/export_library_clip.py --sequence star_spin --force

Writes into fleet-shadow-art/choreographies/, which is exactly what the robot
UI serves as its Play library (render_server's GET /choreographies), so an
exported target is deployable from the UI on the next refresh -- no package
build, and nothing else in this repo has to run.

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


def rows(sweep: str) -> list:
    """Every benchmark row with a solve in `sweep`: (id, class)."""
    out = []
    for line in (BENCH / "metadata.jsonl").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        solved = (BENCH / "optimized" / sweep / r["subset"]
                  / Path(r["target"]).stem / "results.json").exists()
        out.append((r["id"], str(r.get("class", "")), solved))
    return out


def find(query: str, sweep: str) -> list:
    """Same ranking the studio's library rail uses: exact class, then class
    prefix, then id substring. Solved rows only -- an unsolved target has no
    pose to send."""
    q = query.strip().lower()
    pool = [(i, c) for i, c, ok in rows(sweep) if ok]
    for pick in (lambda c: c.lower() == q,
                 lambda c: c.lower().startswith(q),
                 None):
        hits = ([(i, c) for i, c in pool if pick(c)] if pick
                else [(i, c) for i, c in pool if q in i.lower()])
        if hits:
            return hits
    return []


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library-id", action="append", default=[])
    ap.add_argument("--sequence", action="append", default=[],
                    help="a SOLVED sequence id; exports its whole motion, "
                         "not a single pose. Reassembly is not required -- "
                         "that is the video path")
    ap.add_argument("--class", dest="cls", action="append", default=[],
                    metavar="NAME",
                    help="export EVERY solved target of this class "
                         "(e.g. --class bird). Repeatable.")
    ap.add_argument("--list", nargs="?", const="", metavar="QUERY",
                    help="print matching solved library ids and exit; "
                         "no QUERY lists every class")
    ap.add_argument("--sweep", default="big-budget-grounded")
    ap.add_argument("--fps", type=float, default=None,
                    help="source keyframe rate for --sequence clips "
                         "(default: the footage fps where one was recorded, "
                         f"else {pack.DEFAULT_CLIP_FPS:g})")
    ap.add_argument("--force", action="store_true",
                    help="write a clip whose keyframes exceed the planner's "
                         "per-joint bound. Preview only -- the arms cannot "
                         "perform it")
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    a = ap.parse_args()

    if a.list is not None:
        if not a.list:
            seen = {}
            for _i, c, ok in rows(a.sweep):
                if ok:
                    seen[c] = seen.get(c, 0) + 1
            print(f"{len(seen)} class(es) with solved targets:")
            for c in sorted(seen):
                print(f"  {c:<28} {seen[c]:3d}")
            print("\n  --list <class>  to see the ids;  "
                  "--class <class>  to export them all")
        else:
            hits = find(a.list, a.sweep)
            print(f"{len(hits)} solved target(s) matching {a.list!r}:")
            for i, c in hits:
                print(f"  {i:<44} [{c}]")
        return

    for c in a.cls:
        got = [i for i, _c in find(c, a.sweep)]
        if not got:
            sys.exit(f"[!] no solved target of class {c!r} "
                     f"-- try: --list {c}")
        a.library_id += got

    if not a.library_id and not a.sequence:
        sys.exit("[!] nothing to export: pass --library-id, --class "
                 "and/or --sequence  (--list to browse)")

    seen = set()
    a.library_id = [x for x in a.library_id
                    if not (x in seen or seen.add(x))]
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for lid in a.library_id:
        q = solve_pose(lid, a.sweep)
        pack.write_choreo(dest, lid, 5.0, [q])
        print(f"  {lid} -> {dest / (lid + '.json')}")
    for sid in a.sequence:
        # The choreography wants the pose track and nothing else. Going
        # through pack_clip for it also demanded a reassembly -- a VIDEO
        # artifact, since on hardware the rig casts at the fitted position and
        # the fit inverse never happens -- which blocked all 13 generated
        # clips, every one of them solved. clip_poses reads the solve.
        qs, fps, info = pack.clip_poses(sid, a.fps)
        pack.write_choreo(dest, sid, fps, qs, force=a.force)
        note = "" if info["reassembled"] else "  [no reassembly: deploy-only]"
        print(f"  {sid} -> {dest / (sid + '.json')}  "
              f"({len(qs)} poses @ {fps:g} fps){note}")
    n = len(a.library_id) + len(a.sequence)
    print(f"{n} clip(s); refresh Play's library to deploy")


if __name__ == "__main__":
    main()
