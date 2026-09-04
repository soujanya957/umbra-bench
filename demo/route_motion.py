#!/usr/bin/env python3
"""route_motion.py — sort the demo's elements into static / translation / dynamic.

    python demo/route_motion.py            # demo_* clips
    python demo/route_motion.py --all      # the whole sequences track

Three lanes, because the clips genuinely differ in what their motion IS, and
solving them all the same way pays the full per-frame price for content that
does not have per-frame content:

  static       the element barely changes (mean target step IoU >= --static-iou).
               Solve ONE frame and hold it: run_sequence with just f00, then
               08_reassemble --hold replicates the solved pose across the clip's
               frame ids. A 10-frame static A costs one solve instead of ten,
               and its transitions are perfect by construction.

  translation  the element's motion is mostly travel: stabilising it (removing
               per-frame centroid shift) shrinks the union footprint by
               >= --min-gain. Derive the _stab variant, solve that, and
               08_reassemble puts the travel back at composite time. This is
               what recovered scene_06_L (fit scale 0.831 at bound -> 1.312
               free, avg IoU 0.433 -> 0.719).

  dynamic      real articulation or deformation. Solve the full clip as-is;
               the motion is the content and there is no discount for it.

The routing is measured, not guessed: step IoU comes from sequences.jsonl's
target_motion, and the translation test actually derives the stabilisation
(via scripts/stabilize_sequence) rather than trusting a drift heuristic --
clips whose content hugs the canvas edge are not stabilisable and fall
through to dynamic, which is also the safe default.

Writes demo/out/motion_routing.json with the lane and the exact next command
per clip.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent
sys.path.insert(0, str(BENCH / "scripts"))

from stabilize_sequence import stabilize  # noqa: E402


def lane(rec, seq_dir, static_iou, min_gain):
    step = rec["target_motion"]["mean_step_iou"]
    if step >= static_iou:
        return "static", step, None
    try:
        gain, _ = stabilize(rec, str(seq_dir), write=False)
    except ValueError:
        gain = None                     # content at the canvas edge; see docstring
    if gain is not None and gain >= min_gain:
        return "translation", step, gain
    return "dynamic", step, gain


def command(rec, which):
    sid = rec["id"]
    if which == "static":
        return (f"python demo/add_to_library.py --sequence {sid}  # named "
                f"library shape; or solve f00 alone + 08_reassemble --hold")
    if which == "translation":
        return (f"python scripts/stabilize_sequence.py --ids {sid}  # then solve "
                f"sequences/{sid}_stab/ and reassemble {sid}_stab")
    return f"solve sequences/{sid}/ as-is (full clip)"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="route every sequence, not just the demo_* clips")
    ap.add_argument("--static-iou", type=float, default=0.93,
                    help="mean target step IoU at or above which a clip is "
                         "treated as static (default sits between scene_04_M "
                         "at 0.933 and the next clip down at 0.902)")
    ap.add_argument("--min-gain", type=float, default=1.10,
                    help="union-bbox shrink factor from stabilising that "
                         "earns the translation lane")
    ap.add_argument("--out", default=None,
                    help="routing json path (default demo/out/motion_routing.json)")
    a = ap.parse_args()

    recs = [json.loads(l) for l in
            open(BENCH / "sequences.jsonl", encoding="utf-8")]
    # The library, keyed by class: a static element whose shape already has a
    # solved namesake should say so -- reuse beats re-solving, and the choice
    # stays with the person.
    lib = {}
    mp = BENCH / "metadata.jsonl"
    if mp.exists():
        for line in mp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                lib.setdefault(str(r.get("class", "")).lower(), []).append(
                    (str(r.get("class", "")), r["id"]))
    def is_film_cut(rid):
        sj = BENCH / "sequences" / rid / "source.json"
        try:
            return "scene" in json.loads(sj.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
    recs = [r for r in recs if not r["id"].endswith("_stab")
            and (a.all or is_film_cut(r["id"]))]

    routing, counts = {}, {"static": 0, "translation": 0, "dynamic": 0}
    print(f"{'sequence':<24}{'lane':<13}{'step IoU':>9}{'stab gain':>10}   next")
    for r in recs:
        which, step, gain = lane(r, BENCH / "sequences", a.static_iou, a.min_gain)
        counts[which] += 1
        cmd = command(r, which)
        cand = []
        if which == "static":
            cls = str(r.get("class", ""))
            pool = lib.get(cls.lower(), [])
            # exact-case class first: the demo's "A" should meet the library's
            # upper-case A before the lower-case a
            cand = [i for c, i in pool if c == cls][:3] or [i for _, i in pool][:3]
        routing[r["id"]] = {"lane": which, "mean_step_iou": step,
                            "stabilize_gain": gain, "n_frames": r["n_frames"],
                            "library_candidates": cand, "next": cmd}
        if cand:
            print(f"{'':<24}{'':>13}{'':>9}{'':>10}   library has this class: "
                  + ", ".join(cand))
        print(f"{r['id']:<24}{which:<13}{step:>9.3f}"
              f"{(f'{gain:.2f}x' if gain else '-'):>10}   {cmd[:64]}")

    out = Path(a.out) if getattr(a, "out", None) else (
        ROOT / "out" / "motion_routing.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"static_iou": a.static_iou, "min_gain": a.min_gain,
                               "routing": routing}, indent=1), encoding="utf-8")
    n = sum(counts.values())
    saved = sum(routing[k]["n_frames"] - 1 for k in routing
                if routing[k]["lane"] == "static")
    print(f"\n{n} clips: {counts['static']} static, {counts['translation']} "
          f"translation, {counts['dynamic']} dynamic -> {out}")
    if saved:
        print(f"the static lane alone replaces {saved} per-frame solves with holds")


if __name__ == "__main__":
    main()
