#!/usr/bin/env python3
"""Materialise the sequences track as a static target tree, for the baseline.

The comparison the track is built around needs two solves of the same clip:
the sequence-aware one (fleet-shadow-art's run_sequence.py, with warm starts
and the temporal terms) and the frame-independent one -- every frame handed to
the static optimizer as if it were an unrelated target. The second is the
baseline the temporal metrics exist to condemn: it produces good per-frame IoU
and, with nothing coupling consecutive solves, transitions like spinning_star's
291°. This script lays the frames out where `run_base_optimizer.py` can see
them:

    targets_sequences/<sequence_id>/<fXX>.png      subset = sequence, stem = frame

Frames are copied as they are -- NOT normalised, NOT grounded, NOT fitted.
The motion is the content: each frame's position inside the canvas is where
the animation put it, and re-placing frames independently (ground_targets.py,
or a per-frame --fit-target) adds fake motion frame-to-frame, which is exactly
the artefact the one-fit-per-clip rule in run_sequence.py exists to avoid.
Solve this tree WITHOUT --fit-target.

    python scripts/make_sequence_frame_targets.py
    python scripts/run_base_optimizer.py --repo ../fleet-shadow-art \\
        --targets-dir targets_sequences --out optimized/sequences-static \\
        --subsets <sequence ids...> --runs 3 --n-workers 0 ...
    python scripts/sequence_metrics.py --static-sweep optimized/sequences-static

The tree is derived from `sequences/` and is gitignored; re-run after any
sequence import.
"""

import argparse
import json
import os
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default=_BENCH)
    p.add_argument("--out", default="targets_sequences")
    p.add_argument("--only", nargs="*", default=None,
                   help="sequence ids to materialise (default: all)")
    a = p.parse_args()

    index = os.path.join(a.bench, "sequences.jsonl")
    if not os.path.exists(index):
        raise SystemExit(f"[!] {index} not found; run build_sequence_metadata.py")

    out_root = os.path.join(a.bench, a.out)
    n_seq = n_frames = 0
    for line in open(index, encoding="utf-8"):
        r = json.loads(line)
        if a.only and r["id"] not in a.only:
            continue
        d = os.path.join(out_root, r["id"])
        os.makedirs(d, exist_ok=True)
        for fp in r["frames"]:
            src = os.path.join(a.bench, fp)
            dst = os.path.join(d, os.path.basename(fp))
            shutil.copyfile(src, dst)
            n_frames += 1
        n_seq += 1
    print(f"[frame-targets] {n_seq} sequences, {n_frames} frames -> {out_root}")
    print("solve WITHOUT --fit-target: frames carry their own placement, and a "
          "per-frame fit invents motion the animation never had")


if __name__ == "__main__":
    main()
