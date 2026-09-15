#!/usr/bin/env python3
"""Pad targets to the canonical square frame without distorting them.

Every authored target is 512x512, black shape on white. Targets that arrive from
somewhere else -- the teleop captures are 500x383 -- are not, and nothing in the
pipeline resamples them safely: `run_base_optimizer.load_target` does a flat
`resize((size, size))`, so a 500x383 mask is stretched vertically by 1.3x before
it is ever solved. The shape the optimizer chases is then not the shape on disk,
and every attribute in metadata.jsonl describes the undistorted one.

This pads to a square canvas instead of scaling anisotropically: the shape keeps
its aspect ratio and its scale relative to the frame, and the padding is
background. Horizontal placement is centred; vertical placement is left alone,
because `ground_targets.py` is what decides vertical placement and running this
first must not pre-empt it.

Idempotent: a target already square at --size is skipped, so it is safe to
re-run after dropping new captures in. That is the point -- teleop is a growing
subset.

    python scripts/normalize_targets.py --subsets teleop --dry-run
    python scripts/normalize_targets.py --subsets teleop
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def polarity_ok(a: np.ndarray) -> bool:
    """True if this looks like black-shape-on-white (the canonical convention)."""
    corners = [a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]]
    return sum(int(c) > 127 for c in corners) >= 3


def normalize(path: str, size: int, dry: bool) -> str:
    im = Image.open(path).convert("L")
    if im.size == (size, size):
        return "skip"

    a = np.asarray(im)
    if not polarity_ok(a):
        # Refuse rather than guess: inverting a mask that was not inverted turns
        # the shape into its background and every downstream number with it.
        return "POLARITY"

    w, h = im.size
    if w > size or h > size:
        # Scale the long side down to fit, preserving aspect. NEAREST keeps the
        # mask binary; any smooth filter introduces greys that later thresholds
        # would have to guess at.
        s = min(size / w, size / h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.NEAREST)
        w, h = im.size

    canvas = Image.new("L", (size, size), 255)     # white = background
    canvas.paste(im, ((size - w) // 2, (size - h) // 2))

    if not dry:
        canvas.point(lambda v: 255 if v > 127 else 0).convert("L").save(path)
    return "padded"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=BENCH)
    ap.add_argument("--src", default="targets", help="target tree, repo-relative")
    ap.add_argument("--subsets", nargs="+", default=None,
                    help="default: every subset in --src")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    root = os.path.join(a.bench, a.src)
    subsets = a.subsets or sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

    tally = {"padded": 0, "skip": 0, "POLARITY": 0}
    bad = []
    for sub in subsets:
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            print(f"  ! no such subset: {sub}")
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".png") or fn.startswith("_"):
                continue
            r = normalize(os.path.join(d, fn), a.size, a.dry_run)
            tally[r] += 1
            if r == "POLARITY":
                bad.append(f"{sub}/{fn}")

    verb = "would pad" if a.dry_run else "padded"
    print(f"{verb} {tally['padded']}, already {a.size}x{a.size} {tally['skip']}")
    if bad:
        print(f"\n! {len(bad)} not black-on-white, left untouched -- invert them first:")
        for b in bad[:10]:
            print(f"    {b}")
    if a.dry_run:
        print("\ndry run -- drop --dry-run to write")


if __name__ == "__main__":
    main()
