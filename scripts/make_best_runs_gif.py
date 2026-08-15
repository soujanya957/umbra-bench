#!/usr/bin/env python3
"""Flip through the best-runs comparison sheets as an animated GIF.

Scrolling a folder of a few hundred PNGs means deciding when to move on for each one.
A fixed-cadence loop takes that decision away: the sheets go past at a steady rate,
best first, and the point where the shapes stop reading announces itself.

Reads whatever `make_best_runs.py` has already written, so it is instant to re-run and
never re-renders a figure. Sheets are named IoU-first and zero-padded, which makes a
reverse filename sort the intended play order (best → worst) with no metadata lookup.

Frames are quantised to a shared adaptive palette. These sheets use maybe six colours
between them (white, black, the cyan/magenta/blue overlay, and antialiased text), so a
palette shared across every frame stays faithful and keeps the file small — per-frame
palettes would cost more bytes for no visible gain.
"""

import argparse
import os

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        default=None,
        help="default <bench>/optimized/base-optimizer/best-runs",
    )
    p.add_argument("--out", default=None, help="default <src>/best-runs.gif")
    p.add_argument("--seconds", type=float, default=1.0, help="seconds per frame")
    p.add_argument("--scale", type=float, default=1.0, help="resize factor, e.g. 0.7")
    p.add_argument("--colors", type=int, default=64)
    p.add_argument(
        "--worst-first",
        action="store_true",
        help="ascending IoU instead of the default best-first",
    )
    p.add_argument("--limit", type=int, default=None, help="only the first N frames")
    a = p.parse_args()

    src = a.src or os.path.join(_BENCH, "optimized", "base-optimizer", "best-runs")
    out = a.out or os.path.join(src, "best-runs.gif")

    names = sorted(
        f for f in os.listdir(src) if f.startswith("iou") and f.endswith(".png")
    )
    if not names:
        print(f"[!] no iou*.png sheets in {src} — run make_best_runs.py first")
        return
    if not a.worst_first:
        names.reverse()
    if a.limit:
        names = names[: a.limit]

    frames = []
    for n in names:
        img = Image.open(os.path.join(src, n)).convert("RGB")
        if a.scale != 1.0:
            img = img.resize(
                (round(img.width * a.scale), round(img.height * a.scale)),
                Image.LANCZOS,
            )
        frames.append(img)

    # One palette for the whole sequence: frames share a near-identical colour set, and
    # a per-frame palette would make the GIF larger without looking any different.
    master = frames[0].quantize(colors=a.colors, method=Image.MEDIANCUT)
    frames = [f.quantize(palette=master, dither=Image.NONE) for f in frames]

    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(a.seconds * 1000)),
        loop=0,
        optimize=True,
        disposal=2,
    )
    mb = os.path.getsize(out) / 1e6
    print(
        f"[gif] {len(frames)} frames @ {a.seconds:g}s "
        f"({len(frames) * a.seconds / 60:.1f} min loop)  {mb:.1f} MB → {out}"
    )
    print(f"[gif] order: {'worst' if a.worst_first else 'best'} IoU first")


if __name__ == "__main__":
    main()
