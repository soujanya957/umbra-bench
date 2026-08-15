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

A GIF gets separated from its folder the moment anyone shares it, so `--caption-from`
stamps the search budget onto every frame, read back out of the sweep's own
`results.json`. Typing the numbers by hand would let the banner drift from the run it
describes; deriving them means a mislabelled GIF is impossible.
"""

import argparse
import glob
import json
import os

from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)


def _font(size: int):
    """A real TrueType face at an arbitrary size, or the bitmap fallback.

    matplotlib is already required by the sweep tooling and ships DejaVuSans, which
    makes it the one font path that resolves identically on macOS and the Linux box.
    """
    try:
        import matplotlib

        path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def budget_caption(results_dir: str) -> list[str]:
    """Describe a sweep's search budget in three lines, from what it recorded.

    Reads a single `results.json` for the optimizer/rig config — the runner refuses to
    mix budgets within a directory, so any one target speaks for all of them — and scans
    the rest only for the target count.
    """
    found = glob.glob(os.path.join(results_dir, "*", "*", "results.json"))
    if not found:
        raise SystemExit(f"[!] no results.json under {results_dir} to caption from")
    with open(found[0]) as f:
        r = json.load(f)
    c, rig = r["optimizer"], r["rig"]

    adaptive = "on" if c.get("adaptive_final", True) else "off"
    extra = ""
    if r.get("extra_below"):
        extra = f" (+{r.get('n_extra_runs', 0) or 5} if IoU<{r['extra_below']:.2f})"
    gap = rig.get("arm_gap_m")
    return [
        f"UMBRA base optimizer — {os.path.basename(os.path.normpath(results_dir))}",
        f"popsize {c['popsize']} · phase1 {c['phase1_iters']} · phase2 {c['phase2_iters']}"
        f" · final {c['final_iters']} (adaptive {adaptive})"
        f" · {r.get('n_base_runs', r['n_runs'])} runs/target{extra}",
        f"{len(found)} targets · {rig['n_robots']}× SO-101 @ {gap:g} m gap"
        f" · best-of-N IoU, sorted best → worst",
    ]


def add_banner(img: Image.Image, lines: list[str]) -> Image.Image:
    """Grow the frame upward and print `lines` into the new strip.

    Drawn over added height rather than onto the sheet, so the banner never covers a
    shadow. Type is sized off the frame width to survive `--scale`.
    """
    size = max(9, round(img.width * 0.021))
    pad = max(4, size // 2)
    fonts = [_font(round(size * 1.25)), _font(size), _font(size)]
    heights = [round(f.size * 1.45) for f in fonts]
    band = sum(heights) + 2 * pad

    out = Image.new("RGB", (img.width, img.height + band), (24, 24, 28))
    out.paste(img, (0, band))
    d = ImageDraw.Draw(out)
    y = pad
    for text, f, h, color in zip(
        lines, fonts, heights, [(255, 255, 255), (168, 214, 255), (150, 150, 160)]
    ):
        d.text((pad * 2, y), text, font=f, fill=color)
        y += h
    return out


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
    p.add_argument(
        "--caption-from",
        default=None,
        metavar="RESULTS_DIR",
        help="stamp that sweep's budget on every frame, read from its results.json",
    )
    p.add_argument(
        "--caption",
        default=None,
        help="literal banner text, newline-separated; overrides --caption-from",
    )
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

    lines = None
    if a.caption:
        lines = a.caption.split("\n")
    elif a.caption_from:
        lines = budget_caption(a.caption_from)

    # The banner is identical on every frame, so it is rendered once and pasted —
    # re-rasterising the same text 300 times would dominate the runtime of this script.
    banner = None
    frames = []
    for n in names:
        img = Image.open(os.path.join(src, n)).convert("RGB")
        if a.scale != 1.0:
            img = img.resize(
                (round(img.width * a.scale), round(img.height * a.scale)),
                Image.LANCZOS,
            )
        if lines:
            if banner is None or banner.width != img.width:
                banner = add_banner(Image.new("RGB", (img.width, 1)), lines)
            framed = Image.new("RGB", (img.width, img.height + banner.height - 1))
            framed.paste(banner, (0, 0))
            framed.paste(img, (0, banner.height - 1))
            img = framed
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
