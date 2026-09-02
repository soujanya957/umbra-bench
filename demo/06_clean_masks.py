#!/usr/bin/env python3
"""06_clean_masks.py — make SAM's masks castable.

SAM returns the glyph but not a clean region. Measured on this footage, one F
mask has all four of these at once, and they are four different faults needing
four different fixes:

  interior speckle   the glyph's own dark outline strokes fall below the
                     colour test, so the mask is peppered with holes
  detached crumbs    a few small blobs sitting off the shape entirely
  broken skirt       the bottom of the glyph breaks up where it meets its
                     drop shadow
  stair-step edge    ordinary raster jaggies along every diagonal

Order matters. Filling holes before dropping crumbs would keep the crumbs;
smoothing before either would smooth around them and lock them in.

  1. keep the largest connected component      -> crumbs gone
  2. fill enclosed holes                       -> speckle gone
  3. blur the binary field, threshold at 0.5   -> edge smoothed

Step 3 is a Gaussian on the mask rather than a morphological open/close because
it is symmetric: a close alone grows the shape, an open alone shrinks it, and
alternating them leaves the corners chewed. Thresholding a blurred field at 0.5
keeps the area almost unchanged while rounding both protrusions and notches, and
`--sigma` is a legible knob -- roughly the radius of detail it erases.

    python 06_clean_masks.py --in letters_sam2 --out letters_clean
    python 06_clean_masks.py --in letters_sam2 --sigma 3 --compare f0712
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import json

import cv2
import numpy as np
from PIL import Image


def keep_colour(m: np.ndarray, rgb: np.ndarray, s_lo: int, h_lo: int, h_hi: int) -> np.ndarray:
    """Shrink the mask to the glyph's own colour, dropping its dark outline.

    SAM traces the outside of the drawn outline, so ~10% of a mask is the black
    stroke the artist drew around the letter rather than the letter. That stroke
    is a drawing convention, not part of the shape a robot has to throw, and it
    is what reads as a shadow around the cutout. Measured over six M masks the
    glyph body sits at H 26-32 and S 192-195, well clear of the stroke.

    Only the intersection is taken here; the holes this opens along interior
    strokes are closed by fill_holes() afterwards, which is why the order in
    clean() matters.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S = hsv[..., 0], hsv[..., 1]
    return m & (S >= s_lo) & (H >= h_lo) & (H <= h_hi)


def bridge_parts(m: np.ndarray, width: int) -> np.ndarray:
    """Join every remaining component to the body, shortest gap first.

    A cast shadow is one shape. This is a safety net rather than the main event:
    on the raw SAM masks 0 of 55 scene_03 frames have a detached part, because the
    cane is attached through the glyph's own dark outline. It only matters if a
    step upstream cuts something free -- --colour did exactly that, opening a
    5-8 px gap at the cane -- or if SAM drops a stroke.

    The closest pair of points between the fragment and the body is found through
    a distance transform, and a line of `width` px is drawn along it.
    """
    while True:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        if n <= 2:
            return m
        order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
        body = lab == order[0]
        dist = cv2.distanceTransform((~body).astype(np.uint8), cv2.DIST_L2, 5)
        best = None
        for c in order[1:]:
            d = np.where(lab == c, dist, np.inf)
            k = np.unravel_index(int(np.argmin(d)), d.shape)
            if best is None or d[k] < best[0]:
                best = (float(d[k]), k)
        fy, fx = best[1]
        ys, xs = np.where(body)
        j = int(np.argmin((ys - fy) ** 2 + (xs - fx) ** 2))
        canvas = m.astype(np.uint8)
        cv2.line(canvas, (int(fx), int(fy)), (int(xs[j]), int(ys[j])), 1, width)
        m = canvas.astype(bool)


def keep_parts(m: np.ndarray, frac: float, min_abs: int) -> np.ndarray:
    """Keep every component that is a real part; drop only specks.

    Keeping just the largest was too blunt. In scene_03 the I bends into a figure
    leaning on a cane, and the cane is its own component at 3.2-3.7k px -- 5-6% of
    the body -- so "largest only" deleted it from every frame of the sequence. The
    same frames carry 280-390 components, nearly all of them single-pixel specks,
    so something does have to go.

    A fraction of the largest separates the two cleanly: the cane sits at 0.05 and
    the specks three orders of magnitude below. `frac=0` keeps everything above
    `min_abs`, for a glyph that really is in many pieces.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 2:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    thr = max(min_abs, int(frac * areas.max()))
    keep = [i + 1 for i, a in enumerate(areas) if a >= thr]
    if not keep:
        keep = [1 + int(np.argmax(areas))]
    return np.isin(lab, keep)


def fill_holes(m: np.ndarray) -> np.ndarray:
    """Fill everything not reachable from the border: interior holes only."""
    h, w = m.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    inv = (~m).astype(np.uint8)
    cv2.floodFill(inv, ff, (0, 0), 2)
    return m | (inv != 2)


def smooth(m: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return m
    k = int(sigma * 4) | 1                      # odd kernel, ~4 sigma of support
    blurred = cv2.GaussianBlur(m.astype(np.float32), (k, k), sigma)
    return blurred >= 0.5


def clean(m: np.ndarray, sigma: float, keep_frac: float, min_abs: int, do_fill: bool,
          rgb: np.ndarray | None = None, colour=None, bridge: int = 0):
    before = m.copy()
    if rgb is not None and colour is not None:
        m = keep_colour(m, rgb, *colour)
    m = keep_parts(m, keep_frac, min_abs)
    if bridge:
        m = bridge_parts(m, bridge)
    if do_fill:
        m = fill_holes(m)
    m = smooth(m, sigma)
    if do_fill:                                  # smoothing can reopen a pinhole
        m = fill_holes(m)
    # Smoothing can also shed a speck: thresholding a blurred field detaches a
    # couple of pixels off a thin tip. One frame came out 15966 + 2 px. Sweep
    # again at the end so the invariant -- one target, one connected shadow --
    # holds on what is actually written, not on an intermediate.
    m = keep_parts(m, keep_frac, min_abs)
    if bridge:
        m = bridge_parts(m, bridge)
    n_before = cv2.connectedComponentsWithStats(before.astype(np.uint8), 8)[0] - 1
    n_after = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)[0] - 1
    return m, {"area_before": int(before.sum()), "area_after": int(m.sum()),
               "cc_before": n_before, "cc_after": n_after}


def roughness(m: np.ndarray) -> float:
    """Perimeter over the perimeter of a disc of equal area. 1.0 = smoothest."""
    a = int(m.sum())
    if not a:
        return 0.0
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cnts)
    return float(per / (2 * np.sqrt(np.pi * a))) if per else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", default="letters_sam2")
    ap.add_argument("--out", default="letters_clean")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="Gaussian radius in px at full resolution. 0 disables "
                         "smoothing and leaves only the topology fixes.")
    ap.add_argument("--keep-frac", type=float, default=0.03,
                    help="keep any component at least this fraction of the largest. "
                         "The cane in scene_03 is 0.05 of its figure and the specks "
                         "are ~0.0001, so 0.03 separates them. 0 keeps everything "
                         "above --min-area.")
    ap.add_argument("--min-area", type=int, default=200,
                    help="absolute floor in px, applied with --keep-frac")
    ap.add_argument("--bridge", type=int, default=5,
                    help="join surviving components to the body with a line this "
                         "wide, so every target is one connected shadow. 0 leaves "
                         "them apart.")
    ap.add_argument("--no-fill", dest="do_fill", action="store_false")
    ap.add_argument("--colour", action="store_true",
                    help="shrink each mask to the glyph's own hue, dropping the "
                         "dark outline SAM includes. Needs --keypoints for the "
                         "frame -> source image mapping.")
    ap.add_argument("--keypoints", default=str(Path(__file__).resolve().parent / "keypoints.json"))
    ap.add_argument("--s-lo", type=int, default=150)
    ap.add_argument("--h-lo", type=int, default=15)
    ap.add_argument("--h-hi", type=int, default=40)
    ap.add_argument("--compare", nargs="*",
                    help="write a before/after strip for these frame ids")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    masks = sorted(src.glob("by_frame/*/*_mask.png")) or sorted(src.glob("*_mask.png"))
    if not masks:
        raise SystemExit(f"no masks under {src}/by_frame")

    kp = {}
    if a.colour:
        kp = json.loads(Path(a.keypoints).read_text(encoding="utf-8"))["frames"]
    root = Path(a.keypoints).resolve().parent
    colour = (a.s_lo, a.h_lo, a.h_hi) if a.colour else None
    src_cache: dict[str, np.ndarray] = {}

    rows = []
    for p in masks:
        m = np.array(Image.open(p).convert("L")) < 128
        r_before = roughness(m)
        rgb = None
        if a.colour:
            fid = p.parent.name if p.parent.name != src.name else p.name.split("_")[0]
            if fid in kp:
                if fid not in src_cache:
                    src_cache[fid] = np.array(Image.open(root / kp[fid]["file"]).convert("RGB"))
                rgb = src_cache[fid]
        cleaned, st = clean(m, a.sigma, a.keep_frac, a.min_area, a.do_fill,
                            rgb, colour, a.bridge)
        # Flat. The input is by_frame/<fid>/<fid>_<L>_mask.png -- three levels
        # with one file at the bottom, which cannot be skimmed in a file browser.
        # The frame id is already in the name, so the directories carried nothing.
        dst = out / p.name
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.where(cleaned, 0, 255).astype(np.uint8)).save(dst)
        rows.append({"mask": p.name,
                     "roughness_before": round(r_before, 3),
                     "roughness_after": round(roughness(cleaned), 3), **st})

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "clean_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rb = np.mean([r["roughness_before"] for r in rows])
    ra = np.mean([r["roughness_after"] for r in rows])
    da = np.mean([r["area_after"] / max(1, r["area_before"]) for r in rows])
    split = sum(1 for r in rows if r["cc_before"] > 1)
    print(f"{len(rows)} masks -> {out}   sigma={a.sigma}")
    print(f"  roughness (perimeter / equal-area disc)  {rb:.2f} -> {ra:.2f}")
    print(f"  area kept                                 {da:.1%}")
    print(f"  masks that had detached pieces            {split}")
    multi = sum(1 for r in rows if r["cc_after"] > 1)
    print(f"  masks still in more than one piece        {multi}"
          + ("  <- every target should be one shadow" if multi else "  (all connected)"))

    for fid in (a.compare or []):
        srcs = (sorted(src.glob(f"by_frame/{fid}/*_mask.png"))
                or sorted(src.glob(f"{fid}_*_mask.png")))
        if not srcs:
            continue
        tiles = []
        for p in srcs:
            b = np.array(Image.open(p).convert("L")) < 128
            c = np.array(Image.open(out / p.name).convert("L")) < 128
            ys, xs = np.where(b | c)
            box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
            rgb = np.full((*b[box].shape, 3), 255, np.uint8)
            rgb[b[box] & ~c[box]] = (235, 70, 70)      # removed
            rgb[c[box] & ~b[box]] = (70, 170, 235)     # added
            rgb[b[box] & c[box]] = (35, 35, 35)        # kept
            tiles.append(Image.fromarray(rgb))
        H = max(t.height for t in tiles)
        sheet = Image.new("RGB", (sum(t.width for t in tiles) + 10 * len(tiles), H), "white")
        x = 0
        for t in tiles:
            sheet.paste(t, (x, 0)); x += t.width + 10
        sheet.save(out / f"compare_{fid}.png")
        print(f"  compare_{fid}.png  (dark kept, red removed, blue added)")


if __name__ == "__main__":
    main()
