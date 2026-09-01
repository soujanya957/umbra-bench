#!/usr/bin/env python3
"""Segment every teleop capture with no clicking. One command, 29 masks, one sheet.

Automatic THRESHOLDING does not work on these frames — the shadow is 30-60 grey
levels under the wall, the lamp falloff is the same magnitude, and on four
captures the rectified crop reaches past the lit cone so a global cut classifies
half the frame as shadow. But the thing a person was supplying by clicking was
never the boundary; it was two facts: "this is shadow" and "that is wall". Those
can be derived, and then a real segmenter draws the boundary.

So the pipeline is: remove the illumination, pick seeds from the extremes of what
is left, and hand them to SAM 2.

  1. Flat field. Divide by a heavily blurred copy, then CLAHE at clipLimit 5.
     Takes the working range from a std of 6-17 grey levels to 29-45.
  2. Seed. Positive points come from the DARKEST decile — connected, at least
     0.5% of the frame, sampled at the peaks of their distance transform, which
     puts every seed deep inside a shadow region rather than near its edge.
     Negative points come from the brightest quintile plus the frame border.
     Deliberately not Otsu: Otsu is what fails on the four bad captures, and a
     seed only has to be *inside*, which the extremes guarantee and a threshold
     does not.
  3. Segment. Otsu on that enhanced field, by default — and it works now for the
     reason it did not before: Otsu was never the problem, contrast was. On the
     raw flat field it produced four captures at 0.72-0.80 of the frame; on the
     enhanced one all 29 land between 0.10 and 0.22 with no outlier at all. The
     seeds above are still computed, and are what `--backend sam2` and
     `--backend geo` consume when a boundary needs more than a threshold.
  4. Clean with the benchmark's own `_denoise` — sub-0.5% components dropped,
     sub-0.5% holes filled. Never a blanket fill: that erases the counter of an
     'a' and both bowls of a 'B', which is the topology pw_h1 exists to measure.

A contact sheet lands next to the masks so all 29 can be judged at once, and any
capture whose result looks structurally wrong is reported rather than buried.

    python scripts/auto_segment_teleop.py --write            # this is the one
    python scripts/auto_segment_teleop.py --backend sam2 --write
"""
from __future__ import annotations

import argparse, json, os

import cv2
import numpy as np
from PIL import Image

import _rescue_common as rc
from shape_attributes import _denoise
from segment_teleop import flatfield8, geo_backend, sam2_backend, DEFAULT_CKPT


def auto_seeds(ff, n_pos=6, n_neg=8, dark_pct=10, light_pct=80):
    """Points that are certainly inside the shadow, and certainly not."""
    h, w = ff.shape
    dark = (ff <= np.percentile(ff, dark_pct)).astype(np.uint8)
    dark[:6] = dark[-6:] = 0; dark[:, :6] = dark[:, -6:] = 0     # rectification edge
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    pos = []
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    for i in order:
        if stats[i, cv2.CC_STAT_AREA] < 0.005 * h * w:
            break
        comp = (lab == i).astype(np.uint8)
        d = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        for _ in range(max(1, n_pos // 3)):                       # deepest points first
            iy, ix = np.unravel_index(int(d.argmax()), d.shape)
            if d[iy, ix] <= 2:
                break
            pos.append((int(ix), int(iy)))
            cv2.circle(d, (ix, iy), int(max(8, d[iy, ix])), 0, -1)
        if len(pos) >= n_pos:
            break
    light = ff >= np.percentile(ff, light_pct)
    ys, xs = np.nonzero(light)
    if len(xs):
        step = max(1, len(xs) // n_neg)
        neg = [(int(xs[i]), int(ys[i])) for i in range(0, len(xs), step)][:n_neg]
    else:
        neg = []
    neg += [(4, 4), (w - 5, 4), (4, h - 5), (w - 5, h - 5)]
    return pos, neg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["otsu", "sam2", "geo"], default="otsu")
    ap.add_argument("--tol", type=int, default=55, help="geo only: edge pull")
    ap.add_argument("--open-r", type=int, default=2)
    ap.add_argument("--checkpoint", default=os.getenv("SAM2_CHECKPOINT", DEFAULT_CKPT))
    ap.add_argument("--config", default=os.getenv("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml"))
    ap.add_argument("--enhance", action="store_true",
                    help="sam2 only: CLAHE the photo before segmenting it. Off by "
                         "default — the seeds come from the enhanced field, the "
                         "boundary comes from the photo as shot.")
    ap.add_argument("--points", default="Teleops/masks/points.json",
                    help="dashboard markup. Used when present; auto seeds fill the gaps.")
    ap.add_argument("--ignore-points", action="store_true",
                    help="ignore --points and seed every capture automatically.")
    ap.add_argument("--blur", type=float, default=0.0,
                    help="sam2 only: Gaussian sigma on the image SAM 2 sees. 0 = off.")
    ap.add_argument("--ev", type=float, default=0.0,
                    help="sam2 only: exposure in stops. +0.3 is about a third of a stop.")
    ap.add_argument("--contrast", type=float, default=1.0,
                    help="sam2 only: contrast gain about the frame mean. 1.0 = off.")
    ap.add_argument("--out", default="Teleops/masks")
    ap.add_argument("--sheet", default="Teleops/masks/_contact_sheet.png")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    backend = a.backend
    if backend == "sam2":
        try:
            import torch, sam2  # noqa: F401
        except Exception as e:
            print(f"[!] sam2 unavailable ({type(e).__name__}) — falling back to geo")
            backend = "geo"

    # Hand markup wins wherever it exists. auto_seeds is the fallback for
    # captures nobody marked, not the primary source: a percentile rule states
    # "this decile is dark" and hopes that means shadow, while a person pointing
    # at the shadow states it. Deriving the seeds was worth doing because it made
    # 29 captures reachable without 29 sessions of clicking — it was never worth
    # preferring over the clicking that had already happened.
    hand = {}
    pts_p = os.path.join(rc.BENCH, a.points)
    if not a.ignore_points and os.path.exists(pts_p):
        hand = json.load(open(pts_p)).get("captures", {})
        print(f"[points] {len(hand)} hand-marked captures from {a.points}")
    elif not a.ignore_points:
        print(f"[points] {a.points} not found — seeding every capture automatically")

    man_p = os.path.join(rc.BENCH, "Teleops", "masks", "manifest.json")
    man = json.load(open(man_p))
    rows, tiles = [], []
    for rec in man["records"]:
        path = os.path.join(rc.BENCH, rec["rectified"])
        ff = flatfield8(path)
        h = hand.get(rec["capture"])
        if h and h.get("pos"):
            pos = [(float(x), float(y)) for x, y in h["pos"]]
            neg = [(float(x), float(y)) for x, y in h.get("neg", [])]
            src = "hand"
        else:
            pos, neg = auto_seeds(ff)
            src = "auto"
        if not pos and backend != "otsu":
            print(f"  !! no seed found for {rec['capture']}"); continue
        if backend == "otsu":
            from skimage.filters import threshold_otsu
            m = (ff < threshold_otsu(ff)).astype(np.uint8)
            m[:6] = m[-6:] = 0; m[:, :6] = m[:, -6:] = 0     # rectification edge
        elif backend == "sam2":
            m = sam2_backend(path, pos, neg, a.checkpoint, a.config, a.enhance,
                             a.blur, a.ev, a.contrast)
        else:
            m = geo_backend(path, pos, neg, a.tol)
        if a.open_r and m.sum():
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*a.open_r+1, 2*a.open_r+1))
            mo = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
            if mo.sum() > 0.2 * m.sum():
                m = mo
        if m.sum():
            m = _denoise(m, float(m.sum()))
        frac = float(m.mean())
        rows.append((rec["capture"], frac, rc.n_holes_signif(m), len(pos), src))
        if a.write:
            rc.save_mask(m, os.path.join(rc.BENCH, a.out, rec["capture"] + "_mask.png"))
            tag = "auto-" + backend + ("-clahe" if backend == "sam2" and a.enhance else "")
            rec.update(mask_backend=tag, shape_frac=round(frac, 4),
                       n_seeds=len(pos), holes_signif=rc.n_holes_signif(m),
                       seed_source=src)
        # contact-sheet row: flat field, mask, and the seeds that produced it
        vis = cv2.cvtColor(ff, cv2.COLOR_GRAY2BGR)
        # int() here, not at the source: SAM 2 takes the sub-pixel coordinates the
        # dashboard exported, only the drawing needs whole pixels.
        for x, y in pos: cv2.circle(vis, (int(x), int(y)), 5, (0, 200, 0), -1)
        for x, y in neg: cv2.circle(vis, (int(x), int(y)), 4, (0, 90, 240), -1)
        tiles.append(np.hstack([cv2.resize(vis, (200, 153)),
                                cv2.resize(cv2.cvtColor((1-m)*255, cv2.COLOR_GRAY2BGR), (200, 153))]))

    fr = np.array([r[1] for r in rows])
    med = float(np.median(fr)); mad = float(np.median(np.abs(fr - med)) + 1e-9)
    print(f"\nbackend {backend}  ·  {len(rows)} captures")
    n_hand = sum(1 for r in rows if r[4] == "hand")
    print(f"seeds: {n_hand} hand-marked, {len(rows) - n_hand} auto")
    print(f"{'capture':46} {'frac':>6} {'holes':>6} {'seeds':>6} {'from':>5}")
    for cap, frac, hs, ns, src in rows:
        flag = "  <-- outlier" if abs(frac - med) > 4 * mad else ""
        print(f"{cap[:46]:46} {frac:6.3f} {hs:6d} {ns:6d} {src:>5}{flag}")
    print(f"\nshape fraction: median {med:.3f}, range {fr.min():.3f}-{fr.max():.3f}")

    if a.write:
        per = 3
        sheet = np.vstack([np.hstack(tiles[i:i+per] + [np.zeros_like(tiles[0])]*(per-len(tiles[i:i+per])))
                           for i in range(0, len(tiles), per)])
        os.makedirs(os.path.dirname(os.path.join(rc.BENCH, a.sheet)), exist_ok=True)
        cv2.imwrite(os.path.join(rc.BENCH, a.sheet), sheet)
        # Follow --out, so a side-by-side run into a scratch dir cannot overwrite
        # the manifest that describes the masks actually in Teleops/masks.
        out_man = os.path.join(rc.BENCH, a.out, "manifest.json")
        os.makedirs(os.path.dirname(out_man), exist_ok=True)
        json.dump(man, open(out_man, "w"), indent=1)
        print(f"wrote {len(rows)} masks, manifest.json and {a.sheet}")
        print("  sheet columns per capture: flat field with seeds (green = shadow, "
              "orange = wall) | resulting mask")
    else:
        print("\ndry run — pass --write to save masks and the contact sheet")


if __name__ == "__main__":
    main()
