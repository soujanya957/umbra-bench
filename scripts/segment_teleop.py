#!/usr/bin/env python3
"""Segment the teleop captures from clicked point prompts.

Automatic thresholding is not good enough on these frames and the review made
that plain: the shadow sits 30-60 grey levels below the wall, the lamp falloff is
the same magnitude, and on four captures the rectified crop reaches past the lit
cone so there is nothing to normalise against. A person pointing at the shadow
resolves in one click what no global rule gets right.

Points come from the dashboard's Teleop view (`copy points` -> `points.json`).
Two backends consume the same points, so a preview on any machine and the final
run on a GPU box do not need different markup:

  geo    Geodesic competition — each pixel joins whichever seed set it can reach
         more cheaply over an edge-weighted cost. Default, because it is what the
         dashboard previews: markup that does not predict the output is markup
         made blind.
  rw     skimage's random walker. Smoother boundaries, but on these frames a
         single small seed barely diffuses and it returns roughly a tenth of the
         area geo does, so it is an alternative to try rather than the default.
  sam2   Meta's SAM 2 with the checkpoint the render_server already ships.
         Needs torch + the sam2 package.

A flood fill was tried first and is not offered, because it fails in the way
flood fills fail: the shadow is one connected region, so an exclude point landing
anywhere inside it deleted the entire mask. Connectivity is all-or-nothing, and a
second click should refine a segmentation rather than destroy it.

Postprocessing is the benchmark's own `_denoise` (drop sub-0.5% components, fill
sub-0.5% holes) — never a blanket hole fill, which would erase the counter of an
'a' and both bowls of a 'B', the very topology pw_h1 is there to measure.

    python scripts/segment_teleop.py --write                  # geodesic, matches the preview
    python scripts/segment_teleop.py --backend sam2 --write   # if torch is around
"""
from __future__ import annotations

import argparse, json, os

import cv2
import numpy as np
from PIL import Image

import _rescue_common as rc
from shape_attributes import _denoise

TELEOP = os.path.join(rc.BENCH, "Teleops")
# The checkpoint lives in the render_server repo, which is checked out beside
# this one on some machines and under ~/GitHub on others. Probe both instead of
# hardcoding one and making every other machine pass --checkpoint by hand.
_CKPT_REL = os.path.join("Shadow_robot_ui", "render_server", "checkpoints",
                         "sam2.1_hiera_small.pt")
_CKPT_CANDIDATES = [
    os.path.join(os.path.dirname(rc.BENCH), "fleet-shadow-art", _CKPT_REL),
    os.path.expanduser(os.path.join("~", "GitHub", "fleet-shadow-art", _CKPT_REL)),
    os.path.join(rc.BENCH, "checkpoints", "sam2.1_hiera_small.pt"),
]
DEFAULT_CKPT = next((p for p in _CKPT_CANDIDATES if os.path.exists(p)),
                    _CKPT_CANDIDATES[0])


# ── flat field, identical to the one the browser segments on ────────────────
def flatfield8(path, blur_frac=0.25, lo=0.55, hi=1.15, clip=5.0):
    """Illumination divided out, then CLAHE. Must match _build_teleop_payload."""
    g = np.array(Image.open(path).convert("L"), np.float32)
    k = int(blur_frac * max(g.shape)) | 1
    fl = g / np.maximum(cv2.GaussianBlur(g, (k, k), 0), 1e-3)
    q = (np.clip((fl - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(q)


def _markers(shape, pos, neg, r=4, border_step=48):
    """1 = foreground seed, 2 = background.

    The frame border seeds background so that one click on the shadow is already
    a well-posed problem — but SPARSELY, every `border_step` pixels. A random
    walker is a diffusion: seeding the entire border makes the background an
    overwhelming mass that a single interior point cannot compete with, and every
    mask comes back at about 1% of the frame no matter what beta is. Sparse seeds
    state the same fact without stacking the vote. (The geodesic backend is
    immune to this, since a minimum distance does not care how many seeds there
    are; that asymmetry is exactly why this needs saying.)
    """
    h, w = shape
    mk = np.zeros((h, w), np.uint8)
    for x in range(0, w, border_step):
        mk[0, x] = mk[h - 1, x] = 2
    for y in range(0, h, border_step):
        mk[y, 0] = mk[y, w - 1] = 2
    yy, xx = np.ogrid[:h, :w]
    for lbl, pts in ((2, list(neg)), (1, list(pos))):
        for x, y in pts:
            x, y = int(round(x)), int(round(y))
            mk[(xx - x) ** 2 + (yy - y) ** 2 <= r * r] = lbl
    return mk


def rw_backend(path, pos, neg, beta):
    """Random walker: each pixel takes the label whose walker reaches it first.

    The classical answer to seeded segmentation on weak gradients, and unlike a
    flood it degrades gracefully — an extra seed always refines the boundary
    instead of flipping a whole component.
    """
    from skimage.segmentation import random_walker
    ff = flatfield8(path).astype(np.float64) / 255.0
    mk = _markers(ff.shape, pos, neg)
    lab = random_walker(ff, mk, beta=beta, mode="cg_j")
    return (lab == 1).astype(np.uint8)


def geo_backend(path, pos, neg, tol):
    """Geodesic competition: each pixel joins the seed set it can reach cheaper.

    Minimum-cost paths over `1 + gamma * |grad|`, so a path pays for every edge it
    crosses and travels freely inside a uniform region. Computed with skimage's
    MCP_Geometric (Dijkstra in Cython) rather than the browser's raster sweep;
    same quantity, exact instead of approximate, and fast enough to not matter.

    This is the default because it is what the dashboard previews. A backend that
    disagrees with the preview makes the markup a guess, and the whole point of
    clicking was to stop guessing. The agreement is close but not exact: measured
    on three captures at the same seeds and pull, this returns 0.10-0.12 of the
    frame where the browser shows 0.07-0.10 — same region, drawn about a fifth
    more generously, because the exact Dijkstra crosses a soft edge the raster
    sweep's three passes stop short of. Mark against the preview and expect a
    slightly fuller mask, not a different one.
    """
    from skimage.graph import MCP_Geometric
    ff = flatfield8(path).astype(np.float64)
    gx = cv2.Sobel(ff, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(ff, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy) / 8.0
    cost = 1.0 + (tol / 12.0) * grad

    def dist(pts):
        if not pts:
            return np.full(ff.shape, np.inf)
        starts = [(int(round(y)), int(round(x))) for x, y in pts]
        h, w = ff.shape
        starts = [(y, x) for y, x in starts if 0 <= y < h and 0 <= x < w]
        if not starts:
            return np.full(ff.shape, np.inf)
        return MCP_Geometric(cost).find_costs(starts)[0]

    h, w = ff.shape
    border = ([(x, 0) for x in range(0, w, 48)] + [(x, h - 1) for x in range(0, w, 48)]
              + [(0, y) for y in range(0, h, 48)] + [(w - 1, y) for y in range(0, h, 48)])
    return (dist(list(pos)) < dist(list(neg) + border)).astype(np.uint8)


# ── SAM 2, loaded the way render_server/sam_segment.py loads it ─────────────
_PRED = None
def sam2_predictor(ckpt, cfg):
    global _PRED
    if _PRED is not None:
        return _PRED
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}\n"
                         "point --checkpoint at render_server/checkpoints/sam2.1_hiera_small.pt")
    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    _PRED = SAM2ImagePredictor(build_sam2(cfg, ckpt, device=dev))
    print(f"[sam2] loaded on {dev}")
    return _PRED


def expose(rgb, ev=0.0, contrast=1.0):
    """Exposure in stops, then contrast as a gain about the frame's own mean.

    Global on purpose, unlike the CLAHE path below. CLAHE is local, so it lifts
    wall texture and the lamp's edge exactly as hard as it lifts the shadow
    boundary — which is why segmenting the enhanced field went looking for edges
    that were not there. A stop of exposure and a gain about the mean move the
    whole frame together and leave the shadow-to-wall relationship intact.

    Pivoting on the post-exposure mean keeps the two knobs independent: exposure
    sets brightness, contrast stretches around whatever brightness that left.
    """
    x = rgb.astype(np.float32) / 255.0
    if ev:
        x = x * (2.0 ** ev)
    if contrast != 1.0:
        m = float(x.mean())
        x = (x - m) * contrast + m
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def sam2_backend(path, pos, neg, ckpt, cfg, enhance=True, blur=0.0,
                 ev=0.0, contrast=1.0):
    """`blur` is a Gaussian sigma applied last, on what SAM 2 actually sees.

    SAM 2 resizes every input to 1024x1024, so these 500x383 crops reach the
    encoder upsampled 2-2.7x — and bilinear upsampling amplifies sensor noise
    along with everything else. A small sigma spends detail the crop does not
    really have to buy a boundary that is not chasing interpolated grain. It is
    a denoise, not a resolution fix: the fix for resolution is to rectify the
    1280x720 original into something larger than 500x383.
    """
    import torch
    rgb = np.array(Image.open(path).convert("RGB"))
    if enhance:
        # These frames are very low contrast; CLAHE on L keeps them photographic
        # while giving the shadow boundary something to hold on to.
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(3.0, (8, 8)).apply(lab[..., 0])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if ev or contrast != 1.0:
        rgb = expose(rgb, ev, contrast)
    if blur > 0:
        rgb = cv2.GaussianBlur(rgb, (0, 0), blur)
    pts = np.array([[float(x), float(y)] for x, y in list(pos) + list(neg)], np.float32)
    lbl = np.array([1] * len(pos) + [0] * len(neg), np.int32)
    p = sam2_predictor(ckpt, cfg)
    with torch.inference_mode():
        p.set_image(rgb)
        masks, scores, _ = p.predict(point_coords=pts, point_labels=lbl, multimask_output=True)
    return masks[int(np.argmax(scores))].astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", default="Teleops/masks/points.json")
    ap.add_argument("--backend", choices=["geo", "rw", "sam2"], default="geo")
    ap.add_argument("--beta", type=float, default=130.0,
                    help="rw only; higher = the walker respects edges more")
    ap.add_argument("--tol", type=int, default=None,
                    help="geo only; overrides the per-capture edge pull")
    ap.add_argument("--open-r", type=int, default=2)
    ap.add_argument("--checkpoint", default=os.getenv("SAM2_CHECKPOINT", DEFAULT_CKPT))
    ap.add_argument("--config", default=os.getenv("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml"))
    ap.add_argument("--enhance", action="store_true",
                    help="CLAHE the photo before segmenting it. Off by default, "
                         "matching auto_segment_teleop.py.")
    ap.add_argument("--no-enhance", action="store_true",
                    help="kept so old commands still run; enhancement is now off "
                         "by default, so this is a no-op unless --enhance is given.")
    ap.add_argument("--blur", type=float, default=0.0,
                    help="sam2 only: Gaussian sigma on the image SAM 2 sees. 0 = off.")
    ap.add_argument("--ev", type=float, default=0.0,
                    help="sam2 only: exposure in stops. +0.3 is about a third of a stop.")
    ap.add_argument("--contrast", type=float, default=1.0,
                    help="sam2 only: contrast gain about the frame mean. 1.0 = off.")
    ap.add_argument("--out", default="Teleops/masks")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    pts = json.load(open(os.path.join(rc.BENCH, a.points)))
    man = json.load(open(os.path.join(rc.BENCH, "Teleops", "masks", "manifest.json")))
    by_cap = {r["capture"]: r for r in man["records"]}

    done, skipped, rows = 0, 0, []
    for cap, d in sorted(pts.get("captures", pts).items()):
        rec = by_cap.get(cap)
        if rec is None:
            print(f"  ?? unknown capture {cap}"); continue
        pos, neg = d.get("pos", []), d.get("neg", [])
        if not pos:
            skipped += 1; continue
        path = os.path.join(rc.BENCH, rec["rectified"])
        if a.backend == "sam2":
            m = sam2_backend(path, pos, neg, a.checkpoint, a.config,
                             a.enhance and not a.no_enhance,
                             a.blur, a.ev, a.contrast)
        elif a.backend == "geo":
            m = geo_backend(path, pos, neg, a.tol if a.tol is not None else d.get("tol", 55))
        else:
            m = rw_backend(path, pos, neg, a.beta)
        if a.open_r and m.sum():
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*a.open_r+1, 2*a.open_r+1))
            mo = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
            if mo.sum() > 0.2 * m.sum():
                m = mo
        if m.sum():
            m = _denoise(m, float(m.sum()))
        frac = float(m.mean())
        rows.append((cap, frac, rc.n_holes_signif(m), len(pos), len(neg)))
        if a.write:
            rc.save_mask(m, os.path.join(rc.BENCH, a.out, cap + "_mask.png"))
            rec.update(mask_backend=a.backend, n_pos=len(pos), n_neg=len(neg),
                       shape_frac=round(frac, 4), holes_signif=rc.n_holes_signif(m),
                       suspect=False)
        done += 1

    print(f"\nbackend {a.backend}  ·  segmented {done}, skipped {skipped} without points")
    print(f"{'capture':46} {'frac':>6} {'holes':>6} {'+':>3} {'-':>3}")
    for cap, frac, hs, np_, nn in rows:
        print(f"{cap[:46]:46} {frac:6.3f} {hs:6d} {np_:3d} {nn:3d}")
    if a.write:
        json.dump(man, open(os.path.join(rc.BENCH, "Teleops", "masks", "manifest.json"), "w"), indent=1)
        print(f"\nwrote {done} masks + updated manifest.json")
    else:
        print("\ndry run — pass --write to save masks")


if __name__ == "__main__":
    main()
