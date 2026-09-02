#!/usr/bin/env python3
"""RealSense capture -> binary shadow, in one pass -- the v2 teleop pipeline.

The v1 rectification (capture_teleop.py) warps the tags' INNER corners to
500x383, so the AprilTags sit outside the output and anything cast near the
tag line is cropped away with them. This rewrite changes the framing rule and
moves the tag problem to where it belongs:

  * the crop is the tags' OUTER frame -- the quad through each tag's corner
    farthest from the four-tag centroid -- so the frame CONTAINS the tags and
    nothing near the boundary is lost at framing time;
  * the four tag regions are then excluded at SEGMENTATION time instead: their
    rectified footprints (dilated) are masked out of the Otsu statistics and
    forced to background in the output, so the tags can never appear as
    corner blobs in the binary shadow.

Measured constants, not chosen ones: tag family 36h11 with ids 0..3 running
clockwise from top-left (29/29 captures agree); the outer quad's aspect is
1.1956 +/- 0.0014 across all 29 captures, so the output is pinned to 560x468
the same way v1 pinned 500x383 -- one canonical frame, no per-capture drift.

Everything is written beside the v1 outputs, never over them: the 29 legacy
rectified images are the frame the existing masks, manifest and metrics were
made in, and that contract stays intact.

    Teleops/rectified2/<stem>_rectified.png    the outer-frame warp, tags visible
    Teleops/rectified2/<stem>_capture.json     quad, tag footprints, sizes
    Teleops/rectified2/masks/<stem>_mask.png   binary shadow, dark = shadow
    Teleops/rectified2/masks/manifest.json     per-capture stats
    Teleops/rectified2/_contact_sheet.png      raw | rectified | mask, per capture

    python scripts/teleop_pipeline.py                  # every raw capture
    python scripts/teleop_pipeline.py --images "Teleops/digits*.png"
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binarize_teleop import flatten  # noqa: E402  (shared flat-field, one definition)
from shape_attributes import _denoise  # noqa: E402
from auto_segment_teleop import auto_seeds  # noqa: E402
from segment_teleop import _CKPT_CANDIDATES, flatfield8, sam2_backend  # noqa: E402

SAM_CFG = os.getenv("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml")

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
TELEOP = os.path.join(_BENCH, "Teleops")

TAG_DICT = cv2.aruco.DICT_APRILTAG_36h11
TAG_IDS = (0, 1, 2, 3)          # clockwise from top-left; 29/29 captures agree
OUT_W, OUT_H = 654, 548         # pinned from the measured outer-quad aspect 1.1956;
                                # matches the pre-existing tag_rectified/ images
                                # (my warp reproduces them to ~2 grey levels)
TAG_DILATE = 6                  # px of margin around each tag footprint


def detector():
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(TAG_DICT))


def find_outer_quad(gray: np.ndarray, det=None):
    """The quad through each tag's OUTER corner, ordered by tag id (TL TR BR BL).

    Returns (quad 4x2 float32, {tag_id: 4x2 tag corner polygon}) or (None, ids
    actually found) when any of the four tags is missing -- a capture without
    all four has no defined frame and is refused, matching v1's behaviour.
    """
    det = det or detector()
    corners, ids, _ = det.detectMarkers(gray)
    found = set() if ids is None else set(int(i) for i in ids.flatten())
    if not set(TAG_IDS) <= found:
        return None, found
    centre = np.mean([c.mean(axis=1)[0] for c in corners], axis=0)
    outer, polys = {}, {}
    for tid, c in zip(ids.flatten(), corners):
        pts = c[0]
        outer[int(tid)] = pts[np.argmax(np.linalg.norm(pts - centre, axis=1))]
        polys[int(tid)] = pts
    quad = np.array([outer[i] for i in TAG_IDS], dtype=np.float32)
    return (quad, polys)


def rectify(img: np.ndarray, quad: np.ndarray, polys: dict):
    """Warp the outer frame to the pinned size; carry the tag footprints along."""
    dst = np.array([[0, 0], [OUT_W, 0], [OUT_W, OUT_H], [0, OUT_H]], np.float32)
    H = cv2.getPerspectiveTransform(quad, dst)
    out = cv2.warpPerspective(img, H, (OUT_W, OUT_H))
    tag_polys = {
        tid: cv2.perspectiveTransform(p.reshape(-1, 1, 2).astype(np.float32), H)
             .reshape(-1, 2)
        for tid, p in polys.items()
    }
    return out, tag_polys


def inner_region(tag_polys: dict) -> np.ndarray:
    """uint8 mask of the projection area proper: the tags' inner-corner quad.

    The outer frame deliberately contains the tags AND a margin -- and in some
    captures a sliver of the table below the wall. Otsu statistics taken over
    that whole frame split wall-vs-table instead of wall-vs-shadow and the
    faint shadow vanishes (digits1: thr 0.883, mask = one table sliver). The
    projection area is where the light-vs-shadow contrast lives, so the
    threshold is learned there and applied everywhere.
    """
    centre = np.mean([p.mean(axis=0) for p in tag_polys.values()], axis=0)
    inner = []
    for tid in TAG_IDS:
        pts = tag_polys[tid]
        inner.append(pts[np.argmin(np.linalg.norm(pts - centre, axis=1))])
    m = np.zeros((OUT_H, OUT_W), np.uint8)
    cv2.fillPoly(m, [np.round(np.array(inner)).astype(np.int32)], 1)
    return m


def tag_exclusion(tag_polys: dict, dilate: int = TAG_DILATE) -> np.ndarray:
    """uint8 mask, 1 where a (dilated) tag footprint sits in the rectified frame."""
    m = np.zeros((OUT_H, OUT_W), np.uint8)
    for p in tag_polys.values():
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    if dilate:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2)
        m = cv2.dilate(m, k)
    return m


def binarize_v2(rect_path: str, excl: np.ndarray, inner: np.ndarray,
                open_r: int = 2, border: int = 6, inner_min_frac: float = 0.5):
    """binarize_teleop.binarize, adapted to the wider v2 frame.

    Three departures, each answering a failure the wider frame introduces:
    the Otsu threshold is learned on the projection area only (inner quad
    minus tags) but applied to the whole frame, so faint shadows are not
    drowned by the table sliver's wall-vs-table split; the (dilated) tag
    footprints are excluded from statistics and forced to background, so
    near-black tags can never survive as corner blobs; and connected
    components lying mostly outside the projection area (< inner_min_frac
    of their pixels inside) are dropped -- a shadow may bleed past the tag
    line, but a component that mostly lives outside it is furniture.
    """
    flat = flatten(rect_path)
    stats = (inner == 1) & (excl == 0)
    thr = float(threshold_otsu(flat[stats]))
    m = ((flat < thr) & (excl == 0)).astype(np.uint8)
    m[:border] = 0; m[-border:] = 0; m[:, :border] = 0; m[:, -border:] = 0
    n, lab = cv2.connectedComponents(m)
    for i in range(1, n):
        comp = lab == i
        if inner[comp].mean() < inner_min_frac:
            m[comp] = 0
    if open_r:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_r + 1,) * 2)
        mo = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        if mo.sum() > 0.2 * m.sum():
            m = mo
    if m.sum() == 0:
        return m, thr, flat
    return _denoise(m, float(m.sum())), thr, flat


def _postprocess(m: np.ndarray, excl: np.ndarray, inner: np.ndarray,
                 open_r: int = 2, border: int = 6, inner_min_frac: float = 0.5):
    """The shared tail: tags out, border out, off-screen components out."""
    m = (m & (excl == 0)).astype(np.uint8)
    m[:border] = 0; m[-border:] = 0; m[:, :border] = 0; m[:, -border:] = 0
    n, lab = cv2.connectedComponents(m)
    for i in range(1, n):
        comp = lab == i
        if inner[comp].mean() < inner_min_frac:
            m[comp] = 0
    if open_r:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_r + 1,) * 2)
        mo = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        if mo.sum() > 0.2 * m.sum():
            m = mo
    if m.sum() == 0:
        return m
    return _denoise(m, float(m.sum()))


V1_W, V1_H = 500, 383       # capture_teleop's pinned v1 frame


def v1_homography(polys: dict) -> np.ndarray:
    """The v1 (inner-corner) rectification homography, recomputed from the tags.

    points.json holds every capture's hand seeds in v1 rectified coordinates.
    Rather than re-clicking 29 captures, those points ride into the v2 frame
    through raw space: v1 coords -> (inverse v1 H) -> raw -> (v2 H) -> v2.
    Same detection, same tags, both quads -- no new information needed.
    """
    centre = np.mean([p.mean(axis=0) for p in polys.values()], axis=0)
    inner = []
    for tid in TAG_IDS:
        pts = polys[tid]
        inner.append(pts[np.argmin(np.linalg.norm(pts - centre, axis=1))])
    dst = np.array([[0, 0], [V1_W, 0], [V1_W, V1_H], [0, V1_H]], np.float32)
    return cv2.getPerspectiveTransform(np.array(inner, np.float32), dst)


def load_points():
    pj = os.path.join(TELEOP, "masks", "points.json")
    if not os.path.exists(pj):
        return {}
    with open(pj, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("captures", d)


def transfer_points(pts, H1, H2):
    """v1 rectified -> raw -> v2 rectified, for one [(x, y), ...] list."""
    if not pts:
        return []
    a = np.array(pts, np.float32).reshape(-1, 1, 2)
    raw = cv2.perspectiveTransform(a, np.linalg.inv(H1))
    v2 = cv2.perspectiveTransform(raw, H2).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in v2
            if 0 <= x < OUT_W and 0 <= y < OUT_H]


def _resolve_ckpt():
    for c in _CKPT_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def segment_sam2(rect_path: str, excl: np.ndarray, inner: np.ndarray, ckpt: str,
                 hand: dict | None = None):
    """v1's production tier (every shipped mask is auto-sam2), on the v2 frame.

    Seeds come from auto_segment_teleop.auto_seeds on the shared flat field;
    positives landing on a tag or outside the projection area are dropped, and
    each tag centre is added as a negative -- the exclusion expressed as a
    prompt, not only as a postprocess. Divide-by-blur Otsu structurally fails
    on large shadows (the blur window normalises them away: letters_upperN,
    objects_glass), which is exactly why v1 shipped SAM2 masks.
    """
    if hand and hand.get("pos") is not None and not hand["pos"]:
        # a points entry exists but every seed fell outside the frame/inner
        # region -- fall through to auto rather than segmenting from nothing
        hand = None
    if hand and hand.get("pos"):
        # Hand markup wins, exactly as in auto_segment_teleop: these are the
        # dashboard-clicked seeds for the v1 frame, carried across. The faint
        # soft-shadow captures (lowerb: the figure barely rises above wall
        # tone) are the ones auto seeding is structurally fragile on, and the
        # ones a person already disambiguated once.
        pos = [(x, y) for x, y in hand["pos"]
               if inner[int(y), int(x)] == 1 and excl[int(y), int(x)] == 0]
        neg = list(hand.get("neg") or [])
        n_, lab = cv2.connectedComponents(excl)
        for i in range(1, n_):
            ys, xs = np.nonzero(lab == i)
            neg.append((int(xs.mean()), int(ys.mean())))
        if pos:
            return sam2_backend(rect_path, pos, neg, ckpt, SAM_CFG).astype(np.uint8)
    ff = flatfield8(rect_path)
    # The tags are the darkest thing in the v2 frame, so auto_seeds' dark
    # percentile finds THEM, the seed filter then drops those points, and SAM2
    # is left with nothing inside the shadow (objects_glass came out at frac
    # 0.001). Neutralise the tag footprints to the projection area's median
    # before seeding: the darkest-regions question becomes about the shadow
    # again.
    # ...and the same statistics lesson as the Otsu tier, one level up: the
    # percentiles inside auto_seeds are frame-global, and the v2 frame's extra
    # margin shifts them until only the sharpest shadow core seeds, dropping
    # the soft penumbra v1 kept (device9 0.30 -> 0.02). Constant-filling the
    # margin distorts the distribution just as badly (it regressed digits1),
    # so the seeds are computed on the inner region's own bounding box -- the
    # same statistical regime as the v1 frame -- and translated back.
    ff = ff.copy()
    ff[excl == 1] = np.median(ff[(inner == 1) & (excl == 0)])
    ys, xs = np.nonzero(inner)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pos, neg = auto_seeds(ff[y0:y1, x0:x1])
    pos = [(x + x0, y + y0) for x, y in pos]
    neg = [(x + x0, y + y0) for x, y in neg]
    pos = [(x, y) for x, y in pos
           if inner[int(y), int(x)] == 1 and excl[int(y), int(x)] == 0]
    n_, lab = cv2.connectedComponents(excl)
    for i in range(1, n_):
        ys, xs = np.nonzero(lab == i)
        neg.append((int(xs.mean()), int(ys.mean())))
    if not pos:
        return None
    return sam2_backend(rect_path, pos, neg, ckpt, SAM_CFG).astype(np.uint8)


def process(path: str, out_root: str, det=None):
    stem = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path)
    if img is None:
        return dict(capture=stem, error="unreadable")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quad, polys = find_outer_quad(gray, det)
    quad_source = "4tag"
    if quad is None:
        found = polys if isinstance(polys, set) else set()
        median_q = getattr(process, "median_quad", None)
        if (median_q is not None and len(found) == 3
                and not getattr(process, "no_3tag", False)):
            # Three tags seen, camera near-fixed across the session (corner
            # std 2-4 px): fit an affine from the median quad's three matched
            # corners to the detected ones and infer only the missing corner.
            # Recorded as quad_source=3tag-inferred, never silently.
            corners, ids, _ = det.detectMarkers(gray)
            centre = np.mean([c.mean(axis=1)[0] for c in corners], axis=0)
            det_outer, polys = {}, {}
            for tid, c in zip(ids.flatten(), corners):
                pts = c[0]
                det_outer[int(tid)] = pts[
                    np.argmax(np.linalg.norm(pts - centre, axis=1))]
                polys[int(tid)] = pts
            have = sorted(det_outer)
            src = np.array([median_q[i] for i in have], np.float32)
            dst_ = np.array([det_outer[i] for i in have], np.float32)
            A = cv2.getAffineTransform(src, dst_)
            missing = (set(TAG_IDS) - set(have)).pop()
            inferred = (A @ np.array([*median_q[missing], 1.0]))[:2]
            det_outer[missing] = inferred.astype(np.float32)
            # the missing tag's footprint, for exclusion/inner: the session's
            # median polygon for that id, moved by the same affine
            mp = getattr(process, "median_polys", {}).get(missing)
            if mp is not None:
                ones = np.hstack([mp, np.ones((len(mp), 1))])
                polys[missing] = (ones @ A.T).astype(np.float32)
            else:
                return dict(capture=stem,
                            error="3-tag capture but no session median poly")
            quad = np.array([det_outer[i] for i in TAG_IDS], np.float32)
            quad_source = "3tag-inferred"
        else:
            return dict(capture=stem,
                        error=f"tags found: {sorted(found)} (need 0-3)")

    rect, tag_polys = rectify(img, quad, polys)
    dst = np.array([[0, 0], [OUT_W, 0], [OUT_W, OUT_H], [0, OUT_H]], np.float32)
    process.last_H = cv2.getPerspectiveTransform(quad, dst)
    rect_path = os.path.join(out_root, f"{stem}_rectified.png")
    cv2.imwrite(rect_path, rect)

    excl = tag_exclusion(tag_polys)
    inner = inner_region(tag_polys)
    backend, thr = "otsu", None
    mask = None
    if process.ckpt:
        try:
            hand = None
            own = process.set_points.get(stem)
            if own:
                # per-set points.json: clicked on the tag frame itself, used
                # as-is (only bounds/inner-filtered downstream)
                hand = {k: [(float(x), float(y)) for x, y in (own.get(k) or [])
                            if 0 <= x < OUT_W and 0 <= y < OUT_H]
                        for k in ("pos", "neg")}
            else:
                pts = process.points.get(stem)
                if pts:
                    H1, H2 = v1_homography(polys), process.last_H
                    hand = {k: transfer_points(pts.get(k) or [], H1, H2)
                            for k in ("pos", "neg")}
            raw_m = segment_sam2(rect_path, excl, inner, process.ckpt, hand)
            if raw_m is not None:
                mask = _postprocess(raw_m, excl, inner)
                backend = ("sam2-points" if hand and hand.get("pos")
                           else "sam2-auto")
        except Exception as e:
            print(f"  [!] sam2 failed on {stem}: {e}; falling back to otsu")
    if mask is None or mask.sum() == 0:
        mask, thr, _ = binarize_v2(rect_path, excl, inner)
        backend = "otsu"

    masks_dir = process.masks_dir or os.path.join(out_root, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    # dark = shadow on white, the benchmark convention
    Image.fromarray(((1 - mask) * 255).astype(np.uint8)).save(
        os.path.join(masks_dir, f"{stem}_mask.png"))

    with open(os.path.join(out_root, f"{stem}_capture.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "raw": os.path.basename(path),
            "raw_size": list(img.shape[1::-1]),
            "out_size": [OUT_W, OUT_H],
            "tag_family": "36h11",
            "tag_ids": list(TAG_IDS),
            "quad_source": quad_source,
            "quad_outer_corners": quad.tolist(),
            "tag_polys_rectified": {str(t): p.tolist()
                                    for t, p in tag_polys.items()},
            "tag_dilate_px": TAG_DILATE,
        }, f, indent=1)

    rel = lambda q: os.path.relpath(q, _BENCH).replace("\\", "/")
    return dict(capture=stem, mask_backend=backend, quad_source=quad_source,
                otsu_thr=None if thr is None else round(thr, 4),
                shape_frac=round(float(mask.mean()), 4),
                n_components=int(cv2.connectedComponents(mask)[0]) - 1,
                raw=rel(path), rectified=rel(rect_path),
                mask=rel(os.path.join(masks_dir, f"{stem}_mask.png")))


def contact_sheet(rows: list, out_root: str, raw_dir: str, thumb_h: int = 160):
    tiles = []
    for r in rows:
        if "error" in r:
            continue
        stem = r["capture"]
        raw = cv2.imread(os.path.join(raw_dir, f"{stem}.png"))
        if raw is None:
            continue
        rect = cv2.imread(os.path.join(out_root, f"{stem}_rectified.png"))
        mask = cv2.imread(os.path.join(process.masks_dir or
                                       os.path.join(out_root, "masks"),
                                       f"{stem}_mask.png"))
        row = [cv2.resize(x, (int(x.shape[1] * thumb_h / x.shape[0]), thumb_h))
               for x in (raw, rect, mask)]
        tiles.append(cv2.hconcat(row))
    if tiles:
        w = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1],
                                    cv2.BORDER_CONSTANT, value=(30, 30, 30))
                 for t in tiles]
        cv2.imwrite(os.path.join(out_root, "_contact_sheet.png"),
                    cv2.vconcat(tiles))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teleop-root", default=None,
                    help="a dataset dir with raw/ inside (e.g. "
                         "Teleops/teleop_set1); outputs go to its "
                         "tag_rectified/ and masks/")
    ap.add_argument("--images", default=None,
                    help="explicit glob of raw captures (overrides --teleop-root)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--masks-dir", default=None,
                    help="default: <out-dir>/masks, or <root>/masks with "
                         "--teleop-root")
    ap.add_argument("--no-3tag", action="store_true",
                    help="refuse captures with only three tags instead of "
                         "inferring the fourth corner from the session quad")
    ap.add_argument("--backend", choices=["auto", "sam2", "otsu"], default="auto",
                    help="auto = SAM2 when available (v1's production tier), "
                         "otsu fallback")
    ap.add_argument("--no-sheet", action="store_true")
    a = ap.parse_args()

    # Output defaulting follows the set whenever a root is named, whether or
    # not --images narrows the file list -- a two-capture touch-up must land
    # in the set's own tree, not the legacy Teleops/rectified2.
    if a.teleop_root:
        if a.out_dir is None:
            a.out_dir = os.path.join(a.teleop_root, "tag_rectified")
        if a.masks_dir is None:
            a.masks_dir = os.path.join(a.teleop_root, "masks")
    if a.images:
        paths = sorted(glob.glob(a.images))
    elif a.teleop_root:
        paths = sorted(glob.glob(os.path.join(a.teleop_root, "raw", "*.png")))
    else:
        paths = [p for p in sorted(glob.glob(os.path.join(TELEOP, "*.png")))
                 if "_rectified" not in p and "_mask" not in p
                 and "montage" not in p]
    if a.out_dir is None:
        a.out_dir = os.path.join(TELEOP, "rectified2")
    if not paths:
        raise SystemExit("[!] no raw captures matched")

    os.makedirs(a.out_dir, exist_ok=True)
    det = detector()
    process.ckpt = None if a.backend == "otsu" else _resolve_ckpt()
    process.points = load_points()
    sp = (os.path.join(a.teleop_root, "points.json") if a.teleop_root else None)
    if sp and os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            d = json.load(f)
        process.set_points = d.get("captures", d)
        print(f"[teleop-v2] per-set points: {len(process.set_points)} captures")
    else:
        process.set_points = {}
    process.no_3tag = a.no_3tag
    process.masks_dir = a.masks_dir
    # session median quad, for 3-tag inference
    qs, polys_acc = [], {i: [] for i in TAG_IDS}
    for p_ in paths:
        g = cv2.imread(p_)
        if g is None: continue
        q, pl = find_outer_quad(cv2.cvtColor(g, cv2.COLOR_BGR2GRAY), det)
        if q is not None:
            qs.append(q)
            for i in TAG_IDS: polys_acc[i].append(pl[i])
    process.median_quad = np.median(np.stack(qs), axis=0) if len(qs) >= 3 else None
    process.median_polys = ({i: np.median(np.stack(v), axis=0)
                             for i, v in polys_acc.items()} if len(qs) >= 3 else {})
    if a.backend == "sam2" and not process.ckpt:
        raise SystemExit("[!] --backend sam2 but no checkpoint found")
    if process.ckpt is None and a.backend == "auto":
        print("[!] no SAM2 checkpoint found; running the otsu tier only")
    rows, skipped = [], []
    for p in paths:
        r = process(p, a.out_dir, det)
        (skipped if "error" in r else rows).append(r)
        tag = r.get("error") or (f"{r['mask_backend']:9} frac {r['shape_frac']:.3f}")
        print(f"  {r['capture']:44} {tag}")

    # A partial run (--images) must not shrink the set's records to the subset
    # it processed: merge by capture stem into the existing manifest, so
    # relabelling one capture rewrites one record and leaves the other thirty.
    man_path = os.path.join(a.masks_dir or os.path.join(a.out_dir, "masks"),
                            "manifest.json")
    n_run = len(rows)
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as f:
            prev = json.load(f)
        done = {r["capture"] for r in rows}
        rows = ([r for r in prev.get("records", []) if r["capture"] not in done]
                + rows)
        rows.sort(key=lambda r: r["capture"])
        skipped = ([r for r in prev.get("skipped", [])
                    if r.get("capture") not in done] + skipped)

    # labels.csv -- the machine-readable class extraction the capture names
    # encode two different ways: set1's long stems resolve through the
    # selection CSVs (binarize_teleop.match, the v1 mechanism, which also
    # yields the benchmark sample_id), set2's short stems parse as
    # teleop_<class>_<take>.
    import csv as _csv
    import re as _re
    from binarize_teleop import load_selection, resolve as _resolve
    sel = load_selection()
    lab_path = os.path.join(a.masks_dir or a.out_dir, "..", "labels.csv")         if a.teleop_root else os.path.join(a.out_dir, "labels.csv")
    lab_path = os.path.normpath(os.path.join(a.teleop_root, "labels.csv"))         if a.teleop_root else os.path.normpath(lab_path)
    with open(lab_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["capture", "class", "take", "sample_id",
                    "quad_source", "mask_backend", "shape_frac", "mask"])
        for r in rows:
            stem = r["capture"]
            cls, take, sid = "", "", ""
            m2 = _re.match(r"^teleop_([A-Za-z0-9]+)_(\d+)$", stem)
            if m2:
                cls, take = m2.group(1), m2.group(2)
            else:
                hit = _resolve(stem, sel)
                cls = hit.get("class") or ""
                sid = hit.get("sample_id") or ""
            r["class"], r["sample_id"] = cls, sid
            w.writerow([stem, cls, take, sid, r.get("quad_source", ""),
                        r["mask_backend"], r["shape_frac"], r["mask"]])
    print(f"[teleop-v2] labels -> {lab_path}")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump({"out_size": [OUT_W, OUT_H], "tag_dilate_px": TAG_DILATE,
                   "n": len(rows), "records": rows, "skipped": skipped},
                  f, indent=1)
    if not a.no_sheet:
        raw_dir = (os.path.join(a.teleop_root, "raw") if a.teleop_root
                   else os.path.dirname(paths[0]))
        contact_sheet(rows, a.out_dir, raw_dir)
    print(f"[teleop-v2] {n_run} processed ({len(rows)} in manifest), "
          f"{len(skipped)} skipped -> {a.out_dir}")


if __name__ == "__main__":
    main()
