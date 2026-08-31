"""Shape attribute computation for umbra-bench masks.

All functions take a binary mask (H, W) uint8 {0, 255} or bool, **white = shape**.
(Targets on disk are black-on-white, so callers invert: `mask = img < 128`.)

Attributes fall into four groups; see METRICS.md for the full definition of each
and what it is diagnostically for.

  scale/geometry   area_frac, aspect_ratio, solidity, compactness, stroke widths
  thinness         thin_mass_frac, elongation, neck_width_rel   <- strongest IoU predictors
  topology         n_components, n_holes, euler_number, hole areas, persistent homology
  structure        n_limbs, n_junctions, symmetry, contour frequency content

The same function runs on a *shadow* mask, which is the point: attribute deltas
(shadow minus target) are interpretable error metrics -- "lost 0.7 holes", "gained
18% stroke width" -- in a way a single IoU number is not.

Persistent-homology fields require `gudhi`. Without it they come back as None and
everything else still computes, so the module has no hard dependency on it.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

try:  # optional: topological summaries
    import gudhi as _gudhi
except ImportError:  # pragma: no cover
    _gudhi = None

# --- tunables -----------------------------------------------------------------
# All lengths are expressed as a fraction of the image diagonal, so these are
# resolution-independent.
THIN_HALFWIDTH_FRAC = 0.03   # "thin" = local half-width below 3% of the diagonal
LIMB_PRUNE_FRAC = 0.08       # skeleton branches shorter than this are noise, not limbs
HOLE_SIGNIF_FRAC = 0.005     # a hole counts as real at >=0.5% of shape area
CURV_SMOOTH_FRAC = 0.02      # contour smoothing sigma before curvature extrema
CURV_MIN_DEPTH = 0.15        # relative curvature threshold for a "part boundary"
PH_ROBUST_FRAC = 0.01        # a persistent loop is "robust" above this lifetime
PH_NOISE_FRAC = 0.002        # persistence below this is binarization noise, not shape
PH_POCKET_FRAC = 0.02        # a concavity counts as a pocket above this mouth width
COMP_SIGNIF_FRAC = 0.005     # a component counts as real at >=0.5% of shape area


# --- helpers ------------------------------------------------------------------

def _as_binary(mask: np.ndarray) -> np.ndarray:
    """Accept bool, {0,1} uint8, or {0,255} uint8 and return {0,1} uint8.

    The {0,1} case has to be detected rather than thresholded at 127: callers
    inside this repo build masks as `img < 128`, and thresholding those again
    silently produces an all-zero mask.
    """
    a = np.asarray(mask)
    if a.dtype == bool:
        m = a.astype(np.uint8)
    elif a.max() <= 1:
        m = (a > 0).astype(np.uint8)
    else:
        m = (a > 127).astype(np.uint8)
    if m.sum() == 0:
        raise ValueError("empty mask")
    return m


def _denoise(m: np.ndarray, area: float) -> np.ndarray:
    """Fill sub-threshold holes and drop sub-threshold components.

    MPEG-7 silhouettes carry binarization litter: the horse target has 18 holes, the
    largest covering 0.06% of its area. Left in, that litter turns every
    skeleton-derived attribute into noise (the horse's skeleton becomes a mesh of
    189 junctions with zero endpoints, so its four legs vanish from the limb count).
    Structural attributes are therefore computed on this cleaned mask, while the raw
    `n_holes` / `n_components` are still reported from the original -- the gap
    between raw and `*_signif` counts is itself a data-quality signal.
    """
    lab, n = ndimage.label(m)
    if n > 1:
        keep = [i for i in range(1, n + 1)
                if (lab == i).sum() >= COMP_SIGNIF_FRAC * area]
        m = np.isin(lab, keep).astype(np.uint8) if keep else m
    bg, nb = ndimage.label(1 - m)
    border = set(np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]]))) - {0}
    fill = [i for i in range(1, nb + 1)
            if i not in border and (bg == i).sum() < HOLE_SIGNIF_FRAC * area]
    if fill:
        m = (m | np.isin(bg, fill)).astype(np.uint8)
    return m


def _cluster_count(pix: np.ndarray) -> int:
    """Connected-component count of a pixel set.

    Skeleton junctions are 2-4 pixel clusters, not single pixels, so counting
    junction *pixels* over-reports by ~4x. Counting clusters is what makes
    `n_junctions` comparable between a target and a shadow.
    """
    return int(ndimage.label(pix, structure=np.ones((3, 3)))[1])


def _pruned_skeleton(m: np.ndarray, prune_px: float) -> np.ndarray:
    """Skeleton with branches shorter than `prune_px` removed.

    Repeated endpoint deletion: each pass shortens every branch by one pixel, so
    after `prune_px` passes only longer branches survive. This is what separates
    "the horse has 4 legs" from "the horse's outline has 40 skeleton spurs", i.e.
    the difference between a usable limb count and noise.
    """
    sk = skeletonize(m.astype(bool)).astype(np.uint8)
    for _ in range(int(prune_px)):
        ends = _crossing_number(sk) == 1
        if not ends.any():
            break
        sk[ends] = 0
    return sk


# circular order of the 8 neighbours, used by the crossing number
_RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _crossing_number(sk: np.ndarray) -> np.ndarray:
    """Rutovitz crossing number: 1 = endpoint, 2 = path, >=3 = junction.

    Counting raw 8-neighbours instead over-reports junctions badly on organic
    outlines: a Zhang-Suen skeleton traverses jagged boundaries in staircase steps
    whose pixels legitimately have three neighbours. On the MPEG-7 horse that
    inflates the junction count to 25 on a shape that is topologically a tree. The
    crossing number counts 0->1 transitions around the neighbourhood ring instead,
    so a staircase step still reads as 2.
    """
    s = sk.astype(np.int32)
    nb = [np.roll(np.roll(s, dy, 0), dx, 1) for dy, dx in _RING]
    cn = sum(np.abs(nb[i] - nb[(i + 1) % 8]) for i in range(8)) // 2
    return cn * s


def _endpoints_junctions(sk: np.ndarray) -> tuple[int, int]:
    cn = _crossing_number(sk)
    return _cluster_count(cn == 1), _cluster_count(cn >= 3)


def _reflection_iou(m: np.ndarray, axis: int) -> float:
    """Self-IoU under reflection about the centroid -- 1.0 = perfectly symmetric."""
    h, w = m.shape
    ys, xs = np.nonzero(m)
    M = np.float32([[1, 0, w / 2 - xs.mean()], [0, 1, h / 2 - ys.mean()]])
    c = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST)
    f = c[:, ::-1] if axis == 1 else c[::-1, :]
    union = float((c | f).sum())
    return float((c & f).sum()) / union if union else 1.0


def _hole_stats(m: np.ndarray, area: float) -> tuple[int, int, float, float]:
    """Hole count plus how much of the shape's area those holes enclose.

    A count alone treats the eye of a '0' and a 3-pixel binarization artifact as
    the same event. `n_holes_signif` and the area fractions are what make hole
    agreement usable as a metric on real (noisy) shadow captures.
    """
    bg, n_bg = ndimage.label(1 - m)
    border = set(np.unique(np.concatenate(
        [bg[0], bg[-1], bg[:, 0], bg[:, -1]]))) - {0}
    hole_ids = [i for i in range(1, n_bg + 1) if i not in border]
    areas = np.array([float((bg == i).sum()) for i in hole_ids], dtype=float)
    n_signif = int((areas >= HOLE_SIGNIF_FRAC * area).sum())
    return (len(hole_ids), n_signif,
            float(areas.sum() / area) if areas.size else 0.0,
            float(areas.max() / area) if areas.size else 0.0)


def _neck_width_rel(m: np.ndarray, diag: float, max_r: int = 40):
    """Width of the narrowest bridge whose removal splits the shape, else None.

    Found by eroding until the component count rises; the erosion radius at that
    moment is the bridge's half-width. This is the topology-fragility number: a
    shape whose parts hang together by a 2%-of-diagonal neck will come apart in a
    real capture (blur, thresholding, a millimetre of arm droop) even when the
    optimizer's rendered IoU looks fine. None = no such bridge within `max_r`.
    """
    base = ndimage.label(m)[1]
    k = np.ones((3, 3), np.uint8)
    for r in range(1, max_r + 1):
        e = cv2.erode(m, k, iterations=r)
        if e.sum() == 0:
            return None
        if ndimage.label(e)[1] > base:
            return round(2.0 * r / diag, 4)
    return None


def _contour_frequency(cnt: np.ndarray, n: int = 256) -> float:
    """Share of Fourier-descriptor energy above harmonic 8 -- contour detail level.

    Low = a smooth blob describable by a few harmonics; high = a spiky outline
    whose identity lives in fine boundary structure. Predicts which targets lose
    their character when a shadow is softened by penumbra.
    """
    z = cnt[:, 0].astype(np.float64) + 1j * cnt[:, 1].astype(np.float64)
    idx = np.linspace(0, len(z) - 1, n).astype(int)
    Z = np.fft.fft(z[idx])
    Z[0] = 0  # drop translation
    mag = np.abs(Z)
    lo = mag[1:n // 2].sum()
    return float(mag[8:n // 2].sum() / lo) if lo > 0 else 0.0


def _concave_extrema(cnt: np.ndarray, diag: float) -> int:
    """Count of deep negative-curvature minima on the outline.

    Hoffman & Richards' minima rule: human vision segments a silhouette into parts
    at curvature minima. So this is a perceptual part count -- how many pieces a
    viewer reads the shape as having -- which is a different question from how many
    skeleton limbs it has, and closer to what a recognizability study measures.
    """
    if len(cnt) < 16:
        return 0
    p = cnt.astype(np.float64)
    sig = max(1.0, CURV_SMOOTH_FRAC * diag)
    p = np.stack([ndimage.gaussian_filter1d(p[:, i], sig, mode="wrap") for i in (0, 1)], 1)
    d1 = np.gradient(p, axis=0)
    d2 = np.gradient(d1, axis=0)
    denom = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5
    curv = np.divide(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0], denom,
                     out=np.zeros_like(denom), where=denom > 1e-9)
    peak = np.abs(curv).max()
    if not np.isfinite(peak) or peak <= 0:
        return 0
    below = (curv / peak) < -CURV_MIN_DEPTH
    # one run of consecutive below-threshold samples = one concavity
    return int(np.sum(below & ~np.roll(below, 1)))


def _persistence(m: np.ndarray, diag: float) -> dict:
    """Persistent homology of the signed-distance sublevel filtration.

    Betti numbers answer "is there a hole" with a yes/no that a single pixel can
    flip. Persistence answers "how *robustly* is there a hole": each feature gets a
    lifetime equal to the width of the material sustaining it, so the eye of an '8'
    (lifetime ~ the stroke width around it) and a hairline crack (lifetime ~ 0) stop
    being the same event. That grading is what makes topology comparable across a
    clean render and a noisy physical capture.

    Filtration: sublevel sets of the signed distance transform (negative inside), so
    level t is the shape dilated by t (t>0) or eroded by |t| (t<0). Then:

      H0 lifetime      = part inradius + gap to the nearest other part
                         -> how separable a part is. Big for a nearly-detached wing,
                            ~0 for a bump on the torso.
      H1, birth <= 0   = a genuine hole of the shape. Its death is the hole's own
                         inradius -> a size-graded hole count.
      H1, birth  > 0   = a *pocket*: a concavity that only becomes a loop once the
                         shape is dilated. Birth is the dilation radius needed to
                         bridge its mouth -> how open the concavity is.

    Both splits matter and conflating them is the easy mistake. H1 *lifetime* is
    also deliberately unused: it equals hole inradius plus surrounding ring
    thickness, so a one-pixel binarization hole inside a thick body would score as
    persistent as the eye of an '8'. Without the birth-sign split, a fork reports 53
    "holes" -- the gaps between its tines sealing shut under dilation.
    All values normalised by the image diagonal.
    """
    keys = ("ph_h0_total", "ph_n_parts_robust", "ph_holes_total_size",
            "ph_hole_max_size", "ph_n_holes_robust",
            "ph_n_pockets_robust", "ph_pocket_max_mouth", "ph_entropy")
    if _gudhi is None:
        return {k: None for k in keys}
    din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    sdt = ((dout - din).astype(np.float64) / diag)

    cc = _gudhi.CubicalComplex(top_dimensional_cells=sdt)
    cc.compute_persistence(homology_coeff_field=2)
    iv = {}
    for dim in (0, 1):
        a = np.asarray(cc.persistence_intervals_in_dimension(dim), dtype=float)
        a = a.reshape(-1, 2) if a.size else np.zeros((0, 2))
        iv[dim] = a[np.isfinite(a[:, 1])]  # drop the essential class

    # Noise band dropped before summarising: persistence entropy counts features, so
    # a few hundred one-pixel artifacts would dominate it through log(n) alone.
    h0 = iv[0][:, 1] - iv[0][:, 0]
    h0 = h0[h0 >= PH_NOISE_FRAC]

    h1 = iv[1]
    holes = h1[h1[:, 0] <= 0][:, 1]                 # birth <= 0: real holes, size = death
    pockets = h1[h1[:, 0] > 0][:, 0]                # birth  > 0: concavities, mouth = birth
    holes = holes[holes >= PH_NOISE_FRAC]
    pockets = pockets[pockets >= PH_NOISE_FRAC]

    out = {
        "ph_h0_total": round(float(h0.sum()), 4),
        "ph_n_parts_robust": int((h0 >= PH_ROBUST_FRAC).sum()),
        "ph_holes_total_size": round(float(holes.sum()), 4),
        "ph_hole_max_size": round(float(holes.max()), 4) if holes.size else 0.0,
        "ph_n_holes_robust": int((holes >= PH_ROBUST_FRAC).sum()),
        "ph_n_pockets_robust": int((pockets >= PH_POCKET_FRAC).sum()),
        "ph_pocket_max_mouth": round(float(pockets.max()), 4) if pockets.size else 0.0,
    }
    allp = np.concatenate([h0, holes])
    if allp.sum() > 0:
        p = allp / allp.sum()
        p = p[p > 0]
        out["ph_entropy"] = round(float(-(p * np.log(p)).sum()), 4) + 0.0
    else:
        out["ph_entropy"] = 0.0
    return out


# --- main ---------------------------------------------------------------------

def compute_attributes(mask: np.ndarray, with_persistence: bool = True) -> dict:
    """Compute all shape attributes for a binary mask (white = shape).

    Field-by-field definitions and the reason each one exists: METRICS.md, Part A.
    Ordering below is scale -> thinness -> topology -> structure; the returned dict
    keeps that order so `metadata.jsonl` stays readable.
    """
    m = _as_binary(mask)
    h, w = m.shape
    area = float(m.sum())
    diag = float(np.hypot(h, w))

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
    main_cnt = max(contours, key=len).reshape(-1, 2)

    # --- scale / global geometry ---
    hull = cv2.convexHull(all_pts)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 1.0

    max_defect = 0.0
    for c in contours:
        if len(c) < 4:
            continue
        hi = cv2.convexHull(c, returnPoints=False)
        if hi is not None and len(hi) > 3:
            try:
                d = cv2.convexityDefects(c, hi)
            except cv2.error:
                continue
            if d is not None and len(d):
                # OpenCV >= 5 returns (N, 4); OpenCV 4 returns (N, 1, 4).
                dd = d.reshape(-1, 4)
                max_defect = max(max_defect, float(dd[:, 3].max()) / 256.0)

    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    compactness = perimeter ** 2 / (4 * np.pi * area)
    (_, (rw, rh), _) = cv2.minAreaRect(all_pts)
    aspect_ratio = float(min(rw, rh) / max(rw, rh)) if max(rw, rh) > 0 else 1.0

    # --- topology (from the RAW mask; the raw-vs-signif gap is a quality signal) ---
    n_components = int(ndimage.label(m)[1])
    n_holes, n_holes_signif, hole_area_frac, hole_area_frac_max = _hole_stats(m, area)

    # --- thinness / structure (from the DENOISED mask) ---
    mc = _denoise(m, area)
    dist = cv2.distanceTransform(mc, cv2.DIST_L2, 5)
    sk_raw = skeletonize(mc.astype(bool))
    skel_d = dist[sk_raw]
    if skel_d.size:
        p05, p10, p50, p90 = (2.0 * float(np.percentile(skel_d, q)) for q in (5, 10, 50, 90))
    else:  # degenerate: a convex blob can skeletonize to nothing
        p05 = p10 = p50 = p90 = 2.0 * float(dist.max())
    min_sw, med_sw = p05, p50
    inscribed_r = float(dist.max())

    thin_mass_frac = float(((dist > 0) & (dist < THIN_HALFWIDTH_FRAC * diag)).sum()
                           / max(1.0, float(mc.sum())))
    skel_len = float(sk_raw.sum())
    elongation = skel_len / med_sw if med_sw > 0 else 0.0

    sk_pruned = _pruned_skeleton(mc, LIMB_PRUNE_FRAC * diag)
    n_limbs, n_junctions = _endpoints_junctions(sk_pruned)

    attrs = {
        # scale / global geometry
        "area_frac": round(area / (h * w), 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "solidity": round(solidity, 4),
        "compactness": round(compactness, 3),
        "convexity_defect_depth_rel": round(max_defect / diag, 4),
        # thinness (the dominant IoU predictors)
        "min_stroke_width_rel": round(min_sw / diag, 4),
        "median_stroke_width_rel": round(med_sw / diag, 4),
        # p10/p90 = width uniformity. Low means fat torso + thin legs, i.e. the
        # case where IoU is bought by the torso while the legs (which carry the
        # identity) are free to disappear.
        "stroke_width_ratio": round(p10 / p90, 4) if p90 > 0 else 1.0,
        "thin_mass_frac": round(thin_mass_frac, 4),
        "elongation": round(elongation, 3),
        "skel_len_rel": round(skel_len / diag, 3),
        "neck_width_rel": _neck_width_rel(m, diag),
        "closed_region": bool(med_sw > 0.5 * inscribed_r),
        # topology
        "n_components": n_components,
        "n_holes": n_holes,
        "n_holes_signif": n_holes_signif,
        "euler_number": n_components - n_holes,
        "hole_area_frac": round(hole_area_frac, 4),
        "hole_area_frac_max": round(hole_area_frac_max, 4),
        # structure
        "n_limbs": n_limbs,
        "n_junctions": n_junctions,
        "n_concave_extrema": _concave_extrema(main_cnt, diag),
        "sym_h": round(_reflection_iou(m, 1), 4),
        "sym_v": round(_reflection_iou(m, 0), 4),
        "contour_hf_energy": round(_contour_frequency(main_cnt), 4),
    }
    if with_persistence:
        attrs.update(_persistence(m, diag))
    return attrs


def attribute_delta(target_attrs: dict, shadow_attrs: dict) -> dict:
    """Per-attribute error, shadow minus target, keyed `d_<name>`.

    The cheapest interpretable metric in the whole benchmark: it reuses the
    attribute code with no new machinery and turns "IoU 0.64" into statements a
    reader can act on -- the optimizer systematically thickens strokes, drops
    holes, loses limbs. Booleans become 0/1; None on either side yields None.
    """
    out = {}
    for k, tv in target_attrs.items():
        sv = shadow_attrs.get(k)
        if tv is None or sv is None:
            out[f"d_{k}"] = None
        elif isinstance(tv, bool) or isinstance(sv, bool):
            out[f"d_{k}"] = int(bool(sv)) - int(bool(tv))
        else:
            out[f"d_{k}"] = round(float(sv) - float(tv), 4)
    return out
