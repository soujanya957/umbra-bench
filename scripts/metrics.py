"""Target-vs-shadow evaluation metrics for umbra-bench.

IoU is the only metric this benchmark shipped with, and on this data it mostly
measures how fat the target is: across the 546-target big-budget sweep it
correlates rho = +0.64 with median stroke width and +0.51 with area fraction. The
top-scoring samples are `hcircle`, `jar`, `bell` -- convex blobs with no semantic
content -- while `vehicles` tops out at 0.69 because wheels and window frames are
thin. A metric that ranks a featureless disc above a recognisable car is not
measuring what a shadow-art system is for.

So the metrics here are organised by *what question they answer*, and are meant to
be reported together rather than collapsed into one number:

  overlap      iou, dice                    how much area agrees
  boundary     boundary_iou, nsd, chamfer,   where the outline sits -- insensitive
               hd95                          to how much torso is filling the middle
  thin-structure  cldice                     did the thin parts survive at all
  topology     betti_error, hole/part        is it still the same *kind* of shape
               agreement, persistence
               distances
  placement    limb_match                    did the protrusions land in the right
                                             places, not just in the right amount
  descriptor   hu_distance, fourier_distance classical shape-retrieval distances,
                                             comparable to the MPEG-7 literature
  attribute    attribute_delta (in           interpretable, per-property error
               shape_attributes.py)

Alignment: every function takes masks already in the same frame and does *not*
align them, except `aligned_iou`, which searches over similarity transforms. The
gap between `iou` and `aligned_iou` is itself informative -- it separates "the
shape is wrong" from "the shape is right but sitting in the wrong place", which for
a robot rig is the difference between an algorithm problem and a calibration one.

Optional dependencies: `gudhi` (+ `POT`) for persistence distances, `scipy.optimize`
for limb matching. Everything degrades to None rather than failing.
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from PIL import Image

from shape_attributes import (
    _as_binary, _denoise, _crossing_number, _pruned_skeleton,
    _hole_stats, LIMB_PRUNE_FRAC, PH_NOISE_FRAC,
)

try:
    import gudhi as _gudhi
    from gudhi.wasserstein import wasserstein_distance as _wd
except ImportError:  # pragma: no cover
    _gudhi = None
    _wd = None

try:
    from skimage.morphology import skeletonize as _skel
except ImportError:  # pragma: no cover
    _skel = None


# --- io -----------------------------------------------------------------------

def load_mask(path: str, size: int | None = None) -> np.ndarray:
    """Read a umbra-bench PNG (black shape on white) as a uint8 {0,1} mask."""
    img = Image.open(path).convert("L")
    if size:
        img = img.resize((size, size), Image.NEAREST)
    return (np.array(img) < 128).astype(np.uint8)


# --- overlap ------------------------------------------------------------------

def iou(gt: np.ndarray, pred: np.ndarray) -> float:
    """Intersection over union. The incumbent metric; see module docstring for why
    it needs company rather than replacement -- it is still the right headline
    number for "how much of the target did we cover"."""
    g, p = _as_binary(gt).astype(bool), _as_binary(pred).astype(bool)
    u = (g | p).sum()
    return float((g & p).sum() / u) if u else 1.0


def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """Dice / F1 on pixels. Monotone in IoU, so it adds no ranking information;
    reported only because the segmentation literature quotes it and readers will
    look for it."""
    g, p = _as_binary(gt).astype(bool), _as_binary(pred).astype(bool)
    d = g.sum() + p.sum()
    return float(2 * (g & p).sum() / d) if d else 1.0


# --- boundary -----------------------------------------------------------------

def _boundary_band(m: np.ndarray, d: int) -> np.ndarray:
    er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=max(1, d))
    return (m & ~er.astype(bool)) if m.dtype == bool else (m - er)


def boundary_iou(gt: np.ndarray, pred: np.ndarray, d_frac: float = 0.02) -> float:
    """IoU restricted to a band of width `d_frac` * diagonal along each outline.

    (Cheng et al., "Boundary IoU", CVPR 2021.) Plain IoU is dominated by whatever
    fills the interior, so a horse whose torso is covered scores well even with no
    legs. Restricting to the boundary band removes the interior's vote, which is
    exactly the stroke-width bias this benchmark suffers from. This is the single
    highest-value addition to the existing metric set.
    """
    g, p = _as_binary(gt), _as_binary(pred)
    d = max(1, int(d_frac * np.hypot(*g.shape)))
    gb, pb = _boundary_band(g, d), _boundary_band(p, d)
    u = ((gb | pb) > 0).sum()
    return float(((gb & pb) > 0).sum() / u) if u else 1.0


def _surface_distances(gt: np.ndarray, pred: np.ndarray):
    """Distances from each outline pixel of one mask to the other's outline."""
    g, p = _as_binary(gt), _as_binary(pred)
    ge = (g - cv2.erode(g, np.ones((3, 3), np.uint8))) > 0
    pe = (p - cv2.erode(p, np.ones((3, 3), np.uint8))) > 0
    if not ge.any() or not pe.any():
        return None, None
    dg = ndimage.distance_transform_edt(~ge)   # distance to gt outline
    dp = ndimage.distance_transform_edt(~pe)   # distance to pred outline
    return dp[ge], dg[pe]                      # gt->pred, pred->gt


def chamfer(gt: np.ndarray, pred: np.ndarray) -> float:
    """Mean symmetric outline-to-outline distance, as a fraction of the diagonal.

    A continuous error: unlike IoU it keeps grading after the shapes stop
    overlapping, so a run that misses by a little and one that misses entirely are
    distinguishable. Use it when comparing optimizers that are all bad.
    """
    a, b = _surface_distances(gt, pred)
    if a is None:
        return float("nan")
    return float((a.mean() + b.mean()) / 2 / np.hypot(*np.asarray(gt).shape))


def hd95(gt: np.ndarray, pred: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance, relative to the diagonal.

    Worst-case outline error with the top 5% trimmed so one stray pixel cannot set
    the score. Answers "how far off is the *worst* part of this shadow", which is
    usually what a viewer notices first.
    """
    a, b = _surface_distances(gt, pred)
    if a is None:
        return float("nan")
    v = max(np.percentile(a, 95), np.percentile(b, 95))
    return float(v / np.hypot(*np.asarray(gt).shape))


def nsd(gt: np.ndarray, pred: np.ndarray, tol_frac: float = 0.01) -> float:
    """Normalised Surface Dice: share of outline within `tol_frac` of the other.

    (Nikolov et al. 2018.) Boundary agreement with an explicit tolerance, so the
    number means something physical: at tol_frac = 0.01 on a 1 m screen it reads
    "what fraction of the outline lands within a centimetre". That makes it the
    natural metric to state a physical-world pass/fail criterion in.
    """
    a, b = _surface_distances(gt, pred)
    if a is None:
        return float("nan")
    tol = tol_frac * np.hypot(*np.asarray(gt).shape)
    return float(((a <= tol).sum() + (b <= tol).sum()) / (a.size + b.size))


# --- thin structure -----------------------------------------------------------

def cldice(gt: np.ndarray, pred: np.ndarray) -> float:
    """Centreline Dice (Shit et al., CVPR 2021): harmonic mean of

        precision = fraction of the prediction's skeleton lying inside the target
        recall    = fraction of the target's skeleton lying inside the prediction

    Because it scores skeletons against masks, a limb contributes as much as a
    torso regardless of area -- and it cannot be gamed by thickening, since a
    fattened prediction's skeleton drifts out of the thin target. In 2D it is
    provably homotopy-preserving at 1.0, so it is the closest thing here to a
    differentiable topology score, and the natural candidate for an extra term in
    the optimizer's objective rather than just its report.
    """
    if _skel is None:
        return float("nan")
    g, p = _as_binary(gt).astype(bool), _as_binary(pred).astype(bool)
    sg, sp = _skel(g), _skel(p)
    tprec = sp[g].sum() / sp.sum() if sp.sum() else 0.0
    tsens = sg[p].sum() / sg.sum() if sg.sum() else 0.0
    return float(2 * tprec * tsens / (tprec + tsens)) if (tprec + tsens) else 0.0


# --- topology -----------------------------------------------------------------

def betti_error(gt: np.ndarray, pred: np.ndarray, significant: bool = True) -> dict:
    """|dbeta0| + |dbeta1| between target and shadow.

    (The standard metric of the topology-aware segmentation literature, e.g. Hu et
    al., NeurIPS 2019.) This is the "is an 8 still an 8" test: filling both eyes of
    an 8 costs a few points of IoU and destroys the character. Nothing else in the
    metric set detects it.

    `significant=True` uses hole counts thresholded by area, which is essential on
    real data -- the raw count on the MPEG-7 horse target is 18, all of them
    binarization litter under 0.06% of its area.
    """
    g, p = _as_binary(gt), _as_binary(pred)
    ga, pa = float(g.sum()), float(p.sum())
    gh = _hole_stats(g, ga)[1 if significant else 0]
    ph = _hole_stats(p, pa)[1 if significant else 0]
    gc = ndimage.label(_denoise(g, ga) if significant else g)[1]
    pc = ndimage.label(_denoise(p, pa) if significant else p)[1]
    return {
        "d_betti0": pc - gc,
        "d_betti1": ph - gh,
        "betti_error": abs(pc - gc) + abs(ph - gh),
    }


def _diagram(m: np.ndarray):
    """Signed-distance persistence diagram of a mask, normalised by the diagonal."""
    m = _as_binary(m)
    diag = float(np.hypot(*m.shape))
    din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    cc = _gudhi.CubicalComplex(top_dimensional_cells=(dout - din).astype(np.float64) / diag)
    cc.compute_persistence(homology_coeff_field=2)
    out = {}
    for dim in (0, 1):
        a = np.asarray(cc.persistence_intervals_in_dimension(dim), dtype=float)
        a = a.reshape(-1, 2) if a.size else np.zeros((0, 2))
        a = a[np.isfinite(a[:, 1])]
        out[dim] = a[(a[:, 1] - a[:, 0]) >= PH_NOISE_FRAC] if len(a) else a
    return out


def persistence_distance(gt: np.ndarray, pred: np.ndarray, order: float = 1.0) -> dict:
    """Wasserstein distance between the two masks' persistence diagrams, per dimension.

    The graded version of `betti_error`. Betti error is a step function: a hole that
    is 99% closed and one that is wide open score identically, and a single pixel
    can flip the count. Wasserstein distance instead pays the *cost of the edit* --
    a barely-lost hole is cheap, a missing one is expensive -- so it stays stable
    on noisy physical captures where integer topology counts jitter every frame.

    H0 captures parts and their separations, H1 captures loops. Returns None
    without gudhi + POT.
    """
    if _gudhi is None or _wd is None:
        return {"pw_h0": None, "pw_h1": None}
    dg, dp = _diagram(gt), _diagram(pred)
    return {
        "pw_h0": round(float(_wd(dg[0], dp[0], order=order, internal_p=2)), 5),
        "pw_h1": round(float(_wd(dg[1], dp[1], order=order, internal_p=2)), 5),
    }


# --- placement ----------------------------------------------------------------

def limb_match(gt: np.ndarray, pred: np.ndarray, prune_frac: float = LIMB_PRUNE_FRAC) -> dict:
    """Hungarian matching between the two shapes' pruned-skeleton endpoints.

    Counting limbs on each side says whether the right *number* of protrusions came
    out; this says whether they came out in the right *places*. A four-legged
    shadow whose legs point the wrong way scores well on every count-based metric
    and reads as wrong instantly. Returns mean matched offset (fraction of the
    diagonal), plus how many endpoints went unmatched in each direction.
    """
    g, p = _as_binary(gt), _as_binary(pred)
    diag = float(np.hypot(*g.shape))
    out = {}
    pts = []
    for m in (g, p):
        sk = _pruned_skeleton(_denoise(m, float(m.sum())), prune_frac * diag)
        ys, xs = np.nonzero(_crossing_number(sk) == 1)
        pts.append(np.stack([ys, xs], 1).astype(float))
    a, b = pts
    out["n_limbs_target"], out["n_limbs_shadow"] = len(a), len(b)
    if len(a) == 0 or len(b) == 0:
        out["limb_offset_rel"] = None
        out["limbs_unmatched"] = abs(len(a) - len(b))
        return out
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    out["limb_offset_rel"] = round(float(cost[ri, ci].mean() / diag), 4)
    out["limbs_unmatched"] = abs(len(a) - len(b))
    return out


# --- descriptors --------------------------------------------------------------

def hu_distance(gt: np.ndarray, pred: np.ndarray, n_moments: int = 4) -> float:
    """Log-scaled L1 distance between the first `n_moments` Hu invariants.

    Invariant to translation, scale and rotation, so it isolates "is this the same
    shape" from "is it in the same place at the same size" -- which for a shadow rig
    with an uncalibrated throw distance is a distinction worth measuring separately.

    Truncated at 4 on purpose: Hu's high-order invariants sit near zero for most
    silhouettes, so on a log scale their noise dominates. Keeping all 7 gave the
    digit '0' a distance of 50 against a shadow that had reproduced it well, versus
    0.7 for a comparable letter -- two orders of magnitude of pure numerical
    artifact.
    """
    g, p = _as_binary(gt), _as_binary(pred)
    hg = cv2.HuMoments(cv2.moments(g)).flatten()[:n_moments]
    hp = cv2.HuMoments(cv2.moments(p)).flatten()[:n_moments]
    lg = np.sign(hg) * np.log10(np.abs(hg) + 1e-30)
    lp = np.sign(hp) * np.log10(np.abs(hp) + 1e-30)
    return float(np.abs(lg - lp).sum())


def fourier_distance(gt: np.ndarray, pred: np.ndarray, n_harm: int = 32) -> float:
    """L2 distance between scale/rotation/start-point-normalised Fourier descriptors.

    The classical contour-retrieval distance, and the reason it is here is
    comparability: MPEG-7 is a retrieval benchmark with a published protocol, so a
    descriptor distance lets shadow-as-query results be quoted next to forty years
    of shape-matching literature instead of only against ourselves.
    """
    def desc(m):
        c, _ = cv2.findContours(_as_binary(m), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(c, key=len).reshape(-1, 2)
        z = cnt[:, 0] + 1j * cnt[:, 1]
        idx = np.linspace(0, len(z) - 1, 256).astype(int)
        Z = np.fft.fft(z[idx].astype(np.complex128))
        Z[0] = 0                                  # translation
        mag = np.abs(Z[1:n_harm + 1])
        return mag / (mag[0] + 1e-12)             # scale; magnitude drops rotation/phase
    return float(np.linalg.norm(desc(gt) - desc(pred)))


# --- alignment ----------------------------------------------------------------

def _trim(m: np.ndarray):
    ys, xs = np.where(m)
    if not len(ys):
        return None
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def trimmed_centred(gt: np.ndarray, pred: np.ndarray, height: int = 256,
                    pad: int = 6) -> dict:
    """IoU with position and size removed: trim to ink, match height, centre.

    `iou` answers "did the shadow land on the target". On a --fit-target sweep
    that is dominated by the fit itself -- the target is deliberately scaled and
    shifted before anyone solves for it -- so the number mostly reports how far
    the fit moved, not how good the shape is. This answers the other question:
    ignoring where it was placed and how big it was drawn, is it the right shape?

    Height, not width or area, sets the scale. Measured over 571 solves against
    an optimal scale found by search, height lands 0.0200 short of the ceiling
    and within 0.02 of it on 65% of targets; max-side ties it (0.0201 / 67%),
    diagonal is close (0.0221), pixel-area is worse (0.0300) and width is much
    worse (0.0528). Height is also the one that keeps aspect error honest: width
    follows the same factor, so a shadow that is relatively too wide stays too
    wide and is still penalised, which a per-axis fit would hide.

    Returns `tc_iou` plus the two aspect ratios and their relative error, since a
    high `tc_iou` with a large `tc_aspect_error` is a different failure from a
    low one -- the first is a stretched shape, the second a wrong one.
    """
    a, b = _trim(np.asarray(gt, bool)), _trim(np.asarray(pred, bool))
    if a is None or b is None:
        return {"tc_iou": None, "tc_aspect_target": None,
                "tc_aspect_shadow": None, "tc_aspect_error": None}

    def to_h(m):
        h, w = m.shape
        W = max(1, int(round(w * height / h)))
        return np.asarray(Image.fromarray(m).resize((W, height), Image.NEAREST))

    A, B = to_h(a), to_h(b)
    H = height + 2 * pad
    W = max(A.shape[1], B.shape[1]) + 2 * pad

    def place(m):
        o = np.zeros((H, W), bool)
        y0, x0 = (H - m.shape[0]) // 2, (W - m.shape[1]) // 2
        o[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
        return o

    at, as_ = a.shape[1] / a.shape[0], b.shape[1] / b.shape[0]
    return {"tc_iou": round(iou(place(A), place(B)), 4),
            "tc_aspect_target": round(at, 4),
            "tc_aspect_shadow": round(as_, 4),
            "tc_aspect_error": round(abs(as_ - at) / at, 4) if at else None}


def aligned_iou(gt: np.ndarray, pred: np.ndarray,
                scales=(0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15),
                max_shift_frac: float = 0.12) -> dict:
    """Best IoU over scale + translation, and the gap to unaligned IoU.

    Shadow geometry couples size to throw distance, so an uncalibrated rig loses IoU
    for a reason that has nothing to do with the solve. Separating the two tells you
    which problem to fix: a large `align_gain` means the pose is right and the
    calibration is off; a small one means the shape itself is wrong. Report both --
    aligned IoU alone would quietly forgive real scale errors.
    """
    g, p = _as_binary(gt), _as_binary(pred)
    h, w = g.shape
    step = max(1, int(0.01 * max(h, w)))
    lim = int(max_shift_frac * max(h, w))
    base = iou(g, p)
    best, bs, bd = base, 1.0, (0, 0)
    ys, xs = np.nonzero(p)
    cy, cx = ys.mean(), xs.mean()
    for s in scales:
        M = np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])
        ps = cv2.warpAffine(p, M, (w, h), flags=cv2.INTER_NEAREST)
        for dy in range(-lim, lim + 1, step):
            for dx in range(-lim, lim + 1, step):
                q = np.roll(np.roll(ps, dy, 0), dx, 1)
                v = iou(g, q)
                if v > best:
                    best, bs, bd = v, s, (dy, dx)
    return {"iou": round(base, 4), "aligned_iou": round(best, 4),
            "align_gain": round(best - base, 4),
            "align_scale": bs, "align_shift_rel": (round(bd[0] / h, 4), round(bd[1] / w, 4))}


# --- driver -------------------------------------------------------------------

def all_metrics(gt: np.ndarray, pred: np.ndarray, with_alignment: bool = False) -> dict:
    """Every pairwise metric in this module, flat. `with_alignment` is ~40x slower."""
    out = {
        "iou": round(iou(gt, pred), 4),
        "dice": round(dice(gt, pred), 4),
        "boundary_iou": round(boundary_iou(gt, pred), 4),
        "nsd": round(nsd(gt, pred), 4),
        "chamfer": round(chamfer(gt, pred), 5),
        "hd95": round(hd95(gt, pred), 5),
        "cldice": round(cldice(gt, pred), 4),
        "hu_distance": round(hu_distance(gt, pred), 4),
        "fourier_distance": round(fourier_distance(gt, pred), 4),
    }
    out.update(betti_error(gt, pred))
    out.update(persistence_distance(gt, pred))
    out.update(limb_match(gt, pred))
    # Cheap (two bbox crops and a resize), so it is always on rather than behind
    # --align: on a fitted sweep it is the only pairwise number that is about the
    # shape rather than about the fit.
    out.update(trimmed_centred(gt, pred))
    if with_alignment:
        out.update({k: v for k, v in aligned_iou(gt, pred).items() if k != "iou"})
    return out
