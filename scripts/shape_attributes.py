"""Shape attribute computation for umbra-bench target masks.

All functions take a binary mask (H, W) uint8 {0, 255} or bool, white = shape.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def compute_attributes(mask: np.ndarray) -> dict:
    """Compute all shape attributes for a binary target mask."""
    m = (np.asarray(mask) > 127).astype(np.uint8)
    if m.sum() == 0:
        raise ValueError("empty mask")

    h, w = m.shape
    area = float(m.sum())
    diag = float(np.hypot(h, w))

    # topology
    n_components = int(ndimage.label(m)[1])
    # holes: background components not touching the border
    bg_labels, n_bg = ndimage.label(1 - m)
    border = np.unique(
        np.concatenate(
            [bg_labels[0], bg_labels[-1], bg_labels[:, 0], bg_labels[:, -1]]
        )
    )
    n_holes = int(n_bg - len(set(border) - {0}))

    # contour-based measures (on filled shape, largest external contour set)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
    hull = cv2.convexHull(all_pts)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 1.0

    # deepest convexity defect across contours, relative to diag
    max_defect = 0.0
    for c in contours:
        if len(c) < 4:
            continue
        hull_idx = cv2.convexHull(c, returnPoints=False)
        if hull_idx is not None and len(hull_idx) > 3:
            try:
                defects = cv2.convexityDefects(c, hull_idx)
            except cv2.error:
                continue
            if defects is not None:
                max_defect = max(max_defect, float(defects[:, 0, 3].max()) / 256.0)

    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    compactness = perimeter**2 / (4 * np.pi * area)

    # min-area rect aspect ratio
    (_, (rw, rh), _) = cv2.minAreaRect(all_pts)
    aspect_ratio = float(min(rw, rh) / max(rw, rh)) if max(rw, rh) > 0 else 1.0

    # stroke width via distance transform sampled on the skeleton
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    skel = skeletonize(m.astype(bool))
    skel_d = dist[skel]
    if skel_d.size:
        min_sw = 2.0 * float(np.percentile(skel_d, 5))
        med_sw = 2.0 * float(np.median(skel_d))
    else:  # degenerate (e.g. convex blob skeletonized to nothing)
        min_sw = med_sw = 2.0 * float(dist.max())

    inscribed_r = float(dist.max())
    closed_region = bool(med_sw > 0.5 * inscribed_r)

    return {
        "n_components": n_components,
        "n_holes": n_holes,
        "solidity": round(solidity, 4),
        "convexity_defect_depth_rel": round(max_defect / diag, 4),
        "min_stroke_width_rel": round(min_sw / diag, 4),
        "median_stroke_width_rel": round(med_sw / diag, 4),
        "compactness": round(compactness, 3),
        "aspect_ratio": round(aspect_ratio, 4),
        "area_frac": round(area / (h * w), 4),
        "closed_region": closed_region,
    }
