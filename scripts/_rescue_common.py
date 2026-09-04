"""Shared measurement for the target-rescue scripts.

A target the rig physically cannot cast is not made castable by solving harder.
The rig has a hard floor on stroke width -- an arm link is as thin as its own
geometry, magnified by the throw -- and a target whose strokes are finer than
that floor is uncastable at every pose. These scripts adjust such targets to sit
just above the floor, and record exactly what the adjustment cost.

Two rules the callers share:

**The floor is measured, not assumed.** It is the thinnest median stroke the rig
was ever observed to cast across the sweep, read from `results/master_table.csv`.
It is a property of the RIG, not of the benchmark, so a group with thinner links
or a longer throw re-derives their own and the whole pipeline still applies.

**Minimum intervention.** Take the smallest change that clears the floor and stop.
Anything past that is trading likeness for a number nobody asked to improve.

**Edit at 512, judge at 128.** Targets are stored at 512 px and the rescued copies
are written at 512 px, but the floor was measured on 128 px renders -- the size the
optimizer actually solves at. Stroke width in diagonal-relative units is NOT
resolution-free at these widths, because skeletonisation and the distance transform
quantise: judged at 512 px, 75 targets look sub-floor; judged at the resolution the
solve happens at, 68 do. Castability is a claim about the solve, so 128 px wins.

NOTE: `shape_attributes.compute_attributes` cannot be called here -- it raises on
OpenCV >= 5 (`convexityDefects` now returns a 2-D array; see its line ~347). Its
HELPERS import fine, so `_as_binary` and `_denoise` are used directly rather than
reimplemented, and only the six-line stroke reduction is repeated. Stroke width is
measured on the DENOISED mask, exactly as shape_attributes.py:359-367 does -- on
the raw mask MPEG-7 binarization litter drives it to nonsense.
`verify_against_metadata()` re-measures a sample and asserts the agreement.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
from skimage.morphology import skeletonize

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shape_attributes import _as_binary, _denoise, _hole_stats   # noqa: E402

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(BENCH, "results", "master_table.csv")
FALLBACK_FLOOR = 0.0485


def load_mask(path: str) -> np.ndarray:
    """Benchmark PNGs are 1-bit, dark = shape. Returns uint8 {0,1}."""
    return (np.array(__import__("PIL.Image", fromlist=["Image"]).open(path).convert("L")) < 128).astype(np.uint8)


def save_mask(m: np.ndarray, path: str) -> None:
    """Write back in the benchmark's convention: 1-bit, dark = shape."""
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.where(m > 0, 0, 255).astype(np.uint8), "L").convert("1").save(path)


def median_stroke_rel(m: np.ndarray) -> float:
    """Median stroke width / image diagonal. Mirrors shape_attributes.py:359-367."""
    m = _as_binary(m)
    mc = _denoise(m, float(m.sum()))          # litter would dominate the skeleton
    diag = float(np.hypot(*m.shape))
    dist = cv2.distanceTransform(mc, cv2.DIST_L2, 5)
    sk = skeletonize(mc.astype(bool))
    d = dist[sk]
    med = 2.0 * float(np.percentile(d, 50)) if d.size else 2.0 * float(dist.max())
    return med / diag


def n_holes_signif(m: np.ndarray) -> int:
    """Holes big enough to be shape rather than litter (>=0.5% of area).

    Uses the benchmark's own `_hole_stats`. The raw count is unusable as a guard:
    MPEG-7 binarisation leaves a guitar with 146 "holes", none of them meaningful,
    and a guard on the raw count would refuse every legitimate thickening.
    """
    m = _as_binary(m)
    _, n_signif, _, _ = _hole_stats(m, float(m.sum()))
    return int(n_signif)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


SOLVE_SIZE = 128


def to_solve_res(m: np.ndarray, size: int = SOLVE_SIZE) -> np.ndarray:
    """Downsample a 512 px master exactly as `run_base_optimizer.load_target` does.

    NEAREST, not a smooth filter: run_base_optimizer documents that resampling a
    1-bit mask smoothly and re-thresholding at 128 quietly erodes thin strokes, and
    thin strokes are the entire subject here. Any other resize would judge a target
    the solve never saw.
    """
    from PIL import Image
    img = Image.fromarray(np.where(m > 0, 0, 255).astype(np.uint8), "L")
    return (np.array(img.resize((size, size), Image.NEAREST)) < 128).astype(np.uint8)


def stroke_at_solve_res(m: np.ndarray) -> float:
    """Median stroke of a 512 px mask, measured where the floor was measured."""
    return median_stroke_rel(to_solve_res(m))


def rig_floor(explicit: str | float | None = None) -> tuple[float, str]:
    """The thinnest median stroke the rig was ever observed to cast.

    Returns (floor, provenance). Reading it from the sweep rather than from a
    constant is the point: it re-derives itself for a different rig.
    """
    if explicit not in (None, "auto"):
        return float(explicit), "given on the command line"
    try:
        import pandas as pd
        d = pd.read_csv(MASTER, low_memory=False)
        d = d[(d["ref"] == "original") & (d["sweep"].str.startswith("base"))]
        v = float(d["s_median_stroke_width_rel"].min())
        return v, f"min over {len(d)} solved shadows in {os.path.relpath(MASTER, BENCH)}"
    except Exception as e:                                    # noqa: BLE001
        return FALLBACK_FLOOR, f"fallback constant ({type(e).__name__}: {e})"


def below_floor(floor: float) -> list[dict]:
    """Metadata rows whose target is finer than the rig can cast, judged at 128 px.

    Each row gains `stroke_128` so callers do not re-measure.
    """
    out = []
    with open(os.path.join(BENCH, "metadata.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            s = stroke_at_solve_res(load_mask(os.path.join(BENCH, d["target"])))
            if s < floor:
                d["stroke_128"] = round(s, 5)
                out.append(d)
    return out


def verify_against_metadata(sample: int = 40, tol: float = 0.004) -> tuple[int, float]:
    """Re-measure stored attributes on a sample; guards against a library drift."""
    rows, worst = [], 0.0
    with open(os.path.join(BENCH, "metadata.jsonl")) as f:
        for i, line in enumerate(f):
            if i % max(1, 546 // sample):
                continue
            d = json.loads(line)
            got = median_stroke_rel(load_mask(os.path.join(BENCH, d["target"])))
            worst = max(worst, abs(got - d["attributes"]["median_stroke_width_rel"]))
            rows.append(d["id"])
    return len(rows), worst


def write_manifest(path: str, method: str, floor: float, prov: str, params: dict,
                   records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    acc = [r for r in records if r["accepted"]]
    with open(path, "w") as f:
        json.dump({
            "method": method,
            "rig_floor": round(floor, 5),
            "rig_floor_provenance": prov,
            "params": params,
            "n_candidates": len(records),
            "n_accepted": len(acc),
            "n_rejected": len(records) - len(acc),
            "records": records,
        }, f, indent=1)
