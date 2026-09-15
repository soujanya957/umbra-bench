#!/usr/bin/env python3
"""Re-anchor every target to the bottom of the frame instead of its centre.

The dataset authors targets centred with a 10% margin. The rig does not cast
there. Measured over `optimized/big-budget-fitted` (478 targets x 10 seeds), the
vertical centre of mass of the shadows the rig actually produces sits at row 90.5
of 128 while the authored targets sit at 63.4 -- 27 rows, 21% of the frame, above
where the arms can reach. `--fit-target` has been absorbing that gap on its own:
478 of 478 fits moved the target down, and 60% of them stopped only because they
hit `--fit-max-shift`. Half the search grid was being spent on upward shifts the
search never takes, and the half it wanted ran out of room.

This moves the constant part of that offset into the data, so the fit is left
solving the per-target remainder rather than re-deriving one rig constant 546
times. It also lets the target absorb the arm-column and base shadow that is dark
in >90% of every solve regardless of target -- a centred target leaves all of that
as spill, which is most of what makes a shadow unreadable.

**Pure vertical translation. Nothing is scaled, cropped or reflowed.** That is the
point: a grounded target and its authored original differ by a translation and
nothing else, so every shape attribute in `metadata.jsonl` -- area fraction, stroke
widths, topology, limb counts -- still describes it, and any difference in the
results is attributable to placement alone. Scaling here would confound exactly the
comparison this variant exists to make.

Writes a parallel tree rather than editing `targets/` in place, so both conditions
stay runnable from one checkout:

    targets/           authored, centred        (canonical, untouched)
    targets_grounded/  same shapes, bottom-anchored

    python scripts/ground_targets.py
    python scripts/ground_targets.py --subsets digits letters_upper --dry-run
"""

import argparse
import os

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)


def load_mask(path: str) -> np.ndarray:
    """Benchmark PNGs are 1-bit, dark = shape. Returns a bool mask, True = shape."""
    return np.array(Image.open(path).convert("L")) < 128


def save_mask(mask: np.ndarray, path: str) -> None:
    """Write a bool mask back as 1-bit dark-on-white, matching targets/."""
    a = ((~mask) * 255).astype(np.uint8)
    Image.fromarray(a, "L").convert("1").save(path)


def ground(mask: np.ndarray, bottom_margin_frac: float = 0.0,
           recenter_x: bool = False) -> tuple[np.ndarray, int, int]:
    """Translate `mask` so its lowest shape pixel sits at the bottom of the frame.

    Returns (grounded, dy, dx) with dy/dx in pixels, + = down / right.

    The anchor is the shape's true bounding box, not its centroid: "standing on
    the ground" is a statement about the lowest point, so a target with one thin
    protrusion below the body -- a leg, a tail, a stem -- rests on that protrusion
    and the body stays where it is. Anchoring the centroid instead would push such
    a shape off the bottom of the frame.
    """
    H, W = mask.shape
    ys, xs = np.where(mask)
    if len(ys) == 0:                     # empty mask: nothing to anchor
        return mask.copy(), 0, 0
    dy = int(round((H - 1 - int(ys.max())) - bottom_margin_frac * H))
    dx = int(round((W - 1) / 2.0 - (int(xs.min()) + int(xs.max())) / 2.0)) if recenter_x else 0

    out = np.zeros_like(mask)
    # Explicit index arithmetic rather than np.roll: a roll wraps, and a target
    # that wrapped would look like a plausible mask while being nonsense.
    ys2, xs2 = ys + dy, xs + dx
    keep = (ys2 >= 0) & (ys2 < H) & (xs2 >= 0) & (xs2 < W)
    out[ys2[keep], xs2[keep]] = True
    return out, dy, dx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default=_BENCH)
    p.add_argument("--src", default="targets", help="source tree, repo-relative")
    p.add_argument("--out", default="targets_grounded", help="destination tree")
    p.add_argument("--subsets", nargs="+", default=None,
                   help="default: every subset present in --src")
    p.add_argument("--bottom-margin", type=float, default=0.0,
                   help="gap left below the shape, as a fraction of frame height. "
                        "0.0 = resting on the bottom edge")
    p.add_argument("--recenter-x", action="store_true",
                   help="also centre horizontally. Off by default: the authored "
                        "horizontal composition is not what this variant is testing")
    p.add_argument("--force", action="store_true", help="overwrite existing outputs")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    src_root = os.path.join(a.bench, a.src)
    out_root = os.path.join(a.bench, a.out)
    subsets = a.subsets or sorted(
        d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))
    )

    shifts, n_written, n_skipped, clipped = [], 0, 0, []
    for sub in subsets:
        sd = os.path.join(src_root, sub)
        od = os.path.join(out_root, sub)
        if not os.path.isdir(sd):
            print(f"[!] missing subset {sd}")
            continue
        if not a.dry_run:
            os.makedirs(od, exist_ok=True)
        for fn in sorted(f for f in os.listdir(sd) if f.lower().endswith(".png")):
            dst = os.path.join(od, fn)
            if os.path.exists(dst) and not a.force:
                n_skipped += 1
                continue
            m = load_mask(os.path.join(sd, fn))
            g, dy, dx = ground(m, a.bottom_margin, a.recenter_x)
            # A pure translation must not lose mass. If it did, the source shape
            # already ran past the frame and the result is not the same target.
            if g.sum() != m.sum():
                clipped.append(f"{sub}/{fn} ({m.sum() - g.sum()}px)")
            shifts.append((sub, fn, dy, m.shape[0]))
            if not a.dry_run:
                save_mask(g, dst)
            n_written += 1

    if not shifts:
        print("[!] nothing to do" + ("" if a.force else " (all outputs exist; --force to redo)"))
        return

    d = np.array([s[2] for s in shifts], dtype=float)
    H = shifts[0][3]
    print(f"\n[ground] {n_written} written, {n_skipped} skipped"
          f"{' (dry run, nothing on disk)' if a.dry_run else ''}")
    print(f"[ground] {src_root} -> {out_root}")
    print(f"[ground] downward shift, px at {H} (and as a fraction of the frame):")
    print(f"           mean   {d.mean():6.1f}  ({d.mean() / H:.3f})")
    print(f"           median {np.median(d):6.1f}  ({np.median(d) / H:.3f})")
    print(f"           min    {d.min():6.1f}  ({d.min() / H:.3f})")
    print(f"           max    {d.max():6.1f}  ({d.max() / H:.3f})")
    print(f"           p90    {np.percentile(d, 90):6.1f}")
    if clipped:
        print(f"[!] {len(clipped)} target(s) lost mass in translation: {clipped[:5]}")

    print("\n[ground] per subset (mean downward shift, px):")
    for sub in subsets:
        g = [s[2] for s in shifts if s[0] == sub]
        if g:
            print(f"           {sub:<14} n={len(g):3d}  {np.mean(g):6.1f}")


if __name__ == "__main__":
    main()
