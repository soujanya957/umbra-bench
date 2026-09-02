#!/usr/bin/env python3
"""08_reassemble.py — solved shadows back onto the source canvas.

    python 08_reassemble.py --sequence demo_01_scene_06_A
    python 08_reassemble.py --all --fps 5

Stage 5 recorded, per sequence, the crop that ties its 512x512 frames to a
position on the 1920x1080 source. Pasting a solved shadow back through that
crop is *almost* the whole job, and the part it misses is the one that shows.

`--fit-target` applies one similarity transform per clip before solving -- it
has to, because the glyph spans rows the rig cannot reach -- so the solved
shadow lives in fitted coordinates, scaled 0.64-1.02 and shifted 8-41 px in a
128 px frame. Sent through the crop as-is it lands visibly displaced and
resized against the footage it came from. Measured against the authored frames
the wide pass scores 0.167 where it scores 0.681 against its own fitted
targets, and that gap is entirely placement.

The fit is a similarity transform, so it is exactly invertible. Inverting it
per frame before the crop separates the two concerns the pipeline had
conflated: the rig casts wherever it can reach, and the letter still lands
where the ad put it.

The transform, from target_fit.warp_target: a point at authored position i
goes to o = c + d + scale*(i - c), about the canvas centre c. So the inverse
is scale -> 1/scale, rot -> -rot, d -> -(1/scale) * R(-rot) @ d. --check
round-trips it against the authored frames rather than trusting that.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import affine_transform

BENCH = Path(__file__).resolve().parent.parent


def warp(a: np.ndarray, scale=1.0, dy=0.0, dx=0.0, rot=0.0, binarize=True):
    """The same similarity transform run_sequence.py's fit applies."""
    t = np.asarray(a, dtype=np.float32)
    H, W = t.shape
    c = np.array([(H - 1) / 2.0, (W - 1) / 2.0])
    d = np.array([dy, dx], dtype=float)
    ct, st = np.cos(-rot), np.sin(-rot)
    M = np.array([[ct, -st], [st, ct]], dtype=float) / max(scale, 1e-6)
    out = affine_transform(t, M, offset=c - M @ (c + d), order=1,
                           mode="constant", cval=0.0)
    return (out > 0.5).astype(np.float32) if binarize else out.astype(np.float32)


def invert(fit: dict) -> dict:
    """Parameters that undo `fit`."""
    s, rot = float(fit["scale"]), float(fit.get("rot") or 0.0)
    d = np.array([float(fit["dy"]), float(fit["dx"])])
    ct, st = np.cos(-rot), np.sin(-rot)
    dn = -(np.array([[ct, -st], [st, ct]]) @ d) / s
    return {"scale": 1.0 / s, "dy": float(dn[0]), "dx": float(dn[1]), "rot": -rot}


def latest_summary(run_dir: Path):
    j = sorted(run_dir.glob("summary_*.json"))
    return json.loads(j[-1].read_text(encoding="utf-8")) if j else None


def shadow_frames(run_dir: Path):
    return sorted(run_dir.glob("frame_*/best_shadow.png"))


def reassemble(name: str, out_root: Path, fps: float | None, check: bool):
    seq = BENCH / "sequences" / name
    run = BENCH / "optimized" / name
    src = json.loads((seq / "source.json").read_text(encoding="utf-8"))
    summ = latest_summary(run)
    if summ is None:
        print(f"  {name}: not solved yet, skipped")
        return None
    shots = shadow_frames(run)
    if not shots:
        print(f"  {name}: no best_shadow.png, skipped")
        return None

    fit = summ.get("target_fit")
    inv = invert(fit) if fit else None
    crop, canvas = src["crop"], src["canvas"]
    side = crop["pad_side"]
    ox, oy = (side - crop["w"]) // 2, (side - crop["h"]) // 2

    if check and fit:
        # round-trip an authored frame through fit then its inverse
        a = np.asarray(Image.open(seq / "f00.png").convert("L").resize(
            (128, 128), Image.NEAREST)) < 128
        back = warp(warp(a.astype(np.float32), **{k: fit[k] for k in
                                                  ("scale", "dy", "dx")},
                         rot=fit.get("rot") or 0.0), **inv) > 0.5
        inter, union = (a & back).sum(), (a | back).sum()
        print(f"    round-trip IoU {inter / max(union, 1):.4f}  "
              f"(1.0 = exact; resampling costs a little)")

    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for i, p in enumerate(shots):
        m = np.asarray(Image.open(p).convert("L")) > 128     # white = shadow
        if inv:
            m = warp(m.astype(np.float32), **inv) > 0.5
        tile = Image.fromarray(np.where(m, 0, 255).astype(np.uint8), "L")
        tile = tile.resize((side, side), Image.NEAREST)
        region = tile.crop((ox, oy, ox + crop["w"], oy + crop["h"]))
        page = Image.new("L", (canvas["w"], canvas["h"]), 255)
        page.paste(region, (crop["x"], crop["y"]))
        f = out / f"f{i:02d}.png"
        page.save(f)
        written.append(f)

    f = fps or src.get("fps") or 5.0
    (out / "reassembly.json").write_text(json.dumps({
        "sequence": name, "n_frames": len(written), "fps": f,
        "canvas": canvas, "crop": crop,
        "fit_applied_by_solver": fit,
        "inverse_applied_here": inv,
        "source_frame_ids": src.get("frame_ids"),
    }, indent=1), encoding="utf-8")
    print(f"  {name:<26}{len(written):>3} frames -> {out}"
          + ("" if fit else "   (no fit recorded; pasted as solved)"))
    return written, f


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", action="append", default=[])
    ap.add_argument("--all", action="store_true", help="every solved demo_* clip")
    ap.add_argument("--out", default=str(BENCH / "results" / "demo_reassembled"))
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--no-check", action="store_true",
                    help="skip the fit/inverse round-trip check")
    a = ap.parse_args()

    names = list(a.sequence)
    if a.all:
        names += [p.name for p in sorted((BENCH / "sequences").glob("demo_*"))
                  if (BENCH / "optimized" / p.name).is_dir()]
    names = sorted(set(names))
    if not names:
        raise SystemExit("nothing to do: pass --sequence NAME or --all")

    out_root = Path(a.out)
    print(f"{len(names)} sequence(s) -> {out_root}\n")
    done = 0
    for n in names:
        if reassemble(n, out_root, a.fps, not a.no_check):
            done += 1
    print(f"\n{done}/{len(names)} reassembled")
    print("frames are 1-bit on the source canvas, dark = shadow.")
    print("to encode one:  ffmpeg -framerate <fps> -i f%02d.png -pix_fmt yuv420p out.mp4")


if __name__ == "__main__":
    main()
