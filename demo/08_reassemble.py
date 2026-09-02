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
    """The newest summary, and the timestamp that selects its own frames."""
    j = sorted(run_dir.glob("summary_*.json"))
    if not j:
        return None, None
    return json.loads(j[-1].read_text(encoding="utf-8")), j[-1].stem[len("summary_"):]


def shadow_frames(run_dir: Path, ts: str | None):
    """The frames of ONE run.

    A clip solved more than once keeps every attempt as frame_NN_<timestamp>/,
    so globbing frame_* interleaves them: a 14-frame clip re-solved twice
    silently reassembles into 47 frames, in an order that is neither run.
    """
    return sorted(run_dir.glob(f"frame_*_{ts}/best_shadow.png" if ts
                               else "frame_*/best_shadow.png"))


def reassemble(name: str, out_root: Path, fps: float | None, check: bool,
               hold: bool = False):
    seq = BENCH / "sequences" / name
    run = BENCH / "optimized" / name
    src = json.loads((seq / "source.json").read_text(encoding="utf-8"))
    summ, ts = latest_summary(run)
    if summ is None:
        print(f"  {name}: not solved yet, skipped")
        return None
    shots = shadow_frames(run, ts)
    if not shots:
        print(f"  {name}: no best_shadow.png, skipped")
        return None
    # The static lane (route_motion.py): a clip whose element does not move
    # is solved ONCE and held -- one best_shadow replicated across the clip's
    # frame ids, so the composite still gets a frame per source frame.
    held = False
    if hold and len(shots) == 1 and len(src.get("frame_ids") or []) > 1:
        shots = shots * len(src["frame_ids"])
        held = True

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
    if (not held) and summ.get("n_frames") and len(shots) != summ["n_frames"]:
        print(f"  {name}: {len(shots)} frames on disk but the summary says "
              f"{summ['n_frames']} — solve still in flight, skipped")
        return None
    # A stabilised clip was solved with its travel removed; the removed
    # per-frame shifts ride in source.json as trajectory_px (authored-canvas
    # pixels), and re-applying them here is what puts the motion back into
    # the composite. Scaled by side/size because the tile lives in the
    # crop's pad frame, not the authored square.
    traj = src.get("trajectory_px")
    if traj and len(traj) != len(shots):
        print(f"  {name}: {len(traj)} trajectory entries for {len(shots)} "
              f"frames — refusing to guess, skipped")
        return None
    k = side / float(src.get("size") or 512)

    written = []
    for i, p in enumerate(shots):
        m = np.asarray(Image.open(p).convert("L")) > 128     # white = shadow
        if inv:
            m = warp(m.astype(np.float32), **inv) > 0.5
        tile = Image.fromarray(np.where(m, 0, 255).astype(np.uint8), "L")
        tile = tile.resize((side, side), Image.NEAREST)
        if traj:
            # trajectory_px is the shift APPLIED to each original frame to
            # centre it (original -> stabilised); undoing it means
            # subtracting, and getting this sign wrong doubles the travel
            # instead of restoring it -- caught by measuring, frame IoU vs
            # the moving authored letter fell to 0.
            dy, dx = traj[i]
            moved = Image.new("L", (side, side), 255)
            moved.paste(tile, (round(-dx * k), round(-dy * k)))
            tile = moved
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
        "trajectory_applied": bool(traj),
        "held_static": held,
        "stabilized_from": src.get("stabilized_from"),
        "source_frame_ids": src.get("frame_ids"),
    }, indent=1), encoding="utf-8")
    print(f"  {name:<26}{len(written):>3} frames -> {out}"
          + ("" if fit else "   (no fit recorded; pasted as solved)"))
    return written, f


def authored_on_canvas(seq: Path, i: int, crop, canvas) -> np.ndarray:
    """The authored frame put through the same crop, for comparison."""
    side = crop["pad_side"]
    ox, oy = (side - crop["w"]) // 2, (side - crop["h"]) // 2
    t = Image.open(seq / f"f{i:02d}.png").convert("L").resize(
        (side, side), Image.NEAREST)
    page = Image.new("L", (canvas["w"], canvas["h"]), 255)
    page.paste(t.crop((ox, oy, ox + crop["w"], oy + crop["h"])),
               (crop["x"], crop["y"]))
    return np.asarray(page) < 128


def sheet(names, out_root: Path, cols: int = 10, tile_w: int = 240,
          full_canvas: bool = False):
    """One row per sequence: authored glyph in grey, cast shadow in black.

    Cropped to one window per sequence -- the union of both shapes over all its
    frames, plus a margin -- not to each frame's own box. A per-frame crop
    recentres every frame and plays back as a shape twitching in place, the
    same trap stage 5 avoids. One window keeps the motion and keeps the grey
    and black in a fixed relation, so a systematic offset still reads as one.

    On the full 1920x1080 canvas a scene_06 letter is about fifty pixels wide
    and the tile is mostly white, which shows placement and nothing else.
    --full-canvas restores that view.
    """
    from PIL import ImageDraw
    rows = []
    for n in names:
        seq, run = BENCH / "sequences" / n, out_root / n
        fs = sorted(run.glob("f*.png"))
        if not fs:
            continue
        src = json.loads((seq / "source.json").read_text(encoding="utf-8"))
        # A stabilised reassembly re-applies the travel, so the honest grey
        # reference is the PARENT's authored (moving) frames.
        aseq = BENCH / "sequences" / src.get("stabilized_from", n)
        idx = (list(range(len(fs))) if len(fs) <= cols
               else [round(i * (len(fs) - 1) / (cols - 1)) for i in range(cols)])
        pairs = []
        for i in idx:
            sh = np.asarray(Image.open(fs[i]).convert("L")) < 128
            pairs.append((sh, authored_on_canvas(aseq, i, src["crop"], src["canvas"])))

        box = None
        if not full_canvas:
            for sh, au in pairs:
                ys, xs = np.where(sh | au)
                if not len(ys):
                    continue
                b = (xs.min(), ys.min(), xs.max(), ys.max())
                box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]),
                                             max(box[2], b[2]), max(box[3], b[3]))
        if box is not None:
            H, W = pairs[0][0].shape
            m = max(12, int(0.10 * max(box[2] - box[0], box[3] - box[1])))
            box = (max(0, box[0] - m), max(0, box[1] - m),
                   min(W, box[2] + m + 1), min(H, box[3] + m + 1))

        tiles = []
        for sh, au in pairs:
            rgb = np.full(sh.shape + (3,), 255, np.uint8)
            rgb[au] = (205, 205, 205)          # where the letter was authored
            rgb[sh] = (20, 20, 20)             # where the rig casts it
            im = Image.fromarray(rgb)
            if box is not None:
                im = im.crop(box)
            th = max(1, round(tile_w * im.height / im.width))
            tiles.append(im.resize((tile_w, th), Image.BILINEAR))
        rows.append((n, tiles, len(fs)))
    if not rows:
        return None
    # rows now differ in height, so lay them out cumulatively
    tw = rows[0][1][0].width
    heights = [max(t.height for t in tiles) for _, tiles, _ in rows]
    W = 10 + max(len(t) for _, t, _ in rows) * (tw + 6)
    H = 10 + sum(h + 24 for h in heights)
    out = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(out)
    y = 10
    for r, (n, tiles, total) in enumerate(rows):
        th = heights[r]
        for c, t in enumerate(tiles):
            out.paste(t, (10 + c * (tw + 6), y))
        dr.text((10, y + th + 6), f"{n}   {total} frames"
                + ("" if total <= len(tiles) else f", {len(tiles)} shown"),
                fill=(0, 0, 0))
        y += th + 24
    f = out_root / "_all.png"
    out.save(f)
    return f


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", action="append", default=[])
    ap.add_argument("--all", action="store_true", help="every solved demo_* clip")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out" / "reassembled"))
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--no-sheet", action="store_true",
                    help="skip the review sheet")
    ap.add_argument("--full-canvas", action="store_true",
                    help="review sheet on the whole 1920x1080 frame")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the fit/inverse round-trip check")
    ap.add_argument("--hold", action="store_true",
                    help="static lane: replicate a single solved frame across "
                         "the clip's frame ids (see route_motion.py)")
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
        if reassemble(n, out_root, a.fps, not a.no_check, hold=a.hold):
            done += 1
    print(f"\n{done}/{len(names)} reassembled")
    if done and not a.no_sheet:
        f = sheet([n for n in names if (out_root / n).is_dir()], out_root,
                  full_canvas=a.full_canvas)
        if f:
            print(f"review: {f}   (grey = authored, black = cast)")
    print("frames are 1-bit on the source canvas, dark = shadow.")
    print("to encode one:  ffmpeg -framerate <fps> -i f%02d.png -pix_fmt yuv420p out.mp4")


if __name__ == "__main__":
    main()
