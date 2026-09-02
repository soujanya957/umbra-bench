#!/usr/bin/env python3
"""Turn a segmented shot into solvable targets without flattening its motion.

The other subsets are independent shapes, so `ground_targets.py` places each one
on the bottom edge and that is right. A shot is not independent: the glyph moves
39-329 px horizontally and 39-233 px vertically within a scene, and *that motion
is the content*. Normalising each frame on its own bounding box would place every
frame identically and play back as a shape twitching in place rather than
travelling across the wall.

So the crop is computed once per scene, over the union of every frame in it, and
applied unchanged to all of them. Relative position and relative size survive;
what the robot casts moves the way the source moves.

    python scripts/make_demo_targets.py \\
        --masks fleet-shadow-demo/letters_sam/by_frame \\
        --out targets_demo/demo

Output is a flat subset directory named `<scene>_<frame>.png`, which sorts into
playback order. It is deliberately NOT written into `targets/`: these are frames
of one video, not benchmark items, and mixing them in would change every count in
METRICS.md and every per-subset table in the atlas.

Solve it with the fit pinned, or the per-target search puts the motion back where
it started:

    --fit-n-scales 1 --fit-n-shifts 1 --fit-scale-min S --fit-scale-max S
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

FRAME_RE = re.compile(r"(f\d+)")


def ink(path: Path) -> np.ndarray:
    """True where the glyph is. Masks are glyph-dark on white, like the targets."""
    return np.asarray(Image.open(path).convert("L")) < 128


def bbox(m: np.ndarray):
    ys, xs = np.where(m)
    return None if not len(ys) else (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def scene_of(frame_id: str, keypoints: dict) -> str:
    rec = keypoints.get(frame_id)
    return (rec or {}).get("scene") or "scene_00"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", default="fleet-shadow-demo/letters_sam/by_frame",
                    help="directory of <frame>/<frame>_<obj>_mask.png")
    ap.add_argument("--keypoints", default="fleet-shadow-demo/keypoints.json",
                    help="used only for the frame -> scene mapping")
    ap.add_argument("--out", default="targets_demo/demo")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--margin", type=float, default=0.08,
                    help="fraction of the union box added on every side, so the "
                         "shape never touches the frame edge in any frame")
    ap.add_argument("--per-scene", action="store_true", default=True,
                    help="one crop per scene (default). --whole-shot for one "
                         "crop across every scene")
    ap.add_argument("--whole-shot", dest="per_scene", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    masks = sorted(Path(a.masks).glob("*/*_mask.png"))
    if not masks:
        raise SystemExit(f"no masks under {a.masks}")
    kp = {}
    kpf = Path(a.keypoints)
    if kpf.exists():
        kp = json.loads(kpf.read_text(encoding="utf-8")).get("frames", {})

    frames = []
    for p in masks:
        m = FRAME_RE.search(p.name)
        fid = m.group(1) if m else p.stem
        frames.append({"path": p, "frame": fid, "scene": scene_of(fid, kp)})

    groups = defaultdict(list)
    for f in frames:
        groups[f["scene"] if a.per_scene else "_all"].append(f)

    out_dir = Path(a.out)
    rows = []
    print(f"{len(frames)} masks in {len(groups)} group(s)\n")
    for gname, items in sorted(groups.items()):
        items.sort(key=lambda x: x["frame"])
        # the union over the whole group -- one box, so every frame keeps its
        # place inside it
        ub = None
        for it in items:
            b = bbox(ink(it["path"]))
            it["bbox"] = b
            if b is None:
                continue
            ub = b if ub is None else (min(ub[0], b[0]), min(ub[1], b[1]),
                                       max(ub[2], b[2]), max(ub[3], b[3]))
        if ub is None:
            print(f"  {gname}: no ink in any frame, skipped")
            continue

        x0, y0, x1, y1 = ub
        w, h = x1 - x0 + 1, y1 - y0 + 1
        # Crop the union rectangle, not a square around it. Squaring here fails on
        # a wide group: scene_06's FAMILY banner is 1638x272, and a square about
        # its centre is 1900x1900 starting at y=-564 -- off the top of a 1080-line
        # frame, with the letters reduced to a stripe in an empty field. The
        # squaring happens at paste time instead, after the crop, so the content
        # stays as large as the frame allows and the aspect is still not stretched.
        H, W = ink(items[0]["path"]).shape
        mx, my = int(round(w * a.margin)), int(round(h * a.margin))
        sx, sy = max(0, x0 - mx), max(0, y0 - my)
        ex, ey = min(W, x1 + 1 + mx), min(H, y1 + 1 + my)
        cw, ch = ex - sx, ey - sy
        side = max(cw, ch)
        print(f"  {gname}: {len(items):>3} frames  union {w}x{h}"
              f"  -> crop {cw}x{ch} at ({sx},{sy})  padded to {side}x{side}")

        if a.dry_run:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for it in items:
            full = Image.open(it["path"]).convert("L")
            crop = full.crop((sx, sy, ex, ey))
            canvas = Image.new("L", (side, side), 255)      # white = background
            # centred inside the square, identically for every frame of the group,
            # so the padding cannot introduce motion of its own
            canvas.paste(crop, ((side - cw) // 2, (side - ch) // 2))
            img = canvas.resize((a.size, a.size), Image.NEAREST)
            img = img.point(lambda v: 255 if v > 127 else 0)
            name = f"{gname}_{it['frame']}.png" if a.per_scene else f"{it['frame']}.png"
            img.save(out_dir / name)
            b = it["bbox"]
            rows.append({"target": name, "scene": gname, "frame": it["frame"],
                         "crop_x": sx, "crop_y": sy, "crop_side": side,
                         "src_x0": b[0] if b else "", "src_y0": b[1] if b else "",
                         "src_x1": b[2] if b else "", "src_y1": b[3] if b else ""})

    if a.dry_run:
        print("\ndry run — nothing written")
        return
    # the crop is what puts a solved shadow back on the 1920x1080 canvas, so it
    # has to be recorded or the video cannot be reassembled
    man = out_dir.parent / "demo_manifest.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\n{len(rows)} targets -> {out_dir}\n  crop geometry -> {man}")


if __name__ == "__main__":
    main()
