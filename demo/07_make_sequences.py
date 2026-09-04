#!/usr/bin/env python3
"""07_make_sequences.py — one letter in one scene is one sequence.

Takes the cleaned masks out of `by_frame/<fid>/<fid>_<L>_mask.png` -- three
levels deep with a single file at the bottom, which cannot be skimmed -- and
regroups them by (scene, letter). That grouping is also the unit the optimizer
should see: `scene_04_M` is fourteen frames of one glyph moving, not fourteen
unrelated targets.

    python 07_make_sequences.py --in letters_clean_colour
    python 07_make_sequences.py --in letters_clean_colour --dry-run

Output goes into the repo's existing **sequences track**, not `targets/`.
SEQUENCES.md is explicit that an animation is "a separate track, not a tenth
subset", because dropping frames into `targets/` scores them as unrelated static
targets and silently discards the temporal information. The layout and the frame
naming are that track's:

    sequences/family_ad_scene_04_M/f00.png …   1-bit, dark = shape
    sequences/family_ad_scene_04_M/source.json  where the frames came from
    sequences.jsonl                           built by build_sequence_metadata.py

`sequences.jsonl` is not written here. `scripts/build_sequence_metadata.py`
already scans the track, computes `target_motion` and the per-frame attributes,
and detects `loop` by the wrap test -- run it after this and the new sequences
join the existing thirteen.

`source.json` is the part that track has no field for: the crop box that puts a
solved shadow back on the 1920x1080 canvas, the source frame ids behind f00, f01,
and the effective fps. Without it the video cannot be reassembled.

The indices are renumbered because the source ids are not contiguous -- the shot
is sampled at every fifth frame, so a sequence reads f0418, f0423, f0428 and a
loader that assumes i+1 would skip most of it. The original ids are not lost:
`meta.json` keeps `frame_ids` in order, alongside the crop that puts a solved
shadow back on the 1920x1080 canvas and the effective fps to re-encode at.

**The crop is per sequence, not per frame.** The glyph moves across its scene --
39 to 329 px measured on this footage -- and that motion is the content. Cropping
each frame to its own bounding box would centre every frame identically and play
back as a shape twitching in place. One box over the union of the sequence keeps
each frame's position inside it, so the shape travels the way it travels in the
source.

The box is not squared before cropping: scene_06's word is 1638x272, and a square
about its centre starts 564 px above the top of a 1080-line frame. The crop is the
union rectangle clamped to the frame, and the squaring happens by padding
afterwards, identically for every frame of the sequence so the padding cannot
introduce motion of its own.

Solve these with the fit pinned, or `--fit-target` re-places each frame on its own
and puts the motion back where it started:

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

ROOT = Path(__file__).resolve().parent


def ink(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("L")) < 128


def bbox(m):
    ys, xs = np.where(m)
    return None if not len(ys) else (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def sheet(tiles, cols, pad=6, bg=(255, 255, 255)):
    if not tiles:
        return None
    w = max(t.width for t in tiles)
    h = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + pad) + pad), bg)
    for i, t in enumerate(tiles):
        out.paste(t.convert("RGB"),
                  (pad + (i % cols) * (w + pad), pad + (i // cols) * (h + pad)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", default="letters_clean_colour")
    ap.add_argument("--keypoints", default=str(ROOT / "keypoints.json"))
    ap.add_argument("--out", default=str(ROOT.parent / "sequences"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--pad", type=int, default=2,
                    help="digits in the frame index. The track uses f00, f01.")
    ap.add_argument("--src-fps", type=float, default=25.0)
    ap.add_argument("--sample", type=int, default=5,
                    help="the --sample 01_split_scenes.py ran with; src-fps over "
                         "this is the sequence's own frame rate")
    ap.add_argument("--margin", type=float, default=0.08)
    ap.add_argument("--thumb", type=int, default=110, help="review tile size")
    ap.add_argument("--review", default=None,
                    help="where the contact sheets go. NOT inside the track: "
                         "build_sequence_metadata.py treats every directory under "
                         "sequences/ that holds 2+ PNGs as a sequence, so a "
                         "_review folder there would be indexed as one.")
    ap.add_argument("--seq-dir", default="",
                    help="subdirectory under --out; empty writes straight into "
                         "the sequences track")
    ap.add_argument("--demo-id", default="01",
                    help="which demo this is. The names carry it -- "
                         "family_ad_scene_04_M -- so a second shot can be added "
                         "later without colliding, and a sequence name still says "
                         "on its own which source it came from.")
    ap.add_argument("--prefix", default=None,
                    help="override the whole prefix instead of just the id")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="drop these (frame, object) pairs, as <frame>_<letter>. "
                         "Scoped to the object, not the frame, so f0259_I can go "
                         "while f0259_F stays. Empty by default: this used to "
                         "carry f0259_I and f0264_I, which were mislabelled at "
                         "the time and have since been re-clicked -- a hardcoded "
                         "exclusion outlives the problem it was added for.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    prefix = a.prefix if a.prefix is not None else f"demo_{a.demo_id}_"
    kp = json.loads(Path(a.keypoints).read_text(encoding="utf-8"))["frames"]
    src = Path(a.src)
    # flat first (06 writes one directory), nested for older output
    masks = sorted(src.glob("*_mask.png")) or sorted(src.glob("by_frame/*/*_mask.png"))
    if not masks:
        raise SystemExit(f"no masks under {src}/by_frame")

    # A mask's scene comes from the scenes/ tree itself -- keypoints.json
    # only holds the frames somebody clicked, and with video propagation
    # most masked frames were never clicked. Gating on kp silently dropped
    # every propagated-only frame (pixar: 43 of 54).
    scene_of = {}
    scenes_root = Path(a.keypoints).resolve().parent / "scenes"
    for d in sorted(scenes_root.glob("scene_*")):
        for f in d.glob("f*.png"):
            scene_of[f.stem] = d.name
    seqs = defaultdict(list)
    for p in masks:
        m = re.match(r"(f\d+)_(.+)_mask\.png$", p.name)
        if not m:
            continue
        fid, letter = m.group(1), m.group(2)
        sc = scene_of.get(fid) or kp.get(fid, {}).get("scene")
        if not sc or f"{fid}_{letter}" in set(a.exclude or []):
            continue
        seqs[f"{prefix}{sc}_{letter}"].append((fid, p))
    for v in seqs.values():
        v.sort()

    out = Path(a.out)
    rev = Path(a.review) if a.review else out.parent / "results" / "demo_review"
    rev.mkdir(parents=True, exist_ok=True)
    rows, review, seq_meta = [], [], []
    print(f"{len(masks)} masks -> {len(seqs)} sequences\n")
    for name, items in sorted(seqs.items()):
        # one box over the union of the sequence
        ub, H, W = None, None, None
        for fid, p in items:
            mm = ink(p)
            H, W = mm.shape
            b = bbox(mm)
            if b is None:
                continue
            ub = b if ub is None else (min(ub[0], b[0]), min(ub[1], b[1]),
                                       max(ub[2], b[2]), max(ub[3], b[3]))
        if ub is None:
            print(f"  {name}: empty, skipped")
            continue
        x0, y0, x1, y1 = ub
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        mx, my = int(round(bw * a.margin)), int(round(bh * a.margin))
        sx, sy = max(0, x0 - mx), max(0, y0 - my)
        ex, ey = min(W, x1 + 1 + mx), min(H, y1 + 1 + my)
        cw, ch = ex - sx, ey - sy
        side = max(cw, ch)
        print(f"  {name:<16} {len(items):>3} frames  union {bw}x{bh}"
              f"  crop {cw}x{ch}@({sx},{sy})  pad {side}")
        if a.dry_run:
            continue

        d = (out / a.seq_dir / name) if a.seq_dir else (out / name)
        d.mkdir(parents=True, exist_ok=True)
        tiles, frame_ids = [], []
        for idx, (fid, p) in enumerate(items):
            full = Image.open(p).convert("L")
            canvas = Image.new("L", (side, side), 255)
            canvas.paste(full.crop((sx, sy, ex, ey)), ((side - cw) // 2, (side - ch) // 2))
            img = canvas.resize((a.size, a.size), Image.NEAREST).point(
                lambda v: 255 if v > 127 else 0)          # binary, glyph black
            stem = "f" + str(idx).zfill(a.pad)
            img.save(d / f"{stem}.png")
            tiles.append(img.resize((a.thumb, a.thumb), Image.NEAREST))
            frame_ids.append(fid)
            rows.append({"sequence": name, "index": idx, "frame_id": fid,
                         "target": f"{name}/{stem}.png",
                         "crop_x": sx, "crop_y": sy, "crop_w": cw, "crop_h": ch,
                         "pad_side": side, "size": a.size})
        fps = a.src_fps / a.sample if a.sample else a.src_fps
        meta = {"id": name,
                # A cut from a film runs once. The track's loop test is
                # wrap_iou >= 0.9 * mean_step, which a slowly-moving shot passes
                # by accident -- scene_05_I steps at 0.994 and its wrap looks like
                # any other step. Declaring it here stops the metrics scoring a
                # first-to-last transition that is never performed.
                "loop": False, "n_frames": len(frame_ids), "size": a.size,
                "fps": fps, "source_fps": a.src_fps, "sample_every": a.sample,
                # scenes/ tree first: a video-propagated frame (seed not on
                # the scene's first frame -- backward propagation) has no
                # keypoint record, and kp[...] KeyError'd exactly there
                "scene": (scene_of.get(frame_ids[0])
                          or kp.get(frame_ids[0], {}).get("scene")),
                "project": prefix.rstrip("_"),
                "letter": name.split("_")[-1],
                "frame_ids": frame_ids,
                "crop": {"x": sx, "y": sy, "w": cw, "h": ch, "pad_side": side},
                "canvas": {"w": int(W), "h": int(H)}}
        (d / "source.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        seq_meta.append(meta)
        s = sheet(tiles, cols=min(12, len(tiles)))
        s.save(rev / f"{name}.png")
        review.append((name, tiles[0], len(tiles)))

    if a.dry_run:
        print("\ndry run — nothing written")
        return

    # one tile per sequence, so the whole set is a single glance
    from PIL import ImageDraw
    tw = a.thumb
    cols = 5
    rows_n = (len(review) + cols - 1) // cols
    allsheet = Image.new("RGB", (cols * (tw + 10) + 10, rows_n * (tw + 26) + 10), "white")
    dr = ImageDraw.Draw(allsheet)
    for i, (name, tile, n) in enumerate(review):
        x, y = 10 + (i % cols) * (tw + 10), 10 + (i // cols) * (tw + 26)
        allsheet.paste(tile.convert("RGB"), (x, y))
        dr.text((x, y + tw + 4), f"{name}  n={n}", fill=(0, 0, 0))
    allsheet.save(rev / "_all.png")

    with open(rev / "sequences.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} targets in {len(review)} sequences -> {out}")
    print(f"  inspect: {rev / '_all.png'}  (and one sheet per sequence)")
    print(f"  then:    python scripts/build_sequence_metadata.py   # rebuilds sequences.jsonl")


if __name__ == "__main__":
    main()
