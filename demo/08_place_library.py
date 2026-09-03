#!/usr/bin/env python3
"""08_place_library.py — a library shadow takes an element's place in the shot.

    python demo/08_place_library.py --sequence pixar_scene_02_F \\
        --library-id letters_upper_F_dejavusans-bold

The assignment board's second answer: instead of solving this element, cast it
with a shadow the benchmark already solved. The library render supplies the
SHAPE; the authored sequence supplies the PLACEMENT — per frame, the library
silhouette is scaled (aspect kept) into the authored element's bounding box,
so a moving element still moves and a static one holds. Output is a normal
reassembled directory (frames + reassembly.json), which 10_compose_video
consumes exactly like a solved one; if a solved reassembly exists for the same
sequence it is REPLACED — the assignment is the decision.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent


def load_shadow(library_id: str, sweep: str) -> np.ndarray:
    rec = None
    for line in (BENCH / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["id"] == library_id:
                rec = r
                break
    if rec is None:
        sys.exit(f"[!] {library_id} is not in the library (metadata.jsonl)")
    stem = Path(rec["target"]).stem
    p = BENCH / "optimized" / sweep / rec["subset"] / stem / f"{stem}_best.png"
    if not p.exists():
        sys.exit(f"[!] {library_id} has no solve in {sweep}")
    g = np.array(Image.open(p).convert("L"))
    m = g > 128
    # the render's polarity is whichever side is the minority: a shadow does
    # not fill most of the frame
    if m.mean() > 0.5:
        m = ~m
    ys, xs = np.where(m)
    if not len(ys):
        sys.exit(f"[!] {library_id}: empty shadow render")
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--library-id", required=True)
    ap.add_argument("--sweep", default="big-budget-grounded")
    ap.add_argument("--out", default=None,
                    help="default: demo/projects/<project>/out/reassembled/")
    a = ap.parse_args()

    seq = BENCH / "sequences" / a.sequence
    src = json.loads((seq / "source.json").read_text(encoding="utf-8"))
    crop, canvas = src["crop"], src["canvas"]
    side = crop["pad_side"]
    ox, oy = (side - crop["w"]) // 2, (side - crop["h"]) // 2
    shadow = load_shadow(a.library_id, a.sweep)
    sh, sw = shadow.shape

    proj = a.sequence.split("_scene_")[0]
    out = (Path(a.out) if a.out else
           ROOT / "projects" / proj / "out" / "reassembled") / a.sequence
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    frames = sorted(seq.glob("f*.png"))
    # Robust pinning: SIZE comes from the median authored bbox and POSITION
    # from the per-frame centroid unless it is an outlier. A frame where the
    # element is squashed or occluded (the lamp landing on the R leaves a
    # displaced sliver) would otherwise rescale the whole glyph into the
    # sliver -- one frame of the letter jumping and shrinking.
    k = side / 512.0
    stats = []
    for fp in frames:
        au = np.array(Image.open(fp).convert("L")) < 128
        ys, xs = np.where(au)
        if len(ys):
            stats.append(((xs.max() - xs.min() + 1), (ys.max() - ys.min() + 1),
                          xs.mean(), ys.mean()))
        else:
            stats.append(None)
        k = side / au.shape[0]
    ok = [t for t in stats if t]
    if not ok:
        sys.exit(f"[!] {a.sequence}: no authored content in any frame")
    mbw = float(np.median([t[0] for t in ok])) * k
    mbh = float(np.median([t[1] for t in ok])) * k
    mcx = float(np.median([t[2] for t in ok])) * k
    mcy = float(np.median([t[3] for t in ok])) * k
    s_fit = min(mbw / sw, mbh / sh)
    tw, th = max(1, int(sw * s_fit)), max(1, int(sh * s_fit))
    tile = Image.fromarray(np.where(shadow, 0, 255).astype(np.uint8))
    tile = tile.resize((tw, th), Image.NEAREST)
    tol = 0.5 * max(mbw, mbh)
    for i, fp in enumerate(frames):
        page = Image.new("L", (canvas["w"], canvas["h"]), 255)
        t = stats[i]
        if t is not None:
            cx, cy = t[2] * k, t[3] * k
            if abs(cx - mcx) > tol or abs(cy - mcy) > tol:
                cx, cy = mcx, mcy               # outlier frame: hold the pin
        else:
            cx, cy = mcx, mcy
        px = crop["x"] + int(cx - tw / 2) - ox
        py = crop["y"] + int(cy - th / 2) - oy
        page.paste(tile, (px, py))
        page.save(out / f"f{i:02d}.png")

    (out / "reassembly.json").write_text(json.dumps({
        "sequence": a.sequence, "n_frames": len(frames),
        "fps": src.get("fps") or 5.0, "canvas": canvas, "crop": crop,
        "source": "library", "library_id": a.library_id, "sweep": a.sweep,
        "source_frame_ids": src.get("frame_ids"),
    }, indent=1), encoding="utf-8")
    print(f"{a.sequence}: {len(frames)} frames cast from {a.library_id} -> {out}")


if __name__ == "__main__":
    main()
