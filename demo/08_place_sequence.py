#!/usr/bin/env python3
"""08_place_sequence.py — one sequence's cast shadow stands in for another.

    python demo/08_place_sequence.py --sequence pixar_scene_03_I \\
        --donor family_ad_scene_01_I

The assignment board's third answer: replace an element's WHOLE clip with a
sequence you chose. Shape comes from the donor's solve (its best_shadow
renders, content-cropped — position and size independent, as the component
pipeline is); placement and timing come from the acceptor: each acceptor
frame maps to a donor frame by proportional index (timing re-aligned), and
the donor shadow is scaled into the acceptor's authored per-frame bbox, so
the element still enters, moves and holds exactly where the footage put it.

Output is a normal reassembled directory that 10_compose_video consumes;
an existing reassembly for the acceptor is replaced — the assignment is
the decision.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent


def donor_shadows(donor: str) -> list[np.ndarray]:
    """Content-cropped binary shadows from the donor's newest solve."""
    run = BENCH / "optimized" / donor
    js = sorted(run.glob("summary_*.json"))
    if not js:
        sys.exit(f"[!] donor {donor} has no solve in optimized/")
    ts = js[-1].stem[len("summary_"):]
    out = []
    for p in sorted(run.glob(f"frame_*_{ts}/best_shadow.png")):
        m = np.array(Image.open(p).convert("L")) > 128       # white = shadow
        ys, xs = np.where(m)
        if not len(ys):
            out.append(None)
            continue
        out.append(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    if not any(x is not None for x in out):
        sys.exit(f"[!] donor {donor}: every solved frame casts nothing")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", required=True, help="the element to replace")
    ap.add_argument("--donor", required=True, help="the sequence that stands in")
    ap.add_argument("--out", default=str(ROOT / "out" / "reassembled"))
    a = ap.parse_args()

    seq = BENCH / "sequences" / a.sequence
    src = json.loads((seq / "source.json").read_text(encoding="utf-8"))
    crop, canvas = src["crop"], src["canvas"]
    side = crop["pad_side"]
    ox, oy = (side - crop["w"]) // 2, (side - crop["h"]) // 2
    shadows = donor_shadows(a.donor)
    nd = len(shadows)

    frames = sorted(seq.glob("f*.png"))
    na = len(frames)
    out = Path(a.out) / a.sequence
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for i, fp in enumerate(frames):
        # timing re-aligned: acceptor frame i plays donor frame j
        j = round(i * (nd - 1) / max(na - 1, 1)) if na > 1 else 0
        sh = shadows[j]
        if sh is None:                       # donor cast nothing that frame
            sh = next(x for x in shadows if x is not None)
        au = np.array(Image.open(fp).convert("L")) < 128
        page = Image.new("L", (canvas["w"], canvas["h"]), 255)
        ys, xs = np.where(au)
        if len(ys):
            k = side / au.shape[0]
            x0, y0 = xs.min() * k, ys.min() * k
            bw, bh = (xs.max() - xs.min() + 1) * k, (ys.max() - ys.min() + 1) * k
            s = min(bw / sh.shape[1], bh / sh.shape[0])
            tw, th = max(1, int(sh.shape[1] * s)), max(1, int(sh.shape[0] * s))
            tile = Image.fromarray(np.where(sh, 0, 255).astype(np.uint8))
            tile = tile.resize((tw, th), Image.NEAREST)
            px = crop["x"] + int(x0 + (bw - tw) / 2) - ox
            py = crop["y"] + int(y0 + (bh - th) / 2) - oy
            page.paste(tile, (px, py))
        page.save(out / f"f{i:02d}.png")

    (out / "reassembly.json").write_text(json.dumps({
        "sequence": a.sequence, "n_frames": na,
        "fps": src.get("fps") or 5.0, "canvas": canvas, "crop": crop,
        "source": "sequence", "donor": a.donor,
        "source_frame_ids": src.get("frame_ids"),
    }, indent=1), encoding="utf-8")
    print(f"{a.sequence}: {na} frames cast from sequence {a.donor} "
          f"({nd} donor poses, timing re-aligned) -> {out}")


if __name__ == "__main__":
    main()
