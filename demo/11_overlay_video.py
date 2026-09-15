#!/usr/bin/env python3
"""11_overlay_video.py — how much the performance differs, as a film.

    python demo/11_overlay_video.py --project pixar
    python demo/11_overlay_video.py --project ICRA --scene scene_02

The still-image pipeline shows a colourful overlay per frame; this is the
same idea across a whole scene: for every film frame, the AUTHORED
silhouette (what the footage asked for, every element composited on the
source canvas) against the CAST one (what the rig performs, from the
project's reassembled dirs), aligned by source frame id like
10_compose_video. Colours:

    white  — background (neither)
    green  — agreement   (authored AND cast)
    red    — missed      (authored only: the rig failed to cover it)
    blue   — extra       (cast only: shadow where the footage had none)

Output: demo/projects/<P>/out/video/<P>_<scene>_overlay.mp4 at the
authored rate. The mean per-frame IoU is printed per scene — the same
union/intersection the overlay paints, so the video and the number can
be read together.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent


def authored_on_canvas(seq_dir: Path, canvas_hw) -> dict[str, np.ndarray]:
    """{frame_id: bool mask on the full canvas} from a sequence's authored
    frames, placed through its crop (the exact inverse 07 recorded)."""
    src = json.loads((seq_dir / "source.json").read_text(encoding="utf-8"))
    crop = src["crop"]
    side = crop["pad_side"]
    ox, oy = (side - crop["w"]) // 2, (side - crop["h"]) // 2
    H, W = canvas_hw
    out = {}
    for fid, fp in zip(src["frame_ids"], sorted(seq_dir.glob("f*.png"))):
        au = np.array(Image.open(fp).convert("L")) < 128
        k = side / au.shape[0]
        ys, xs = np.where(au)
        m = np.zeros((H, W), bool)
        if len(ys):
            cy = np.clip((ys * k).astype(int) + crop["y"] - oy, 0, H - 1)
            cx = np.clip((xs * k).astype(int) + crop["x"] - ox, 0, W - 1)
            m[cy, cx] = True
            # nearest-pixel scatter leaves pinholes at k>1; close them so
            # the overlay reads as shapes, not dust
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((3, 3), np.uint8)).astype(bool)
        out[fid] = m
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--fps", type=float, default=None)
    a = ap.parse_args()

    re_root = ROOT / "projects" / a.project / "out" / "reassembled"
    out_dir = ROOT / "projects" / a.project / "out" / "video"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = defaultdict(list)              # scene -> [(seq_dir, re_dir)]
    for d in sorted(re_root.glob(f"{a.project}_*")) if re_root.is_dir() else []:
        seq = BENCH / "sequences" / d.name
        sj = seq / "source.json"
        if not sj.exists():
            continue
        src = json.loads(sj.read_text(encoding="utf-8"))
        sc = src.get("scene")
        if not sc or (a.scene and sc != a.scene):
            continue
        scenes[sc].append((seq, d, src))

    if not scenes:
        raise SystemExit(f"nothing reassembled for {a.project}"
                         + (f"/{a.scene}" if a.scene else ""))

    for sc, items in sorted(scenes.items()):
        canvas = items[0][2]["canvas"]
        H, W = int(canvas["h"]), int(canvas["w"])
        auth = defaultdict(lambda: np.zeros((H, W), bool))
        cast = defaultdict(lambda: np.zeros((H, W), bool))
        fps = a.fps or items[0][2].get("fps") or 5.0
        for seq, red, src in items:
            for fid, m in authored_on_canvas(seq, (H, W)).items():
                auth[fid] |= m
            rj = json.loads((red / "reassembly.json").read_text(encoding="utf-8"))
            fids = rj.get("source_frame_ids") or src["frame_ids"]
            for fid, fp in zip(fids, sorted(red.glob("f*.png"))):
                g = np.array(Image.open(fp).convert("L")) < 128
                if g.shape != (H, W):
                    g = np.array(Image.open(fp).convert("L").resize(
                        (W, H), Image.NEAREST)) < 128
                cast[fid] |= g
        ids = sorted(set(auth) | set(cast))
        name = f"{a.project}_{sc}_overlay.mp4"
        vw = cv2.VideoWriter(str(out_dir / name),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        ious = []
        for fid in ids:
            A, C = auth[fid], cast[fid]
            frame = np.full((H, W, 3), 255, np.uint8)
            frame[A & ~C] = (60, 60, 220)       # BGR red   — missed
            frame[C & ~A] = (220, 120, 40)      # BGR blue  — extra
            frame[A & C] = (70, 160, 40)        # BGR green — agreement
            vw.write(frame)
            u = (A | C).sum()
            if u:
                ious.append((A & C).sum() / u)
        vw.release()
        print(f"  {a.project}_{sc}: {len(ids)} frames, mean overlay IoU "
              f"{np.mean(ious):.3f} -> {out_dir / name}")


if __name__ == "__main__":
    main()
