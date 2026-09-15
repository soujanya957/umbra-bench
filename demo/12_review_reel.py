#!/usr/bin/env python3
"""12_review_reel.py — one whole-cut film PER MODE, and the video package.

    python demo/12_review_reel.py --project ICRA

10_compose already writes the composite whole cut (<P>.mp4); the overlay
and mujoco films existed per scene only. This concatenates each mode's
scenes in order into:

    projects/<P>/out/video/<P>_overlay.mp4
    projects/<P>/out/video/<P>_mujoco.mp4

so every mode has a scene-by-scene AND a whole-project film, and the
package's video/ folder (pack.py copies out/video) carries the full set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent


def concat(files, out: Path) -> bool:
    vw = None
    n = 0
    for f in files:
        cap = cv2.VideoCapture(str(f))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if vw is None:
                h, w = frame.shape[:2]
                vw = cv2.VideoWriter(str(out),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (w, h))
            elif frame.shape[1::-1] != (vw_w, vw_h):
                frame = cv2.resize(frame, (vw_w, vw_h))
            if vw is not None and n == 0:
                vw_w, vw_h = frame.shape[1], frame.shape[0]
            vw.write(frame)
            n += 1
        cap.release()
    if vw:
        vw.release()
        print(f"  {out.name}: {len(files)} scene(s), {n} frames")
        return True
    return False


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    a = ap.parse_args()

    vdir = ROOT / "projects" / a.project / "out" / "video"
    made = 0
    for suffix in ("overlay", "mujoco"):
        files = sorted(vdir.glob(f"{a.project}_scene_*_{suffix}.mp4"))
        if not files:
            print(f"  no per-scene {suffix} films yet -- run "
                  f"{'11_overlay_video' if suffix == 'overlay' else 'ensemble_mujoco --record'} first")
            continue
        made += concat(files, vdir / f"{a.project}_{suffix}.mp4")
    if made:
        print("repack (studio: 8 pack) to fold these into the package's video/")


if __name__ == "__main__":
    main()
