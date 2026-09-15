#!/usr/bin/env python3
"""00_trim.py — white frames in, separate clips out. The quick trimmer.

    python demo/00_trim.py edited.mp4                # -> edited_01.mp4 ...
    python demo/00_trim.py edited.mp4 --dry-run      # just show the cuts

Drop pure-white frames into your edit wherever a clip should end; this splits
the video at those separators and writes one mp4 per segment, white frames
dropped. Segmentation logic is imported from `01_split_scenes.py` (same
white_runs/segments_from_white, same thresholds), so what trims here splits
identically there — but the frame I/O is OpenCV, not ffmpeg, because this
machine has no ffmpeg (see 10_compose_video.py's header) and a trimmer that
cannot run where the videos are is a diagram, not a tool.

The difference from 01: 01 extracts frames into `scenes/` for the pipeline;
this writes plain clips for your editor, for review, or for feeding
`run_demo.py` one clip at a time. Audio is dropped — the pipeline is
silhouettes; keep sound in your editor.

If no white separators are found the video is left alone and the tool says
so, rather than guessing at content cuts. White frames are an explicit
instruction; content detection is 01's fallback business.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

# 01 starts with a digit, so a plain import cannot name it
_spec = importlib.util.spec_from_file_location("split_scenes",
                                               ROOT / "01_split_scenes.py")
_ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ss)


def luma_stats_cv(video: str):
    """Per-frame (mean, min) luma at low res — cv2 stand-in for 01's
    ffmpeg-signalstats version, same (YAVG, YMIN) contract."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stats = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (96, 54)), cv2.COLOR_BGR2GRAY)
        stats.append((float(g.mean()), float(g.min())))
    cap.release()
    return stats, fps, w, h


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out-dir", default=None,
                    help="default: next to the input")
    ap.add_argument("--white-yavg", type=float, default=225.0,
                    help="mean luma at or above this is 'white' (as in 01)")
    ap.add_argument("--white-ymin", type=float, default=180.0,
                    help="darkest pixel must also clear this (as in 01)")
    ap.add_argument("--min-frames", type=int, default=3,
                    help="segments shorter than this are dropped as debris")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = Path(a.video)
    if not src.exists():
        sys.exit(f"{src} not found")
    stats, fps, w, h = luma_stats_cv(str(src))
    if not stats:
        sys.exit("no frames read — is this a video?")
    runs = _ss.white_runs(stats, a.white_yavg, a.white_ymin)
    if not runs:
        sys.exit("no white separators found — nothing to trim. Insert pure-"
                 "white frames at the cut points, or lower --white-yavg "
                 f"(brightest frame mean seen: {max(s[0] for s in stats):.0f}).")
    segs = [s for s in _ss.segments_from_white(len(stats), runs)
            if s[1] - s[0] + 1 >= a.min_frames]

    out_dir = Path(a.out_dir) if a.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{src.name}: {len(stats)} frames @ {fps:g} fps, {w}x{h} — "
          f"{len(runs)} white run(s) -> {len(segs)} clip(s)\n")
    keep = set()
    writers = []
    for i, (s0, s1) in enumerate(segs, 1):
        dst = out_dir / f"{src.stem}_{i:02d}.mp4"
        print(f"  {dst.name:<30} frames {s0:>5}-{s1:<5} "
              f"{(s1 - s0 + 1) / fps:6.2f}s")
        writers.append((s0, s1, dst))
        keep.update(range(s0, s1 + 1))
    if a.dry_run:
        print("\ndry run — drop --dry-run to write the clips")
        return

    cap = cv2.VideoCapture(str(src))
    cur, vw = None, None
    idx = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        seg = next((t for t in writers if t[0] <= idx <= t[1]), None)
        if seg is not None:
            if cur is not seg:
                if vw is not None:
                    vw.release()
                cur = seg
                vw = cv2.VideoWriter(str(seg[2]),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (w, h))
                if not vw.isOpened():
                    sys.exit(f"could not open writer for {seg[2]}")
            vw.write(f)
        idx += 1
    if vw is not None:
        vw.release()
    cap.release()
    print(f"\n{len(writers)} clip(s) -> {out_dir}")
    print("next: python demo/run_demo.py --video <clip> --demo-id NN")


if __name__ == "__main__":
    main()
