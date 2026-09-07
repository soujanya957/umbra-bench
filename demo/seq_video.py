#!/usr/bin/env python3
"""seq_video.py — watch a solved sequence, without a project behind it.

    python demo/seq_video.py --sequence flower
    python demo/seq_video.py --sequence flower --mode shadow --slow 4
    python demo/seq_video.py --all

10_compose_video.py composites a PROJECT: reassembled clips, aligned by
source frame id, back onto the 1920x1080 canvas the footage came from. That
is the demo film, and a clip with no footage behind it -- all 13 generated
motions -- has no place in it and so has no way to be watched at all. This is
the other thing: one clip, the solve as it was solved, playable.

Two modes, and neither of them derives anything:

  compare  the solver's own per-frame `final_comparison.png` -- target,
           shadow, overlay, with that frame's IoU in the caption. The target
           it draws is the FITTED one the solver was actually given, which is
           the whole reason to use this file rather than pairing the authored
           frames against the shadow here: on a --fit-target clip that pairing
           reads as mis-registered, and none of it is solver error (the same
           trap atlas/README.md documents for the static cards).
  shadow   `best_shadow.png` alone, inverted to the repo's dark-on-white
           convention -- what the rig casts, with no comparison question in
           the picture at all.

Writes both an .mp4 (mp4v, the codec 10_compose_video.py settled on -- it
plays in QuickTime and VLC) and a .gif, because mp4v does NOT play in a
browser <video> and the studio page has to be able to show something.

The clip's own rate is 5 fps by construction (SEQUENCES.md: the demo family
is every 5th frame of 25 fps footage, and the generated family has no rate of
its own). That is real time and it is fast to read, so --slow repeats each
frame and --loop repeats the clip; both change the playback, never the
motion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent
OUT = ROOT / "out" / "seqvideo"
DEFAULT_FPS = 5.0


def latest_ts(run: Path) -> str | None:
    js = sorted(run.glob("summary_*.json"))
    return js[-1].stem[len("summary_"):] if js else None


def clip_fps(sid: str) -> float:
    """The clip's own rate: the footage fps where a reassembly recorded one."""
    proj = sid.split("_scene_")[0]
    for d in (ROOT / "projects" / proj / "out" / "reassembled" / sid,
              ROOT / "out" / "reassembled" / sid):
        p = d / "reassembly.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            if r.get("fps"):
                return float(r["fps"])
    return DEFAULT_FPS


def solved_sequences() -> list[str]:
    """Every sequence with a solve that has frames to show."""
    out = []
    for d in sorted((BENCH / "sequences").glob("*")):
        if not d.is_dir():
            continue
        run = BENCH / "optimized" / d.name
        ts = latest_ts(run) if run.is_dir() else None
        if ts and sorted(run.glob(f"frame_*_{ts}/best_shadow.png")):
            out.append(d.name)
    return out


def frames_for(sid: str, mode: str) -> list[np.ndarray]:
    """BGR frames, uniform size, in clip order."""
    run = BENCH / "optimized" / sid
    ts = latest_ts(run)
    if ts is None:
        sys.exit(f"[!] {sid}: no solve in optimized/{sid}")
    name = "final_comparison.png" if mode == "compare" else "best_shadow.png"
    paths = sorted(run.glob(f"frame_*_{ts}/{name}"))
    if not paths:
        sys.exit(f"[!] {sid}: no {name} under optimized/{sid}/frame_*_{ts}/ "
                 f"-- the solve predates it; re-solve, or use "
                 f"--mode {'shadow' if mode == 'compare' else 'compare'}")
    ims = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None:
            sys.exit(f"[!] {sid}: unreadable {p}")
        if im.ndim == 3 and im.shape[2] == 4:                 # flatten alpha
            a = im[:, :, 3:4].astype(np.float32) / 255.0
            im = (im[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
        if im.ndim == 2:
            # best_shadow is white-on-black; the repo's convention everywhere
            # else -- targets, sequences/, the atlas plates -- is dark ink on
            # white, so invert rather than shipping the one odd-looking film.
            im = cv2.cvtColor(255 - im, cv2.COLOR_GRAY2BGR)
        ims.append(im)
    h = max(i.shape[0] for i in ims)
    w = max(i.shape[1] for i in ims)
    # a shadow frame is 128px: nearest-neighbour, so the pixels stay pixels
    # instead of being blurred into a shape the solver never cast
    return [cv2.resize(i, (w, h), interpolation=cv2.INTER_NEAREST)
            if i.shape[:2] != (h, w) else i for i in ims]


def upscale(ims: list[np.ndarray], min_w: int) -> list[np.ndarray]:
    h, w = ims[0].shape[:2]
    k = max(1, int(round(min_w / w)))
    if k == 1:
        return ims
    return [cv2.resize(i, (w * k, h * k), interpolation=cv2.INTER_NEAREST)
            for i in ims]


def write_mp4(path: Path, ims: list[np.ndarray], fps: float) -> None:
    h, w = ims[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (w, h))
    if not vw.isOpened():
        sys.exit(f"[!] cannot open a writer for {path}")
    for im in ims:
        vw.write(im)
    vw.release()


def write_gif(path: Path, ims: list[np.ndarray], fps: float,
              max_w: int = 720) -> None:
    """The browser-playable copy. mp4v does not decode in a <video> tag, and
    the studio page needs to show the clip without a plugin or a re-encode."""
    from PIL import Image
    h, w = ims[0].shape[:2]
    if w > max_w:
        k = max_w / w
        ims = [cv2.resize(i, (max_w, max(1, int(round(h * k)))),
                          interpolation=cv2.INTER_AREA) for i in ims]
    pil = [Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB)) for i in ims]
    pil[0].save(path, save_all=True, append_images=pil[1:], loop=0,
                duration=max(20, int(round(1000.0 / max(fps, 0.1)))),
                optimize=True)


def render(sid: str, mode: str, fps: float | None, slow: int, loop: int,
           out_dir: Path, min_w: int) -> Path:
    ims = upscale(frames_for(sid, mode), min_w)
    n_src = len(ims)
    if slow > 1:
        ims = [im for im in ims for _ in range(slow)]
    if loop > 1:
        ims = ims * loop
    rate = fps if fps else clip_fps(sid)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{sid}_{mode}"
    write_mp4(stem.with_suffix(".mp4"), ims, rate)
    write_gif(stem.with_suffix(".gif"), ims, rate)
    h, w = ims[0].shape[:2]
    print(f"  {sid:<28} {mode:<7} {n_src} frames -> {len(ims)} @ {rate:g} fps "
          f"({w}x{h})  {stem.with_suffix('.mp4').relative_to(ROOT)} + .gif")
    return stem.with_suffix(".mp4")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", action="append", default=[],
                    help="a solved sequence id. Repeatable.")
    ap.add_argument("--all", action="store_true",
                    help="every solved sequence")
    ap.add_argument("--list", action="store_true",
                    help="print the solved sequence ids and exit")
    ap.add_argument("--mode", default="compare",
                    choices=("compare", "shadow"))
    ap.add_argument("--fps", type=float, default=None,
                    help=f"playback rate (default: the clip's own, "
                         f"else {DEFAULT_FPS:g})")
    ap.add_argument("--slow", type=int, default=1, metavar="N",
                    help="hold each frame N times -- playback only")
    ap.add_argument("--loop", type=int, default=1, metavar="N",
                    help="repeat the clip N times -- playback only")
    ap.add_argument("--min-width", type=int, default=384,
                    help="nearest-neighbour upscale target for small frames")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    if a.list:
        ids = solved_sequences()
        print(f"{len(ids)} solved sequence(s):")
        for i in ids:
            print(f"  {i}")
        return
    ids = solved_sequences() if a.all else a.sequence
    if not ids:
        sys.exit("[!] nothing to render: pass --sequence, --all  (--list to "
                 "browse)")
    for sid in ids:
        render(sid, a.mode, a.fps, max(1, a.slow), max(1, a.loop),
               Path(a.out), a.min_width)
    print(f"{len(ids)} clip(s) -> {a.out}")


if __name__ == "__main__":
    main()
