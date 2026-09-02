#!/usr/bin/env python3
"""10_compose_video.py — the solved shadows, back together as the shot.

    python 10_compose_video.py
    python 10_compose_video.py --scene scene_06 --fps 5

Stage 7 put each solved shadow back on the 1920x1080 canvas at the position the
glyph occupied in the source. That makes the last step possible and it is the
one the sequences track cannot do on its own: **letters that shared a frame in
the ad get composited back into that frame.** scene_06 is not six clips of one
letter each, it is F A M I L Y being spelled out, and L only enters at source
frame 0712 and Y at 0717 — the word assembles, which is the shot.

Alignment is by source frame id, not by index. The sequences have different
lengths because a letter is only tracked while it is on screen (scene_06: four
frames of L against ten of A), so pairing f03 with f03 would put different
moments of the ad in the same picture. `reassembly.json` carries
`source_frame_ids` and that is the key.

The composite is a union of the shadow masks. A rig casting two letters casts
both; where they overlap there is simply more shadow.

Output is one mp4 per scene plus one for the whole cut, written with OpenCV --
there is no ffmpeg on this machine and imageio-ffmpeg is not installed, so
cv2.VideoWriter with the mp4v fourcc is what is actually available.

Frame rate is the sequences' own: the shot was sampled every 5th frame of 25
fps footage, so 5 fps is real time. It looks slow because it is — playing it
faster would show motion the rig was never solved for.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def load_scene(re_root: Path):
    """Group reassembled sequences by scene, keyed on source frame id."""
    scenes = defaultdict(lambda: {"frames": defaultdict(list), "fps": None,
                                  "letters": set(), "canvas": None})
    for d in sorted(re_root.glob("demo_01_*")):
        rj = d / "reassembly.json"
        sj = ROOT / "sequences" / d.name / "source.json"
        if not rj.exists() or not sj.exists():
            continue
        r = json.loads(rj.read_text(encoding="utf-8"))
        s = json.loads(sj.read_text(encoding="utf-8"))
        ids = r.get("source_frame_ids") or s["frame_ids"]
        pngs = sorted(d.glob("f*.png"))
        if len(pngs) != len(ids):
            print(f"  {d.name}: {len(pngs)} frames but {len(ids)} ids, skipped")
            continue
        sc = scenes[s["scene"]]
        sc["fps"] = r.get("fps") or s.get("fps") or 5.0
        sc["letters"].add(s["letter"])
        sc["canvas"] = (s["canvas"]["w"], s["canvas"]["h"])
        for fid, p in zip(ids, pngs):
            sc["frames"][fid].append(p)
    return scenes


def composite(paths, size):
    """Union of the shadow masks: every letter the rig casts in this frame."""
    w, h = size
    out = np.full((h, w), 255, np.uint8)
    for p in paths:
        a = np.asarray(Image.open(p).convert("L"))
        if a.shape != (h, w):
            a = np.asarray(Image.fromarray(a).resize((w, h), Image.NEAREST))
        out = np.minimum(out, a)                    # dark = shadow
    return out


def write_mp4(frames, path: Path, fps: float):
    import cv2
    if not frames:
        return False
    h, w = frames[0].shape
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        return False
    for f in frames:
        vw.write(np.stack([f] * 3, axis=-1))
    vw.release()
    return path.exists() and path.stat().st_size > 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reassembled", default=str(ROOT / "results" / "demo_reassembled"))
    ap.add_argument("--out", default=str(ROOT / "results" / "demo_video"))
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--fps", type=float, default=None,
                    help="override; the default is the sequences' own rate")
    ap.add_argument("--no-film", action="store_true",
                    help="skip the concatenated cut")
    a = ap.parse_args()

    scenes = load_scene(Path(a.reassembled))
    names = a.scene or sorted(scenes)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    film, made = [], 0
    for sc in names:
        d = scenes.get(sc)
        if not d or not d["frames"]:
            print(f"  {sc}: nothing reassembled, skipped")
            continue
        ids = sorted(d["frames"])
        frames = [composite(d["frames"][i], d["canvas"]) for i in ids]
        fps = a.fps or d["fps"]
        p = out / f"{sc}.mp4"
        ok = write_mp4(frames, p, fps)
        letters = "".join(sorted(d["letters"]))
        print(f"  {sc:<10} {len(frames):>3} frames  {letters:<7} "
              f"{ids[0]}..{ids[-1]}  {fps:g}fps  -> {p.name}"
              + ("" if ok else "   WRITE FAILED"))
        made += ok
        film.extend(frames)

    if film and not a.no_film:
        fps = a.fps or 5.0
        p = out / "demo_01.mp4"
        ok = write_mp4(film, p, fps)
        print(f"\n  whole cut: {len(film)} frames at {fps:g}fps -> {p}"
              + ("" if ok else "   WRITE FAILED"))

    print(f"\n{made}/{len(names)} scenes written to {out}")
    print("dark = cast shadow, on the source 1920x1080 canvas.")
    print("frames are aligned by source frame id, so a letter that enters")
    print("part-way through a scene enters part-way through the video.")


if __name__ == "__main__":
    main()
