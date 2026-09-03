#!/usr/bin/env python3
"""
01_split_scenes.py — split an edited cut into scenes, extract every frame,
and build one review sheet per scene.

    python3 01_split_scenes.py FAMILY_trimmed.mp4

Boundary detection, in order of preference:

  white  — you separated the clips with pure-white frames. Those frames are
           the delimiters; they are detected, used as cut points, and left
           OUT of the extracted frames.
  scene  — no separators present; fall back to content-based cut detection.
           Butt-joined clips produce very strong scores, so this is reliable
           when the cuts are hard.

`--mode auto` (default) tries white first and falls back to scene.

Output
------
    scenes/scene_01/f0001.png …   every frame, split by scene, global ids
    review/scene_01.jpg …         ONE contact sheet per scene
    frames_manifest.csv           frame_id, file, scene, timestamp, src frame

Each scene is extracted by its own ffmpeg call, so no single step runs long
enough to be interrupted and leave a half-written frame set behind. Use
--scenes to redo just one, and --no-extract to rebuild sheets only.

Frame ids are GLOBAL and continuous across the whole video — they do not
restart per scene. That keeps them unique once step 2 regroups cutouts into
by_letter/, where per-scene numbering would collide.

Then segment a scene at a time:

    python3 02_segment_letters.py --frames-dir scenes/scene_01 --all
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ffmpeg_works() -> bool:
    """A which() hit is not enough: a conda ffmpeg with mismatched DLLs dies
    with STATUS_ENTRYPOINT_NOT_FOUND (0xC0000139), zero stderr, before main().
    Only an actually-runnable ffprobe counts; otherwise OpenCV does the work."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return False
    try:
        return run(["ffprobe", "-version"]).returncode == 0
    except OSError:
        return False


USE_CV2 = False
FRAME_OFF = 0


def need(tool: str) -> None:
    if shutil.which(tool) is None:
        sys.exit(f"{tool} not found. Install it with:  brew install ffmpeg")


# ---------------------------------------------------------------- probing
def probe(video: str) -> tuple[float, int, int, int]:
    """-> (fps, n_frames, width, height)"""
    p = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate,nb_frames,width,height",
             "-of", "default=noprint_wrappers=1", video])
    if p.returncode != 0:
        sys.exit(f"ffprobe failed on {video}:\n{p.stderr.strip()}")
    d = dict(l.split("=", 1) for l in p.stdout.strip().splitlines() if "=" in l)
    num, _, den = d.get("r_frame_rate", "25/1").partition("/")
    fps = float(num) / float(den or 1)
    try:
        n = int(d.get("nb_frames", "0"))
    except ValueError:
        n = 0
    return fps, n, int(d.get("width", 0)), int(d.get("height", 0))


def luma_stats(video: str) -> list[tuple[float, float]]:
    """Per-frame (YAVG, YMIN) at low res. Used to spot white separators."""
    p = run(["ffmpeg", "-v", "error", "-i", video, "-an",
             "-vf", "scale=96:54,signalstats,metadata=print:file=-",
             "-f", "null", "-"])
    avg, mn = [], []
    for line in p.stdout.splitlines():
        if "YAVG" in line:
            avg.append(float(line.rsplit("=", 1)[1]))
        elif "YMIN" in line:
            mn.append(float(line.rsplit("=", 1)[1]))
    if not avg:
        return []
    if len(mn) != len(avg):
        mn = [0.0] * len(avg)
    return list(zip(avg, mn))


def probe_cv(video: str):
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"OpenCV could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def luma_stats_cv(video: str):
    """One decode pass: per-frame (mean, min) luma at low res, plus the
    mean-absolute-difference to the previous frame for the scene fallback."""
    import cv2
    cap = cv2.VideoCapture(video)
    stats, diffs, prev = [], [], None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (96, 54)), cv2.COLOR_BGR2GRAY)
        stats.append((float(g.mean()), float(g.min())))
        diffs.append(float(abs(g.astype("f4") - prev).mean()) if prev is not None else 0.0)
        prev = g.astype("f4")
    cap.release()
    return stats, diffs


def scene_cuts_cv(diffs: list[float], fps: float, thresh: float) -> list[float]:
    """Cut timestamps from the decode pass's frame diffs. ffmpeg's `scene`
    score is normalised [0,1]; mean-abs-diff/255 is close enough for the
    butt-joined cuts this mode exists for (they score near the ceiling)."""
    return [i / fps for i, d in enumerate(diffs) if d / 255.0 > thresh]


def extract_scene_cv(video: str, out: Path, seg, fmt: str, sample: int) -> int:
    import cv2
    a, b = seg
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob(f"f*.{fmt}"):
        stale.unlink()
    keep = set(kept_frames(seg, sample))
    cap = cv2.VideoCapture(video)
    idx = 0
    while idx <= b:
        ok, f = cap.read()
        if not ok:
            break
        if idx in keep:
            cv2.imwrite(str(out / f"f{idx+1+FRAME_OFF:04d}.{fmt}"), f)
        idx += 1
    cap.release()
    return len(list(out.glob(f"f*.{fmt}")))


def scene_cuts(video: str, thresh: float) -> list[float]:
    """Content-change cut timestamps, in seconds."""
    p = run(["ffmpeg", "-v", "error", "-i", video, "-an",
             "-vf", f"select='gt(scene,{thresh})',metadata=print:file=-",
             "-f", "null", "-"])
    times, pending = [], None
    for line in p.stdout.splitlines():
        m = re.search(r"pts_time:([0-9.]+)", line)
        if m:
            pending = float(m.group(1))
        elif "scene_score" in line and pending is not None:
            times.append(pending)
            pending = None
    return times


# ---------------------------------------------------------------- cutting
def white_runs(stats, yavg_min: float, ymin_min: float) -> list[tuple[int, int]]:
    """Runs of near-white frames, as inclusive 0-based [start, end] pairs."""
    flags = [(a >= yavg_min and m >= ymin_min) for a, m in stats]
    runs, i = [], 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j + 1 < len(flags) and flags[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def segments_from_white(n: int, runs) -> list[tuple[int, int]]:
    """Frame spans between white runs, 0-based inclusive. Separators dropped."""
    segs, cur = [], 0
    for a, b in runs:
        if a > cur:
            segs.append((cur, a - 1))
        cur = b + 1
    if cur <= n - 1:
        segs.append((cur, n - 1))
    return segs


def segments_from_cuts(n: int, cut_frames: list[int]) -> list[tuple[int, int]]:
    bounds = [0] + sorted(set(cut_frames)) + [n]
    return [(bounds[i], bounds[i + 1] - 1)
            for i in range(len(bounds) - 1) if bounds[i + 1] - 1 >= bounds[i]]


# ---------------------------------------------------------------- outputs
def kept_frames(seg, sample: int) -> list[int]:
    """0-based frame indices this scene keeps, after sampling."""
    a, b = seg
    return list(range(a, b + 1, max(1, sample)))


def extract_scene(video: str, out: Path, seg, fmt: str, sample: int) -> int:
    """Extract one scene's frames, numbered with GLOBAL ids.

    One ffmpeg call per scene rather than one for the whole video: each
    finishes in well under a minute, so a long single job can't be cut off
    part-way and leave a half-written frame set behind.

    select='between(n,A,B)' is frame-exact. -ss seeking would be faster but
    lands on keyframes, which would shift the numbering. Sampling is folded
    into the same expression, anchored at the scene's own first frame so
    every scene keeps its opening pose.
    """
    a, b = seg
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob(f"f*.{fmt}"):
        stale.unlink()

    sel = f"between(n\\,{a}\\,{b})"
    if sample > 1:
        sel += f"*not(mod(n-{a}\\,{sample}))"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats",
           "-i", video, "-an", "-vf", f"select='{sel}'"]
    probe_h = run(["ffmpeg", "-hide_banner", "-h", "full"])
    cmd += (["-fps_mode", "passthrough"] if "fps_mode" in probe_h.stdout
            else ["-vsync", "0"])
    if fmt != "png":
        cmd += ["-q:v", "2"]

    # The image2 muxer numbers its outputs consecutively from -start_number,
    # so with sampling it would emit f0001,f0002,f0003 for what are really
    # frames 1,6,11. Write to a scratch dir, then rename to the true global
    # ids. Renaming out of a separate directory also avoids any collision
    # between temporary and final names.
    raw = out / ".raw"
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True)
    cmd += ["-start_number", "1", str(raw / f"f%05d.{fmt}")]

    p = subprocess.run(cmd)
    if p.returncode != 0:
        shutil.rmtree(raw, ignore_errors=True)
        sys.exit(f"extraction failed for {out}")

    produced = sorted(raw.glob(f"f*.{fmt}"))
    want = kept_frames(seg, sample)
    if len(produced) != len(want):
        print(f"  [!] ffmpeg produced {len(produced)} frame(s), "
              f"expected {len(want)} — ids may be off")
    for src, idx in zip(produced, want):
        src.rename(out / f"f{idx+1+FRAME_OFF:04d}.{fmt}")
    shutil.rmtree(raw, ignore_errors=True)
    return len(list(out.glob(f"f*.{fmt}")))


def sheet_for_scene(frames: Path, seg, idx: int, review: Path,
                    fmt: str, max_width: int, max_tiles: int,
                    sample: int) -> list[Path]:
    """One contact sheet per scene, with the frame id burned into each tile.

    Sampling leaves gaps in the numbering (f0001, f0006, f0011…), which the
    image2 demuxer's %04d pattern would stop at. So the tiles are fed from a
    concat list of the actual files, and each tile's label is computed from
    its position: id = scene start + tile index * sample.

    Long scenes get a wider grid rather than a taller one, and only split
    across several sheets past --max-tiles.
    """
    a, _ = seg
    files = sorted(frames.glob(f"f*.{fmt}"))
    if not files:
        print(f"  [warn] no frames in {frames}")
        return []
    review.mkdir(parents=True, exist_ok=True)
    made = []

    for ci in range(0, len(files), max_tiles):
        chunk = files[ci:ci + max_tiles]
        n = len(chunk)
        cols = max(1, math.ceil(math.sqrt(n)))          # roughly square sheet
        rows = math.ceil(n / cols)
        tw = max(120, min(360, max_width // cols))
        tw -= tw % 2
        multi = len(files) > max_tiles
        name = (f"scene_{idx:02d}_{ci//max_tiles+1}.jpg" if multi
                else f"scene_{idx:02d}.jpg")

        # Label from the files actually on disk, not from seg + index*sample.
        # If the scene folder was trimmed by hand after extraction, the
        # computed ids would drift; reading them back keeps the tile numbers
        # honest. Spacing is still uniform when trimming only took off ends.
        ids = [int(re.sub(r"\D", "", p.stem) or 0) for p in chunk]
        first_id = ids[0]
        steps = {b - a_ for a_, b in zip(ids, ids[1:])}
        step = steps.pop() if len(steps) == 1 else max(1, sample)
        if len(steps) > 0:
            print(f"  [warn] {frames.name} frame ids are unevenly spaced; "
                  f"tile labels past the first may be off")
        vf = (f"drawtext=text='%{{eif\\:n*{step}+{first_id}\\:d}}':"
              f"x=8:y=8:fontsize=42:fontcolor=white:box=1:"
              f"boxcolor=black@0.65:boxborderw=6,"
              f"scale={tw}:-1,tile={cols}x{rows}:padding=4:margin=6:"
              f"color=0x111111")

        if USE_CV2:
            import cv2
            import numpy as np
            tiles = []
            for p_, fid in zip(chunk, ids):
                im = cv2.imread(str(p_))
                th = max(1, round(tw * im.shape[0] / im.shape[1]))
                im = cv2.resize(im, (tw, th))
                cv2.rectangle(im, (4, 4), (4 + 30 + 13 * len(str(fid)), 34),
                              (0, 0, 0), -1)
                cv2.putText(im, f"f{fid:04d}", (8, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                tiles.append(im)
            th = max(t.shape[0] for t in tiles)
            blank = np.full((th, tw, 3), 17, np.uint8)
            tiles = [t if t.shape[0] == th else
                     cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0,
                                        cv2.BORDER_CONSTANT, value=(17, 17, 17))
                     for t in tiles]
            grid_rows = []
            for r in range(rows):
                row = tiles[r * cols:(r + 1) * cols]
                row += [blank] * (cols - len(row))
                grid_rows.append(cv2.hconcat(row))
            cv2.imwrite(str(review / name), cv2.vconcat(grid_rows),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            made.append(review / name)
            continue

        lst = frames / f".sheet_{ci}.txt"
        lst.write_text("".join(f"file '{p.name}'\n" for p in chunk))
        p = run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-f", "concat", "-safe", "0", "-r", "1", "-i", str(lst),
                 "-frames:v", "1", "-vf", vf, "-q:v", "3", "-y",
                 str(review / name)])
        lst.unlink(missing_ok=True)
        if p.returncode == 0:
            made.append(review / name)
        else:
            print(f"  [warn] sheet {name} failed: {p.stderr.strip()[:200]}")
    return made


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split an edited cut into scenes and review sheets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("video", nargs="?", help="edited video; default: the only one here")
    ap.add_argument("--mode", choices=["auto", "white", "scene", "manual"],
                    default="auto")
    ap.add_argument("--scene-thresh", type=float, default=0.30,
                    help="content-change score that counts as a cut")
    # Limited-range ("TV range") video codes white as Y=235, not 255, so a
    # genuinely white separator frame measures ~235. Content in this spot
    # tops out around 197, so 225 clears the separators with margin on both
    # sides. Raise toward 250 only for full-range footage.
    ap.add_argument("--white-yavg", type=float, default=225.0,
                    help="mean luma for a frame to count as a white separator")
    ap.add_argument("--white-ymin", type=float, default=180.0,
                    help="darkest pixel, so a bright shot isn't mistaken for white")
    ap.add_argument("--cuts", nargs="*", type=int, default=None,
                    help="manual cut frames (1-based, first frame of each new scene)")
    ap.add_argument("--fmt", choices=["png", "jpg"], default="png")
    ap.add_argument("--sample", type=int, default=1, metavar="N",
                    help="keep every Nth frame within each scene "
                         "(25 fps source: 5 -> 5 fps, 12 -> ~2 fps). "
                         "Anchored at each scene's first frame.")
    ap.add_argument("--max-sheet-width", type=int, default=4200)
    ap.add_argument("--max-tiles", type=int, default=400,
                    help="split a scene's sheet past this many tiles")
    ap.add_argument("--no-extract", action="store_true",
                    help="reuse already-extracted scene folders, just resheet")
    ap.add_argument("--scene-start", type=int, default=1,
                    help="number the first scene this (a numbered trim of a "
                         "project continues its predecessor's scene numbers, "
                         "so pixar_02's scenes never collide with pixar_01's)")
    ap.add_argument("--frame-start", type=int, default=0,
                    help="offset added to every global frame id (a later trim "
                         "of the same project continues its predecessor's ids, "
                         "so keypoints.json entries never collide)")
    ap.add_argument("--scenes", nargs="*", type=int, default=None,
                    metavar="N",
                    help="only process these scene numbers, e.g. --scenes 1 2. "
                         "Useful to spread a big job over several runs.")
    args = ap.parse_args()

    global USE_CV2, FRAME_OFF
    FRAME_OFF = args.frame_start
    USE_CV2 = not ffmpeg_works()
    if USE_CV2:
        print("backend: OpenCV (no runnable ffmpeg found -- a which() hit "
              "with broken DLLs also lands here)")
    here = Path(__file__).resolve().parent
    os.chdir(here)

    video = args.video
    if not video:
        vids = sorted(p for p in here.iterdir()
                      if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".m4v"))
        if len(vids) != 1:
            sys.exit("Name the video explicitly — found: "
                     + ", ".join(v.name for v in vids))
        video = str(vids[0])
    print(f"source : {video}")

    if USE_CV2:
        fps, n_probe, w, h = probe_cv(video)
        cv_stats, cv_diffs = luma_stats_cv(video)
    else:
        fps, n_probe, w, h = probe(video)
        cv_stats = cv_diffs = None
    print(f"         {w}x{h}  {fps:g} fps  {n_probe} frames")

    # ---- decide the boundaries -------------------------------------
    n = n_probe
    mode_used, segs = args.mode, None

    if args.mode == "manual":
        if not args.cuts:
            sys.exit("--mode manual needs --cuts")
        segs = segments_from_cuts(n, [c - 1 for c in args.cuts])
    else:
        if args.mode in ("auto", "white"):
            print("scanning for white separator frames...")
            stats = cv_stats if USE_CV2 else luma_stats(video)
            if stats:
                n = len(stats)
                runs = white_runs(stats, args.white_yavg, args.white_ymin)
                if runs:
                    segs, mode_used = segments_from_white(n, runs), "white"
                    dropped = sum(b - a + 1 for a, b in runs)
                    print(f"  found {len(runs)} separator run(s), "
                          f"{dropped} frame(s) dropped")
                else:
                    brightest = max(a for a, _ in stats)
                    print(f"  none found (brightest frame averages "
                          f"{brightest:.0f}/255, needs >= {args.white_yavg:.0f})")
                    if args.mode == "white":
                        sys.exit("No white separators. Re-export with them, "
                                 "or use --mode scene.")
        if segs is None:
            print(f"detecting cuts by content (score > {args.scene_thresh})...")
            times = (scene_cuts_cv(cv_diffs, fps, args.scene_thresh)
                     if USE_CV2 else scene_cuts(video, args.scene_thresh))
            cut_frames = sorted({int(round(t * fps)) for t in times})
            cut_frames = [c for c in cut_frames if 0 < c < n]
            segs, mode_used = segments_from_cuts(n, cut_frames), "scene"
            print(f"  {len(cut_frames)} cut(s) at frames "
                  f"{', '.join(str(c+1) for c in cut_frames) or '(none)'}")

    if not segs:
        sys.exit("No scenes resolved.")
    S = args.scene_start - 1
    if S:
        print(f"scene numbering continues at scene_{args.scene_start:02d}")

    print(f"\n{len(segs)} scene(s) by {mode_used}:")
    for i, (a, b) in enumerate(segs, 1):
        print(f"  scene_{i+S:02d}  frames f{a+1:04d}-f{b+1:04d}  "
              f"({b-a+1:>4} frames, {(b-a+1)/fps:5.2f}s)")

    # ---- extract + sheet, one scene at a time -----------------------
    scenes_root = Path("scenes")
    review = Path("review")
    review.mkdir(parents=True, exist_ok=True)

    todo = args.scenes or list(range(1, len(segs) + 1))
    bad = [s for s in todo if not 1 <= s <= len(segs)]
    if bad:
        sys.exit(f"--scenes out of range: {bad} (have 1..{len(segs)})")

    counts = {}
    for i in todo:
        seg = segs[i - 1]
        d = scenes_root / f"scene_{i+S:02d}"
        if args.no_extract:
            counts[i] = len(list(d.glob(f"f*.{args.fmt}")))
            print(f"\nscene_{i+S:02d}: reusing {counts[i]} frame(s)")
        else:
            print(f"\nscene_{i+S:02d}: extracting f{seg[0]+1:04d}-f{seg[1]+1:04d} "
                  f"-> {d}/")
            counts[i] = (extract_scene_cv if USE_CV2 else extract_scene)(
                video, d, seg, args.fmt, args.sample)
            want = len(kept_frames(seg, args.sample))
            flag = "" if counts[i] == want else f"  [!] expected {want}"
            print(f"  got {counts[i]} frame(s){flag}")
        for m in sheet_for_scene(d, seg, i + S, review, args.fmt,
                                 args.max_sheet_width, args.max_tiles,
                                 args.sample):
            print(f"  sheet: {m}")

    count = sum(counts.values())

    # ---- manifest ---------------------------------------------------
    # Written only on a full run; a partial --scenes pass would otherwise
    # drop the scenes it didn't touch.
    if len(todo) == len(segs):
        with open("frames_manifest.csv", "w", newline="") as fh:
            wtr = csv.writer(fh)
            wtr.writerow(["frame_id", "file", "scene",
                          "timestamp_sec", "source_frame"])
            # Built from the frames actually on disk, not from the computed
            # segments. After a plain run they're identical; after a manual
            # trim of the scene folders only the disk is right.
            for i, _ in enumerate(segs, 1):
                d = scenes_root / f"scene_{i+S:02d}"
                for p in sorted(d.glob(f"f*.{args.fmt}")):
                    k = int(re.sub(r"\D", "", p.stem) or 0) - 1
                    wtr.writerow([p.stem, str(p), f"scene_{i+S:02d}",
                                  f"{k/fps:.3f}", k])
        print(f"\nmanifest : frames_manifest.csv")
    else:
        print(f"\nmanifest : skipped (partial run: scenes {todo})")

    print("done.")
    print(f"  scenes : scenes/scene_NN/   ({count} frames total, global ids)")
    print(f"  sheets : review/scene_NN.jpg   (one per scene)")
    print(f"\nNext:  python3 02_segment_letters.py "
          f"--frames-dir scenes/scene_01 --all")


if __name__ == "__main__":
    main()
