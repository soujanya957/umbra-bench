"""Solve the ICRA story keyframes: one chained sequence per (clip, subject), on its stage.

Takes results/icra_story/keyframes/targets.json (scripts/story_keyframes.py). Each subject's
keyframes, in film order (video part, then frame), become one sequence for fleet's
run_sequence.py: frame 1 starts from the stage's pose library, every later keyframe from the
one before, so consecutive poses stay close and can be interpolated. --fit-target fits one
similarity per sequence, so the subject is cast as large as the stage casts it well (the
size is free; the relative size and ground line inside a sequence are kept).

Stages (decided 2026-11-13): slider clips on LeSlider, arm_transformation clips on three
arms in a row, no LeSlider; the cutout clip (the rising '+') is not an arm solve and is left
out here. Every subject is centred, displacement comes later.

    python scripts/story_solve.py prep            # square targets + the run list
    python scripts/story_solve.py run [--only c02]
    python scripts/story_solve.py report

--independent (2026-11-13, after the chained run cast every subject small and in the base
shadows): every keyframe is cropped tight to its own subject and solved on its own, with its
own fit (scale up to --fit-scale-max) and a pose-library start, into results/icra_story/
solve_indep/. Sizes then differ between keyframes; poses are no longer chained.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import cv2
import numpy as np

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KF = os.path.join(BENCH, "results", "icra_story", "keyframes")
OUT = os.path.join(BENCH, "results", "icra_story", "solve")
FLEET = os.path.join(os.path.dirname(BENCH), "fleet-shadow-art", "motion-aware-shadow")
PY = r"C:\Users\hexia\miniconda3\envs\fh_l1\python.exe"
STAGES = {
    "leslider": os.path.join(BENCH, "results", "fastshadow", "pose_library_leslider", "stage.json"),
    "three_in_a_row": os.path.join(BENCH, "results", "fastshadow", "pose_library_row3", "stage.json"),
}
SLIDER_CLIPS = {1, 3, 4, 6}
CUTOUT_CLIPS = {5}
MARGIN = 40  # film px of white kept around the widest frame of a sequence
TIGHT = 0.06  # --independent: margin around a keyframe's own subject, as a fraction of its size


def part_no(video):
    m = re.search(r"-(\d+)\.\w+$", video)
    return int(m.group(1)) if m else 0


def prep(a):
    T = json.load(open(os.path.join(KF, "targets.json"), encoding="utf-8"))["targets"]
    seqs = {}
    for t in T:
        cno = int(re.match(r"clip_(\d+)", t["clip"]).group(1))
        if cno in CUTOUT_CLIPS:
            continue
        seqs.setdefault((cno, t["subject"]), []).append(t)
    runs = []
    for (cno, subj), ts in sorted(seqs.items()):
        # film order: the video's folder, its part number, then the frame
        ts.sort(key=lambda t: (os.path.dirname(t["video"]), part_no(t["video"]), t["frame"]))
        name = f"c{cno:02d}_{re.sub(r'[^0-9A-Za-z+]+', '-', subj).strip('-')}"
        d = os.path.join(OUT, name, "targets")
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, "*.png")):
            os.remove(f)
        side = max(1080, max(t["width_film_px"] for t in ts) + 2 * MARGIN, max(t["height_film_px"] for t in ts) + MARGIN)
        frames = []
        for i, t in enumerate(ts):
            im = cv2.imread(os.path.join(KF, t["id"] + ".png"), cv2.IMREAD_GRAYSCALE)
            H, W = im.shape
            if a.independent:  # a square just around this keyframe's subject
                ys, xs = np.nonzero(im < 128)
                h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
                side = int(max(h, w) * (1 + 2 * TIGHT))
                sq = np.full((side, side), 255, np.uint8)
                sub = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                y0, x0 = (side - h) // 2, (side - w) // 2
                sq[y0:y0 + h, x0:x0 + w] = sub
            else:
                sq = np.full((side, side), 255, np.uint8)
                # centred horizontally, ground (canvas bottom row) kept at the square's bottom
                x0 = (W - side) // 2
                src = im[max(0, H - side):, max(0, x0):max(0, x0) + min(side, W)]
                sq[side - src.shape[0]:, (side - src.shape[1]) // 2:(side - src.shape[1]) // 2 + src.shape[1]] = src
            p = os.path.join(d, f"kf_{i:02d}.png")
            cv2.imwrite(p, sq)
            frames.append({"png": p, **{k: t[k] for k in ("id", "video", "frame", "subject")}})
        stage = "leslider" if cno in SLIDER_CLIPS else "three_in_a_row"
        runs.append({"name": name, "clip": cno, "subject": subj, "stage": stage, "frames": frames})
        print(f"{name:<22} {stage:<15} {len(frames):2d} keyframes  square {side}px")
    json.dump(runs, open(os.path.join(OUT, "runs.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(runs)} sequences, {sum(len(r['frames']) for r in runs)} keyframes")


def run(a):
    runs = json.load(open(os.path.join(OUT, "runs.json"), encoding="utf-8"))
    if a.only:
        runs = [r for r in runs if r["name"].startswith(a.only)]
    for r in runs:
        stage = json.load(open(STAGES[r["stage"]], encoding="utf-8"))
        jobs = ([(os.path.join(OUT, r["name"], f"solve_k{i:02d}"), [f["png"]]) for i, f in enumerate(r["frames"])]
                if a.independent else [(os.path.join(OUT, r["name"], "solve"), [f["png"] for f in r["frames"]])])
        for out, pngs in jobs:
            if glob.glob(os.path.join(out, "summary_*.json")) and not a.force:
                print(f"{os.path.relpath(out, OUT)}: done, skipped", flush=True)
                continue
            cmd = [PY, "scripts/run_sequence.py", "--urdf", "urdf/SO101/so101_new_calib.urdf",
                   "--stage-json", STAGES[r["stage"]], "--n-robots", str(len(stage["bases"])),
                   "--renderer", "nvdiffrast", "--floor", "mesh", "--fit-target",
                   "--fit-scale-max", str(a.fit_scale_max),
                   "--pose-library", "auto", "--outdir", out, "--targets", *pngs]
            t0 = time.time()
            os.makedirs(out, exist_ok=True)
            log = open(os.path.join(out, "solve.log"), "w", encoding="utf-8")
            p = subprocess.run(cmd, cwd=FLEET, stdout=log, stderr=subprocess.STDOUT)
            print(f"{os.path.relpath(out, OUT)}: {len(pngs)} target(s) on {r['stage']}, exit {p.returncode} "
                  f"in {time.time() - t0:.0f} s", flush=True)


def solved_frames(name, out=None):
    """The solved keyframes of a sequence in order (summary frame records), chained or not."""
    out = out or OUT
    one = sorted(glob.glob(os.path.join(out, name, "solve", "summary_*.json")))
    if one:
        return json.load(open(one[-1], encoding="utf-8"))["frames"]
    fr = []
    for d in sorted(glob.glob(os.path.join(out, name, "solve_k*"))):
        s = sorted(glob.glob(os.path.join(d, "summary_*.json")))
        if s:
            fr += json.load(open(s[-1], encoding="utf-8"))["frames"]
    return fr


def report(a):
    runs = json.load(open(os.path.join(OUT, "runs.json"), encoding="utf-8"))
    print(f"{'sequence':<22}{'stage':<16}{'kf':>3}  IoU per keyframe")
    for r in runs:
        ious = [f["iou"] for f in solved_frames(r["name"])]
        if not ious:
            print(f"{r['name']:<22}{r['stage']:<16}{len(r['frames']):>3}  (not solved)")
            continue
        print(f"{r['name']:<22}{r['stage']:<16}{len(r['frames']):>3}  " + " ".join(f"{x:.2f}" for x in ious if x is not None))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prep", "run", "report"])
    p.add_argument("--only", default=None, help="run: sequences whose name starts with this")
    p.add_argument("--force", action="store_true")
    p.add_argument("--independent", action="store_true", help="tight crop and one solve per keyframe")
    p.add_argument("--fit-scale-max", type=float, default=1.6)
    a = p.parse_args()
    if a.independent:
        OUT = os.path.join(BENCH, "results", "icra_story", "solve_indep")
    {"prep": prep, "run": run, "report": report}[a.cmd](a)
