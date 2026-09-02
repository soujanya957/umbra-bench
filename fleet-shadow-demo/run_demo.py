#!/usr/bin/env python3
"""run_demo.py — video in, sequences track out, with one human step in the middle.

    python run_demo.py --video FAMILY_trimmed.mp4 --demo-id 02

Runs every stage that can be automated, stops at the one that cannot, and picks
up where it left off when you run it again. Each stage is skipped when its output
is already there, so re-running after fixing a few labels is cheap.

    1  split scenes + extract frames        01_split_scenes.py     auto
    2  label one object per glyph           03_label_keypoints.py  ← YOU
    3  segment from the keypoints           04_sam_segment.py      auto
    4  clean the masks                      06_clean_masks.py      auto
    5  group into sequences, crop per clip  07_make_sequences.py   auto
    6  index the track                      build_sequence_metadata.py  auto

Only step 2 needs a person, and only because deciding *which* shape in a frame is
the subject is not something the pixels answer.

What the defaults encode
------------------------

**SAM2 small, not large.** Measured over 159 masks from this footage: large
produced 866 interior holes and 707 detached fragments, small 213 and 54, and
small was also smoother (1.96 vs 2.02) and three times faster. The large model
resolves the glyph's own dark outline and the anti-aliased edge and excludes
them, which is exactly the detail a silhouette target does not want. Bigger is
worse here; `--sam-model large` if you want to re-check that on new footage.

**One named object per glyph, not many points in one object.** SAM answers the
question it is asked. Points for F, A, M, I, L and Y inside a single object asks
for "the thing containing these", and it returns one 38k-pixel mask spanning the
whole word. Label each glyph as its own object and `04_sam_segment.py` prompts
once per object. `05_split_objects.py` exists to recover the other case.

**Per-clip crop, never per-frame.** A glyph travels 39-329 px across its scene
and that motion is the content. Cropping each frame to its own bounding box
centres every frame identically and plays back as a shape twitching in place.
07 computes one box over the union of each sequence.

**Every target is one connected shadow.** A rig cannot cast a floating piece, so
06 drops specks, bridges anything large enough to be a real part (the cane in
scene_03 is 5% of its figure and 5-8 px from the hand), and verifies the result
is a single component.

**The outline stays.** `06 --colour` shrinks each mask to the glyph's own hue,
which also severs the cane at the dark stroke where it meets the hand. Off by
default; the black border is part of the shape.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent

SAM_CKPT = {
    "small": ("C:/Users/hexia/Documents/GitHub/animal_inspired_BC/thirdparty/sam2/"
              "checkpoints/sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "large": ("C:/Users/hexia/Documents/GitHub/animal_inspired_BC/FetchHound_Bench/"
              "ckpt/sam2/sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}


def run(cmd: list[str], label: str) -> None:
    print(f"\n=== {label}\n    {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"[{label}] exited {r.returncode}")


def count(pattern: str, root: Path = ROOT) -> int:
    return sum(1 for _ in root.glob(pattern))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=None, help="source mp4; omit to reuse scenes/")
    ap.add_argument("--demo-id", default="01", help="names become demo_<id>_<scene>_<letter>")
    ap.add_argument("--sample", type=int, default=5, help="keep every Nth frame")
    ap.add_argument("--sam-model", choices=("small", "large"), default="small")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sigma", type=float, default=3.0, help="mask edge smoothing")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--from-stage", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="re-run stages that have output")
    a = ap.parse_args()
    py = a.python

    kp = ROOT / "keypoints.json"

    # 1 — frames
    if a.from_stage <= 1 and (a.force or not count("scenes/*/*.png")):
        if not a.video:
            raise SystemExit("no scenes/ yet and no --video given")
        run([py, "01_split_scenes.py", a.video, "--sample", str(a.sample)], "1  scenes + frames")
    n_frames = count("scenes/*/*.png")
    print(f"[1] {n_frames} frames in scenes/")

    # 2 — the human step
    n_lab = 0
    if kp.exists():
        n_lab = len(json.loads(kp.read_text(encoding="utf-8")).get("frames", {}))
    if n_lab < n_frames:
        print(f"\n[2] {n_lab}/{n_frames} frames labelled — this is the manual step.\n"
              f"    python 03_label_keypoints.py\n"
              f"    Label each glyph as its OWN named object (F, A, M, …), not\n"
              f"    several points inside one object. Re-run run_demo.py after.")
        return
    print(f"[2] {n_lab}/{n_frames} frames labelled")

    ckpt, cfg = SAM_CKPT[a.sam_model]
    if not Path(ckpt).exists():
        raise SystemExit(f"SAM2 checkpoint not found: {ckpt}")

    # 3 — segment
    seg = ROOT / f"letters_sam2_{a.sam_model}"
    if a.from_stage <= 3 and (a.force or not count("by_frame/*/*_mask.png", seg)):
        run([py, "04_sam_segment.py", "--backend", "sam2", "--device", a.device,
             "--sam2-checkpoint", ckpt, "--sam2-config", cfg,
             "--out", seg.name], f"3  SAM2 {a.sam_model}")
    print(f"[3] {count('by_frame/*/*_mask.png', seg)} masks in {seg.name}/")

    # 4 — clean
    clean = ROOT / "letters_clean"
    if a.from_stage <= 4 and (a.force or not count("*_mask.png", clean)):
        run([py, "06_clean_masks.py", "--in", seg.name, "--out", clean.name,
             "--sigma", str(a.sigma)], "4  clean")
    print(f"[4] {count('*_mask.png', clean)} cleaned masks")

    # 5 — sequences
    if a.from_stage <= 5:
        run([py, "07_make_sequences.py", "--in", clean.name,
             "--demo-id", a.demo_id], "5  sequences")

    # 6 — index
    if a.from_stage <= 6:
        run([a.python, str(BENCH / "scripts" / "build_sequence_metadata.py")], "6  index")

    seqs = sorted(p.name for p in (BENCH / "sequences").glob(f"demo_{a.demo_id}_*"))
    print(f"\ndone — {len(seqs)} sequences in the track")
    for s in seqs:
        print(f"  {s}  ({count('f*.png', BENCH / 'sequences' / s)} frames)")
    print(f"\nreview: {BENCH / 'results' / 'demo_review' / '_all.png'}")
    print("solve:  see README.md § Solving")


if __name__ == "__main__":
    main()
