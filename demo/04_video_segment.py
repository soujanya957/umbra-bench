#!/usr/bin/env python3
"""04_video_segment.py — click one frame per scene, propagation does the rest.

    python 04_video_segment.py                       # SAM2 video propagation
    python 04_video_segment.py --backend sam3        # SAM3 tracker (downloads
                                                     # facebook/sam3 on first use)

The per-frame segmenter (04_sam_segment.py) needs points on EVERY frame, which
is why labelling family_ad meant clicking through 92 frames. A video predictor
needs points on ONE frame per (scene, object): it tracks the object through
the clip, forward and backward from the seed frame. Label a scene's first
clear frame, run this, and the whole scene is masked.

Seeds come from the same keypoints.json the labeller writes — for each
(scene, object) the earliest labelled frame is the seed and any OTHER labelled
frames of that object become correction seeds (extra clicks where the track
drifted). Reuse-marked labels are skipped exactly as in 04.

Outputs are byte-compatible with 04_sam_segment (same by_frame/ layout via its
own write_outputs), so 06_clean_masks and everything after run unchanged.

SAM2's video predictor wants a directory of JPEGs, so each scene's PNGs are
mirrored into a temp dir; frame index mapping rides on the sorted order.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("sam_seg", ROOT / "04_sam_segment.py")
_seg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_seg)

SAM2_CKPT = ("C:/Users/hexia/Documents/GitHub/animal_inspired_BC/thirdparty/"
             "sam2/checkpoints/sam2.1_hiera_small.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"


def scene_frames(kp: dict, scene: str) -> list[tuple[str, Path]]:
    """(fid, path) for every frame of the scene ON DISK, sorted by id."""
    d = Path("scenes") / scene
    out = []
    for p in sorted(d.glob("f*.png")):
        out.append((p.stem, p))
    return out


def seeds_for(kp: dict, scene: str):
    """{label: [(frame_pos, points, labels), ...]} — earliest first."""
    frames = scene_frames(kp, scene)
    pos_of = {fid: i for i, (fid, _) in enumerate(frames)}
    decisions = kp.get("decisions", {}).get(scene, {})
    out: dict[str, list] = {}
    for fid, rec in sorted(kp.get("frames", {}).items()):
        if rec.get("scene") != scene or fid not in pos_of:
            continue
        for lab, o in rec.get("objects", {}).items():
            if lab in decisions:                     # reuse-marked: skip
                continue
            if not o.get("points") or not any(o.get("labels", [])):
                continue
            out.setdefault(lab, []).append(
                (pos_of[fid], np.array(o["points"], np.float32),
                 np.array(o["labels"], np.int32)))
    return out, frames


def run_sam2(device: str):
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    pred = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)
    return pred, torch


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=None,
                    help="project folder holding scenes/ and keypoints.json")
    ap.add_argument("--keypoints", default="keypoints.json")
    ap.add_argument("--out", default="letters_sam2_small")
    ap.add_argument("--backend", choices=["sam2", "sam3"], default="sam2")
    ap.add_argument("--sam3-model", default="facebook/sam3",
                    help="HF id; downloads several GB on first use")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--no-overlay", action="store_true")
    a = ap.parse_args()

    import os
    os.chdir(Path(a.workdir).resolve() if a.workdir else ROOT)
    kp = json.loads(Path(a.keypoints).read_text(encoding="utf-8"))
    scenes = sorted({r["scene"] for r in kp.get("frames", {}).values()
                     if r.get("objects")})
    if a.scenes:
        scenes = [s for s in scenes if s in a.scenes]
    if not scenes:
        sys.exit("no labelled frames in keypoints.json — label at least one "
                 "frame per scene in the studio (or 03) first")

    out = Path(a.out)
    if a.backend == "sam3":
        sys.exit("sam3 backend: checkpoint download not yet approved -- run "
                 "with --backend sam2, or say the word and it gets wired "
                 "(Sam3TrackerVideoModel is already in transformers 5.15).")

    pred, torch = run_sam2(a.device)
    total = 0
    for scene in scenes:
        seeds, frames = seeds_for(kp, scene)
        if not seeds:
            print(f"{scene}: labels present but all reuse-marked/negative — skipped")
            continue
        print(f"{scene}: {len(frames)} frames, objects: "
              + ", ".join(f"{l} (seed f@{s[0][0]})" for l, s in seeds.items()))

        with tempfile.TemporaryDirectory(prefix="s2v_") as td:
            for i, (_, p) in enumerate(frames):
                Image.open(p).convert("RGB").save(f"{td}/{i:05d}.jpg",
                                                  quality=95)
            # SAM2 stores its memory bank in bfloat16 by design and expects
            # autocast at inference (Meta's reference usage). Without it the
            # reverse pass -- any seed not on frame 0 -- dies with
            # "mat1 and mat2 must have the same dtype".
            import contextlib
            ac = (torch.autocast("cuda", dtype=torch.bfloat16)
                  if a.device == "cuda" else contextlib.nullcontext())
            with torch.inference_mode(), ac:
                state = pred.init_state(video_path=td)
                obj_ids = {}
                for oi, (lab, seedlist) in enumerate(sorted(seeds.items()), 1):
                    obj_ids[oi] = lab
                    for fpos, pts, labs in seedlist:
                        pred.add_new_points_or_box(
                            state, frame_idx=fpos, obj_id=oi,
                            points=pts, labels=labs)
                masks: dict[int, dict[int, np.ndarray]] = {}
                for reverse in (False, True):
                    for fidx, oids, logits in pred.propagate_in_video(
                            state, reverse=reverse):
                        for k, oid in enumerate(oids):
                            m = (logits[k] > 0).squeeze().cpu().numpy()
                            masks.setdefault(fidx, {})[int(oid)] = m

        for i, (fid, p) in enumerate(frames):
            items = []
            img = __import__("cv2").imread(str(p))
            for oid, m in sorted(masks.get(i, {}).items()):
                lab = obj_ids[oid]
                mb = m.astype(bool)
                if not mb.any():
                    continue
                # seed clicks, when this frame carries them, for the overlay
                rec = kp["frames"].get(fid, {}).get("objects", {}).get(lab, {})
                pts = np.array(rec.get("points", []), np.float32).reshape(-1, 2)
                labs = np.array(rec.get("labels", []), np.int32)
                _seg.write_outputs(out, fid, scene, lab, img, mb,
                                   full_mask=True, white_on_black=False)
                items.append((lab, mb, pts, labs, 1.0))
                total += 1
            if items and not a.no_overlay:
                _seg.write_overlay(out, fid, img, items)
    print(f"\n{total} masks -> {out}/ (video-propagated)")

    print("next: 06_clean_masks.py, or the studio's clean button")


if __name__ == "__main__":
    main()
