#!/usr/bin/env python3
"""Build `sequences.jsonl` -- one record per sequence, the index for the track.

Mirrors `build_metadata.py` for the static side, with three differences that follow
from a sequence being an ordered thing rather than a bag of frames:

  * `frames` is ordered and authoritative. Frame order is data, not filesystem luck.
  * `target_motion` records how much the TARGET moves between frames. Every
    shadow-side temporal metric has to be read against it: a spinning star is
    supposed to change a lot between frames and a slow bloom is not, so raw shadow
    change is uninterpretable on its own.
  * `loop` is detected, not assumed, and it matters. `star_spin` covers one full
    72-degree visual period of a 5-pointed star in 5 frames, so the wrap from the
    last frame back to the first is a real transition of the same size as the
    others. If the wrap is not scored, a solver can unwind the whole rotation in
    the gap between the last frame and the first and pay nothing for it.

    The test is `wrap_iou` vs the mean step IoU, NOT `last frame == first frame`.
    A loop returns to the same *appearance*, and for a rotationally symmetric shape
    that happens before the frames repeat: star_spin's last frame is nowhere near
    identical to its first (IoU 0.565), yet its wrap step is the same size as every
    interior step (0.558-0.573), which is exactly what a closed loop looks like.

Per-frame shape attributes are filled in when `shape_attributes` can be imported
(needs cv2/scipy/skimage -- see requirements-eval.txt); without them the field is
left null and everything else still builds, matching how metrics.py degrades.

    python scripts/build_sequence_metadata.py
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)

try:
    from shape_attributes import compute_attributes as _attrs
except Exception:                       # optional; see module docstring
    _attrs = None

PROMPTS = {
    "star_spin": "a five-pointed star, rotating",
    "wiper": "a windshield wiper sweeping back and forth",
    "triangle": "a triangle moving across the frame",
    "flower": "a two-petal flower opening from a stem",
    "plant": "a plant growing",
    "bird": "a bird flapping its wings",
    "stick_wave": "a stick figure waving",
    "cheer": "a figure raising both arms in a cheer",
    "two_arm_wave": "a figure waving with both arms",
    "reeds": "reeds bending in the wind",
    "windmill": "a windmill turning",
}


def load(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) < 128


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def motion_stats(masks: list[np.ndarray], declared_loop: bool | None = None) -> dict:
    """How much the target itself moves -- the denominator for every shadow metric."""
    steps = [iou(masks[i], masks[i + 1]) for i in range(len(masks) - 1)]
    wrap = iou(masks[-1], masks[0])
    mean_step = float(np.mean(steps))
    return {
        "mean_step_iou": round(mean_step, 4),
        "min_step_iou": round(float(np.min(steps)), 4),
        "max_step_iou": round(float(np.max(steps)), 4),
        "wrap_iou": round(wrap, 4),
        # A closed loop wraps with a step the same size as its interior steps.
        # A declaration from the importer wins: the test cannot tell a slow shot
        # from a loop, and getting it wrong makes the metrics score a wrap that
        # is never performed.
        "loop": bool(wrap >= 0.90 * mean_step) if declared_loop is None else bool(declared_loop),
        "loop_source": "wrap-test" if declared_loop is None else "declared",
        "step_iou": [round(s, 4) for s in steps],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default=_BENCH)
    p.add_argument("--seq-dir", default="sequences")
    p.add_argument("--out", default="sequences.jsonl")
    p.add_argument("--no-attributes", action="store_true")
    a = p.parse_args()

    root = os.path.join(a.bench, a.seq_dir)
    if not os.path.isdir(root):
        p.error(f"{root} not found; run scripts/import_sequences.py first")
    if _attrs is None and not a.no_attributes:
        print("[!] shape_attributes unavailable (needs cv2/scipy/skimage); "
              "frame_attributes will be null -- re-run in the eval env to fill them")

    records = []
    for sid in sorted(os.listdir(root)):
        d = os.path.join(root, sid)
        if not os.path.isdir(d):
            continue
        # A sequence may declare what it is. The wrap test is a good heuristic
        # for generated loops and a bad one for a film cut, which moves slowly
        # enough between frames that its wrap looks like an interior step. An
        # importer that knows the answer writes it into source.json.
        source = {}
        sj = os.path.join(d, "source.json")
        if os.path.exists(sj):
            # A declaration that exists but cannot be parsed must not degrade
            # to the wrap test: that silently reverts the exact mislabel the
            # declaration was written to prevent. Fail with the filename.
            try:
                with open(sj, encoding="utf-8") as f:
                    source = json.load(f)
            except Exception as e:
                sys.exit(f"[!] {sj} exists but is unreadable: {e}")
        declared = source.get("loop")
        files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        if len(files) < 2:
            continue
        paths = [os.path.join(a.seq_dir, sid, f).replace("\\", "/") for f in files]
        masks = [load(os.path.join(d, f)) for f in files]
        H, W = masks[0].shape
        base = sid.rsplit("_n", 1)[0] if sid.endswith(("_n3", "_n5")) else sid
        # A film-cut letter carries its glyph in source.json; use the static
        # track's class convention (the bare letter) so the CLIP machinery can
        # score these frames against the same glyph set as letters_upper.
        cls = source.get("letter", base)
        prompt = (f"the letter {cls}, moving as it does in the source footage"
                  if "letter" in source
                  else PROMPTS.get(base, f"an animated {base.replace('_', ' ')}"))
        rec = {
            "id": sid,
            "track": "sequences",
            "class": cls,
            "prompt": prompt,
            "n_frames": len(files),
            "frame_size": [W, H],
            "frames": paths,
            "target_motion": motion_stats(masks, declared),
            "frame_attributes": (
                None if (_attrs is None or a.no_attributes)
                else [_attrs(m.astype(np.uint8) * 255) for m in masks]
            ),
            # One capture per source, as ordered frames. `joints` carries the pose
            # per frame: joint-space continuity is the metric the per-frame IoU
            # average cannot see, and it is unrecoverable from the masks alone.
            "shadows": {
                src: {"frames": None, "joints": None, "captured_at": None,
                      "operator": None, "run_id": None, "config": None, "notes": None}
                for src in ("hand", "teleop", "optimizer")
            },
            "rig": {"light": None, "screen_distance_m": None, "camera": None,
                    "n_arms": 5 if sid.endswith("_n5") else (3 if sid.endswith("_n3") else None)},
        }
        records.append(rec)

    out = os.path.join(a.bench, a.out)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    nf = sum(r["n_frames"] for r in records)
    print(f"[metadata] {len(records)} sequences, {nf} frames -> {out}\n")
    print(f"  {'id':<18}{'frames':>7}{'size':>10}{'loop':>7}{'mean step IoU':>15}")
    for r in records:
        m = r["target_motion"]
        print(f"  {r['id']:<18}{r['n_frames']:>7}{str(r['frame_size'][0]):>10}"
              f"{str(m['loop']):>7}{m['mean_step_iou']:>15.3f}")


if __name__ == "__main__":
    main()
