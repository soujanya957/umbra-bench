#!/usr/bin/env python3
"""05_split_objects.py — FALLBACK: recover one mask per object when the
keypoints put several objects into one.

Use `04_sam_segment.py` first. It already prompts SAM once per named object, so
if each glyph is labelled as its own object it produces one mask per glyph and
this script is unnecessary.

This exists for the case where that did not happen: every click of a frame
landed in a single object, so SAM is asked for "the thing containing these
points" and returns one blob spanning all of them. Measured on this footage,
scene_06 came back as the whole word — F, A, M, I, L and Y as one 38k-pixel mask
reaching from x=1189 to x=1794. No larger checkpoint fixes that;
sam2.1_hiera_large already is the largest, and the prompt is asking for exactly
what it returns.

This prompts SAM **once per positive point**, with that point's siblings as
negatives, and then merges the results that turn out to be the same object.

The merge is what makes it safe to run on every scene. Two clicks refining one
figure produce near-identical masks; two clicks on different glyphs produce
disjoint ones. Measured on this footage:

    f0001  2 clicks on one I-figure     masks overlap 97.4%  -> merge
    f0712  5 clicks on F A M I L        masks overlap  0.3%  -> keep apart

So the rule is read off the masks rather than guessed per scene, and a frame that
mixes both -- f0567 has small distinct bits and large shared ones -- resolves
correctly without being special-cased.

    python 05_split_objects.py --checkpoint <path>/sam2.1_hiera_large.pt
    python 05_split_objects.py --frames f0707 f0712      # just these
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def merge_groups(masks: list[np.ndarray], thr: float) -> list[list[int]]:
    """Union-find over pairwise IoU: same object if they mostly coincide."""
    parent = list(range(len(masks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if iou(masks[i], masks[j]) >= thr:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(masks)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keypoints", default=str(ROOT / "keypoints.json"))
    ap.add_argument("--out", default=str(ROOT / "letters_split"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frames", nargs="*")
    ap.add_argument("--merge-iou", type=float, default=0.55,
                    help="masks at or above this are the same object. The gap in "
                         "the data is wide -- 0.97 for same, 0.003 for different -- "
                         "so the exact value is not delicate.")
    ap.add_argument("--min-area", type=int, default=800,
                    help="drop specks; the smallest real glyph here is ~4.8k px")
    a = ap.parse_args()

    frames = json.loads(Path(a.keypoints).read_text(encoding="utf-8"))["frames"]
    ids = [f for f in sorted(frames) if not a.frames or f in a.frames]
    if not ids:
        raise SystemExit("no matching frames")

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print(f"[sam2] {Path(a.checkpoint).name} on {a.device}")
    predictor = SAM2ImagePredictor(build_sam2(a.config, a.checkpoint, device=a.device))

    out = Path(a.out)
    (out / "by_frame").mkdir(parents=True, exist_ok=True)
    (out / "overlay").mkdir(parents=True, exist_ok=True)
    palette = [(230, 60, 50), (60, 190, 80), (60, 130, 235), (245, 175, 30),
               (195, 60, 210), (40, 200, 200), (250, 120, 160), (150, 110, 60)]
    rows, n_obj = [], 0

    for n, fid in enumerate(ids, 1):
        rec = frames[fid]
        img = np.array(Image.open(ROOT / rec["file"]).convert("RGB"))
        predictor.set_image(img)
        # Every object, not just the first. A frame may already be labelled
        # correctly for some glyphs and lumped for others, and a fallback that
        # keeps objects[0] and drops the rest loses data without saying so.
        masks, scores, owners = [], [], []
        for lab, obj in rec["objects"].items():
            pos = [p for p, l in zip(obj["points"], obj["labels"]) if l == 1]
            neg = [p for p, l in zip(obj["points"], obj["labels"]) if l == 0]
            if not pos:
                continue
            # siblings within this object are negatives; points belonging to a
            # *different* labelled object are too, since they are not this shape
            other = [p for k, o in rec["objects"].items() if k != lab
                     for p, l in zip(o["points"], o["labels"]) if l == 1]
            for i in range(len(pos)):
                pts = ([pos[i]] + [pos[j] for j in range(len(pos)) if j != i]
                       + neg + other)
                lbl = ([1] + [0] * (len(pos) - 1) + [0] * len(neg)
                       + [0] * len(other))
                m, sc, _ = predictor.predict(point_coords=np.array(pts, np.float32),
                                             point_labels=np.array(lbl, np.int32),
                                             multimask_output=False)
                masks.append(m[0].astype(bool))
                scores.append(float(sc[0]))
                owners.append(lab)
        if not masks:
            continue

        groups = merge_groups(masks, a.merge_iou)
        merged = []
        for g in groups:
            mm = np.zeros_like(masks[0])
            for i in g:
                mm |= masks[i]
            if mm.sum() < a.min_area:
                continue
            xs = np.where(mm.any(axis=0))[0]
            labs = {owners[i] for i in g}
            merged.append({"mask": mm, "x0": int(xs.min()),
                           "score": max(scores[i] for i in g), "n_pts": len(g),
                           "label": sorted(labs)[0] if len(labs) == 1 else "+".join(sorted(labs))})
        merged.sort(key=lambda d: d["x0"])          # reading order

        fdir = out / "by_frame" / fid
        fdir.mkdir(parents=True, exist_ok=True)
        comp = img.astype(np.float32)
        for k, d in enumerate(merged):
            mm = d["mask"]
            tag = f"{d['label']}{k}" if d.get("label") else f"obj{k}"
            Image.fromarray(np.where(mm, 0, 255).astype(np.uint8)).save(
                fdir / f"{fid}_{tag}_mask.png")
            ys, xs = np.where(mm)
            c = palette[k % len(palette)]
            for ch in range(3):
                comp[..., ch] = np.where(mm, 0.45 * comp[..., ch] + 0.55 * c[ch], comp[..., ch])
            rows.append({"frame_id": fid, "scene": rec.get("scene", ""), "obj": k,
                         "label": d.get("label", ""),
                         "x": int(xs.min()), "y": int(ys.min()),
                         "w": int(xs.max() - xs.min() + 1), "h": int(ys.max() - ys.min() + 1),
                         "area": int(mm.sum()), "score": round(d["score"], 4),
                         "n_points": d["n_pts"],
                         "mask": f"by_frame/{fid}/{fid}_{tag}_mask.png"})
        Image.fromarray(comp.clip(0, 255).astype(np.uint8)).save(
            out / "overlay" / f"{fid}.jpg", quality=85)
        n_obj += len(merged)
        print(f"  {n}/{len(ids)}  {fid}  {len(masks)} prompts -> {len(merged)} object(s)")

    with open(out / "objects_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{n_obj} objects from {len(ids)} frames -> {out}")
    print(f"  overlay/ and objects_manifest.csv written")


if __name__ == "__main__":
    main()
