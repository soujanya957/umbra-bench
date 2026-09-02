#!/usr/bin/env python3
"""
04_sam_segment.py — turn the clicked keypoints into masks with SAM.

Run this on the GPU box. It reads `keypoints.json` from 03_label_keypoints.py
and writes the same output tree 02_segment_letters.py produced, so anything
downstream keeps working.

    python3 04_sam_segment.py \
        --sam2-checkpoint checkpoints/sam2.1_hiera_large.pt \
        --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml

    python3 04_sam_segment.py --dry-run      # no GPU: check the JSON first

Outputs (under --out, default `letters/`)
    by_frame/f0001/f0001_I.png        RGBA cutout, tight crop
    by_frame/f0001/f0001_I_mask.png   full-canvas mask, glyph BLACK on WHITE
    by_letter/I/f0001_I.png           regrouped per glyph
    overlay/f0001.jpg                 QC render
    letters_manifest.csv              frame, scene, letter, bbox, area, score

Backends
--------
`--backend sam2` uses SAM2's SAM2ImagePredictor, which takes point_coords /
point_labels directly. `--backend hf` uses transformers' SamModel, which
covers SAM 1 and any SAM checkpoint published on the Hub.

If you are on SAM 3, check its predictor's call signature before trusting the
sam2 path — the point-prompt API is what matters here, and only the two
backends above are verified. `--dry-run` tells you whether your JSON is sound
without loading any model at all.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv missing:  pip install opencv-python")


# ==========================================================================
# Backends
# ==========================================================================
class Sam2Backend:
    """Meta's SAM2 image predictor."""

    def __init__(self, checkpoint: str, config: str, device: str | None):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
        self.device = device
        self.predictor = SAM2ImagePredictor(
            build_sam2(config, checkpoint, device=device))
        print(f"[sam2] {Path(checkpoint).name} on {device}")

    def set_image(self, rgb: np.ndarray) -> None:
        self.predictor.set_image(rgb)

    def predict(self, pts: np.ndarray, labs: np.ndarray):
        masks, scores, _ = self.predictor.predict(
            point_coords=pts, point_labels=labs, multimask_output=True)
        masks = np.asarray(masks)
        scores = np.asarray(scores).ravel()
        best = int(scores.argmax())          # multimask + pick best by score
        return masks[best].astype(bool), float(scores[best])


class HfSamBackend:
    """transformers SamModel — works for SAM 1 and Hub-published checkpoints."""

    def __init__(self, model_id: str, device: str | None):
        import torch
        from transformers import SamModel, SamProcessor

        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
        self.torch = torch
        self.device = device
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(device).eval()
        self._rgb = None
        print(f"[hf] {model_id} on {device}")

    def set_image(self, rgb: np.ndarray) -> None:
        self._rgb = rgb

    def predict(self, pts: np.ndarray, labs: np.ndarray):
        inputs = self.processor(
            self._rgb,
            input_points=[[[[float(x), float(y)] for x, y in pts]]],
            input_labels=[[[int(v) for v in labs]]],
            return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs, multimask_output=True)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())[0][0].numpy()
        scores = out.iou_scores.cpu().numpy().ravel()
        best = int(scores.argmax())
        return masks[best].astype(bool), float(scores[best])


# ==========================================================================
def write_outputs(out: Path, fid: str, scene: str, label: str,
                  img_bgr: np.ndarray, mask: np.ndarray,
                  full_mask: bool, white_on_black: bool) -> dict:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {}
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)

    crop = img_bgr[y:y + h, x:x + w]
    alpha = mask[y:y + h, x:x + w].astype(np.uint8) * 255
    rgba = np.dstack([crop, alpha])
    name = f"{fid}_{label}.png"

    fd = out / "by_frame" / fid
    fd.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(fd / name), rgba)

    ld = out / "by_letter" / label
    ld.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ld / name), rgba)

    mask_rel = ""
    if full_mask:
        m = mask.astype(np.uint8) * 255
        if not white_on_black:
            m = 255 - m          # glyph black on white, like a cast shadow
        mp = fd / f"{fid}_{label}_mask.png"
        cv2.imwrite(str(mp), m)
        mask_rel = str(mp.relative_to(out))

    return dict(frame_id=fid, scene=scene, letter=label,
                x=x, y=y, w=w, h=h, area=int(mask.sum()),
                cutout=str((fd / name).relative_to(out)), mask=mask_rel)


def write_overlay(out: Path, fid: str, img_bgr: np.ndarray, items) -> None:
    vis = img_bgr.copy()
    tint = np.zeros_like(vis)
    pal = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0),
           (0, 128, 255), (255, 255, 0), (128, 0, 255), (0, 200, 128)]
    for i, (label, mask, pts, labs, score) in enumerate(items):
        tint[mask] = pal[i % len(pal)]
    vis = cv2.addWeighted(vis, 0.55, tint, 0.45, 0)
    for i, (label, mask, pts, labs, score) in enumerate(items):
        c = pal[i % len(pal)]
        ys, xs = np.nonzero(mask)
        if len(xs):
            cv2.rectangle(vis, (int(xs.min()), int(ys.min())),
                          (int(xs.max()), int(ys.max())), c, 3)
            tag = f"{label} {score:.2f}"
            cv2.putText(vis, tag, (int(xs.min()) + 6, max(int(ys.min()) - 12, 32)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 7)
            cv2.putText(vis, tag, (int(xs.min()) + 6, max(int(ys.min()) - 12, 32)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, c, 3)
        for (px, py), pl in zip(pts, labs):
            cv2.drawMarker(vis, (int(px), int(py)),
                           (255, 255, 255) if pl else (0, 0, 255),
                           cv2.MARKER_CROSS if pl else cv2.MARKER_TILTED_CROSS,
                           26, 4)
    d = out / "overlay"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{fid}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="SAM masks from clicked points.")
    ap.add_argument("--keypoints", default="keypoints.json")
    ap.add_argument("--out", default="letters")
    ap.add_argument("--backend", choices=["sam2", "hf"], default="sam2")
    ap.add_argument("--sam2-checkpoint", default="checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--hf-model", default="facebook/sam-vit-huge")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu")
    ap.add_argument("--frames", nargs="*", help="only these frame ids")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="skip a mask whose predicted IoU is below this")
    ap.add_argument("--no-full-mask", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--mask-white-on-black", action="store_true",
                    help="invert mask polarity; default is glyph black on white")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the JSON and exit without loading a model")
    args = ap.parse_args()

    kp_path = Path(args.keypoints)
    if not kp_path.exists():
        sys.exit(f"{kp_path} not found — run 03_label_keypoints.py first.")
    data = json.loads(kp_path.read_text())
    frames = data.get("frames", {})

    todo = []
    problems = []
    for fid, f in sorted(frames.items()):
        if args.frames and fid not in args.frames:
            continue
        objs = {k: v for k, v in f.get("objects", {}).items() if v.get("points")}
        if not objs:
            continue
        p = Path(f["file"])
        if not p.exists():
            problems.append(f"{fid}: image missing at {p}")
            continue
        for lab, o in objs.items():
            if not any(o["labels"]):
                problems.append(f"{fid}/{lab}: only negative points, "
                                f"SAM needs at least one positive")
        todo.append((fid, f.get("scene", ""), p, objs))

    n_obj = sum(len(o) for _, _, _, o in todo)
    n_pts = sum(len(v["points"]) for _, _, _, o in todo for v in o.values())
    print(f"{len(todo)} frame(s), {n_obj} object(s), {n_pts} point(s)")
    labs_seen = sorted({l for _, _, _, o in todo for l in o})
    print(f"labels: {', '.join(labs_seen) or '—'}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for m in problems[:20]:
            print("  " + m)
    if args.dry_run:
        print("\ndry run — no model loaded, nothing written")
        return
    if not todo:
        sys.exit("nothing to segment")

    if args.backend == "sam2":
        try:
            backend = Sam2Backend(args.sam2_checkpoint, args.sam2_config,
                                  args.device)
        except ImportError as e:
            sys.exit(f"SAM2 not importable ({e}).\n"
                     "  pip install 'git+https://github.com/facebookresearch/sam2.git'\n"
                     "or use --backend hf --hf-model facebook/sam-vit-huge")
    else:
        try:
            backend = HfSamBackend(args.hf_model, args.device)
        except ImportError as e:
            sys.exit(f"transformers not importable ({e}).\n"
                     "  pip install transformers torch")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []

    for n, (fid, scene, path, objs) in enumerate(todo, 1):
        img = cv2.imread(str(path))
        if img is None:
            skipped.append(f"{fid}: unreadable"); continue
        backend.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        items = []
        for label, o in sorted(objs.items()):
            pts = np.array(o["points"], dtype=np.float32)
            labs = np.array(o["labels"], dtype=np.int32)
            if not labs.any():
                skipped.append(f"{fid}/{label}: no positive point"); continue
            mask, score = backend.predict(pts, labs)
            if score < args.min_score:
                skipped.append(f"{fid}/{label}: score {score:.2f} below cutoff")
                continue
            row = write_outputs(out, fid, scene, label, img, mask,
                                not args.no_full_mask,
                                args.mask_white_on_black)
            if row:
                row["score"] = round(score, 4)
                rows.append(row)
                items.append((label, mask, o["points"], o["labels"], score))

        if items and not args.no_overlay:
            write_overlay(out, fid, img, items)
        if n % 10 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}")

    if rows:
        cols = ["frame_id", "scene", "letter", "x", "y", "w", "h", "area",
                "score", "cutout", "mask"]
        with open(out / "letters_manifest.csv", "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=cols)
            wtr.writeheader()
            for r in rows:
                wtr.writerow({c: r.get(c, "") for c in cols})

    print(f"\n{len(rows)} mask(s) from {len(todo)} frame(s)")
    print(f"  by_frame/ by_letter/ overlay/  +  letters_manifest.csv")
    if rows:
        s = [r["score"] for r in rows]
        print(f"  score: min {min(s):.2f}  median {sorted(s)[len(s)//2]:.2f}"
              f"  max {max(s):.2f}   (check the low ones in overlay/)")
    if skipped:
        print(f"\n{len(skipped)} skipped:")
        for m in skipped[:20]:
            print("  " + m)


if __name__ == "__main__":
    main()
