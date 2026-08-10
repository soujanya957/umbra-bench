"""Curate a small exemplar subset of HaSPeR into targets/hand_shadow/.

HaSPeR (15,000 real hand-shadow photos, 15 animal classes) is NOT vendored in this
repo — it is large and its source clips are used under fair use, so we only keep a
few binarized exemplar silhouettes per class and reference the rest.

1. Download the dataset locally (see external/README.md):
       hf download Starscream-11813/HaSPeR --repo-type dataset --local-dir external/HaSPeR
2. Run:  python scripts/curate_hasper.py

For each class directory found, picks the first PICK_PER_CLASS images
(deterministic: sorted filenames), extracts the shadow region (Otsu threshold,
dark side = shadow, largest connected component), converts to canonical format
(512x512 white-on-black, centered, 10% margin), and records provenance in
targets/hand_shadow/_sources.json.

Review the outputs visually — real photos vary; delete bad masks and adjust
PICK_OFFSET or the per-class overrides to pick cleaner frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "external" / "HaSPeR"
OUT = ROOT / "targets" / "hand_shadow"
SIZE = 512
MARGIN = 0.10
PICK_PER_CLASS = 2
PICK_OFFSET = 0  # skip the first N candidates per class if the defaults are bad
IMG_EXTS = {".jpg", ".jpeg", ".png"}
# per-file extra trim (fractions: left, top, right, bottom) for images with
# dark panels/bars the automatic border trim misses — keyed by source stem
TRIM_OVERRIDES: dict[str, tuple[float, float, float, float]] = {
    "elephant_2": (0.15, 0.0, 0.0, 0.0),
}


def extract_shadow_mask(img: Image.Image, trim: tuple[float, float, float, float] = (0, 0, 0, 0)) -> np.ndarray:
    g = np.asarray(img.convert("L"))
    h, w = g.shape
    l, t, r, b = trim
    g = g[int(h * t) : h - int(h * b), int(w * l) : w - int(w * r)]
    # trim border (screenshots often have dark window frames)
    ty, tx = max(2, g.shape[0] // 33), max(2, g.shape[1] // 33)
    g = g[ty:-ty, tx:-tx]
    _, th = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = 1 - th  # shadow = dark side
    # opening cuts thin frame lines connecting the shadow to border junk
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    labels, n = ndimage.label(opened)
    if n == 0:
        raise ValueError("no shadow region found")
    # score components by size, discounted by distance from image center
    # (frame/border remnants are large but peripheral)
    h, w = opened.shape
    best, best_score = 0, -1.0
    for i in range(1, n + 1):
        comp = labels == i
        area = float(comp.sum())
        if area < 0.005 * h * w:
            continue
        ys, xs = np.nonzero(comp)
        d = np.hypot(ys.mean() / h - 0.5, xs.mean() / w - 0.5)  # 0 center, ~0.7 corner
        score = area * float(np.exp(-4.0 * d))
        if score > best_score:
            best, best_score = i, score
    comp = (labels == best).astype(np.uint8)
    # restore detail lost by opening: closing, then intersect with original dark mask
    comp = cv2.morphologyEx(comp, cv2.MORPH_DILATE, kernel) & mask
    return comp


def to_canonical(mask: np.ndarray) -> Image.Image:
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] * 255
    crop_img = Image.fromarray(crop)
    target = int(SIZE * (1 - 2 * MARGIN))
    scale = target / max(crop_img.size)
    crop_img = crop_img.resize(
        (max(1, round(crop_img.width * scale)), max(1, round(crop_img.height * scale))),
        Image.NEAREST,
    )
    canvas = Image.new("L", (SIZE, SIZE), 0)
    canvas.paste(crop_img, ((SIZE - crop_img.width) // 2, (SIZE - crop_img.height) // 2))
    # canonical: black shape on white background (shadow on screen), 1-bit
    return canvas.point(lambda p: 0 if p > 127 else 255).convert("1")


def gather_by_class(root: Path) -> dict[str, list[Path]]:
    """Group source images by class.

    Supports two layouts:
    - HF download: class directories containing images (train/swan/*.jpg)
    - flat files named after the class (swan.png, elephant_2.PNG, ...)
    """
    out: dict[str, list[Path]] = {}
    for d in sorted(root.rglob("*")):
        if d.is_dir():
            imgs = sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)
            if imgs and d.name.lower() not in out:  # first split wins (e.g. train/)
                out[d.name.lower()] = imgs
    if not out:  # flat layout
        for f in sorted(root.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                cls = f.stem.split("_")[0].lower()
                out.setdefault(cls, []).append(f)
    return out


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found — download HaSPeR first (see external/README.md)")
    OUT.mkdir(parents=True, exist_ok=True)
    sources: dict[str, str] = {}
    n = 0
    for cls, imgs in gather_by_class(SRC).items():
        picked = 0
        for f in imgs[PICK_OFFSET:]:
            if picked >= PICK_PER_CLASS:
                break
            stem = f"{cls}_hasper-{picked + 1:02d}"
            try:
                trim = TRIM_OVERRIDES.get(f.stem, (0, 0, 0, 0))
                to_canonical(extract_shadow_mask(Image.open(f), trim)).save(OUT / f"{stem}.png")
            except Exception as e:  # noqa: BLE001 — skip unreadable/degenerate frames
                print(f"skip {f.name}: {e}")
                continue
            sources[stem] = f"hasper:{f.relative_to(SRC)}"
            picked += 1
            n += 1
    (OUT / "_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(f"wrote {n} hand-shadow targets to {OUT} — review them visually!")


if __name__ == "__main__":
    main()
