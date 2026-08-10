"""Curate MPEG-7 CE-Shape-1 silhouettes into typed target subsets.

MPEG-7 is NOT vendored in this repo. Download it first (see external/README.md):
    external/MPEG7/  containing  <class>-<idx>.gif  files (e.g. bird-3.gif)

Then:  python scripts/curate_mpeg7.py

Fills four subsets (see SUBSETS below):
    animals/   — semantic, organic outlines; overlaps HaSPeR classes for human-baseline comparison
    objects/   — semantic, man-made; handles/holes/thin protrusions (fork, key, cup)
    vehicles/  — semantic, boxy + wheels; distinct attribute profile from organic shapes
    abstract/  — NO semantic prior (device0-9, heart, comma...) — isolates pure geometric
                 matching from recognizability; the control group for perceptual metrics

Converts each pick to canonical format (512x512 binary PNG, white on black,
centered, 10% margin) and records provenance in targets/<subset>/_sources.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "external" / "MPEG7"
SIZE = 512
MARGIN = 0.10

# MPEG-7 class names as they appear in filenames (case varies by mirror; matching
# is case-insensitive). Animal picks overlap HaSPeR's hand-shadow classes where
# MPEG-7 allows (bird, chicken, cattle~cow, deer, dog, elephant, horse, sea_snake).
SUBSETS = {
    "animals": [
        "bird", "chicken", "cattle", "deer", "dog", "elephant", "horse",
        "sea_snake", "butterfly", "fish", "turtle", "octopus", "camel", "frog",
        "bat", "beetle", "fly", "flatfish", "lizzard", "lmfish", "rat", "ray",
    ],
    "objects": [
        "cup", "fork", "spoon", "hammer", "key", "pencil", "shoe", "hat",
        "bell", "guitar", "watch", "teddy", "bottle", "glas", "brick", "fountain",
        "apple", "cellular_phone", "crown", "horseshoe", "jar", "pocket", "tree",
    ],
    "vehicles": [
        "car", "personal_car", "carriage", "truck", "chopper", "classic",
    ],
    "figures": [  # human figures — very relevant for shadow theatre storyboards
        "children", "face",
    ],
    "abstract": [
        "device0", "device1", "device2", "device3", "device4",
        "device5", "device6", "device7", "device8", "device9",
        "heart", "comma", "hcircle", "spring", "bone", "misk", "stef",
    ],
}
# collect generously, filter later — bad/duplicate picks are cheap to delete,
# and build_metadata.py just follows whatever is on disk
PICK_PER_CLASS = 5  # first N indices (deterministic)


def to_canonical(img: Image.Image) -> Image.Image:
    m = np.asarray(img.convert("L"))
    m = (m > 127).astype(np.uint8)
    # polarity: the border is background (MPEG-7 shapes don't touch the frame).
    # (a global-mean heuristic misfires on shapes covering >half the image)
    border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    if border.mean() > 0.5:  # white border => white is background — invert
        m = 1 - m
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        raise ValueError("empty mask")
    crop = m[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] * 255
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


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found — download MPEG-7 first (see external/README.md)")
    # case-insensitive index of all source files, normalizing zero-padded
    # indices (some classes ship as car-01.gif, others as bird-1.gif)
    index: dict[str, Path] = {}
    for f in sorted(SRC.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".gif", ".png", ".bmp") and "-" in f.stem:
            name, _, num = f.stem.rpartition("-")
            if num.isdigit():
                index.setdefault(f"{name.lower()}-{int(num)}", f)

    total = 0
    for subset, classes in SUBSETS.items():
        out = ROOT / "targets" / subset
        out.mkdir(parents=True, exist_ok=True)
        sources: dict[str, str] = {}
        n = 0
        for cls in classes:
            picked = 0
            for idx in range(1, 21):
                if picked >= PICK_PER_CLASS:
                    break
                src_file = index.get(f"{cls.lower()}-{idx}")
                if src_file is None:
                    continue
                stem = f"{cls.replace('_', '-')}_mpeg7-{idx:02d}"
                to_canonical(Image.open(src_file)).save(out / f"{stem}.png")
                sources[stem] = f"mpeg7-ce-shape-1:{src_file.name}"
                picked += 1
                n += 1
            if picked == 0:
                print(f"warning: no files found for class '{cls}' ({subset})")
        (out / "_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
        print(f"{subset}: wrote {n} targets to {out}")
        total += n
    print(f"total: {total} MPEG-7 targets")


if __name__ == "__main__":
    main()
