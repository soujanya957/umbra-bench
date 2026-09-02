#!/usr/bin/env python3
"""
02_segment_letters.py — cut the letter shapes out of selected frames.

The letters in this spot are high-saturation yellow/gold on a desaturated
warm background, so an HSV threshold isolates them cleanly with no GPU.
SAM2 refinement is available as an optional second pass for the hard frames
(letters morphing into figures, letters overlapping the city background).

Typical use
-----------
    # everything in frames/
    python3 02_segment_letters.py --frames-dir frames --all

    # just the frames you picked off the contact sheets
    python3 02_segment_letters.py --frames f0092 f0117 f0161

    # from a text file, one frame id per line
    python3 02_segment_letters.py --frames-file selected.txt

    # per-frame letter labels (frames don't all show the whole word)
    python3 02_segment_letters.py --all --words words.csv

    # optional GPU refinement
    python3 02_segment_letters.py --all --sam2 \
        --sam2-checkpoint checkpoints/sam2.1_hiera_large.pt \
        --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml

Outputs (under --out, default `letters/`)
-----------------------------------------
    by_frame/f0092/f0092_F.png     RGBA cutout, tight crop, transparent bg
    by_frame/f0092/f0092_F_mask.png  binary mask, full 1920x1080 canvas
    by_letter/F/f0092_F.png        same cutouts regrouped by letter
    overlay/f0092.jpg              QC image: boxes + labels, eyeball this
    letters_manifest.csv           frame, letter, bbox, area, status

`by_letter/` is the layout to feed your per-letter optimization: one folder
per glyph, every frame's instance of that glyph inside it. `bbox` in the
manifest is what puts each letter back on the 1920x1080 canvas afterwards.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Defaults measured off this video's actual pixels (see README).
# Letters sit around H 15-40, S 195+, V 150+; background warm beige sits
# near S 130. S>=165 splits them with margin on both sides.
# --------------------------------------------------------------------------
DEFAULTS = dict(
    # Hue window (OpenCV 0-179), warm yellow/orange. h_lo=18 is the important
    # one: the room set has a reddish-brown window frame and door outline that
    # sit at H 10-12 and otherwise come through as extra components. Measured
    # over 405 components, glyphs land at H 20-32 and those artifacts at
    # H 10-12, with nothing in between — 18 sits in the gap.
    h_lo=18, h_hi=45,
    s_lo=165,               # saturation floor — the main discriminator
    v_lo=140,               # value floor, drops dark outlines/shadow
    warm_tol=25,            # keep px where R >= G - warm_tol. Kills the green
                            # park trees, which otherwise survive the hue
                            # window. 25 measured as the point that drops
                            # every tree while leaving glyphs untouched;
                            # at 0 it starts eating holes in the letters.
    # px; below this a blob is noise. Do NOT push this higher without checking
    # the last scene: there the whole word is laid out small and its smallest
    # glyph is only ~3.8k px, while the specks to kill top out around 3.1k.
    # Area alone can't separate glyphs from artifacts — hue does that.
    min_area=3200,
    merge_area=600,         # px; blobs this small get merged into a neighbour
    # Reject a blob whose bounding box covers more than this fraction of the
    # frame. On cross-dissolves between clips the whole washed-out set passes
    # the colour test and comes through as one blob spanning 63-70% of frame.
    # The widest real glyph here is the F whose bar runs over the whole word,
    # at 37%, so 0.45 sits clear of both.
    max_bbox_frac=0.45,
    # A blob nested inside another's bbox and smaller than this fraction of it
    # is a broken-off piece of that glyph, not a separate one.
    contain_ratio=0.25,
    close_px=5,             # morphological close, seals stroke gaps
    open_px=3,              # morphological open, kills speckle
    top_crop=0.02,          # ignore top 2% (channel logo)
    bottom_crop=0.14,       # ignore bottom 14% (burned-in subtitles)
)

# Strict -> loose (saturation, value). The late frames fade the word almost
# into the background and need roughly s=110 before anything survives.
RELAX_LADDER = [(130, 125), (110, 110), (95, 100)]


# ==========================================================================
# Blob extraction
# ==========================================================================
@dataclass
class Blob:
    """One connected letter-ish region."""
    mask: np.ndarray                     # full-canvas bool
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float
    label: str = ""
    refined: bool = False
    merged_from: list = field(default_factory=list)

    @property
    def bbox(self):
        return (self.x, self.y, self.w, self.h)

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h


def build_color_mask(img_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """HSV threshold + morphology -> uint8 {0,255} letter mask."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    keep = (
        (h >= cfg["h_lo"]) & (h <= cfg["h_hi"])
        & (s >= cfg["s_lo"])
        & (v >= cfg["v_lo"])
    )

    # Warm-channel test: yellow/gold glyphs have R >= G, foliage has G > R.
    # The hue window alone lets the park trees through; this drops them.
    if cfg["warm_tol"] is not None and cfg["warm_tol"] >= 0:
        bgr = img_bgr.astype(np.int16)
        keep &= bgr[..., 2] >= bgr[..., 1] - cfg["warm_tol"]

    mask = keep.astype(np.uint8) * 255

    # Zero out the bands where the subtitles live.
    H, W = mask.shape
    t = int(H * cfg["top_crop"])
    b = int(H * (1.0 - cfg["bottom_crop"]))
    mask[:t] = 0
    mask[b:] = 0

    # The channel bug sits at x 0.86-0.91, y 0.08-0.13 — the same vertical
    # band as the tops of the letters, so a horizontal crop can't remove it
    # without clipping glyphs. Knock out the corner box instead. Checked
    # over 60 random frames: letters never enter it.
    for (x1, y1, x2, y2) in cfg.get("exclude", ()):
        mask[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)] = 0

    if cfg["open_px"] > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["open_px"],) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if cfg["close_px"] > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["close_px"],) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def extract_blobs(mask: np.ndarray, cfg: dict) -> list[Blob]:
    """Connected components -> Blobs, with small fragments merged in.

    Merging matters because of accent marks: the dot on an 'i' and the
    crossbar fragments come out as separate components. Anything between
    merge_area and min_area is folded into the nearest big blob whose
    horizontal span it overlaps (that's the letter it belongs to).
    """
    n, lab, stats, cents = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )

    frame_px = mask.shape[0] * mask.shape[1]
    cap = cfg.get("max_bbox_frac", 1.0)

    big, small = [], []
    for i in range(1, n):  # 0 is background
        x, y, w, h, a = stats[i]
        if a < cfg["merge_area"]:
            continue
        if frame_px and (w * h) / frame_px > cap:
            continue  # dissolve wash, not a glyph
        blob = Blob(mask=(lab == i), x=int(x), y=int(y), w=int(w), h=int(h),
                    area=int(a), cx=float(cents[i][0]), cy=float(cents[i][1]))
        (big if a >= cfg["min_area"] else small).append(blob)

    # Fold each small fragment into the best big blob.
    for frag in small:
        best, best_score = None, None
        for b in big:
            # horizontal overlap of the fragment with the candidate letter
            ox = min(frag.x2, b.x2) - max(frag.x, b.x)
            # vertical gap between them
            gap = max(b.y - frag.y2, frag.y - b.y2, 0)
            if ox <= 0:
                # no overlap: fall back to horizontal distance, penalised
                ox = -min(abs(frag.cx - b.cx), 1e6) * 0.25
            score = ox - gap * 0.5
            if best_score is None or score > best_score:
                best, best_score = b, score
        if best is not None and best_score is not None and best_score > -200:
            best.mask |= frag.mask
            best.merged_from.append(frag.bbox)
            xs = min(best.x, frag.x)
            ys = min(best.y, frag.y)
            best.w = max(best.x2, frag.x2) - xs
            best.h = max(best.y2, frag.y2) - ys
            best.x, best.y = xs, ys
            best.area += frag.area

    # Fold any blob lying entirely inside another's bounding box into it.
    # When a glyph morphs it can break into pieces that are too big to be
    # caught by merge_area — the F's lower stroke splits off at ~3.7k px,
    # just over min_area. Area thresholds can't separate that from a real
    # small glyph (scene_06's F is 3.8k), but containment can: the letters
    # here stand side by side, so one never nests inside another.
    ratio = cfg.get("contain_ratio", 0.25)
    big.sort(key=lambda b: -b.area)
    absorbed = set()
    for i, outer in enumerate(big):
        if i in absorbed:
            continue
        for j in range(i + 1, len(big)):
            if j in absorbed:
                continue
            inner = big[j]
            if (inner.x >= outer.x and inner.y >= outer.y
                    and inner.x2 <= outer.x2 and inner.y2 <= outer.y2
                    and inner.area < outer.area * ratio):
                outer.mask |= inner.mask      # bbox unchanged: inner is inside
                outer.area += inner.area
                outer.merged_from.append(inner.bbox)
                absorbed.add(j)
    big = [b for i, b in enumerate(big) if i not in absorbed]

    # Order left-to-right by the LEFT EDGE, not the centroid. In this spot the
    # 'F' carries a long bar over the whole word, so its centroid lands in the
    # middle of the frame while its true reading order is first.
    big.sort(key=lambda b: b.x)
    return big


def segment_frame(img_bgr: np.ndarray, cfg: dict, expected_n: int | None,
                  relax: bool) -> tuple[list[Blob], tuple[int, int]]:
    """Threshold + blob extraction, relaxing saturation only when needed.

    Strictest-that-works wins. We walk from the strict default down the
    ladder and stop at the first rung that finds anything, because a looser
    threshold pulls in background. The one exception: if we know how many
    glyphs to expect and a later rung hits that count exactly, that rung is
    better evidence than a strict rung that came up short.
    """
    rungs = [(cfg["s_lo"], cfg["v_lo"])]
    if relax:
        rungs += [r for r in RELAX_LADDER if r[0] < cfg["s_lo"]]

    first_hit = None
    for s_lo, v_lo in rungs:
        c = dict(cfg, s_lo=s_lo, v_lo=v_lo)
        blobs = extract_blobs(build_color_mask(img_bgr, c), c)
        if expected_n and len(blobs) == expected_n:
            return blobs, (s_lo, v_lo)
        if blobs and first_hit is None:
            first_hit = (blobs, (s_lo, v_lo))
            if not expected_n:
                return first_hit
    if first_hit is not None:
        return first_hit
    return [], (cfg["s_lo"], cfg["v_lo"])


# ==========================================================================
# Optional SAM2 refinement
# ==========================================================================
class Sam2Refiner:
    """Wraps SAM2 image prediction, prompting with the HSV blob boxes.

    Only worth turning on for frames where the letters overlap busy
    background or have morphed into figures. On the clean studio frames the
    HSV mask is already tighter than SAM will give you.
    """

    def __init__(self, checkpoint: str, config: str, device: str | None = None):
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise SystemExit(
                f"SAM2 not importable ({e}).\n"
                "  pip install 'git+https://github.com/facebookresearch/sam2.git'\n"
                "and download a checkpoint from that repo. Or drop --sam2 and use "
                "the HSV pass, which handles the clean frames fine."
            )
        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        model = build_sam2(config, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(model)
        print(f"[sam2] loaded on {device}")

    def refine(self, img_bgr: np.ndarray, blobs: list[Blob]) -> list[Blob]:
        if not blobs:
            return blobs
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        boxes = np.array([[b.x, b.y, b.x2, b.y2] for b in blobs], dtype=np.float32)
        masks, scores, _ = self.predictor.predict(
            box=boxes, multimask_output=False
        )
        masks = np.asarray(masks)
        if masks.ndim == 4:      # (N,1,H,W)
            masks = masks[:, 0]
        elif masks.ndim == 3 and len(blobs) == 1:
            masks = masks[:1]

        for blob, m, sc in zip(blobs, masks, np.atleast_1d(scores).ravel()):
            m = m.astype(bool)
            # Guard against SAM latching onto the background: keep its mask
            # only if it still covers most of the colour blob and hasn't
            # ballooned. Otherwise keep the HSV mask.
            inter = np.logical_and(m, blob.mask).sum()
            if blob.area == 0:
                continue
            covers = inter / blob.area
            growth = m.sum() / max(blob.area, 1)
            if covers > 0.6 and growth < 3.0:
                blob.mask = m
                ys, xs = np.nonzero(m)
                if len(xs):
                    blob.x, blob.y = int(xs.min()), int(ys.min())
                    blob.w = int(xs.max() - xs.min() + 1)
                    blob.h = int(ys.max() - ys.min() + 1)
                    blob.area = int(m.sum())
                    blob.refined = True
        return blobs


# ==========================================================================
# Labelling
# ==========================================================================
def assign_labels(blobs: list[Blob], word: str | None) -> str:
    """Name blobs left-to-right from `word`. Returns a status string.

    On a count mismatch nothing is dropped — blobs fall back to positional
    names (c00, c01, ...) and the frame is flagged so you can fix it in
    words.csv and re-run just that frame.
    """
    if not blobs:
        return "empty"
    if not word:
        for i, b in enumerate(blobs):
            b.label = f"c{i:02d}"
        return "unlabelled"

    letters = [c for c in word if not c.isspace()]
    if len(letters) != len(blobs):
        for i, b in enumerate(blobs):
            b.label = f"c{i:02d}"
        return f"mismatch(word={len(letters)},found={len(blobs)})"

    seen: dict[str, int] = {}
    for b, ch in zip(blobs, letters):
        seen[ch] = seen.get(ch, 0) + 1
        # disambiguate repeats within one frame: A, A2, A3...
        b.label = ch if seen[ch] == 1 else f"{ch}{seen[ch]}"
    return "ok"


# ==========================================================================
# Writing
# ==========================================================================
def write_blob(out: Path, frame_id: str, img_bgr: np.ndarray, blob: Blob,
               write_full_mask: bool,
               mask_white_on_black: bool = False) -> tuple[Path, Path | None]:
    """Tight RGBA cutout into by_frame/ and by_letter/; optional full mask."""
    x, y, w, h = blob.bbox
    crop = img_bgr[y:y + h, x:x + w]
    alpha = (blob.mask[y:y + h, x:x + w].astype(np.uint8)) * 255
    rgba = np.dstack([crop, alpha])

    name = f"{frame_id}_{blob.label}.png"

    frame_dir = out / "by_frame" / frame_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    cutout_path = frame_dir / name
    cv2.imwrite(str(cutout_path), rgba)

    letter_dir = out / "by_letter" / blob.label
    letter_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(letter_dir / name), rgba)

    mask_path = None
    if write_full_mask:
        mask_path = frame_dir / f"{frame_id}_{blob.label}_mask.png"
        m = blob.mask.astype(np.uint8) * 255
        if not mask_white_on_black:
            # Glyph black on white — a shadow is dark on a lit wall, so this
            # is the target image the optimizer wants, not its negative.
            m = 255 - m
        cv2.imwrite(str(mask_path), m)

    return cutout_path, mask_path


def write_overlay(out: Path, frame_id: str, img_bgr: np.ndarray,
                  blobs: list[Blob], status: str) -> None:
    vis = img_bgr.copy()
    tint = np.zeros_like(vis)
    palette = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0),
               (0, 128, 255), (255, 255, 0), (128, 0, 255), (0, 200, 128)]
    for i, b in enumerate(blobs):
        col = palette[i % len(palette)]
        tint[b.mask] = col
    vis = cv2.addWeighted(vis, 0.55, tint, 0.45, 0)
    for i, b in enumerate(blobs):
        col = palette[i % len(palette)]
        cv2.rectangle(vis, (b.x, b.y), (b.x2, b.y2), col, 3)
        tag = b.label + ("*" if b.refined else "")
        cv2.putText(vis, tag, (b.x + 6, max(b.y - 12, 32)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 7)
        cv2.putText(vis, tag, (b.x + 6, max(b.y - 12, 32)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
    banner = f"{frame_id}  n={len(blobs)}  {status}"
    cv2.putText(vis, banner, (24, vis.shape[0] - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
    cv2.putText(vis, banner, (24, vis.shape[0] - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    d = out / "overlay"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{frame_id}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ==========================================================================
# Frame selection
# ==========================================================================
def resolve_frames(args) -> list[Path]:
    """Collect frames from one or more directories.

    Taking every scene in a single run matters: the run writes one manifest
    and one shared by_letter/ tree, so cutouts from every scene land in the
    same per-glyph folders. Six separate runs would each rewrite the
    manifest, leaving only the last scene's rows.
    """
    dirs = [Path(d) for d in args.frames_dir]
    missing = [d for d in dirs if not d.is_dir()]
    if missing:
        sys.exit("frames dir not found: " + ", ".join(str(m) for m in missing))

    everything, seen = [], set()
    for d in dirs:
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            if p.stem in seen:
                sys.exit(f"duplicate frame id {p.stem} across --frames-dir "
                         f"({p}). Frame ids must be unique.")
            seen.add(p.stem)
            everything.append(p)
    everything.sort(key=lambda p: p.stem)
    if args.all:
        return everything

    wanted: list[str] = list(args.frames or [])
    if args.frames_file:
        for line in Path(args.frames_file).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                wanted.append(line)
    if not wanted:
        sys.exit("Nothing selected. Pass --all, --frames f0092 ..., "
                 "or --frames-file selected.txt")

    by_stem = {p.stem: p for p in everything}
    out, absent = [], []
    for wid in wanted:
        stem = Path(wid).stem
        if stem in by_stem:
            out.append(by_stem[stem])
        else:
            absent.append(wid)
    if absent:
        print(f"[warn] not found: {', '.join(absent)}")
    return out


def load_words(path: str | None) -> dict[str, str]:
    """words.csv -> {frame_id: word}. Header optional; 2 columns."""
    if not path:
        return {}
    words: dict[str, str] = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            fid, word = row[0].strip(), row[1].strip()
            if fid.lower() in ("frame_id", "frame"):
                continue
            words[Path(fid).stem] = word
    return words


# ==========================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description="Segment letter components out of demo frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--frames-dir", nargs="+", default=["frames"],
                   metavar="DIR",
                   help="one or more frame folders, e.g. scenes/scene_*")
    p.add_argument("--frames", nargs="*", help="frame ids, e.g. f0092 f0117")
    p.add_argument("--frames-file", help="text file, one frame id per line")
    p.add_argument("--all", action="store_true", help="every frame in --frames-dir")
    p.add_argument("--out", default="letters")
    p.add_argument("--word", default="FAMILY",
                   help="default left-to-right labels; '' to number instead")
    p.add_argument("--words", help="words.csv with per-frame overrides")
    p.add_argument("--scene-words", nargs="*", default=[], metavar="DIR=WORD",
                   help="per-folder labels, e.g. scene_01=I scene_06=FAMILY. "
                        "Each scene here shows only part of the word, so this "
                        "is usually what you want instead of --word.")
    p.add_argument("--no-full-mask", action="store_true",
                   help="skip the full-canvas mask PNGs (saves disk)")
    p.add_argument("--mask-white-on-black", action="store_true",
                   help="invert the mask polarity: glyph white on black. "
                        "Default is glyph BLACK on WHITE, matching how a "
                        "shadow reads on a lit wall.")
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--no-auto-relax", action="store_true",
                   help="don't lower the saturation floor on empty frames")
    p.add_argument("--exclude", nargs="*", default=["0.85,0.0,1.0,0.16"],
                   metavar="x1,y1,x2,y2",
                   help="normalised boxes to blank out; default is the "
                        "top-right channel logo. Pass --exclude to disable.")

    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)

    p.add_argument("--sam2", action="store_true", help="SAM2 refinement pass")
    p.add_argument("--sam2-checkpoint", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--device", default=None, help="cuda / mps / cpu")

    args = p.parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULTS}

    zones = []
    for spec in (args.exclude or []):
        try:
            x1, y1, x2, y2 = (float(v) for v in spec.split(","))
        except ValueError:
            sys.exit(f"--exclude wants x1,y1,x2,y2 normalised 0-1, got {spec!r}")
        zones.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    cfg["exclude"] = zones

    frames = resolve_frames(args)
    if not frames:
        sys.exit("No frames to process.")
    words = load_words(args.words)

    scene_words = {}
    for spec in args.scene_words:
        if "=" not in spec:
            sys.exit(f"--scene-words wants DIR=WORD, got {spec!r}")
        k, _, v = spec.partition("=")
        scene_words[Path(k.strip()).name] = v.strip()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    refiner = None
    if args.sam2:
        refiner = Sam2Refiner(args.sam2_checkpoint, args.sam2_config, args.device)

    rows, flagged = [], []
    print(f"segmenting {len(frames)} frame(s) -> {out}/")

    for n, path in enumerate(frames, 1):
        frame_id = path.stem
        img = cv2.imread(str(path))
        if img is None:
            print(f"[warn] unreadable: {path}")
            continue

        # per-frame CSV wins, then per-scene folder, then the global default
        word = words.get(frame_id, scene_words.get(path.parent.name, args.word))
        expected = len([c for c in word if not c.isspace()]) if word else None

        blobs, (used_s, used_v) = segment_frame(
            img, cfg, expected, relax=not args.no_auto_relax
        )
        if refiner is not None:
            blobs = refiner.refine(img, blobs)
            blobs.sort(key=lambda b: b.x)

        status = assign_labels(blobs, word)
        if status not in ("ok",):
            flagged.append((frame_id, status, len(blobs)))

        for b in blobs:
            cut, mp = write_blob(out, frame_id, img, b,
                                 not args.no_full_mask,
                                 args.mask_white_on_black)
            rows.append(dict(
                frame_id=frame_id, scene=path.parent.name,
                letter=b.label, status=status,
                x=b.x, y=b.y, w=b.w, h=b.h, area=b.area,
                cx=round(b.cx, 1), cy=round(b.cy, 1),
                refined=int(b.refined),
                s_lo=used_s, v_lo=used_v,
                cutout=str(cut.relative_to(out)),
                mask=str(mp.relative_to(out)) if mp else "",
            ))

        if not args.no_overlay:
            write_overlay(out, frame_id, img, blobs, status)

        if n % 25 == 0 or n == len(frames):
            print(f"  {n}/{len(frames)}")

    man = out / "letters_manifest.csv"
    if rows:
        with open(man, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)

    print(f"\n{len(rows)} letter cutouts from {len(frames)} frames")
    print(f"  by_frame/   per-frame folders")
    print(f"  by_letter/  regrouped per glyph  <- feed this to optimization")
    print(f"  overlay/    QC images, check these first")
    print(f"  {man}")

    if flagged:
        print(f"\n{len(flagged)} frame(s) need a label fix "
              f"(component count != word length):")
        for fid, st, k in flagged[:20]:
            print(f"  {fid:>8}  {st}")
        if len(flagged) > 20:
            print(f"  ... and {len(flagged)-20} more")
        print("Put the right word for those in words.csv (frame_id,word) "
              "and re-run with --words words.csv")


if __name__ == "__main__":
    main()
