#!/usr/bin/env python3
"""Evaluate letter and digit masks with CLIP retrieval.

Example:
    python tests/clip_eval.py

The default input is Teleops/masks and results are written to
results/clip_eval.  The candidate order starts with uppercase A-Z, lowercase a-z,
then 0-9, but applies the filtering and equivalent-label rules below before CLIP
runs.  The generated CSV records each image's original and scoring labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from semantic_metrics import clip_retrieval


DEFAULT_INPUT_DIR = ROOT / "Teleops" / "masks"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "clip_eval"
# The evaluation intentionally excludes low-confidence lowercase *classes*, while
# retaining their captures: c/k/m/p/s/v/w/z are scored as C/K/M/P/S/V/W/Z.
# Lowercase n is an independent class and remains distinct from uppercase N.
# It also collapses visual equivalents so I/l/1 is scored as 1, O/o/0 as 0, and
# q/9 as q. Keep these rules here: they define the benchmark's class set.
SOURCE_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
REMOVED_LABELS = frozenset("ckmopsvwz")
LOWERCASE_TO_UPPERCASE = {
    "c": "C", "k": "K", "m": "M", "p": "P",
    "s": "S", "v": "V", "w": "W", "z": "Z",
}
LABEL_ALIASES = {
    **LOWERCASE_TO_UPPERCASE,
    "I": "1", "l": "1", "1": "1",
    "O": "0", "o": "0", "0": "0",
    "q": "q", "9": "q",
}
ALL_LABELS = [
    label for label in SOURCE_LABELS
    if label not in REMOVED_LABELS
    and LABEL_ALIASES.get(label, label) == label
]
_GLYPH_CAPTURE = re.compile(
    r"^(?:letters_upper([A-Z])|letters_lower([a-z])|digits([0-9]))_.*_mask\.png$"
)


def label_from_filename(path: Path) -> str:
    """Return the true glyph label encoded in a rectified capture filename."""
    match = _GLYPH_CAPTURE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Filename does not encode a letter or digit label: {path.name}")
    return next(label for label in match.groups() if label is not None)


def canonicalize_label(label: str) -> str | None:
    """Return the scoring class for a source label, or None when it is excluded."""
    canonical = LABEL_ALIASES.get(label, label)
    if canonical in REMOVED_LABELS:
        return None
    return canonical if canonical in ALL_LABELS else None


def glyph_prompt(label: str) -> str:
    """Make case explicit so CLIP can distinguish uppercase from lowercase."""
    if label.isupper():
        return f"a shadow of the uppercase letter {label}"
    if label.islower():
        return f"a shadow of the lowercase letter {label}"
    return f"a shadow of the digit {label}"


def label_index(label: str) -> int:
    """Return the filtered class index for a glyph label."""
    canonical = canonicalize_label(label)
    if canonical is None:
        raise ValueError(f"Excluded glyph label: {label}")
    try:
        return ALL_LABELS.index(canonical)
    except ValueError as exc:
        raise ValueError(f"Unsupported glyph label: {label}") from exc


def find_mask_images(input_dir: Path) -> list[Path]:
    """Find valid glyph masks, excluding other masks such as objects."""
    return sorted(
        path for path in input_dir.glob("*_mask.png")
        if (_GLYPH_CAPTURE.fullmatch(path.name) is not None
            and canonicalize_label(label_from_filename(path)) is not None)
    )


def evaluate(input_dir: Path, output_dir: Path, model_name: str,
             pretrained: str, batch_size: int, top_k: int = 10) -> dict:
    """Run CLIP retrieval and persist per-image and aggregate results."""
    image_paths = find_mask_images(input_dir)
    if not image_paths:
        raise ValueError(
            f"No letter/digit files ending in _mask.png found in {input_dir}"
        )

    source_labels = [label_from_filename(path) for path in image_paths]
    labels = [canonicalize_label(label) for label in source_labels]
    true_idx = [label_index(label) for label in source_labels]
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    result = clip_retrieval(
        images=images,
        class_names=ALL_LABELS,
        true_idx=true_idx,
        model_name=model_name,
        pretrained=pretrained,
        prompt_tmpl=glyph_prompt,
        batch_size=batch_size,
        top_k=top_k,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "clip_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=(
                "path", "filename", "source_label", "true_label", "true_idx", "rank", "top1",
                "clip_top1_label", "clip_top1_similarity", "clip_top_predictions",
            )
        )
        writer.writeheader()
        for path, source_label, label, index, rank, predictions in zip(
                image_paths, source_labels, labels, true_idx, result["rank"], result["top_predictions"]):
            writer.writerow({
                "path": str(path),
                "filename": path.name,
                "source_label": source_label,
                "true_label": label,
                "true_idx": index,
                "rank": rank,
                "top1": int(rank == 1),
                "clip_top1_label": predictions[0]["label"],
                "clip_top1_similarity": predictions[0]["similarity"],
                "clip_top_predictions": json.dumps({
                    position: [prediction["label"], prediction["similarity"]]
                    for position, prediction in enumerate(predictions)
                }),
            })

    summary = {
        "input_dir": str(input_dir),
        "n_images": len(image_paths),
        "class_names": ALL_LABELS,
        "model_name": model_name,
        "pretrained": pretrained,
        "prompt": {
            "uppercase": "a shadow of the uppercase letter {label}",
            "lowercase": "a shadow of the lowercase letter {label}",
            "digit": "a shadow of the digit {label}",
        },
        **result,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of highest-scoring labels to save per image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(**vars(args))
    print(json.dumps({
        "n_images": summary["n_images"],
        "top1": summary["top1"],
        "top5": summary["top5"],
        "mrr": summary["mrr"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
