#!/usr/bin/env python3
"""Evaluate letter and digit shadows with CLIP retrieval, against a target ceiling.

Example:
    python tests/clip_eval.py                 # one 49-way retrieval over the
                                              # benchmark's combined class set
    python tests/clip_eval.py --by-subset     # digits, lowercase, uppercase as
                                              # three independent retrievals

The default input is Teleops/masks and results are written to results/clip_eval.
What this does beyond a bare CLIP score (METRICS.md Part C):

  **Score the targets too.** The reference glyphs in `targets/` are run through the
  identical pipeline -- same class list, same prompts, same raw-mask input -- as a
  recognizability ceiling. An absolute shadow top-1 of 0.16 is uninterpretable
  alone: nobody knows what CLIP scores on the targets. The number to report is
  `recognizability_ratio` = shadow / target, written to `summary.json`.

Masks are fed to CLIP as-is; `render_shadow()` is deliberately not applied here.
Both conditions get the same treatment, so the ratio stays valid.

Two modes:

  * default -- the benchmark's combined 49-class set (uppercase A-Z minus I/O,
    16 lowercase, digits 0-8), with the ambiguous-lowercase and visual-equivalent
    collapses defined below. A digit shadow is ranked against letters too.
  * ``--by-subset`` -- one retrieval per subset (``digits``, ``letters_lower``,
    ``letters_upper``), each self-contained: only that subset's captures, only its
    targets, its own full single-case alphabet as the candidate set, and the
    matching prompt (``a shadow of the digit 7`` / ``... lowercase letter b`` /
    ``... uppercase letter B``). No collapses, no cross-subset routing.
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

from semantic_metrics import clip_retrieval, recognizability_ratio


DEFAULT_INPUT_DIR = ROOT / "Teleops" / "masks"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "clip_eval"
DEFAULT_TARGETS_ROOT = ROOT / "targets"
# The combined evaluation intentionally excludes low-confidence lowercase *classes*,
# while retaining their captures: c/k/m/p/s/v/w/z are scored as C/K/M/P/S/V/W/Z.
# Lowercase n is an independent class and remains distinct from uppercase N.
# It also collapses visual equivalents so I/l/1 is scored as 1, O/o/0 as 0, and
# q/9 as q. Keep these rules here: they define the benchmark's combined class set.
# --by-subset uses none of this -- see SUBSET_ALPHABETS.
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
_SUBSET_DIRS = ("letters_upper", "letters_lower", "digits")
# --by-subset: each subset scored against its own full alphabet, nothing collapsed.
SUBSET_ALPHABETS = {
    "digits": list("0123456789"),
    "letters_lower": list("abcdefghijklmnopqrstuvwxyz"),
    "letters_upper": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
}


def _capture_subset(path: Path) -> str | None:
    """The targets/ subdirectory a capture belongs to, or None if not a glyph mask."""
    match = _GLYPH_CAPTURE.fullmatch(path.name)
    if match is None:
        return None
    return _SUBSET_DIRS[next(i for i, g in enumerate(match.groups()) if g is not None)]


def label_from_filename(path: Path) -> str:
    """Return the true glyph label encoded in a rectified capture filename."""
    match = _GLYPH_CAPTURE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Filename does not encode a letter or digit label: {path.name}")
    return next(label for label in match.groups() if label is not None)


def glyph_subset(path: Path) -> str:
    """Return the targets/ subdirectory holding the reference glyph for a capture."""
    subset = _capture_subset(path)
    if subset is None:
        raise ValueError(f"Filename does not encode a letter or digit label: {path.name}")
    return subset


def canonicalize_label(label: str) -> str | None:
    """Return the combined-set scoring class for a source label, or None if excluded."""
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
    """Return the combined-set class index for a glyph label."""
    canonical = canonicalize_label(label)
    if canonical is None:
        raise ValueError(f"Excluded glyph label: {label}")
    try:
        return ALL_LABELS.index(canonical)
    except ValueError as exc:
        raise ValueError(f"Unsupported glyph label: {label}") from exc


def find_mask_images(input_dir: Path, subset: str | None = None) -> list[Path]:
    """Glyph masks in ``input_dir``.

    With ``subset`` (a targets/ dir name) return every capture of that subset. With
    ``subset=None`` return the combined-set captures -- glyph masks whose label
    survives the exclusion rules -- and drop objects, abstract shapes, etc.
    """
    def keep(path: Path) -> bool:
        if subset is not None:
            return _capture_subset(path) == subset
        return (_GLYPH_CAPTURE.fullmatch(path.name) is not None
                and canonicalize_label(label_from_filename(path)) is not None)

    return sorted(path for path in input_dir.glob("*_mask.png") if keep(path))


def target_paths(label: str, subset: str, targets_root: Path) -> list[Path]:
    """Every shipped font's mask of the reference glyph a capture was posed against.

    All fonts are kept -- the ceiling is "how well CLIP reads *this glyph*", and
    averaging over the shipped fonts is a less brittle estimate than picking one.
    """
    found = sorted((targets_root / subset).glob(f"{label}_*.png"))
    if not found:
        raise ValueError(
            f"No reference glyph for {subset}/{label} under {targets_root}. "
            "The recognizability ceiling needs it: add the target or drop the capture."
        )
    return found


def load_glyph(path: Path) -> Image.Image:
    """Load a 1-bit glyph mask as RGB, fed to CLIP as-is (no shadow render)."""
    return Image.open(path).convert("RGB")


def _predictions_blob(predictions: list[dict]) -> str:
    return json.dumps({
        position: [prediction["label"], prediction["similarity"]]
        for position, prediction in enumerate(predictions)
    })


def _write_shadow_csv(path: Path, image_paths, source_labels, score_labels, true_idx,
                      result: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=(
            "path", "filename", "source_label", "true_label", "true_idx", "rank",
            "top1", "clip_top1_label", "clip_top1_similarity", "clip_top_predictions",
        ))
        writer.writeheader()
        for path_, source_label, score_label, index, rank, predictions in zip(
                image_paths, source_labels, score_labels, true_idx,
                result["rank"], result["top_predictions"]):
            writer.writerow({
                "path": str(path_),
                "filename": path_.name,
                "source_label": source_label,
                "true_label": score_label,
                "true_idx": index,
                "rank": rank,
                "top1": int(rank == 1),
                "clip_top1_label": predictions[0]["label"],
                "clip_top1_similarity": predictions[0]["similarity"],
                "clip_top_predictions": _predictions_blob(predictions),
            })


def _write_target_csv(path: Path, target_rows, result: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=(
            "target_path", "filename", "capture", "glyph", "true_idx", "rank",
            "top1", "clip_top1_label", "clip_top1_similarity", "clip_top_predictions",
        ))
        writer.writeheader()
        for (capture, glyph, index, target_path), rank, predictions in zip(
                target_rows, result["rank"], result["top_predictions"]):
            writer.writerow({
                "target_path": str(target_path),
                "filename": target_path.name,
                "capture": capture.name,
                "glyph": glyph,
                "true_idx": index,
                "rank": rank,
                "top1": int(rank == 1),
                "clip_top1_label": predictions[0]["label"],
                "clip_top1_similarity": predictions[0]["similarity"],
                "clip_top_predictions": _predictions_blob(predictions),
            })


def _evaluate(image_paths: list[Path], score_labels: list[str], glob_labels: list[str],
              class_names: list[str], output_dir: Path, input_dir: Path,
              targets_root: Path, model_name: str, pretrained: str, batch_size: int,
              top_k: int, extra_summary: dict | None = None) -> dict:
    """Shadow + ceiling retrieval against one candidate class list.

    ``score_labels`` are the classes ranks are measured against; ``glob_labels`` are
    the literal glyphs used to locate reference PNGs in ``targets/`` (they differ
    only in the combined mode, where e.g. a ``w`` capture is scored as ``W``).
    """
    index_of = {label: i for i, label in enumerate(class_names)}
    missing = sorted({label for label in score_labels if label not in index_of})
    if missing:
        raise ValueError(f"Candidate list is missing scored classes: {missing}")
    true_idx = [index_of[label] for label in score_labels]

    # Resolve the ceiling's reference glyphs first: a missing target is a cheap
    # failure to hit before paying for any CLIP encoding.
    target_rows = [
        (capture, glob_label, index_of[score_label], target_path)
        for capture, glob_label, score_label in zip(image_paths, glob_labels, score_labels)
        for target_path in target_paths(glob_label, glyph_subset(capture), targets_root)
    ]

    retrieval_kwargs = dict(
        class_names=class_names,
        model_name=model_name,
        pretrained=pretrained,
        prompt_tmpl=glyph_prompt,
        batch_size=batch_size,
        top_k=top_k,
    )

    # Shadow condition: the captured shadow mask, fed to CLIP as-is.
    shadow_result = clip_retrieval(
        images=[load_glyph(path) for path in image_paths],
        true_idx=true_idx,
        **retrieval_kwargs,
    )

    # Ceiling condition: the reference glyphs, every font, through the same path.
    target_result = clip_retrieval(
        images=[load_glyph(row[3]) for row in target_rows],
        true_idx=[row[2] for row in target_rows],
        **retrieval_kwargs,
    )

    ratio = {
        key: recognizability_ratio(shadow_result, target_result, key)
        for key in ("top1", "top5", "mrr")
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_shadow_csv(output_dir / "clip_results.csv", image_paths,
                      glob_labels, score_labels, true_idx, shadow_result)
    _write_target_csv(output_dir / "clip_targets.csv", target_rows, target_result)

    summary = {
        "input_dir": str(input_dir),
        "targets_root": str(targets_root),
        "n_images": len(image_paths),
        "n_target_images": len(target_rows),
        "n_classes": len(class_names),
        "class_names": class_names,
        "model_name": model_name,
        "pretrained": pretrained,
        "prompt": {
            "uppercase": "a shadow of the uppercase letter {label}",
            "lowercase": "a shadow of the lowercase letter {label}",
            "digit": "a shadow of the digit {label}",
        },
        "input": "raw 1-bit masks; render_shadow() not applied",
        **(extra_summary or {}),
        # Per-item top-k candidates live in the CSVs; the summary keeps only ranks.
        "shadow": {key: shadow_result[key]
                   for key in ("top1", "top5", "mrr", "chance_top1", "rank")},
        "target": {key: target_result[key]
                   for key in ("top1", "top5", "mrr", "chance_top1", "rank")},
        "recognizability_ratio": ratio,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def evaluate(input_dir: Path, output_dir: Path, model_name: str, pretrained: str,
             batch_size: int, top_k: int = 10,
             targets_root: Path = DEFAULT_TARGETS_ROOT) -> dict:
    """Run the shadow and ceiling conditions over the benchmark's combined class set."""
    image_paths = find_mask_images(input_dir)
    if not image_paths:
        raise ValueError(
            f"No letter/digit files ending in _mask.png found in {input_dir}"
        )
    source_labels = [label_from_filename(path) for path in image_paths]
    score_labels = [canonicalize_label(label) for label in source_labels]
    return _evaluate(image_paths, score_labels, source_labels, ALL_LABELS,
                     output_dir, input_dir, targets_root, model_name, pretrained,
                     batch_size, top_k)


def evaluate_by_subset(input_dir: Path, output_dir: Path, model_name: str,
                       pretrained: str, batch_size: int, top_k: int = 10,
                       targets_root: Path = DEFAULT_TARGETS_ROOT) -> dict:
    """One self-contained retrieval per subset: digits, lowercase, uppercase.

    Each subset sees only its own captures and targets, ranked against its own full
    single-case alphabet with the matching prompt. No exclusions, no collapses.
    """
    subsets: dict[str, dict] = {}
    for subset, alphabet in SUBSET_ALPHABETS.items():
        image_paths = find_mask_images(input_dir, subset=subset)
        if not image_paths:
            continue
        labels = [label_from_filename(path) for path in image_paths]
        subsets[subset] = _evaluate(
            image_paths, labels, labels, alphabet,
            output_dir / subset, input_dir, targets_root, model_name, pretrained,
            batch_size, top_k, extra_summary={"subset": subset})

    if not subsets:
        raise ValueError(
            f"No letter/digit files ending in _mask.png found in {input_dir}"
        )

    combined = {
        "input_dir": str(input_dir),
        "mode": "by_subset",
        "model_name": model_name,
        "pretrained": pretrained,
        "subsets": {
            subset: {
                "n_images": summary["n_images"],
                "n_target_images": summary["n_target_images"],
                "n_classes": summary["n_classes"],
                "shadow": {key: summary["shadow"][key]
                           for key in ("top1", "top5", "mrr", "chance_top1")},
                "target": {key: summary["target"][key]
                           for key in ("top1", "top5", "mrr", "chance_top1")},
                "recognizability_ratio": summary["recognizability_ratio"],
            }
            for subset, summary in subsets.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "by_subset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
        f.write("\n")
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of highest-scoring labels to save per image.")
    parser.add_argument("--by-subset", action="store_true",
                        help="Evaluate digits, lowercase and uppercase as three "
                             "self-contained retrievals over their own alphabets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = vars(args)
    by_subset = kwargs.pop("by_subset")
    output_dir = kwargs["output_dir"]

    if by_subset:
        combined = evaluate_by_subset(**kwargs)
        print(json.dumps({**combined, "output_dir": str(output_dir)}, indent=2))
        return

    summary = evaluate(**kwargs)
    print(json.dumps({
        "n_images": summary["n_images"],
        "n_target_images": summary["n_target_images"],
        "shadow": {key: summary["shadow"][key] for key in ("top1", "top5", "mrr")},
        "target": {key: summary["target"][key] for key in ("top1", "top5", "mrr")},
        "recognizability_ratio": summary["recognizability_ratio"],
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
