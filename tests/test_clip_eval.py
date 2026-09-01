from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from clip_eval import (
    ALL_LABELS,
    canonicalize_label,
    evaluate,
    find_mask_images,
    label_from_filename,
    label_index,
)


def test_class_list_filters_ambiguous_lowercase_labels_and_aliases() -> None:
    assert ALL_LABELS[0] == "A"
    assert "C" in ALL_LABELS
    assert "c" not in ALL_LABELS
    assert "n" in ALL_LABELS
    assert "I" not in ALL_LABELS
    assert "l" not in ALL_LABELS
    assert "O" not in ALL_LABELS
    assert "o" not in ALL_LABELS
    assert "9" not in ALL_LABELS
    assert {"0", "1", "q"} <= set(ALL_LABELS)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("letters_upperH_A4_structure_mask.png", "H"),
        ("letters_lowerb_C4_human_study_mask.png", "b"),
        ("digits7_D1_medium_tier_mask.png", "7"),
    ],
)
def test_label_from_filename(filename: str, expected: str) -> None:
    assert label_from_filename(Path(filename)) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("A", 0),
        ("Z", ALL_LABELS.index("Z")),
        ("a", ALL_LABELS.index("a")),
        ("0", ALL_LABELS.index("0")),
        ("I", ALL_LABELS.index("1")),
        ("l", ALL_LABELS.index("1")),
        ("9", ALL_LABELS.index("q")),
    ],
)
def test_label_index(label: str, expected: int) -> None:
    assert label_index(label) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("I", "1"), ("l", "1"), ("1", "1"),
        ("O", "0"), ("o", "0"), ("0", "0"),
        ("q", "q"), ("9", "q"),
        ("c", "C"), ("k", "K"), ("m", "M"), ("n", "n"),
        ("p", "P"), ("s", "S"), ("v", "V"), ("w", "W"), ("z", "Z"),
    ],
)
def test_canonicalize_label_applies_aliases_and_exclusions(
        label: str, expected: str | None) -> None:
    assert canonicalize_label(label) == expected


def test_label_from_filename_rejects_non_glyph_names() -> None:
    with pytest.raises(ValueError, match="does not encode"):
        label_from_filename(Path("objects_fork_D1_honest_failure_mask.png"))


def test_find_mask_images_filters_to_glyph_masks(tmp_path: Path) -> None:
    for name in (
        "letters_upperA_D1_mask.png",
        "digits7_D1_mask.png",
        "letters_lowerc_D1_mask.png",
        "objects_fork_D1_mask.png",
        "letters_lowerb_D1.png",
    ):
        (tmp_path / name).touch()

    assert [path.name for path in find_mask_images(tmp_path)] == [
        "digits7_D1_mask.png",
        "letters_lowerc_D1_mask.png",
        "letters_upperA_D1_mask.png",
    ]


def test_evaluate_writes_csv_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_dir = tmp_path / "captured_images"
    input_dir.mkdir()
    output_dir = tmp_path / "results" / "clip_eval"

    filenames = [
        "digits7_D1_medium_tier_mask.png",
        "letters_upperA_D1_attribute_space_coverage_mask.png",
        "letters_lowerb_C4_human_study_mask.png",
    ]
    for name in filenames:
        Image.new("RGB", (8, 8), color=(128, 128, 128)).save(input_dir / name)

    def fake_clip_retrieval(*, images, class_names, true_idx, model_name, pretrained,
                            prompt_tmpl, batch_size, top_k):
        assert len(images) == 3
        assert class_names == ALL_LABELS
        assert true_idx == [ALL_LABELS.index("7"), ALL_LABELS.index("b"), ALL_LABELS.index("A")]
        assert model_name == "ViT-B-32"
        assert pretrained == "laion2b_s34b_b79k"
        assert prompt_tmpl("N") == "a shadow of the uppercase letter N"
        assert prompt_tmpl("n") == "a shadow of the lowercase letter n"
        assert prompt_tmpl("7") == "a shadow of the digit 7"
        assert batch_size == 16
        assert top_k == 10
        return {
            "rank": [1, 2, 5],
            "top_predictions": [
                [{"label": "7", "index": 59, "similarity": 0.7}],
                [{"label": "B", "index": 1, "similarity": 0.6}],
                [{"label": "A", "index": 0, "similarity": 0.5}],
            ],
            "top1": 1 / 3,
            "top5": 1.0,
            "mrr": 0.5666666667,
            "chance_top1": 1 / len(ALL_LABELS),
        }

    monkeypatch.setattr("clip_eval.clip_retrieval", fake_clip_retrieval)

    summary = evaluate(
        input_dir=input_dir,
        output_dir=output_dir,
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        batch_size=16,
    )

    assert summary["n_images"] == 3
    assert summary["rank"] == [1, 2, 5]

    csv_path = output_dir / "clip_results.csv"
    summary_path = output_dir / "summary.json"
    assert csv_path.is_file()
    assert summary_path.is_file()

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["filename"] for row in rows] == sorted(filenames)
    assert [row["true_label"] for row in rows] == ["7", "b", "A"]
    assert [int(row["true_idx"]) for row in rows] == [
        ALL_LABELS.index("7"), ALL_LABELS.index("b"), ALL_LABELS.index("A")
    ]
    assert [int(row["rank"]) for row in rows] == [1, 2, 5]
    assert [int(row["top1"]) for row in rows] == [1, 0, 0]
    assert [row["clip_top1_label"] for row in rows] == ["7", "B", "A"]
    assert [float(row["clip_top1_similarity"]) for row in rows] == [0.7, 0.6, 0.5]
    assert json.loads(rows[0]["clip_top_predictions"]) == {"0": ["7", 0.7]}

    with summary_path.open(encoding="utf-8") as f:
        persisted_summary = json.load(f)
    assert persisted_summary == summary
