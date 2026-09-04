from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from eval_metrics_screen import METRICS, fit_global, metric_columns


def test_global_fit_is_shared_similarity_transform() -> None:
    target = np.zeros((12, 12), np.uint8)
    target[4:8, 4:8] = 1
    capture = np.zeros((12, 12), np.uint8)
    capture[5:9, 4:8] = 1
    fit = fit_global([capture, capture], [target, target], capture.shape)
    assert fit[0] == fit[1]  # isotropic scale only
    assert fit[3] == 0.0     # no rotation is searched


def test_metric_columns_are_unambiguous_and_cap_zero_is_checked() -> None:
    columns = metric_columns({"iou": 0.4, "pw_h1": None}, {"iou": 0.4, "pw_h1": None})
    assert columns["iou_original"] == columns["iou_warped"] == 0.4
    assert "iou" not in columns
    assert columns["pw_h1_original"] is None


def test_unimplemented_hd95_bordersafe_is_not_a_candidate() -> None:
    assert "hd95_bordersafe" not in METRICS
