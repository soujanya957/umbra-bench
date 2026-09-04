"""Boundary tests for the motion-aware-shadow mask adapter."""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from mas_bridge import from_mas, to_mas

MAS_ROOT = Path("/Users/liusimin/Documents/Code/fleet-shadow-art/motion-aware-shadow")


def _mas_module(name: str):
    if not MAS_ROOT.is_dir():
        pytest.skip("motion-aware-shadow checkout is unavailable")
    if str(MAS_ROOT) not in sys.path:
        sys.path.insert(0, str(MAS_ROOT))
    # distortion_sweep imports its renderer at module import time even though
    # load_target itself is pure image IO.  Stub those unused heavy dependencies
    # so this test exercises the canonical loader without requiring MuJoCo.
    if name == "scripts.distortion_sweep":
        optimizer = types.ModuleType("optimizer")
        optimizer.OptimizerConfig = object
        optimizer.optimize_staged = lambda *args, **kwargs: None
        renderer = types.ModuleType("renderer")
        renderer.ShadowRenderer = object
        renderer.build_scene = lambda *args, **kwargs: None
        sys.modules.setdefault("optimizer", optimizer)
        sys.modules.setdefault("renderer", renderer)
    return importlib.import_module(name)


def test_round_trip_preserves_ink_pixels() -> None:
    mask = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    converted = to_mas(mask)
    assert converted.dtype == np.float32
    assert np.array_equal(from_mas(converted), mask)


def test_all_benchmark_targets_pass_mas_corner_check() -> None:
    sweep = _mas_module("scripts.distortion_sweep")
    inverted = []
    for path in ROOT.glob("targets/**/*.png"):
        raw = to_mas(__import__("metrics").load_mask(str(path), 128))
        loaded = sweep.load_target(str(path), 128)
        # This is exactly load_target's inversion predicate.  Do not compare
        # pixels: MAS intentionally resizes with LANCZOS while benchmark loading
        # uses NEAREST, so edge pixels legitimately differ.
        corners = raw[0, 0] + raw[0, -1] + raw[-1, 0] + raw[-1, -1]
        if corners >= 3.0 and raw.mean() > 0.5:
            inverted.append(str(path.relative_to(ROOT)))
        assert loaded.dtype == np.float32
    assert not inverted, f"inverted targets: {', '.join(inverted)}"


def test_identity_warp_survives_adapter_round_trip() -> None:
    warp = _mas_module("target_warp")
    source = np.zeros((8, 8), dtype=np.uint8)
    source[2:6, 3:5] = 1
    result = warp.apply_warp(to_mas(source), warp.WarpField.identity(4), binarize=True)
    assert np.array_equal(from_mas(result), source)
