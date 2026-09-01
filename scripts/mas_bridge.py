"""Explicit dtype/polarity boundary for motion-aware-shadow masks.

umbra-bench uses uint8 masks with ink=1.  motion-aware-shadow uses float32
masks with the same polarity and a >0.5 foreground test.  Keep that conversion
in one place so an accidental inversion cannot silently enter an evaluation.
"""
from __future__ import annotations

import numpy as np


def to_mas(mask: np.ndarray) -> np.ndarray:
    """Return a float32 `{0.0, 1.0}` ink mask for motion-aware-shadow."""
    return (np.asarray(mask) > 0).astype(np.float32)


def from_mas(mask: np.ndarray) -> np.ndarray:
    """Return a uint8 `{0, 1}` ink mask from motion-aware-shadow output."""
    return (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.uint8)
