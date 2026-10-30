"""Aggregate results/fastshadow/pose_library (scripts/pose_library_study.py) for the solver bench page."""

from __future__ import annotations

import json
import os

import numpy as np

PAIRS = [("lib_cma_f30", "cold_8k"), ("lib_cma_f16", "cold_3k"), ("lib_cma_f4", "cold_300"),
         ("lib_adam_64x100", "adam_64x300")]


def load(bench: str):
    d = os.path.join(bench, "results", "fastshadow", "pose_library")
    p = os.path.join(d, "runs.jsonl")
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    ids = sorted({r["id"] for r in rows})
    per = lambda m, k: np.array([np.mean([r[k] for r in rows if r["id"] == i and r["method"] == m]) for i in ids])  # noqa: E731
    g = np.random.default_rng(0)

    def ci(x):
        b = x[g.integers(0, len(x), (10000, len(x)))].mean(1)
        return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    methods = [{"m": m, "iou": float(per(m, "iou").mean()), "tc": float(per(m, "tc_iou").mean()),
                "sec": float(per(m, "seconds").mean()),
                "clean": float(np.mean([r["clean"] for r in rows if r["method"] == m]))}
               for m in sorted({r["method"] for r in rows})]
    methods.sort(key=lambda x: x["sec"])
    pairs = [{"a": a, "b": b, "speedup": float(per(b, "seconds").mean() / per(a, "seconds").mean()),
              "iou": ci(per(a, "iou") - per(b, "iou")), "tc": ci(per(a, "tc_iou") - per(b, "tc_iou"))} for a, b in PAIRS]
    return {"n": len(ids), "lib": json.load(open(os.path.join(d, "library_n3.json"), encoding="utf-8"))["n"],
            "methods": methods, "pairs": pairs}
