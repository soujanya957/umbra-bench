"""Aggregate results/fastshadow/arrangement_study for the report page."""

from __future__ import annotations

import base64
import io
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

CMA_INK = np.array([15, 110, 102])
ADAM_INK = np.array([58, 63, 176])
INK = np.array([38, 36, 44])
WALL = np.array([247, 245, 240])


def _uri(mask, ink):
    img = np.tile(WALL, mask.shape + (1,))
    img[mask] = ink
    b = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def _mask(path):
    return np.array(Image.open(path).convert("L")) < 128


def load(bench: str):
    d = os.environ.get("ARRANGEMENT_DIR") or os.path.join(bench, "results", "fastshadow", "arrangement_study")
    p = os.path.join(d, "runs.jsonl")
    if not os.path.exists(p):
        return None
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    runs = [json.loads(l) for l in open(p, encoding="utf-8")]
    configs = [c for c in meta["configs"] if any(r["config"] == c for r in runs)]
    opts = [o for o in ("cma", "adam") if any(r["optimizer"] == o for r in runs)]
    by = defaultdict(list)
    for r in runs:
        by[(r["id"], r["config"], r["optimizer"])].append(r)
    ids = [i for i in meta["targets"] if all((i, c, o) in by for c in configs for o in opts)]
    if not ids:
        return None

    def best(rs):  # best of seeds by IoU against the shown target
        return max(rs, key=lambda r: r["iou"])

    summary = []
    for c in configs:
        for o in opts:
            groups = [by[(i, c, o)] for i in ids]
            bests = [best(g) for g in groups]
            allr = [r for g in groups for r in g]
            summary.append({
                "config": c, "opt": o,
                "best_iou": float(np.mean([b["iou"] for b in bests])),
                "mean_iou": float(np.mean([r["iou"] for r in allr])),
                "best_tc": float(np.mean([b["tc_iou"] or 0 for b in bests])),
                "best_cls": float(np.mean([b["cls_top1"] for b in bests])),
                "cf": float(np.mean([r["collision_free"] for r in allr])),
                "sec": float(np.mean([r["seconds"] for r in allr])),
                "sec_target": float(np.mean([sum(r["seconds"] for r in g) for g in groups])),
                "seeds": int(np.median([len(g) for g in groups])),
                "gap_x": float(np.mean([b["gaps_m"][0] for b in bests])),
                "gap_y": float(np.mean([b["gaps_m"][1] for b in bests])),
                "at_lo": float(np.mean([min(b["gaps_m"]) <= 0.1205 for b in bests])) if c != "fixed" else None,
            })
    # budget curves: best-of-k (seeds in order) vs cumulative seconds, mean over digits
    curves = {}
    for c in configs:
        for o in opts:
            groups = [sorted(by[(i, c, o)], key=lambda r: r["seed"]) for i in ids]
            K = min(len(g) for g in groups)
            pts = []
            for k in range(1, K + 1):
                pts.append({"k": k, "sec": float(np.mean([sum(r["seconds"] for r in g[:k]) for g in groups])),
                            "iou": float(np.mean([max(r["iou"] for r in g[:k]) for g in groups]))})
            curves[f"{c}|{o}"] = pts
    # per digit gallery
    root = os.path.join(bench, "optimized", "big-budget-grounded")
    digits = []
    for i in ids:
        sub, stem = i.split("/")
        T = _mask(os.path.join(root, sub, stem, f"{stem}_shown.png"))
        cells = []
        for c in configs:
            cell = {"config": c}
            for o in opts:
                b = best(by[(i, c, o)])
                S = _mask(os.path.join(d, b["mask"]))
                cell[o] = {"img": _uri(S, CMA_INK if o == "cma" else ADAM_INK), "iou": b["iou"], "sec": b["seconds"],
                           "gaps": b["gaps_m"], "bases": b["bases_xy"], "cf": b["collision_free"], "cls": b["cls_top1"],
                           "sec_all": float(sum(r["seconds"] for r in by[(i, c, o)])), "n": len(by[(i, c, o)])}
            cells.append(cell)
        digits.append({"id": i, "digit": stem.split("_")[0], "target": _uri(T, INK), "cells": cells})
    return {"meta": {"objective": meta["objective"], "arrangements": meta["arrangements"], "args": meta["args"]},
            "configs": configs, "opts": opts, "n": len(ids), "summary": summary, "curves": curves, "digits": digits}
