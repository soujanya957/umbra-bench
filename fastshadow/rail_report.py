"""Aggregate results/fastshadow/rail_study (scripts/rail_study.py) for the solver bench page."""

from __future__ import annotations

import base64
import io
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

INK = np.array([38, 36, 44])
TGT = np.array([150, 160, 178])
WALL = np.array([247, 245, 240])


def _uri(mask, color):
    img = np.tile(WALL, mask.shape + (1,))
    img[mask] = color
    b = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def _mask(path):
    return np.array(Image.open(path).convert("L")) < 128


def load(bench: str, n_examples: int = 6):
    d = os.path.join(bench, "results", "fastshadow", "rail_study")
    p, sp = os.path.join(d, "runs.jsonl"), os.path.join(d, "summary.json")
    if not (os.path.exists(p) and os.path.exists(sp)):
        return None
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    summ = json.load(open(sp, encoding="utf-8"))
    runs = [json.loads(l) for l in open(p, encoding="utf-8")]
    by = defaultdict(lambda: defaultdict(list))
    for r in runs:
        by[(r["config"], r["optimizer"])][r["id"]].append(r)
    ids = sorted(set.intersection(*[set(v) for v in by.values()]))
    # examples: the targets where moving the carriages changed CMA-ES's best seed the most, both ways
    gain = {i: max(r["iou"] for r in by[("rail_free", "cma")][i]) - max(r["iou"] for r in by[("rail_even", "cma")][i])
            for i in ids}
    order = sorted(ids, key=lambda i: -gain[i])
    pick = order[: n_examples // 2] + order[-(n_examples - n_examples // 2):]
    ex = []
    for i in pick:
        sub, stem = i.split("/")
        T = _mask(os.path.join(d, "fitted", f"{sub}__{stem}.png"))
        row = {"id": i, "ch": stem.split("_")[0], "target": _uri(T, TGT), "gain": gain[i]}
        for c in ("rail_even", "rail_free"):
            b = max(by[(c, "cma")][i], key=lambda r: r["iou"])
            row[c] = {"img": _uri(_mask(os.path.join(d, b["mask"])), INK), "iou": b["iou"], "s": b["s_m"]}
        ex.append(row)
    # where the free carriages went: every rail_free solve's positions
    spots = {o: [r["s_m"] for i in ids for r in by[("rail_free", o)][i]] for o in ("cma", "adam")}
    # other rail placements (scripts/rail_study.py --light-dist / --screen-dist), one folder each
    variants = []
    base = os.path.join(bench, "results", "fastshadow")
    for name in sorted(os.listdir(base)):
        vd = os.path.join(base, name)
        if not name.startswith("rail_study") or not os.path.exists(os.path.join(vd, "summary.json")):
            continue
        vm = json.load(open(os.path.join(vd, "meta.json"), encoding="utf-8"))
        vs = json.load(open(os.path.join(vd, "summary.json"), encoding="utf-8"))
        tr = vm["track"]
        throw = tr["light_dist"] + tr["length"] + tr["screen_dist"]
        variants.append({"light_dist": tr["light_dist"], "screen_dist": tr["screen_dist"],
                         "mag": [throw / (tr["light_dist"] + tr["s_max"]), throw / (tr["light_dist"] + tr["s_min"])],
                         "cells": vs["cells"], "free_minus_even": vs["free_minus_even"],
                         "rail_minus_old_tc": vs.get("rail_minus_old_tc", {}), "n_targets": vs["n_targets"]})
    variants.sort(key=lambda v: (v["light_dist"], v["screen_dist"]))
    return {"meta": {"track": meta["track"], "s_even": meta["s_even"], "n": meta["n"], "n_targets": len(ids)},
            "summary": summ, "examples": ex, "spots": spots, "variants": variants}
