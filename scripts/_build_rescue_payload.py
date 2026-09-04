#!/usr/bin/env python3
"""Payload for the dashboard's Rescue view — the v1 -> v2 replacement.

Reads `dropped.jsonl` (the quarantined originals) and joins each to the v2 that
replaced it, so the view shows what left the benchmark next to what took its
place. The v1 files are read from `dropped/`, where quarantine_dropped.py put
them — they are out of every scanned tree but still on disk, which is the whole
point of moving rather than deleting.
"""
import base64, io, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
import _rescue_common as rc

def png(m, px=256):
    im = Image.fromarray(np.where(np.asarray(m) > 0, 0, 255).astype(np.uint8), "L")
    return base64.b64encode(
        (lambda b: (im.resize((px, px), Image.NEAREST).convert("1").save(b, "PNG", optimize=True), b)[1])
        (io.BytesIO()).getvalue()).decode("ascii")

live = [json.loads(l) for l in open(os.path.join(rc.BENCH, "metadata.jsonl"))]
v2 = {r["rescue"]["derived_from"]: r for r in live if r.get("version") == 2}
drop = [json.loads(l) for l in open(os.path.join(rc.BENCH, "dropped.jsonl"))]
floor, prov = rc.rig_floor()

items = []
for r in drop:
    p1 = os.path.join(rc.BENCH, r.get("quarantined_target", r["target"]))
    if not os.path.exists(p1):
        continue
    m1 = rc.load_mask(p1)
    e = {"id": r["id"], "subset": r["subset"], "cls": r.get("class", ""),
         "o": png(m1), "stroke0": round(rc.stroke_at_solve_res(m1), 5)}
    n = v2.get(r["id"])
    if n:
        e.update({"nid": n["id"], "n": png(rc.load_mask(os.path.join(rc.BENCH, n["target"]))),
                  **{k: v for k, v in n["rescue"].items()
                     if k in ("op", "radius_px", "stroke_after", "iou_vs_original",
                              "area_gain", "holes_signif_before", "holes_signif_after",
                              "reviewed")}})
        e["status"] = n["rescue"]["op"]
    else:
        e["status"] = "unrescuable"
    items.append(e)

solved = sum(1 for r in live if r.get("version") != 2)
out = {"px": 256, "floor": round(floor, 5), "floor_provenance": prov,
       "n_index": len(live), "n": len(items),
       "n_v2": len(v2), "n_pending": len(live) - solved,
       "items": items}
json.dump(out, open(os.path.join(rc.BENCH, "results", "rescue_payload.json"), "w"),
          separators=(",", ":"))
from collections import Counter
print(Counter(i["status"] for i in items), f"| index {len(live)} | v2 {len(v2)}")
