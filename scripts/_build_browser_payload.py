#!/usr/bin/env python3
"""Build the per-target browser payload: thumbnails + metrics, one JSON."""
import base64, io, json, os, sys
import numpy as np, pandas as pd
from PIL import Image

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PX = int(sys.argv[1]) if len(sys.argv) > 1 else 128
OUT = os.path.join(BENCH, "results", "browser_payload.json")

SWEEPS = {"big": ("base-big", "optimized/base-optimizer/big-budget"),
          "small": ("base-small", "optimized/base-optimizer/small-budget")}
# The four-metric panel (METRICS.md): area / outline / topology / pose-invariant shape.
# iou,nsd higher-better; pw_h1,hu_distance lower-better. betti_error deliberately excluded.
METRICS = ["iou", "nsd", "pw_h1", "hu_distance", "boundary_iou", "cldice", "chamfer"]
ATTRS = ["area_frac", "solidity", "compactness", "median_stroke_width_rel",
         "min_stroke_width_rel", "thin_mass_frac", "elongation", "n_holes", "n_holes_signif",
         "n_components", "n_limbs", "aspect_ratio"]


def thumb(path):
    """1-bit mask -> compact base64 PNG. Source: dark = shape."""
    im = Image.open(path).convert("L").resize((PX, PX), Image.LANCZOS)
    im = im.point(lambda v: 255 if v > 127 else 0).convert("1")
    b = io.BytesIO(); im.save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode("ascii")


meta = {}
with open(os.path.join(BENCH, "metadata.jsonl")) as f:
    for line in f:
        d = json.loads(line); meta[d["id"]] = d

df = pd.read_csv(os.path.join(BENCH, "results", "master_table.csv"), low_memory=False)
df = df[df["ref"] == "original"]
rows = {}
for key, (sweep, _) in SWEEPS.items():
    sub = df[df["sweep"] == sweep].set_index("sample_id")
    rows[key] = sub

samples, missing = [], 0
for sid, d in meta.items():
    tpath = os.path.join(BENCH, d["target"])
    if not os.path.exists(tpath):
        missing += 1; continue
    rec = {"id": sid, "subset": d["subset"], "cls": d.get("class", ""),
           "t": thumb(tpath),
           "a": {k: d["attributes"].get(k) for k in ATTRS}}
    for key, (sweep, base) in SWEEPS.items():
        stem = os.path.splitext(os.path.basename(d["target"]))[0]
        sp = os.path.join(BENCH, base, d["subset"], stem, f"{stem}_best.png")
        if not os.path.exists(sp):
            continue
        e = {"s": thumb(sp)}
        if sid in rows[key].index:
            r = rows[key].loc[sid]
            for m in METRICS:
                v = r.get(m)
                e[m] = None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)
            for k, c in (("best", "best_iou_reported"), ("mean", "mean_iou_reported"),
                         ("std", "std_iou_reported"), ("sec", "seconds")):
                v = r.get(c)
                e[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)
        rec[key] = e
    samples.append(rec)

payload = {"px": PX, "n": len(samples), "metrics": METRICS, "attrs": ATTRS,
           "subsets": sorted({s["subset"] for s in samples}), "samples": samples}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(payload, f, separators=(",", ":"))
print(f"samples={len(samples)} missing_target={missing} px={PX}")
print(f"bytes={os.path.getsize(OUT)/1e6:.2f} MB -> {OUT}")
