#!/usr/bin/env python3
"""Payload for the dashboard's Teleop view: photo -> rectified -> mask -> target.

Photographs go out as JPEG and masks as 1-bit PNG; a 1 MB wall photo carries no
more information at review size than a 12 KB one, and the mask is what the review
is actually about.

Each capture is also checked against the CURRENT index, because two of them were
posed against a target that the v1->v2 replacement has since retired. A capture
whose target no longer exists is not a broken capture, but it is not directly
comparable either, and the view says so rather than quietly showing the v2.
"""
import base64, io, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
import _rescue_common as rc

def jpg(path, w=300, q=72):
    im = Image.open(path).convert("RGB")
    im = im.resize((w, max(1, round(w * im.height / im.width))), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=q, optimize=True)
    return base64.b64encode(b.getvalue()).decode("ascii")

def flatfield(path, blur_frac=0.25, lo=0.55, hi=1.15, clip=5.0):
    """Illumination removed, then local contrast stretched. 0 = shadow, 255 = wall.

    Two steps, and both are needed. Dividing by a heavily blurred copy removes the
    lamp's falloff, but what is left still only spans about a fifth of the range —
    the shadow is 30-60 grey levels under the wall and that is all. CLAHE then
    stretches contrast per tile, which is what gives the boundary an actual
    gradient for a seeded method to hold on to; a global stretch cannot, because
    the residual falloff varies across the frame.

    clipLimit 5 rather than a higher one: a global stretch ahead of CLAHE pushes
    the range further still (std 42-48 against 29-45) but starts printing the
    wall's own texture, and grain is worse than flatness here — a geodesic cost
    reads spurious texture as spurious edges.

    Shipped at native resolution: the browser segments ON this array, so the pixel
    that gets clicked has to be the pixel that gets read.
    """
    import cv2 as _cv
    g = np.array(Image.open(path).convert("L"), np.float32)
    k = int(blur_frac * max(g.shape)) | 1
    fl = g / np.maximum(_cv.GaussianBlur(g, (k, k), 0), 1e-3)
    q = (np.clip((fl - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    q = _cv.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(q)
    b = io.BytesIO()
    Image.fromarray(q, "L").save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode("ascii"), g.shape


def png1(path, w=300):
    im = Image.open(path).convert("L")
    im = im.resize((w, max(1, round(w * im.height / im.width))), Image.NEAREST).convert("1")
    b = io.BytesIO(); im.save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode("ascii")

man = json.load(open(os.path.join(rc.BENCH, "Teleops", "masks", "manifest.json")))
live = {json.loads(l)["id"]: json.loads(l)
        for l in open(os.path.join(rc.BENCH, "metadata.jsonl"))}
drop = {json.loads(l)["id"]: json.loads(l)
        for l in open(os.path.join(rc.BENCH, "dropped.jsonl"))}
v2 = {r["rescue"]["derived_from"]: r for r in live.values() if r.get("version") == 2}

items = []
for r in man["records"]:
    e = {k: r[k] for k in ("capture", "subset", "class", "part", "otsu_thr",
                           "shape_frac", "n_components", "holes_signif", "suspect", "match")}
    # How the committed mask was actually made. The view states it rather than
    # implying the browser's own segmenter produced what is on screen.
    e["mask_backend"] = r.get("mask_backend")
    e["reason"] = r.get("reason", "")
    e["sample_id"] = r.get("sample_id")
    e["raw"] = jpg(os.path.join(rc.BENCH, r["raw"]), 300) if os.path.exists(
        os.path.join(rc.BENCH, r["raw"])) else None
    e["rect"] = jpg(os.path.join(rc.BENCH, r["rectified"]), 300)
    ff, shp = flatfield(os.path.join(rc.BENCH, r["rectified"]))
    e["ff"] = ff
    e["h"], e["w"] = int(shp[0]), int(shp[1])
    e["mask"] = png1(os.path.join(rc.BENCH, r["mask"]), 300)
    sid = r.get("sample_id")
    if sid and sid in live:
        e["target"] = png1(os.path.join(rc.BENCH, live[sid]["target"]), 240)
        e["target_status"] = "live"
    elif sid and sid in v2:
        e["target"] = png1(os.path.join(rc.BENCH, drop[sid]["quarantined_target"]), 240)
        e["target_status"] = "replaced_by_v2"
        e["v2_id"] = v2[sid]["id"]
        e["target_v2"] = png1(os.path.join(rc.BENCH, v2[sid]["target"]), 240)
    else:
        e["target"] = None
        e["target_status"] = "unmatched"
    items.append(e)

out = {"n": len(items), "open_r": man["open_r"], "ff_window": [0.55, 1.15],
       "median_frac": float(np.median([i["shape_frac"] for i in items])),
       "n_suspect": sum(1 for i in items if i["suspect"]),
       "n_live": sum(1 for i in items if i["target_status"] == "live"),
       "items": items}
p = os.path.join(rc.BENCH, "results", "teleop_payload.json")
json.dump(out, open(p, "w"), separators=(",", ":"))
from collections import Counter
print(Counter(i["target_status"] for i in items), "| suspect", out["n_suspect"],
      "|", round(os.path.getsize(p)/1e3), "KB")
