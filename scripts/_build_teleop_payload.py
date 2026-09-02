#!/usr/bin/env python3
"""Payload for the dashboard's Teleop view: photo -> rectified -> mask -> target.

Photographs go out as JPEG and masks as 1-bit PNG; a 1 MB wall photo carries no
more information at review size than a 12 KB one, and the mask is what the review
is actually about.

Each manifest passed via --manifest becomes one switchable set in the view,
synced to its folder: points exported there go to <set>/points.json and a rerun
of teleop_pipeline.py --teleop-root <set> regenerates exactly that folder's
masks. The default is the two tag-rectified sets; the retired v1 flat layout
(Teleops/masks/manifest.json) still bakes if named, minus the deleted photos.

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

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", nargs="+",
                default=[os.path.join(rc.BENCH, "Teleops", "source", d,
                                      "masks", "manifest.json")
                         for d in ("teleop_set1", "teleop_set2")],
                help="masks manifests to browse/label; each becomes a "
                     "switchable set in the view, synced to its folder")
a = ap.parse_args()
live = {json.loads(l)["id"]: json.loads(l)
        for l in open(os.path.join(rc.BENCH, "metadata.jsonl"))}
drop = {json.loads(l)["id"]: json.loads(l)
        for l in open(os.path.join(rc.BENCH, "dropped.jsonl"))}
v2 = {r["rescue"]["derived_from"]: r for r in live.values() if r.get("version") == 2}


def build_items(man):
    items = []
    for r in man["records"]:
        # .get throughout: the v2 pipeline's manifest carries a different field
        # set (quad_source instead of suspect/match); absent keys render as null.
        e = {k: r.get(k) for k in ("capture", "subset", "class", "part", "otsu_thr",
                                   "shape_frac", "n_components", "holes_signif",
                                   "suspect", "match", "quad_source")}
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
        # Tag footprints in the rectified frame, for the browser segmenter to
        # exclude the way the pipeline itself does (background seeds). Absent
        # for the v1 layout, whose capture JSONs predate the field.
        cj = os.path.join(rc.BENCH, os.path.dirname(r["rectified"]),
                          r["capture"] + "_capture.json")
        e["tags"] = None
        if os.path.exists(cj):
            with open(cj, encoding="utf-8") as f:
                tp = json.load(f).get("tag_polys_rectified")
            if tp:
                e["tags"] = [[[int(round(x)), int(round(y))] for x, y in poly]
                             for poly in tp.values()]
        sid = r.get("sample_id")
        if sid and sid in live:
            e["target"] = png1(os.path.join(rc.BENCH, live[sid]["target"]), 240)
            e["target_status"] = "live"
        elif sid and sid in v2:
            e["target"] = png1(os.path.join(rc.BENCH, drop[sid]["quarantined_target"]), 240)
            e["target_status"] = "replaced_by_v2"
            e["v2_id"] = v2[sid]["id"]
            e["target_v2"] = png1(os.path.join(rc.BENCH, v2[sid]["target"]), 240)
            # Point sample_id at the target that still exists. The atlas keys its
            # hand-cast badge on this field, and a retired v1 id matches no card, so
            # two real captures were silently losing their badge. link_teleop.py
            # already re-points the same two when it fills shadows.teleop; this keeps
            # the payload agreeing with metadata.jsonl. The v1 id stays for
            # provenance -- the capture really was posed against the retired shape.
            e["sample_id_v1"] = sid
            e["sample_id"] = v2[sid]["id"]
        else:
            e["target"] = None
            e["target_status"] = "unmatched"
        items.append(e)
    return items


def set_name(src):
    parts = src.split("/")
    return parts[2] if parts[:2] == ["Teleops", "source"] else "teleop_v1"


def build_set(manifest_path):
    man = json.load(open(manifest_path))
    items = build_items(man)
    src = os.path.relpath(os.path.abspath(manifest_path),
                          rc.BENCH).replace(os.sep, "/")
    return {"name": set_name(src),
            # src doubles as the folder pointer the view's sync/rerun hints are
            # built from; None marks the retired v1 layout, which has no folder
            # a per-set rerun could land in.
            "src": None if src == "Teleops/masks/manifest.json" else src,
            "n": len(items), "open_r": man.get("open_r", 2),
            "ff_window": [0.55, 1.15],
            "median_frac": float(np.median([i["shape_frac"] for i in items])),
            "n_suspect": sum(1 for i in items if i.get("suspect")),
            "n_live": sum(1 for i in items if i["target_status"] == "live"),
            "items": items}


sets = [build_set(m) for m in a.manifest]
out = {"n": sum(t["n"] for t in sets), "sets": sets}
p = os.path.join(rc.BENCH, "results", "teleop_payload.json")
json.dump(out, open(p, "w"), separators=(",", ":"))
from collections import Counter
for t in sets:
    print(t["name"], dict(Counter(i["target_status"] for i in t["items"])),
          "| suspect", t["n_suspect"])
print("total", out["n"], "|", round(os.path.getsize(p) / 1e3), "KB")
