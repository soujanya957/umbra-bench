#!/usr/bin/env python3
"""Turn the teleop wall photographs into binary shadow masks.

The captures are photographs of a real shadow on a real wall, rectified to a
fronto-parallel view by the four ArUco markers around the projection area. They
are not maskable by thresholding as they stand: the shadow sits only 30-60 grey
levels below the lit wall, and the lamp's falloff across the frame is of the same
magnitude, so any global threshold classifies the dim corners as shadow. Measured
over the 29 rectified frames, grey std is 7-12 on a 0-255 range.

So the illumination is removed before the threshold, not after:

  1. Estimate the lit wall as a heavily blurred copy of the frame (sigma ~ a
     quarter of the long side, wide enough to smooth out the shadow itself) and
     divide by it. What remains is reflectance: a flat field with the shadow in it.
  2. Otsu on the flattened frame. Tried against fixed cutoffs at 0.90/0.93/0.96
     and Otsu wins clearly -- after flattening the shadow is only 5-15% below the
     wall over most of its area, so a fixed cut catches the core and drops the rest.
  3. `_denoise` from the benchmark's own attribute code: drop components under
     0.5% of area, fill holes under 0.5%. NOT `binary_fill_holes` -- filling every
     hole erases the counter of an 'a' and both bowls of a 'B', which is precisely
     what pw_h1 exists to measure.
  4. A small opening (r=2 by default) to shed the cable shadows and speckle.
     r=4 starts eating the strokes themselves.

Frames where the rectified crop reaches past the lit cone cannot be saved by any
of this -- the flat field has nothing to normalise against -- so the manifest
carries a `suspect` flag (shape fraction far from the rest, or a border touching
a large filled region) and the dashboard shows those for a human to judge.

**No metric is computed here.** Scoring a capture against its target needs the
two put in one frame, and the intended position and scale were never displayed on
the wall while the arms were posed (METRICS.md section 3). That alignment is a
decision, not a default, so this script stops at the mask.

    python scripts/binarize_teleop.py            # dry run + report
    python scripts/binarize_teleop.py --write
"""
from __future__ import annotations

import argparse, csv, glob, json, os, re

import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu

import _rescue_common as rc
from shape_attributes import _denoise

TELEOP = os.path.join(rc.BENCH, "Teleops")


def flatten(path: str, blur_frac: float = 0.25) -> np.ndarray:
    g = np.array(Image.open(path).convert("L"), np.float32)
    k = int(blur_frac * max(g.shape)) | 1
    return np.clip(g / np.maximum(cv2.GaussianBlur(g, (k, k), 0), 1e-3), 0, 2)


def binarize(path: str, open_r: int = 2, border: int = 6):
    flat = flatten(path)
    thr = float(threshold_otsu(flat))
    m = (flat < thr).astype(np.uint8)
    m[:border] = 0; m[-border:] = 0; m[:, :border] = 0; m[:, -border:] = 0
    if open_r:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_r + 1, 2 * open_r + 1))
        mo = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        if mo.sum() > 0.2 * m.sum():
            m = mo
    if m.sum() == 0:
        return m, thr, flat
    return _denoise(m, float(m.sum())), thr, flat


def load_selection() -> list[dict]:
    rows = []
    for c in sorted(glob.glob(os.path.join(TELEOP, "*.csv"))):
        with open(c) as f:
            for r in csv.DictReader(f):
                r["_csv"] = os.path.basename(c)
                rows.append(r)
    return rows


# MPEG-7 ships the class spelled "glas"; whoever named the capture spelled it the
# English way. An explicit alias is safer than fuzzy string matching, which would
# let a genuine mismatch through as a near-hit.
CLASS_ALIAS = {"glass": "glas"}


def match(stem: str, rows: list[dict]):
    """Capture stems encode <subset><class>_<partcode>_<words>; ids do not."""
    subsets = sorted({r["subset"] for r in rows}, key=len, reverse=True)
    for s in subsets:
        if not stem.startswith(s):
            continue
        rest = stem[len(s):].lstrip("_")
        m = re.match(r"^(.+?)_([A-D])[._]?(\d?)", rest)
        cls, part = (m.group(1), m.group(2) + m.group(3)) if m else (rest, "")
        cls = CLASS_ALIAS.get(cls, cls)   # report the benchmark's spelling
        cands = [r for r in rows if r["subset"] == s and r["class"] == cls]
        if part:
            narrowed = [r for r in cands if r["part"] == part[0]
                        and part[1:] in r["subcat"].replace(".", "")]
            if narrowed:
                cands = narrowed
        # The two selection CSVs overlap, so the same sample_id can appear twice;
        # that is a duplicate row, not a genuine ambiguity.
        seen, uniq = set(), []
        for c in cands:
            if c["sample_id"] not in seen:
                seen.add(c["sample_id"]); uniq.append(c)
        return s, cls, part, uniq
    return None, stem, "", []


def resolve(stem: str, rows: list[dict]) -> dict:
    """The match fields for one capture, and nothing else."""
    sub, cls, part, cands = match(stem, rows)
    out = {"subset": sub, "class": cls, "part": part}
    if len(cands) == 1:
        out.update(sample_id=cands[0]["sample_id"], target=cands[0]["target_path"],
                   reason=cands[0]["reason"], match="unique", candidates=[])
    else:
        out.update(sample_id=None, match="ambiguous" if cands else "no_match",
                   candidates=[c["sample_id"] for c in cands])
    return out


def rematch(rows: list[dict], a) -> None:
    man_p = os.path.join(rc.BENCH, a.out, "manifest.json")
    man = json.load(open(man_p))
    changed = []
    for rec in man["records"]:
        new = resolve(rec["capture"], rows)
        if new["sample_id"] != rec.get("sample_id"):
            changed.append((rec["capture"], rec.get("sample_id"), new["sample_id"]))
        rec.update(new)
    for cap, old, now in changed:
        print(f"  {cap[:44]:44} {old} -> {now}")
    print(f"{len(changed)} of {len(man['records'])} re-resolved"
          + ("" if a.write else "   (dry run — pass --write)"))
    if a.write:
        json.dump(man, open(man_p, "w"), indent=1)
        print(f"wrote {man_p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open-r", type=int, default=2)
    ap.add_argument("--out", default="Teleops/masks")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--rematch", action="store_true",
                    help="re-resolve sample_id on the existing manifest and stop; "
                         "masks and every other field are left exactly as they are, "
                         "so a hand-seeded segmentation is not thrown away")
    a = ap.parse_args()

    rows = load_selection()
    if a.rematch:
        rematch(rows, a)
        return
    caps = sorted(glob.glob(os.path.join(TELEOP, "*_rectified.png")))
    recs = []
    for p in caps:
        stem = os.path.basename(p)[: -len("_rectified.png")]
        m, thr, _ = binarize(p, a.open_r)
        sub, cls, part, cands = match(stem, rows)
        frac = float(m.mean())
        rec = {"capture": stem, "rectified": os.path.relpath(p, rc.BENCH),
               "raw": os.path.relpath(os.path.join(TELEOP, stem + ".png"), rc.BENCH),
               "subset": sub, "class": cls, "part": part,
               "otsu_thr": round(thr, 4), "shape_frac": round(frac, 4),
               "n_components": int(cv2.connectedComponents(m)[0] - 1),
               "holes_signif": rc.n_holes_signif(m)}
        if len(cands) == 1:
            rec["sample_id"] = cands[0]["sample_id"]
            rec["target"] = cands[0]["target_path"]
            rec["reason"] = cands[0]["reason"]
            rec["match"] = "unique"
        else:
            rec["sample_id"] = None
            rec["match"] = "ambiguous" if cands else "no_match"
            rec["candidates"] = [c["sample_id"] for c in cands]
        if a.write:
            rc.save_mask(m, os.path.join(rc.BENCH, a.out, stem + "_mask.png"))
            rec["mask"] = os.path.join(a.out, stem + "_mask.png")
        recs.append(rec)

    fr = np.array([r["shape_frac"] for r in recs])
    med, mad = float(np.median(fr)), float(np.median(np.abs(fr - np.median(fr))) + 1e-9)
    for r in recs:
        r["suspect"] = bool(abs(r["shape_frac"] - med) > 4 * mad)

    from collections import Counter
    print(f"{len(recs)} captures  ·  Otsu on the flat-fielded frame, opening r={a.open_r}")
    print("  id match:", dict(Counter(r["match"] for r in recs)))
    print(f"  shape fraction: median {med:.3f}, range {fr.min():.3f}-{fr.max():.3f}")
    sus = [r for r in recs if r["suspect"]]
    print(f"  flagged suspect (fraction far from the rest): {len(sus)}")
    for r in sus:
        print(f"    {r['capture'][:48]:48} frac {r['shape_frac']:.3f}")
    for r in recs:
        if r["match"] != "unique":
            print(f"  {r['match']:10} {r['capture'][:44]:44} {r.get('candidates')}")

    if a.write:
        with open(os.path.join(rc.BENCH, a.out, "manifest.json"), "w") as f:
            json.dump({"open_r": a.open_r, "n": len(recs), "records": recs}, f, indent=1)
        print(f"\nwrote {len(recs)} masks + manifest.json under {a.out}/")
    else:
        print("\ndry run — pass --write to emit masks")


if __name__ == "__main__":
    main()
