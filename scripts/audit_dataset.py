#!/usr/bin/env python3
"""
audit_dataset.py -- release gate for umbra-bench.

Everything a stranger cloning this repo could trip over, checked before they do.
Read-only: writes a report, never touches the dataset.

  A. metadata <-> filesystem agreement
  B. image contract (size, mode, polarity, binarity, emptiness, margin)
  C. attributes recomputed from pixels and compared to what is recorded
  D. duplicates -- exact and near, WITHIN and ACROSS subsets
  E. provenance and licensing coverage
  F. the documented loader interface, exercised the way a user would
  G. reference results (what a submission is compared against)
  H. the `uncastable_*` columns vs achieved IoU  (need-to-collect.md 0.2)

Exit code is the number of ERROR-level findings, so CI can gate on it.

  python scripts/audit_dataset.py [--root .] [--out docs/AUDIT.md]
"""
import argparse, collections, hashlib, json, os, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from shape_attributes import compute_attributes as _ds_attrs
except Exception as _e:          # cv2/skimage absent -> skip section C
    _ds_attrs = None
    print(f"  [note ] shape_attributes unavailable ({_e}); skipping attribute check")

ERRORS, WARNS, NOTES = [], [], []
def err(s):  ERRORS.append(s); print(f"  [ERROR] {s}")
def warn(s): WARNS.append(s);  print(f"  [WARN ] {s}")
def note(s): NOTES.append(s);  print(f"  [note ] {s}")

# The contract stated in DATASET.md.
EXP_SIZE = (512, 512)
EXP_MARGIN_FRAC = 0.10


# ── loading ───────────────────────────────────────────────────────────────────
def load_metadata(root):
    rows, bad = [], 0
    with open(os.path.join(root, "metadata.jsonl")) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                bad += 1
                err(f"metadata.jsonl line {i} is not valid JSON: {e}")
    return rows, bad


def as_mask(path):
    """Canonical boolean mask: True = shape. Targets are black-on-white."""
    im = Image.open(path)
    a = np.asarray(im.convert("L"))
    return a < 128, im, a


# ── A. metadata <-> filesystem ────────────────────────────────────────────────
def check_index(root, rows):
    print("\nA. metadata <-> filesystem")
    ids = [r["id"] for r in rows]
    dupe_ids = [k for k, v in collections.Counter(ids).items() if v > 1]
    if dupe_ids:
        err(f"{len(dupe_ids)} duplicate ids in metadata.jsonl: {dupe_ids[:5]}")

    listed = set()
    for r in rows:
        p = os.path.join(root, r["target"])
        if not os.path.exists(p):
            err(f"{r['id']}: target missing on disk -> {r['target']}")
        else:
            listed.add(os.path.relpath(p, root))
        # id should be derivable from subset + filename
        stem = os.path.splitext(os.path.basename(r["target"]))[0]
        want = f"{r['subset']}_{stem}"
        if r["id"] != want:
            warn(f"{r['id']}: id does not match subset_stem ({want})")

    on_disk = set()
    tdir = os.path.join(root, "targets")
    for dp, _, fns in os.walk(tdir):
        for fn in fns:
            if fn.lower().endswith(".png"):
                on_disk.add(os.path.relpath(os.path.join(dp, fn), root))
    orphans = sorted(on_disk - listed)
    if orphans:
        err(f"{len(orphans)} PNGs under targets/ are NOT in metadata.jsonl "
            f"(a user walking the directory sees more data than the index "
            f"admits): {orphans[:8]}")
    note(f"{len(rows)} metadata rows, {len(on_disk)} PNGs under targets/")

    by_subset = collections.Counter(r["subset"] for r in rows)
    note("per-subset counts: " + ", ".join(f"{k}={v}" for k, v in sorted(by_subset.items())))
    return orphans


# ── B + C. image contract and attributes ──────────────────────────────────────
def _attrs(mask):
    """Recompute the subset of `attributes` that is cheap and unambiguous."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    inv, nh = ndimage.label(~mask)
    # holes = background components not touching the border
    border = set(inv[0, :]) | set(inv[-1, :]) | set(inv[:, 0]) | set(inv[:, -1])
    holes = len([i for i in range(1, nh + 1) if i not in border])
    area = int(mask.sum())
    out = {"n_components": int(n), "n_holes": holes,
           "area_frac": round(area / mask.size, 4)}
    ys, xs = np.nonzero(mask)
    if len(ys):
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        out["aspect_ratio"] = round(min(w, h) / max(w, h), 4)
        out["_bbox"] = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
    return out


def check_images(root, rows, sample=None):
    print("\nB. image contract   C. recorded attributes vs pixels")
    sizes, modes = collections.Counter(), collections.Counter()
    n_nonbinary = n_empty = n_polarity = n_margin = 0
    attr_mismatch = collections.Counter()
    subset = rows if sample is None else rows[:sample]

    for r in subset:
        p = os.path.join(root, r["target"])
        if not os.path.exists(p):
            continue
        mask, im, a = as_mask(p)
        sizes[im.size] += 1
        modes[im.mode] += 1

        vals = np.unique(a)
        if len(vals) > 2:
            n_nonbinary += 1
            if n_nonbinary <= 3:
                warn(f"{r['id']}: {len(vals)} grey levels, not a binary mask "
                     f"(min={vals.min()} max={vals.max()})")
        if mask.sum() == 0:
            n_empty += 1
            err(f"{r['id']}: target mask is empty")
            continue
        got = _attrs(mask)
        rec_af = r.get("attributes", {}).get("area_frac")
        border_fg = float(np.mean(np.concatenate(
            [mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])))
        if rec_af is not None and abs(got["area_frac"] - (1.0 - rec_af)) < 0.005 \
                and abs(got["area_frac"] - rec_af) > 0.02:
            n_polarity += 1
            err(f"{r['id']}: pixels are the INVERSE of the recorded area_frac "
                f"({got['area_frac']:.3f} vs {rec_af:.3f}) -- mask polarity flipped "
                f"since metadata was built")
        elif border_fg > 0.5:
            n_polarity += 1
            err(f"{r['id']}: {border_fg:.0%} of the image border is 'shape' -- "
                f"polarity is probably inverted")
        if mask.mean() > 0.45:
            note(f"{r['id']}: fills {mask.mean():.0%} of the frame "
                 f"(near-degenerate target for a silhouette benchmark)")

        bb = got.pop("_bbox", None)
        if bb is not None:
            H, W = mask.shape
            m = min(bb[0], bb[1], H - 1 - bb[2], W - 1 - bb[3]) / max(H, W)
            if m < EXP_MARGIN_FRAC * 0.5:
                n_margin += 1

        rec = r.get("attributes", {})
        if _ds_attrs is not None:
            # build_metadata.py: mask = 255 - L(target). Same convention here,
            # so a mismatch means metadata is stale, not that we disagree on
            # a definition.
            fresh = _ds_attrs(255 - a)
            for k, v in fresh.items():
                if k not in rec or rec[k] is None:
                    continue
                ok = (abs(rec[k] - v) <= 1e-3 + 0.02 * abs(v)
                      if isinstance(v, float) else rec[k] == v)
                if not ok:
                    attr_mismatch[k] += 1
                    if attr_mismatch[k] <= 2:
                        warn(f"{r['id']}: attributes.{k} recorded {rec[k]}, "
                             f"recomputed {v}")

    note(f"sizes: {dict(sizes)}")
    note(f"PIL modes: {dict(modes)}")
    if set(sizes) != {EXP_SIZE}:
        err(f"DATASET.md says targets are {EXP_SIZE[0]}x{EXP_SIZE[1]}; "
            f"found {dict(sizes)}")
    if set(modes) - {"1"}:
        warn(f"DATASET.md says 1-bit PNG; found modes {dict(modes)}")
    if n_nonbinary:
        warn(f"{n_nonbinary}/{len(subset)} targets are not 2-valued")
    if n_margin:
        warn(f"{n_margin}/{len(subset)} targets sit closer to the border than "
             f"half the documented {EXP_MARGIN_FRAC:.0%} margin")
    if attr_mismatch:
        err(f"attributes disagree with pixels: {dict(attr_mismatch)} "
            f"(metadata is stale relative to targets/)")
    else:
        note("recomputed attributes agree with metadata")


# ── D. duplicates ─────────────────────────────────────────────────────────────
def _phash(mask, n=16):
    """Downsample-and-threshold hash. Coarse on purpose: near-duplicate silhouettes,
    not pixel-identical files, are what break a public benchmark."""
    m = mask.astype(np.float32)
    H, W = m.shape
    ph = m.reshape(n, H // n, n, W // n).mean(axis=(1, 3))
    return (ph > ph.mean()).flatten()


def check_duplicates(root, rows):
    print("\nD. duplicates")
    sha, ph, ids = {}, [], []
    for r in rows:
        p = os.path.join(root, r["target"])
        if not os.path.exists(p):
            continue
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        sha.setdefault(h, []).append(r["id"])
        mask, _, _ = as_mask(p)
        ph.append(_phash(mask))
        ids.append((r["id"], r["subset"], r.get("class")))

    exact = {h: v for h, v in sha.items() if len(v) > 1}
    if exact:
        err(f"{len(exact)} groups of BYTE-IDENTICAL targets: "
            + "; ".join(" == ".join(v) for v in list(exact.values())[:5]))
    else:
        note("no byte-identical targets")

    P = np.array(ph)
    if len(P) < 2:
        return []
    # Hamming distance over the 256-bit hash, upper triangle only.
    D = (P[:, None, :] != P[None, :, :]).sum(axis=2)
    iu = np.triu_indices(len(P), k=1)
    d = D[iu]
    THRESH = 8  # <=8/256 bits differ
    near = [(ids[i], ids[j], int(D[i, j]))
            for i, j in zip(*iu) if D[i, j] <= THRESH]
    cross = [n for n in near if n[0][1] != n[1][1]]
    note(f"near-duplicate pairs (hamming<={THRESH}/256): {len(near)} total, "
         f"{len(cross)} of them ACROSS subsets")
    if cross:
        err(f"{len(cross)} near-duplicate pairs cross a subset boundary -- a "
            f"split that separates subsets will leak. Examples: "
            + "; ".join(f"{a[0]}~{b[0]}(d={c})" for a, b, c in cross[:5]))
    elif near:
        warn(f"{len(near)} within-subset near-duplicates; examples: "
             + "; ".join(f"{a[0]}~{b[0]}(d={c})" for a, b, c in near[:5]))
    note(f"pairwise silhouette-hash distance: median {np.median(d):.1f}, "
         f"1st percentile {np.percentile(d, 1):.1f} of 256 bits")
    return near


# ── E. provenance ─────────────────────────────────────────────────────────────
def check_provenance(root, rows):
    print("\nE. provenance and licensing")
    missing = [r["id"] for r in rows if not r.get("target_source")]
    if missing:
        err(f"{len(missing)} samples have no target_source")
    srcs = collections.Counter(str(r.get("target_source", "")).split(":")[0]
                               for r in rows)
    note("target_source prefixes: " + ", ".join(f"{k}={v}" for k, v in srcs.most_common()))

    cit = os.path.join(root, "CITATIONS.md")
    if not os.path.exists(cit):
        err("CITATIONS.md is missing; the release ships third-party derivatives")
        return
    text = open(cit).read().lower()
    for pfx in srcs:
        if pfx and pfx not in ("generated",) and pfx.split("-")[0] not in text:
            warn(f"source '{pfx}' is used by {srcs[pfx]} samples but is not "
                 f"mentioned in CITATIONS.md")
    if "licen" not in text:
        warn("CITATIONS.md never says the word 'licence'/'license'")


# ── F. the documented loader ──────────────────────────────────────────────────
def check_loader(root, rows):
    print("\nF. documented loader interface")
    for kind in ("hand", "teleop", "optimizer"):
        have = sum(1 for r in rows
                   if (r.get("shadows", {}).get(kind) or {}).get("path"))
        fn = err if kind == "optimizer" and have == 0 else note
        fn(f"shadows.{kind}: {have}/{len(rows)} populated")
    broken = [r["id"] for r in rows
              for k in ("hand", "teleop", "optimizer")
              if (r.get("shadows", {}).get(k) or {}).get("path")
              and not os.path.exists(os.path.join(root, r["shadows"][k]["path"]))]
    if broken:
        err(f"{len(broken)} shadow paths are recorded but missing on disk")
    rig = sum(1 for r in rows if (r.get("rig") or {}).get("light"))
    if rig == 0:
        warn("rig.light is null on every sample -- a shadow cannot be "
             "reproduced without the rig that cast it")


# ── G + H. reference results, and the uncastable columns ──────────────────────
def check_results(root, rows):
    print("\nG. reference results   H. the uncastable_* columns")
    import csv
    tables = {}
    for name in ("big-budget-fitted", "small-budget-fitted"):
        p = os.path.join(root, "optimized", name, "summary.csv")
        if not os.path.exists(p):
            warn(f"{name}/summary.csv missing")
            continue
        tables[name] = list(csv.DictReader(open(p)))

    for name, t in tables.items():
        iou = np.array([float(r["best_iou"]) for r in t])
        orig = np.array([float(r["best_iou_vs_original"]) for r in t
                         if r.get("best_iou_vs_original")])
        note(f"{name}: n={len(t)}  best_iou mean {iou.mean():.4f} "
             f"median {np.median(iou):.4f} min {iou.min():.4f} max {iou.max():.4f}")
        if len(orig):
            note(f"{name}: IoU vs ORIGINAL (un-fitted) target mean {orig.mean():.4f} "
                 f"-- the fitted number above is {iou.mean() - orig.mean():+.4f} higher; "
                 f"a release must say which one a submission is scored on")
        missing = [r["shadow_name"] for r in t
                   if not os.path.exists(os.path.join(root, "optimized", name,
                                                      r["best_shadow_png"]))]
        if missing:
            err(f"{name}: {len(missing)} best_shadow_png rows point at nothing")

    t = tables.get("big-budget-fitted")
    if not t:
        return
    per = collections.defaultdict(list)
    for r in t:
        per[r["subset"]].append(float(r["best_iou"]))
    note("per-subset best_iou (the leaderboard baseline): " + ", ".join(
        f"{k}={np.mean(v):.3f}(n={len(v)})" for k, v in sorted(per.items())))

    # H -- does `uncastable_after` bound achieved IoU?
    ok = viol = 0
    worst = []
    for r in t:
        try:
            ua = float(r["uncastable_after"]); iou = float(r["best_iou"])
        except (TypeError, ValueError):
            continue
        bound = 1.0 - ua
        if iou > bound + 1e-6:
            viol += 1
            worst.append((iou - bound, r["shadow_name"], iou, bound))
        else:
            ok += 1
    if viol:
        worst.sort(reverse=True)
        err(f"uncastable_after does NOT bound achieved IoU: {viol}/{viol+ok} "
            f"({100*viol/(viol+ok):.1f}%) targets score above 1-uncastable_after. "
            f"Worst: " + "; ".join(f"{n} {i:.3f}>{b:.3f}" for _, n, i, b in worst[:3])
            + ". Document what these columns measure or drop them.")

    if len(tables) == 2:
        big = {r["shadow_name"]: float(r["best_iou"]) for r in tables["big-budget-fitted"]}
        sml = {r["shadow_name"]: float(r["best_iou"]) for r in tables["small-budget-fitted"]}
        common = sorted(set(big) & set(sml))
        d = np.array([big[k] - sml[k] for k in common])
        note(f"big minus small budget over {len(common)} shared targets: "
             f"mean {d.mean():+.4f}, median {np.median(d):+.4f}, "
             f"big wins on {100*(d>0).mean():.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample", type=int, default=None,
                    help="check only the first N images (fast smoke)")
    a = ap.parse_args()

    print(f"umbra-bench audit -- {a.root}")
    rows, _ = load_metadata(a.root)
    check_index(a.root, rows)
    check_images(a.root, rows, a.sample)
    check_duplicates(a.root, rows)
    check_provenance(a.root, rows)
    check_loader(a.root, rows)
    check_results(a.root, rows)

    print(f"\n{'='*70}\n{len(ERRORS)} errors, {len(WARNS)} warnings, {len(NOTES)} notes")
    for s in ERRORS:
        print(f"  ERROR  {s}")
    for s in WARNS:
        print(f"  WARN   {s}")

    if a.out:
        with open(a.out, "w") as f:
            f.write("# umbra-bench audit\n\nGenerated by `scripts/audit_dataset.py`.\n\n")
            for title, items in (("Errors", ERRORS), ("Warnings", WARNS), ("Notes", NOTES)):
                f.write(f"## {title} ({len(items)})\n\n")
                for s in items:
                    f.write(f"- {s}\n")
                f.write("\n")
        print(f"wrote {a.out}")
    return len(ERRORS)


if __name__ == "__main__":
    sys.exit(main())
