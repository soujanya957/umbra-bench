#!/usr/bin/env python3
"""Build the per-target browser payload: thumbnails + metrics, one JSON.

Reads whatever is on disk at the time it runs -- the target tree named by
--targets-dir, the sweeps named by --big/--small, and results/master_table.csv --
so re-running it is how the dashboard picks up new targets or a re-solved sweep.
Nothing about the sweep set is baked in any more; the two names below are only
defaults.

    python scripts/_build_browser_payload.py                 # grounded, 128px
    python scripts/_build_browser_payload.py --list          # what is available
    python scripts/_build_browser_payload.py --targets-dir targets \\
        --big big-budget-fitted --small small-budget-fitted  # the centred view

`big` and `small` stay the payload's two sweep keys because the atlas template
hardcodes those literals in eight places (the delta sort at 549-550, the
benchmark view's `bm` at 1003, the pcard at 1104-1105, the default at 536). They
are slot names, not claims about budget -- what each slot points at is the
--big/--small argument, recorded in the payload as `sweep_dirs` so a reader can
tell which run they are looking at.
"""
import argparse
import base64
import io
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTIMIZED = os.path.join(BENCH, "optimized")

# The four-metric panel (METRICS.md): area / outline / topology / pose-invariant shape.
# iou,nsd higher-better; pw_h1,hu_distance lower-better. betti_error deliberately excluded.
METRICS = ["iou", "nsd", "pw_h1", "hu_distance", "boundary_iou", "cldice", "chamfer",
           # position- and size-free: the only pairwise number on a fitted sweep
           # that is about the shape rather than about the fit (metrics.py).
           "tc_iou", "tc_aspect_error"]
ATTRS = ["area_frac", "solidity", "compactness", "median_stroke_width_rel",
         "min_stroke_width_rel", "thin_mass_frac", "elongation", "n_holes", "n_holes_signif",
         "n_components", "n_limbs", "aspect_ratio"]


def discover_sweeps() -> list[str]:
    """Sweep directory names under optimized/ that actually hold solves."""
    out = []
    for name in sorted(os.listdir(OPTIMIZED)) if os.path.isdir(OPTIMIZED) else []:
        d = os.path.join(OPTIMIZED, name)
        if not os.path.isdir(d):
            continue
        for sub in os.listdir(d):
            sd = os.path.join(d, sub)
            if sub in ("gallery", "sheets", "best-runs") or not os.path.isdir(sd):
                continue
            if any(os.path.exists(os.path.join(sd, s, "results.json"))
                   for s in os.listdir(sd)[:5] if os.path.isdir(os.path.join(sd, s))):
                out.append(name)
                break
    return out


def _enc(im):
    b = io.BytesIO()
    im.convert("1").save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode("ascii")


def thumb(path, px):
    """1-bit mask -> compact base64 PNG. Source: dark = shape."""
    im = Image.open(path).convert("L").resize((px, px), Image.LANCZOS)
    return _enc(im.point(lambda v: 255 if v > 127 else 0))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--px", type=int, default=128)
    ap.add_argument("--targets-dir", default="targets_grounded",
                    help="target tree the thumbnails come from. Must be the tree "
                         "the sweeps were solved against, or target and shadow "
                         "in the overlay are different shapes.")
    ap.add_argument("--big", default="big-budget-grounded",
                    help="sweep directory under optimized/ for the `big` slot")
    ap.add_argument("--small", default="small-budget-grounded",
                    help="sweep directory under optimized/ for the `small` slot")
    ap.add_argument("--ref", default="shown", choices=("shown", "original"),
                    help="which comparison the card metrics report. `shown` is "
                         "the shadow against the shape the optimizer was given, "
                         "i.e. after --fit-target has scaled and shifted it -- "
                         "the aligned comparison, and the frame every plate in "
                         "the atlas draws. `original` is against the target as "
                         "authored, which folds the fit's scale and shift in as "
                         "error and is the end-to-end number; it will not line "
                         "up with the pictures.")
    ap.add_argument("--clip-dir", default=os.path.join("results", "clip_dataset"),
                    help="directory holding clip_per_image.csv from "
                         "scripts/clip_eval_dataset.py. Absent is fine -- the "
                         "recognizability metric is simply omitted.")
    ap.add_argument("--out", default=os.path.join(BENCH, "results", "browser_payload.json"))
    ap.add_argument("--list", action="store_true",
                    help="show the sweeps and target trees on disk, then exit")
    a = ap.parse_args()

    if a.list:
        print("sweeps under optimized/:")
        for s in discover_sweeps():
            n = sum(1 for _ in _iter_results(os.path.join(OPTIMIZED, s)))
            print(f"  {s:<26} {n:>4} solved")
        print("\ntarget trees:")
        for d in sorted(os.listdir(BENCH)):
            if d.startswith("targets") and os.path.isdir(os.path.join(BENCH, d)):
                n = sum(len([f for f in fs if f.endswith('.png')])
                        for _, _, fs in os.walk(os.path.join(BENCH, d)))
                print(f"  {d:<26} {n:>4} png")
        return

    sweeps = {"big": a.big, "small": a.small}
    for slot, name in sweeps.items():
        if not os.path.isdir(os.path.join(OPTIMIZED, name)):
            raise SystemExit(f"--{slot}: no such sweep directory: optimized/{name}\n"
                             f"available: {', '.join(discover_sweeps()) or '(none)'}")

    meta = {}
    with open(os.path.join(BENCH, "metadata.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            meta[d["id"]] = d

    master = os.path.join(BENCH, "results", "master_table.csv")
    if not os.path.exists(master):
        raise SystemExit(f"missing {master}\n"
                         "run scripts/make_master_table.py (which reads the "
                         "metrics_*.csv that compute_metrics.py writes).")
    df = pd.read_csv(master, low_memory=False)
    df = df[df["ref"] == a.ref]
    rows = {slot: df[df["sweep"] == name].set_index("sample_id")
            for slot, name in sweeps.items()}
    for slot, name in sweeps.items():
        if rows[slot].empty:
            print(f"  ! no rows in master_table for sweep={name!r} -- "
                  f"the `{slot}` slot will have metrics missing")

    # CLIP recognizability, if it has been run. Keyed by (id, condition) where the
    # condition is the sweep slot, plus `target` for the ceiling. Carried as
    # `clip_rr` = 1/rank: continuous, higher-better, and its mean over a set is
    # exactly the MRR that clip_eval_dataset.py reports, so the card and the
    # summary table cannot disagree. Optional -- the dashboard predates it.
    clip, clip_summary = {}, []
    clip_csv = os.path.join(BENCH, a.clip_dir, "clip_per_image.csv")
    if os.path.exists(clip_csv):
        cdf = pd.read_csv(clip_csv)
        for r in cdf.itertuples(index=False):
            clip[(r.id, r.condition)] = {
                "clip_rr": round(1.0 / int(r.rank), 4),
                "clip_rank": int(r.rank),
                "clip_top1": int(r.top1),
                # 0/1 per item, so the readout's mean over a selection is the
                # "correctly identified" ratio for exactly what is on screen.
                "clip_top5": int(int(r.rank) <= 5),
                "clip_n": int(r.n_classes),
            }
        print(f"  clip: {len(cdf)} scored images from {a.clip_dir}")
        # The subset-level table too. top1 and mrr are both carried because they
        # disagree in a way that matters: hand_shadow is 0.000 by top1, since no
        # shadow ever ranks first, and 0.407 by MRR, since they rank respectably
        # and just never win. One column alone misreports that subset either way.
        csum = os.path.join(BENCH, a.clip_dir, "clip_summary.csv")
        if os.path.exists(csum):
            keep = ["subset", "condition", "n", "n_classes", "chance_top1", "top1",
                    "top5", "mrr", "top1_over_chance", "ratio_vs_target",
                    "mrr_ratio_vs_target"]
            sdf = pd.read_csv(csum)
            clip_summary = [
                {k: (None if pd.isna(v) else (float(v) if isinstance(v, float) else v))
                 for k, v in r.items() if k in keep}
                for r in sdf.to_dict("records")]
    else:
        print(f"  clip: {a.clip_dir}/clip_per_image.csv absent -- "
              f"run scripts/clip_eval_dataset.py to add the recognizability metric")

    samples, missing_target, missing_shadow = [], [], {k: 0 for k in sweeps}
    for sid, d in meta.items():
        # The thumbnail must come from the tree the sweep solved, not from the
        # `target` recorded in metadata.jsonl (which is always the authored tree).
        rel = d["target"].replace("\\", "/")
        stem = os.path.splitext(os.path.basename(rel))[0]
        tpath = os.path.join(BENCH, a.targets_dir, d["subset"], f"{stem}.png")
        if not os.path.exists(tpath):
            missing_target.append(sid)
            continue
        rec = {"id": sid, "subset": d["subset"], "cls": d.get("class", ""),
               "t": thumb(tpath, a.px),
               "a": {k: d["attributes"].get(k) for k in ATTRS}}
        ct = clip.get((sid, "target"))
        if ct:
            # The ceiling. A shadow scoring badly means nothing until you know
            # whether CLIP could read the target at all.
            rec["clip_t_rr"] = ct["clip_rr"]
            rec["clip_t_rank"] = ct["clip_rank"]
            rec["clip_n"] = ct["clip_n"]
        for slot, name in sweeps.items():
            sp = os.path.join(OPTIMIZED, name, d["subset"], stem, f"{stem}_best.png")
            if not os.path.exists(sp):
                missing_shadow[slot] += 1
                continue
            e = {"s": thumb(sp, a.px)}
            # `_shown.png` is the target AFTER --fit-target scaled and shifted it
            # -- the shape the optimizer was actually asked to cast. Overlaying
            # the shadow on the authored target instead compares against
            # something nobody solved for: the fit is scale 0.83 and ~14px of
            # shift on average, so every card looks mis-registered. Carry both,
            # and let the plate choose which comparison it is making.
            wp = os.path.join(OPTIMIZED, name, d["subset"], stem, f"{stem}_shown.png")
            if os.path.exists(wp):
                e["w"] = thumb(wp, a.px)
            if sid in rows[slot].index:
                r = rows[slot].loc[sid]
                if isinstance(r, pd.DataFrame):        # duplicate sample_id
                    r = r.iloc[0]
                for m in METRICS:
                    v = r.get(m)
                    e[m] = None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)
                for k, c in (("best", "best_iou_reported"), ("mean", "mean_iou_reported"),
                             ("std", "std_iou_reported"), ("sec", "seconds")):
                    v = r.get(c)
                    e[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)
            cs = clip.get((sid, slot))
            if cs:
                e["clip_rr"] = cs["clip_rr"]
                e["clip_rank"] = cs["clip_rank"]
                e["clip_top1"] = cs["clip_top1"]
                e["clip_top5"] = cs["clip_top5"]
                if ct and ct["clip_rr"]:
                    e["clip_ratio"] = round(cs["clip_rr"] / ct["clip_rr"], 4)
            rec[slot] = e
        samples.append(rec)

    payload = {"px": a.px, "n": len(samples), "metrics": METRICS, "attrs": ATTRS,
               "targets_dir": a.targets_dir, "sweep_dirs": sweeps, "ref": a.ref,
               "clip_summary": clip_summary,
               "subsets": sorted({s["subset"] for s in samples}), "samples": samples}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"samples={len(samples)}  px={a.px}  targets={a.targets_dir}  ref={a.ref}")
    print(f"  subsets: {', '.join(payload['subsets'])}")
    for slot, name in sweeps.items():
        print(f"  {slot:<6} -> {name:<26} missing shadow: {missing_shadow[slot]}")
    if missing_target:
        print(f"  ! {len(missing_target)} targets absent from {a.targets_dir}/ "
              f"(e.g. {', '.join(missing_target[:3])})")
    print(f"bytes={os.path.getsize(a.out)/1e6:.2f} MB -> {a.out}")


def _iter_results(root):
    for sub in os.listdir(root):
        sd = os.path.join(root, sub)
        if sub in ("gallery", "sheets", "best-runs") or not os.path.isdir(sd):
            continue
        for stem in os.listdir(sd):
            if os.path.exists(os.path.join(sd, stem, "results.json")):
                yield sub, stem


if __name__ == "__main__":
    main()
