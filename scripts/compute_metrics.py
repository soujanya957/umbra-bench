#!/usr/bin/env python3
"""Compute every geometric/topological metric + attribute delta for a sweep.

Reads an optimizer sweep laid out as `<results>/<subset>/<stem>/<stem>_best.png`
(what `run_base_optimizer.py` writes) or the dataset's own
`shadows/<sample_id>/{hand,teleop,optimizer}.png`, and writes one wide CSV to
`results/`. That CSV is the input to the exploratory analysis -- one row per
(sample, source), with the target's attributes, the shadow's attributes, their
deltas, and every pairwise metric side by side, so any question about which metric
tracks which shape property is a groupby away.

Nothing is written back into the dataset: `metadata.jsonl` holds attributes of
*targets* only, and derived measurements live in `results/` per DATASET.md.

    python scripts/compute_metrics.py --results optimized/base-optimizer/big-budget
    python scripts/compute_metrics.py --shadows --sources hand teleop optimizer
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from metrics import load_mask, all_metrics  # noqa: E402
from shape_attributes import compute_attributes, attribute_delta  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)


def _one(job):
    """(row_stub, target_path, shadow_path, size, with_alignment) -> flat dict."""
    stub, tp, sp, size, align = job
    try:
        T = load_mask(tp, size)
        S = load_mask(sp, size)
        ta = compute_attributes(T)
        sa = compute_attributes(S)
        row = dict(stub)
        row.update(all_metrics(T, S, with_alignment=align))
        row.update({f"t_{k}": v for k, v in ta.items()})
        row.update({f"s_{k}": v for k, v in sa.items()})
        row.update(attribute_delta(ta, sa))
        return row
    except Exception as e:  # a single unreadable capture must not kill the sweep
        return dict(stub, error=f"{type(e).__name__}: {e}")


def jobs_from_sweep(bench, results, size, align, targets_dir="targets"):
    """One job per (sample, reference).

    `--fit-target` sweeps solve for a *placed* target -- rescaled and shifted into
    the rig's reachable region -- and store it as `<stem>_shown.png`. Scoring those
    runs only against the original target conflates two different failures, and the
    gap is enormous: `digits/0_dejavusans-bold` in `big-budget-fitted` reports
    best_iou 0.813 against what it was actually asked to cast and 0.220 against the
    original. So both references are emitted, tagged by `ref`:

      ref=shown     did the optimizer solve the problem it was given
      ref=original  how much of the requested shape survived, fit included

    Unfitted sweeps have no `_shown.png` and emit ref=original only.
    """
    for sub in sorted(os.listdir(results)):
        sd = os.path.join(results, sub)
        if not os.path.isdir(sd) or sub in ("gallery", "sheets", "best-runs",
                                            "reels", "phrases"):
            continue
        for stem in sorted(os.listdir(sd)):
            shadow = os.path.join(sd, stem, f"{stem}_best.png")
            target = os.path.join(bench, targets_dir, sub, f"{stem}.png")
            shown = os.path.join(sd, stem, f"{stem}_shown.png")
            rj = os.path.join(sd, stem, "results.json")
            if not (os.path.exists(shadow) and os.path.exists(target)):
                continue
            stub = {"sample_id": f"{sub}_{stem}", "subset": sub, "stem": stem,
                    "source": "optimizer"}
            if os.path.exists(rj):
                with open(rj) as f:
                    r = json.load(f)
                fit = r.get("fit") or {}
                stub.update({"best_iou_reported": r.get("best_iou"),
                             "mean_iou_reported": r.get("mean_iou"),
                             "std_iou_reported": r.get("std_iou"),
                             "best_iou_vs_original_reported": r.get("best_iou_vs_original"),
                             "fit_scale": fit.get("scale", ""),
                             "fit_dx": fit.get("dx", ""),
                             "fit_dy": fit.get("dy", ""),
                             "uncastable_before": fit.get("uncastable_before", ""),
                             "uncastable_after": fit.get("uncastable_after", ""),
                             "seconds": r.get("seconds")})
            refs = [("original", target)]
            if os.path.exists(shown):
                refs.append(("shown", shown))
            for tag, ref in refs:
                yield (dict(stub, ref=tag), ref, shadow, size, align)


def jobs_from_shadows(bench, sources, size, align):
    meta = os.path.join(bench, "metadata.jsonl")
    with open(meta) as f:
        for line in f:
            r = json.loads(line)
            for src in sources:
                p = (r.get("shadows", {}).get(src) or {}).get("path")
                if not p:
                    continue
                target = os.path.join(bench, r["target"])
                shadow = os.path.join(bench, p)
                if not (os.path.exists(shadow) and os.path.exists(target)):
                    continue
                yield ({"sample_id": r["id"], "subset": r["subset"],
                        "stem": os.path.splitext(os.path.basename(r["target"]))[0],
                        "source": src, "ref": "original", "class": r.get("class")},
                       target, shadow, size, align)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default=_BENCH)
    p.add_argument("--results", default=os.path.join(
        _BENCH, "optimized", "base-optimizer", "big-budget"),
        help="optimizer sweep root (<subset>/<stem>/<stem>_best.png)")
    p.add_argument("--targets-dir", default="targets",
                   help="target tree the sweep was solved against; must match the "
                        "run's --targets-dir or ref=original compares to the wrong "
                        "shape (see targets_dir in the sweep's BUDGET.md)")
    p.add_argument("--shadows", action="store_true",
                   help="read captures from metadata.jsonl instead of a sweep")
    p.add_argument("--sources", nargs="+", default=["hand", "teleop", "optimizer"])
    p.add_argument("--size", type=int, default=128,
                   help="working resolution; 128 = the optimizer's render size")
    p.add_argument("--align", action="store_true",
                   help="also search scale+shift for aligned_iou (~40x slower)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", default=None)
    p.add_argument("--tag", default=None, help="output filename tag")
    a = p.parse_args()

    jobs = list(jobs_from_shadows(a.bench, a.sources, a.size, a.align) if a.shadows
                else jobs_from_sweep(a.bench, a.results, a.size, a.align,
                                     a.targets_dir))
    if not jobs:
        print("[!] nothing to do")
        return
    print(f"[metrics] {len(jobs)} pairs, {a.workers} workers, size {a.size}"
          f"{', with alignment' if a.align else ''}")

    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, row in enumerate(ex.map(_one, jobs, chunksize=4), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    bad = [r for r in rows if "error" in r]
    if bad:
        print(f"[!] {len(bad)} failed, e.g. {bad[0]['sample_id']}: {bad[0]['error']}")

    out_dir = a.out or os.path.join(a.bench, "results")
    os.makedirs(out_dir, exist_ok=True)
    tag = a.tag or ("shadows" if a.shadows else os.path.basename(a.results.rstrip("/")))
    path = os.path.join(out_dir, f"metrics_{tag}.csv")

    cols = []
    for r in rows:  # union of keys, first-seen order
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[metrics] -> {path}  ({len(rows)} rows x {len(cols)} cols)")

    ok = [r for r in rows if "error" not in r]
    for ref in sorted({r.get("ref", "original") for r in ok}):
        sel = [r for r in ok if r.get("ref", "original") == ref]
        print(f"  [ref={ref}]  n={len(sel)}")
        for k in ("iou", "boundary_iou", "nsd", "cldice", "betti_error", "pw_h1"):
            v = np.array([r[k] for r in sel if r.get(k) is not None], dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                print(f"    {k:<14} mean {v.mean():.4f}  median {np.median(v):.4f}")


if __name__ == "__main__":
    main()
