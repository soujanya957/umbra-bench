#!/usr/bin/env python3
"""
backfill_shadows.py -- link the optimizer solutions into metadata.jsonl.

need-to-collect.md 0.1: metadata.jsonl advertises a three-source design
(hand / teleop / optimizer) and ships every path null, so anyone loading the
benchmark by its documented interface sees an empty dataset even though
optimized/<run>/ holds 546 solved shadows.

This fills `shadows.optimizer` and `rig` from a solved run, and records the
target-fit transform that solution was scored against -- WITHOUT it, the IoU in
summary.csv is not reproducible, because it is measured against a target that
has been rescaled and translated (mean shift 23% of the frame).

  python scripts/backfill_shadows.py --run big-budget-fitted [--apply]

Default is a dry run.
"""
import argparse, csv, json, os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="big-budget-fitted")
    ap.add_argument("--apply", action="store_true", help="write; default is dry-run")
    a = ap.parse_args()

    run_dir = os.path.join(ROOT, "optimized", a.run)
    summary = os.path.join(run_dir, "summary.csv")
    if not os.path.exists(summary):
        sys.exit(f"no such run: {summary}")

    by_id, missing_png = {}, []
    for r in csv.DictReader(open(summary)):
        rel = os.path.join("optimized", a.run, r["best_shadow_png"])
        if not os.path.exists(os.path.join(ROOT, rel)):
            missing_png.append(r["shadow_name"])
            continue
        stem = os.path.basename(os.path.dirname(r["best_shadow_png"]))
        sid = f"{r['subset']}_{stem}"
        rj = os.path.join(run_dir, os.path.dirname(r["best_shadow_png"]), "results.json")
        det = json.load(open(rj)) if os.path.exists(rj) else {}
        by_id[sid] = dict(row=r, rel=rel, det=det)

    rows = [json.loads(l) for l in open(os.path.join(ROOT, "metadata.jsonl")) if l.strip()]
    hit = miss = 0
    for rec in rows:
        e = by_id.get(rec["id"])
        if e is None:
            miss += 1
            continue
        hit += 1
        r, det = e["row"], e["det"]
        rig = det.get("rig", {})
        fit = det.get("fit", {})
        rec["shadows"]["optimizer"] = {
            "path": e["rel"],
            "captured_at": None,          # simulated, not captured
            "run_id": a.run,
            "config": {
                "method": det.get("method"),
                "n_runs": int(r["n_runs"]),
                "best_run": int(r["best_run"]),
                "seconds": float(r["seconds"]),
                "optimizer": det.get("optimizer"),
            },
            # The IoU in summary.csv is measured against the target AFTER this
            # transform. Reproducing it requires applying the transform first.
            "scored_against": {
                "target_transform": {
                    "scale": fit.get("scale"), "dx": fit.get("dx"),
                    "dy": fit.get("dy"), "rot": fit.get("rot"),
                    "frame_px": rig.get("render_size"),
                },
                "iou_vs_transformed": float(r["best_iou"]),
                "iou_vs_original": float(r["best_iou_vs_original"]),
                "mean_iou_vs_transformed": float(r["mean_iou"]),
                "std_iou": float(r["std_iou"]),
            },
            "notes": "simulated shadow; best of n_runs seeds",
        }
        rec["rig"] = {
            "light": {"to_front_m": rig.get("light_to_front_m"),
                      "back_to_wall_m": rig.get("back_to_wall_m")},
            "screen_distance_m": rig.get("back_to_wall_m"),
            "camera": None,
            "n_robots": rig.get("n_robots"),
            "arm_model": "SO-101",
            "arm_gap_m": rig.get("arm_gap_m"),
            "render_size": rig.get("render_size"),
        }

    print(f"run={a.run}  matched {hit}/{len(rows)}  unmatched {miss}  "
          f"missing png {len(missing_png)}")
    if miss:
        print("  unmatched ids:", [r["id"] for r in rows
                                   if r["id"] not in by_id][:8])
    if not a.apply:
        print("dry run; pass --apply to write metadata.jsonl")
        return 0

    meta = os.path.join(ROOT, "metadata.jsonl")
    shutil.copy(meta, meta + ".bak")
    with open(meta, "w") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {meta}  (backup at {meta}.bak)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
