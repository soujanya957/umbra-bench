#!/usr/bin/env python3
"""Flatten <dir>/<budget>-n<N>/**/results.json into one CSV.

usage: collect_runs.py <results_dir> <out.csv>
"""
import csv, json, os, sys

BASE, OUT = sys.argv[1], sys.argv[2]
FIELDS = ["condition", "budget", "n_robots", "id", "subset", "best_iou",
          "best_iou_vs_original", "mean_iou", "mean_iou_vs_original",
          "std_iou", "min_iou", "n_runs", "seconds"]

rows = []
for cond in sorted(os.listdir(BASE)):
    cdir = os.path.join(BASE, cond)
    if not os.path.isdir(cdir) or "-n" not in cond:
        continue
    budget, _, n = cond.partition("-n")
    for dirpath, _, files in os.walk(cdir):
        if "results.json" not in files:
            continue
        d = json.load(open(os.path.join(dirpath, "results.json")))
        rows.append({
            "condition": cond, "budget": budget, "n_robots": int(n),
            "id": d.get("id"), "subset": d.get("subset"),
            "best_iou": d.get("best_iou"),
            "best_iou_vs_original": d.get("best_iou_vs_original"),
            "mean_iou": d.get("mean_iou"),
            "mean_iou_vs_original": d.get("mean_iou_vs_original"),
            "std_iou": d.get("std_iou"), "min_iou": d.get("min_iou"),
            "n_runs": d.get("n_runs"), "seconds": d.get("seconds"),
        })

rows.sort(key=lambda r: (r["budget"], r["n_robots"], r["subset"] or "", r["id"] or ""))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> {OUT}")
for c in sorted({r["condition"] for r in rows}):
    sub = [r for r in rows if r["condition"] == c]
    print(f"  {c:10} n={len(sub):3}  mean best_iou {sum(r['best_iou'] for r in sub)/len(sub):.4f}")
