"""Evaluation layer: score every solved shadow with the Ivy judge.

Adds "ivy_judge" = P(Ivy marks it good) under the full reader (fastshadow/ivy_reader.pt)
to results/fastshadow/{objective_study, objective_study_adam}/graded.jsonl and to
results/fastshadow/arrangement_study/runs.jsonl. Run it after `objective_study.py
grade` (which rewrites graded.jsonl) and after retraining the reader.

The judge is only for evaluation and selection. Objectives that optimize the same
reader ("ivy", "ivy_mix") score themselves here, so their judge score is circular;
the reports mark it.

    python scripts/ivy_judge.py
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastshadow  # noqa: E402

BENCH = fastshadow.BENCH
sys.path.insert(0, os.path.join(BENCH, "scripts"))
from run_base_optimizer import load_target  # noqa: E402

FS = os.path.join(BENCH, "results", "fastshadow")
ROOT = os.path.join(BENCH, "optimized", "big-budget-grounded")


def score_file(path, folder, judge, Target):
    import torch

    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    groups = defaultdict(list)
    for k, r in enumerate(rows):
        groups[r["id"]].append(k)
    for gid, ks in groups.items():
        sub, stem = gid.split("/")
        t = Target(load_target(os.path.join(ROOT, sub, stem, f"{stem}_shown.png"), 128), sub, stem.split("_")[0])
        for s in range(0, len(ks), 256):
            chunk = ks[s:s + 256]
            S = torch.tensor(np.stack([load_target(os.path.join(folder, rows[k]["mask"]), 128) for k in chunk]), device="cuda")
            p = judge(S, t).cpu().numpy()
            for k, v in zip(chunk, p):
                rows[k]["ivy_judge"] = round(float(v), 5)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def main():
    from fastshadow.ivy_reader import judge
    from fastshadow.objectives import Target

    for d, name in (("objective_study", "graded.jsonl"), ("objective_study_adam", "graded.jsonl"),
                    ("arrangement_study", "runs.jsonl")):
        p = os.path.join(FS, d, name)
        if os.path.exists(p):
            print(d, score_file(p, os.path.join(FS, d), judge, Target), "rows scored")


if __name__ == "__main__":
    main()
