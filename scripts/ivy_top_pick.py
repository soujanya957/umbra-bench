"""The Ivy judge's favourite shadow for every letter and digit, from every stored solve.

Candidates, all on the standard 3-arm rig at 128 px:
  objective_study (CMA-ES) and objective_study_adam: every objective and seed, except "ivy" and
      "ivy_mix", which optimized this very judge, and "cls" and "shape3_cls", which optimized the
      glyph classifier whose outputs are among the judge's features (their scores would be circular);
  seeds_dt2 (12 dt2 seeds), floorfix_dt2 (fixed floor check);
  optimized/big-budget-grounded: the benchmark's own solves (10 runs per target, 3 fonts).
Left out: compare_letters_digits and fastcma_letters_digits (no joint angles stored, so the
pose cannot be checked) and the arrangement study (bases moved, a different rig).

A candidate must be physically possible: no arm collision (MuJoCo contacts) and no moving part
more than 5 mm into the table (FastShadowRenderer.moving_min_z). Among those, the highest judge
score wins per character, whatever the font of its target.

    set FLEET_SHADOW_ART=...\\fleet-shadow-art-batched
    python scripts/ivy_top_pick.py
"""

import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastshadow  # noqa: E402

BENCH = fastshadow.BENCH
sys.path.insert(0, os.path.join(BENCH, "scripts"))
from run_base_optimizer import load_target  # noqa: E402

FS = os.path.join(BENCH, "results", "fastshadow")
ROOT = os.path.join(BENCH, "optimized", "big-budget-grounded")
OUT = os.path.join(FS, "ivy_top")
SUBS = ("letters_upper", "letters_lower", "digits")
CIRCULAR = {"ivy", "ivy_mix", "cls", "shape3_cls"}  # steered by the judge, or by the classifier the judge reads
TABLE_MM = -5.0


def candidates():
    out = []
    for d, tk in (("objective_study", "cma"), ("objective_study_adam", "adam"), ("seeds_dt2", "cma"), ("floorfix_dt2", "cma")):
        q = {(r["id"], r["objective"], r["seed"]): r for r in map(json.loads, open(os.path.join(FS, d, "solves.jsonl"), encoding="utf-8"))}
        for r in map(json.loads, open(os.path.join(FS, d, "graded.jsonl"), encoding="utf-8")):
            if r["objective"] in CIRCULAR:
                continue
            s = q.get((r["id"], r["objective"], r["seed"]))
            if s is None:
                continue
            out.append({"id": r["id"], "sub": r["subset"], "label": r["label"], "source": d, "optimizer": tk,
                        "objective": r["objective"], "seed": r["seed"], "mask": os.path.join(FS, d, r["mask"]),
                        "iou": r["iou"], "tc_iou": r.get("tc_iou"), "q": s["q_rad"], "collision_free": s["collision_free"]})
    for sub in SUBS:
        for f in sorted(glob.glob(os.path.join(ROOT, sub, "*", "results.json"))):
            res = json.load(open(f, encoding="utf-8"))
            stem = os.path.basename(os.path.dirname(f))
            for run in res["runs"]:
                m = os.path.join(os.path.dirname(f), f"{stem}_run{run['run']:02d}.png")
                if os.path.exists(m):
                    out.append({"id": f"{sub}/{stem}", "sub": sub, "label": stem.split("_")[0], "source": "benchmark",
                                "optimizer": "cma", "objective": "current", "seed": run["seed"], "mask": m,
                                "iou": run["iou"], "tc_iou": None, "q": run["q_rad"], "collision_free": None})
    return out


def main():
    fastshadow.use_fleet()
    import torch
    from renderer import ShadowRenderer, build_scene

    from fastshadow.ivy_reader import judge
    from fastshadow.objectives import Target
    from fastshadow.renderer import FastShadowRenderer

    C = candidates()
    print(len(C), "candidate shadows", Counter(c["source"] for c in C))
    model, data, n = build_scene("urdf/SO101/so101_new_calib.urdf", n_robots=3, arm_gap=0.2)
    mj = ShadowRenderer(model, data, size=128, n_robots=n)
    fr = FastShadowRenderer(mj)
    Q = np.array([c["q"] for c in C])
    with torch.no_grad():
        low = torch.cat([fr.moving_min_z(Q[s:s + 1024]) for s in range(0, len(Q), 1024)]).cpu().numpy()
    for c, z in zip(C, low):
        c["lowest_mm"] = float(1000 * z)
        if c["collision_free"] is None:  # benchmark runs: MuJoCo contacts, CPU
            mj.set_qpos(np.asarray(c["q"]))
            _, _, cross, self_ = mj.eval_q(np.asarray(c["q"]))
            c["collision_free"] = bool(cross <= 0 and self_ <= 0)
    by_id = defaultdict(list)
    for c in C:
        by_id[c["id"]].append(c)
    for gid, cs in by_id.items():
        sub, stem = gid.split("/")
        t = Target(load_target(os.path.join(ROOT, sub, stem, f"{stem}_shown.png"), 128), sub, stem.split("_")[0])
        for s in range(0, len(cs), 256):
            chunk = cs[s:s + 256]
            S = torch.tensor(np.stack([load_target(c["mask"], 128) for c in chunk]), device="cuda")
            with torch.no_grad():
                p = judge(S, t).cpu().numpy()
            for c, v in zip(chunk, p):
                c["judge"] = float(v)
    by_char = defaultdict(list)
    for c in C:
        by_char[(c["sub"], c["label"])].append(c)
    top, n_infeasible = [], 0
    for (sub, ch), cs in sorted(by_char.items(), key=lambda kv: (SUBS.index(kv[0][0]), kv[0][1])):
        ok = [c for c in cs if c["collision_free"] and c["lowest_mm"] >= TABLE_MM]
        n_infeasible += len(cs) - len(ok)
        best = max(ok, key=lambda c: c["judge"])
        best_iou = max(ok, key=lambda c: c["iou"])
        rec = {k: best[k] for k in ("id", "sub", "label", "source", "optimizer", "objective", "seed", "iou", "tc_iou",
                                    "judge", "lowest_mm")}
        rec["mask"] = os.path.relpath(best["mask"], BENCH)
        rec["n_candidates"], rec["n_feasible"] = len(cs), len(ok)
        rec["iou_pick_judge"], rec["iou_pick_iou"] = best_iou["judge"], best_iou["iou"]
        rec["same_as_iou_pick"] = best is best_iou
        rec["max_judge_any"] = max(c["judge"] for c in cs)  # incl. infeasible, to see what the filter cost
        top.append(rec)
    summ = {"n_candidates": len(C), "n_infeasible": n_infeasible, "by_source": dict(Counter(c["source"] for c in C)),
            "winner_source": dict(Counter(r["source"] for r in top)),
            "winner_optimizer": dict(Counter(r["optimizer"] for r in top)),
            "winner_objective": dict(Counter(r["objective"] for r in top).most_common()),
            "same_as_iou_pick": sum(r["same_as_iou_pick"] for r in top), "n_chars": len(top),
            "mean_judge": float(np.mean([r["judge"] for r in top])), "mean_iou": float(np.mean([r["iou"] for r in top])),
            "mean_iou_of_iou_pick": float(np.mean([r["iou_pick_iou"] for r in top])),
            "mean_judge_of_iou_pick": float(np.mean([r["iou_pick_judge"] for r in top]))}
    os.makedirs(OUT, exist_ok=True)
    json.dump({"summary": summ, "top": top}, open(os.path.join(OUT, "top.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
