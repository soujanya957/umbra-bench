"""Are some solved poses fragile? (plan experiment E11)

Real arms are off by about a degree per joint (visual vs refined calibrations differ by up to
1.5 deg). Each stored solve is re-rendered under random joint errors (sigma 1 and 2 deg, 32
draws, one nvdiffrast batch) and scored against its target: mean and 10th-percentile IoU.
Within a glyph, if seeds differ in robustness by more than they differ in nominal IoU, picking
the most robust solve (same geometric loss, nothing learned) is worth testing on the real rig.

    set FLEET_SHADOW_ART=...\\fleet-shadow-art-batched
    python scripts/robustness_check.py [--study objective_study] [--objective dt2]
"""

import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="objective_study")
    ap.add_argument("--objective", default="dt2")
    ap.add_argument("--draws", type=int, default=32)
    a = ap.parse_args()
    fastshadow.use_fleet()
    import torch
    from renderer import ShadowRenderer, build_scene

    from fastshadow.renderer import FastShadowRenderer

    rows = [json.loads(l) for l in open(os.path.join(FS, a.study, "solves.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r["objective"] == a.objective]
    root = os.path.join(BENCH, "optimized", "big-budget-grounded")
    first = json.load(open(os.path.join(root, *rows[0]["id"].split("/"), "results.json"), encoding="utf-8"))
    rig = first["rig"]
    model, data, n = build_scene("urdf/SO101/so101_new_calib.urdf", n_robots=rig["n_robots"], arm_gap=rig["arm_gap_m"])
    mj = ShadowRenderer(model, data, size=rig["render_size"], n_robots=n)
    fr = FastShadowRenderer(mj)
    g = np.random.default_rng(0)
    out = []
    for r in rows:
        sub, stem = r["id"].split("/")
        T = torch.tensor(load_target(os.path.join(root, sub, stem, f"{stem}_shown.png"), mj.size) > 0.5, device="cuda")
        q = np.array(r["q_rad"])
        rec = {"id": r["id"], "seed": r["seed"]}
        for sd in (1.0, 2.0):
            E = g.normal(0, np.deg2rad(sd), size=(a.draws, len(q)))  # same draws for every pose would be fairer:
            Q = np.vstack([q[None], q[None] + E])                      # but independent draws are what a rig sees
            with torch.no_grad():
                M = fr.render(np.clip(Q, mj.q_lower, mj.q_upper)) > 0.5
            inter = (M & T).flatten(1).sum(1).float()
            uni = (M | T).flatten(1).sum(1).float()
            iou = (inter / uni.clamp(min=1)).cpu().numpy()
            rec["nominal"] = float(iou[0])
            rec[f"mean_{sd:g}"] = float(iou[1:].mean())
            rec[f"p10_{sd:g}"] = float(np.percentile(iou[1:], 10))
        out.append(rec)
    by = defaultdict(list)
    for rec in out:
        by[rec["id"]].append(rec)
    multi = {k: v for k, v in by.items() if len(v) > 1}
    summ = {"n_solves": len(out), "n_glyphs_multi_seed": len(multi)}
    for sd in ("1", "2"):
        drop = np.array([o["nominal"] - o[f"mean_{sd}"] for o in out])
        p10drop = np.array([o["nominal"] - o[f"p10_{sd}"] for o in out])
        spread = [max(o[f"p10_{sd}"] for o in v) - min(o[f"p10_{sd}"] for o in v) for v in multi.values()]
        nom_spread = [max(o["nominal"] for o in v) - min(o["nominal"] for o in v) for v in multi.values()]
        changes = sum(max(v, key=lambda o: o["nominal"]) is not max(v, key=lambda o: o[f"p10_{sd}"]) for v in multi.values())
        # what you lose in nominal IoU and gain in p10 by picking the most robust instead of the best nominal
        gain = [(max(v, key=lambda o: o[f"p10_{sd}"])[f"p10_{sd}"] - max(v, key=lambda o: o["nominal"])[f"p10_{sd}"],
                 max(v, key=lambda o: o["nominal"])["nominal"] - max(v, key=lambda o: o[f"p10_{sd}"])["nominal"])
                for v in multi.values()]
        summ[f"sigma_{sd}deg"] = {
            "mean_iou_drop": float(drop.mean()), "p10_iou_drop": float(p10drop.mean()),
            "worst_p10_drop": float(p10drop.max()),
            "within_glyph_p10_spread_median": float(np.median(spread)),
            "within_glyph_nominal_spread_median": float(np.median(nom_spread)),
            "glyphs_where_robust_pick_differs": int(changes),
            "robust_pick_p10_gain_mean": float(np.mean([x for x, _ in gain])),
            "robust_pick_nominal_cost_mean": float(np.mean([y for _, y in gain])),
        }
    od = os.path.join(FS, "robustness")
    os.makedirs(od, exist_ok=True)
    json.dump({"summary": summ, "per_solve": out}, open(os.path.join(od, f"{a.study}_{a.objective}.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
