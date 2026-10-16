"""Summaries for the overnight trials (plan experiments E5 and the floor fix), written to
results/fastshadow/research/ for make_research_notes.py.

    python scripts/research_trials.py seeds   # E5: 12 dt2 seeds vs 13 objectives, same budget
    python scripts/research_trials.py floor   # floor barrier on the moving bodies' real lowest point
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastshadow  # noqa: E402

FS = os.path.join(fastshadow.BENCH, "results", "fastshadow")
RES = os.path.join(FS, "research")
MENU = ["current", "iou", "tversky_fn", "tversky_fp", "dt2", "biou", "cldice", "holes", "tc",
        "cls", "clip", "shape3", "shape3_cls"]  # the 13 objectives of the objective study
METRICS = ["iou", "tc_iou", "cls_top1", "holes_filled", "nsd"]


def graded(study):
    return [json.loads(l) for l in open(os.path.join(FS, study, "graded.jsonl"), encoding="utf-8")]


def glyph_ci(vals, n=4000, seed=0):
    v = np.array([np.nan if x is None else x for x in vals], float)
    v = v[~np.isnan(v)]
    g = np.random.default_rng(seed)
    b = [v[g.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def seeds():
    S = defaultdict(list)
    for r in graded("seeds_dt2"):
        S[r["id"]].append(r)
    M = defaultdict(dict)
    for r in graded("objective_study"):
        if r["seed"] == 0 and r["objective"] in MENU:
            M[r["id"]][r["objective"]] = r
    ids = sorted(i for i in S if len(S[i]) >= 12 and len(M[i]) == len(MENU))
    g = np.random.default_rng(0)
    out = {"n_glyphs": len(ids), "k_curve": {}, "pick": {}}
    for k in (1, 2, 4, 8, 12):
        a, b = [], []
        for i in ids:
            seeds_i, menu_i = S[i][:12], list(M[i].values())
            # average over random k-subsets so the curve does not depend on seed order
            a.append(np.mean([max(r["iou"] for r in g.choice(seeds_i, k, replace=False)) for _ in range(50)]))
            b.append(np.mean([max(r["iou"] for r in g.choice(menu_i, min(k, len(menu_i)), replace=False)) for _ in range(50)]))
        out["k_curve"][k] = {"same_objective": float(np.mean(a)), "objective_menu": float(np.mean(b))}
    for name, pool in (("12 tries of dt2", lambda i: S[i][:12]), ("13 objectives, 1 try each", lambda i: list(M[i].values())),
                       ("1 try of dt2", lambda i: [S[i][0]])):
        picks = [max(pool(i), key=lambda r: r["iou"]) for i in ids]
        out["pick"][name] = {m: glyph_ci([p[m] for p in picks]) for m in METRICS if m in picks[0]}
        out["pick"][name]["objective_counts"] = dict(sorted(
            ((o, sum(p["objective"] == o for p in picks)) for o in {p["objective"] for p in picks}), key=lambda x: -x[1]))
    d = [max(S[i][:12], key=lambda r: r["iou"])["iou"] - max(M[i].values(), key=lambda r: r["iou"])["iou"] for i in ids]
    out["iou_diff_seeds_minus_menu"] = glyph_ci(d)
    out["wins_seeds"] = int(sum(x > 1e-4 for x in d))
    out["wins_menu"] = int(sum(x < -1e-4 for x in d))
    json.dump(out, open(os.path.join(RES, "seeds_vs_objectives.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def floor():
    fastshadow.use_fleet()
    from renderer import ShadowRenderer, build_scene

    from fastshadow.renderer import FastShadowRenderer

    model, data, n = build_scene("urdf/SO101/so101_new_calib.urdf", n_robots=3, arm_gap=0.2)
    mj = ShadowRenderer(model, data, size=128, n_robots=n)
    fr = FastShadowRenderer(mj)

    def lowest(rows):
        Q = np.array([r["q_rad"] for r in rows])
        return np.concatenate([fr.moving_min_z(Q[s:s + 512]).cpu().numpy() for s in range(0, len(Q), 512)])

    new = {(r["id"], r["seed"]): r for r in graded("floorfix_dt2")}
    sol_new = {(r["id"], r["seed"]): r for r in map(json.loads, open(os.path.join(FS, "floorfix_dt2", "solves.jsonl"), encoding="utf-8"))}
    old = {(r["id"], r["seed"]): r for r in graded("objective_study") if r["objective"] == "dt2"}
    sol_old = {(r["id"], r["seed"]): r for r in map(json.loads, open(os.path.join(FS, "objective_study", "solves.jsonl"), encoding="utf-8"))
               if r["objective"] == "dt2"}
    keys = sorted(set(new) & set(old) & set(sol_new) & set(sol_old))
    zo, zn = lowest([sol_old[k] for k in keys]), lowest([sol_new[k] for k in keys])
    out = {"n_solves": len(keys), "n_glyphs": len({k[0] for k in keys})}
    for name, z in (("before", zo), ("after", zn)):
        out[name] = {"below_5mm": float(np.mean(z < -0.005)), "below_0": float(np.mean(z < 0)),
                     "below_margin_10mm_above": float(np.mean(z < 0.01)), "lowest_mm": float(1000 * z.min()),
                     "collision_free": float(np.mean([(sol_old if name == "before" else sol_new)[k]["collision_free"] for k in keys]))}
    for m in METRICS:
        if m in new[keys[0]]:
            nn = lambda x: np.nan if x is None else x  # noqa: E731
            out["before"][m] = float(np.nanmean([nn(old[k][m]) for k in keys]))
            out["after"][m] = float(np.nanmean([nn(new[k][m]) for k in keys]))
    byg = defaultdict(list)
    for k in keys:
        byg[k[0]].append(new[k]["iou"] - old[k]["iou"])
    out["iou_change_glyph_ci"] = glyph_ci([np.mean(v) for v in byg.values()])
    # best of the 3 seeds, as the benchmark reports it
    bo = [max((old[k] for k in keys if k[0] == gid), key=lambda r: r["iou"])["iou"] for gid in byg]
    bn = [max((new[k] for k in keys if k[0] == gid), key=lambda r: r["iou"])["iou"] for gid in byg]
    out["best_of_3_iou"] = {"before": float(np.mean(bo)), "after": float(np.mean(bn)),
                            "change_glyph_ci": glyph_ci(np.array(bn) - np.array(bo))}
    out["seconds_per_solve"] = {"before": float(np.mean([sol_old[k]["seconds"] for k in keys])),
                                "after": float(np.mean([sol_new[k]["seconds"] for k in keys]))}
    json.dump(out, open(os.path.join(RES, "floor_fix.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def timing(n_glyphs=6):
    """Seconds per solve with each floor input, back to back on the same glyphs (run on an idle GPU)."""
    import contextlib
    import io
    import time

    fastshadow.use_fleet()
    from optimizer import OptimizerConfig, optimize_staged
    from renderer import ShadowRenderer, build_scene
    from run_base_optimizer import load_target

    from fastshadow.cma_bridge import BatchedShadowRenderer
    from fastshadow.renderer import FastShadowRenderer

    meta = json.load(open(os.path.join(FS, "floorfix_dt2", "meta.json"), encoding="utf-8"))
    rig, ocfg = meta["rig"], meta["optimizer"]
    model, data, n = build_scene("urdf/SO101/so101_new_calib.urdf", n_robots=rig["n_robots"], arm_gap=rig["arm_gap_m"])
    mj = ShadowRenderer(model, data, size=rig["render_size"], n_robots=n)
    fast = FastShadowRenderer(mj)
    import optimizer as opt
    from fastshadow.objectives import TargetCache, use_objective

    root = os.path.join(fastshadow.BENCH, "optimized", "big-budget-grounded")
    out = {"mujoco": [], "mesh": []}
    for tid in meta["targets"][:n_glyphs]:
        sub, stem = tid.split("/")
        T = load_target(os.path.join(root, sub, stem, f"{stem}_shown.png"), mj.size)
        for fl in ("mujoco", "mesh", "mujoco", "mesh"):  # alternate, so drift hits both
            br = BatchedShadowRenderer(mj, fast, floor=fl)
            use_objective(opt, br, "dt2", TargetCache(sub, stem.split("_")[0]))
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                optimize_staged(br, T, OptimizerConfig(seed=0, **ocfg))
            out[fl].append(time.perf_counter() - t0)
            use_objective(opt, br, None)
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res["n"] = len(out["mesh"])
    json.dump(res, open(os.path.join(RES, "floor_timing.json"), "w"), indent=1)
    print(res)


if __name__ == "__main__":
    {"seeds": seeds, "floor": floor, "timing": timing}[sys.argv[1]]()
