"""Gradient solve (Adam on the nvdiffrast soft renderer) for every target in a manifest.

The solver is fastshadow.grad_opt with the README's chosen settings (64 starts x 300
steps, worse half re-seeded at each blur stage) and its recommended defaults: the `dt2`
objective (1 - IoU + 0.5 (DT_out on spill + DT_in on missed target)) and the floor
barrier on the moving bodies' real lowest point (`mesh`). Each seed's top-k starts are
re-scored on MuJoCo with that same objective and floor, and the best kept; every
reported number is MuJoCo-measured (render, contacts, loss). `--objective current
--floor mujoco` is the CMA-ES loss as-is.

Rig and penalty weights come from a CMA-ES results tree (default
optimized/big-budget-grounded, the grounded-target condition these targets match), and
the collision proxy is calibrated on that tree's contact-free solves.

Each target is first placed into the rig's reachable support with the same similarity
transform (uniform scale + shift, proportions kept) and search settings that tree used
(run_base_optimizer --fit-target). Without it a target can sit partly where no pose
casts a shadow, and the solver fills area instead of the shape. IoU is recorded against
both the shown (fitted) target and the original. --no-fit solves the original as is.

    python scripts/solve_adam_targets.py                         # movies/v1/targets -> movies/v1/results
    python scripts/solve_adam_targets.py --seeds 3 --ids v1_k09_f0152_p0_three --force

Layout:

    <out>/<stem>/
        <stem>_original.png          the target at render size, as authored
        <stem>_shown.png             after the fit: what the solver chased
        <stem>_seed0.png ...         MuJoCo shadow of each seed's pick
        <stem>_best.png              the seed with the lowest MuJoCo loss (objective + floor + contacts)
        results.json                 every seed's IoU, loss, contacts, joints, timing
    <out>/summary.jsonl              one row per target
    <out>/meta.json                  rig, grad config, proxy

Finished targets are skipped unless --force.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastshadow  # noqa: E402

BENCH = fastshadow.BENCH
sys.path.insert(0, os.path.join(BENCH, "scripts"))
from run_base_optimizer import load_target, save_mask  # noqa: E402


def load_solved(root, n=3000):
    """Joint vectors of every stored CMA-ES run, shuffled, for the collision proxy."""
    solved = []
    for sub in os.listdir(root):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for stem in os.listdir(d):
            f = os.path.join(d, stem, "results.json")
            if os.path.exists(f):
                solved += [r["q_rad"] for r in json.load(open(f, encoding="utf-8"))["runs"]]
    solved = np.array(solved)
    np.random.default_rng(0).shuffle(solved)
    return solved[:n]


def iou(a, b):
    a, b = np.asarray(a) > 0.5, np.asarray(b) > 0.5
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def first_results(root):
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for stem in sorted(os.listdir(d)):
            f = os.path.join(d, stem, "results.json")
            if os.path.exists(f):
                return json.load(open(f, encoding="utf-8"))
    raise FileNotFoundError(f"no results.json under {root}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=os.path.join(BENCH, "movies", "v1", "targets", "manifest.jsonl"))
    p.add_argument("--out", default=os.path.join(BENCH, "movies", "v1", "results"))
    p.add_argument("--rig-from", default=os.path.join(BENCH, "optimized", "big-budget-grounded"),
                   help="CMA-ES results tree supplying the rig, penalty weights and proxy calibration")
    p.add_argument("--ids", nargs="*", default=None, help="only these manifest ids")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-init", type=int, default=2048)
    p.add_argument("--n-restarts", type=int, default=64)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--reseed-sigma", type=float, default=0.15)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--objective", choices=["dt2", "current"], default="dt2",
                   help="dt2: the README's default; current: the CMA-ES loss, 1 - IoU + 0.5 DT_out")
    p.add_argument("--floor", choices=["mesh", "mujoco"], default="mesh",
                   help="mesh: barrier on the moving bodies' lowest vertex; mujoco: MuJoCo's min_z, "
                   "pinned at -1.83 cm by the static base (a constant)")
    p.add_argument("--no-fit", dest="fit", action="store_false",
                   help="solve the target as authored, without placing it into the reachable support")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    # use_fleet() chdirs into fleet-shadow-art, so resolve every path first
    manifest, out, root = (os.path.abspath(x) for x in (a.manifest, a.out, a.rig_from))
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    if a.ids:
        rows = [r for r in rows if r["id"] in a.ids]
    os.makedirs(out, exist_ok=True)

    fastshadow.use_fleet()
    import torch
    from loss import shadow_loss
    from optimizer import floor_penalty_loss
    from renderer import ShadowRenderer, build_scene

    from fastshadow.grad_opt import FnObjective, GradConfig, Objective, Penalties, solve
    from fastshadow.objectives import OBJECTIVES, TargetCache
    from fastshadow.renderer import FastShadowRenderer

    first = first_results(root)
    rig, ocfg = first["rig"], first["optimizer"]
    model, data, n = build_scene("urdf/SO101/so101_new_calib.urdf", n_robots=rig["n_robots"], arm_gap=rig["arm_gap_m"],
                                 light_to_front=rig["light_to_front_m"], back_to_wall=rig["back_to_wall_m"])
    size = rig["render_size"]
    mj = ShadowRenderer(model, data, size=size, n_robots=n)
    fr = FastShadowRenderer(mj)
    cfg = GradConfig(
        floor_penalty=ocfg["floor_penalty"], collision_penalty=ocfg["collision_penalty"],
        self_collision_penalty=ocfg["self_collision_penalty"],
        n_init=a.n_init, n_restarts=a.n_restarts, iters=a.iters, lr=a.lr, reseed_sigma=a.reseed_sigma,
        top_k=a.top_k,
    )
    pen = Penalties(fr, solved_q=load_solved(root), floor=a.floor)
    fn = None if a.objective == "current" else OBJECTIVES[a.objective]

    fit_cfg = rig["target_fit"] if a.fit else None
    if fit_cfg:
        from reachability import build_reachability_map, uncastable_fraction
        from target_fit import fit_target

        reach = build_reachability_map(mj, n_samples=fit_cfg["reach_samples"], verbose=False)
        support = reach.support()
        print(f"reach map ready ({fit_cfg['reach_samples']} samples)", flush=True)
    print(f"{len(rows)} targets; proxy: {len(pen.s_radius)} spheres, {len(pen.pi)} pairs, "
          f"calibrated on {pen.n_calib} contact-free solves; fast renderer {fr.n_tri} tris", flush=True)
    meta = {"args": vars(a), "objective": a.objective, "floor": a.floor, "rig": rig, "rig_from": root, "target_fit": fit_cfg, "grad_config": cfg.__dict__,
            "n_tri": fr.n_tri, "proxy_spheres": len(pen.s_radius), "proxy_pairs": len(pen.pi)}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2, default=str)

    summary = os.path.join(out, "summary.jsonl")
    for ti, row in enumerate(rows):
        stem = os.path.splitext(os.path.basename(row["target"]))[0]
        d = os.path.join(out, stem)
        rpath = os.path.join(d, "results.json")
        if os.path.exists(rpath) and not a.force:
            print(f"[{ti + 1}/{len(rows)}] {stem}: done, skipping", flush=True)
            continue
        os.makedirs(d, exist_ok=True)
        T0 = load_target(os.path.join(BENCH, row["target"]), size)
        T, fit_info = T0, None
        if fit_cfg:
            fit = fit_target(
                T0, reach,
                scale_range=tuple(fit_cfg["scale_range"]),
                n_scales=fit_cfg["n_scales"],
                max_shift_frac=fit_cfg["max_shift_frac"],
                dy_range=None if fit_cfg["dy_range"] == "symmetric" else tuple(fit_cfg["dy_range"]),
                n_shifts=fit_cfg["n_shifts"],
                scale_penalty=fit_cfg["scale_penalty"],
                min_retained=fit_cfg["min_retained"],
                verbose=False,
            )
            T = fit.target
            fit_info = dict(fit.as_dict())
            fit_info["uncastable_before"] = round(float(uncastable_fraction(T0, support)), 5)
            fit_info["uncastable_after"] = round(float(uncastable_fraction(T, support)), 5)
            print(f"    fit: scale {fit_info['scale']}  dx {fit_info['dx']}  dy {fit_info['dy']}  "
                  f"uncastable {fit_info['uncastable_before']:.3f} -> {fit_info['uncastable_after']:.3f}", flush=True)
        save_mask(T0, os.path.join(d, f"{stem}_original.png"))
        save_mask(T, os.path.join(d, f"{stem}_shown.png"))

        tc = TargetCache("movies", row.get("label") or stem)

        def score(q):
            """MuJoCo render and contacts, scored with the objective and floor the
            solver chased (the fast renderer and sphere proxy only steer the search)."""
            mask, mj_min_z, cross, self_ = mj.eval_q(q)
            moving_z = float(fr.moving_min_z(np.asarray(q)[None])[0])
            if fn is None:
                sh = float(shadow_loss(mask, T, cfg.alpha, cfg.beta))
            else:
                with torch.no_grad():
                    sh = float(fn(torch.tensor(mask, dtype=torch.float32, device=fr.device)[None], tc(T))[0])
            min_z = moving_z if a.floor == "mesh" else mj_min_z
            floor = floor_penalty_loss(min_z, cfg.floor_z, cfg.floor_penalty, cfg.floor_margin, cfg.floor_barrier)
            return {"iou": iou(mask, T), "loss": sh + floor + cfg.collision_penalty * cross
                    + cfg.self_collision_penalty * self_, "shadow_loss": sh, "floor_loss": float(floor),
                    "cross_pen_m": float(cross), "self_pen_m": float(self_), "mujoco_min_z": float(mj_min_z),
                    "lowest_moving_z": round(moving_z, 5), "mask": mask}

        runs, masks = [], []
        for s in range(a.seeds):
            cfg.seed = s
            obj = Objective(fr, pen, T, cfg) if fn is None else FnObjective(fr, pen, T, cfg, fn, tc(T))
            r = solve(fr, pen, T, cfg, obj=obj)
            t0 = time.perf_counter()
            cands = [score(q) for q in r.top_q]
            k = int(np.argmin([c["loss"] for c in cands]))
            q, v = r.top_q[k], cands[k]
            m = v.pop("mask")
            v["iou_vs_original"] = iou(m, T0)
            # MuJoCo re-scoring of the top-k is part of the method, so it is timed
            v.update(seed=s, seconds=r.seconds + time.perf_counter() - t0,
                     collision_free=bool(v["cross_pen_m"] <= 0 and v["self_pen_m"] <= 0),
                     n_hard=r.n_hard_renders, n_soft=r.n_soft_renders,
                     q_rad=[round(float(x), 6) for x in q])
            runs.append(v)
            masks.append(m)
            save_mask(m, os.path.join(d, f"{stem}_seed{s}.png"))
            print(f"    seed {s}: IoU {v['iou']:.3f} (vs original {v['iou_vs_original']:.3f})  loss {v['loss']:.4f}  "
                  f"lowest {100 * v['lowest_moving_z']:+.1f} cm  "
                  f"{'ok' if v['collision_free'] else 'CONTACT'}  {v['seconds']:.1f}s", flush=True)

        bi = int(np.argmin([x["loss"] for x in runs]))
        save_mask(masks[bi], os.path.join(d, f"{stem}_best.png"))
        ious = [x["iou"] for x in runs]
        res = {"id": row["id"], "label": row.get("label"), "target": row["target"],
               "method": "adam" + ("+fit" if fit_cfg else ""), "objective": a.objective, "floor": a.floor,
               "rig": rig, "fit": fit_info,
               "grad_config": cfg.__dict__, "n_runs": len(runs), "best_run": bi,
               "best_iou": runs[bi]["iou"], "best_iou_vs_original": runs[bi]["iou_vs_original"], "max_iou": max(ious), "mean_iou": float(np.mean(ious)),
               "std_iou": float(np.std(ious)), "seconds": float(sum(x["seconds"] for x in runs)),
               "runs": runs}
        json.dump(res, open(rpath, "w"), indent=2, default=str)
        with open(summary, "a", encoding="utf-8") as f:
            f.write(json.dumps({k: res[k] for k in ("id", "label", "method", "objective", "floor", "best_run", "best_iou",
                                                    "best_iou_vs_original", "max_iou", "mean_iou", "seconds")}
                               | {"collision_free": runs[bi]["collision_free"],
                                  "lowest_moving_z": runs[bi]["lowest_moving_z"]}) + "\n")
        print(f"[{ti + 1}/{len(rows)}] {stem}: best {res['best_iou']:.3f} (seed {bi})  "
              f"mean {res['mean_iou']:.3f}", flush=True)


if __name__ == "__main__":
    main()
