"""Robot arrangement + spacing as optimization variables (digits 0-9).

For each digit, each arrangement in fastshadow.spacing (plus the fixed 0.2 m column
the benchmark has used so far), and each optimizer:

  cma   staged CMA-ES (fleet-shadow-art main, batched hooks), populations on
        nvdiffrast, MuJoCo kinematics + contacts. The two spacing variables are
        extra dimensions of q: the per-arm phases hold them, the joint stage moves
        them with all 18 joints. Seed 0 starts at the default 0.2 m spacing, other
        seeds at a random spacing.
  adam  multi-start Adam on the soft renderer; the spacing gets a gradient through
        the base positions in FK. Final pick among the top 8 starts by the same
        objective on MuJoCo masks plus MuJoCo floor and contact penalties.

Objective dt2 (1 - IoU + 0.5 two-sided DT), the shape objective people preferred.
Every result is re-rendered and contact-checked by MuJoCo with the optimized base
positions. Light, wall and camera never move.

    set FLEET_SHADOW_ART=...\\fleet-shadow-art   (main; the default when unset)
    python scripts/arrangement_study.py run
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastshadow  # noqa: E402

BENCH = fastshadow.BENCH
sys.path.insert(0, os.path.join(BENCH, "scripts"))
from metrics import trimmed_centred  # noqa: E402
from objective_study import all_glyphs, target_list  # noqa: E402
from run_base_optimizer import load_target, save_mask  # noqa: E402

OUT = os.path.join(BENCH, "results", "fastshadow", "arrangement_study")
CONFIGS = ["fixed", "column", "row", "triangle", "diagonal"]
OBJECTIVE = "dt2"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run"])
    p.add_argument("--configs", nargs="*", default=CONFIGS)
    p.add_argument("--optimizers", nargs="*", default=["cma", "adam"])
    p.add_argument("--cma-seeds", type=int, default=8)
    p.add_argument("--adam-seeds", type=int, default=3)
    p.add_argument("--adam-restarts", type=int, default=256)
    p.add_argument("--adam-iters", type=int, default=300)
    p.add_argument("--cma-popsize", type=int, default=64)
    p.add_argument("--cma-final-iters", type=int, default=60)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=OUT)
    a = p.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "masks"), exist_ok=True)

    ms = fastshadow.use_fleet()
    import optimizer as opt
    import torch
    from optimizer import OptimizerConfig, floor_penalty_loss, optimize_staged
    from renderer import ShadowRenderer, build_scene

    assert "batch_shadow_loss" in open(opt.__file__, encoding="utf-8").read(), \
        f"{ms} lacks the batched hooks (fleet-shadow-art main); set FLEET_SHADOW_ART"
    from fastshadow.cma_bridge import BatchedShadowRenderer
    from fastshadow.glyph_cls import GlyphClassifier
    from fastshadow.grad_opt import FnObjective, GradConfig, Penalties, solve
    from fastshadow.objectives import OBJECTIVES, TargetCache, use_objective
    from fastshadow.renderer import FastShadowRenderer
    from fastshadow.spacing import ARRANGEMENTS, SpacedMJ

    root = os.path.join(BENCH, "optimized", "big-budget-grounded")
    ids = [i for i in all_glyphs(root, target_list("letters_digits")) if i.startswith("digits/")]
    if a.limit:
        ids = ids[: a.limit]
    first = json.load(open(os.path.join(root, *ids[0].split("/"), "results.json"), encoding="utf-8"))
    rig, ocfg = first["rig"], dict(first["optimizer"])
    model, data, n = build_scene("urdf/SO101/so101_new_calib.urdf", n_robots=rig["n_robots"], arm_gap=rig["arm_gap_m"])
    mj = ShadowRenderer(model, data, size=rig["render_size"], n_robots=n)
    home = mj.get_current_q().copy()
    fr0 = FastShadowRenderer(mj)
    # collision proxy for Adam, calibrated on contact-free solves of the fixed rig
    solved = []
    for sub in os.listdir(root):
        d = os.path.join(root, sub)
        if os.path.isdir(d) and sub != "digits":
            for stem in os.listdir(d):
                f = os.path.join(d, stem, "results.json")
                if os.path.exists(f):
                    solved += [r["q_rad"] for r in json.load(open(f, encoding="utf-8"))["runs"]]
    solved = np.array(solved)
    np.random.default_rng(0).shuffle(solved)
    pen = Penalties(fr0, solved_q=solved[:3000])
    clf = GlyphClassifier()
    fn = OBJECTIVES[OBJECTIVE]
    frs = {c: (fr0 if c == "fixed" else FastShadowRenderer(mj, arrangement=ARRANGEMENTS[c])) for c in a.configs}

    path = os.path.join(out, "runs.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["id"], r["config"], r["optimizer"], r["seed"]) for r in map(json.loads, open(path, encoding="utf-8"))}
    json.dump({"objective": OBJECTIVE, "configs": a.configs, "targets": ids, "rig": rig, "args": vars(a),
               "arrangements": {k: {"label": v.label, "x": v.x_desc, "y": v.y_desc, "lo": list(v.lo), "hi": list(v.hi)}
                                for k, v in ARRANGEMENTS.items()}},
              open(os.path.join(out, "meta.json"), "w"), indent=2)
    cma_cfg = dict(ocfg, popsize=a.cma_popsize, final_iters=a.cma_final_iters)

    def mujoco_view(config, u0=None):
        """The MuJoCo renderer for a config, joints at home, bases placed."""
        mj.set_qpos(home)
        if config == "fixed":
            SpacedMJ(mj, ARRANGEMENTS["column"]).restore_default()
            mj.set_qpos(home)
            return mj
        sm = SpacedMJ(mj, ARRANGEMENTS[config], u0)
        sm.set_qpos(sm.get_current_q())
        return sm

    def record(tid, config, optname, seed, q, dt, T, T0, extra):
        view = mujoco_view(config)
        mask, min_z, cross, self_ = view.eval_q(q)
        S = mask > 0.5
        sub, stem = tid.split("/")
        mp = os.path.join("masks", f"{stem}__{config}__{optname}__s{seed}.png")
        save_mask(mask, os.path.join(out, mp))
        inter, union = (S & (T > 0.5)).sum(), (S | (T > 0.5)).sum()
        rk, pc = clf.rank(torch.tensor(S[None], dtype=torch.float32), sub, stem.split("_")[0])
        rec = {"id": tid, "digit": stem.split("_")[0], "config": config, "optimizer": optname, "seed": seed,
               "seconds": round(dt, 3), "iou": float(inter / union) if union else 0.0,
               "tc_iou": trimmed_centred(T0 > 0.5, S)["tc_iou"], "cls_top1": int(rk[0] == 1), "cls_p": float(pc[0]),
               "collision_free": bool(cross <= 0 and self_ <= 0), "cross_pen_m": float(cross), "self_pen_m": float(self_),
               "min_z": float(min_z), "mask": mp, "q": [round(float(v), 6) for v in q], **extra}
        if config != "fixed":
            arr = ARRANGEMENTS[config]
            u = np.asarray(q[18:20], float)
            rec["gaps_m"] = [round(float(g), 4) for g in arr.gaps(u)]
            rec["bases_xy"] = np.round(arr.positions(u[None])[0], 4).tolist()
        else:
            rec["gaps_m"] = [0.2, 0.2]
            rec["bases_xy"] = [[0.0, 0.0], [0.0, -0.2], [0.0, -0.4]]
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    t_all = time.time()
    for ti, tid in enumerate(ids):
        sub, stem = tid.split("/")
        d = os.path.join(root, sub, stem)
        T = load_target(os.path.join(d, f"{stem}_shown.png"), mj.size)
        T0 = load_target(os.path.join(d, f"{stem}_original.png"), mj.size)
        tc = TargetCache(sub, stem.split("_")[0])
        for config in a.configs:
            fr = frs[config]
            if "cma" in a.optimizers:
                for s in range(a.cma_seeds):
                    if (tid, config, "cma", s) in done:
                        continue
                    u0 = None if (s == 0 or config == "fixed") else np.random.default_rng(1000 + s).uniform(0.05, 0.95, 2)
                    view = mujoco_view(config, u0)
                    br = BatchedShadowRenderer(view, fr)
                    use_objective(opt, br, OBJECTIVE, tc)
                    t0 = time.perf_counter()
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            res = optimize_staged(br, T, OptimizerConfig(seed=s, **cma_cfg))
                    finally:
                        use_objective(opt, br, None)
                    dt = time.perf_counter() - t0
                    record(tid, config, "cma", s, np.asarray(res.best_q), dt, T, T0,
                           {"n_evals": int(res.n_evals), "u0": None if u0 is None else [float(x) for x in u0]})
            if "adam" in a.optimizers:
                for s in range(a.adam_seeds):
                    if (tid, config, "adam", s) in done:
                        continue
                    cfg = GradConfig(floor_penalty=ocfg["floor_penalty"], collision_penalty=ocfg["collision_penalty"],
                                     self_collision_penalty=ocfg["self_collision_penalty"], reseed_sigma=0.15, seed=s,
                                     n_restarts=a.adam_restarts, iters=a.adam_iters)
                    obj = FnObjective(fr, pen, T, cfg, fn, tc(T))
                    r = solve(fr, pen, T, cfg, obj=obj)
                    t0 = time.perf_counter()
                    view = mujoco_view(config)
                    best = None
                    for q in r.top_q:  # finalists on MuJoCo, same objective + MuJoCo penalties
                        mask, min_z, cross, self_ = view.eval_q(q)
                        with torch.no_grad():
                            sh = float(fn(torch.tensor(mask, device="cuda")[None], tc(T))[0])
                        tot = sh + floor_penalty_loss(min_z, cfg.floor_z, cfg.floor_penalty, cfg.floor_margin,
                                                      cfg.floor_barrier) + cfg.collision_penalty * cross \
                            + cfg.self_collision_penalty * self_
                        if best is None or tot < best[0]:
                            best = (tot, q)
                    dt = r.seconds + time.perf_counter() - t0
                    record(tid, config, "adam", s, np.asarray(best[1]), dt, T, T0,
                           {"n_restarts": a.adam_restarts, "iters": a.adam_iters})
        print(f"[{ti + 1}/{len(ids)}] {tid}  {(time.time() - t_all) / 60:.1f} min", flush=True)
    SpacedMJ(mj, ARRANGEMENTS["column"]).restore_default()


if __name__ == "__main__":
    main()
