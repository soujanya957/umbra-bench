"""CMA-ES on MuJoCo vs CMA-ES on nvdiffrast, and Adam's start-point sweep.

Same targets as compare_grad.py --tag <src> (its per_target.jsonl). Needs the
fleet-shadow-art main (optimizer.optimize uses
renderer.eval_q_batch when present); the default lookup finds it, or set FLEET_SHADOW_ART.

  cma        CMA-ES, MuJoCo renders everything (unchanged path)
  cma_fast   the same CMA-ES, populations rendered by nvdiffrast, MuJoCo still
             doing kinematics + contacts (fastshadow.cma_bridge)
  adam_N     gradient solve with N start points (restarts), everything else fixed

Every number is MuJoCo-measured (grad_opt.verify).

    set FLEET_SHADOW_ART=...\\fleet-shadow-art   (main; the default when unset)
    python scripts/compare_fast_cma.py --src letters_digits --seeds 3 --restarts 8 16 32 64 128 256
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
from run_base_optimizer import load_target, save_mask  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="letters_digits")
    p.add_argument("--results", default=os.path.join(BENCH, "optimized", "big-budget-grounded"))
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--adam-seeds", type=int, default=2)
    p.add_argument("--restarts", type=int, nargs="*", default=[8, 16, 32, 64, 128, 256])
    p.add_argument("--skip-cma", action="store_true")
    a = p.parse_args()
    root = os.path.abspath(a.results)
    src = os.path.join(BENCH, "results", "fastshadow", f"compare_{a.src}")
    outdir = os.path.join(BENCH, "results", "fastshadow", f"fastcma_{a.src}")
    os.makedirs(outdir, exist_ok=True)
    ids = [json.loads(l)["id"] for l in open(os.path.join(src, "per_target.jsonl"), encoding="utf-8")]

    ms = fastshadow.use_fleet()
    import optimizer as _opt
    import torch  # noqa: F401
    from optimizer import OptimizerConfig, optimize_staged
    from renderer import ShadowRenderer, build_scene

    assert "eval_many" in open(_opt.__file__, encoding="utf-8").read(), \
        f"{ms} lacks the batched hooks (fleet-shadow-art main); set FLEET_SHADOW_ART"
    from fastshadow.cma_bridge import BatchedShadowRenderer
    from fastshadow.grad_opt import GradConfig, Penalties, pick_by_mujoco, solve, verify
    from fastshadow.renderer import FastShadowRenderer

    first = json.load(open(os.path.join(root, *ids[0].split("/"), "results.json"), encoding="utf-8"))
    rig, ocfg = first["rig"], first["optimizer"]
    model, data, n = build_scene("urdf/SO101/so101_new_calib.urdf", n_robots=rig["n_robots"], arm_gap=rig["arm_gap_m"])
    mj = ShadowRenderer(model, data, size=rig["render_size"], n_robots=n)
    fr = FastShadowRenderer(mj)
    br = BatchedShadowRenderer(mj, fr)
    gcfg = GradConfig(floor_penalty=ocfg["floor_penalty"], collision_penalty=ocfg["collision_penalty"],
                      self_collision_penalty=ocfg["self_collision_penalty"], reseed_sigma=0.15)

    targets = {}
    for tid in ids:
        sub, stem = tid.split("/")
        d = os.path.join(root, sub, stem)
        targets[tid] = (load_target(os.path.join(d, f"{stem}_shown.png"), mj.size),
                        load_target(os.path.join(d, f"{stem}_original.png"), mj.size))

    def measure(q, T, T0):
        v = verify(mj, np.asarray(q, np.float64), T, gcfg)
        m = v.pop("mask")
        v["tc_iou"] = trimmed_centred(T0 > 0.5, m > 0.5)["tc_iou"]
        v["collision_free"] = bool(v["cross_pen_m"] <= 0 and v["self_pen_m"] <= 0)
        return v, m

    path = os.path.join(outdir, "runs.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["id"], r["method"], r["seed"]) for r in map(json.loads, open(path, encoding="utf-8"))}

    def log(rec):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # ── CMA-ES: MuJoCo vs nvdiffrast, interleaved per target ─────────────
    if not a.skip_cma:
        for ti, tid in enumerate(ids):
            T, T0 = targets[tid]
            for s in range(a.seeds):
                for meth, rend in (("cma", mj), ("cma_fast", br)):
                    if (tid, meth, s) in done:
                        continue
                    t0 = time.perf_counter()
                    with contextlib.redirect_stdout(io.StringIO()):
                        res = optimize_staged(rend, T, OptimizerConfig(seed=s, **ocfg))
                    dt = time.perf_counter() - t0
                    v, m = measure(res.best_q, T, T0)
                    save_mask(m, os.path.join(outdir, f"{tid.replace('/', '__')}__{meth}{s}.png"))
                    log({"id": tid, "method": meth, "seed": s, "seconds": dt, "n_evals": int(res.n_evals),
                         **{k: v[k] for k in ("iou", "loss", "tc_iou", "collision_free")}})
            print(f"[cma {ti + 1}/{len(ids)}] {tid}", flush=True)

    # ── Adam: number of start points ─────────────────────────────────────
    solved = []
    picked = set(ids)
    for sub in os.listdir(root):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for stem in os.listdir(d):
            f = os.path.join(d, stem, "results.json")
            if f"{sub}/{stem}" not in picked and os.path.exists(f):
                solved += [r["q_rad"] for r in json.load(open(f, encoding="utf-8"))["runs"]]
    solved = np.array(solved)
    np.random.default_rng(0).shuffle(solved)
    pen = Penalties(fr, solved_q=solved[:3000])
    for R in a.restarts:
        for ti, tid in enumerate(ids):
            T, T0 = targets[tid]
            for s in range(a.adam_seeds):
                meth = f"adam_{R}"
                if (tid, meth, s) in done:
                    continue
                gcfg.n_restarts, gcfg.seed, gcfg.top_k = R, s, min(8, R)
                r = solve(fr, pen, T, gcfg)
                t0 = time.perf_counter()
                q, _ = pick_by_mujoco(mj, r.top_q, T, gcfg)
                dt = r.seconds + time.perf_counter() - t0
                v, m = measure(q, T, T0)
                if s == 0:
                    save_mask(m, os.path.join(outdir, f"{tid.replace('/', '__')}__{meth}.png"))
                log({"id": tid, "method": meth, "seed": s, "seconds": dt, "restarts": R,
                     "n_soft": r.n_soft_renders, "n_hard": r.n_hard_renders,
                     **{k: v[k] for k in ("iou", "loss", "tc_iou", "collision_free")}})
        print(f"[adam R={R}] done", flush=True)


if __name__ == "__main__":
    main()
