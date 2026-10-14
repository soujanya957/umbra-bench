"""Gradient solve on the nvdiffrast renderer vs the CMA-ES results already on disk.

For each target in a stratified sample of an existing results tree (default
optimized/big-budget-grounded, 10 CMA-ES solves per target), on the exact target
the CMA-ES solver was shown (<stem>_shown.png):

  cma         the stored solves, re-measured here with MuJoCo
  grad        pure gradient: random multi-start + blurred-IoU Adam, `--seeds` solves
  cma+polish  each stored CMA-ES solve refined by Adam (fine blur only)
  cma_local   (--time-cma K) CMA-ES re-run on this machine for the first K targets,
              so wall-clock is compared on the same hardware

Every number is MuJoCo-measured (render, contacts, loss) via grad_opt.verify, so the
fast renderer and the sphere collision proxy can only affect the search, not the
score.

    python scripts/compare_grad.py --per-subset 6 --seeds 3 --time-cma 10
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
from metrics import trimmed_centred  # noqa: E402
from run_base_optimizer import load_target, save_mask  # noqa: E402


def pick_targets(root, per_subset, subsets=None):
    out = []
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d) or (subsets and sub not in subsets):
            continue
        stems = sorted(s for s in os.listdir(d) if os.path.exists(os.path.join(d, s, "results.json")))
        if not stems:
            continue
        idx = np.unique(np.linspace(0, len(stems) - 1, min(per_subset, len(stems))).round().astype(int))
        out += [(sub, stems[i]) for i in idx]
    return out


def summarise(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.path.join(BENCH, "optimized", "big-budget-grounded"))
    p.add_argument("--per-subset", type=int, default=6)
    p.add_argument("--subsets", nargs="*", default=None)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-init", type=int, default=2048)
    p.add_argument("--n-restarts", type=int, default=64)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--reseed-sigma", type=float, default=0.0)
    p.add_argument("--polish-iters", type=int, default=100)
    p.add_argument("--no-polish", action="store_true")
    p.add_argument("--time-cma", type=int, default=0)
    p.add_argument("--tag", default="v1")
    a = p.parse_args()

    root = os.path.abspath(a.results)
    outdir = os.path.join(BENCH, "results", "fastshadow", f"compare_{a.tag}")
    os.makedirs(outdir, exist_ok=True)
    fastshadow.use_fleet()
    import torch
    from optimizer import OptimizerConfig, optimize_staged
    from renderer import ShadowRenderer, build_scene

    from fastshadow.grad_opt import GradConfig, Penalties, pick_by_mujoco, polish, solve, verify
    from fastshadow.renderer import FastShadowRenderer

    jobs = pick_targets(root, a.per_subset, a.subsets)
    first = json.load(open(os.path.join(root, *jobs[0], "results.json"), encoding="utf-8"))
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
    )
    # Calibrate the collision proxy on solves MuJoCo accepted, from targets NOT in
    # this sample.
    picked = set(jobs)
    solved = []
    for sub in os.listdir(root):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for stem in os.listdir(d):
            f = os.path.join(d, stem, "results.json")
            if (sub, stem) in picked or not os.path.exists(f):
                continue
            solved += [r["q_rad"] for r in json.load(open(f, encoding="utf-8"))["runs"]]
    solved = np.array(solved)
    np.random.default_rng(0).shuffle(solved)
    pen = Penalties(fr, solved_q=solved[:3000])
    print(f"{len(jobs)} targets; proxy: {len(pen.s_radius)} spheres, {len(pen.pi)} pairs, "
          f"calibrated on {pen.n_calib} contact-free solves; fast renderer {fr.n_tri} tris", flush=True)
    meta = {"args": vars(a), "rig": rig, "baseline_optimizer": ocfg, "grad_config": cfg.__dict__,
            "n_tri": fr.n_tri, "proxy_spheres": len(pen.s_radius), "proxy_pairs": len(pen.pi)}
    json.dump(meta, open(os.path.join(outdir, "meta.json"), "w"), indent=2, default=str)

    rows_path = os.path.join(outdir, "per_target.jsonl")
    done = set()
    if os.path.exists(rows_path):
        done = {json.loads(l)["id"] for l in open(rows_path, encoding="utf-8")}
    for ti, (sub, stem) in enumerate(jobs):
        tid = f"{sub}/{stem}"
        if tid in done:
            continue
        d = os.path.join(root, sub, stem)
        res = json.load(open(os.path.join(d, "results.json"), encoding="utf-8"))
        shown = os.path.join(d, f"{stem}_shown.png")
        T = load_target(shown, size) if os.path.exists(shown) else load_target(os.path.join(BENCH, res["target"]), size)
        orig = os.path.join(d, f"{stem}_original.png")
        T0 = load_target(orig, size) if os.path.exists(orig) else T

        def measure(q):
            v = verify(mj, np.asarray(q, np.float64), T, cfg)
            m = v.pop("mask")
            v["tc_iou"] = trimmed_centred(T0 > 0.5, m > 0.5)["tc_iou"]
            v["collision_free"] = bool(v["cross_pen_m"] <= 0 and v["self_pen_m"] <= 0)
            return v, m

        row = {"id": tid, "subset": sub}
        # ── stored CMA-ES solves ──────────────────────────────────────────
        cm = [measure(r["q_rad"]) for r in res["runs"]]
        cma = [c for c, _ in cm]
        bi = int(np.argmin([c["loss"] for c in cma]))
        save_mask(T, os.path.join(outdir, f"{sub}__{stem}__target.png"))
        for k, (_, m) in enumerate(cm):
            save_mask(m, os.path.join(outdir, f"{sub}__{stem}__cma{k}.png"))
        row["cma"] = {"best_iou": max(c["iou"] for c in cma), "mean_iou": float(np.mean([c["iou"] for c in cma])),
                      "best_loss_run": bi, "runs_iou": [c["iou"] for c in cma],
                      "best_loss_iou": cma[bi]["iou"], "best_loss_tc_iou": cma[bi]["tc_iou"],
                      "best_loss": cma[bi]["loss"], "n": len(cma),
                      "collision_free_frac": float(np.mean([c["collision_free"] for c in cma])),
                      "stored_best_iou": res["best_iou"], "stored_seconds_per_solve": res["seconds"] / max(1, res["n_runs"])}
        # ── pure gradient ─────────────────────────────────────────────────
        g = []
        for s in range(a.seeds):
            cfg.seed = s
            r = solve(fr, pen, T, cfg)
            t0 = time.perf_counter()
            q, _ = pick_by_mujoco(mj, r.top_q, T, cfg)
            v, m = measure(q)
            # MuJoCo re-scoring of the top-k is part of the method, so it is timed
            v.update(seconds=r.seconds + time.perf_counter() - t0, n_hard=r.n_hard_renders, n_soft=r.n_soft_renders)
            g.append(v)
            save_mask(m, os.path.join(outdir, f"{sub}__{stem}__grad{s}.png"))
        bi = int(np.argmin([x["loss"] for x in g]))
        row["grad"] = {"best_iou": max(x["iou"] for x in g), "mean_iou": float(np.mean([x["iou"] for x in g])),
                       "best_loss_run": bi,
                       "best_loss_iou": g[bi]["iou"], "best_loss_tc_iou": g[bi]["tc_iou"], "best_loss": g[bi]["loss"],
                       "n": len(g), "seconds_per_solve": float(np.mean([x["seconds"] for x in g])),
                       "collision_free_frac": float(np.mean([x["collision_free"] for x in g])),
                       "runs": [{k: x[k] for k in ("iou", "loss", "tc_iou", "seconds", "collision_free")} for x in g]}
        # ── CMA-ES + Adam polish ─────────────────────────────────────────
        if not a.no_polish:
            t0 = time.perf_counter()
            Q0 = np.array([r["q_rad"] for r in res["runs"]])
            Qp, _ = polish(fr, pen, T, Q0, cfg, iters=a.polish_iters)
            # accept a polished run only if MuJoCo agrees it is better
            pv = []
            for q0, q1, c in zip(Q0, Qp, cma):
                v1 = measure(q1)[0]
                pv.append(v1 if v1["loss"] < c["loss"] else c)
            tp = time.perf_counter() - t0
            bi = int(np.argmin([x["loss"] for x in pv]))
            row["cma_polish"] = {"best_iou": max(x["iou"] for x in pv), "mean_iou": float(np.mean([x["iou"] for x in pv])),
                                 "best_loss_iou": pv[bi]["iou"], "best_loss_tc_iou": pv[bi]["tc_iou"],
                                 "best_loss": pv[bi]["loss"], "seconds_all_runs": tp,
                                 "collision_free_frac": float(np.mean([x["collision_free"] for x in pv])),
                                 "per_run_iou_gain": float(np.mean([x["iou"] - c["iou"] for x, c in zip(pv, cma)]))}
        # ── CMA-ES timed on this machine ─────────────────────────────────
        if ti < a.time_cma:
            oc = OptimizerConfig(seed=0, **ocfg)
            t0 = time.perf_counter()
            r = optimize_staged(mj, T, oc)
            v, m = measure(r.best_q)
            v["seconds"] = time.perf_counter() - t0
            save_mask(m, os.path.join(outdir, f"{sub}__{stem}__cmalocal.png"))
            row["cma_local"] = {k: v[k] for k in ("iou", "loss", "tc_iou", "seconds", "collision_free")}
        with open(rows_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        c, gg = row["cma"], row["grad"]
        print(f"[{ti + 1}/{len(jobs)}] {tid}: cma best {c['best_iou']:.3f} mean {c['mean_iou']:.3f} | "
              f"grad best {gg['best_iou']:.3f} mean {gg['mean_iou']:.3f} ({gg['seconds_per_solve']:.1f}s/solve)"
              + (f" | polish best {row['cma_polish']['best_iou']:.3f} mean {row['cma_polish']['mean_iou']:.3f}"
                 if "cma_polish" in row else "")
              + (f" | cma_local {row['cma_local']['iou']:.3f} {row['cma_local']['seconds']:.1f}s" if "cma_local" in row else ""),
              flush=True)


if __name__ == "__main__":
    main()

