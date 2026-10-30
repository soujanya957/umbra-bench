"""Does starting from a library of stored solves make a solve faster at the same quality?

Library: every stored 3-arm solve on the bench's fixed column (fastshadow.pose_library),
contact- and floor-filtered on this rig. The 30 test glyphs are held out by character,
in every font and in both cases, so a test target never finds its own solve.

Methods, each on the bench's 30 fitted targets (scored against them as IoU and against the
original glyph as tc_iou, every number re-measured by MuJoCo):
  cold_*     staged CMA-ES (Hungarian + Voronoi + greedy phases) at a ladder of budgets
  adam_*     Adam, random starts
  lib_only   the library's best pose, nothing else
  lib_cma_*  library's best pose, then only the joint final pass of CMA-ES
  lib3_cma_* the same from the 3 best different poses, best kept
  lib_adam_* Adam from the library's top poses

    python scripts/pose_library_study.py build
    python scripts/pose_library_study.py run --seeds 3
    python scripts/pose_library_study.py report
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
from run_base_optimizer import load_target  # noqa: E402

OUT = os.path.join(BENCH, "results", "fastshadow", "pose_library")
ROOT = os.path.join(BENCH, "optimized", "big-budget-grounded")
FLOOR_TOL = -0.005  # m: lowest moving point allowed under the table
SIZES = (1000, 5000, 20000)


def bench_ids():
    src = os.path.join(BENCH, "results", "fastshadow", "compare_letters_digits", "per_target.jsonl")
    return [json.loads(l)["id"] for l in open(src, encoding="utf-8")]


def held_out():
    """Characters of the test glyphs, lower-cased: 'O' also removes 'o' in every font."""
    return {i.split("/")[1].split("_")[0].lower() for i in bench_ids()}


def rig():
    fastshadow.use_fleet()
    from renderer import ShadowRenderer, build_scene

    from fastshadow.cma_bridge import BatchedShadowRenderer
    from fastshadow.renderer import FastShadowRenderer

    first = json.load(open(os.path.join(ROOT, *bench_ids()[0].split("/"), "results.json"), encoding="utf-8"))
    r = first["rig"]
    model, data, n = build_scene("urdf/SO101/so101_new_calib.urdf", n_robots=r["n_robots"], arm_gap=r["arm_gap_m"])
    mj = ShadowRenderer(model, data, size=r["render_size"], n_robots=n)
    fr = FastShadowRenderer(mj)
    return mj, fr, BatchedShadowRenderer(mj, fr, floor="mesh"), first["optimizer"]


def build(a):
    import torch

    from fastshadow.pose_library import harvest

    mj, fr, br, _ = rig()
    chars = held_out()

    def skip(top, subset, stem):
        return ("letter" in subset or "digit" in subset) and stem.split("_")[0].lower() in chars

    t0 = time.perf_counter()
    Q, sources = harvest(os.path.join(BENCH, "optimized"), 3, skip)
    print(f"harvested {len(Q)} unique poses in {time.perf_counter() - t0:.0f} s from {sources}", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        z = torch.cat([fr.moving_min_z(Q[i: i + 4096].astype(np.float64)) for i in range(0, len(Q), 4096)]).cpu().numpy()
    keep = z >= FLOOR_TOL
    print(f"floor: {keep.mean():.3f} of poses pass ({time.perf_counter() - t0:.0f} s)", flush=True)
    Q = Q[keep]
    t0 = time.perf_counter()
    pen = np.array([br._penalties(q.astype(np.float64)) for q in Q])
    ok = (pen[:, 1] <= 0) & (pen[:, 2] <= 0)
    print(f"contacts: {ok.mean():.3f} pass ({time.perf_counter() - t0:.0f} s)", flush=True)
    Q = Q[ok]
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "library_n3.npz"), Q=Q)
    json.dump({"n": len(Q), "sources": sources, "held_out": sorted(chars)}, open(os.path.join(OUT, "library_n3.json"), "w"), indent=1)
    print("library:", len(Q), "poses")


def run(a):
    mj, fr, br, ocfg = rig()
    import optimizer as opt  # noqa: F401
    import torch
    from optimizer import OptimizerConfig, optimize_staged

    from fastshadow.grad_opt import GradConfig, Penalties, pick_by_mujoco, solve, verify
    from fastshadow.pose_library import PoseLibrary

    Q = np.load(os.path.join(OUT, "library_n3.npz"))["Q"]
    t0 = time.perf_counter()
    lib = PoseLibrary.build(fr, Q)
    print(f"library {len(lib)} poses rendered and packed in {time.perf_counter() - t0:.1f} s, "
          f"{lib.bits.numel() * 8 / 1e6:.0f} MB on the GPU", flush=True)
    gcfg = GradConfig(floor_penalty=ocfg["floor_penalty"], collision_penalty=ocfg["collision_penalty"],
                      self_collision_penalty=ocfg["self_collision_penalty"], reseed_sigma=0.15)
    g = np.random.default_rng(0)
    pen = Penalties(fr, solved_q=Q[g.permutation(len(Q))[:3000]].astype(np.float64))
    subs = {n: lib.subset(np.sort(g.permutation(len(lib))[:n])) for n in SIZES if n < len(lib)}

    ids = bench_ids()
    targets = {}
    for tid in ids:
        sub, stem = tid.split("/")
        d = os.path.join(ROOT, sub, stem)
        targets[tid] = (load_target(os.path.join(d, f"{stem}_shown.png"), mj.size),
                        load_target(os.path.join(d, f"{stem}_original.png"), mj.size))

    path = os.path.join(OUT, "runs.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["id"], r["method"], r["seed"]) for r in map(json.loads, open(path, encoding="utf-8"))}

    def measure(q, T, T0):
        v = verify(mj, np.asarray(q, np.float64), T, gcfg)
        m = v.pop("mask")
        v["tc_iou"] = trimmed_centred(T0 > 0.5, m > 0.5)["tc_iou"]
        with torch.no_grad():
            v["floor_z"] = float(fr.moving_min_z(np.asarray(q, np.float64)[None])[0])
        v["clean"] = bool(v["cross_pen_m"] <= 0 and v["self_pen_m"] <= 0 and v["floor_z"] >= FLOOR_TOL)
        return v

    def cma(T, seed, budget, x0=None):
        p, p1, p2, f = budget
        kw = {**ocfg, "popsize": p, "phase1_iters": p1, "phase2_iters": p2, "final_iters": f}
        cfg = OptimizerConfig(seed=seed, skip_greedy=x0 is not None, **kw)
        with contextlib.redirect_stdout(io.StringIO()):
            return optimize_staged(br, T, cfg, x0=None if x0 is None else x0.astype(np.float64))

    COLD = {"cold_8k": (48, 16, 16, 30), "cold_3k": (24, 12, 12, 16), "cold_1k": (16, 6, 6, 10), "cold_300": (8, 4, 4, 6)}
    WARM = {"lib_cma_f30": (48, 0, 0, 30), "lib_cma_f16": (24, 0, 0, 16), "lib_cma_f8": (16, 0, 0, 8), "lib_cma_f4": (12, 0, 0, 4)}
    ADAM = {"adam_64x300": (2048, 64, 300), "adam_16x100": (2048, 16, 100)}
    LADAM = {"lib_adam_64x100": (64, 100), "lib_adam_16x100": (16, 100), "lib_adam_16x40": (16, 40)}

    def log(rec):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    for ti, tid in enumerate(ids):
        T, T0 = targets[tid]

        def rec(meth, s, q, dt, extra=None):
            v = measure(q, T, T0)
            log({"id": tid, "method": meth, "seed": s, "seconds": dt,
                 **{k: v[k] for k in ("iou", "tc_iou", "loss", "clean", "floor_z")}, **(extra or {})})

        for n, L in [*subs.items(), (len(lib), lib)]:
            meth = f"lib_only_{n}"
            if (tid, meth, 0) not in done:
                t0 = time.perf_counter()
                q, _ = L.query(T, 1)
                torch.cuda.synchronize()
                rec(meth, 0, q[0], time.perf_counter() - t0)
        for s in range(a.seeds):
            for m, b in COLD.items():
                if (tid, m, s) not in done:
                    t0 = time.perf_counter()
                    r = cma(T, s, b)
                    rec(m, s, r.best_q, time.perf_counter() - t0, {"n_evals": int(r.n_evals)})
            for m, b in WARM.items():
                if (tid, m, s) not in done:
                    t0 = time.perf_counter()
                    q, _ = lib.query(T, 1)
                    r = cma(T, s, b, q[0])
                    rec(m, s, r.best_q, time.perf_counter() - t0, {"n_evals": int(r.n_evals)})
                m3 = m.replace("lib_", "lib3_")
                if (tid, m3, s) not in done and m in ("lib_cma_f8", "lib_cma_f4"):
                    t0 = time.perf_counter()
                    qs, _ = lib.query(T, 3, min_dist=0.3)
                    rs = [cma(T, s + 10 * i, b, q) for i, q in enumerate(qs)]
                    best, _ = pick_by_mujoco(mj, np.array([x.best_q for x in rs]), T, gcfg)
                    rec(m3, s, best, time.perf_counter() - t0, {"n_evals": sum(int(x.n_evals) for x in rs)})
            for m, (ni, R, it) in ADAM.items():
                if (tid, m, s) not in done:
                    gcfg.n_init, gcfg.n_restarts, gcfg.iters, gcfg.seed, gcfg.top_k = ni, R, it, s, min(8, R)
                    r = solve(fr, pen, T, gcfg)
                    t0 = time.perf_counter()
                    q, _ = pick_by_mujoco(mj, r.top_q, T, gcfg)
                    rec(m, s, q, r.seconds + time.perf_counter() - t0)
            for m, (R, it) in LADAM.items():
                if (tid, m, s) not in done:
                    t0 = time.perf_counter()
                    qs, _ = lib.query(T, R, min_dist=0.05)
                    tq = time.perf_counter() - t0
                    gcfg.n_init, gcfg.n_restarts, gcfg.iters, gcfg.seed, gcfg.top_k = R, R, it, s, min(8, R)
                    r = solve(fr, pen, T, gcfg, init_q=qs)
                    t0 = time.perf_counter()
                    q, _ = pick_by_mujoco(mj, r.top_q, T, gcfg)
                    rec(m, s, q, tq + r.seconds + time.perf_counter() - t0)
        print(f"[{ti + 1}/{len(ids)}] {tid}", flush=True)


def report(a):
    rows = [json.loads(l) for l in open(os.path.join(OUT, "runs.jsonl"), encoding="utf-8")]
    ms = sorted({r["method"] for r in rows})
    tab = []
    for m in ms:
        R = [r for r in rows if r["method"] == m]
        ids = sorted({r["id"] for r in R})
        per = lambda k: np.array([np.mean([r[k] for r in R if r["id"] == i]) for i in ids])  # noqa: E731
        tab.append((m, len(ids), per("iou").mean(), per("tc_iou").mean(), np.median([r["seconds"] for r in R]),
                    np.mean([r["clean"] for r in R])))
    tab.sort(key=lambda x: x[4])
    print(f"{'method':<20}{'n':>4}{'IoU':>8}{'tc_iou':>8}{'s/solve':>9}{'clean':>7}")
    for t in tab:
        print(f"{t[0]:<20}{t[1]:>4}{t[2]:>8.3f}{t[3]:>8.3f}{t[4]:>9.3f}{t[5]:>7.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "run", "report"])
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()
    {"build": build, "run": run, "report": report}[a.cmd](a)
