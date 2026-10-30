"""Pose library on the LeSlider stage: 4 arms on two rails, carriages solved with the joints.

The stage is the UI's own (fleet-shadow-art leRobot-control/configs/leslider.json, through
the same transform as POST /optimize/scene), solved by the same renderer the UI uses
(rails.wrap_renderer: CMA-ES populations on nvdiffrast, the free carriages appended to q).

A library pose here is 24 joints plus 4 carriage positions. The bench's stored solves are
3-arm poses on another rig, so they are split into single-arm poses:
  build   4 stored arm poses at random + a random legal carriage layout (combo library),
          and the same number of uniform random poses, as the control
  grow    real solves on this stage, on the bench's other targets (solve library)
Test characters are held out of both, in every font and both cases.

Targets: the solver bench's 30 characters, fitted once to this stage's reach. Every number
is re-measured by MuJoCo with the carriages where the solver put them.

    python scripts/pose_library_leslider.py build --n 300000
    python scripts/pose_library_leslider.py grow  --limit 300
    python scripts/pose_library_leslider.py run   --seeds 3
    python scripts/pose_library_leslider.py report
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

OUT = os.path.join(BENCH, "results", "fastshadow", "pose_library_leslider")
ROOT = os.path.join(BENCH, "optimized", "big-budget-grounded")
FLOOR_TOL = -0.005
N = 4
COLD = {"cold_8k": (48, 16, 16, 30), "cold_3k": (24, 12, 12, 16), "cold_300": (8, 4, 4, 6)}
WARM = {"f30": (48, 0, 0, 30), "f16": (24, 0, 0, 16), "f4": (12, 0, 0, 4)}


def bench_ids():
    src = os.path.join(BENCH, "results", "fastshadow", "compare_letters_digits", "per_target.jsonl")
    return [json.loads(l)["id"] for l in open(src, encoding="utf-8")]


def stage_json():
    """leslider.json -> the solver rig, as render_server/optimize.py builds it."""
    fleet = os.path.dirname(fastshadow.use_fleet())
    c = json.load(open(os.path.join(fleet, "leRobot-control", "configs", "leslider.json"), encoding="utf-8"))
    ids = [r["id"] for r in c["robots"]]
    bases = [{"lat": r["base"]["y"], "dep": -r["base"]["x"], "yaw_deg": float(r["base"].get("yaw", 0))} for r in c["robots"]]
    rails = []
    for r, sign in zip(c["rails"], (-1.0, 1.0)):  # rail_A runs toward the screen, rail_B toward the light
        rails.append({"id": r["id"], "dir": [0.0, sign], "s_max": 0.81784, "gap": 0.1786,
                      "carriages": [{"arm": ids.index(k["robotId"]), "s": max(float(k["s"]), 1e-5), "locked": False}
                                    for k in r["carriages"]]})
    # light and screen in the solver frame: same numbers as the UI's saved 2-arm stage
    return {"bases": bases, "light": {"lat": c["scene"]["lightY"], "dep": -c["scene"]["lightX"], "h": 0.0566},
            "screen": {"dep": 0.0, "half": 1.105, "ch": 0.7366}, "rails": rails}


def rig():
    fastshadow.use_fleet()
    from rails import wrap_renderer
    from renderer import ShadowRenderer, build_scene

    stage = stage_json()
    model, data, n = build_scene("urdf/SO101/so101_new_calib.urdf", N, stage=stage)
    mj = ShadowRenderer(model, data, size=128, n_robots=n)
    mj.floor_mode = "mesh"
    br = wrap_renderer(mj, stage, True)
    return stage, mj, br


def hold(tid=None):
    return {i.split("/")[1].split("_")[0].lower() for i in bench_ids()}


def filtered(br, Q, chunk=2048):
    """Keep the poses with no contact and nothing under the table, on this stage."""
    Q = np.clip(Q, br.q_lower, br.q_upper)
    keep = np.zeros(len(Q), bool)
    for i in range(0, len(Q), chunk):
        _, z, cross, self_ = br.eval_q_batch(Q[i: i + chunk])
        keep[i: i + chunk] = (z >= FLOOR_TOL) & (cross <= 0) & (self_ <= 0)
    return Q[keep].astype(np.float32), float(keep.mean())


def build(a):
    stage, mj, br = rig()
    os.makedirs(OUT, exist_ok=True)
    arms = np.load(os.path.join(BENCH, "results", "fastshadow", "pose_library", "library_n3.npz"))["Q"].reshape(-1, 6)
    g = np.random.default_rng(0)
    nj, nv = br.n_joint, br.n_dof - br.n_joint
    print(f"stage: {N} arms, {nv} free carriages, {br.n_dof} variables; arm pool {len(arms)}", flush=True)
    t0 = time.perf_counter()
    Qa = arms[g.integers(0, len(arms), (a.n, N))].reshape(a.n, nj)
    lo, hi = br.q_lower[nj:], br.q_upper[nj:]
    Ua = g.uniform(lo, hi, (a.n, nv))
    combo, f1 = filtered(br, np.hstack([Qa, Ua]))
    print(f"combo: {len(combo)} of {a.n} pass ({f1:.3f}), {time.perf_counter() - t0:.0f} s", flush=True)
    t0 = time.perf_counter()
    Qr = g.uniform(br.q_lower, br.q_upper, (a.n, br.n_dof))
    rnd, f2 = filtered(br, Qr)
    print(f"random: {len(rnd)} of {a.n} pass ({f2:.3f}), {time.perf_counter() - t0:.0f} s", flush=True)
    np.savez_compressed(os.path.join(OUT, "libraries.npz"), combo=combo, rnd=rnd)
    json.dump(stage, open(os.path.join(OUT, "stage.json"), "w"), indent=1)


def fit_targets(mj, ids):
    """Each original glyph fitted once to the stage's reach, cached."""
    from reachability import build_reachability_map
    from target_fit import fit_target

    d = os.path.join(OUT, "fitted")
    os.makedirs(d, exist_ok=True)
    reach, out = None, {}
    for tid in ids:
        sub, stem = tid.split("/")
        p = os.path.join(d, f"{sub}__{stem}.png")
        if not os.path.exists(p):
            reach = reach or build_reachability_map(mj, n_samples=300, verbose=False)
            T0 = load_target(os.path.join(ROOT, sub, stem, f"{stem}_original.png"), 128)
            f = fit_target(T0, reach, scale_range=(0.35, 1.6), n_scales=14, max_shift_frac=0.22, n_shifts=15,
                           scale_penalty=0.0, min_retained=0.98, verbose=False)
            save_mask(f.target > 0.5, p)
        out[tid] = load_target(p, 128)
    return out


def solver():
    from optimizer import OptimizerConfig, optimize_staged

    def cma(br, T, seed, b, x0=None):
        p, p1, p2, f = b
        cfg = OptimizerConfig(seed=seed, skip_greedy=x0 is not None, popsize=p, phase1_iters=p1, phase2_iters=p2,
                              final_iters=f, floor_penalty=40.0, collision_penalty=400.0, self_collision_penalty=200.0,
                              adaptive_final=False)
        with contextlib.redirect_stdout(io.StringIO()):
            return optimize_staged(br, T, cfg, x0=None if x0 is None else np.asarray(x0, np.float64))

    return cma


def grow(a):
    """Real solves on this stage, on targets outside the test characters."""
    stage, mj, br = rig()
    cma = solver()
    chars = hold()
    ids = []
    for sub in sorted(os.listdir(ROOT)):
        d = os.path.join(ROOT, sub)
        if os.path.isdir(d):
            ids += [f"{sub}/{s}" for s in sorted(os.listdir(d))
                    if not (("letter" in sub or "digit" in sub) and s.split("_")[0].lower() in chars)]
    ids = [i for i in ids if os.path.exists(os.path.join(ROOT, *i.split("/"), i.split("/")[1] + "_original.png"))]
    np.random.default_rng(1).shuffle(ids)
    ids = ids[: a.limit]
    path = os.path.join(OUT, "grown.jsonl")
    done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
    from loss import compute_iou

    Ts = fit_targets(mj, ids)
    for k, tid in enumerate(ids):
        if tid in done:
            continue
        r = cma(br, Ts[tid], k, COLD["cold_8k"])
        m, z, cross, self_ = br.eval_q(r.best_q)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": tid, "q": [float(v) for v in r.best_q], "iou": float(compute_iou(m, Ts[tid])),
                                "clean": bool(z >= FLOOR_TOL and cross <= 0 and self_ <= 0)}) + "\n")
        if k % 20 == 0:
            print(f"[{k}/{len(ids)}]", flush=True)


def run(a):
    stage, mj, br = rig()
    import torch

    from fastshadow.pose_library import PoseLibrary
    from loss import compute_iou

    cma = solver()
    fr = br._fast
    d = np.load(os.path.join(OUT, "libraries.npz"))
    libs = {"combo": d["combo"], "rnd": d["rnd"]}
    gp = os.path.join(OUT, "grown.jsonl")
    if os.path.exists(gp):
        rows = [json.loads(l) for l in open(gp, encoding="utf-8")]
        libs["grown"] = np.array([r["q"] for r in rows if r["clean"]], np.float32)
        libs["both"] = np.vstack([libs["combo"], libs["grown"]])
    L = {}
    for k, Q in libs.items():
        t0 = time.perf_counter()
        L[k] = PoseLibrary.build(fr, Q)
        print(f"library {k}: {len(Q)} poses, {time.perf_counter() - t0:.1f} s", flush=True)

    ids = bench_ids()
    Ts = fit_targets(mj, ids)
    orig = {t: load_target(os.path.join(ROOT, *t.split("/"), t.split("/")[1] + "_original.png"), 128) for t in ids}
    path = os.path.join(OUT, "runs.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["id"], r["method"], r["seed"]) for r in map(json.loads, open(path, encoding="utf-8"))}

    def rec(tid, meth, s, q, dt):
        m, z, cross, self_ = br.eval_q(np.asarray(q, np.float64))
        lay = br.rail_arr.layout
        bad = lay.violations(br.rail_arr.S(np.asarray(q, float)[br.n_joint:]))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": tid, "method": meth, "seed": s, "seconds": dt, "iou": float(compute_iou(m, Ts[tid])),
                                "tc_iou": trimmed_centred(orig[tid] > 0.5, m > 0.5)["tc_iou"], "floor_z": float(z),
                                "clean": bool(z >= FLOOR_TOL and cross <= 0 and self_ <= 0 and not bad)}) + "\n")

    for ti, tid in enumerate(ids):
        T = Ts[tid]
        for k, lib in L.items():
            m = f"{k}_only"
            if (tid, m, 0) not in done:
                t0 = time.perf_counter()
                q, _ = lib.query(T, 1)
                torch.cuda.synchronize()
                rec(tid, m, 0, q[0], time.perf_counter() - t0)
        for s in range(a.seeds):
            for m, b in COLD.items():
                if (tid, m, s) not in done:
                    t0 = time.perf_counter()
                    r = cma(br, T, s, b)
                    rec(tid, m, s, r.best_q, time.perf_counter() - t0)
            for k, lib in L.items():
                for w, b in WARM.items():
                    m = f"{k}_{w}"
                    if (tid, m, s) not in done and (k != "rnd" or w == "f30"):
                        t0 = time.perf_counter()
                        q, _ = lib.query(T, 1)
                        r = cma(br, T, s, b, q[0])
                        rec(tid, m, s, r.best_q, time.perf_counter() - t0)
        print(f"[{ti + 1}/{len(ids)}] {tid}", flush=True)


def export(a):
    """The library the solvers start from: every clean solve, then the combo poses."""
    fleet = fastshadow.use_fleet()
    d = np.load(os.path.join(OUT, "libraries.npz"))
    rows = [json.loads(l) for l in open(os.path.join(OUT, "grown.jsonl"), encoding="utf-8")]
    grown = np.array([r["q"] for r in rows if r["clean"]], np.float32)
    combo = d["combo"]
    Q = np.vstack([grown, combo[np.random.default_rng(0).permutation(len(combo))]])
    st = stage_json()
    home = np.array([c["s"] for r in st["rails"] for c in r["carriages"]])
    order = np.argsort([c["arm"] for r in st["rails"] for c in r["carriages"]])
    out = os.path.join(fleet, "datasets", "pose_library")
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "source.npz"), Q=Q, home=home[order],
                        bases=np.array([[b["lat"], b["dep"]] for b in st["bases"]]),
                        yaw=np.array([b["yaw_deg"] for b in st["bases"]]))
    print("exported", len(Q), "poses,", len(grown), "of them real solves")


def report(a):
    rows = [json.loads(l) for l in open(os.path.join(OUT, "runs.jsonl"), encoding="utf-8")]
    ms = sorted({r["method"] for r in rows})
    ids = sorted({r["id"] for r in rows})
    per = lambda m, k: np.array([np.mean([r[k] for r in rows if r["id"] == i and r["method"] == m]) for i in ids if any(r["id"] == i and r["method"] == m for r in rows)])  # noqa: E731
    tab = sorted(((m, per(m, "iou").mean(), per(m, "tc_iou").mean(), per(m, "seconds").mean(),
                   np.mean([r["clean"] for r in rows if r["method"] == m])) for m in ms), key=lambda x: x[3])
    print(f"{'method':<16}{'IoU':>8}{'tc_iou':>8}{'s/solve':>9}{'clean':>7}")
    for t in tab:
        print(f"{t[0]:<16}{t[1]:>8.3f}{t[2]:>8.3f}{t[3]:>9.3f}{t[4]:>7.2f}")
    g = np.random.default_rng(0)
    for x, y in [(f"{k}_{w}", c) for k in ("combo", "grown", "both") for w, c in (("f30", "cold_8k"), ("f16", "cold_3k"))]:
        if x not in ms or y not in ms:
            continue
        for key in ("iou", "tc_iou"):
            dd = per(x, key) - per(y, key)
            b = dd[g.integers(0, len(dd), (10000, len(dd)))].mean(1)
            print(f"{x} - {y} {key}: {dd.mean():+.3f} [{np.percentile(b, 2.5):+.3f} {np.percentile(b, 97.5):+.3f}]")
        print(f"  speedup {per(y, 'seconds').mean() / per(x, 'seconds').mean():.1f}x")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "grow", "run", "report", "export"])
    p.add_argument("--n", type=int, default=300000)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()
    {"build": build, "grow": grow, "run": run, "report": report, "export": export}[a.cmd](a)
