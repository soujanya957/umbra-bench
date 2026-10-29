"""Gradient-solver sweep for the solver bench: Adam vs L-BFGS vs Levenberg-Marquardt, Adam
with basin hopping (against Adam with twice the starts, about the same time) and Adam with
annealed Langevin noise. (SVGD was dropped after tuning: with a kernel narrow enough not to mix
gradients from different basins, the starts never interact.)

Setup is compare_fast_cma.py's, unchanged: the same 30 letters/digits (compare_<src>
per_target.jsonl), the same objective and penalties (sphere proxy calibrated on solves of
other glyphs), the same start selection (2048 random poses, best R by hard loss, identical
across methods for a seed), the same blur schedule and the same final pick (best 8 by fast
hard loss, re-measured and chosen by MuJoCo). Only the local search differs. Every number is
MuJoCo-measured; the glyph classifier (DejaVu held out) also reads every final shadow.

  --set tune   8 other letters (characters not in the bench, all three DejaVu faces), 1 seed
  --set bench  the bench's 30 targets, --seeds seeds (default 3: gradients from the nvdiffrast
               backward are not bit-reproducible, so a seed fixes the starts, not the result)

Protocol (after review): one untimed warm-up; target-major order with the method order
rotated per target and seed; every row stores its pose, the true floor check (lowest moving
vertex) and its mask, and a hash of the method config + solver code (resume key).

Methods are named <local>[_<variant>]_R<starts>; see METHODS. Output:
results/fastshadow/gradsweep_<src>/{tune,runs}.jsonl and masks/ (every seed).

    python scripts/grad_sweep.py --set tune
    python scripts/grad_sweep.py --set bench --seeds 3
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

# local search, starts, its iterations and keyword arguments; "hop" = basin-hopping rounds.
# Every method reseeds at blur-stage boundaries like Adam does (reseed_sigma 0.15). Tuning
# history in tune.jsonl: pass 1 (langevin 0.03, plain-median SVGD, L-BFGS/LM without
# reseeding, R256) and pass 2 (*_reseed, langevin01, svgdlocal, adam_hop) ran before the code
# review; v2 names run the reviewed solvers (fastshadow.local_opt) and the fixed protocol.
METHODS = {
    "adam_R64": dict(local="adam", R=64, iters=300),
    "adam_R128": dict(local="adam", R=128, iters=300),
    "adam_langevin01_R64": dict(local="adam", R=64, iters=300, langevin=0.01),
    "adam_hop2_R64": dict(local="adam", R=64, iters=300, hop=3, hop_iters=60),
    "lbfgs2_R64": dict(local="lbfgs", R=64, iters=60),
    "lm2_R64": dict(local="lm", R=64, iters=40),
}


def config_hash(meth):
    import hashlib

    src = "".join(open(os.path.join(BENCH, "fastshadow", f), encoding="utf-8").read() for f in ("local_opt.py", "grad_opt.py"))
    return hashlib.sha1((json.dumps(METHODS[meth], sort_keys=True) + src).encode()).hexdigest()[:10]


def tune_ids(root, bench_ids, n=8):
    """Letters of the same results tree whose character is not a bench character (any font),
    round-robin over the three DejaVu faces so serif and mono are tuned on too."""
    used = {(t.split("/")[0], t.split("/")[1].split("_")[0]) for t in bench_ids}
    by_font = {}
    for sub in ("digits", "letters_upper", "letters_lower"):
        d = os.path.join(root, sub)
        for s in sorted(os.listdir(d)):
            if os.path.exists(os.path.join(d, s, "results.json")) and (sub, s.split("_")[0]) not in used:
                by_font.setdefault(s.split("_", 1)[1], []).append(f"{sub}/{s}")
    g = np.random.default_rng(0)
    pools = {f: list(g.permutation(v)) for f, v in sorted(by_font.items())}
    out, seen = [], set()
    while len(out) < n and any(pools.values()):
        for f in sorted(pools):
            while pools[f]:
                t = str(pools[f].pop())
                ch = (t.split("/")[0], t.split("/")[1].split("_")[0])
                if ch not in seen:  # one font per character
                    seen.add(ch)
                    out.append(t)
                    break
            if len(out) == n:
                break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="letters_digits")
    p.add_argument("--results", default=os.path.join(BENCH, "optimized", "big-budget-grounded"))
    p.add_argument("--set", choices=["tune", "bench"], default="bench")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--methods", nargs="*", default=list(METHODS))
    a = p.parse_args()
    root = os.path.abspath(a.results)
    src = os.path.join(BENCH, "results", "fastshadow", f"compare_{a.src}")
    outdir = os.path.join(BENCH, "results", "fastshadow", f"gradsweep_{a.src}")
    os.makedirs(os.path.join(outdir, "masks"), exist_ok=True)
    bench_ids = [json.loads(l)["id"] for l in open(os.path.join(src, "per_target.jsonl"), encoding="utf-8")]
    ids = bench_ids if a.set == "bench" else tune_ids(root, bench_ids)
    seeds = a.seeds if a.set == "bench" else 1

    fastshadow.use_fleet()
    import torch
    from renderer import ShadowRenderer, build_scene

    from fastshadow.glyph_cls import GlyphClassifier
    from fastshadow.grad_opt import GradConfig, Objective, Penalties, pick_by_mujoco, verify
    from fastshadow.local_opt import basin_hop, random_starts, refine
    from fastshadow.renderer import FastShadowRenderer

    first = json.load(open(os.path.join(root, *bench_ids[0].split("/"), "results.json"), encoding="utf-8"))
    rig, ocfg = first["rig"], first["optimizer"]
    model, data, n = build_scene("urdf/SO101/so101_new_calib.urdf", n_robots=rig["n_robots"], arm_gap=rig["arm_gap_m"])
    mj = ShadowRenderer(model, data, size=rig["render_size"], n_robots=n)
    fr = FastShadowRenderer(mj)
    clf = GlyphClassifier()

    # penalties calibrated as compare_fast_cma.py does (solves of non-bench glyphs); in tune
    # mode the tuning glyphs are left out of the calibration as well
    solved, picked = [], set(bench_ids) | set(ids)
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

    path = os.path.join(outdir, "runs.jsonl" if a.set == "bench" else "tune.jsonl")
    hashes = {m: config_hash(m) for m in a.methods}
    done = set()
    if os.path.exists(path):
        done = {(r["id"], r["method"], r["seed"], r.get("config")) for r in map(json.loads, open(path, encoding="utf-8"))}

    def gcfg_of(meth, s):
        M = METHODS[meth]
        return GradConfig(floor_penalty=ocfg["floor_penalty"], collision_penalty=ocfg["collision_penalty"],
                          self_collision_penalty=ocfg["self_collision_penalty"], reseed_sigma=0.15,
                          n_restarts=M["R"], seed=s, top_k=8, langevin=M.get("langevin", 0.0),
                          svgd_T=M.get("svgd_T", 0.0), svgd_bw=M.get("svgd_bw", 1.0))

    def solve_one(meth, T, gcfg, iters_scale=1.0):
        M = dict(METHODS[meth])
        local, iters = M.pop("local"), max(2, int(M.pop("iters") * iters_scale))
        M.pop("R")
        rounds, hop_iters = M.pop("hop", 0), M.pop("hop_iters", 0)
        kw = {k: v for k, v in M.items() if k not in ("langevin", "svgd_T", "svgd_bw")}
        obj = Objective(fr, pen, T, gcfg)
        Q0 = random_starts(obj, gcfg)
        if rounds:
            return local, basin_hop(local, obj, Q0, gcfg, iters, gcfg.blur_sigmas, rounds, max(2, int(hop_iters * iters_scale)), **kw)
        return local, refine(local, obj, Q0, gcfg, iters, gcfg.blur_sigmas, **kw)

    # warm-up: CUDA, nvdiffrast and autograd kernels, untimed, every local search once
    sub0, stem0 = ids[0].split("/")
    Tw = load_target(os.path.join(root, sub0, stem0, f"{stem0}_shown.png"), mj.size)
    for meth in {METHODS[m]["local"]: m for m in a.methods}.values():
        w = gcfg_of(meth, 0)
        w.n_init, w.n_restarts = 64, 8
        solve_one(meth, Tw, w, iters_scale=0.1)
    torch.cuda.synchronize()

    # target-major: every method on a target before the next target, method order rotated per
    # target and seed, so GPU contention and drift do not line up with one method
    for ti, tid in enumerate(ids):
        sub, stem = tid.split("/")
        d = os.path.join(root, sub, stem)
        T = load_target(os.path.join(d, f"{stem}_shown.png"), mj.size)
        T0 = load_target(os.path.join(d, f"{stem}_original.png"), mj.size)
        for s in range(seeds):
            k = (ti * seeds + s) % len(a.methods)
            for meth in a.methods[k:] + a.methods[:k]:
                if (tid, meth, s, hashes[meth]) in done:
                    continue
                gcfg = gcfg_of(meth, s)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                local, (bq, bl, nh, ns) = solve_one(meth, T, gcfg)
                top = bq[torch.argsort(bl)[: gcfg.top_k]].cpu().numpy().astype(np.float64)
                q, _ = pick_by_mujoco(mj, top, T, gcfg)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                v = verify(mj, q, T, gcfg)
                m = v.pop("mask")
                rk, pc = clf.rank(torch.tensor(m > 0.5, dtype=torch.float32)[None], sub, stem.split("_")[0])
                low = float(fr.moving_min_z(q[None])[0]) * 1000
                save_mask(m, os.path.join(outdir, "masks", f"{tid.replace('/', '__')}__{meth}__s{s}.png"))
                M = METHODS[meth]
                rec = {"id": tid, "method": meth, "seed": s, "config": hashes[meth], "seconds": dt, "local": local,
                       "restarts": M["R"], "iters": M["iters"], "hop": M.get("hop", 0), "hop_iters": M.get("hop_iters", 0),
                       "n_soft": int(ns), "soft_kind": "fwd" if local == "lm" else "fwd+bwd", "n_hard": int(gcfg.n_init + nh),
                       "iou": v["iou"], "loss": v["loss"], "tc_iou": trimmed_centred(T0 > 0.5, m > 0.5)["tc_iou"],
                       "collision_free": bool(v["cross_pen_m"] <= 0 and v["self_pen_m"] <= 0),
                       "lowest_mm": round(low, 1), "floor_ok": bool(low >= -5),
                       "cls_top1": bool(int(rk[0]) == 1), "cls_p": float(pc[0]), "langevin": gcfg.langevin,
                       "svgd_T": gcfg.svgd_T, "svgd_bw": gcfg.svgd_bw, "q_rad": [round(float(x), 6) for x in q]}
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
        print(f"[{ti + 1}/{len(ids)}] {tid}", flush=True)


if __name__ == "__main__":
    main()
