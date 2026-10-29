"""LeSlider rail: carriages that slide along the light axis, optimized with the joints.

Rig: fleet-shadow-art's 1 m LeSlider track (track.leslider_1m: light 1 m before the rail,
screen 1 m after it, base travel 0.13-0.95 m from the light end, carriages >= 0.179 m apart)
with 3 SO-101s, built by fleet's build_scene(track=...). Each carriage's position is one more
variable (fastshadow.spacing.RailArrangement), so the search moves arm i's distance to the
light (its shadow's size) and where its shadow lands together with its joints, in one
batched solve on the GPU renderer, instead of fleet's outer search (a short MuJoCo solve
per candidate layout).

Configs, same targets, same solvers:
  rail_even   carriages held evenly spread (the rail without moving carriages)
  rail_free   carriage positions optimized with the 18 joints (21 variables)
Optimizers: staged CMA-ES (populations on nvdiffrast, MuJoCo kinematics + contacts, mesh
floor) and multi-start Adam (soft renderer; the carriages get a gradient through FK).

Targets: the solver bench's 30 letters/digits, each original glyph fitted once to this rig's
reach (fleet target_fit, reach map with the carriages evenly spread, so the fit if anything
favours rail_even); both configs chase the same fitted target. Across rigs (the old fixed
0.2 m column), compare tc_iou against the original glyph and the classifier, not IoU.

Every result is re-rendered and contact-checked by MuJoCo with the carriages where the
solver put them.

    python scripts/rail_study.py check     # GPU vs MuJoCo on random poses + layouts
    python scripts/rail_study.py fit       # targets fitted to the rail's reach
    python scripts/rail_study.py run       # --cma-seeds 3 --adam-seeds 2
    python scripts/rail_study.py fit|run|report --light-dist 0.5   # another rail placement
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

OUT = os.path.join(BENCH, "results", "fastshadow", "rail_study")
GEOM = {"light_dist": 1.0, "screen_dist": 1.0}  # fleet track.leslider_1m defaults; --light-dist / --screen-dist
ROOT = os.path.join(BENCH, "optimized", "big-budget-grounded")
CONFIGS = ["rail_even", "rail_free"]
N = 3


def bench_ids():
    src = os.path.join(BENCH, "results", "fastshadow", "compare_letters_digits", "per_target.jsonl")
    return [json.loads(l)["id"] for l in open(src, encoding="utf-8")]


def scene():
    fastshadow.use_fleet()
    from renderer import ShadowRenderer, build_scene
    from track import leslider_1m

    track = leslider_1m(**GEOM)
    s0 = track.evenly_spread(N)
    first = json.load(open(os.path.join(ROOT, *bench_ids()[0].split("/"), "results.json"), encoding="utf-8"))
    model, data, n = build_scene("urdf/SO101/so101_new_calib.urdf", n_robots=N, track=track, track_s=list(s0))
    mj = ShadowRenderer(model, data, size=first["rig"]["render_size"], n_robots=n)
    return track, s0, mj, first["optimizer"]


def check(a):
    import torch

    from fastshadow.renderer import FastShadowRenderer
    from fastshadow.spacing import RailArrangement, SpacedMJ

    track, s0, mj, _ = scene()
    arr = RailArrangement(track, N)
    g = np.random.default_rng(0)
    U = g.uniform(0, 1, (4000, N))
    S = arr.s_of(U)
    ok = np.all(np.diff(S, axis=1) >= track.min_spacing - 1e-9, axis=1) & (S[:, 0] >= track.s_min - 1e-9) \
        & (S[:, -1] <= track.s_max + 1e-9)
    inv = max(float(np.abs(arr.s_of(arr.u_of(s)[None])[0] - s).max()) for s in S[:200])
    print(f"stick-breaking: {ok.mean():.3f} of 4000 layouts feasible, inverse max error {inv:.2e} m, "
          f"evenly spread u = {np.round(arr.u_of(s0), 3)}")
    fr = FastShadowRenderer(mj, arrangement=arr)
    sm = SpacedMJ(mj, arr)
    out = []
    for k in range(24):
        q = np.concatenate([g.uniform(mj.q_lower, mj.q_upper), g.uniform(0, 1, N)])
        m_mj = sm.get_shadow_mask(q) > 0.5
        m_fr = fr.render(torch.tensor(q[None], dtype=torch.float32, device=fr.device))[0].cpu().numpy() > 0.5
        u = (m_mj | m_fr).sum()
        out.append(float((m_mj & m_fr).sum() / u) if u else 1.0)
    print(f"GPU vs MuJoCo, 24 random poses + layouts: IoU mean {np.mean(out):.4f}, min {np.min(out):.4f}")
    mags = [(track.light_y - track.wall_y) / (track.light_y - y) for y in track.world_y([track.s_min, track.s_max])]
    print(f"rail: s {track.s_min:.3f}-{track.s_max:.3f} m, spacing {track.min_spacing:.4f} m, slack {arr.slack:.3f} m, "
          f"magnification {mags[1]:.2f}x (far end) to {mags[0]:.2f}x (light end)")


def fit(a):
    fastshadow.use_fleet()
    from reachability import build_reachability_map
    from target_fit import fit_target

    _, _, mj, _ = scene()
    reach = build_reachability_map(mj, n_samples=300, verbose=False)
    os.makedirs(os.path.join(OUT, "fitted"), exist_ok=True)
    fits = {}
    for tid in bench_ids():
        sub, stem = tid.split("/")
        T0 = load_target(os.path.join(ROOT, sub, stem, f"{stem}_original.png"), 128)
        f = fit_target(T0, reach, scale_range=(0.35, 1.6), n_scales=14, max_shift_frac=0.22, n_shifts=15,
                       scale_penalty=0.0, min_retained=0.98, verbose=False)
        save_mask(f.target > 0.5, os.path.join(OUT, "fitted", f"{sub}__{stem}.png"))
        fits[tid] = f.as_dict()
    json.dump(fits, open(os.path.join(OUT, "fits.json"), "w"), indent=1)
    print("fitted", len(fits), "targets")


def run(a):
    track, s0, mj, ocfg = scene()  # puts fleet-shadow-art on the path
    import optimizer as opt
    import torch
    from optimizer import OptimizerConfig, optimize_staged

    from fastshadow.cma_bridge import BatchedShadowRenderer
    from fastshadow.glyph_cls import GlyphClassifier
    from fastshadow.grad_opt import GradConfig, Penalties, pick_by_mujoco, solve
    from fastshadow.objectives import TargetCache, use_objective
    from fastshadow.renderer import FastShadowRenderer
    from fastshadow.spacing import RailArrangement, SpacedMJ

    assert "batch_shadow_loss" in open(opt.__file__, encoding="utf-8").read(), "fleet-shadow-art lacks the batched hooks"
    arr = RailArrangement(track, N)
    u_even = arr.u_of(s0)
    home = mj.get_current_q().copy()
    fr0 = FastShadowRenderer(mj)  # bases where the model was built: evenly spread
    frs = {"rail_even": fr0, "rail_free": FastShadowRenderer(mj, arrangement=arr)}
    # collision proxy calibrated on contact-free solves of other glyphs, carriages evenly spread
    solved, picked = [], set(bench_ids())
    for sub in os.listdir(ROOT):
        d = os.path.join(ROOT, sub)
        if os.path.isdir(d):
            for stem in os.listdir(d):
                f = os.path.join(d, stem, "results.json")
                if f"{sub}/{stem}" not in picked and os.path.exists(f):
                    solved += [r["q_rad"] for r in json.load(open(f, encoding="utf-8"))["runs"]]
    solved = np.array(solved)
    np.random.default_rng(0).shuffle(solved)
    pen = Penalties(fr0, solved_q=solved[:3000], floor="mesh")
    clf = GlyphClassifier()
    ids = bench_ids()[: a.limit or None]
    os.makedirs(os.path.join(OUT, "masks"), exist_ok=True)
    path = os.path.join(OUT, "runs.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["id"], r["config"], r["optimizer"], r["seed"]) for r in map(json.loads, open(path, encoding="utf-8"))}
    json.dump({"track": track.as_dict(), "n": N, "s_even": [float(v) for v in s0], "configs": CONFIGS, "targets": ids,
               "args": vars(a)}, open(os.path.join(OUT, "meta.json"), "w"), indent=2)

    def view(config, u=None):
        """MuJoCo with the carriages placed (rail_even: evenly spread, q has 18 entries)."""
        sm = SpacedMJ(mj, arr, u_even if u is None else u)
        sm.set_qpos(np.concatenate([home, u_even if u is None else u]))
        return sm if config == "rail_free" else mj

    def full_q(config, q):
        return np.concatenate([q, u_even]) if config == "rail_even" else np.asarray(q, float)

    def record(tid, config, optname, seed, q, dt, T, T0, extra):
        qf = full_q(config, q)
        sm = SpacedMJ(mj, arr, qf[18:])
        mask, min_z, cross, self_ = sm.eval_q(qf)
        S = mask > 0.5
        sub, stem = tid.split("/")
        mp = os.path.join("masks", f"{sub}__{stem}__{config}__{optname}__s{seed}.png")
        save_mask(mask, os.path.join(OUT, mp))
        inter, union = (S & (T > 0.5)).sum(), (S | (T > 0.5)).sum()
        rk, pc = clf.rank(torch.tensor(S[None], dtype=torch.float32), sub, stem.split("_")[0])
        low = float(frs["rail_free"].moving_min_z(qf[None])[0]) * 1000
        s = arr.s_of(qf[18:][None])[0]
        rec = {"id": tid, "config": config, "optimizer": optname, "seed": seed, "seconds": round(dt, 3),
               "iou": float(inter / union) if union else 0.0, "tc_iou": trimmed_centred(T0 > 0.5, S)["tc_iou"],
               "cls_top1": bool(int(rk[0]) == 1), "cls_p": float(pc[0]),
               "collision_free": bool(cross <= 0 and self_ <= 0), "lowest_mm": round(low, 1), "floor_ok": bool(low >= -5),
               "s_m": [round(float(v), 4) for v in s],
               "magnification": [round((track.light_y - track.wall_y) / (track.light_y - y), 3) for y in track.world_y(s)],
               "mask": mp, "q": [round(float(v), 6) for v in qf], **extra}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    t_all = time.time()
    for ti, tid in enumerate(ids):
        sub, stem = tid.split("/")
        T = load_target(os.path.join(OUT, "fitted", f"{sub}__{stem}.png"), mj.size)
        T0 = load_target(os.path.join(ROOT, sub, stem, f"{stem}_original.png"), mj.size)
        tc = TargetCache(sub, stem.split("_")[0])
        for config in CONFIGS:
            fr = frs[config]
            for s in range(a.cma_seeds):
                if (tid, config, "cma", s) in done:
                    continue
                u0 = None if (s == 0 or config == "rail_even") else np.random.default_rng(1000 + s).uniform(0, 1, N)
                v = view(config, u0)
                br = BatchedShadowRenderer(v, fr, floor="mesh")
                use_objective(opt, br, "dt2", tc)
                t0 = time.perf_counter()
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        res = optimize_staged(br, T, OptimizerConfig(seed=s, **ocfg))
                finally:
                    use_objective(opt, br, None)
                record(tid, config, "cma", s, np.asarray(res.best_q), time.perf_counter() - t0, T, T0,
                       {"n_evals": int(res.n_evals)})
            for s in range(a.adam_seeds):
                if (tid, config, "adam", s) in done:
                    continue
                cfg = GradConfig(floor_penalty=ocfg["floor_penalty"], collision_penalty=ocfg["collision_penalty"],
                                 self_collision_penalty=ocfg["self_collision_penalty"], reseed_sigma=0.15, seed=s,
                                 n_restarts=a.adam_restarts)
                t0 = time.perf_counter()
                r = solve(fr, pen, T, cfg)
                v = SpacedMJ(mj, arr) if config == "rail_free" else view(config)
                q, _ = pick_by_mujoco(v, r.top_q, T, cfg)
                record(tid, config, "adam", s, q, time.perf_counter() - t0, T, T0, {"n_restarts": a.adam_restarts})
        print(f"[{ti + 1}/{len(ids)}] {tid}  {(time.time() - t_all) / 60:.1f} min", flush=True)


def report(a):
    """summary.json: per config x optimizer means; rail_free minus rail_even paired by character
    (IoU on the shared fitted target, tc_iou against the original glyph); the old fixed 0.2 m
    column from the solver bench (fastcma_letters_digits: cma_fast seeds 0-2, adam_64 seeds
    0-1) on tc_iou and the classifier, since its IoU is against a different fitted target."""
    from collections import defaultdict

    import torch

    from fastshadow.glyph_cls import GlyphClassifier

    rows = [json.loads(l) for l in open(os.path.join(OUT, "runs.jsonl"), encoding="utf-8")]
    ids = sorted({r["id"] for r in rows})
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["config"], r["optimizer"])][r["id"]].append(r)
    g = np.random.default_rng(0)

    def ci(d):
        d = np.asarray(d, float)
        b = [d[g.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        return [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)),
                int((d > 0.005).sum()), int((d < -0.005).sum())]

    def per_target(key, cell, best=False):
        return np.array([max(r[key] for r in cell[i]) if best else np.mean([r[key] for r in cell[i]]) for i in ids])

    out = {"n_targets": len(ids), "cells": {}, "free_minus_even": {}}
    for (config, optname), cell in sorted(by.items()):
        rr = [r for i in ids for r in cell[i]]
        out["cells"][f"{config}/{optname}"] = {
            "iou": float(per_target("iou", cell).mean()), "iou_best": float(per_target("iou", cell, True).mean()),
            "tc_iou": float(per_target("tc_iou", cell).mean()), "cls_top1": float(np.mean([r["cls_top1"] for r in rr])),
            "collision_free": float(np.mean([r["collision_free"] for r in rr])), "floor_ok": float(np.mean([r["floor_ok"] for r in rr])),
            "seconds": float(np.mean([r["seconds"] for r in rr])), "n_seeds": len(next(iter(cell.values()))),
            "s_mean": np.round(np.mean([r["s_m"] for r in rr], 0), 3).tolist(),
            "mag_mean": np.round(np.mean([r["magnification"] for r in rr], 0), 3).tolist()}
    for optname in ("cma", "adam"):
        e, f = by[("rail_even", optname)], by[("rail_free", optname)]
        if all(i in e and i in f for i in ids):
            out["free_minus_even"][optname] = {
                k: ci(per_target(k, f) - per_target(k, e)) for k in ("iou", "tc_iou", "cls_p")}
            out["free_minus_even"][optname]["iou_best"] = ci(per_target("iou", f, True) - per_target("iou", e, True))
    # the old rig: the bench's fixed 0.2 m column
    fd = os.path.join(BENCH, "results", "fastshadow", "fastcma_letters_digits")
    old = defaultdict(lambda: defaultdict(list))
    for r in map(json.loads, open(os.path.join(fd, "runs.jsonl"), encoding="utf-8")):
        if r["id"] in ids and ((r["method"] == "cma_fast" and r["seed"] < 3) or (r["method"] == "adam_64" and r["seed"] < 2)):
            old[r["method"]][r["id"]].append(r)
    clf = GlyphClassifier()
    reads = []
    for i in ids:
        sub, stem = i.split("/")
        for s in range(3):
            m = load_target(os.path.join(fd, f"{sub}__{stem}__cma_fast{s}.png"), 128) > 0.5
            rk, _ = clf.rank(torch.tensor(m[None], dtype=torch.float32), sub, stem.split("_")[0])
            reads.append(int(rk[0]) == 1)
    out["old_rig"] = {"cma_fast": {"tc_iou": float(per_target("tc_iou", old["cma_fast"]).mean()),
                                   "cls_top1": float(np.mean(reads)), "n_seeds": 3},
                      "adam_64": {"tc_iou": float(per_target("tc_iou", old["adam_64"]).mean()), "n_seeds": 2}}
    for optname, om in (("cma", "cma_fast"), ("adam", "adam_64")):
        for config in CONFIGS:
            cell = by[(config, optname)]
            if all(i in cell for i in ids):
                out.setdefault("rail_minus_old_tc", {})[f"{config}/{optname}"] = ci(per_target("tc_iou", cell) - per_target("tc_iou", old[om]))
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["check", "fit", "run", "report"])
    p.add_argument("--cma-seeds", type=int, default=3)
    p.add_argument("--adam-seeds", type=int, default=2)
    p.add_argument("--adam-restarts", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--light-dist", type=float, default=1.0, help="light -> rail light end (m)")
    p.add_argument("--screen-dist", type=float, default=1.0, help="rail far end -> screen (m)")
    a = p.parse_args()
    global OUT
    GEOM.update(light_dist=a.light_dist, screen_dist=a.screen_dist)
    if (a.light_dist, a.screen_dist) != (1.0, 1.0):  # each geometry in its own folder
        OUT = f"{OUT}_light{a.light_dist:g}_screen{a.screen_dist:g}"
    {"check": check, "fit": fit, "run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
