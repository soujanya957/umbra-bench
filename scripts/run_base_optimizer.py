#!/usr/bin/env python3
"""Run the base UMBRA optimizer over the benchmark: N independent solves per target.

"Base" means the optimizer as it stands — staged CMA-ES, no target deformation. Each
target is solved `--runs` times from different seeds, because a single CMA-ES run is
a sample, not a measurement: on this rig the same target with the same budget spans
~0.09 IoU across three seeds. Reporting one number per target would be reporting
which restart got lucky.

Layout mirrors `targets/`, one folder per input:

    optimized/base-optimizer/<subset>/<stem>/
        <stem>_run00.png … <stem>_run09.png   cast shadow, 1-bit, dark = shadow
        <stem>_best.png                       the run with the highest IoU
        results.json                          every run's IoU, joints, config

Scene and renderer are built ONCE per process and reused across every target — the
build costs more than several solves do. Work is split by `--shard`, so N independent
processes cover the benchmark without sharing a GL context (MuJoCo's EGL context does
not survive being forked).

Already-finished targets are skipped unless `--force`, so an interrupted overnight run
resumes instead of restarting.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)


def load_target(path: str, size: int) -> np.ndarray:
    """Benchmark PNGs are 1-bit, dark = shape — the same convention run.py uses.

    Resized with NEAREST rather than LANCZOS: these are binary masks, and a smooth
    filter turns a 1-bit edge into a grey ramp that then gets re-thresholded at 128,
    which quietly erodes thin strokes. The benchmark deliberately includes targets
    with `min_stroke_width_rel` near 0.02 — about 2px at 128 — so that erosion is not
    hypothetical.
    """
    img = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    t = (np.array(img) < 128).astype(np.float32)
    corners = t[0, 0] + t[0, -1] + t[-1, 0] + t[-1, -1]
    if corners >= 3.0 and t.mean() > 0.5:
        raise ValueError(f"{path} looks stored inverted (fill {100 * t.mean():.0f}%)")
    return t


def save_mask(mask: np.ndarray, path: str):
    """Write a 0/1 mask as 1-bit dark-on-white, matching targets/."""
    m = (np.asarray(mask) > 0.5).astype(np.uint8)
    Image.fromarray(((1 - m) * 255).astype(np.uint8), "L").convert("1").save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=os.path.expanduser("~/dev/fleet-shadow-art"))
    p.add_argument("--bench", default=_BENCH)
    p.add_argument(
        "--subsets", nargs="+", default=["letters_upper", "digits", "animals", "abstract"]
    )
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--n-robots", type=int, default=3)
    p.add_argument("--arm-gap", type=float, default=0.20)
    p.add_argument("--light-to-front", type=float, default=None)
    p.add_argument("--back-to-wall", type=float, default=None)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--popsize", type=int, default=32)
    p.add_argument("--phase1-iters", type=int, default=8)
    p.add_argument("--phase2-iters", type=int, default=8)
    p.add_argument("--final-iters", type=int, default=10)
    p.add_argument("--floor-penalty", type=float, default=40.0)
    p.add_argument("--collision-penalty", type=float, default=400.0)
    p.add_argument("--self-collision-penalty", type=float, default=200.0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--n-workers",
        type=int,
        default=0,
        help="Optimizer threads per process (0 = auto, which picks 8). Running S "
        "shards at W workers oversubscribes above S*W = core count",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    ms = os.path.join(a.repo, "motion-aware-shadow")
    sys.path.insert(0, ms)
    sys.path.insert(0, os.path.join(ms, "targets"))
    os.chdir(ms)  # URDF meshes resolve relative to the package root

    from loss import compute_iou
    from optimizer import OptimizerConfig, optimize_staged
    from renderer import ShadowRenderer, build_scene

    outroot = a.out or os.path.join(a.bench, "optimized", "base-optimizer")

    # ── Collect work ─────────────────────────────────────────────────────────
    jobs = []
    for sub in a.subsets:
        d = os.path.join(a.bench, "targets", sub)
        if not os.path.isdir(d):
            print(f"[!] missing subset {d}", flush=True)
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(".png"):
                jobs.append((sub, os.path.join(d, fn)))
    jobs = [j for i, j in enumerate(jobs) if i % a.num_shards == a.shard]
    print(f"[shard {a.shard}/{a.num_shards}] {len(jobs)} targets × {a.runs} runs", flush=True)

    # ── Scene: built once, reused for every target ───────────────────────────
    urdf = os.path.join(ms, "urdf/SO101/so101_new_calib.urdf")
    model, data, n_robots = build_scene(
        urdf,
        n_robots=a.n_robots,
        arm_gap=a.arm_gap,
        light_to_front=a.light_to_front,
        back_to_wall=a.back_to_wall,
    )
    renderer = ShadowRenderer(model, data, size=a.size, n_robots=n_robots)
    print(f"[shard {a.shard}] n_dof={renderer.n_dof} arm_gap={a.arm_gap}", flush=True)

    cfg_common = dict(
        popsize=a.popsize,
        phase1_iters=a.phase1_iters,
        phase2_iters=a.phase2_iters,
        final_iters=a.final_iters,
        floor_penalty=a.floor_penalty,
        collision_penalty=a.collision_penalty,
        self_collision_penalty=a.self_collision_penalty,
        n_workers=a.n_workers,
    )
    rig = {
        "n_robots": n_robots,
        "arm_gap_m": a.arm_gap,
        "light_to_front_m": a.light_to_front,
        "back_to_wall_m": a.back_to_wall,
        "render_size": a.size,
        "distortion": False,
    }

    t_start = time.time()
    done = 0
    for sub, tpath in jobs:
        stem = os.path.splitext(os.path.basename(tpath))[0]
        odir = os.path.join(outroot, sub, stem)
        rjson = os.path.join(odir, "results.json")
        if os.path.exists(rjson) and not a.force:
            done += 1
            continue
        os.makedirs(odir, exist_ok=True)

        try:
            T = load_target(tpath, a.size)
        except Exception as e:
            print(f"[shard {a.shard}] SKIP {sub}/{stem}: {e}", flush=True)
            continue

        runs, best_i, best_iou = [], -1, -1.0
        t0 = time.time()
        for k in range(a.runs):
            res = optimize_staged(
                renderer, T, OptimizerConfig(seed=k, **cfg_common)
            )
            shadow = renderer.get_shadow_mask(res.best_q)
            iou = float(compute_iou(shadow, T))
            save_mask(shadow, os.path.join(odir, f"{stem}_run{k:02d}.png"))
            runs.append(
                {
                    "run": k,
                    "seed": k,
                    "iou": round(iou, 4),
                    "loss": round(float(res.best_loss), 4),
                    "n_evals": int(res.n_evals),
                    "q_rad": [round(float(v), 6) for v in res.best_q],
                }
            )
            if iou > best_iou:
                best_iou, best_i = iou, k
                save_mask(shadow, os.path.join(odir, f"{stem}_best.png"))

        ious = [r["iou"] for r in runs]
        with open(rjson, "w") as f:
            json.dump(
                {
                    "id": f"{sub}_{stem}",
                    "subset": sub,
                    "target": os.path.relpath(tpath, a.bench),
                    "method": "base-optimizer",
                    "rig": rig,
                    "optimizer": cfg_common,
                    "n_runs": a.runs,
                    "best_run": best_i,
                    "best_iou": round(best_iou, 4),
                    "mean_iou": round(float(np.mean(ious)), 4),
                    "std_iou": round(float(np.std(ious)), 4),
                    "min_iou": round(float(np.min(ious)), 4),
                    "seconds": round(time.time() - t0, 1),
                    "runs": runs,
                },
                f,
                indent=2,
            )
        done += 1
        el = time.time() - t_start
        eta = el / max(done, 1) * (len(jobs) - done)
        print(
            f"[shard {a.shard}] {done}/{len(jobs)} {sub}/{stem}  "
            f"best={best_iou:.3f} mean={np.mean(ious):.3f}±{np.std(ious):.3f}  "
            f"{time.time() - t0:.0f}s  eta {eta / 3600:.1f}h",
            flush=True,
        )

    renderer.close()
    print(f"[shard {a.shard}] DONE {done} targets in {(time.time() - t_start) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
