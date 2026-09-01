#!/usr/bin/env python3
"""Run the base UMBRA optimizer over the benchmark: N independent solves per target.

"Base" means the optimizer as it stands — staged CMA-ES, target dropped in at canvas
scale, dead centre. `--fit-target` adds the one intervention that survived the
distortion study: a similarity transform (uniform scale + translation) placing the
target where the rig can actually cast it. Proportions never change, so recognizability
is preserved by construction. It is a separate condition and must go to its own --out;
IoU is then recorded against both the shown and the authored target. Each
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


def _default_repo() -> str:
    """Where fleet-shadow-art lives.

    The historical default, ~/dev/fleet-shadow-art, exists on none of the machines
    this has run on since, so every invocation either passed --repo or failed deep
    inside an import with nothing pointing at the cause. Checked in order: an
    explicit env var, a checkout sitting beside this one (the usual layout, both
    repos cloned into the same parent), then the historical default so a machine
    where it did work is unaffected.
    """
    for cand in (
        os.environ.get("FLEET_SHADOW_ART"),
        os.path.join(os.path.dirname(_BENCH), "fleet-shadow-art"),
        os.path.expanduser("~/dev/fleet-shadow-art"),
    ):
        if cand and os.path.isdir(os.path.join(cand, "motion-aware-shadow")):
            return cand
    return os.path.expanduser("~/dev/fleet-shadow-art")


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


def write_budget_md(
    path: str, a, cfg: dict, rig: dict, n_targets: int, repo: str, note: str = ""
):
    """Record what this sweep actually cost, next to its results.

    A results folder whose settings live only in a shell history is not comparable
    against anything later. The render count is derived from the same numbers the
    optimizer uses, so the estimate moves whenever the config does.
    """
    n_arm = 6
    n_rob = rig["n_robots"]
    ph_pop = max(32, cfg["popsize"])
    hung = n_rob * n_rob * 8
    init = 16 * n_rob
    ph1 = a.phase1_iters * n_rob * ph_pop
    ph2 = a.phase2_iters * n_rob * ph_pop if n_rob > 1 else 0
    fin_iters = a.final_iters if a.adaptive_final is False else max(12, round(a.final_iters * 0.55))
    fin = fin_iters * cfg["popsize"]
    fd = 180
    total = hung + init + ph1 + ph2 + fin + fd
    # What the optimizer will actually use, not what was asked for. On win32 the
    # thread count is clamped to 1 whatever the flag says, and a BUDGET.md that
    # records the request instead of the result is how "n_workers = 2", true only
    # on the EGL machine that produced it, ends up reading like a setting any
    # machine can reproduce.
    eff_workers = (
        1 if sys.platform == "win32"
        else (a.n_workers if a.n_workers >= 1 else "auto")
    )

    import subprocess

    try:
        sha = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        sha = "unknown"

    with open(path, "w") as f:
        f.write(
            f"""# Budget — `{os.path.basename(os.path.dirname(path))}`

Generated {time.strftime("%Y-%m-%d %H:%M:%S %Z")} on `{os.uname().nodename}`,
fleet-shadow-art @ `{sha}`.
{note}
## Optimizer

| setting | value |
|---|---|
| popsize | {cfg["popsize"]} |
| phase1_iters (per robot, {n_arm}-D) | {a.phase1_iters} |
| phase2_iters (per robot, {n_arm}-D) | {a.phase2_iters} |
| final_iters (joint, {n_rob * n_arm}-D) | {a.final_iters} |
| adaptive_final | {a.adaptive_final} |
| floor / collision / self-collision penalty | {a.floor_penalty} / {a.collision_penalty} / {a.self_collision_penalty} |
| n_workers per process | {a.n_workers} requested / {eff_workers} effective |

## Renders per solve (derived)

| stage | renders |
|---|---|
| Hungarian pre-assignment | {hung:,} |
| init sampling | {init:,} |
| phase 1 — forward greedy | {ph1:,} |
| phase 2 — backward pass | {ph2:,} |
| final — joint refinement ({fin_iters} iters) | {fin:,} |
| FD refinement | ~{fd:,} |
| **total** | **~{total:,}** |

## Sampling

- {a.runs} independent solves per target, seeds `0…{a.runs - 1}`.
- Targets finishing below **IoU {a.extra_below}** get **{a.extra_runs} extra** solves
  (seeds `{a.runs}…{a.runs + a.extra_runs - 1}`), all of them, not stopping at the first
  to clear the bar. `results.json` marks these with `"extra": true`.
- Reported statistic is best-of-N. Within-target seed spread on this rig is
  σ ≈ 0.022 IoU, so a single solve is a sample, not a measurement.

## Rig

| setting | value |
|---|---|
| robots | {n_rob} × SO-101 |
| arm gap | {rig["arm_gap_m"]} m |
| light-to-front / back-to-wall | {rig["light_to_front_m"]} / {rig["back_to_wall_m"]} (None = default) |
| render size | {rig["render_size"]} px |
| target deformation (free-form warp) | {rig["distortion"]} |
| target fit (similarity transform) | {rig["target_fit"]} |

## Scale

{n_targets} targets × {a.runs} runs ≈ **{n_targets * a.runs * total / 1e6:.1f}M renders**
before extras.

Subsets: {", ".join(a.subsets)}
"""
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=_default_repo())
    p.add_argument("--bench", default=_BENCH)
    p.add_argument(
        "--targets-dir",
        default="targets",
        help="target tree to solve, relative to --bench. `targets_grounded` is the "
        "same shapes translated to rest on the bottom of the frame (see "
        "scripts/ground_targets.py). A different tree is a different experimental "
        "condition, so give it its own --out",
    )
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
    p.add_argument(
        "--no-adaptive-final",
        dest="adaptive_final",
        action="store_false",
        help="Spend the full --final-iters on the joint pass. By default the optimizer "
        "shrinks it once a frame looks solved, which is right for sequences but caps "
        "the one stage that searches all robots together — the thin part of a big budget",
    )
    p.add_argument(
        "--extra-runs",
        type=int,
        default=0,
        help="Extra solves granted to targets that finish below --extra-below. Restarts "
        "are the only lever a fixed budget has on a target the search keeps missing",
    )
    p.add_argument("--extra-below", type=float, default=0.50)
    p.add_argument(
        "--fit-target",
        action="store_true",
        help="Place each target with a similarity transform (uniform scale + shift) "
        "into the rig's reachable support before solving. Proportions never change, "
        "so a smaller, higher 'B' is still a 'B' — unlike the free-form warp in "
        "target_warp.py there is no fidelity traded away. Writes to a SEPARATE --out",
    )
    p.add_argument("--fit-scale-min", type=float, default=0.35)
    p.add_argument("--fit-scale-max", type=float, default=1.60)
    p.add_argument("--fit-max-shift", type=float, default=0.22)
    p.add_argument("--fit-scale-penalty", type=float, default=0.0)
    p.add_argument("--fit-n-scales", type=int, default=14)
    p.add_argument("--fit-n-shifts", type=int, default=15)
    p.add_argument(
        "--fit-dy-min",
        type=float,
        default=None,
        help="Vertical search range as a fraction of --size, + = down. Pass with "
        "--fit-dy-max to replace the symmetric +/- --fit-max-shift on the vertical "
        "axis alone. Symmetric is the wrong shape for a rig whose reachable band "
        "sits below the canvas centre: in big-budget-fitted 478/478 fits chose "
        "dy > 0 and 60%% stopped exactly on the bound, so half the grid went to "
        "shifts the search never takes while the half it wanted ran out of room",
    )
    p.add_argument("--fit-dy-max", type=float, default=None)
    p.add_argument(
        "--fit-min-retained",
        type=float,
        default=0.98,
        help="Reject a placement losing more than 1 - this fraction of the scaled "
        "target off the canvas. Default unchanged, so existing sweeps stay "
        "reproducible; what is new is that the chosen placement's actual loss is "
        "recorded as fit.clip_frac whether or not it cleared the threshold",
    )
    p.add_argument(
        "--reach-samples",
        type=int,
        default=300,
        help="Forward-kinematics samples used to build the reachability map. A property "
        "of the rig, not of any target, so it is built once per process",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    # The fit runs are a different experimental condition, not more rows of the same
    # one. Sharing an output folder would silently mix them and make the resume check
    # skip fit targets because the no-fit results.json already exists.
    if a.fit_target and not a.out:
        p.error("--fit-target changes the condition; pass an explicit --out so it "
                "does not land on top of the no-fit results")
    if (a.fit_dy_min is None) != (a.fit_dy_max is None):
        p.error("--fit-dy-min and --fit-dy-max define one range; pass both or neither")
    if a.fit_dy_min is not None and a.fit_dy_min >= a.fit_dy_max:
        p.error("--fit-dy-min must be below --fit-dy-max")

    # Resolve paths against the launch directory before chdir'ing away from it, or a
    # relative --out silently lands under the package root instead of the benchmark.
    a.bench = os.path.abspath(a.bench)
    if a.out:
        a.out = os.path.abspath(a.out)

    ms = os.path.join(a.repo, "motion-aware-shadow")
    if not os.path.isdir(ms):
        p.error(
            f"--repo {a.repo!r} contains no motion-aware-shadow/. Pass --repo, or "
            f"set FLEET_SHADOW_ART, or clone fleet-shadow-art beside this repo."
        )
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
        d = os.path.join(a.bench, a.targets_dir, sub)
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

    reach = support = None
    if a.fit_target:
        from reachability import build_reachability_map, uncastable_fraction
        from target_fit import fit_target

        reach = build_reachability_map(renderer, n_samples=a.reach_samples)
        support = reach.support()
        print(f"[shard {a.shard}] reach map ready ({a.reach_samples} samples)", flush=True)

    cfg_common = dict(
        popsize=a.popsize,
        phase1_iters=a.phase1_iters,
        phase2_iters=a.phase2_iters,
        final_iters=a.final_iters,
        floor_penalty=a.floor_penalty,
        collision_penalty=a.collision_penalty,
        self_collision_penalty=a.self_collision_penalty,
        n_workers=a.n_workers,
        adaptive_final=a.adaptive_final,
    )
    rig = {
        "n_robots": n_robots,
        "arm_gap_m": a.arm_gap,
        "light_to_front_m": a.light_to_front,
        "back_to_wall_m": a.back_to_wall,
        "render_size": a.size,
        "targets_dir": a.targets_dir,
        "distortion": False,
        "target_fit": (
            {
                "scale_range": [a.fit_scale_min, a.fit_scale_max],
                "max_shift_frac": a.fit_max_shift,
                "dy_range": (
                    [a.fit_dy_min, a.fit_dy_max]
                    if a.fit_dy_min is not None
                    else "symmetric"
                ),
                "min_retained": a.fit_min_retained,
                "scale_penalty": a.fit_scale_penalty,
                "n_scales": a.fit_n_scales,
                "n_shifts": a.fit_n_shifts,
                "reach_samples": a.reach_samples,
            }
            if a.fit_target
            else False
        ),
    }

    # One shard writes it, so parallel shards don't race on the same file.
    if a.shard == 0:
        os.makedirs(outroot, exist_ok=True)
        n_all = sum(
            len([f for f in os.listdir(os.path.join(a.bench, "targets", s)) if f.lower().endswith(".png")])
            for s in a.subsets
            if os.path.isdir(os.path.join(a.bench, "targets", s))
        )
        write_budget_md(
            os.path.join(outroot, "BUDGET.md"), a, cfg_common, rig, n_all, a.repo
        )
        print(f"[shard 0] wrote {os.path.join(outroot, 'BUDGET.md')}", flush=True)

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
            T0 = load_target(tpath, a.size)
        except Exception as e:
            print(f"[shard {a.shard}] SKIP {sub}/{stem}: {e}", flush=True)
            continue

        # T is what the solver is shown; T0 is the target as authored. With --fit-target
        # they differ by a similarity transform, so IoU is recorded against both: vs
        # shown is what the rig was asked to reproduce, vs original keeps the placement
        # visible instead of hiding it inside the headline number.
        T, fit_info = T0, None
        if a.fit_target:
            fit = fit_target(
                T0, reach,
                scale_range=(a.fit_scale_min, a.fit_scale_max),
                n_scales=a.fit_n_scales,
                max_shift_frac=a.fit_max_shift,
                dy_range=(
                    (a.fit_dy_min * a.size, a.fit_dy_max * a.size)
                    if a.fit_dy_min is not None
                    else None
                ),
                n_shifts=a.fit_n_shifts,
                scale_penalty=a.fit_scale_penalty,
                min_retained=a.fit_min_retained,
                verbose=False,
            )
            T = fit.target
            fit_info = dict(fit.as_dict())
            fit_info["uncastable_before"] = round(
                float(uncastable_fraction(T0, support)), 5
            )
            fit_info["uncastable_after"] = round(
                float(uncastable_fraction(T, support)), 5
            )
            # All three masks live together so a folder is self-contained: what was
            # authored, what the solver was actually shown, and what the rig cast.
            # Comparing a shadow against the wrong one of the first two is the easiest
            # way to misread this experiment.
            save_mask(T0, os.path.join(odir, f"{stem}_original.png"))
            save_mask(T, os.path.join(odir, f"{stem}_shown.png"))

        runs, best_i, best_iou = [], -1, -1.0
        t0 = time.time()

        def do_run(k: int, extra: bool = False):
            nonlocal best_i, best_iou
            res = optimize_staged(renderer, T, OptimizerConfig(seed=k, **cfg_common))
            shadow = renderer.get_shadow_mask(res.best_q)
            iou = float(compute_iou(shadow, T))
            save_mask(shadow, os.path.join(odir, f"{stem}_run{k:02d}.png"))
            runs.append(
                {
                    "run": k,
                    "seed": k,
                    "iou": round(iou, 4),
                    "iou_vs_original": round(float(compute_iou(shadow, T0)), 4),
                    "loss": round(float(res.best_loss), 4),
                    "n_evals": int(res.n_evals),
                    "extra": extra,
                    "q_rad": [round(float(v), 6) for v in res.best_q],
                }
            )
            if iou > best_iou:
                best_iou, best_i = iou, k
                save_mask(shadow, os.path.join(odir, f"{stem}_best.png"))

        for k in range(a.runs):
            do_run(k)

        # Targets the search keeps missing get more restarts. All granted extras are
        # run, rather than stopping at the first one to clear the bar: stopping early
        # would pile the reported scores up just above the threshold and make the
        # low tail look like a cliff that is really an artefact of when we quit.
        n_extra = 0
        if a.extra_runs > 0 and best_iou < a.extra_below:
            print(
                f"[shard {a.shard}] {sub}/{stem} best={best_iou:.3f} "
                f"< {a.extra_below} → {a.extra_runs} extra runs",
                flush=True,
            )
            for k in range(a.runs, a.runs + a.extra_runs):
                do_run(k, extra=True)
                n_extra += 1

        ious = [r["iou"] for r in runs]
        with open(rjson, "w") as f:
            json.dump(
                {
                    "id": f"{sub}_{stem}",
                    "subset": sub,
                    "target": os.path.relpath(tpath, a.bench),
                    "method": "base-optimizer" + ("+fit" if a.fit_target else ""),
                    "rig": rig,
                    "fit": fit_info,
                    "optimizer": cfg_common,
                    "n_runs": len(runs),
                    "n_base_runs": a.runs,
                    "n_extra_runs": n_extra,
                    "extra_below": a.extra_below if a.extra_runs else None,
                    "best_run": best_i,
                    "best_iou": round(best_iou, 4),
                    "best_iou_vs_original": round(
                        float(max(r["iou_vs_original"] for r in runs)), 4
                    ),
                    "mean_iou": round(float(np.mean(ious)), 4),
                    "mean_iou_vs_original": round(
                        float(np.mean([r["iou_vs_original"] for r in runs])), 4
                    ),
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
            + (
                f"fit(s={fit_info.get('scale')}, unc "
                f"{100 * fit_info['uncastable_before']:.0f}%→"
                f"{100 * fit_info['uncastable_after']:.0f}%)  "
                if fit_info
                else ""
            )
            + f"{time.time() - t0:.0f}s  eta {eta / 3600:.1f}h",
            flush=True,
        )

    renderer.close()
    print(f"[shard {a.shard}] DONE {done} targets in {(time.time() - t_start) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
