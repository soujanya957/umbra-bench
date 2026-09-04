#!/usr/bin/env python3
"""Run the UMBRA optimizer over the sequence axis: chained frames, N solves each.

Same optimizer, same budgets, same output conventions as
`run_base_optimizer.py` — one thing changes, and it is the whole point of the
sequence axis:

    frame k is solved from frame k-1's *best* pose.

Not from frame k-1's run-0 pose. Each frame gets `--runs` independent solves,
all of them seeded from the previous frame's winner and anchored to it by the
temporal + reachability terms; the winner of *those* becomes the prior for
frame k+1, and so on down the sequence. Chaining run i to run i is the obvious
mistake here and it silently measures something else: it would propagate five
independent mediocre trajectories instead of one good one.

    frame 00:  [run0 run1 run2 run3 run4]  -> best -> q_prev
    frame 01:  [run0 run1 run2 run3 run4]  <- all five start from q_prev
                       |
                       +-> best -> q_prev
    frame 02:  ...

**Which run wins.** By default the winner is chosen planner-first: among the
runs, prefer those within the direct-quintic threshold of the prior
(`LARGE_Q_JUMP`, 1.2 rad/joint), and only then by IoU. A run can win on raw IoU
while jumping further than the planner can connect, and chaining *that* pose
poisons every frame after it — a sequence has no way to recover from a prior it
cannot reach. `--chain-by iou` restores naive best-IoU selection for ablation;
both scores are recorded either way, so the choice is auditable after the fact.

Layout mirrors `sequences/`, one folder per sequence:

    optimized/anim-optimizer/<budget>/<group>/<name>/
        frame_00_run00.png … frame_NN_run04.png   every solve (gitignored)
        frame_00_best.png  … frame_NN_best.png    the chained trajectory
        results.json                              per-frame runs, chain, dq

Sharding is by sequence, never by frame: frames are serially dependent, so a
sequence is the smallest unit that can be run independently.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)

# motion_planner.LARGE_Q_JUMP — the per-joint displacement above which the
# planner abandons a direct quintic and falls back to STOMP/sequential/via.
LARGE_Q_JUMP = 1.2


def load_target(path: str, size: int) -> np.ndarray:
    """Sequence frames use the same 1-bit dark=shape convention as targets/."""
    img = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    t = (np.array(img) < 128).astype(np.float32)
    corners = t[0, 0] + t[0, -1] + t[-1, 0] + t[-1, -1]
    if corners >= 3.0 and t.mean() > 0.5:
        raise ValueError(f"{path} looks stored inverted (fill {100 * t.mean():.0f}%)")
    return t


def save_mask(mask: np.ndarray, path: str):
    m = (np.asarray(mask) > 0.5).astype(np.uint8)
    Image.fromarray(((1 - m) * 255).astype(np.uint8), "L").convert("1").save(path)


def write_budget_md(path, a, cfg, rig, seqs, repo, chain_by):
    n_arm, n_rob = 6, rig["n_robots"]
    ph_pop = max(32, cfg["popsize"])
    hung = n_rob * n_rob * 8
    init = 16 * n_rob
    ph1 = a.phase1_iters * n_rob * ph_pop
    ph2 = a.phase2_iters * n_rob * ph_pop if n_rob > 1 else 0
    fin_iters = a.final_iters if a.adaptive_final is False else max(12, round(a.final_iters * 0.55))
    fin = fin_iters * cfg["popsize"]
    fd = 180
    total = hung + init + ph1 + ph2 + fin + fd
    n_frames = sum(s["n_frames"] for s in seqs)
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"

    with open(path, "w") as f:
        f.write(f"""# Budget — `{os.path.basename(os.path.dirname(path))}` (sequence axis)

Generated {time.strftime("%Y-%m-%d %H:%M:%S %Z")} on `{os.uname().nodename}`,
fleet-shadow-art @ `{sha}`.

## Optimizer

| setting | value |
|---|---|
| popsize | {cfg["popsize"]} |
| phase1_iters (per robot, {n_arm}-D) | {a.phase1_iters} |
| phase2_iters (per robot, {n_arm}-D) | {a.phase2_iters} |
| final_iters (joint, {n_rob * n_arm}-D) | {a.final_iters} |
| adaptive_final | {a.adaptive_final} |
| floor / collision / self-collision penalty | {a.floor_penalty} / {a.collision_penalty} / {a.self_collision_penalty} |
| n_workers per process | {a.n_workers} |

Identical to the static sweep of the same name, so a sequence frame and a
`targets/` sample cost the same search. The greedy phases are **not** skipped on
warm frames: the sequence solver in `run_sequence.py` does skip them once a
warm start looks good, which is right for production but would make these
numbers incomparable to the static ones.

## Chaining

| setting | value |
|---|---|
| solves per frame | {a.runs} |
| prior for frame k | best run of frame k-1 (`x0` **and** `q_ref`) |
| winner selected by | `{chain_by}` |
| temporal_weight (Ph1/Ph2) | {a.temporal_weight} |
| final_temporal_weight | {a.final_temporal_weight} (0 = curriculum: final pass free) |
| reachability_penalty | {a.reachability_penalty} (barrier at {LARGE_Q_JUMP} rad/joint) |

Frame 00 has no prior and is solved cold — identical to a static target.

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

## Rig

| setting | value |
|---|---|
| robots | {n_rob} × SO-101 |
| arm gap | {rig["arm_gap_m"]} m |
| light-to-front / back-to-wall | {rig["light_to_front_m"]} / {rig["back_to_wall_m"]} (None = default) |
| render size | {rig["render_size"]} px |
| target fit (similarity transform) | {rig["target_fit"]} |
| target deformation (free-form warp) | {rig["distortion"]} |

## Scale

{len(seqs)} sequences / {n_frames} frames × {a.runs} runs ≈ **{n_frames * a.runs * total / 1e6:.2f}M renders**
""")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=os.path.expanduser("~/dev/fleet-shadow-art"))
    p.add_argument("--bench", default=_BENCH)
    p.add_argument("--groups", nargs="+", default=None,
                   help="Restrict to these sequence groups (default: all)")
    p.add_argument("--only", nargs="+", default=None,
                   help="Restrict to these sequence_ids, e.g. gesture/wiper")
    p.add_argument("--runs", type=int, default=5)
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
    p.add_argument("--no-adaptive-final", dest="adaptive_final", action="store_false")
    p.add_argument("--n-workers", type=int, default=0)
    p.add_argument("--temporal-weight", type=float, default=0.3)
    p.add_argument("--final-temporal-weight", type=float, default=0.0)
    p.add_argument("--reachability-penalty", type=float, default=100.0)
    p.add_argument("--chain-by", choices=["reachable", "iou"], default="reachable",
                   help="'reachable' prefers runs within LARGE_Q_JUMP of the prior "
                        "before comparing IoU; 'iou' ignores reachability (ablation)")
    p.add_argument("--fit-target", action="store_true",
                   help="Fit ONE similarity transform for the whole sequence and "
                        "apply it to every frame. Per-frame fitting would re-centre "
                        "each frame independently and delete the motion")
    p.add_argument("--fit-scale-min", type=float, default=0.35)
    p.add_argument("--fit-scale-max", type=float, default=1.60)
    p.add_argument("--fit-max-shift", type=float, default=0.22)
    p.add_argument("--fit-n-scales", type=int, default=14)
    p.add_argument("--fit-n-shifts", type=int, default=15)
    p.add_argument("--reach-samples", type=int, default=300)
    p.add_argument("--distort", action="store_true",
                   help="Let the figure DEFORM (bounded free-form warp) beyond what "
                        "a similarity transform can fix. ONE deformation per clip, "
                        "never per-frame — a per-frame warp makes the character "
                        "breathe and wobble. Writes to a SEPARATE --out")
    p.add_argument("--distort-max", type=float, default=0.06,
                   help="Cap on how far any pixel may travel, as a fraction of frame")
    p.add_argument("--distort-grid", type=int, default=4)
    p.add_argument("--distort-bending", type=float, default=0.03,
                   help="Thin-plate penalty. Do NOT set to 0 — an unregularised warp "
                        "crumples and fits worse (see target_warp.fit_warp)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    if (a.fit_target or a.distort) and not a.out:
        p.error("--fit-target/--distort change the condition; pass an explicit --out "
                "so they do not land on top of the baseline results")

    a.bench = os.path.abspath(a.bench)
    if a.out:
        a.out = os.path.abspath(a.out)

    ms = os.path.join(a.repo, "motion-aware-shadow")
    sys.path.insert(0, ms)
    os.chdir(ms)  # URDF meshes resolve relative to the package root

    from loss import compute_iou
    from optimizer import OptimizerConfig, optimize_staged
    from renderer import ShadowRenderer, build_scene

    outroot = a.out or os.path.join(a.bench, "optimized", "anim-optimizer")

    seqs = [json.loads(l) for l in open(os.path.join(a.bench, "sequences.jsonl"))]
    if a.groups:
        seqs = [s for s in seqs if s["group"] in a.groups]
    if a.only:
        seqs = [s for s in seqs if s["sequence_id"] in set(a.only)]
    all_seqs = list(seqs)
    seqs = [s for i, s in enumerate(seqs) if i % a.num_shards == a.shard]
    print(f"[shard {a.shard}/{a.num_shards}] {len(seqs)} sequences "
          f"({sum(s['n_frames'] for s in seqs)} frames) × {a.runs} runs", flush=True)

    urdf = os.path.join(ms, "urdf/SO101/so101_new_calib.urdf")
    model, data, n_robots = build_scene(
        urdf, n_robots=a.n_robots, arm_gap=a.arm_gap,
        light_to_front=a.light_to_front, back_to_wall=a.back_to_wall)
    renderer = ShadowRenderer(model, data, size=a.size, n_robots=n_robots)
    print(f"[shard {a.shard}] n_dof={renderer.n_dof} arm_gap={a.arm_gap}", flush=True)

    reach = None
    if a.fit_target:
        from reachability import build_reachability_map
        reach = build_reachability_map(renderer, n_samples=a.reach_samples)
        print(f"[shard {a.shard}] reach map ready", flush=True)

    cfg_common = dict(
        popsize=a.popsize, phase1_iters=a.phase1_iters, phase2_iters=a.phase2_iters,
        final_iters=a.final_iters, floor_penalty=a.floor_penalty,
        collision_penalty=a.collision_penalty,
        self_collision_penalty=a.self_collision_penalty,
        n_workers=a.n_workers, adaptive_final=a.adaptive_final,
    )
    rig = {
        "n_robots": n_robots, "arm_gap_m": a.arm_gap,
        "light_to_front_m": a.light_to_front, "back_to_wall_m": a.back_to_wall,
        "render_size": a.size, "distortion": False,
        "target_fit": ({"scale_range": [a.fit_scale_min, a.fit_scale_max],
                        "max_shift_frac": a.fit_max_shift,
                        "sequence_level": True} if a.fit_target else False),
    }
    rig["distortion"] = ({"grid": a.distort_grid,
                          "max_disp_frac": a.distort_max,
                          "bending": a.distort_bending,
                          "sequence_level": True} if a.distort else False)

    if a.shard == 0:
        os.makedirs(outroot, exist_ok=True)
        write_budget_md(os.path.join(outroot, "BUDGET.md"), a, cfg_common, rig,
                        all_seqs, a.repo, a.chain_by)
        print(f"[shard 0] wrote {outroot}/BUDGET.md", flush=True)

    t_start, done = time.time(), 0
    for seq in seqs:
        group, name = seq["group"], seq["name"]
        odir = os.path.join(outroot, group, name)
        rjson = os.path.join(odir, "results.json")
        if os.path.exists(rjson) and not a.force:
            done += 1
            continue
        os.makedirs(odir, exist_ok=True)

        T0s = [load_target(os.path.join(a.bench, f), a.size) for f in seq["frames"]]

        # One transform for the whole sequence. Fitting each frame on its own
        # would give every frame its own scale and offset, which is exactly the
        # motion the sequence is supposed to contain — the fit would eat it.
        Ts, fit_info = T0s, None
        if a.fit_target:
            from target_fit import apply_fit, fit_target_sequence
            fit = fit_target_sequence(
                T0s, reach,
                scale_range=(a.fit_scale_min, a.fit_scale_max),
                n_scales=a.fit_n_scales,
                max_shift_frac=a.fit_max_shift,
                n_shifts=a.fit_n_shifts, verbose=False)
            fit_info = dict(fit.as_dict())
            fit_info["fitted_on"] = "union of all frames"
            Ts = [apply_fit(T, fit) for T in T0s]

        # One deformation for the whole clip, fit from a single solve of the union
        # frame. run_sequence.py makes the same call for the same reason: a warp fit
        # per frame re-bends the character to each frame's own shadow and the figure
        # visibly breathes. Fitting from the union solve rather than from a full
        # first pass costs one extra solve instead of doubling the clip.
        warp, warp_info = None, None
        if a.distort:
            from target_warp import apply_warp, fit_warp_sequence
            u_target = np.clip(np.sum([t > 0.5 for t in Ts], axis=0), 0, 1).astype(np.float32)
            u_res = optimize_staged(
                renderer, u_target,
                OptimizerConfig(seed=0, **cfg_common))
            u_shadow = renderer.get_shadow_mask(u_res.best_q)
            warp = fit_warp_sequence(
                [u_shadow], [u_target],
                grid=a.distort_grid, max_disp_frac=a.distort_max,
                bending=a.distort_bending, seed=0, verbose=False)
            if warp.is_identity():
                print(f"[shard {a.shard}] {group}/{name} deformation bought nothing "
                      f"— frames left as authored", flush=True)
                warp = None
            else:
                warp_info = warp.as_dict()
                print(f"[shard {a.shard}] {group}/{name} clip warp "
                      f"rms {warp.magnitude():.2f}px peak {warp.peak():.2f}px "
                      f"union IoU {warp.iou_original:.3f} → {warp.iou_warped:.3f}",
                      flush=True)
            # T0s stays undeformed on purpose. `iou_vs_original` is the only number
            # comparable across the distort / no-distort conditions: IoU against a
            # target the run itself bent measures the bend, not the shadow.
            if warp is not None:
                Ts = [apply_warp(T, warp) for T in Ts]

        frames_out, prev_q, t0 = [], None, time.time()
        for k, (T, T0) in enumerate(zip(Ts, T0s)):
            runs = []
            for s in range(a.runs):
                cfg = OptimizerConfig(
                    seed=k * 1000 + s,
                    temporal_weight=a.temporal_weight if prev_q is not None else 0.0,
                    final_temporal_weight=a.final_temporal_weight,
                    reachability_penalty=a.reachability_penalty,
                    **cfg_common)
                res = optimize_staged(renderer, T, cfg, x0=prev_q, q_ref=prev_q)
                shadow = renderer.get_shadow_mask(res.best_q)
                iou = float(compute_iou(shadow, T))
                dq = (float(np.max(np.abs(res.best_q - prev_q)))
                      if prev_q is not None else 0.0)
                save_mask(shadow, os.path.join(odir, f"frame_{k:02d}_run{s:02d}.png"))
                runs.append({
                    "run": s, "seed": cfg.seed,
                    "iou": round(iou, 4),
                    "iou_vs_original": round(float(compute_iou(shadow, T0)), 4),
                    "loss": round(float(res.best_loss), 4),
                    "n_evals": int(res.n_evals),
                    "dq_max_from_prior": round(dq, 4),
                    "cheap": bool(dq < LARGE_Q_JUMP),
                    "q_rad": [round(float(v), 6) for v in res.best_q],
                })

            # Winner. Planner-first by default: a run that lands outside the
            # direct-connect radius is not just a worse frame, it is a worse
            # *prior*, and every later frame inherits it.
            if a.chain_by == "reachable":
                win = max(range(len(runs)), key=lambda i: (runs[i]["cheap"], runs[i]["iou"]))
            else:
                win = max(range(len(runs)), key=lambda i: runs[i]["iou"])
            best = runs[win]
            prev_q = np.array(best["q_rad"], dtype=float)

            import shutil
            shutil.copyfile(os.path.join(odir, f"frame_{k:02d}_run{win:02d}.png"),
                            os.path.join(odir, f"frame_{k:02d}_best.png"))
            frames_out.append({
                "frame": k,
                "target": seq["frames"][k],
                "best_run": win,
                "best_iou": best["iou"],
                "best_iou_vs_original": best["iou_vs_original"],
                "dq_max_from_prior": best["dq_max_from_prior"],
                "cheap": best["cheap"],
                # What best-IoU selection would have taken, so the cost of the
                # planner-first rule is visible without re-running the sweep.
                "argmax_iou_run": int(max(range(len(runs)), key=lambda i: runs[i]["iou"])),
                "runs": runs,
            })
            print(f"[shard {a.shard}] {group}/{name} f{k:02d} "
                  f"best={best['iou']:.3f} (run {win}) dq={best['dq_max_from_prior']:.2f} "
                  f"{'cheap' if best['cheap'] else 'FALLBACK'}", flush=True)

        # Wrap-around: only meaningful when the sequence actually closes.
        loop_dq = None
        if seq["cyclic"] and len(frames_out) > 1:
            q_first = np.array(frames_out[0]["runs"][frames_out[0]["best_run"]]["q_rad"])
            loop_dq = round(float(np.max(np.abs(prev_q - q_first))), 4)

        ious = [f["best_iou"] for f in frames_out]
        interior = [f for f in frames_out[1:]]
        with open(rjson, "w") as f:
            json.dump({
                "sequence_id": seq["sequence_id"],
                "group": group, "name": name,
                "n_frames": len(frames_out),
                "cyclic": seq["cyclic"],
                "method": ("anim-optimizer" + ("+fit" if a.fit_target else "")
                           + ("+distort" if a.distort else "")),
                "warp": warp_info,
                "chain_by": a.chain_by,
                "rig": rig, "fit": fit_info, "optimizer": cfg_common,
                "temporal": {
                    "temporal_weight": a.temporal_weight,
                    "final_temporal_weight": a.final_temporal_weight,
                    "reachability_penalty": a.reachability_penalty,
                    "large_q_jump": LARGE_Q_JUMP,
                },
                "runs_per_frame": a.runs,
                "mean_iou": round(float(np.mean(ious)), 4),
                "min_iou": round(float(np.min(ious)), 4),
                "first_frame_iou": ious[0],
                "mean_iou_vs_original": round(
                    float(np.mean([f["best_iou_vs_original"] for f in frames_out])), 4),
                "mean_dq": round(float(np.mean(
                    [f["dq_max_from_prior"] for f in interior])), 4) if interior else None,
                "max_dq": round(float(np.max(
                    [f["dq_max_from_prior"] for f in interior])), 4) if interior else None,
                "cheap_frac": round(float(np.mean(
                    [f["cheap"] for f in interior])), 4) if interior else None,
                "loop_dq": loop_dq,
                "loop_cheap": (loop_dq < LARGE_Q_JUMP) if loop_dq is not None else None,
                "seconds": round(time.time() - t0, 1),
                "frames": frames_out,
            }, f, indent=2)

        done += 1
        el = time.time() - t_start
        eta = el / max(done, 1) * (len(seqs) - done)
        print(f"[shard {a.shard}] {done}/{len(seqs)} {group}/{name} "
              f"mean={np.mean(ious):.3f} cheap={100 * np.mean([f['cheap'] for f in interior]):.0f}% "
              f"{time.time() - t0:.0f}s  eta {eta / 3600:.1f}h", flush=True)

    renderer.close()
    print(f"[shard {a.shard}] DONE {done} sequences in "
          f"{(time.time() - t_start) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
