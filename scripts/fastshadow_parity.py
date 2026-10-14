"""Does the nvdiffrast renderer cast the same shadow as MuJoCo, and how much faster?

Poses: random joint vectors inside the limits, plus the solved poses stored in an
existing results tree (default big-budget-grounded), which is where an optimizer
actually spends its time. For each pose both renderers produce a mask; reported is
the per-pose IoU between them and the pixel disagreement, per renderer setting.

    python scripts/fastshadow_parity.py --n-random 300 --n-solved 300
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastshadow  # noqa: E402


def iou(a, b):
    i = (a * b).sum(-1).sum(-1)
    u = a.sum(-1).sum(-1) + b.sum(-1).sum(-1) - i
    return np.where(u > 0, i / np.maximum(u, 1e-9), 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.path.join(fastshadow.BENCH, "optimized", "big-budget-grounded"))
    p.add_argument("--n-random", type=int, default=300)
    p.add_argument("--n-solved", type=int, default=300)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--arm-gap", type=float, default=0.20)
    p.add_argument("--settings", nargs="+", default=["1.0:1:0.5", "1.0:2:0.25", "1.0:4:0.25", "0:4:0.25", "2.0:4:0.25"],
                   help="cluster_mm:supersample:coverage_threshold")
    p.add_argument("--out", default=os.path.join(fastshadow.BENCH, "results", "fastshadow", "parity.json"))
    a = p.parse_args()

    results = os.path.abspath(a.results)
    out = os.path.abspath(a.out)
    fastshadow.use_fleet()
    import torch
    from renderer import ShadowRenderer, build_scene
    from fastshadow.renderer import FastShadowRenderer

    model, data, n = build_scene("urdf/SO101/so101_new_calib.urdf", n_robots=3, arm_gap=a.arm_gap)
    mj = ShadowRenderer(model, data, size=a.size, n_robots=n)

    rng = np.random.default_rng(0)
    Qr = rng.uniform(mj.q_lower, mj.q_upper, size=(a.n_random, mj.n_dof))
    solved = []
    for f in sorted(glob.glob(os.path.join(results, "*", "*", "results.json"))):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("rig", {}).get("n_robots") == n:
            solved += [run["q_rad"] for run in r["runs"]]
    solved = np.array(solved)
    Qs = solved[rng.choice(len(solved), size=min(a.n_solved, len(solved)), replace=False)]
    Q = np.concatenate([Qr, Qs])
    kind = np.array(["random"] * len(Qr) + ["solved"] * len(Qs))
    print(f"{len(Qr)} random + {len(Qs)} solved poses (of {len(solved)} in {results})")

    t = time.perf_counter()
    M_mj = np.stack([mj.get_shadow_mask(q) for q in Q])
    t_mj = (time.perf_counter() - t) / len(Q)
    print(f"MuJoCo: {1e3 * t_mj:.3f} ms/pose (one process, one GL context)")

    report = {"n_random": len(Qr), "n_solved": len(Qs), "size": a.size,
              "mujoco_ms_per_pose": round(1e3 * t_mj, 4), "settings": []}
    for spec in a.settings:
        cm, ss, thr = spec.split(":")
        fr = FastShadowRenderer(mj, cluster_mm=float(cm), supersample=int(ss), coverage_threshold=float(thr))
        Qt = torch.tensor(Q, dtype=torch.float32, device="cuda")
        fr.render(Qt[:64])  # warm-up
        torch.cuda.synchronize()
        M_fr = fr.render(Qt).cpu().numpy()
        times = {}
        for B in (1, 48, 192, 1024):
            Qb = Qt[:B] if B <= len(Qt) else Qt.repeat((B + len(Qt) - 1) // len(Qt), 1)[:B]
            reps = max(3, 2000 // B)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(reps):
                fr.render(Qb)
            torch.cuda.synchronize()
            times[B] = (time.perf_counter() - t) / (reps * B)
        # soft-render + backward, the cost of one gradient step per pose
        Qg = Qt[:48].clone().requires_grad_(True)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(20):
            fr.render(Qg, soft=True).sum().backward()
        torch.cuda.synchronize()
        t_grad = (time.perf_counter() - t) / (20 * 48)

        ious = iou(M_fr, M_mj)
        diff = np.abs(M_fr - M_mj).sum((1, 2))
        row = {
            "cluster_mm": float(cm), "supersample": int(ss), "coverage_threshold": float(thr),
            "triangles": fr.n_tri, "triangles_raw": fr.n_tri_raw,
            "ms_per_pose": {str(B): round(1e3 * v, 4) for B, v in times.items()},
            "speedup_vs_mujoco_at_192": round(t_mj / times[192], 1),
            "soft_fwd_bwd_ms_per_pose_b48": round(1e3 * t_grad, 4),
        }
        for k in ("random", "solved"):
            sel = kind == k
            row[k] = {
                "mean_iou": round(float(ious[sel].mean()), 4),
                "p05_iou": round(float(np.percentile(ious[sel], 5)), 4),
                "min_iou": round(float(ious[sel].min()), 4),
                "mean_px_diff": round(float(diff[sel].mean()), 2),
                "mean_mj_px": round(float(M_mj[sel].sum((1, 2)).mean()), 1),
            }
        report["settings"].append(row)
        print(json.dumps(row))
        # worst solved case, for eyeballing
        if spec == a.settings[0]:
            w = int(np.argmin(np.where(kind == "solved", ious, 2)))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            from PIL import Image
            vis = np.stack([M_mj[w], M_fr[w], np.zeros_like(M_mj[w])], -1)
            Image.fromarray((255 * (1 - vis)).astype(np.uint8)).resize((384, 384), 0).save(
                os.path.join(os.path.dirname(out), "worst_solved_pose.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
