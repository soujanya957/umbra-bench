"""Is fleet-shadow-art's diffrender/soft_shadow.py right, and is it useful for optimization?

Checks, on the 3 x SO-101 rig at 128 px, against MuJoCo as the reference, with
fastshadow's nvdiffrast renderer as the control:

  1. hard mask vs MuJoCo (IoU), on solved poses and random poses
  2. soft mask contrast: mean value inside vs outside MuJoCo's shadow
  3. gradient usefulness: does a step along -grad(soft 1-IoU vs a glyph target) raise the
     REAL (MuJoCo) IoU, and how well does the soft gradient align with finite differences
     of the real IoU? (autograd vs finite differences of the soft loss itself, which
     p0_diffrender.py checks, only tests the differentiation, not the direction.)
  4. time per render and per render+backward

Needs FLEET_SHADOW_ART at a fleet-shadow-art checkout that has diffrender/, plus
pytorch_kinematics, trimesh, urdf_parser_py (and fast_simplification for --decimate).

    python scripts/verify_soft_shadow.py
"""

import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastshadow  # noqa: E402

BENCH = fastshadow.BENCH
sys.path.insert(0, os.path.join(BENCH, "scripts"))
from run_base_optimizer import load_target  # noqa: E402

OUT = os.path.join(BENCH, "results", "fastshadow", "soft_shadow_check")


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def main():
    ms = fastshadow.use_fleet()
    import mujoco
    import pytorch_kinematics as pk
    import torch
    from diffrender.soft_shadow import SoftShadowRenderer, WallGeom
    from renderer import ShadowRenderer, build_scene

    sys.path.insert(0, os.path.join(ms, "scripts"))
    from p0_diffrender import load_link_meshes

    from fastshadow.renderer import FastShadowRenderer

    os.makedirs(OUT, exist_ok=True)
    urdf = "urdf/SO101/so101_new_calib.urdf"
    model, data, n = build_scene(urdf, n_robots=3, arm_gap=0.2)
    mj = ShadowRenderer(model, data, size=128, n_robots=n)
    fr = FastShadowRenderer(mj)
    geom = WallGeom.from_mujoco(model)
    with open(urdf, "rb") as fh:
        chain = pk.build_chain_from_urdf(fh.read()).to(dtype=torch.float32, device="cuda")
    offs = []
    for i in range(n):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot_{chr(ord('a') + i)}")
        offs.append((float(model.body_pos[bid][0]), float(model.body_pos[bid][1])))

    # poses: solved digits (realistic) and random
    root = os.path.join(BENCH, "optimized", "big-budget-grounded")
    solved, targets = [], []
    for f in sorted(glob.glob(os.path.join(root, "digits", "*", "results.json"))):
        r = json.load(open(f, encoding="utf-8"))
        stem = os.path.basename(os.path.dirname(f))
        T = load_target(os.path.join(os.path.dirname(f), f"{stem}_shown.png"), 128) > 0.5
        for run in r["runs"][:3]:
            solved.append(np.array(run["q_rad"]))
            targets.append(T)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(solved), 30, replace=False)
    Qs = [solved[i] for i in pick]
    Ts = [targets[i] for i in pick]
    Qr = [rng.uniform(mj.q_lower, mj.q_upper) for _ in range(20)]
    M_mj_s = [mj.get_shadow_mask(q) > 0.5 for q in Qs]
    M_mj_r = [mj.get_shadow_mask(q) > 0.5 for q in Qr]

    def mj_iou(q, T):
        return iou(mj.get_shadow_mask(np.clip(q, mj.q_lower, mj.q_upper)) > 0.5, T)

    report = {}
    variants = [("soft_shadow (full meshes)", 0), ("soft_shadow (2000 faces/link)", 2000), ("soft_shadow (400 faces/link)", 400)]
    renderers = []
    for name, dec in variants:
        meshes = load_link_meshes(urdf, torch.float32, dec)
        ntri = sum(int(f.shape[0]) for _, f in meshes.values()) * n
        sr = SoftShadowRenderer(chain, meshes, geom, size=128, device="cuda", base_offsets=offs)
        renderers.append((name, ntri, "soft", sr))
    renderers.append(("nvdiffrast (fastshadow)", fr.n_tri, "nvdr", fr))

    def render(kind, r, q, soft, tau=0.7):
        qt = torch.as_tensor(q, dtype=torch.float32, device="cuda")
        if kind == "soft":
            return r.render(qt, tau=tau, hard=not soft)
        return r.render(qt[None], soft=soft)[0]

    for name, ntri, kind, r in renderers:
        print(f"\n== {name}: {ntri} triangles", flush=True)
        res = {"triangles": ntri}
        # 1. hard masks
        ns = 30 if not (kind == "soft" and ntri > 200000) else 6  # full meshes are slow
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.no_grad():
            Hs = [render(kind, r, q, False).cpu().numpy() > 0.5 for q in Qs[:ns]]
        torch.cuda.synchronize(); t_hard = (time.perf_counter() - t) / ns
        with torch.no_grad():
            Hr = [render(kind, r, q, False).cpu().numpy() > 0.5 for q in Qr[:ns]]
        res["hard_iou_solved"] = float(np.mean([iou(a, b) for a, b in zip(Hs, M_mj_s)]))
        res["hard_iou_random"] = float(np.mean([iou(a, b) for a, b in zip(Hr, M_mj_r)]))
        res["ms_hard"] = 1000 * t_hard
        print(f"  1. hard mask vs MuJoCo IoU: solved {res['hard_iou_solved']:.4f}, random {res['hard_iou_random']:.4f}; "
              f"{res['ms_hard']:.1f} ms/pose", flush=True)
        # 2. soft contrast
        with torch.no_grad():
            Ss = [render(kind, r, q, True).cpu().numpy() for q in Qs[:ns]]
        ins = np.concatenate([s[m] for s, m in zip(Ss, M_mj_s)])
        out = np.concatenate([s[~m] for s, m in zip(Ss, M_mj_s)])
        res["soft_inside"], res["soft_outside"] = float(ins.mean()), float(out.mean())
        res["soft_inside_below_half"] = float((ins < 0.5).mean())
        print(f"  2. soft mask: inside mean {ins.mean():.3f} (share < 0.5: {100 * (ins < 0.5).mean():.1f}%), "
              f"outside mean {out.mean():.3f}", flush=True)
        # 3. gradient usefulness on real targets, from solved poses nudged off their optimum
        cos, wins, gains, t_g = [], 0, [], 0.0
        ng = 8 if not (kind == "soft" and ntri > 200000) else 2
        for k in range(ng):
            T = Ts[k]
            q0 = Qs[k] + rng.normal(0, 0.15, size=len(Qs[k]))  # off the optimum, so there is something to fix
            q0 = np.clip(q0, mj.q_lower, mj.q_upper)
            Tt = torch.tensor(T, dtype=torch.float32, device="cuda")
            qv = torch.tensor(q0, dtype=torch.float32, device="cuda", requires_grad=True)
            torch.cuda.synchronize(); t = time.perf_counter()
            S = render(kind, r, qv, True)
            inter = (S * Tt).sum()
            loss = 1 - inter / (S.sum() + Tt.sum() - inter + 1e-6)
            loss.backward()
            torch.cuda.synchronize(); t_g += time.perf_counter() - t
            g = qv.grad.detach().cpu().numpy()
            # finite differences of the REAL loss (MuJoCo), eps 0.02 rad
            fd = np.zeros_like(g)
            for j in range(len(g)):
                e = np.zeros_like(q0); e[j] = 0.02
                fd[j] = ((1 - mj_iou(q0 + e, T)) - (1 - mj_iou(q0 - e, T))) / 0.04
            cos.append(float(g @ fd / (np.linalg.norm(g) * np.linalg.norm(fd) + 1e-12)))
            base = mj_iou(q0, T)
            d = -g / (np.linalg.norm(g) + 1e-12)
            best = max(mj_iou(q0 + s * d, T) for s in (0.01, 0.03, 0.1))
            gains.append(best - base)
            wins += best > base + 1e-4
        res["grad_cos_vs_real_fd"] = float(np.mean(cos))
        res["step_improves_real_iou"] = wins / ng
        res["mean_real_iou_gain"] = float(np.mean(gains))
        res["ms_soft_fwd_bwd"] = 1000 * t_g / ng
        print(f"  3. gradient: cosine with finite differences of the real IoU {np.mean(cos):.3f}; "
              f"a step along it raises the real IoU in {wins}/{ng} cases (mean +{np.mean(gains):.4f}); "
              f"{res['ms_soft_fwd_bwd']:.0f} ms per forward+backward", flush=True)
        report[name] = res
    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2)
    print("\nwrote", os.path.join(OUT, "report.json"))


if __name__ == "__main__":
    main()
