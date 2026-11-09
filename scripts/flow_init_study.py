"""Does a conditional flow-matching model give better solve starts than the pose library?

Same protocol as pose_library_study.py: the bench's 30 fitted targets, held out by character
(every font, both cases), 3 seeds, the same final-pass CMA-ES budgets, and every number
re-measured by MuJoCo. Three ways to start the same refinement:

  rand_*  uniform random poses inside the joint limits, floor-clean, best by IoU
  lib_*   the pose library (stored solves, rendered on this rig, scored by true IoU)
  fm_*    K poses sampled from p(q | target), rendered, floor- and contact-checked, best by IoU

  *_only_K     the start alone (K candidates rendered and ranked)
  *_cma_fN     best start, then the joint final pass of CMA-ES for N iterations
  *3_cma_fN    the 3 best different starts (0.3 rad apart), each refined, best kept

Training data (`data`): every stored 3-arm solve whose pose is in the pose library (so it is
floor- and contact-clean and no held-out character is in it), paired with the target it was
solved for, plus every library pose paired with its own shadow rendered on this rig. The model
sees the targets it will be asked about (unreachable glyphs) as well as reachable shadows.

    python scripts/flow_init_study.py data
    python scripts/flow_init_study.py train
    python scripts/flow_init_study.py select            # snapshot by validation start IoU
    python scripts/flow_init_study.py run --seeds 3
    python scripts/flow_init_study.py report

Result (2026-11-09, results/fastshadow/flow_init, 30 targets x 3 seeds, K=64):
  start alone    IoU / tc_iou   rand 0.413/0.348  lib 0.652/0.561 (2 ms)  fm 0.531/0.476 (29 ms)
  + CMA-ES f30   IoU / tc_iou   rand 0.644/0.509  lib 0.714/0.606         fm 0.686/0.596
  fm - lib: start IoU -0.122 (95% -0.137..-0.105); after f30 -0.028 (-0.041..-0.015), tc_iou -0.010 (-0.050..+0.033)
  good starts (>=0.3 rad apart, within 0.05 IoU of best): lib 13.0, fm 4.1, rand 3.0
  fm samples clearing the table: 82% (random poses 44%)
  More samples help, guidance and more Euler steps do not: fm start IoU 0.544 / 0.580 / 0.605 / 0.623
  at K = 64 / 256 / 1024 / 4096, still under the library's one lookup (0.652).
  Held-out flow-matching MSE is a poor model-selection signal here: it rises from step 2k
  while validation start IoU keeps improving to step 14k.
So on this rig the flow model trained on the library's own solves only interpolates them,
and loses to exact retrieval over the same solves on quality, diversity and time.
"""

import argparse
import contextlib
import glob
import io
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastshadow  # noqa: E402
from pose_library_study import FLOOR_TOL, ROOT, bench_ids, held_out, rig  # noqa: E402

BENCH = fastshadow.BENCH
sys.path.insert(0, os.path.join(BENCH, "scripts"))
from metrics import trimmed_centred  # noqa: E402
from run_base_optimizer import load_target  # noqa: E402

LIB = os.path.join(BENCH, "results", "fastshadow", "pose_library", "library_n3.npz")
OUT = os.path.join(BENCH, "results", "fastshadow", "flow_init")
COND = 64  # condition image side; targets and shadows are 128 on the rig, average-pooled to this


def key(Q):
    return [tuple(r) for r in np.round(np.asarray(Q) / 0.01).astype(np.int32)]


def pool(m, size=COND):
    """(B,H,W) torch coverage -> (B,size,size) float in [0,1]."""
    import torch.nn.functional as F
    return F.adaptive_avg_pool2d(m[:, None].float(), size)[:, 0]


# ---------------------------------------------------------------- data

def data(a):
    import torch

    mj, fr, br, _ = rig()
    Q = np.load(LIB)["Q"]
    lib_keys = {k: i for i, k in enumerate(key(Q))}
    chars = held_out()

    # 1. library poses with their own shadows on this rig
    with torch.no_grad():
        R = torch.cat([pool(fr.render(Q[i: i + 2048])).cpu() for i in range(0, len(Q), 2048)])
    print(f"rendered {len(Q)} library shadows", flush=True)

    # 2. (target, solve) pairs from every stored run whose pose is in the library
    root = os.path.join(BENCH, "optimized")
    T, TQ, tids, cache = [], [], [], {}
    for f in glob.glob(os.path.join(root, "**", "results.json"), recursive=True):
        parts = os.path.relpath(f, root).replace("\\", "/").split("/")
        if parts[0] in ("base-optimizer", "_attic"):
            continue
        d = json.load(open(f, encoding="utf-8"))
        r = d.get("rig", {})
        if r.get("n_robots") != 3 or r.get("arm_gap_m") != 0.2 or r.get("distortion"):
            continue
        stem, subset = parts[-2], parts[-3] if len(parts) >= 3 else ""
        if ("letter" in subset or "digit" in subset) and stem.split("_")[0].lower() in chars:
            continue
        shown = os.path.join(os.path.dirname(f), f"{stem}_shown.png")
        if os.path.exists(shown):
            path = shown
        elif not r.get("target_fit") and d.get("target"):
            path = os.path.join(BENCH, d["target"])
        else:
            continue
        if not os.path.exists(path):
            continue
        qs = [q["q_rad"] for q in d.get("runs", []) if "q_rad" in q]
        idx = [lib_keys.get(k) for k in key(qs)] if qs else []
        idx = [i for i in idx if i is not None]
        if not idx:
            continue
        if path not in cache:
            img = torch.as_tensor(load_target(path, mj.size), dtype=torch.float32)
            cache[path] = len(T)
            T.append(pool(img[None])[0])
            tids.append(f"{subset}/{stem}")
        for i in idx:
            TQ.append((cache[path], i))
    T = torch.stack(T)
    TQ = np.array(TQ, np.int64)
    print(f"{len(T)} targets, {len(TQ)} (target, solve) pairs, "
          f"{len(np.unique(TQ[:, 1]))} distinct poses among them", flush=True)

    # 3. uniform random poses with their shadows: the reachable-shadow space at large, too
    # many to memorise (floor and contacts not checked: they teach geometry, not good solves)
    lo, hi = fr.q_lower[: Q.shape[1]], fr.q_upper[: Q.shape[1]]
    RQ = (lo + (hi - lo) * np.random.default_rng(0).random((a.random, Q.shape[1]))).astype(np.float32)
    with torch.no_grad():
        RR = torch.cat([(pool(fr.render(RQ[i: i + 2048])) * 255).round().to(torch.uint8).cpu()
                        for i in range(0, len(RQ), 2048)])
    print(f"rendered {len(RQ)} random poses", flush=True)

    os.makedirs(OUT, exist_ok=True)
    torch.save({"Q": torch.as_tensor(Q), "R": (R * 255).round().to(torch.uint8),
                "RQ": torch.as_tensor(RQ), "RR": RR,
                "T": (T * 255).round().to(torch.uint8), "TQ": torch.as_tensor(TQ), "tids": tids,
                "q_lower": torch.as_tensor(fr.q_lower[: Q.shape[1]], dtype=torch.float32),
                "q_upper": torch.as_tensor(fr.q_upper[: Q.shape[1]], dtype=torch.float32)},
               os.path.join(OUT, "data.pt"))


# ---------------------------------------------------------------- model

def make_model(n_dof, width=512, depth=4, dropout=0.1):
    import torch
    import torch.nn as nn

    class Encoder(nn.Module):
        """Keeps where the shadow is: the strided convs end in a 4x4 grid that is flattened."""
        def __init__(self, out=512):
            super().__init__()
            ch = [1, 32, 64, 128, 256]
            layers = []
            for i in range(4):
                layers += [nn.Conv2d(ch[i], ch[i + 1], 4, 2, 1), nn.GroupNorm(8, ch[i + 1]), nn.SiLU()]
            self.conv = nn.Sequential(*layers)
            self.fc = nn.Linear(256 * (COND // 16) ** 2, out)

        def forward(self, x):
            return self.fc(self.conv(x[:, None]).flatten(1))

    class Block(nn.Module):
        def __init__(self, w, c):
            super().__init__()
            self.norm = nn.LayerNorm(w)
            self.film = nn.Linear(c, 2 * w)
            self.mlp = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Dropout(dropout), nn.Linear(w, w))

        def forward(self, h, c):
            s, b = self.film(c).chunk(2, -1)
            return h + self.mlp(self.norm(h) * (1 + s) + b)

    class Flow(nn.Module):
        """v_theta(x_t, t, image): velocity of the straight path from noise to the pose
        (in joint-limit units, [-1, 1])."""
        def __init__(self):
            super().__init__()
            self.enc = Encoder()
            self.null = nn.Parameter(torch.zeros(512))  # condition-dropout token
            self.tfreq = nn.Parameter(torch.randn(32) * 4.0, requires_grad=False)
            c = 512 + 64
            self.inp = nn.Linear(n_dof, width)
            self.blocks = nn.ModuleList([Block(width, c) for _ in range(depth)])
            self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, n_dof))
            nn.init.zeros_(self.out[1].weight)
            nn.init.zeros_(self.out[1].bias)

        def cond(self, img, drop=None):
            e = self.enc(img)
            if drop is not None:
                e = torch.where(drop[:, None], self.null.expand_as(e), e)
            return e

        def forward(self, x, t, e):
            a = t[:, None] * self.tfreq[None] * 2 * math.pi
            c = torch.cat([e, torch.sin(a), torch.cos(a)], -1)
            h = self.inp(x)
            for b in self.blocks:
                h = b(h, c)
            return self.out(h)

    return Flow()


class Sampler:
    def __init__(self, path, device="cuda"):
        import torch
        ck = torch.load(path, map_location=device, weights_only=False)
        self.lo, self.hi = ck["q_lower"].to(device), ck["q_upper"].to(device)
        self.net = make_model(len(self.lo), ck["args"]["width"]).to(device)
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.device = device

    def sample(self, target, k, steps=16, guidance=1.0, seed=0):
        """target (H,W) array -> (k, n_dof) poses inside the joint limits (Euler on the flow)."""
        import torch
        g = torch.Generator(device=self.device).manual_seed(seed)
        with torch.no_grad():
            t = target if torch.is_tensor(target) else torch.as_tensor(np.asarray(target))
            img = pool(t.to(self.device, torch.float32)[None])
            e = self.net.cond(img).expand(k, -1)
            e0 = self.net.null[None].expand(k, -1) if guidance != 1.0 else None
            x = torch.randn(k, len(self.lo), device=self.device, generator=g)
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((k,), i * dt, device=self.device)
                v = self.net(x, t, e)
                if e0 is not None:
                    v = self.net(x, t, e0) + guidance * (v - self.net(x, t, e0))
                x = x + dt * v
            u = x.clamp(-1, 1)
        return ((u + 1) / 2 * (self.hi - self.lo) + self.lo).cpu().numpy()


# ---------------------------------------------------------------- train

def train(a):
    import torch

    torch.manual_seed(0)
    dev = "cuda"
    D = torch.load(os.path.join(OUT, "data.pt"))
    lo, hi = D["q_lower"].to(dev), D["q_upper"].to(dev)
    U = (2 * (D["Q"].to(dev) - lo) / (hi - lo) - 1).clamp(-1, 1)
    R = D["R"].to(dev)
    RU = (2 * (D["RQ"].to(dev) - lo) / (hi - lo) - 1).clamp(-1, 1)
    RR = D["RR"].to(dev)
    T = D["T"].to(dev)
    TQ = D["TQ"].to(dev)
    # hold out 5% of the targets (by target) to watch over-fitting
    g = np.random.default_rng(0)
    val_t = torch.as_tensor(g.permutation(len(T))[: max(1, len(T) // 20)], device=dev)
    is_val = torch.isin(TQ[:, 0], val_t)
    TQtr, TQva = TQ[~is_val], TQ[is_val]
    print(f"train pairs: {len(TQtr)} target + {len(U)} library rendered + {len(RU)} random rendered; "
          f"val target pairs {len(TQva)}", flush=True)

    net = make_model(U.shape[1], a.width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    B = a.batch

    def batch(src_t, n):
        # (target, solve), (library shadow, pose), (random shadow, pose) in a.mix proportions
        nt, nl = int(n * a.mix[0]), int(n * a.mix[1])
        nr = n - nt - nl
        j = src_t[torch.randint(len(src_t), (nt,), device=dev)]
        il = torch.randint(len(U), (nl,), device=dev)
        ir = torch.randint(len(RU), (nr,), device=dev)
        img = torch.cat([T[j[:, 0]], R[il], RR[ir]])
        return img.float() / 255, torch.cat([U[j[:, 1]], U[il], RU[ir]])

    def loss_on(img, x1, drop_p):
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1), device=dev)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        drop = torch.rand(len(x1), device=dev) < drop_p
        v = net(xt, t, net.cond(img, drop))
        return ((v - (x1 - x0)) ** 2).mean()

    log = open(os.path.join(OUT, "train_log.txt"), "w")
    t0 = time.perf_counter()
    best = float("inf")
    for step in range(a.steps):
        img, x1 = batch(TQtr, B)
        loss = loss_on(img, x1, 0.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == a.steps - 1:
            net.eval()
            with torch.no_grad():
                torch.manual_seed(1)
                vl = np.mean([loss_on(T[j[:, 0]].float() / 255, U[j[:, 1]], 0.0).item()
                              for j in TQva[torch.randint(len(TQva), (8, 1024), device=dev)]])
            net.train()
            msg = f"step {step:6d}  train {loss.item():.4f}  val(target pairs) {vl:.4f}  {time.perf_counter() - t0:.0f} s"
            # MSE on held-out targets is only a proxy for a multi-solution problem (a sharper
            # model can score worse), so keep snapshots and pick by start IoU (`select`).
            ck = {"model": net.state_dict(), "q_lower": D["q_lower"], "q_upper": D["q_upper"],
                  "args": vars(a), "step": step, "val": float(vl), "val_targets": val_t.cpu()}
            if vl < best:
                best = vl
                msg += "  *best val"
                torch.save(ck, os.path.join(OUT, "flow_bestval.pt"))
            if step % 2000 == 0 or step == a.steps - 1:
                os.makedirs(os.path.join(OUT, "snapshots"), exist_ok=True)
                torch.save(ck, os.path.join(OUT, "snapshots", f"flow_s{step:06d}.pt"))
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()


# ---------------------------------------------------------------- select

def select(a):
    """Score every snapshot by the best floor-clean start IoU on the validation targets (never
    the 30 test glyphs), at the condition resolution; copy the winner to flow.pt."""
    import shutil

    import torch

    mj, fr, br, _ = rig()
    D = torch.load(os.path.join(OUT, "data.pt"))
    snaps = sorted(glob.glob(os.path.join(OUT, "snapshots", "flow_s*.pt")))
    val_t = torch.load(snaps[0], weights_only=False)["val_targets"].numpy()
    val_t = val_t[np.random.default_rng(0).permutation(len(val_t))[: a.n_val]]
    T = (D["T"][val_t].float() / 255).cuda()

    def score(fm, k, steps, guidance):
        ious, t0 = [], time.perf_counter()
        for i in range(len(T)):
            Qc = fm.sample(T[i], k, steps=steps, guidance=guidance, seed=i)
            with torch.no_grad():
                m = pool(fr.render(Qc)) > 0.5
                t = T[i] > 0.5
                inter = (m & t).flatten(1).sum(1)
                iou = inter / (m.flatten(1).sum(1) + t.sum() - inter).clamp(min=1)
                ok = fr.moving_min_z(np.asarray(Qc, np.float64)) >= FLOOR_TOL
            ious.append(float(torch.where(ok, iou, iou - 2).max()))
        torch.cuda.synchronize()
        return float(np.mean(ious)), (time.perf_counter() - t0) / len(T)

    if a.sweep:  # sampling settings for the selected model; the library is not comparable here
        fm = Sampler(os.path.join(OUT, "flow.pt"))  # (its source holds these targets' own solves)
        for k in (64, 256, 1024):
            for steps in (16, 32):
                for gd in (1.0, 2.0, 3.0):
                    v, dt = score(fm, k, steps, gd)
                    print(f"K {k:5d}  steps {steps:3d}  guidance {gd:.1f}  start IoU {v:.4f}  {dt * 1000:.0f} ms/target", flush=True)
        return
    res = {}
    for s in snaps:
        res[s] = score(Sampler(s), a.k, a.steps, 1.0)[0]
        print(f"{os.path.basename(s)}  val loss {torch.load(s, weights_only=False)['val']:.4f}  best-of-{a.k} start IoU {res[s]:.4f}", flush=True)
    win = max(res, key=res.get)
    shutil.copy(win, os.path.join(OUT, "flow.pt"))
    print("selected", os.path.basename(win))


# ---------------------------------------------------------------- run

def run(a):
    mj, fr, br, ocfg = rig()
    import optimizer as opt  # noqa: F401
    import torch
    from optimizer import OptimizerConfig, optimize_staged

    from fastshadow.grad_opt import GradConfig, pick_by_mujoco, verify
    from fastshadow.pose_library import PoseLibrary

    Q = np.load(LIB)["Q"]
    lib = PoseLibrary.build(fr, Q)
    fm = Sampler(os.path.join(OUT, "flow.pt"))
    gcfg = GradConfig(floor_penalty=ocfg["floor_penalty"], collision_penalty=ocfg["collision_penalty"],
                      self_collision_penalty=ocfg["self_collision_penalty"])
    mj.floor_mode = "mesh"
    lo, hi = fr.q_lower[: Q.shape[1]], fr.q_upper[: Q.shape[1]]

    targets = {}
    for tid in bench_ids():
        sub, stem = tid.split("/")
        d = os.path.join(ROOT, sub, stem)
        targets[tid] = (load_target(os.path.join(d, f"{stem}_shown.png"), mj.size),
                        load_target(os.path.join(d, f"{stem}_original.png"), mj.size))

    path = os.path.join(OUT, "runs.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["id"], r["method"], r["seed"]) for r in map(json.loads, open(path, encoding="utf-8"))}

    def log(rec):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def measure(q, T, T0):
        v = verify(mj, np.asarray(q, np.float64), T, gcfg)
        m = v.pop("mask")
        v["tc_iou"] = trimmed_centred(T0 > 0.5, m > 0.5)["tc_iou"]
        with torch.no_grad():
            v["floor_z"] = float(fr.moving_min_z(np.asarray(q, np.float64)[None])[0])
        v["clean"] = bool(v["cross_pen_m"] <= 0 and v["self_pen_m"] <= 0 and v["floor_z"] >= FLOOR_TOL)
        return v

    def contact_free(q):
        p = br._penalties(np.asarray(q, np.float64))
        return p[1] <= 0 and p[2] <= 0

    def rank(Qc, T, k, min_dist=0.0, check=True):
        """Render candidates, drop floor hits, order by IoU; walk down, keep contact-free ones
        (and min_dist apart) until k. Returns poses, their IoU, and the candidate stats."""
        t = torch.as_tensor(np.asarray(T) > 0.5, device=fr.device)
        with torch.no_grad():
            m = fr.render(Qc) > 0.5
            inter = (m & t).flatten(1).sum(1)
            iou = (inter / (m.flatten(1).sum(1) + t.sum() - inter).clamp(min=1)).float()
            floor_ok = fr.moving_min_z(np.asarray(Qc, np.float64)) >= FLOOR_TOL
        iou = torch.where(floor_ok, iou, iou - 2).cpu().numpy()
        picked, n_checked = [], 0
        for j in np.argsort(-iou):
            if iou[j] < -1:
                break
            if any(np.abs(Qc[j] - Qc[p]).max() < min_dist for p in picked):
                continue
            n_checked += 1
            if check and not contact_free(Qc[j]):
                continue
            picked.append(j)
            if len(picked) == k:
                break
        if not picked:  # nothing clean: fall back to the best IoU
            picked = [int(np.argmax(iou))]
        stats = {"floor_ok": float(floor_ok.float().mean()), "n_checked": n_checked}
        return Qc[picked], iou[picked], stats

    def starts(src, T, s, k, min_dist=0.0):
        """(poses, start stats) for one start source; timing includes everything."""
        if src == "lib":
            q, v = lib.query(T, k, min_dist=min_dist)
            return q, {}
        if src == "fm":
            Qc = fm.sample(T, a.k, steps=a.steps, guidance=a.guidance, seed=1000 * s + 7)
        else:
            g = np.random.default_rng(1000 * s + 7)
            Qc = (lo + (hi - lo) * g.random((a.k, len(lo)))).astype(np.float32)
        q, _, st = rank(Qc, T, k, min_dist)
        return q, st

    def cma(T, seed, iters, x0):
        p = {4: 12, 8: 16, 16: 24, 30: 48}[iters]
        kw = {**ocfg, "popsize": p, "phase1_iters": 0, "phase2_iters": 0, "final_iters": iters}
        with contextlib.redirect_stdout(io.StringIO()):
            return optimize_staged(br, T, OptimizerConfig(seed=seed, skip_greedy=True, **kw), x0=x0.astype(np.float64))

    def diversity(T, src, s):
        """How many different (>=0.3 rad apart) clean starts within 0.05 IoU of the best."""
        if src == "lib":
            q, v = lib.query(T, 16, min_dist=0.3)
        elif src == "fm":
            q, v, _ = rank(fm.sample(T, a.k, steps=a.steps, guidance=a.guidance, seed=1000 * s + 7), T, 16, 0.3)
        else:
            g = np.random.default_rng(1000 * s + 7)
            q, v, _ = rank((lo + (hi - lo) * g.random((a.k, len(lo)))).astype(np.float32), T, 16, 0.3)
        return int((np.asarray(v) >= np.max(v) - 0.05).sum())

    sync = torch.cuda.synchronize
    # warm up the GPU paths once so the first target is not charged for it
    T_, _ = targets[bench_ids()[0]]
    fm.sample(T_, a.k, steps=a.steps)
    lib.query(T_, 1)
    sync()

    srcs = ("rand", "lib", "fm")
    for ti, tid in enumerate(bench_ids()):
        T, T0 = targets[tid]

        def rec(meth, s, q, dt, extra=None):
            v = measure(q, T, T0)
            log({"id": tid, "method": meth, "seed": s, "seconds": dt,
                 **{k: v[k] for k in ("iou", "tc_iou", "loss", "clean", "floor_z")}, **(extra or {})})

        for s in range(a.seeds):
            for src in srcs:
                m = f"{src}_only"
                if (tid, m, s) not in done:
                    t0 = time.perf_counter()
                    q, st = starts(src, T, s, 1)
                    sync()
                    rec(m, s, q[0], time.perf_counter() - t0, {**st, "n_good_starts": diversity(T, src, s)})
                for it in (30, 8, 4):
                    m = f"{src}_cma_f{it}"
                    if (tid, m, s) not in done:
                        t0 = time.perf_counter()
                        q, _ = starts(src, T, s, 1)
                        r = cma(T, s, it, q[0])
                        rec(m, s, r.best_q, time.perf_counter() - t0, {"n_evals": int(r.n_evals)})
                for it in (8, 4):
                    m = f"{src}3_cma_f{it}"
                    if (tid, m, s) not in done:
                        t0 = time.perf_counter()
                        qs, _ = starts(src, T, s, 3, min_dist=0.3)
                        rs = [cma(T, s + 10 * i, it, q) for i, q in enumerate(qs)]
                        best, _ = pick_by_mujoco(mj, np.array([x.best_q for x in rs]), T, gcfg)
                        dt = time.perf_counter() - t0
                        fin = [measure(x.best_q, T, T0) for x in rs]
                        top = max(f["iou"] for f in fin)
                        rec(m, s, best, dt, {"n_evals": sum(int(x.n_evals) for x in rs), "n_starts": len(qs),
                                             "start_ious": [f["iou"] for f in fin],
                                             "n_within_0.02": sum(f["iou"] >= top - 0.02 and f["clean"] for f in fin)})
        print(f"[{ti + 1}/30] {tid}", flush=True)


# ---------------------------------------------------------------- report

def report(a):
    rows = [json.loads(l) for l in open(os.path.join(OUT, "runs.jsonl"), encoding="utf-8")]
    old = os.path.join(BENCH, "results", "fastshadow", "pose_library", "runs.jsonl")
    rows += [dict(r, method="ref_" + r["method"]) for r in map(json.loads, open(old, encoding="utf-8"))
             if r["method"] in ("cold_8k", "cold_1k", "cold_300")]
    ms = sorted({r["method"] for r in rows})
    tab = []
    for m in ms:
        R = [r for r in rows if r["method"] == m]
        ids = sorted({r["id"] for r in R})
        per = lambda k: np.array([np.mean([r[k] for r in R if r["id"] == i]) for i in ids])  # noqa: E731
        opt = lambda k: np.mean([r[k] for r in R]) if k in R[0] else float("nan")  # noqa: E731
        tab.append((m, len(ids), per("iou").mean(), per("tc_iou").mean(), np.median([r["seconds"] for r in R]),
                    np.mean([r["clean"] for r in R]), opt("n_good_starts"), opt("n_within_0.02"), opt("floor_ok")))
    tab.sort(key=lambda x: (x[0].split("_", 1)[1] if not x[0].startswith("ref") else "~", x[0]))
    hdr = f"{'method':<16}{'n':>4}{'IoU':>8}{'tc_iou':>8}{'s/solve':>9}{'clean':>7}{'goodstarts':>11}{'kept/3':>8}{'floor_ok':>9}"
    print(hdr)
    lines = [hdr]
    for t in tab:
        line = f"{t[0]:<16}{t[1]:>4}{t[2]:>8.3f}{t[3]:>8.3f}{t[4]:>9.3f}{t[5]:>7.2f}{t[6]:>11.1f}{t[7]:>8.2f}{t[8]:>9.2f}"
        print(line)
        lines.append(line)
    # paired differences fm - lib per target (mean over seeds), bootstrap 95% CI
    g = np.random.default_rng(0)
    for suffix in ("only", "cma_f30", "cma_f8", "cma_f4"):
        for k in ("iou", "tc_iou"):
            d = []
            for tid in sorted({r["id"] for r in rows if r["method"] == f"fm_{suffix}"}):
                f = np.mean([r[k] for r in rows if r["id"] == tid and r["method"] == f"fm_{suffix}"])
                l = np.mean([r[k] for r in rows if r["id"] == tid and r["method"] == f"lib_{suffix}"])
                d.append(f - l)
            if not d:
                continue
            d = np.array(d)
            bs = [g.choice(d, len(d)).mean() for _ in range(5000)]
            line = f"fm - lib  {suffix:<8} {k:<7} {d.mean():+.3f}  95% {np.percentile(bs, 2.5):+.3f}..{np.percentile(bs, 97.5):+.3f}"
            print(line)
            lines.append(line)
    open(os.path.join(OUT, "report.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["data", "train", "select", "run", "report"])
    p.add_argument("--n-val", type=int, default=80, help="select: validation targets to score")
    p.add_argument("--sweep", action="store_true", help="select: K, Euler steps and guidance for flow.pt")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=16, help="run: Euler steps; train: optimizer steps")
    p.add_argument("--k", type=int, default=64, help="candidates sampled per target")
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--mix", type=float, nargs=3, default=(0.4, 0.3, 0.3),
                   help="train batch shares: (target, solve), (library shadow, pose), (random shadow, pose)")
    p.add_argument("--random", type=int, default=100000, help="data: random poses to render")
    a = p.parse_args()
    if a.cmd == "train" and a.steps == 16:
        a.steps = 20000
    {"data": data, "train": train, "select": select, "run": run, "report": report}[a.cmd](a)
