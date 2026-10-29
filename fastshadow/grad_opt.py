"""Gradient-based shadow solve on the nvdiffrast renderer.

The objective is the CMA-ES one term for term (optimizer._full_loss with the
run_base_optimizer settings), so a difference in the result is the search, not the
target being chased:

    alpha * (1 - IoU) + beta * mean(DT_out * S)          shadow (loss.shadow_loss)
  + floor_penalty_loss(min_z)                            same barrier, same min_z
  + collision_penalty * cross + self_collision_penalty * self

`min_z` is the MuJoCo renderer's own formula (geom centre z minus capped bounding
radius), which is already a differentiable function of the joint angles.
Penetration depth is MuJoCo's convex-mesh contact solver there; here it is a sphere
proxy (k-means spheres per link). The proxy only steers the search: every reported
number is re-measured with MuJoCo (verify()), contacts included.

Search: many restarts in one batch. A large random draw is scored with the hard
renderer and the best `n_restarts` start Adam on the soft renderer. The IoU term is
computed on Gaussian-blurred shadow and target, with the blur shrinking to zero over
the run: the antialiased mask only has gradient at its silhouette, and blurring the
target is what lets an edge feel a target that is still several pixels away.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


@dataclass
class GradConfig:
    alpha: float = 1.0
    beta: float = 0.5
    floor_z: float = 0.0
    floor_penalty: float = 40.0
    floor_margin: float = 0.01
    floor_barrier: float = 1500.0
    collision_penalty: float = 400.0
    self_collision_penalty: float = 200.0

    n_init: int = 2048  # random poses scored before choosing starts
    n_restarts: int = 64  # Adam runs in parallel
    iters: int = 300
    lr: float = 0.05
    lr_final: float = 0.003
    blur_sigmas: tuple = (6.0, 3.0, 1.5, 0.0)  # px, one stage each, equal length
    eval_every: int = 10
    top_k: int = 8  # restarts handed back for MuJoCo re-scoring
    # At each blur-stage boundary, replace the worse half of the restarts with
    # perturbed copies of the better half (fresh Adam state). 0 = independent runs.
    reseed_sigma: float = 0.0
    # Population variants (solver bench sweep; both off by default):
    # langevin: Gaussian noise of this size (rad) added after every Adam step, annealed
    #   with the learning rate, so a start can hop out of a shallow basin early on.
    # svgd_T: Stein variational gradient descent. Each start's gradient is the kernel-
    #   weighted mean of the starts' gradients minus a repulsion of this strength, so
    #   starts that drift into the same basin push each other apart.
    langevin: float = 0.0
    svgd_T: float = 0.0
    svgd_bw: float = 1.0  # kernel bandwidth as a fraction of the median heuristic (smaller = more local)
    seed: int = 0


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """(B,H,W) separable blur; sigma 0 is the identity."""
    if sigma <= 0:
        return x
    r = int(math.ceil(3 * sigma))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    y = x[:, None]
    y = F.conv2d(F.pad(y, (r, r, 0, 0)), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r)), k.view(1, 1, -1, 1))
    return y[:, 0]


class Penalties:
    """Floor + collision terms, batched and differentiable."""

    def __init__(self, fr, solved_q: np.ndarray | None = None, spheres_per_body: int = 6, seed: int = 0,
                 floor: str = "mujoco"):
        """floor: "mujoco" is ShadowRenderer.eval_q's min_z (pinned at -1.83 cm by the
        static base, so the barrier is a constant); "mesh" is the lowest vertex of the
        moving bodies, as BatchedShadowRenderer(floor="mesh") does for CMA-ES."""
        import mujoco

        assert floor in ("mujoco", "mesh")
        ref, model, dev = fr.ref, fr.model, fr.device
        self.fr, self.floor = fr, floor
        if floor == "mesh":
            fr.moving_min_z(np.zeros((1, fr.n_dof)))  # builds fr._moving_vert
        # ── floor: exactly ShadowRenderer.eval_q's min_z ────────────────────
        g = ref.all_arm_geom_ids
        self.g_body = torch.tensor(model.geom_bodyid[g], device=dev)
        self.g_pos = torch.tensor(model.geom_pos[g], dtype=torch.float32, device=dev)
        self.g_zext = torch.tensor(ref.arm_geom_zext, dtype=torch.float32, device=dev)

        # ── collision proxy: k-means spheres per body, from the casting verts ─
        rng = np.random.default_rng(seed)
        vloc = fr.verts_local.cpu().numpy()
        geom_to_robot = ref._geom_to_robot
        body_robot = {}
        for gi, r in geom_to_robot.items():
            body_robot[int(model.geom_bodyid[gi])] = r
        # contype/conaffinity: a body whose geoms can never collide gets no spheres
        collides = {}
        for gi in range(model.ngeom):
            b = int(model.geom_bodyid[gi])
            if model.geom_contype[gi] or model.geom_conaffinity[gi]:
                collides[b] = True
        centers, radii, s_body = [], [], []
        for b, s0, n in fr._body_blocks:
            if b not in body_robot or not collides.get(b):
                continue
            V = vloc[s0 : s0 + n]
            k = min(spheres_per_body, max(1, n // 50))
            C = V[rng.choice(n, k, replace=False)]
            for _ in range(20):
                lab = np.argmin(((V[:, None] - C[None]) ** 2).sum(-1), 1)
                C = np.stack([V[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
            lab = np.argmin(((V[:, None] - C[None]) ** 2).sum(-1), 1)
            for j in range(k):
                d = np.linalg.norm(V[lab == j] - C[j], axis=1)
                if len(d) == 0:
                    continue
                centers.append(C[j])
                radii.append(float(d.max()))
                s_body.append(b)
        self.s_center = torch.tensor(np.array(centers), dtype=torch.float32, device=dev)
        self.s_radius = torch.tensor(radii, dtype=torch.float32, device=dev)
        self.s_body = torch.tensor(s_body, device=dev)
        s_body = np.array(s_body)
        parent = fr.parent
        excluded = set()
        for ex in range(model.nexclude):
            sig = int(model.exclude_signature[ex])
            excluded.add((sig >> 16, sig & 0xFFFF))
            excluded.add((sig & 0xFFFF, sig >> 16))
        ii, jj = np.triu_indices(len(s_body), 1)
        bi, bj = s_body[ii], s_body[jj]
        ri = np.array([body_robot[b] for b in bi])
        rj = np.array([body_robot[b] for b in bj])
        adjacent = np.array([parent[a] == b or parent[b] == a for a, b in zip(bi, bj)])
        excl = np.array([(int(a), int(b)) in excluded for a, b in zip(bi, bj)])
        keep = (bi != bj) & ~adjacent & ~excl
        self.cross = ri != rj
        rsum = self.s_radius[torch.as_tensor(ii)] + self.s_radius[torch.as_tensor(jj)]
        # Calibration. Max-radius spheres are fatter than MuJoCo's convex hulls, so
        # raw sphere overlap flags nearly every pose MuJoCo accepts (measured: self
        # overlap in 99.8% of collision-free solved poses). Each pair's contact
        # distance is instead capped at the closest that pair ever came in a solved
        # pose MuJoCo found contact-free: the proxy is then exactly zero on all of
        # them and only pushes back when a pose goes closer than any accepted one.
        self.n_calib = 0
        if solved_q is not None and len(solved_q):
            free = np.array([q for q in solved_q if sum(ref.eval_q(np.asarray(q))[2:]) <= 0])
            self.n_calib = len(free)
            with torch.no_grad():
                dmin = None
                for s in range(0, len(free), 512):
                    W = self._sphere_world(fr.pose(fr.clip(fr.as_q(free[s : s + 512]))))
                    d = (W[:, torch.as_tensor(ii, device=dev)] - W[:, torch.as_tensor(jj, device=dev)]).norm(dim=-1).amin(0)
                    dmin = d if dmin is None else torch.minimum(dmin, d)
            rsum = torch.minimum(rsum, dmin)
        self.pi = torch.tensor(ii[keep], device=dev)
        self.pj = torch.tensor(jj[keep], device=dev)
        self.prsum = rsum[torch.as_tensor(keep, device=dev)]
        self.pcross = torch.tensor(self.cross[keep], device=dev)

    def _sphere_world(self, Rp):
        R, p = Rp
        Rb, pb = R[:, self.s_body], p[:, self.s_body]
        return torch.einsum("bkij,kj->bki", Rb, self.s_center) + pb

    def min_z(self, Rp) -> torch.Tensor:
        if self.floor == "mesh":
            return self.fr.world_verts(None, Rp=Rp)[:, self.fr._moving_vert, 2].amin(1)
        R, p = Rp
        c = torch.einsum("bkij,kj->bki", R[:, self.g_body], self.g_pos) + p[:, self.g_body]
        return (c[..., 2] - self.g_zext).amin(1)

    def collision(self, Rp):
        W = self._sphere_world(Rp)
        d = (W[:, self.pi] - W[:, self.pj]).norm(dim=-1)
        pen = torch.relu(self.prsum - d)
        cross = (pen * self.pcross).sum(1)
        self_ = (pen * ~self.pcross).sum(1)
        return cross, self_

    def total(self, Rp, cfg: GradConfig) -> torch.Tensor:
        mz = self.min_z(Rp)
        line = cfg.floor_z + cfg.floor_margin
        floor = cfg.floor_penalty * torch.relu(line - mz) + cfg.floor_penalty * cfg.floor_barrier * torch.relu(cfg.floor_z - mz) ** 2
        cross, self_ = self.collision(Rp)
        return floor + cfg.collision_penalty * cross + cfg.self_collision_penalty * self_


class Objective:
    def __init__(self, fr, pen: Penalties, target: np.ndarray, cfg: GradConfig):
        self.fr, self.pen, self.cfg = fr, pen, cfg
        H, W = target.shape
        self.T = torch.tensor(target, dtype=torch.float32, device=fr.device)
        dt = distance_transform_edt(target == 0).astype(np.float32) / float(np.sqrt(H * W))
        self.DT = torch.tensor(dt, device=fr.device)
        self._Tb = {}

    def T_blur(self, sigma):
        if sigma not in self._Tb:
            self._Tb[sigma] = gaussian_blur(self.T[None], sigma)[0]
        return self._Tb[sigma]

    def shadow_terms(self, S, sigma=0.0):
        Sb = gaussian_blur(S, sigma)
        Tb = self.T_blur(sigma)
        inter = (Sb * Tb).sum((1, 2))
        union = Sb.sum((1, 2)) + Tb.sum() - inter
        iou_l = 1.0 - inter / (union + 1e-6)
        dt_l = (self.DT * S).mean((1, 2))
        return self.cfg.alpha * iou_l + self.cfg.beta * dt_l

    def __call__(self, Q, soft: bool, sigma: float = 0.0):
        """Per-pose loss (B,). Hard: the CMA-ES loss with proxy collisions."""
        Q = self.fr.clip(Q)
        Rp = self.fr.pose(Q)
        S = self.fr.render(Q, soft=soft, Rp=Rp)
        return self.shadow_terms(S, sigma if soft else 0.0) + self.pen.total(Rp, self.cfg), S

    def hard(self, Q, chunk=512):
        with torch.no_grad():
            return torch.cat([self(Q[i : i + chunk], soft=False)[0] for i in range(0, len(Q), chunk)])


@dataclass
class GradResult:
    best_q: np.ndarray
    best_loss: float
    seconds: float
    n_hard_renders: int
    n_soft_renders: int
    history: list = field(default_factory=list)
    top_q: np.ndarray | None = None  # best restarts, by fast hard loss, best first


def adam_refine(obj: Objective, Q0: torch.Tensor, cfg: GradConfig, iters: int, sigmas, lr, lr_final, history=None):
    """Parallel Adam from each row of Q0. Returns per-row best (q, hard loss)."""
    fr = obj.fr
    Q = Q0.clone().requires_grad_(True)
    opt = torch.optim.Adam([Q], lr=lr)
    best_l = obj.hard(Q0.detach())
    best_q = Q0.detach().clone()
    n_soft = n_hard = 0
    n_hard += len(Q0)
    per = max(1, iters // len(sigmas))
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)
    gen_noise = torch.Generator(device="cpu").manual_seed(cfg.seed + 2)  # Langevin: keep the reseed draws common
    for it in range(iters):
        sigma = sigmas[min(it // per, len(sigmas) - 1)]
        if cfg.reseed_sigma > 0 and it > 0 and it % per == 0 and it // per < len(sigmas) and len(Q) >= 4:
            with torch.no_grad():
                cur = obj.hard(Q.detach())
                n_hard += len(Q)
                order = torch.argsort(cur)
                h = len(Q) // 2
                good, bad = order[:h], order[h : 2 * h]
                noise = torch.randn(h, Q.shape[1], generator=gen).to(Q.device) * cfg.reseed_sigma
                Q[bad] = fr.clip(Q[good] + noise)
                st = opt.state.get(Q)
                if st:
                    st["exp_avg"][bad] = 0
                    st["exp_avg_sq"][bad] = 0
        frac = it / max(iters - 1, 1)
        for g in opt.param_groups:  # cosine decay
            g["lr"] = lr_final + 0.5 * (lr - lr_final) * (1 + math.cos(math.pi * frac))
        opt.zero_grad(set_to_none=True)
        loss, _ = obj(Q, soft=True, sigma=sigma)
        loss.sum().backward()
        if cfg.svgd_T > 0 and len(Q) > 1:
            with torch.no_grad():
                diff = Q[:, None, :] - Q[None, :, :]  # q_i - q_j
                d2 = (diff ** 2).sum(-1)
                bw = cfg.svgd_bw * d2.median() / math.log(len(Q) + 1) + 1e-12  # median heuristic, scaled
                K = torch.exp(-d2 / bw)
                Q.grad = (K @ Q.grad - cfg.svgd_T * (2.0 / bw) * (K[..., None] * diff).sum(1)) / len(Q)
        opt.step()
        with torch.no_grad():
            if cfg.langevin > 0:
                Q.add_(torch.randn(Q.shape, generator=gen_noise).to(Q.device) * cfg.langevin * (opt.param_groups[0]["lr"] / lr))
            Q.copy_(fr.clip(Q))
        n_soft += len(Q)
        if (it + 1) % cfg.eval_every == 0 or it == iters - 1:
            l = obj.hard(Q.detach())
            n_hard += len(Q)
            better = l < best_l
            best_l = torch.where(better, l, best_l)
            best_q[better] = Q.detach()[better]
            if history is not None:
                history.append((it + 1, float(best_l.min())))
    return best_q, best_l, n_hard, n_soft


class FnObjective(Objective):
    """Objective whose shadow term is any fastshadow.objectives function. During
    the blurred stages the shadow is blurred before scoring, which is what lets a
    silhouette edge feel target pixels several pixels away."""

    def __init__(self, fr, pen, target, cfg, fn, tconst):
        super().__init__(fr, pen, target, cfg)
        self.fn, self.tconst = fn, tconst

    def shadow_terms(self, S, sigma=0.0):
        Sb = gaussian_blur(S, sigma).clamp(0, 1) if sigma > 0 else S
        return self.fn(Sb, self.tconst)


def solve(fr, pen: Penalties, target: np.ndarray, cfg: GradConfig, init_q: np.ndarray | None = None,
          obj: Objective | None = None) -> GradResult:
    """Pure gradient solve: random multi-start + Adam. `init_q` (k, n_dof) adds
    caller-chosen starts (e.g. CMA-ES solutions) ahead of the random ones. `obj`
    replaces the default CMA-ES objective (e.g. an FnObjective)."""
    t0 = time.perf_counter()
    obj = obj or Objective(fr, pen, target, cfg)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    lo, hi = fr._qlo.cpu(), fr._qhi.cpu()
    Qr = (lo + (hi - lo) * torch.rand(cfg.n_init, fr.n_dof, generator=g)).to(fr.device)
    l0 = obj.hard(Qr)
    starts = Qr[torch.argsort(l0)[: cfg.n_restarts]]
    if init_q is not None:
        starts = torch.cat([fr.as_q(init_q), starts])[: max(cfg.n_restarts, len(init_q))]
    hist = []
    bq, bl, nh, ns = adam_refine(obj, starts, cfg, cfg.iters, cfg.blur_sigmas, cfg.lr, cfg.lr_final, hist)
    order = torch.argsort(bl)
    k = int(order[0])
    torch.cuda.synchronize()
    return GradResult(bq[k].cpu().numpy().astype(np.float64), float(bl[k]), time.perf_counter() - t0,
                      cfg.n_init + nh, ns, hist, bq[order[: cfg.top_k]].cpu().numpy().astype(np.float64))


def polish(fr, pen: Penalties, target: np.ndarray, Q0: np.ndarray, cfg: GradConfig, iters=100, lr=0.01):
    """CMA-ES → Adam: refine given solutions at fine blur only. Returns (Q, hard loss)
    per row; a row never gets worse than its start under the hard loss."""
    obj = Objective(fr, pen, target, cfg)
    bq, bl, _, _ = adam_refine(obj, fr.as_q(Q0), cfg, iters, (1.0, 0.0), lr, lr * 0.1)
    return bq.cpu().numpy().astype(np.float64), bl.cpu().numpy()


def pick_by_mujoco(mj, Q: np.ndarray, target: np.ndarray, cfg: GradConfig):
    """Final selection on the reference renderer: the fast renderer disagrees with
    MuJoCo on ~0.5% of pixels, which is the same size as the differences between
    the last few candidates. Costs one MuJoCo render per candidate."""
    vs = [verify(mj, q, target, cfg) for q in Q]
    k = int(np.argmin([v["loss"] for v in vs]))
    return Q[k], vs[k]


def verify(mj, q: np.ndarray, target: np.ndarray, cfg: GradConfig):
    """Re-measure a pose the way the CMA-ES results were measured: MuJoCo render,
    MuJoCo contacts, the optimizer's own loss functions."""
    from loss import compute_iou, shadow_loss
    from optimizer import floor_penalty_loss

    mask, min_z, cross, self_ = mj.eval_q(q)
    loss = shadow_loss(mask, target, cfg.alpha, cfg.beta)
    floor = floor_penalty_loss(min_z, cfg.floor_z, cfg.floor_penalty, cfg.floor_margin, cfg.floor_barrier)
    return {
        "iou": float(compute_iou(mask, target)),
        "loss": float(loss + floor + cfg.collision_penalty * cross + cfg.self_collision_penalty * self_),
        "cross_pen_m": float(cross),
        "self_pen_m": float(self_),
        "min_z": float(min_z),
        "mask": mask,
    }
