"""Second-order local searches and basin hopping for the gradient solve (grad_opt.solve),
for the solver bench sweep (scripts/grad_sweep.py). Same objective, same random-start
selection, same blur schedule and the same MuJoCo pick at the end as grad_opt.solve; only
the local search differs.

  lbfgs   batched projected L-BFGS on the soft loss: every start keeps its own curvature
          memory, Armijo backtracking (re-rendering only the starts still searching) and
          step cap (reset each blur stage, x2 / x0.5); joints pushing out of the box are
          frozen for the step; memory resets when the blur changes.
  lm      Levenberg-Marquardt on residuals whose sum of squares is the loss term for term
          (IoU block weighted by the union at the current pose, DT spill, floor, one residual
          per collision sphere pair). Jacobian by finite differences (backward at an upper
          joint limit; nvdiffrast has no forward-mode derivative), recomputed only for starts
          that moved; steps accepted on the actual soft loss; Nielsen gain-ratio damping.
  hop     basin hopping around either: perturb every start's best pose, refine it again at
          the fine blur stages without reseeding, keep it where the hard loss improves.
Gradients from the nvdiffrast backward are not bit-reproducible (atomic accumulation), so a
seed fixes the starts, not the result.
Adam with annealed Langevin noise or an SVGD kernel is grad_opt.adam_refine
(GradConfig.langevin / svgd_T).
"""

from __future__ import annotations

import math

import torch

from fastshadow.grad_opt import GradConfig, Objective, adam_refine, gaussian_blur


def random_starts(obj: Objective, cfg: GradConfig) -> torch.Tensor:
    """Exactly grad_opt.solve's start selection: n_init uniform poses, the best n_restarts
    by hard loss (same generator, so every method starts from the same poses)."""
    fr = obj.fr
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    lo, hi = fr._qlo.cpu(), fr._qhi.cpu()
    Qr = (lo + (hi - lo) * torch.rand(cfg.n_init, fr.n_dof, generator=g)).to(fr.device)
    l0 = obj.hard(Qr)
    return Qr[torch.argsort(l0)[: cfg.n_restarts]]


class _Best:
    """Per-start best pose under the hard loss (what grad_opt.adam_refine tracks)."""

    def __init__(self, obj, Q):
        self.obj = obj
        self.l = obj.hard(Q)
        self.q = Q.detach().clone()
        self.n_hard = len(Q)

    def update(self, Q):
        l = self.obj.hard(Q.detach())
        self.n_hard += len(Q)
        b = l < self.l
        self.l = torch.where(b, l, self.l)
        self.q[b] = Q.detach()[b]


def _reseed(obj, Q, cfg, gen):
    """grad_opt.adam_refine's population step at a blur-stage boundary: the worse half of the
    starts (hard loss) becomes the better half plus N(0, reseed_sigma). Returns (Q, rows reseeded)."""
    with torch.no_grad():
        cur = obj.hard(Q.detach())
        order = torch.argsort(cur)
        h = len(Q) // 2
        good, bad = order[:h], order[h : 2 * h]
        Q = Q.detach().clone()
        Q[bad] = obj.fr.clip(Q[good] + torch.randn(h, Q.shape[1], generator=gen).to(Q.device) * cfg.reseed_sigma)
    return Q, bad


def _loss_grad(obj, Q, sigma):
    Qv = Q.detach().clone().requires_grad_(True)
    loss, _ = obj(Qv, soft=True, sigma=sigma)
    (g,) = torch.autograd.grad(loss.sum(), Qv)
    return loss.detach(), g


def lbfgs_refine(obj: Objective, Q0: torch.Tensor, cfg: GradConfig, iters: int, sigmas, m: int = 8,
                 max_step: float = 0.3, first_step: float = 0.05, c1: float = 1e-4, ls_max: int = 5,
                 min_pair: float = 1e-2, history=None):
    """Projected L-BFGS, one memory per start. Joints at a limit whose gradient points out of
    the box are frozen for the step; backtracking renders only the starts still searching;
    the per-start step cap resets on every blur stage and moves x2 / x0.5; a curvature pair is
    kept only if the step is at least `min_pair` rad (smaller ones are mostly gradient noise)."""
    fr = obj.fr
    lo, hi = fr._qlo, fr._qhi
    Q = fr.clip(Q0.detach().clone())
    B, D = Q.shape
    dev = Q.device
    ar = torch.arange(B, device=dev)
    best = _Best(obj, Q)
    n_soft = 0
    per = max(1, iters // len(sigmas))
    Sm, Ym = torch.zeros(m, B, D, device=dev), torch.zeros(m, B, D, device=dev)
    rho = torch.zeros(m, B, device=dev)
    ptr = torch.zeros(B, dtype=torch.long, device=dev)  # next slot per start
    cnt = torch.zeros(B, dtype=torch.long, device=dev)  # valid pairs per start
    gam_last = torch.full((B,), float("nan"), device=dev)
    cap = torch.full((B,), max_step, device=dev)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)
    cur = None
    for it in range(iters):
        sigma = sigmas[min(it // per, len(sigmas) - 1)]
        if sigma != cur:  # the loss changed: old curvature pairs no longer apply
            if cur is not None and cfg.reseed_sigma > 0 and B >= 4:
                Q, _ = _reseed(obj, Q, cfg, gen)
                best.n_hard += B
            cur = sigma
            Sm.zero_(), Ym.zero_(), rho.zero_(), ptr.zero_(), cnt.zero_()
            gam_last.fill_(float("nan"))
            cap.fill_(max_step)
            f, g = _loss_grad(obj, Q, sigma)
            n_soft += B
        act = ((Q <= lo + 1e-7) & (g > 0)) | ((Q >= hi - 1e-7) & (g < 0))
        gf = g.masked_fill(act, 0.0)
        q, al = gf.clone(), []
        for i in range(m):  # newest pair first
            jj = (ptr - 1 - i) % m
            Si, Yi = Sm[jj, ar], Ym[jj, ar]
            ri = rho[jj, ar] * (i < cnt)
            a = ri * (Si * q).sum(1)
            q -= a[:, None] * Yi
            al.append((Si, Yi, ri, a))
        gnorm = gf.norm(dim=1).clamp_min(1e-12)
        gam = torch.where(torch.isfinite(gam_last), gam_last, first_step / gnorm)
        r = gam[:, None] * q
        for Si, Yi, ri, a in reversed(al):
            r += Si * (a - ri * (Yi * r).sum(1))[:, None]
        d = (-r).masked_fill(act, 0.0)
        bad = (gf * d).sum(1) >= 0
        if bad.any():  # not a descent direction: steepest descent
            d[bad] = -gf[bad] * (first_step / gnorm[bad])[:, None]
        d = d * torch.clamp(cap / d.norm(dim=1).clamp_min(1e-12), max=1.0)[:, None]
        t = torch.ones(B, device=dev)
        acc = torch.zeros(B, dtype=torch.bool, device=dev)
        Qn, fn, gn = Q.clone(), f.clone(), g.clone()
        for _ in range(ls_max):
            idx = (~acc).nonzero()[:, 0]
            if len(idx) == 0:
                break
            Qt = fr.clip(Q[idx] + t[idx, None] * d[idx])
            ft, gt = _loss_grad(obj, Qt, sigma)
            n_soft += len(idx)
            ok = ft <= f[idx] + c1 * (g[idx] * (Qt - Q[idx])).sum(1)  # Armijo on the projected step
            sel = idx[ok]
            Qn[sel], fn[sel], gn[sel] = Qt[ok], ft[ok], gt[ok]
            acc[sel] = True
            t[idx[~ok]] *= 0.5
        cap = torch.where(acc, (cap * 2).clamp_max(max_step), (cap * 0.5).clamp_min(1e-3))
        s, y = Qn - Q, gn - g
        sy, sn, yn = (s * y).sum(1), s.norm(dim=1), y.norm(dim=1)
        keep = acc & (sy > 1e-8 * sn * yn) & (sn >= min_pair)
        gi = keep.nonzero()[:, 0]
        if len(gi):
            slot = ptr[gi]
            Sm[slot, gi], Ym[slot, gi], rho[slot, gi] = s[gi], y[gi], 1.0 / sy[gi]
            ptr[gi] = (ptr[gi] + 1) % m
            cnt[gi] = (cnt[gi] + 1).clamp_max(m)
            gam_last[gi] = sy[gi] / (yn[gi] ** 2).clamp_min(1e-20)
        Q, f, g = Qn, fn, gn
        if (it + 1) % cfg.eval_every == 0 or it == iters - 1:
            best.update(Q)
            if history is not None:
                history.append((it + 1, float(best.l.min())))
    return best.q, best.l, best.n_hard, n_soft


def _lm_eval(obj: Objective, Q, sigma, w1=None, chunk=512):
    """Residuals r (B, M), the actual soft loss (B,) and the blurred union (B,).

    r = [ w1 (blur(S) - blur(T)),  sqrt(beta DT / N) S,  sqrt(floor),  sqrt(w_pair pen_pair) ]
    With w1 = sqrt(alpha / union) (the union at the current pose, held fixed while it is the
    current pose), the first block's sum of squares equals alpha (1 - IoU) for binary masks; the
    rest sum to the DT term and the penalties exactly (grad_opt.Objective, term for term)."""
    fr, cfg, pen = obj.fr, obj.cfg, obj.pen
    Tb = obj.T_blur(sigma)
    tsum = Tb.sum()
    w2 = torch.sqrt(cfg.beta * obj.DT / obj.DT.numel())
    wc = torch.where(pen.pcross, torch.tensor(cfg.collision_penalty, device=Q.device),
                     torch.tensor(cfg.self_collision_penalty, device=Q.device))
    R, Ls, Us = [], [], []
    for i in range(0, len(Q), chunk):
        Qc = fr.clip(Q[i : i + chunk])
        Rp = fr.pose(Qc)
        S = fr.render(Qc, soft=True, Rp=Rp)
        Sb = gaussian_blur(S, sigma)
        inter = (Sb * Tb).sum((1, 2))
        union = Sb.sum((1, 2)) + tsum - inter
        mz = pen.min_z(Rp)
        floor = cfg.floor_penalty * torch.relu(cfg.floor_z + cfg.floor_margin - mz) \
            + cfg.floor_penalty * cfg.floor_barrier * torch.relu(cfg.floor_z - mz) ** 2
        W = pen._sphere_world(Rp)
        pp = torch.relu(pen.prsum - (W[:, pen.pi] - W[:, pen.pj]).norm(dim=-1))
        Ls.append(cfg.alpha * (1 - inter / (union + 1e-6)) + cfg.beta * (obj.DT * S).mean((1, 2)) + floor + (wc * pp).sum(1))
        Us.append(union)
        wi = (torch.sqrt(cfg.alpha / (union + 1e-6)) if w1 is None else w1[i : i + chunk])[:, None]
        R.append(torch.cat([(Sb - Tb).flatten(1) * wi, (S * w2).flatten(1),
                            torch.sqrt(floor + 1e-12)[:, None], torch.sqrt(wc * pp + 1e-12)], 1))
    return torch.cat(R), torch.cat(Ls), torch.cat(Us)


@torch.no_grad()
def lm_refine(obj: Objective, Q0: torch.Tensor, cfg: GradConfig, iters: int, sigmas, h: float = 4e-3,
              h_fine: float = 1e-2, lam0: float = 1e-2, max_step: float = 0.3, history=None):
    """Levenberg-Marquardt with a forward-difference Jacobian (backward at an upper joint limit,
    so a joint there can come back), recomputed only for starts that moved; steps accepted on
    the actual soft loss; damping by Nielsen's gain-ratio rule, reset on every blur stage."""
    fr = obj.fr
    hi = fr._qhi
    Q = fr.clip(Q0.detach().clone())
    B, D = Q.shape
    dev = Q.device
    best = _Best(obj, Q)
    n_soft = 0
    lam = torch.full((B,), lam0, device=dev)
    nu = torch.full((B,), 2.0, device=dev)
    per = max(1, iters // len(sigmas))
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)
    n_img = obj.T.numel()
    cur = None
    for it in range(iters):
        sigma = sigmas[min(it // per, len(sigmas) - 1)]
        if sigma != cur:
            if cur is not None and cfg.reseed_sigma > 0 and B >= 4:
                Q, _ = _reseed(obj, Q, cfg, gen)
                best.n_hard += B
            cur = sigma
            lam.fill_(lam0)
            nu.fill_(2.0)
            r, L, U = _lm_eval(obj, Q, sigma)
            w1 = torch.sqrt(obj.cfg.alpha / (U + 1e-6))
            n_soft += B
            A = torch.zeros(B, D, D, device=dev)
            gv = torch.zeros(B, D, device=dev)
            fresh = torch.ones(B, dtype=torch.bool, device=dev)
        hs = h_fine if sigma == 0 else h
        fi = fresh.nonzero()[:, 0]
        if len(fi):
            Qf = Q[fi]
            step = torch.where(Qf + hs <= hi, torch.full_like(Qf, hs), torch.full_like(Qf, -hs))
            P = fr.clip(Qf[:, None, :] + torch.diag_embed(step))
            hh = torch.diagonal(P - Qf[:, None, :], dim1=1, dim2=2)  # actual step per column
            J, _, _ = _lm_eval(obj, P.reshape(-1, D), sigma, w1[fi].repeat_interleave(D))
            n_soft += len(fi) * D
            J = J.view(len(fi), D, -1)
            J.sub_(r[fi][:, None, :])
            J.div_(torch.where(hh.abs() > 0, hh, torch.ones_like(hh))[..., None])
            J.mul_((hh.abs() > 0)[..., None])
            A[fi] = J @ J.transpose(1, 2)
            gv[fi] = (J @ r[fi][:, :, None])[..., 0]
            del J
        dg = torch.diagonal(A, dim1=1, dim2=2)
        damp = dg + 1e-6 * dg.mean(1, keepdim=True) + 1e-12
        delta = -torch.linalg.solve(A + lam[:, None, None] * torch.diag_embed(damp), gv[..., None])[..., 0]
        delta = delta * torch.clamp(max_step / delta.norm(dim=1).clamp_min(1e-12), max=1.0)[:, None]
        Qt = fr.clip(Q + delta)
        rt, Lt, Ut = _lm_eval(obj, Qt, sigma, w1)
        n_soft += B
        ok = Lt < L
        c, ct = 0.5 * (r * r).sum(1), 0.5 * (rt * rt).sum(1)
        pred = -((gv * delta).sum(1) + 0.5 * (delta[:, None, :] @ A @ delta[..., None])[:, 0, 0])
        gain = (c - ct) / pred.clamp_min(1e-12)
        lam = torch.where(ok, lam * torch.clamp(1 - (2 * gain - 1) ** 3, min=1 / 3), lam * nu).clamp(1e-7, 1e7)
        nu = torch.where(ok, torch.full_like(nu, 2.0), nu * 2)
        w1n = torch.sqrt(obj.cfg.alpha / (Ut + 1e-6))
        rt[:, :n_img] *= (w1n / w1)[:, None]  # re-weight the IoU block at the new pose
        Q = torch.where(ok[:, None], Qt, Q)
        r = torch.where(ok[:, None], rt, r)
        L = torch.where(ok, Lt, L)
        w1 = torch.where(ok, w1n, w1)
        fresh = ok
        if (it + 1) % cfg.eval_every == 0 or it == iters - 1:
            best.update(Q)
            if history is not None:
                history.append((it + 1, float(best.l.min())))
    return best.q, best.l, best.n_hard, n_soft


def refine(local: str, obj, Q0, cfg, iters, sigmas, **kw):
    if local == "adam":
        extra = set(kw) - {"lr", "lr_final"}
        if extra:
            raise ValueError(f"adam_refine takes no {sorted(extra)}")
        return adam_refine(obj, Q0, cfg, iters, sigmas, kw.get("lr", cfg.lr), kw.get("lr_final", cfg.lr_final))
    if local == "lbfgs":
        return lbfgs_refine(obj, Q0, cfg, iters, sigmas, **kw)
    if local == "lm":
        return lm_refine(obj, Q0, cfg, iters, sigmas, **kw)
    raise ValueError(local)


def basin_hop(local: str, obj, Q0, cfg, iters, sigmas, rounds: int, hop_iters: int, hop_sigmas=(1.5, 0.0),
              hop_scale: float = 0.15, **kw):
    """Full refine, then `rounds` times: perturb each start's best pose (N(0, hop_scale) rad),
    refine it at the fine blur stages without reseeding (a reseed would hand row i a copy of
    row j and shrink the population), keep it where the hard loss improves."""
    import dataclasses

    bq, bl, nh, ns = refine(local, obj, Q0, cfg, iters, sigmas, **kw)
    hcfg = dataclasses.replace(cfg, reseed_sigma=0.0)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed + 7)
    for _ in range(rounds):
        P = obj.fr.clip(bq + hop_scale * torch.randn(bq.shape, generator=g).to(bq.device))
        q2, l2, nh2, ns2 = refine(local, obj, P, hcfg, hop_iters, hop_sigmas, **kw)
        b = l2 < bl
        bq, bl = torch.where(b[:, None], q2, bq), torch.where(b, l2, bl)
        nh, ns = nh + nh2, ns + ns2
    return bq, bl, nh, ns
