"""Shadow objectives for the glyph study, batched on the GPU.

Each objective maps hard masks S (B,H,W) and a target T to a loss (B,), lower is
better, on the same scale as the current 1 - IoU + 0.5 DT. The search's
accept/keep decisions use quality = 1 - loss (see use_objective), so the whole
staged pipeline chases one objective rather than CMA-ES chasing it while the
stage logic still keeps whatever has the best IoU.

Shape: iou, tversky_fn / tversky_fp, dt2, biou, cldice, holes, tc
Semantic: cls (glyph classifier), clip (ViT-B-16 laion2b, not the benchmark's
          ViT-B-32 grader), each added to 1 - IoU
Combined: shape3 (IoU + boundary + skeleton + holes), shape3_cls
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes, distance_transform_edt

SEM_W = 0.5  # weight of (1 - p) for the semantic terms


def _sum(x):
    return x.sum((-1, -2))


def iou_t(S, T):
    i = _sum(S * T)
    return i / (_sum(S) + _sum(T) - i).clamp_min(1e-6)


def erode(M, r):
    return 1 - F.max_pool2d(1 - M[:, None], 2 * r + 1, 1, r)[:, 0]


def band(M, r):
    return M - erode(M, r)


def soft_skel(M, iters=14):
    """Shit et al. 2021 soft skeleton; exact enough on binary masks."""
    x = M[:, None]

    def er(z):
        return torch.minimum(-F.max_pool2d(-z, (3, 1), 1, (1, 0)), -F.max_pool2d(-z, (1, 3), 1, (0, 1)))

    def dil(z):
        return F.max_pool2d(z, 3, 1, 1)

    skel = F.relu(x - dil(er(x)))
    for _ in range(iters):
        x = er(x)
        d = F.relu(x - dil(er(x)))
        skel = skel + F.relu(d - skel * d)
    return skel[:, 0]


class Target:
    """Per-target constants, built once per solve."""

    def __init__(self, T: np.ndarray, subset: str, label: str, device="cuda"):
        H, W = T.shape
        dev = torch.device(device)
        self.T = torch.tensor(T, dtype=torch.float32, device=dev)
        n = float(np.sqrt(H * W))
        self.dt_out = torch.tensor(distance_transform_edt(T == 0) / n, dtype=torch.float32, device=dev)
        self.dt_in = torch.tensor(distance_transform_edt(T > 0.5) / n, dtype=torch.float32, device=dev)
        holes = binary_fill_holes(T > 0.5) & ~(T > 0.5)
        self.holes = torch.tensor(holes, dtype=torch.float32, device=dev)
        self.n_holes = float(holes.sum())
        self.skel = soft_skel(self.T[None])[0]
        self.band = band(self.T[None], 3)[0]
        ys, xs = np.where(T > 0.5)
        self.bb = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
        self.subset, self.label = subset, label


def _cldice(S, t: Target):
    sk = soft_skel(S)
    tprec = _sum(sk * t.T) / _sum(sk).clamp_min(1e-6)
    tsens = _sum(t.skel * S) / _sum(t.skel).clamp_min(1e-6)
    return 2 * tprec * tsens / (tprec + tsens).clamp_min(1e-6)


def _biou(S, t: Target):
    b = band(S, 3)
    i = _sum(b * t.band)
    return i / (_sum(b) + _sum(t.band) - i).clamp_min(1e-6)


def _holes(S, t: Target):
    if t.n_holes == 0:
        return torch.zeros(S.shape[0], device=S.device)
    return _sum(S * t.holes) / t.n_holes


def _tc_iou(S, t: Target):
    """IoU after moving S's ink box onto T's (uniform scale by height, centred)."""
    B, H, W = S.shape
    dev = S.device
    rows, cols = S.amax(2) > 0.5, S.amax(1) > 0.5
    ar = torch.arange(H, device=dev)
    big = 10**6
    y0 = torch.where(rows, ar, big).amin(1).float()
    y1 = torch.where(rows, ar, -1).amax(1).float() + 1
    x0 = torch.where(cols, ar, big).amin(1).float()
    x1 = torch.where(cols, ar, -1).amax(1).float() + 1
    empty = y1 <= 0
    ty0, ty1, tx0, tx1 = t.bb
    s = ((y1 - y0) / (ty1 - ty0)).clamp_min(1e-3)
    cyS, cxS = (y0 + y1) / 2, (x0 + x1) / 2
    cyT, cxT = (ty0 + ty1) / 2, (tx0 + tx1) / 2
    yy, xx = torch.meshgrid(torch.arange(H, device=dev) + 0.5, torch.arange(W, device=dev) + 0.5, indexing="ij")
    sy = cyS[:, None, None] + (yy - cyT) * s[:, None, None]
    sx = cxS[:, None, None] + (xx - cxT) * s[:, None, None]
    grid = torch.stack([sx / W * 2 - 1, sy / H * 2 - 1], -1)
    if S.requires_grad:  # Adam: soft resample so the gradient reaches S
        Sw = F.grid_sample(S[:, None], grid, mode="bilinear", align_corners=False)[:, 0]
    else:
        Sw = (F.grid_sample(S[:, None], grid, mode="nearest", align_corners=False)[:, 0] > 0.5).float()
    v = iou_t(Sw, t.T[None])
    return torch.where(empty, torch.zeros_like(v), v)


_SEM = {}


def _p_cls(S, t):
    if "cls" not in _SEM:
        from fastshadow.glyph_cls import GlyphClassifier
        _SEM["cls"] = GlyphClassifier()
    return _SEM["cls"].prob(S, t.subset, t.label)


class _Clip:
    """Steering CLIP: ViT-B-16 laion2b. The benchmark grades with ViT-B-32."""

    def __init__(self, dev):
        import open_clip
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
        from clip_eval import glyph_prompt
        from fastshadow.glyph_cls import SUBSET_CLASSES

        self.m, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
        self.m = self.m.to(dev).eval().half()
        tok = open_clip.get_tokenizer("ViT-B-16")
        self.text, self.vocab = {}, SUBSET_CLASSES
        with torch.no_grad():
            for s, cl in SUBSET_CLASSES.items():
                tf = self.m.encode_text(tok([glyph_prompt(c) for c in cl]).to(dev))
                self.text[s] = tf / tf.norm(dim=-1, keepdim=True)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)

    def prob(self, S, subset, label):  # differentiable; scoring callers use no_grad
        # what the grader sees: a dark shadow on a white wall, the whole frame
        img = (1 - S)[:, None].expand(-1, 3, -1, -1)
        img = F.interpolate(img, size=224, mode="bicubic", align_corners=False).clamp(0, 1)
        f = self.m.encode_image(((img - self.mean) / self.std).half())
        f = f / f.norm(dim=-1, keepdim=True)
        logits = 100 * (f @ self.text[subset].T).float()
        return logits.softmax(1)[:, self.vocab[subset].index(label)]


def _p_clip(S, t):
    if "clip" not in _SEM:
        _SEM["clip"] = _Clip(S.device)
    out = []
    for i in range(0, S.shape[0], 64):
        out.append(_SEM["clip"].prob(S[i : i + 64], t.subset, t.label))
    return torch.cat(out)


def _shape3(S, t):
    return 1 - (iou_t(S, t.T) + _biou(S, t) + _cldice(S, t)) / 3 + _holes(S, t)


OBJECTIVES = {
    "iou": lambda S, t: 1 - iou_t(S, t.T),
    "tversky_fn": lambda S, t: 1 - _sum(S * t.T) / (_sum(S * t.T) + 0.3 * _sum(S * (1 - t.T)) + 0.7 * _sum((1 - S) * t.T)).clamp_min(1e-6),
    "tversky_fp": lambda S, t: 1 - _sum(S * t.T) / (_sum(S * t.T) + 0.7 * _sum(S * (1 - t.T)) + 0.3 * _sum((1 - S) * t.T)).clamp_min(1e-6),
    "dt2": lambda S, t: 1 - iou_t(S, t.T) + 0.5 * ((t.dt_out * S).mean((1, 2)) + (t.dt_in * (t.T - S).clamp_min(0)).mean((1, 2))),
    "biou": lambda S, t: 1 - 0.5 * iou_t(S, t.T) - 0.5 * _biou(S, t),
    "cldice": lambda S, t: 1 - 0.5 * iou_t(S, t.T) - 0.5 * _cldice(S, t),
    "holes": lambda S, t: 1 - iou_t(S, t.T) + _holes(S, t),
    "tc": lambda S, t: 1 - _tc_iou(S, t),
    "cls": lambda S, t: 1 - iou_t(S, t.T) + SEM_W * (1 - _p_cls(S, t)),
    "clip": lambda S, t: 1 - iou_t(S, t.T) + SEM_W * (1 - _p_clip(S, t)),
    "shape3": _shape3,
    "shape3_cls": lambda S, t: _shape3(S, t) + SEM_W * (1 - _p_cls(S, t)),
}


class TargetCache:
    """Target constants for whatever target array the optimizer passes. Phase 1 of
    optimize_staged scores each arm against its own zone-blended target, so the
    objective must follow the array it is given, not assume the glyph."""

    def __init__(self, subset, label, device="cuda"):
        self.subset, self.label, self.device = subset, label, device
        self._c = {}

    def __call__(self, T) -> Target:
        T = np.asarray(T)
        k = (id(T), T.shape)
        hit = self._c.get(k)
        if hit is None or hit[0] is not T:
            if len(self._c) > 16:
                self._c.clear()
            hit = (T, Target(T.astype(np.float32), self.subset, self.label, self.device))
            self._c[k] = hit
        return hit[1]


def use_objective(optimizer_module, renderer, name: str | None, targets: TargetCache | None = None):
    """Point the CMA-ES pipeline at one objective for the next solves.

    name None restores the original (1 - IoU + 0.5 DT for CMA-ES, IoU for the stage
    decisions). Otherwise optimizer.shadow_loss (single masks), the batched renderer
    path (renderer.batch_shadow_loss) and optimizer.compute_iou (every accept/keep
    decision in optimize_staged, as 1 - loss) all use the objective.
    """
    import loss as _loss

    if name is None:
        optimizer_module.shadow_loss = _loss.shadow_loss
        optimizer_module.compute_iou = _loss.compute_iou
        renderer.objective = None
        return
    fn = OBJECTIVES[name]
    dev = torch.device(targets.device)

    def single(mask, T, *_a, **_k):
        S = torch.as_tensor(np.asarray(mask), dtype=torch.float32, device=dev)[None]
        with torch.no_grad():
            return float(fn(S, targets(T))[0])

    optimizer_module.shadow_loss = single
    optimizer_module.compute_iou = lambda mask, T, *_a, **_k: 1.0 - single(mask, T)
    renderer.objective = (fn, targets)
