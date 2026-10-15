"""A small glyph classifier (A-Z, a-z, 0-9) for scoring and steering shadows.

Trained only on the machine's system fonts, never on the DejaVu faces the
benchmark targets are drawn in, with augmentation for what a shadow does to a
glyph: thickening/thinning, shear, rotation, aspect, stray blobs, bites. The
input is cropped to its ink and centred, so the score is about shape, not where
the shadow landed.

    python -m fastshadow.glyph_cls train      # writes fastshadow/glyph_cls.pt
    python -m fastshadow.glyph_cls check      # accuracy on the benchmark targets
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)] + [str(i) for i in range(10)]
SUBSET_CLASSES = {
    "letters_upper": CLASSES[:26],
    "letters_lower": CLASSES[26:52],
    "digits": CLASSES[52:],
}
SIZE = 32
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "glyph_cls.pt")
_SKIP = ("symbol", "wingding", "webding", "marlett", "segmdl", "holomdl", "emoji", "mtextra", "bssym", "outlook",
         "refspcl", "msreference", "segoe fluent", "segoeicons", "dejavu")


class GlyphNet(nn.Module):
    def __init__(self, n=len(CLASSES)):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128))
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, n))

    def forward(self, x):
        return self.h(self.f(x))


def crop_center(M: torch.Tensor, size: int = SIZE, pad: float = 0.12) -> torch.Tensor:
    """(B,H,W) masks → (B,1,size,size): ink bbox, square, padded, resampled. An
    empty mask stays empty."""
    B, H, W = M.shape
    dev = M.device
    rows, cols = M.amax(2) > 0.5, M.amax(1) > 0.5
    ar_h, ar_w = torch.arange(H, device=dev), torch.arange(W, device=dev)
    big = 10**6
    y0 = torch.where(rows, ar_h, big).amin(1).float()
    y1 = torch.where(rows, ar_h, -1).amax(1).float()
    x0 = torch.where(cols, ar_w, big).amin(1).float()
    x1 = torch.where(cols, ar_w, -1).amax(1).float()
    empty = y1 < 0
    y0, y1, x0, x1 = [torch.where(empty, torch.zeros_like(v), v) for v in (y0, y1, x0, x1)]
    cy, cx = (y0 + y1 + 1) / 2, (x0 + x1 + 1) / 2
    half = torch.maximum(y1 - y0 + 1, x1 - x0 + 1) / 2 * (1 + 2 * pad)
    # affine_grid maps output [-1,1] to input normalised coords
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = half / (W / 2)
    theta[:, 1, 1] = half / (H / 2)
    theta[:, 0, 2] = cx / (W / 2) - 1
    theta[:, 1, 2] = cy / (H / 2) - 1
    grid = F.affine_grid(theta, (B, 1, size, size), align_corners=False)
    out = F.grid_sample(M[:, None].float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return torch.where(empty[:, None, None, None], torch.zeros_like(out), out)


class GlyphClassifier:
    def __init__(self, device="cuda", weights=WEIGHTS):
        self.dev = torch.device(device)
        self.net = GlyphNet().to(self.dev).eval()
        self.net.load_state_dict(torch.load(weights, map_location=self.dev))
        self.idx = {s: torch.tensor([CLASSES.index(c) for c in cl], device=self.dev) for s, cl in SUBSET_CLASSES.items()}

    def prob(self, M: torch.Tensor, subset: str, label: str) -> torch.Tensor:
        """p(label) among the subset's classes, (B,). Differentiable in M (for
        Adam); callers that only score wrap it in torch.no_grad()."""
        logits = self.net(crop_center(M.to(self.dev)))[:, self.idx[subset]]
        return logits.softmax(1)[:, SUBSET_CLASSES[subset].index(label)]

    @torch.no_grad()
    def logits_embed(self, M: torch.Tensor):
        """All-class logits and the 256-d penultimate embedding of the ink crop."""
        h = self.net.f(crop_center(M.to(self.dev)))
        e = self.net.h[3](self.net.h[2](self.net.h[1](self.net.h[0](h))))  # flatten, dropout, linear, relu
        return self.net.h[4](e), e

    @torch.no_grad()
    def rank(self, M: torch.Tensor, subset: str, label: str):
        logits = self.net(crop_center(M.to(self.dev)))[:, self.idx[subset]]
        t = SUBSET_CLASSES[subset].index(label)
        p = logits.softmax(1)
        rank = (logits > logits[:, t : t + 1]).sum(1) + 1
        return rank, p[:, t]


# ── training data ─────────────────────────────────────────────────────────
def _fonts():
    out = []
    for f in sorted(glob.glob(r"C:\Windows\Fonts\*.ttf") + glob.glob(r"C:\Windows\Fonts\*.otf")):
        n = os.path.basename(f).lower()
        if any(s in n for s in _SKIP):
            continue
        out.append(f)
    return out


def render_bank(px=96):
    """One binary bitmap per (font, class) that renders as a real, distinct glyph."""
    from PIL import Image, ImageDraw, ImageFont

    bank, labels = [], []
    for f in _fonts():
        try:
            font = ImageFont.truetype(f, int(px * 0.7))
        except Exception:
            continue
        imgs = []
        for c in CLASSES:
            im = Image.new("L", (px, px), 0)
            d = ImageDraw.Draw(im)
            try:
                bb = d.textbbox((0, 0), c, font=font)
            except Exception:
                imgs = None
                break
            d.text(((px - (bb[2] - bb[0])) / 2 - bb[0], (px - (bb[3] - bb[1])) / 2 - bb[1]), c, fill=255, font=font)
            imgs.append(np.array(im) > 127)
        if imgs is None:
            continue
        a = np.stack(imgs)
        # fonts that draw boxes or identical shapes for letters are not text fonts
        if (a.reshape(len(CLASSES), -1).sum(1) < 20).any():
            continue
        flat = a.reshape(len(CLASSES), -1)
        if len({r.tobytes() for r in flat[:26]}) < 24:
            continue
        bank.append(a)
        labels.append(np.arange(len(CLASSES)))
    return np.concatenate(bank), np.concatenate(labels)


def augment(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """(B,1,H,W) binary glyphs → shadow-like variants."""
    B, _, H, W = x.shape
    dev = x.device
    r = lambda *s: torch.rand(*s, generator=g, device="cpu").to(dev)  # noqa: E731
    ang = (r(B) - 0.5) * np.deg2rad(30)
    sx = 0.75 + 0.5 * r(B)
    sy = 0.75 + 0.5 * r(B)
    sh = (r(B) - 0.5) * 0.5
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0] = torch.cos(ang) / sx
    th[:, 0, 1] = -torch.sin(ang) / sx + sh
    th[:, 1, 0] = torch.sin(ang) / sy
    th[:, 1, 1] = torch.cos(ang) / sy
    th[:, :, 2] = (r(B, 2) - 0.5) * 0.2
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), align_corners=False)
    x = (x > 0.5).float()
    # thicken / thin by up to 5 px (shadows are fatter than fonts, sometimes thinner)
    k = (r(B) * 6).long()
    out = x.clone()
    for kk in range(1, 6):
        sel = k == kk
        if sel.any():
            out[sel] = F.max_pool2d(x[sel], 2 * kk + 1, 1, kk)
    thin = r(B) < 0.2
    if thin.any():
        out[thin] = 1 - F.max_pool2d(1 - out[thin], 3, 1, 1)
    # blobs (stray arm shadow) and bites (missed ink)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    for _ in range(2):
        cy, cx = r(B, 1, 1) * H, r(B, 1, 1) * W
        rad = 3 + r(B, 1, 1) * 9
        blob = ((yy - cy) ** 2 + (xx - cx) ** 2 < rad**2).float()[:, None]
        add = (r(B) < 0.35)[:, None, None, None]
        bite = (r(B) < 0.25)[:, None, None, None]
        out = torch.where(add, torch.maximum(out, blob), out)
        out = torch.where(bite, out * (1 - blob), out)
    return out


def train(epochs=12, bs=512, seed=0):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    X, y = render_bank()
    print(f"bank: {len(X) // len(CLASSES)} fonts × {len(CLASSES)} classes", flush=True)
    X = torch.tensor(X, dtype=torch.float32)[:, None]
    y = torch.tensor(y)
    net = GlyphNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    steps = epochs * (len(X) * 4 // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    step = 0
    for ep in range(epochs):
        net.train()
        tot = corr = 0
        for _ in range(len(X) * 4 // bs):
            i = torch.randint(0, len(X), (bs,), generator=g)
            xb = augment(X[i].to(dev), g)
            xb = crop_center(xb[:, 0])
            logits = net(xb)
            loss = F.cross_entropy(logits, y[i].to(dev), label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            tot += bs
            corr += int((logits.argmax(1).cpu() == y[i]).sum())
        print(f"epoch {ep + 1}: train acc (augmented, 62-way) {corr / tot:.3f}", flush=True)
    torch.save(net.state_dict(), WEIGHTS)
    print("saved", WEIGHTS)


def check():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
    from run_base_optimizer import load_target

    clf = GlyphClassifier()
    bench = os.path.dirname(HERE)
    for sub in SUBSET_CLASSES:
        paths = sorted(glob.glob(os.path.join(bench, "targets_grounded", sub, "*.png")))
        M = torch.tensor(np.stack([load_target(p, 128) for p in paths]))
        labs = [os.path.basename(p).split("_")[0] for p in paths]
        ok = [int(clf.rank(M[k : k + 1], sub, lab)[0]) == 1 for k, lab in enumerate(labs)]
        print(f"{sub}: targets top-1 {np.mean(ok):.3f} ({len(ok)})")
        # stored CMA-ES shadows (best run) for a sense of the shadow domain
        sh = [os.path.join(bench, "optimized", "big-budget-grounded", sub, os.path.basename(p)[:-4],
                           os.path.basename(p)[:-4] + "_best.png") for p in paths]
        sh = [(s, l) for s, l in zip(sh, labs) if os.path.exists(s)]
        if sh:
            M = torch.tensor(np.stack([load_target(s, 128) for s, _ in sh]))
            ok = [int(clf.rank(M[k : k + 1], sub, lab)[0]) == 1 for k, (_, lab) in enumerate(sh)]
            print(f"{sub}: stored CMA-ES shadows top-1 {np.mean(ok):.3f} ({len(ok)})")


if __name__ == "__main__":
    {"train": train, "check": check}[sys.argv[1]]()
