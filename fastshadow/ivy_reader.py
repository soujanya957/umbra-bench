"""Ivy reader: a model of one person's judgement of shadow glyphs.

Trained on everything Ivy did in the picker page:
  marks     for each character and optimizer, which of the 13 objectives' shadows
            looked like the target (target visible)          -> P(good)
  guesses   shadow shown alone, which character is it         -> P(read correctly)
  A/B       CMA-ES vs Adam shadow of the same objective,
            "which looks more like it"                         -> P(A preferred over B)

One shared score s(shadow, target) = MLP(standardized features); each task has its
own calibration (marks: σ(a·s + b), guesses: σ(c·s + d), A/B: σ(s_A − s_B)).

Features are computed on the GPU from hard or soft masks by the same code that
serves the objective, so the reader can drive CMA-ES or Adam directly:
  iou, tc_iou, boundary iou, clDice, holes filled, spill (DT outside), gaps (DT
  inside, uncovered), log area ratio, classifier p(target) within the subset,
  classifier margin, cosine of classifier embeddings (shadow vs target)
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

from fastshadow.objectives import _biou, _cldice, _holes, _tc_iou, iou_t

FEATURES = ["iou", "tc_iou", "biou", "cldice", "holes", "spill", "gaps", "log_area", "cls_p", "cls_margin", "emb_cos"]
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "ivy_reader.pt")


class FeatureMaker:
    def __init__(self, clf=None, device="cuda"):
        from fastshadow.glyph_cls import CLASSES, SUBSET_CLASSES, GlyphClassifier

        self.clf = clf or GlyphClassifier(device)
        self.dev = torch.device(device)
        self.idx = {s: torch.tensor([CLASSES.index(c) for c in cl], device=self.dev) for s, cl in SUBSET_CLASSES.items()}
        self.sub_classes = SUBSET_CLASSES
        self._temb = {}

    def _target_emb(self, t):
        k = id(t)
        if k not in self._temb:
            with torch.no_grad():
                _, e = self.clf.logits_embed(t.T[None])
            self._temb[k] = (t, e[0])
        return self._temb[k][1]

    def __call__(self, S: torch.Tensor, t) -> torch.Tensor:
        """S (B,H,W) in [0,1] (hard or soft), t: objectives.Target -> (B, F)."""
        T = t.T
        n = float(np.sqrt(T.numel()))
        iou = iou_t(S, T)
        tc = _tc_iou(S, t)
        bi = _biou(S, t)
        cl = _cldice(S, t)
        ho = _holes(S, t)
        spill = (t.dt_out * S).mean((1, 2)) * n / 10  # ~pixels of spill distance, rescaled
        gaps = (t.dt_in * (T - S).clamp_min(0)).mean((1, 2)) * n / 10
        area = torch.log((S.sum((1, 2)) + 1) / (T.sum() + 1))
        logits, emb = self.clf.logits_embed(S)
        sub = logits[:, self.idx[t.subset]]
        k = self.sub_classes[t.subset].index(t.label)
        p = sub.softmax(1)[:, k]
        other = torch.cat([sub[:, :k], sub[:, k + 1:]], 1).amax(1)
        margin = (sub[:, k] - other).clamp(-10, 10) / 10
        te = self._target_emb(t)
        cos = torch.nn.functional.cosine_similarity(emb, te[None], dim=1)
        return torch.stack([iou, tc, bi, cl, ho, spill, gaps, area, p, margin, cos], 1)


class IvyReader(nn.Module):
    def __init__(self, n_feat=len(FEATURES), hidden=16):
        super().__init__()
        self.register_buffer("mu", torch.zeros(n_feat))
        self.register_buffer("sd", torch.ones(n_feat))
        self.net = nn.Sequential(nn.Linear(n_feat, hidden), nn.Tanh(), nn.Linear(hidden, 1)) if hidden else nn.Linear(n_feat, 1)
        self.mark = nn.Parameter(torch.tensor([1.0, 0.0]))  # scale, bias
        self.guess = nn.Parameter(torch.tensor([1.0, 0.0]))

    def score(self, F):
        return self.net((F - self.mu) / self.sd)[:, 0]

    def p_good(self, F):
        return torch.sigmoid(self.mark[0] * self.score(F) + self.mark[1])


def load_reader(device="cuda", path=WEIGHTS):
    blob = torch.load(path, map_location=device)
    m = IvyReader(hidden=blob["hidden"]).to(device)
    m.load_state_dict(blob["state"])
    m.eval()
    return m


_CACHE = {}


def ivy_loss(S, t):
    """Objective: 1 - P(Ivy marks this shadow good). Differentiable in S."""
    if "fm" not in _CACHE:
        _CACHE["fm"] = FeatureMaker()
        _CACHE["m"] = load_reader()
    return 1 - _CACHE["m"].p_good(_CACHE["fm"](S, t))


def ivy_mix_loss(S, t):
    """Half reader, half plain IoU: the reader steers, IoU keeps the shape honest."""
    return 0.5 * ivy_loss(S, t) + 0.5 * (1 - iou_t(S, t.T))
