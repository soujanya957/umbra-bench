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
# Two layers. The judge sees everything and is for evaluation and selection only.
# The objective sees shape alone: every time a classifier-style signal was optimized
# (classifier, CLIP, the full reader) the search learned to please it instead of the
# person, and keeping those features out of the objective leaves the judge a part
# of the evidence the optimizer never pushed on.
SHAPE_FEATURES = ["iou", "tc_iou", "biou", "cldice", "holes", "spill", "gaps", "log_area"]
# Allowed sign of each objective-layer weight (+1 may only help, -1 may only hurt,
# 0 = no weight). Unconstrained, the data gives spill, gaps and area positive weights
# (bold shadows read better), and the optimizer turns that into blobs.
SHAPE_SIGNS = {"iou": 1, "tc_iou": 1, "biou": 1, "cldice": 1, "holes": -1, "spill": -1, "gaps": -1, "log_area": 0}
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "ivy_reader.pt")  # judge (all features)
WEIGHTS_OPT = os.path.join(HERE, "ivy_reader_opt.pt")  # objective (shape features)


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

    def __call__(self, S: torch.Tensor, t, classifier: bool = True) -> torch.Tensor:
        """S (B,H,W) in [0,1] (hard or soft), t: objectives.Target -> (B, F).
        classifier=False leaves the three classifier columns at 0 (the shape-only
        reader never reads them) and skips the network."""
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
        if not classifier:
            z = torch.zeros_like(iou)
            return torch.stack([iou, tc, bi, cl, ho, spill, gaps, area, z, z, z], 1)
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
    """F is always the full FEATURES matrix; `features` picks the columns used."""

    def __init__(self, features=FEATURES, hidden=16):
        super().__init__()
        n_feat = len(features)
        self.features = list(features)
        self.register_buffer("cols", torch.tensor([FEATURES.index(f) for f in features]))
        self.register_buffer("mu", torch.zeros(n_feat))
        self.register_buffer("sd", torch.ones(n_feat))
        # Trust region: features are clamped to the range seen in training, so an
        # optimizer gains nothing by pushing one past where the reader has data.
        self.register_buffer("lo", torch.full((n_feat,), -1e9))
        self.register_buffer("hi", torch.full((n_feat,), 1e9))
        self.net = nn.Sequential(nn.Linear(n_feat, hidden), nn.Tanh(), nn.Linear(hidden, 1)) if hidden else nn.Linear(n_feat, 1)
        self.mark = nn.Parameter(torch.tensor([1.0, 0.0]))  # scale, bias
        self.guess = nn.Parameter(torch.tensor([1.0, 0.0]))

    def score(self, F):
        X = torch.minimum(torch.maximum(F[:, self.cols], self.lo), self.hi)
        return self.net((X - self.mu) / self.sd)[:, 0]

    def p_good(self, F):
        return torch.sigmoid(self.mark[0] * self.score(F) + self.mark[1])


def load_reader(device="cuda", path=WEIGHTS):
    blob = torch.load(path, map_location=device)
    m = IvyReader(features=blob.get("features", FEATURES), hidden=blob["hidden"]).to(device)
    m.load_state_dict(blob["state"])
    m.eval()
    return m


_CACHE = {}


def _reader(path):
    if "fm" not in _CACHE:
        _CACHE["fm"] = FeatureMaker()
    if path not in _CACHE:
        _CACHE[path] = load_reader(path=path)
    return _CACHE["fm"], _CACHE[path]


def ivy_loss(S, t):
    """1 - P(good) under the full reader (the judge). Kept for the record: as an
    objective it is gamed, see the report."""
    fm, m = _reader(WEIGHTS)
    return 1 - m.p_good(fm(S, t))


def ivy_mix_loss(S, t):
    """Half full reader, half plain IoU."""
    return 0.5 * ivy_loss(S, t) + 0.5 * (1 - iou_t(S, t.T))


def ivy_opt_reader_loss(S, t):
    """1 - P(good) under the shape-only reader (linear, features clamped to the
    training range)."""
    fm, m = _reader(WEIGHTS_OPT)
    return 1 - m.p_good(fm(S, t, classifier=False))


def ivy_opt_loss(S, t):
    """The optimization layer: half the shape-only reader, half plain IoU. A shape-
    only MLP reader on its own was driven to IoU 0.26-0.33: the search found inputs
    outside its training range. Linear + clamped features + an IoU anchor."""
    return 0.5 * ivy_opt_reader_loss(S, t) + 0.5 * (1 - iou_t(S, t.T))


@torch.no_grad()
def judge(S, t):
    """The evaluation layer: P(Ivy marks this shadow good), full reader."""
    fm, m = _reader(WEIGHTS)
    return m.p_good(fm(S, t))
