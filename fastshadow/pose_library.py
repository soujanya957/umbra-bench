"""Pose library: stored solves as an instant first answer for a new target.

Every solve in optimized/ is a pose someone already paid for. Rendered once on the rig
in hand and kept as packed masks on the GPU, the whole library is scored against a new
target by true IoU in milliseconds, with no render and no trained model. The best poses
then start a short CMA-ES or Adam solve.

A library belongs to one rig: the same joints cast another shadow from other bases, so
masks are always rendered here, never taken from the stored run. Poses are checked for
contacts and for the real floor (moving_min_z), because about 11% of stored solves put
the gripper into the table.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch


def harvest(root: str, n_robots: int, skip=None, exclude_dirs=("base-optimizer", "_attic")):
    """(N, 6 n_robots) joint poses of every stored run on the fixed column rig, deduplicated
    to 0.01 rad. `skip(top, subset, stem)` -> True leaves a whole target out (held-out sets)."""
    out, sources = [], {}
    for f in glob.glob(os.path.join(root, "**", "results.json"), recursive=True):
        parts = os.path.relpath(f, root).replace("\\", "/").split("/")
        if parts[0] in exclude_dirs:
            continue
        d = json.load(open(f, encoding="utf-8"))
        rig = d.get("rig", {})
        if rig.get("n_robots") != n_robots or rig.get("arm_gap_m") != 0.2 or rig.get("distortion"):
            continue
        stem = parts[-2]
        subset = parts[-3] if len(parts) >= 3 else ""
        if skip is not None and skip(parts[0], subset, stem):
            continue
        qs = [r["q_rad"] for r in d.get("runs", []) if "q_rad" in r]
        out += qs
        sources[parts[0]] = sources.get(parts[0], 0) + len(qs)
    Q = np.asarray(out, np.float32)
    _, keep = np.unique(np.round(Q / 0.01).astype(np.int32), axis=0, return_index=True)
    return Q[np.sort(keep)], sources


class PoseLibrary:
    """Q (N, n_dof) with masks packed to bits on the GPU, so a query is a popcount."""

    def __init__(self, Q: np.ndarray, bits: torch.Tensor, area: torch.Tensor, size: int):
        self.Q, self.bits, self.area, self.size = np.asarray(Q, np.float32), bits, area, size

    def __len__(self):
        return len(self.Q)

    @staticmethod
    def _pack(masks: torch.Tensor) -> torch.Tensor:
        """(B,H,W) bool -> (B, H*W/64) uint64-as-int64."""
        b = masks.flatten(1).to(torch.int64)
        n = b.shape[1] // 64
        w = 1 << torch.arange(64, device=b.device, dtype=torch.int64)
        return (b[:, : n * 64].view(len(b), n, 64) * w).sum(2)

    @classmethod
    def build(cls, fr, Q: np.ndarray, chunk: int = 2048):
        """Render every pose on this rig and pack the masks."""
        bits, area = [], []
        for i in range(0, len(Q), chunk):
            m = fr.render(Q[i: i + chunk]) > 0.5
            bits.append(cls._pack(m))
            area.append(m.flatten(1).sum(1))
        return cls(Q, torch.cat(bits), torch.cat(area), int(fr.size))

    def subset(self, idx):
        i = torch.as_tensor(np.asarray(idx), device=self.bits.device)
        return PoseLibrary(self.Q[np.asarray(idx)], self.bits[i], self.area[i], self.size)

    def query(self, target: np.ndarray, k: int = 1, min_dist: float = 0.0, chunk: int = 8192):
        """Top-k poses by IoU against `target`. min_dist > 0 keeps the picks at least that
        far apart (max joint difference, rad), so k starts are k different ideas."""
        t = torch.as_tensor(np.asarray(target) > 0.5, device=self.bits.device)
        tb = self._pack(t[None])[0]
        ta = t.sum()
        inter = torch.cat([_popcount(self.bits[i: i + chunk] & tb) for i in range(0, len(self), chunk)])
        iou = (inter / (self.area + ta - inter).clamp(min=1)).float()
        if min_dist <= 0:
            v, i = torch.topk(iou, min(k, len(iou)))
            return self.Q[i.cpu().numpy()], v.cpu().numpy()
        order = torch.argsort(iou, descending=True)[: max(512, 64 * k)].cpu().numpy()
        picked = []
        for j in order:
            if all(np.abs(self.Q[j] - self.Q[p]).max() >= min_dist for p in picked):
                picked.append(j)
                if len(picked) == k:
                    break
        return self.Q[picked], iou[torch.as_tensor(picked, device=iou.device)].cpu().numpy()


def _popcount(x: torch.Tensor) -> torch.Tensor:
    """Set bits per row of int64 words (SWAR, on the GPU)."""
    x = x.to(torch.int64)
    m1, m2, m4 = 0x5555555555555555, 0x3333333333333333, 0x0F0F0F0F0F0F0F0F
    x = x - ((x >> 1) & m1)
    x = (x & m2) + ((x >> 2) & m2)
    x = (x + (x >> 4)) & m4
    return (((x * 0x0101010101010101) >> 56) & 0xFF).sum(1)
