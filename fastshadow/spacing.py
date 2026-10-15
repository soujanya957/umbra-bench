"""Robot arrangements with optimizable spacing (the LeSlider study).

Three SO-101s, light, wall and camera fixed where the default 3-robot rig puts
them (light y = 1.0, wall y = -2.4, the 0.2 m column). Only the bases move, in the
table plane. Each arrangement has two spacing variables, x and y (metres), mapped
from optimizer units u in [0, 1]:  gap = lo + (hi - lo) * u. Working in u keeps a
spacing step the same size as a joint step for CMA-ES's sigma and Adam's lr.

Coordinates: x lateral, y depth (light at +y, wall at -y), z up. Robots are
robot_a, robot_b, robot_c; yaw stays 0 (they all face the wall as before).

  column    a (0, 0)        b (0, -x)          c (0, -x-y)           depth line
  row       a (-x, -0.2)    b (0, -0.2)        c (y, -0.2)           lateral line
  triangle  a (0, 0)        b (-x/2, -y)       c (x/2, -y)           one front, two back
  diagonal  a (0, 0)        b (x, -x)/√2       c ((x+y), -(x+y))/√2  45° line
"""

from __future__ import annotations

import numpy as np
import torch

LO, HI = 0.12, 0.45  # metres; 0.12 keeps the bases apart, MuJoCo still checks contacts
DEFAULT_GAP = 0.20  # the fixed rig the benchmark has used so far
ROOTS = ("robot_a", "robot_b", "robot_c")
R2 = float(np.sqrt(0.5))


class Arrangement:
    def __init__(self, name, label, x_desc, y_desc, lo=(LO, LO), hi=(HI, HI)):
        self.name, self.label, self.x_desc, self.y_desc = name, label, x_desc, y_desc
        self.lo, self.hi = np.array(lo, float), np.array(hi, float)

    def gaps(self, u):
        """u (..., 2) in [0,1] -> metres; works on numpy or torch."""
        lo = torch.as_tensor(self.lo, dtype=u.dtype, device=u.device) if torch.is_tensor(u) else self.lo
        hi = torch.as_tensor(self.hi, dtype=u.dtype, device=u.device) if torch.is_tensor(u) else self.hi
        return lo + (hi - lo) * u

    def u_of(self, gaps):
        return (np.asarray(gaps, float) - self.lo) / (self.hi - self.lo)

    def positions(self, u):
        """u (B, 2) -> base xy (B, 3, 2), numpy or torch."""
        g = self.gaps(u)
        x, y = g[..., 0], g[..., 1]
        z = x * 0
        if self.name == "column":
            pts = [(z, z), (z, -x), (z, -x - y)]
        elif self.name == "row":
            d = z - 0.2
            pts = [(-x, d), (z, d), (y, d)]
        elif self.name == "triangle":
            pts = [(z, z), (-x / 2, -y), (x / 2, -y)]
        elif self.name == "diagonal":
            pts = [(z, z), (x * R2, -x * R2), ((x + y) * R2, -(x + y) * R2)]
        else:
            raise ValueError(self.name)
        if torch.is_tensor(u):
            return torch.stack([torch.stack(p, -1) for p in pts], -2)
        return np.stack([np.stack(p, -1) for p in pts], -2)


ARRANGEMENTS = {
    "column": Arrangement("column", "Column (depth line)", "robot 1 → 2, along depth", "robot 2 → 3, along depth"),
    "row": Arrangement("row", "Row (side by side)", "robot 1 → 2, lateral", "robot 2 → 3, lateral"),
    "triangle": Arrangement("triangle", "Triangle (1 front, 2 back)", "back pair, lateral spacing", "front → back pair, depth"),
    "diagonal": Arrangement("diagonal", "Diagonal (45° line)", "robot 1 → 2, along the diagonal", "robot 2 → 3, along the diagonal"),
}


def root_ids(model):
    import mujoco

    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ROOTS]


class SpacedMJ:
    """MuJoCo ShadowRenderer with two extra 'joints': the arrangement's spacing.

    q = [18 joint angles, u_x, u_y]. Every call writes the base positions into the
    model before kinematics, so renders, floor and contact checks (base-to-base
    included) are MuJoCo's own. Everything else is delegated."""

    def __init__(self, mj, arr: Arrangement, u0=None):
        self._mj, self.arr = mj, arr
        self._roots = root_ids(mj.model)
        self.n_joint = mj.n_dof
        self.n_dof = mj.n_dof + 2
        self.q_lower = np.concatenate([mj.q_lower, [0.0, 0.0]])
        self.q_upper = np.concatenate([mj.q_upper, [1.0, 1.0]])
        u0 = arr.u_of([DEFAULT_GAP, DEFAULT_GAP]) if u0 is None else np.asarray(u0, float)
        self._q = np.concatenate([mj.get_current_q(), np.clip(u0, 0, 1)])

    def __getattr__(self, name):
        return getattr(self._mj, name)

    def _apply(self, q):
        xy = self.arr.positions(np.asarray(q[self.n_joint:], float)[None])[0]
        for rid, p in zip(self._roots, xy):
            self._mj.model.body_pos[rid, :2] = p

    def set_qpos(self, q):
        q = np.clip(np.asarray(q, float), self.q_lower, self.q_upper)
        self._apply(q)
        self._mj.set_qpos(q[: self.n_joint])
        self._q = q.copy()

    def get_shadow_mask(self, q):
        q = np.clip(np.asarray(q, float), self.q_lower, self.q_upper)
        self._apply(q)
        self._q = q.copy()
        return self._mj.get_shadow_mask(q[: self.n_joint])

    def eval_q(self, q):
        q = np.clip(np.asarray(q, float), self.q_lower, self.q_upper)
        self._apply(q)
        self._q = q.copy()
        return self._mj.eval_q(q[: self.n_joint])

    def get_current_q(self):
        return self._q.copy()

    def set_spacing(self, u):
        q = self._q.copy()
        q[self.n_joint:] = np.clip(u, 0, 1)
        self.set_qpos(q)

    def clone(self):
        return self

    def restore_default(self):
        """Put the bases back at the 0.2 m column the model was built with."""
        col = ARRANGEMENTS["column"]
        xy = col.positions(col.u_of([DEFAULT_GAP, DEFAULT_GAP])[None])[0]
        for rid, p in zip(self._roots, xy):
            self._mj.model.body_pos[rid, :2] = p
