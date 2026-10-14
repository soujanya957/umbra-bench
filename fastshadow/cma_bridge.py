"""Let the existing CMA-ES optimizer render whole populations on nvdiffrast.

optimizer.optimize() (fleet-shadow-art branch `batched-render`) calls
`renderer.eval_q_batch(Q)` when the renderer has it, and scores every row with the
unchanged loss. This wrapper is that renderer: shadows come from FastShadowRenderer
in one GPU call; min_z and contact penetration still come from MuJoCo
(mj_fwdPosition, CPU only, no GL), so the penalties are bit-for-bit the original
ones. Everything else (single-pose renders in Hungarian / ICP / snapshots) is
delegated to the MuJoCo ShadowRenderer.
"""

from __future__ import annotations

import numpy as np


class BatchedShadowRenderer:
    def __init__(self, mj, fast):
        self._mj = mj
        self._fast = fast
        self.n_batched = 0  # poses rendered through eval_q_batch

    def __getattr__(self, name):
        return getattr(self._mj, name)

    def clone(self):
        return self  # only reached with n_workers > 1, which Windows never uses

    def _penalties(self, q):
        mj = self._mj
        mj.set_qpos(q)
        d = mj.data
        min_z = (
            float(np.min(d.geom_xpos[mj.all_arm_geom_ids, 2] - mj.arm_geom_zext))
            if len(mj.all_arm_geom_ids) > 0
            else 0.0
        )
        cross = self_ = 0.0
        g2r = mj._geom_to_robot
        for c in range(d.ncon):
            con = d.contact[c]
            if con.dist >= 0.0:
                continue
            r1, r2 = g2r.get(con.geom1, -1), g2r.get(con.geom2, -1)
            if r1 == -1 or r2 == -1:
                continue
            if r1 != r2:
                cross -= con.dist
            else:
                self_ -= con.dist
        return min_z, cross, self_

    def eval_q_batch(self, Q: np.ndarray):
        """(B, n_dof) → masks (B,H,W) float32, min_z (B,), cross (B,), self (B,).
        Same contract as calling ShadowRenderer.eval_q on each row."""
        Q = np.clip(np.asarray(Q, np.float64), self._mj.q_lower, self._mj.q_upper)
        masks = self._fast.render(Q).cpu().numpy()  # async-free: .cpu() syncs
        pen = np.array([self._penalties(q) for q in Q])
        self.n_batched += len(Q)
        return masks, pen[:, 0], pen[:, 1], pen[:, 2]
