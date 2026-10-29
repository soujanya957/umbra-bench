"""Batched, differentiable shadow renderer on nvdiffrast.

The MuJoCo renderer (fleet-shadow-art renderer.ShadowRenderer) gets a shadow by
rendering the lit scene through `shadow_cam` with an 8192^2 shadow map, reading the
RGB back and thresholding it: one pose per GL round trip. None of that is needed.
The rig is a point light and a flat wall, so the shadow of a triangle is that
triangle projected from the light onto the wall plane, and a central projection is
a linear map in homogeneous coordinates. Every vertex therefore goes straight to
nvdiffrast clip space with

    w   = L_y - P_y                       (depth from the light along -y)
    x_c = -((L_x - C_x) w + (P_x - L_x) s) / h
    y_c =  ((L_z - C_z) w + (P_z - L_z) s) / h

where s = L_y - wall_face_y, C is shadow_cam and h the half extent the camera sees
on the wall face. A whole CMA-ES population is one rasterize call.

MuJoCo is read once at construction (kinematic tree, meshes, light, camera, wall)
so the geometry is the one the original renderer draws; nothing calls into MuJoCo
per pose. Collision and floor penalties are not here: see FastEvaluator.

Masks come back as (B, H, W) float tensors, rows top-down like
mujoco.Renderer.render(). `soft=True` returns nvdiffrast's analytically
antialiased coverage, whose gradient w.r.t. the joint angles flows through the
silhouette edges. That is the gradient path; the hard mask is for scoring.
"""

from __future__ import annotations

import numpy as np
import torch

# mjtGeom
_PLANE, _HFIELD, _SPHERE, _CAPSULE, _ELLIPSOID, _CYLINDER, _BOX, _MESH = range(8)
_HINGE = 3
NOCAST_GROUP = 2  # renderer.NOCAST_GROUP: hidden from the shadow camera in cutout mode


def _quat_to_mat_np(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def cluster_mesh(V: np.ndarray, F: np.ndarray, cell: float):
    """Vertex-clustering decimation: snap vertices to a `cell`-sized grid, merge each
    cell to its centroid, drop the faces that collapse. The outline moves by at most
    about one cell, and at 128 px one pixel is ~3 mm of arm (wall pixel ~8.7 mm over
    2.4-3.4x magnification), so a 1 mm cell is invisible in the mask while cutting
    the 320k-face SO-101 set by an order of magnitude."""
    if cell <= 0:
        return V.astype(np.float32), F.astype(np.int64)
    key = np.floor(V / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    Vn = np.stack([np.bincount(inv, weights=V[:, k], minlength=n) for k in range(3)], 1)
    Vn /= cnt[:, None]
    Fn = inv[F]
    keep = (Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 0] != Fn[:, 2])
    Fn = Fn[keep]
    Fn = np.unique(np.sort(Fn, axis=1), axis=0)  # orientation is irrelevant: no culling
    used = np.unique(Fn)
    remap = -np.ones(n, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return Vn[used].astype(np.float32), remap[Fn]


def _box_mesh(size):
    sx, sy, sz = size[:3]
    V = np.array(
        [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)], dtype=np.float64
    )
    F = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    )
    return V, F


def _uv_sphere(radii, n=16):
    th = np.linspace(0, np.pi, n)
    ph = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    V = np.array([[np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)] for t in th for p in ph])
    V = V * np.asarray(radii)[None, :]
    F = []
    m = len(ph)
    for i in range(n - 1):
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m
            c, d = a + m, b + m
            F += [[a, b, d], [a, d, c]]
    return V, np.array(F)


def _capsule_or_cylinder(r, hl, caps: bool, n=16):
    """A capsule's silhouette is a cylinder's plus two end spheres; their union is
    all the rasterizer needs, so overlapping pieces are fine."""
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([r * np.cos(ph), r * np.sin(ph)], 1)
    V = np.concatenate([np.c_[ring, -hl * np.ones(n)], np.c_[ring, hl * np.ones(n)],
                        [[0, 0, -hl], [0, 0, hl]]])
    F = []
    for j in range(n):
        k = (j + 1) % n
        F += [[j, k, n + k], [j, n + k, n + j], [2 * n, k, j], [2 * n + 1, n + j, n + k]]
    V, F = [V], [np.array(F)]
    if caps:
        for zc in (-hl, hl):
            sv, sf = _uv_sphere([r, r, r], n=max(8, n // 2))
            F.append(sf + sum(len(v) for v in V))
            V.append(sv + [0, 0, zc])
    return np.concatenate(V), np.concatenate(F)


def geom_mesh(model, g: int):
    """Geom-frame triangles for any casting geom type."""
    t = int(model.geom_type[g])
    size = model.geom_size[g]
    if t == _MESH:
        m = int(model.geom_dataid[g])
        v0, nv = model.mesh_vertadr[m], model.mesh_vertnum[m]
        f0, nf = model.mesh_faceadr[m], model.mesh_facenum[m]
        return model.mesh_vert[v0 : v0 + nv].astype(np.float64), model.mesh_face[f0 : f0 + nf].astype(np.int64)
    if t == _BOX:
        return _box_mesh(size)
    if t == _SPHERE:
        return _uv_sphere([size[0]] * 3)
    if t == _ELLIPSOID:
        return _uv_sphere(size[:3])
    if t in (_CAPSULE, _CYLINDER):
        return _capsule_or_cylinder(size[0], size[1], caps=(t == _CAPSULE))
    raise NotImplementedError(f"geom type {t} cannot cast here")


def _rodrigues_basis(axis: np.ndarray):
    """R(θ) = I + sinθ K + (1-cosθ) K², so a hinge needs only K and K² precomputed."""
    x, y, z = axis / np.linalg.norm(axis)
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return K, K @ K


class FastShadowRenderer:
    """Drop-in for the shadow half of ShadowRenderer, batched over poses.

    ref : a renderer.ShadowRenderer built on the same model. Only its DOF ordering
          and joint limits are read, so q vectors mean exactly what they mean to the
          existing optimizer and results.json files.
    """

    def __init__(
        self,
        ref,
        device: str = "cuda",
        cluster_mm: float = 2.0,
        supersample: int = 1,
        coverage_threshold: float = 0.5,
        size: int | None = None,
        fk_device: str | None = "cpu",
        arrangement=None,
    ):
        """fk_device: where forward kinematics runs. FK is ~100 tiny ops on (B, n, 3, 3)
        tensors, so its cost is kernel launches, which the CPU does not pay; the GPU
        gets only R and p. Autograd crosses the copy, so gradients still reach Q."""
        import nvdiffrast.torch as dr

        self.dr = dr
        self.ref = ref
        model = ref.model
        self.model = model
        self.device = torch.device(device)
        self.size = int(size or ref.size)
        self.ss = int(supersample)
        self.cov_thr = float(coverage_threshold)
        self.n_robots = ref.n_robots
        self.n_dof = ref.n_dof
        self.all_dof_ids = ref.all_dof_ids
        self.q_lower = ref.q_lower.copy()
        self.q_upper = ref.q_upper.copy()
        dt = torch.float32
        dev = self.device
        kdev = torch.device(fk_device) if fk_device else dev
        self.kdev = kdev
        self._qlo = torch.tensor(self.q_lower, dtype=dt, device=dev)
        self._qhi = torch.tensor(self.q_upper, dtype=dt, device=dev)

        # ── Kinematic tree ────────────────────────────────────────────────
        nb = model.nbody
        self.parent = [int(model.body_parentid[b]) for b in range(nb)]
        self.body_pos = torch.tensor(model.body_pos, dtype=dt, device=kdev)
        self.body_R = torch.tensor(
            np.stack([_quat_to_mat_np(q) for q in model.body_quat]), dtype=dt, device=kdev
        )
        dof_col = {int(j): i for i, j in enumerate(self.all_dof_ids)}
        self.body_joints: list[list[tuple]] = [[] for _ in range(nb)]
        for j in range(model.njnt):
            b = int(model.jnt_bodyid[j])
            if int(model.jnt_type[j]) != _HINGE:
                raise NotImplementedError(f"joint {j} is not a hinge")
            if j not in dof_col:
                # A hinge the optimizer doesn't drive stays at qpos0, which is what
                # MuJoCo's ShadowRenderer leaves it at.
                col, q0 = None, float(model.qpos0[model.jnt_qposadr[j]])
            else:
                col, q0 = dof_col[j], float(model.qpos0[model.jnt_qposadr[j]])
            K, K2 = _rodrigues_basis(model.jnt_axis[j])
            self.body_joints[b].append(
                (
                    col,
                    q0,
                    torch.tensor(K, dtype=dt, device=kdev),
                    torch.tensor(K2, dtype=dt, device=kdev),
                    torch.tensor(model.jnt_pos[j], dtype=dt, device=kdev),
                )
            )
        self._eye = torch.eye(3, dtype=dt, device=kdev)

        # Level-order FK: a body's parent sits exactly one tree level up, so a whole
        # level (the same link of all N arms, plus whatever else is that deep) is one
        # gather + a few batched matmuls. Kernel launches, not FLOPs, are the cost at
        # these sizes, and on Windows each launch is expensive.
        depth = [0] * nb
        for b in range(1, nb):
            depth[b] = depth[self.parent[b]] + 1
        self._levels = []  # per level: (bodies, parent index within previous level, ...)
        self._level_pos = {0: 0}
        self._fk_fast = all(len(js) <= 1 for js in self.body_joints)
        prev = [0]
        for d in range(1, max(depth) + 1):
            bodies = [b for b in range(nb) if depth[b] == d]
            for i, b in enumerate(bodies):
                self._level_pos[b] = i
            ppos = [prev.index(self.parent[b]) for b in bodies]
            cols, q0s, Ks, K2s, jps = [], [], [], [], []
            for b in bodies:
                if self.body_joints[b]:
                    col, q0, K, K2, jp = self.body_joints[b][0]
                    cols.append(self.n_dof if col is None else col)  # n_dof → zero column
                    q0s.append(q0)
                    Ks.append(K), K2s.append(K2), jps.append(jp)
                else:
                    cols.append(self.n_dof)
                    q0s.append(0.0)
                    z = torch.zeros(3, 3, dtype=dt, device=kdev)
                    Ks.append(z), K2s.append(z), jps.append(torch.zeros(3, dtype=dt, device=kdev))
            self._levels.append(
                dict(
                    bodies=torch.tensor(bodies, device=kdev),
                    ppos=torch.tensor(ppos, device=kdev),
                    bR=self.body_R[bodies],
                    bp=self.body_pos[bodies],
                    cols=torch.tensor(cols, device=kdev),
                    q0=torch.tensor(q0s, dtype=dt, device=kdev),
                    K=torch.stack(Ks),
                    K2=torch.stack(K2s),
                    jp=torch.stack(jps),
                )
            )
            prev = bodies
        ofs = [0, 1]
        for lv in self._levels:
            ofs.append(ofs[-1] + len(lv["bodies"]))
        self._cat_to_body = torch.tensor(
            [ofs[depth[b]] + self._level_pos[b] for b in range(nb)], device=kdev
        )

        # ── Casting geometry ──────────────────────────────────────────────
        # Mirror what mjv_updateScene puts in the shadow camera's scene: default
        # geomgroup visibility [1,1,1,0,0,0], NOCAST_GROUP hidden when cutout mode
        # put anything there, transparent geoms skipped. Wall and floor receive the
        # shadow; they don't cast into the frame.
        import mujoco

        visible = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
        if int((model.geom_group == NOCAST_GROUP).sum()) > 0:
            visible[NOCAST_GROUP] = False
        wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "backwall")
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        verts_by_body: dict[int, list] = {}
        faces, vofs, n_raw = [], 0, 0
        for g in range(model.ngeom):
            if g in (wall_id, floor_id) or int(model.geom_type[g]) in (_PLANE, _HFIELD):
                continue
            grp = int(model.geom_group[g])
            if grp >= 6 or not visible[grp] or model.geom_rgba[g, 3] <= 0:
                continue
            V, F = geom_mesh(model, g)
            n_raw += len(F)
            V, F = cluster_mesh(V, F, cluster_mm * 1e-3)
            if len(F) == 0:
                continue
            Rg = _quat_to_mat_np(model.geom_quat[g])
            Vb = V @ Rg.T + model.geom_pos[g]  # geom frame → body frame
            b = int(model.geom_bodyid[g])
            verts_by_body.setdefault(b, []).append((Vb, F))
        # Vertices grouped by body so FK is one matmul per body, not a gather.
        self._body_blocks = []  # (body, start, count)
        allV = []
        for b in sorted(verts_by_body):
            start = vofs
            for Vb, F in verts_by_body[b]:
                faces.append(F + vofs)
                allV.append(Vb)
                vofs += len(Vb)
            self._body_blocks.append((b, start, vofs - start))
        self.verts_local = torch.tensor(np.concatenate(allV), dtype=dt, device=dev)
        # Padded (blocks, max_n, 3) so all bodies transform in one einsum; `_valid`
        # picks the real rows back out in the order the triangles index.
        maxn = max(n for _, _, n in self._body_blocks)
        pad = torch.zeros(len(self._body_blocks), maxn, 3, dtype=dt, device=dev)
        valid = []
        for k, (b, s0, n) in enumerate(self._body_blocks):
            pad[k, :n] = self.verts_local[s0 : s0 + n]
            valid.append(k * maxn + np.arange(n))
        self._pad_verts = pad
        self._valid = torch.tensor(np.concatenate(valid), device=dev)
        self._block_body = torch.tensor([b for b, _, _ in self._body_blocks], device=dev)
        self.tri = torch.tensor(np.concatenate(faces), dtype=torch.int32, device=dev).contiguous()
        self.n_tri_raw, self.n_tri = n_raw, int(self.tri.shape[0])

        # ── Light / wall / camera → clip-space projection ──────────────────
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "shadow_cam")
        L = model.light_pos[0].astype(np.float64)
        C = model.cam_pos[cam_id].astype(np.float64)
        wall_face_y = float(model.geom_pos[wall_id][1] + model.geom_size[wall_id][1])
        s = L[1] - wall_face_y
        h = (C[1] - wall_face_y) * np.tan(np.deg2rad(model.cam_fovy[cam_id]) / 2.0)
        # clip = P_h @ M.T, P_h = (x, y, z, 1)
        M = np.zeros((4, 4))
        # w = L_y - P_y
        M[3] = [0, -1, 0, L[1]]
        # x_c = -((L_x - C_x) w + (P_x - L_x) s) / h
        M[0] = -(np.array([s, 0, 0, -L[0] * s]) + (L[0] - C[0]) * M[3]) / h
        # y_c = ((L_z - C_z) w + (P_z - L_z) s) / h
        M[1] = (np.array([0, 0, s, -L[2] * s]) + (L[2] - C[2]) * M[3]) / h
        M[2] = 0.0  # every shadow at ndc z = 0: there is no occlusion to resolve
        self._M = torch.tensor(M, dtype=dt, device=dev)
        self.light = L

        self._glctx = dr.RasterizeCudaContext(device=dev)
        self._topo = None  # antialias topology hash, built on first soft render

        # Optional movable bases (fastshadow.spacing): Q gains two columns, the
        # arrangement's spacing in [0,1], which set the robot roots' table-plane
        # position inside FK, so Adam gets a gradient for the spacing too.
        self.n_joint = self.n_dof
        self.arrangement = arrangement
        if arrangement is not None:
            from fastshadow.spacing import root_ids

            lv0 = self._levels[0]["bodies"].tolist()
            self._root_idx = torch.tensor([lv0.index(r) for r in root_ids(model)], device=kdev)
            k = arrangement.n_vars
            self.n_dof = self.n_joint + k
            self.q_lower = np.concatenate([self.q_lower, np.zeros(k)])
            self.q_upper = np.concatenate([self.q_upper, np.ones(k)])
            self._qlo = torch.tensor(self.q_lower, dtype=dt, device=dev)
            self._qhi = torch.tensor(self.q_upper, dtype=dt, device=dev)

    # ── Kinematics ─────────────────────────────────────────────────────────
    def fk(self, Q: torch.Tensor):
        """Q (B, n_dof), already clipped → world (R (B,nb,3,3), p (B,nb,3)), body order."""
        if not self._fk_fast:
            return self._fk_loop(Q)
        B = Q.shape[0]
        nj = self.n_joint
        Qz = torch.cat([Q[:, :nj], Q.new_zeros(B, 1)], 1)  # column n_joint = "no joint"
        Rs = [self._eye.expand(B, 1, 3, 3)]
        ps = [Q.new_zeros(B, 1, 3)]
        for li, lv in enumerate(self._levels):
            Rp = Rs[-1][:, lv["ppos"]]  # (B,n,3,3)
            pp = ps[-1][:, lv["ppos"]]
            Rb = Rp @ lv["bR"]
            pb = pp + (Rp @ lv["bp"][..., None])[..., 0]
            if li == 0 and self.arrangement is not None:  # movable bases: roots hang off the world
                pb = pb.clone()
                pb[:, self._root_idx, :2] = self.arrangement.positions(Q[:, nj:nj + self.arrangement.n_vars])
            a = Qz[:, lv["cols"]] - lv["q0"]  # (B,n)
            Rj = self._eye + torch.sin(a)[..., None, None] * lv["K"] + (1 - torch.cos(a))[..., None, None] * lv["K2"]
            anchor = pb + (Rb @ lv["jp"][..., None])[..., 0]
            Rb = Rb @ Rj
            pb = anchor - (Rb @ lv["jp"][..., None])[..., 0]
            Rs.append(Rb)
            ps.append(pb)
        R = torch.cat(Rs, 1)[:, self._cat_to_body]
        p = torch.cat(ps, 1)[:, self._cat_to_body]
        return R, p

    def _fk_loop(self, Q: torch.Tensor):
        """Body-by-body FK for trees with multi-joint bodies."""
        B = Q.shape[0]
        nb = len(self.parent)
        # Every joint's sin/cos in two kernels instead of two per joint.
        S, Cq = torch.sin(Q), torch.cos(Q)
        R = [self._eye.expand(B, 3, 3)] + [None] * (nb - 1)
        p = [Q.new_zeros(B, 3)] + [None] * (nb - 1)
        for b in range(1, nb):
            pa = self.parent[b]
            Rb = R[pa] @ self.body_R[b]
            pb = p[pa] + (R[pa] @ self.body_pos[b])  # (B,3)
            for col, q0, K, K2, jp in self.body_joints[b]:
                anchor = pb + Rb @ jp
                if col is None:
                    s, c = np.sin(-q0), np.cos(-q0)
                    Rj = self._eye + s * K + (1 - c) * K2
                elif q0 == 0.0:
                    Rj = self._eye + S[:, col, None, None] * K + (1 - Cq[:, col, None, None]) * K2
                else:
                    a = Q[:, col, None, None] - q0
                    Rj = self._eye + torch.sin(a) * K + (1 - torch.cos(a)) * K2
                Rb = Rb @ Rj
                pb = anchor - Rb @ jp
            R[b], p[b] = Rb, pb
        return torch.stack(R, 1), torch.stack(p, 1)

    def pose(self, Q: torch.Tensor):
        """Clipped Q (on self.device) → body (R, p) on self.device."""
        R, p = self.fk(Q.to(self.kdev))
        return R.to(self.device, non_blocking=True), p.to(self.device, non_blocking=True)

    def world_verts(self, Q: torch.Tensor, Rp=None) -> torch.Tensor:
        R, p = Rp if Rp is not None else self.pose(Q)
        Rb, pb = R[:, self._block_body], p[:, self._block_body]  # (B,K,3,3), (B,K,3)
        V = torch.einsum("bkij,knj->bkni", Rb, self._pad_verts) + pb[:, :, None, :]
        return V.reshape(V.shape[0], -1, 3)[:, self._valid]

    def clip(self, q):
        return torch.minimum(torch.maximum(q, self._qlo), self._qhi)

    # ── Rendering ──────────────────────────────────────────────────────────
    def as_q(self, Q) -> torch.Tensor:
        if not torch.is_tensor(Q):
            Q = torch.as_tensor(np.asarray(Q), dtype=torch.float32, device=self.device)
        Q = Q.to(self.device, torch.float32)
        return Q[None] if Q.ndim == 1 else Q

    def render(self, Q, soft: bool = False, chunk: int = 256, Rp=None) -> torch.Tensor:
        """Q: (B, n_dof) array/tensor → (B, H, W) coverage. Hard masks are {0,1};
        soft masks are antialiased coverage in [0,1] carrying gradient to Q.
        Rp: body poses from self.pose(clip(Q)), to share one FK with penalties."""
        Q = self.as_q(Q)
        if Rp is None and Q.shape[0] > chunk:
            return torch.cat([self.render(Q[i : i + chunk], soft, chunk) for i in range(0, Q.shape[0], chunk)])
        Q = self.clip(Q)
        Vw = self.world_verts(Q, Rp)
        Vh = torch.cat([Vw, torch.ones_like(Vw[..., :1])], -1)
        pos = (Vh @ self._M.T).contiguous()
        res = self.size * (1 if soft else self.ss)
        rast, _ = self.dr.rasterize(self._glctx, pos, self.tri, resolution=[res, res])
        cov = (rast[..., 3:4] > 0).float()
        if soft:
            if self._topo is None:
                self._topo = self.dr.antialias_construct_topology_hash(self.tri)
            cov = self.dr.antialias(cov.contiguous(), rast, pos, self.tri, topology_hash=self._topo)
            return cov[..., 0].flip(1)
        cov = cov[..., 0]
        if self.ss > 1:
            cov = torch.nn.functional.avg_pool2d(cov[:, None], self.ss)[:, 0]
            cov = (cov > self.cov_thr).float()
        return cov.flip(1)

    def get_shadow_mask(self, q: np.ndarray) -> np.ndarray:
        """Same contract as ShadowRenderer.get_shadow_mask."""
        return self.render(q)[0].cpu().numpy()

    def min_z(self, Q) -> torch.Tensor:
        """Exact lowest casting-vertex height per pose (B,). Tighter than the MuJoCo
        renderer's geom-centre-minus-capped-radius estimate, so it is not a drop-in
        for its floor penalty; FastEvaluator keeps MuJoCo's for parity."""
        return self.world_verts(self.clip(Q))[..., 2].amin(1)

    def moving_min_z(self, Q) -> torch.Tensor:
        """Lowest casting vertex of the bodies a joint can move, per pose (B,).

        The MuJoCo renderer's min_z (geom centre z minus bounding radius capped at
        3.5 cm, over every arm geom) is pinned at -1.83 cm by the static base, so the
        floor barrier never sees the 1 cm margin or the first 1.8 cm under the table,
        and the cap under-reports the gripper (bounding radius 8.4 cm). This is the
        real lowest point of what can move, so the same barrier sits where meant."""
        if not hasattr(self, "_moving_vert"):
            moving = [False] * len(self.parent)
            for b in range(1, len(self.parent)):
                moving[b] = moving[self.parent[b]] or any(c is not None for c, *_ in self.body_joints[b])
            m = np.zeros(int(self.verts_local.shape[0]), bool)
            for b, s0, n in self._body_blocks:
                m[s0:s0 + n] = moving[b]
            self._moving_vert = torch.tensor(m, device=self.device)
        return self.world_verts(self.clip(self.as_q(Q)))[:, self._moving_vert, 2].amin(1)
