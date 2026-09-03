#!/usr/bin/env python3
"""ensemble_mujoco.py — the whole scene, native MuJoCo, one window.

    fleet-shadow python demo/ensemble_mujoco.py --ensemble pixar_scene_03
    ... --snapshot out.png        # offscreen render instead of a window

Reads an ensemble manifest (demo/make_ensemble.py) plus its choreography
clips and builds ONE MuJoCo scene: per element a rigid rig of three SO-101
arms and its OWN spot light (the owner's design — per-group light, shared
screen), placed at the manifest's solved depth/lateral so every shadow
lands where the footage put the element, at its footage size. Playback is
kinematic (qpos written directly, mj_forward, no dynamics) on the film
clock: before its entry a trio holds the default pose and blends into its
first solved pose; travelling elements glide as a group along lat_path_m.

Frame convention follows the solver rig (renderer.py): x lateral, y depth
with the WALL at the scene's minimum y, z up, arm bases at z=0.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parent.parent
FSA = BENCH.parent / "fleet-shadow-art"
SO101_XML = (FSA / "Shadow_robot_ui" / "assets" / "SO101_urdf"
             / "so101_new_calib.xml")
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
SCREEN_Z = 2.4          # manifest coords: light -1.0 ... arms ... wall 2.4
WALL_H = 2.6


def load(name: str):
    man = json.loads((FSA / "ensembles" / f"{name}.json")
                     .read_text(encoding="utf-8"))
    clips = {}
    for e in man["entries"]:
        c = e["clip"]
        if c not in clips:
            clips[c] = json.loads((FSA / "choreographies" / f"{c}.json")
                                  .read_text(encoding="utf-8"))
    return man, clips


def build(man):
    import mujoco
    spec = mujoco.MjSpec()
    spec.visual.quality.shadowsize = 8192
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.25, 0.25, 0.25]
    spec.visual.map.znear = 0.01
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[12, 12, 0.1], rgba=[0.85, 0.86, 0.88, 1])
    w = max(float(man.get("screen_w_m", 2.21)), 2.6) / 2 + 0.6
    # wall at the minimum y of the scene (solver convention); manifest z ->
    # scene y via y = SCREEN_Z - z, so the wall sits at y = 0
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=[0, -0.03, WALL_H / 2],
                            size=[w, 0.03, WALL_H / 2],
                            rgba=[0.96, 0.96, 0.94, 1])
    rigs = []
    for i, e in enumerate(man["entries"]):
        lat = e["lat_path_m"][0]
        y_mid = SCREEN_Z - e["depth_m"]
        arms = []
        for j, dz in enumerate((0.0, 0.2, 0.4)):
            # manifest arm depths depth_m-0.2 / depth_m / depth_m+0.2 hold
            # SR101 nearest the light; z -> y flips the order, same rig
            y = y_mid + 0.2 - dz
            pfx = f"e{i}r{j}_"
            frame = spec.worldbody.add_frame(pos=[lat, y, 0])
            # attach prefixes the child IN PLACE, so each arm gets a fresh spec
            spec.attach(mujoco.MjSpec.from_file(str(SO101_XML)),
                        prefix=pfx, frame=frame)
            arms.append(pfx)
        ly = y_mid + (-e.get("light", {}).get("ddepth", -1.2))
        lz = e.get("light", {}).get("h", 0.30)
        aim = np.array([0.0, -ly, WALL_H * 0.45 - lz])
        aim /= np.linalg.norm(aim)
        lt = spec.worldbody.add_light(pos=[lat, ly, lz], dir=list(aim))
        lt.castshadow = True
        lt.cutoff = 50          # tight cone: neighbour pools barely overlap,
                                # so each shadow keeps its contrast
        lt.ambient = [0, 0, 0]
        lt.diffuse = [0.9, 0.9, 0.85]
        rigs.append((arms, lat))
    model = spec.compile()
    return model, rigs


def clip_index(t, fps, entry, n, clip_n, head, itf, blend_s):
    """(mode, idx, blend) — the same mapping the web Ensemble view uses."""
    tf = t * fps
    e_t = entry / fps
    if tf < entry:
        if blend_s > 0 and t >= e_t - blend_s:
            first = min(head, clip_n - 1)
            return "blend", first, (t - (e_t - blend_s)) / blend_s
        return "default", 0, 0.0
    k = min(int(tf) - entry, n - 1)
    frac = tf - int(tf)
    return "clip", min(head + k * itf + round(frac * itf), clip_n - 1), 1.0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--snapshot", default=None,
                    help="render frame at --at seconds to a PNG and exit")
    ap.add_argument("--at", type=float, default=1.0)
    ap.add_argument("--cam", default=None,
                    help="az,el,dist,lx,ly,lz override")
    ap.add_argument("--regen", default=None,
                    help="project:scene -- refresh the manifest first")
    a = ap.parse_args()

    if a.regen:
        import subprocess
        import sys as _sys
        proj, sc = a.regen.split(":", 1)
        subprocess.run([_sys.executable,
                        str(Path(__file__).with_name("make_ensemble.py")),
                        "--project", proj, "--scene", sc], check=True)
    import mujoco
    man, clips = load(a.ensemble)
    model, rigs = build(man)
    data = mujoco.MjData(model)
    fps = man["fps"]
    head, itf = man["choreo_head_pad"], man["interp_factor"]
    blend_s = man.get("default_blend_s", 1.0)
    total = man["n_frames"] / fps

    # qpos addresses + per-arm frames, robot order by base.x (SR101 first)
    plans = []
    for e, (arms, _) in zip(man["entries"], rigs):
        clip = clips[e["clip"]]
        robots = sorted(clip["robots"], key=lambda r: r["base"]["x"])
        adrs, frames = [], []
        for pfx, r in zip(arms, robots):
            adrs.append([model.joint(pfx + jn).qposadr[0]
                         for jn in JOINT_NAMES])
            frames.append(np.asarray(r["frames"], dtype=float))
        plans.append((e, adrs, frames, clip["n_frames"]))

    def pose_at(t):
        for (e, adrs, frames, clip_n), (arms, _) in zip(plans, rigs):
            mode, idx, blend = clip_index(t, fps, e["entry"], e["n_frames"],
                                          clip_n, head, itf, blend_s)
            for aj, fr in zip(adrs, frames):
                q = fr[idx] * (blend if mode == "blend" else
                               0.0 if mode == "default" else 1.0)
                for adr, v in zip(aj, q):
                    data.qpos[adr] = v
        mujoco.mj_forward(model, data)

    if a.snapshot:
        pose_at(min(a.at, total - 1e-3))
        ren = mujoco.Renderer(model, height=720, width=1280)
        cam = mujoco.MjvCamera()
        lats = [e["lat_path_m"][0] for e in man["entries"]]
        cam.lookat = [float(np.mean(lats)), 0.6, 0.8]
        cam.distance = 7.0
        cam.azimuth = 90
        cam.elevation = -14
        if a.cam:
            v = [float(x) for x in a.cam.split(",")]
            cam.azimuth, cam.elevation, cam.distance = v[0], v[1], v[2]
            if len(v) == 6:
                cam.lookat = v[3:6]
        ren.update_scene(data, camera=cam)
        from PIL import Image
        Image.fromarray(ren.render()).save(a.snapshot)
        print(f"snapshot t={a.at:.2f}s -> {a.snapshot}")
        return

    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.lookat = [0, 1.0, 0.9]
        v.cam.distance = 6.5
        v.cam.azimuth = 90
        v.cam.elevation = -12
        t0 = time.time()
        while v.is_running():
            t = (time.time() - t0) * a.speed
            if t > total + 1.0:
                if not a.loop:
                    break
                t0, t = time.time(), 0.0
            pose_at(min(t, total - 1e-3))
            v.sync()
            time.sleep(1 / 30)


if __name__ == "__main__":
    main()
