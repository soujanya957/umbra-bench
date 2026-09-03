#!/usr/bin/env python3
"""ensemble_mujoco.py — the whole scene, native MuJoCo, one window.

    fleet-shadow python demo/ensemble_mujoco.py --ensemble pixar_scene_03
    ... --snapshot out.png        # offscreen render instead of a window

Reads an ensemble manifest (demo/make_ensemble.py) plus its choreography
clips and builds ONE MuJoCo scene: per element a rigid rig of three SO-101
arms and its OWN vertical strip light (three stacked dim spots — the
owner's design: per-group light, shared screen), the trios standing on a
TABLE so the wall continues below their baseline and shadow bottoms are
not cut by the floor horizon. Each rig hangs on a lateral slide joint, so
travelling elements glide as a group — light included — along the
footage's trajectory. Playback is kinematic (qpos written directly,
mj_forward, no dynamics) on the film clock: before its entry a trio holds
the default pose and blends into its first solved pose.

Handedness: the audience stands on the SAME side as the arms and looks at
the wall (front projection), so canvas-right must be audience-right.
Scene frame: x lateral (mirrored from the manifest to keep that true),
y depth with the wall at y=0 and arms/light at +y, z up.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

LIGHT = {"cutoff": 26, "exponent": 2, "diffuse": 0.8, "fill": 0.08}

BENCH = Path(__file__).resolve().parent.parent
FSA = BENCH.parent / "fleet-shadow-art"
SO101_XML = (FSA / "Shadow_robot_ui" / "assets" / "SO101_urdf"
             / "so101_new_calib.xml")
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
SCREEN_Z = 2.4          # manifest coords: light -1.0 ... arms ... wall 2.4
WALL_H = 2.8
TABLE_H = 0.55          # arms stand on the lab table, wall runs below it
# One shadow light per group. A "strip" of stacked spots reads wrong in
# MuJoCo: each spot maps its own hard shadow (triple ghosting) and fills
# the other two's umbra to 1/3 depth -- grey mush. Softness comes from the
# spot exponent instead.


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
    # dim, even fill: the strip lights carry the picture, the headlight
    # must not wash the wall or the shadows vanish
    f = LIGHT["fill"]
    spec.visual.headlight.ambient = [f, f, f]
    spec.visual.headlight.diffuse = [f * 0.8, f * 0.8, f * 0.8]
    spec.visual.map.znear = 0.01
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[14, 14, 0.1], rgba=[0.55, 0.56, 0.58, 1])
    all_lats = [abs(x) for e in man["entries"] for x in e["lat_path_m"]]
    w = max(float(man.get("screen_w_m", 2.21)) / 2,
            max(all_lats) + 1.5, 1.3) + 0.6
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=[0, -0.03, WALL_H / 2],
                            size=[w, 0.03, WALL_H / 2],
                            rgba=[0.96, 0.96, 0.94, 1])
    # one table under all trios: from a hand's width off the wall out past
    # the deepest rig's light, so downward rays clear its far edge and the
    # wall keeps receiving shadow below the arms' baseline
    y_mids = [SCREEN_Z - e["depth_m"] for e in man["entries"]]
    a_rig = max(-e.get("light", {}).get("ddepth", -1.2)
                for e in man["entries"])
    lats = [-x for e in man["entries"] for x in e["lat_path_m"]]
    tbl_near, tbl_far = 0.12, max(y_mids) + a_rig + 0.4
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[float(np.mean(lats)), (tbl_near + tbl_far) / 2, TABLE_H / 2],
        size=[(max(lats) - min(lats)) / 2 + 0.8,
              (tbl_far - tbl_near) / 2, TABLE_H / 2],
        rgba=[0.28, 0.28, 0.30, 1])

    rigs = []
    for i, e in enumerate(man["entries"]):
        lat0 = -e["lat_path_m"][0]          # audience-side handedness
        y_mid = SCREEN_Z - e["depth_m"]
        grp = spec.worldbody.add_body(name=f"grp{i}",
                                      pos=[lat0, y_mid, TABLE_H])
        grp.add_joint(name=f"grp{i}_slide",
                      type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
        arms = []
        for j, dz in enumerate((0.0, 0.2, 0.4)):
            # SR101 (base.x 0) nearest the light in the manifest frame;
            # z -> y flips depth order, same rig
            frame = grp.add_frame(pos=[0, 0.2 - dz, 0])
            pfx = f"e{i}r{j}_"
            # attach prefixes the child IN PLACE: fresh spec per arm
            spec.attach(mujoco.MjSpec.from_file(str(SO101_XML)),
                        prefix=pfx, frame=frame)
            arms.append(pfx)
        ly = -e.get("light", {}).get("ddepth", -1.2)
        lz = e.get("light", {}).get("h", 0.30)
        # aim where the shadow actually lands: arm centre (~0.25 above base)
        # magnified from the light -- not some fixed fraction of the wall
        mag = float(e.get("mag", 2.0))
        hit = (TABLE_H + lz) + mag * ((TABLE_H + 0.25) - (TABLE_H + lz))
        aim = np.array([0.0, -(y_mid + ly), hit - (TABLE_H + lz)])
        aim /= np.linalg.norm(aim)
        lt = grp.add_light(pos=[0, ly, lz], dir=[float(v) for v in aim])
        lt.castshadow = True
        lt.cutoff = LIGHT["cutoff"]
        lt.exponent = LIGHT["exponent"]
        lt.ambient = [0, 0, 0]
        d = LIGHT["diffuse"]
        lt.diffuse = [d, d, d * 0.96]
        rigs.append((arms, f"grp{i}_slide", lat0))
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


def lat_at(e, t, fps):
    """Group lateral (manifest sign) at film time t, lerped inside frames."""
    lp = e["lat_path_m"]
    tf = t * fps - e["entry"]
    if tf <= 0:
        return lp[0]
    k = min(int(tf), len(lp) - 1)
    k2 = min(k + 1, len(lp) - 1)
    return lp[k] + (lp[k2] - lp[k]) * (tf - int(tf))


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
    ap.add_argument("--light", default=None,
                    help="cutoff,exponent,diffuse,fill override")
    a = ap.parse_args()

    if a.light:
        v = [float(x) for x in a.light.split(",")]
        LIGHT.update(cutoff=v[0], exponent=v[1], diffuse=v[2], fill=v[3])
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

    plans = []
    for e, (arms, slide, lat0) in zip(man["entries"], rigs):
        clip = clips[e["clip"]]
        robots = sorted(clip["robots"], key=lambda r: r["base"]["x"])
        adrs = [[model.joint(pfx + jn).qposadr[0] for jn in JOINT_NAMES]
                for pfx, r in zip(arms, robots)]
        frames = [np.asarray(r["frames"], dtype=float) for r in robots]
        slide_adr = model.joint(slide).qposadr[0]
        plans.append((e, adrs, frames, clip["n_frames"], slide_adr, lat0))

    def pose_at(t):
        for e, adrs, frames, clip_n, slide_adr, lat0 in plans:
            mode, idx, blend = clip_index(t, fps, e["entry"], e["n_frames"],
                                          clip_n, head, itf, blend_s)
            for aj, fr in zip(adrs, frames):
                q = fr[idx] * (blend if mode == "blend" else
                               0.0 if mode == "default" else 1.0)
                for adr, v in zip(aj, q):
                    data.qpos[adr] = v
            # group glide, light riding along (slide axis is scene-x,
            # manifest lateral is mirrored)
            data.qpos[slide_adr] = -lat_at(e, t, fps) - lat0
        mujoco.mj_forward(model, data)

    def default_cam(cam):
        lats = [-e["lat_path_m"][0] for e in man["entries"]]
        cam.lookat = [float(np.mean(lats)), 0.6, 1.0]
        cam.distance = 7.0
        cam.azimuth = -90           # from the arms' side, facing the wall
        cam.elevation = -12
        if a.cam:
            v = [float(x) for x in a.cam.split(",")]
            cam.azimuth, cam.elevation, cam.distance = v[0], v[1], v[2]
            if len(v) == 6:
                cam.lookat = v[3:6]

    if a.snapshot:
        pose_at(min(a.at, total - 1e-3))
        ren = mujoco.Renderer(model, height=720, width=1280)
        cam = mujoco.MjvCamera()
        default_cam(cam)
        ren.update_scene(data, camera=cam)
        from PIL import Image
        Image.fromarray(ren.render()).save(a.snapshot)
        print(f"snapshot t={a.at:.2f}s -> {a.snapshot}")
        return

    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as v:
        default_cam(v.cam)
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
