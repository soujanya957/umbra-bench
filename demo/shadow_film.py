#!/usr/bin/env python3
"""shadow_film.py -- a directed film of a shadow-art ensemble.

    fleet-shadow python demo/shadow_film.py --ensemble SHADOW_ART_scene_01 \
        --out demo/projects/SHADOW_ART/out/video/SHADOW_ART_reveal_1080p.mp4

Three beats on one clock. The room is dark; every group's lamp comes up; then
the arms lift out of their home pose into the solved pose one group at a time,
left to right, so the word writes itself on the wall. The camera starts tight
on the first group and pulls back to the hero framing while that happens.

The library letter clips hold ONE pose for all their frames, so the movement in
frame is the rise out of the home pose -- authored here, not in the clip.

Renders through the light-linked compositor of projects/SHADOW/tools/
linked_render.py (--relight): MuJoCo's OpenGL renderer draws at most eight
lights, so a ten-element ensemble cannot be filmed natively -- the last groups
stay black. Every exposure constant is locked to one reference frame, or the
auto-exposure breathes as the camera moves.

    --backend native      fast draft, timing and camera only (<=7 groups lit)
    --frames A:B --frame-dir D    render a slice to PNGs (several processes)
    --assemble D          mux a PNG directory into the mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ensemble_mujoco as em  # noqa: E402

# A shot is a camera path plus the clock the lamps and arms run on. The path is
# parameterised by u in [0, 1] -- one eased move from node to node -- so the
# camera never stops and restarts at an interior key the way per-segment easing
# would. Nodes are (u, azimuth, elevation, distance, lookat x, y, z, fovy).
#
# fovy is on the path because distance alone cannot frame one group: the arms
# stand 1.2 m off the wall, so a 45-degree lens far enough back to keep arms and
# their shadow in one frame already sees five letters. The reveals open on a
# long lens and zoom out to the wide as they pull back.
SHOTS = {
    # tight on the first group, pull back to the whole word. The open sits low,
    # on the parked arms; the lookat climbs with the letters as they rise, so
    # the camera tilts up into the word instead of waiting on empty wall.
    "reveal": {
        "duration": 7.8,
        "move": (0.8, 6.2),          # camera moves between these seconds
        "path": [(0.00, -102.0, -3.0, 3.20, 2.66, 0.55, 0.92, 24.0),
                 (0.45, -98.0, -6.0, 4.40, 1.45, 0.50, 1.24, 30.0),
                 (1.00, -94.0, -7.0, 5.60, -0.02, 0.45, 1.24, 42.0)],
        "lamp": (0.20, 1.10),        # lamp ramps up over these seconds
        "arms": (1.50, 0.26, 0.75),  # first group starts, per-group stagger, rise
    },
    # the mirrored move: start on the last group, sweep the other way
    "reveal_right": {
        "duration": 7.8,
        "move": (0.8, 6.2),
        "path": [(0.00, -78.0, -3.0, 3.20, -2.90, 0.55, 0.92, 24.0),
                 (0.45, -82.0, -6.0, 4.40, -1.55, 0.50, 1.24, 30.0),
                 (1.00, -94.0, -7.0, 5.60, -0.02, 0.45, 1.24, 42.0)],
        "lamp": (0.20, 1.10),
        "arms": (1.50, 0.26, 0.75),
    },
    # in on the arms themselves, swinging round to the wall they are drawing on
    "robot": {
        "duration": 9.0,
        "move": (0.8, 7.2),
        "path": [(0.00, -130.0, -2.0, 2.60, 1.70, 0.95, 0.85, 22.0),
                 (0.40, -115.0, -5.0, 3.90, 1.10, 0.70, 1.15, 30.0),
                 (1.00, -94.0, -7.0, 5.60, -0.02, 0.45, 1.24, 42.0)],
        "lamp": (0.20, 1.10),
        "arms": (1.60, 0.28, 0.80),
    },
    # no move: the hero framing, lamps up, the word writes itself
    "static": {
        "duration": 7.0,
        "move": (0.0, 0.1),
        "path": [(0.00, -94.0, -7.0, 5.60, -0.02, 0.45, 1.24, 42.0),
                 (1.00, -94.0, -7.0, 5.60, -0.02, 0.45, 1.24, 42.0)],
        "lamp": (0.20, 1.10),
        "arms": (1.40, 0.26, 0.75),
    },
}


def smooth(x):
    """smoothstep, clamped -- the only easing this film uses."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


FOVY_ASPECT = 16 / 9   # the aspect a path's fovy values are written for


def cam_at(shot, t, cam, model=None, aspect=FOVY_ASPECT):
    """Write the shot's camera at film time t into an MjvCamera.

    The lens lives on the model, not the camera, so `model` is needed for a
    shot whose nodes carry a fovy. MuJoCo's fovy is vertical, but what frames
    this subject is its WIDTH -- a word nearly 7 m across and 1.5 m tall -- so a
    node's fovy is read as the 16:9 framing and converted to hold the same
    horizontal coverage at whatever aspect is being rendered. A scope frame
    (2.35:1) then crops the dead floor and ceiling rather than widening."""
    t0, t1 = shot["move"]
    u = float(smooth((t - t0) / max(t1 - t0, 1e-6)))
    nodes = shot["path"]
    v = list(nodes[-1][1:])
    for na, nb in zip(nodes, nodes[1:]):
        if u <= nb[0] or nb is nodes[-1]:
            f = float(np.clip((u - na[0]) / max(nb[0] - na[0], 1e-6), 0.0, 1.0))
            v = [x + (y - x) * f for x, y in zip(na[1:], nb[1:])]
            break
    cam.azimuth, cam.elevation, cam.distance = v[0], v[1], v[2]
    cam.lookat = v[3:6]
    if model is not None and len(v) > 6:
        half = np.tan(np.radians(v[6]) / 2) * FOVY_ASPECT / aspect
        model.vis.global_.fovy = float(np.degrees(2 * np.arctan(half)))
    return cam


def lamp_at(shot, t):
    """0 -> 1 dimmer on every lamp. A real fixture does not snap on: the ramp
    is eased and overshoots a hair before it settles, which reads as a strike."""
    t0, dur = shot["lamp"]
    x = (t - t0) / max(dur, 1e-6)
    return float(smooth(x) * (1.0 + 0.10 * np.exp(-((x - 1.15) ** 2) / 0.02)))


def build_plans(man, model, clips, rigs, shot):
    """Per entry: the qpos addresses, its clip, and when this group rises.

    Groups rise in screen order -- scene x descending is audience-left first,
    because the scene mirrors the manifest's lateral axis to keep canvas-right
    on the audience's right."""
    t0, step, dur = shot["arms"]
    scene_x = [-e["lat_path_m"][0] for e in man["entries"]]
    order = np.argsort(scene_x)[::-1]
    rise = {int(i): t0 + k * step for k, i in enumerate(order)}
    plans = []
    for i, (e, (arms, slide, lat0)) in enumerate(zip(man["entries"], rigs)):
        clip = clips[e["clip"]]
        robots = sorted(clip["robots"], key=lambda r: r["base"]["x"])
        adrs = [[model.joint(pfx + jn).qposadr[0] for jn in em.JOINT_NAMES]
                for pfx, r in zip(arms, robots)]
        frames = [np.asarray(r["frames"], dtype=float) for r in robots]
        plans.append((e, adrs, frames, clip["n_frames"],
                      model.joint(slide).qposadr[0],
                      model.joint(slide.replace("_slide", "_lift")).qposadr[0],
                      lat0, rise[i], dur))
    return plans


def make_pose_at(model, data, man, plans):
    import mujoco
    fps = man["fps"]
    head, itf = man["choreo_head_pad"], man["interp_factor"]

    def pose_at(t):
        for e, adrs, frames, clip_n, slide_adr, lift_adr, lat0, t0, dur in plans:
            b = float(smooth((t - t0) / max(dur, 1e-6)))
            # the clip's own clock only starts once the group has finished
            # rising; a held-pose library letter does not notice either way
            k = max(0.0, t - (t0 + dur)) * fps
            idx = min(head + int(round(k * itf)), clip_n - 1)
            for aj, fr in zip(adrs, frames):
                for adr, v in zip(aj, fr[idx] * b):
                    data.qpos[adr] = v
            data.qpos[slide_adr] = -em.lat_at(e, t, fps) - lat0
            data.qpos[lift_adr] = em.lift_at(e, t, fps)
        mujoco.mj_forward(model, data)

    return pose_at


class Relight:
    """The still pipeline's light-linked compositor, held open across frames.

    Per frame: which group owns each pixel (depth), a headlight-only fill, and
    one pass per lamp with only that lamp on and only its own group standing --
    MuJoCo has no light linking, and its GL renderer draws at most eight lights
    anyway, so lamps are switched with light_active and every pass carries one.

    Two looks, both from linked_render.py:

    "sum" (the chosen one) adds the passes. Neighbouring pools still fill each
    shadow to mid grey, which is why it reads as cast light rather than pasted
    artwork; its only fault was being too hot, which --exposure handles. It also
    needs two renders per lamp instead of five.

    "relight" additionally renders a plate (same lamp, arms hidden) and a broad
    wash. pass/plate is that group's own shadow transmission, multiplied onto an
    even wash, so no neighbour can fill an umbra -- cleaner, but flatter.

    The balance weights and the output exposure are measured ONCE, on a
    reference frame, and reused -- per-frame auto-exposure would pump."""

    def __init__(self, model, data, entries, W, H, cfg):
        import mujoco
        self.model, self.data, self.entries = model, data, entries
        self.W, self.H, self.cfg = W, H, cfg
        self.ren = mujoco.Renderer(model, H, W)
        self.dep = mujoco.Renderer(model, H, W)
        self.dep.enable_depth_rendering()
        self.hide = mujoco.MjvOption()
        self.hide.geomgroup[2] = self.hide.geomgroup[3] = 0
        self.lamp_d = model.light_diffuse.copy()
        self.lamp_s = model.light_specular.copy()
        self.lamp_exp = model.light_exponent.copy()
        self.lamp_cut = model.light_cutoff.copy()
        self.head = (model.vis.headlight.ambient.copy(),
                     model.vis.headlight.diffuse.copy(),
                     model.vis.headlight.specular.copy())
        self.lifts = [model.joint(f"grp{i}_lift").qposadr[0]
                      for i in range(len(entries))]
        self.w = None        # locked wash balance
        self.ref = None      # locked exposure denominator
        self.right_sign = 1  # screen-right handedness of cross(forward, up)

    # -- raw renders -----------------------------------------------------
    def _shot(self, cam, opt=None):
        import mujoco
        mujoco.mj_forward(self.model, self.data)
        if opt is None:
            self.ren.update_scene(self.data, camera=cam)
        else:
            self.ren.update_scene(self.data, camera=cam, scene_option=opt)
        return self.ren.render().astype(np.float32)

    def _depth(self, cam):
        import mujoco
        mujoco.mj_forward(self.model, self.data)
        self.dep.update_scene(self.data, camera=cam)
        return self.dep.render().astype(np.float32)

    def _gain(self, img):
        peak = float(np.percentile(img.max(axis=2), 99.9))
        return float(np.clip(235.0 / max(peak, 0.5), 1.0, 128.0))

    def _exposed(self, set_gain, cam):
        set_gain(1.0)
        img = self._shot(cam)
        k = self._gain(img)
        if k > 1.05:
            set_gain(k)
            img = self._shot(cam) / k
        return img

    def _set_head(self, g):
        m = self.model
        m.vis.headlight.ambient[:] = self.head[0] * g
        m.vis.headlight.diffuse[:] = self.head[1] * g
        m.vis.headlight.specular[:] = self.head[2] * g

    # -- the pass stack for one frame ------------------------------------
    def passes(self, cam):
        m, d, n = self.model, self.data, len(self.entries)
        z0s = [float(d.qpos[la]) for la in self.lifts]

        def place(only):
            for j, (la, z0) in enumerate(zip(self.lifts, z0s)):
                d.qpos[la] = z0 if j == only else z0 - 50.0

        def restore():
            for la, z0 in zip(self.lifts, z0s):
                d.qpos[la] = z0

        # which group owns each pixel: nearer than the empty room, nearest wins
        place(None)
        room = self._depth(cam)
        near = np.full((self.H, self.W), np.inf, np.float32)
        owner = np.full((self.H, self.W), -1, int)
        for i in range(n):
            place(i)
            dep = self._depth(cam)
            mk = (dep < room - 1e-3) & (dep < near)
            near[mk], owner[mk] = dep[mk], i
        restore()

        # MuJoCo takes the first eight lights whatever their diffuse, so lamps
        # are switched with light_active: every pass carries only its own light
        lamp_on, head_on = m.light_active.copy(), int(m.vis.headlight.active)
        m.light_active[:] = 0
        fill = self._exposed(self._set_head, cam)
        self._set_head(1.0)
        m.vis.headlight.active = 0

        passes, plates, washes = [], [], []
        for i in range(n):
            m.light_active[:] = 0
            m.light_active[i] = 1

            def set_lamp(g, i=i):
                m.light_diffuse[i] = self.lamp_d[i] * g
                m.light_specular[i] = self.lamp_s[i] * g

            if self.cfg["look"] == "sum":
                place(i)
                passes.append(self._exposed(set_lamp, cam))
                set_lamp(1.0)
                continue

            # pass and plate share one gain, so their ratio is the shadow alone
            place(i)
            set_lamp(1.0)
            plate = self._shot(cam, self.hide)
            k = self._gain(plate)
            if k > 1.05:
                set_lamp(k)
                plate = self._shot(cam, self.hide) / k
            passes.append(self._shot(cam) / k if k > 1.05 else self._shot(cam))
            plates.append(plate)
            # the wash: the same lamp, broader and softer, with no arm in it
            m.light_exponent[i] = self.cfg["env_exponent"]
            m.light_cutoff[i] = self.cfg["env_cutoff"]
            set_lamp(1.0)
            wash = self._shot(cam, self.hide)
            kw = self._gain(wash)
            if kw > 1.05:
                set_lamp(kw)
                wash = self._shot(cam, self.hide) / kw
            washes.append(wash)
            m.light_exponent[i], m.light_cutoff[i] = self.lamp_exp[i], self.lamp_cut[i]
            set_lamp(1.0)

        restore()
        m.light_active[:] = lamp_on
        m.vis.headlight.active = head_on
        return dict(room=room, owner=owner, fill=fill,
                    passes=passes, plates=plates, washes=washes)

    # -- composite --------------------------------------------------------
    def _blur_env(self, env, room, owner):
        """Ten broad washes each throw the table edge onto the wall and floor
        at a different place, which sums to stepped panels; smooth the wash
        along x within each surface. The letters live in `trans`, so they stay
        sharp and no light leaks across a surface boundary."""
        import cv2
        gl = self.ren.scene.camera[0]
        pos, fwd, up = (np.asarray(v, np.float64)
                        for v in (gl.pos, gl.forward, gl.up))
        half_w = gl.frustum_width or 0.5 * (self.W / self.H) * (
            gl.frustum_top - gl.frustum_bottom)
        xn = gl.frustum_center + (2.0 * (np.arange(self.W) + 0.5) / self.W - 1.0) * half_w
        yn = gl.frustum_bottom + (1.0 - (np.arange(self.H) + 0.5) / self.H) * (
            gl.frustum_top - gl.frustum_bottom)
        r = np.cross(fwd, up) * self.right_sign
        dvec = fwd * gl.frustum_near + r * xn[None, :, None] + up * yn[:, None, None]
        P = pos + dvec * (room / gl.frustum_near)[..., None]
        roomm = owner < 0
        wall = roomm & (P[..., 1] > -0.08) & (P[..., 1] < 0.02) & (P[..., 2] > 0.02)
        floor = roomm & (P[..., 2] < 0.02) & ~wall
        table = roomm & ~wall & ~floor
        s = self.cfg["env_blur"] * self.W
        for mk in (wall, floor, table):
            if not mk.any():
                continue
            mf = mk.astype(np.float32)
            num = cv2.GaussianBlur(env * mf[..., None], (0, 0), sigmaX=s, sigmaY=0.3)
            den = cv2.GaussianBlur(mf, (0, 0), sigmaX=s, sigmaY=0.3)
            env = np.where(mk[..., None], num / np.maximum(den, 1e-4)[..., None], env)
        return env

    def composite(self, p, dim=1.0, lock=False):
        cfg = self.cfg
        roomm = p["owner"] < 0
        if cfg["look"] == "sum":
            # no balancing: in oblique views it over-boosts the far lamps, and
            # --equal-lamps has already undone build()'s crowd dimming
            self.w = [1.0] * len(p["passes"])
            out = cfg["fill_weight"] * p["fill"] + sum(p["passes"])
            for i in range(len(self.entries)):
                mk = p["owner"] == i
                if mk.any():
                    out[mk] = cfg["fill_weight"] * p["fill"][mk] + p["passes"][i][mk]
            if lock or self.ref is None:
                self.ref = float(np.percentile(out.mean(axis=2)[roomm], 99.5))
            out *= dim * cfg["exposure"] * 235.0 / max(self.ref, 1e-3)
            return np.clip(out, 0, 255).astype(np.uint8)
        if lock or self.w is None:
            lum = [w_.mean(axis=2) for w_ in p["washes"]]
            peaks = [float(np.percentile(l_[roomm], 99.5)) if roomm.any() else 1.0
                     for l_ in lum]
            self.w = [float(np.mean(peaks)) / max(pk, 1e-3) for pk in peaks]
        env = cfg["fill_weight"] * p["fill"] + sum(
            wi * wa for wi, wa in zip(self.w, p["washes"]))
        if cfg["env_blur"] > 0:
            env = self._blur_env(env, p["room"], p["owner"])
        trans = np.ones((self.H, self.W), np.float32)
        for pa, pl in zip(p["passes"], p["plates"]):
            lp, ll = pa.mean(axis=2), pl.mean(axis=2)
            peak = float(np.percentile(ll, 99.5))
            # at the edge of a lamp's cone both images are a few grey levels and
            # their ratio is noise that prints as faint panels: fade the shadow
            # in as the plate brightens
            conf = np.clip((ll / max(peak, 1e-6) - 0.05) / 0.20, 0.0, 1.0)
            conf = conf * conf * (3.0 - 2.0 * conf)
            ratio = np.clip(lp / np.maximum(ll, 1e-6), 0.0, 1.0)
            trans *= (1.0 - cfg["strength"] * conf * (1.0 - ratio)).astype(np.float32)
        out = env * trans[..., None]
        # the wall sums several overlapping washes while an arm sees only its
        # own lamp, so lift the arms back to the wash's level
        for i in range(len(self.entries)):
            mk = p["owner"] == i
            if mk.any():
                out[mk] = (cfg["fill_weight"] * p["fill"][mk]
                           + cfg["arm_gain"] * self.w[i] * p["passes"][i][mk])
        if lock or self.ref is None:
            self.ref = float(np.percentile(out.mean(axis=2)[roomm], 99.5))
        out *= dim * cfg["exposure"] * 235.0 / max(self.ref, 1e-3)
        return np.clip(out, 0, 255).astype(np.uint8)

    def frame(self, cam, dim=1.0, lock=False):
        return self.composite(self.passes(cam), dim, lock)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", default="SHADOW_ART_scene_01")
    ap.add_argument("--shot", default="reveal", choices=sorted(SHOTS))
    ap.add_argument("--out", default=None, help="mp4 to write")
    ap.add_argument("--backend", default="linked", choices=("linked", "native"))
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=None,
                    help="override the shot's length in seconds")
    ap.add_argument("--still", default=None, help="render one frame to PNG and exit")
    ap.add_argument("--at", type=float, default=None, help="--still: film time")
    ap.add_argument("--frames", default=None,
                    help="A:B -- render this slice of frames to --frame-dir")
    ap.add_argument("--frame-dir", default=None)
    ap.add_argument("--assemble", default=None,
                    help="mux this PNG directory into --out and exit")
    ap.add_argument("--look", default="sum", choices=("sum", "relight"),
                    help="sum: the plain light-linked sum the stage stills "
                         "settled on (and two renders per lamp, not five); "
                         "relight: transmission on an even wash")
    ap.add_argument("--equal-lamps", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="undo build()'s crowd dimming, which otherwise leaves "
                         "the end letters' lamps brightest")
    # the still pipeline's knobs, same names and defaults
    ap.add_argument("--strength", type=float, default=0.8)
    ap.add_argument("--env-exponent", type=float, default=4.0)
    ap.add_argument("--env-cutoff", type=float, default=70.0)
    ap.add_argument("--env-blur", type=float, default=0.08)
    ap.add_argument("--arm-gain", type=float, default=2.5)
    ap.add_argument("--fill-weight", type=float, default=1.0)
    ap.add_argument("--exposure", type=float, default=None,
                    help="default 0.75 for --look sum (the stills' value), "
                         "1.0 for --look relight")
    ap.add_argument("--numquads", type=int, default=128)
    ap.add_argument("--wall-h", type=float, default=4.2,
                    help="screen height (m); the still pipeline's 2.8 leaves the "
                         "wall's top edge and the black room above it in frame")
    a = ap.parse_args()

    if a.assemble:
        from PIL import Image
        files = sorted(Path(a.assemble).glob("f*.png"))
        if not files:
            ap.error(f"no frames in {a.assemble}")
        W, H = Image.open(files[0]).size
        with em.video_writer(a.out, W, H, a.fps) as write:
            for f in files:
                write(np.asarray(Image.open(f).convert("RGB")))
        print(f"assembled {len(files)} frames ({len(files) / a.fps:.1f}s, "
              f"{W}x{H}) -> {a.out}")
        return

    W, H = (int(v) for v in a.size.lower().split("x"))
    shot = dict(SHOTS[a.shot])
    if a.duration:
        shot["duration"] = a.duration
    exposure = a.exposure if a.exposure is not None else (
        0.75 if a.look == "sum" else 1.0)
    cfg = dict(look=a.look, strength=a.strength, env_exponent=a.env_exponent,
               env_cutoff=a.env_cutoff, env_blur=a.env_blur, arm_gain=a.arm_gain,
               fill_weight=a.fill_weight, exposure=exposure)

    import mujoco
    em.WALL_H = a.wall_h
    man, clips = em.load(a.ensemble)
    em.LIGHT.update(man.get("mujoco", {}).get("light", {}))
    model, rigs, _ = em.build(man)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    model.vis.quality.numquads = a.numquads
    data = mujoco.MjData(model)
    plans = build_plans(man, model, clips, rigs, shot)
    pose_at = make_pose_at(model, data, man, plans)
    cam = mujoco.MjvCamera()
    total = shot["duration"]

    if a.backend == "native":
        ren = mujoco.Renderer(model, H, W)
        base = model.light_diffuse.copy()

        def render(t):
            pose_at(t)
            model.light_diffuse[:] = base * lamp_at(shot, t)
            ren.update_scene(data, camera=cam_at(shot, t, cam, model, W / H))
            return ren.render()
    else:
        if a.equal_lamps:
            model.light_diffuse[:] = model.light_diffuse.mean(axis=0)
        R = Relight(model, data, man["entries"], W, H, cfg)
        # lock exposure on the hero frame: full light, final framing, every
        # group risen -- the whole film is then one camera setting
        pose_at(total)
        R.frame(cam_at(shot, total, cam, model, W / H), 1.0, lock=True)
        print(f"locked exposure: ref {R.ref:.1f}, weights "
              f"{[round(float(x), 2) for x in R.w]}", flush=True)

        def render(t):
            pose_at(t)
            return R.frame(cam_at(shot, t, cam, model, W / H), lamp_at(shot, t))

    if a.still is not None:
        from PIL import Image
        t = total if a.at is None else a.at
        Image.fromarray(render(t)).save(a.still)
        print(f"still t={t:.2f}s ({W}x{H}) -> {a.still}")
        return

    n = int(round(total * a.fps))
    if a.frames:
        from PIL import Image
        lo, hi = (int(v) for v in a.frames.split(":"))
        d = Path(a.frame_dir or ".")
        d.mkdir(parents=True, exist_ok=True)
        for k in range(lo, min(hi, n)):
            Image.fromarray(render(k / a.fps)).save(d / f"f{k:05d}.png")
        print(f"frames {lo}:{min(hi, n)} -> {d}")
        return

    with em.video_writer(a.out, W, H, a.fps) as write:
        for k in range(n):
            write(render(k / a.fps))
    print(f"filmed {n} frames ({n / a.fps:.1f}s, {W}x{H}, {a.backend}) -> {a.out}")


if __name__ == "__main__":
    main()
