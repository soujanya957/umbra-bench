"""Light-linked render of an ensemble: every lamp lights only its own trio.

MuJoCo has no light linking, so it is emulated with passes:
  - depth: the empty room, then each trio alone -> which trio (or the room) each pixel sees
  - fill: headlight only, all trios present
  - lamp i: only lamp i on, headlight off, every other trio lifted far below the floor
Room pixels (wall/table/floor) = fill + sum of all lamp passes, so each letter's
shadow comes only from its own lamp and the neighbour ghosts are gone. Arm pixels
of trio i = fill + lamp pass i.

Summing 8-bit passes bands badly and clips, so every pass is exposed on its own:
render once, find its peak, render again with the light scaled so the peak sits
near 235, and divide back. Sums are then taken in float, optionally with each
lamp's pool balanced to the same peak, and one exposure maps the result to 8 bits.
Box faces are lit per vertex, so the wall is subdivided (--numquads) to remove
the diagonal streaks a 4-quad wall shows under spot lights.

--relight separates the shadows from the light instead. For each lamp, a pass with
its own trio and a plate with no trio are rendered at the same gain, so
pass / plate is that letter's shadow transmission, shaped by its own lamp. The
room is then lit by one even wash (each lamp's plate re-rendered with a broader,
softer spot and balanced to the same peak), and every transmission multiplies
it at --strength. No lamp outshines another and no window cuts the light.

  <fleet-shadow python> linked_render.py --ensemble SHADOW_scene_01 --out x.png
        (--cam=az,el,dist,lx,ly,lz | --view front|left|right|side) [--size 3840x2160]
        [--exponent E] [--cutoff C] [--balance] [--numquads 128] [--exposure 1.0]
        [--window --feather F] [--relight --strength 0.8 --env-exponent 4 --env-cutoff 70]
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

REPO = Path(r"C:/Users/hexia/Documents/GitHub/umbra-bench")
sys.path.insert(0, str(REPO / "demo"))
import ensemble_mujoco as em  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", default=None)
    ap.add_argument("--view", default="front", choices=sorted(em.VIEWS))
    ap.add_argument("--size", default="3840x2160")
    ap.add_argument("--exponent", type=float, default=None, help="override every lamp's spot exponent")
    ap.add_argument("--cutoff", type=float, default=None, help="override every lamp's cutoff (>=35)")
    ap.add_argument("--balance", action="store_true", help="scale lamps so every pool peaks equally")
    ap.add_argument("--fill-weight", type=float, default=1.0, help="scale the headlight fill in the sum")
    ap.add_argument("--numquads", type=int, default=128)
    ap.add_argument("--exposure", type=float, default=1.0, help="extra exposure after auto (1 = room p99.5 -> 235)")
    ap.add_argument("--window", action="store_true",
                    help="virtual barn doors: weight each lamp's room light by a smooth lateral window "
                         "around its own group, so neighbour pools cannot fill a letter's shadow")
    ap.add_argument("--feather", type=float, default=0.35, help="window feather, as a fraction of the gap between groups")
    ap.add_argument("--relight", action="store_true",
                    help="shadow transmission from each lamp's own pass/plate, multiplied onto one even wash")
    ap.add_argument("--strength", type=float, default=0.8, help="--relight: how much light a shadow removes (1 = black)")
    ap.add_argument("--env-exponent", type=float, default=4.0, help="--relight: spot exponent of the wash plates")
    ap.add_argument("--env-cutoff", type=float, default=70.0, help="--relight: spot cutoff of the wash plates")
    ap.add_argument("--arm-gain", type=float, default=2.5, help="--relight: lift the arms to sit in the wash")
    a = ap.parse_args()
    W, H = (int(v) for v in a.size.lower().split("x"))

    man, clips = em.load(a.ensemble)
    em.LIGHT.update(man.get("mujoco", {}).get("light", {}))
    model, rigs, _ = em.build(man)
    entries = man["entries"]
    assert model.nlight == len(entries), (model.nlight, len(entries))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    model.vis.quality.numquads = a.numquads
    if a.exponent is not None:
        model.light_exponent[:] = a.exponent
    if a.cutoff is not None:
        model.light_cutoff[:] = a.cutoff
    data = mujoco.MjData(model)

    # static pose: frame 0 of each clip, rig at its first lateral/lift
    lifts = []
    for e, (arms, slide, lat0) in zip(entries, rigs):
        robots = sorted(clips[e["clip"]]["robots"], key=lambda r: r["base"]["x"])
        for pfx, r in zip(arms, robots):
            for jn, v in zip(em.JOINT_NAMES, r["frames"][0]):
                data.qpos[model.joint(pfx + jn).qposadr[0]] = v
        data.qpos[model.joint(slide).qposadr[0]] = -e["lat_path_m"][0] - lat0
        la = model.joint(slide.replace("_slide", "_lift")).qposadr[0]
        data.qpos[la] = float((e.get("lift_path_m") or [0.0])[0])
        lifts.append((la, float(data.qpos[la])))

    cam = mujoco.MjvCamera()
    lats = [-e["lat_path_m"][0] for e in entries]
    if a.cam:
        v = [float(x) for x in a.cam.split(",")]
        cam.azimuth, cam.elevation, cam.distance = v[0], v[1], v[2]
        cam.lookat = v[3:6] if len(v) == 6 else [float(np.mean(lats)), 0.6, 1.0]
    else:                                   # the same presets as ensemble_mujoco.default_cam
        if a.view == "front":
            cam.lookat, cam.distance = [float(np.mean(lats)), 0.6, 1.0], 7.0
        else:
            y_rig = float(np.mean([em.SCREEN_Z - e["depth_m"] for e in entries]))
            if len(entries) == 1:
                cam.lookat, cam.distance = [float(np.mean(lats)), 0.5 * y_rig, 0.9], 4.0
            else:
                cam.lookat = [float(np.mean(lats)), 0.6 * y_rig, 1.0]
                cam.distance = max(5.5, 1.1 * (max(lats) - min(lats)) + 4.5)
        cam.azimuth, cam.elevation = em.VIEWS[a.view]

    ren = mujoco.Renderer(model, H, W)
    dep = mujoco.Renderer(model, H, W)
    dep.enable_depth_rendering()

    def place(only):
        """Lift every trio except `only` far below the floor (None: all of them)."""
        for j, (la, z0) in enumerate(lifts):
            data.qpos[la] = z0 if j == only else z0 - 50.0

    def restore():
        for la, z0 in lifts:
            data.qpos[la] = z0

    def depth():
        mujoco.mj_forward(model, data)
        dep.update_scene(data, camera=cam)
        return dep.render().astype(np.float32)

    def shot():
        mujoco.mj_forward(model, data)
        ren.update_scene(data, camera=cam)
        return ren.render().astype(np.float32)

    def gain_for(img):
        peak = float(np.percentile(img.max(axis=2), 99.9))
        return float(np.clip(235.0 / max(peak, 0.5), 1.0, 128.0))

    def exposed(set_gain):
        """Float image in gain-1 units, rendered with its own peak lifted near 235."""
        set_gain(1.0)
        img = shot()
        k = gain_for(img)
        if k > 1.05:
            set_gain(k)
            img = shot() / k
        return img

    # which trio each pixel sees, from depth: a trio owns a pixel when it is
    # nearer than the empty room, and the nearest such trio wins
    place(None)
    room = depth()
    near = np.full((H, W), np.inf, np.float32)
    owner = np.full((H, W), -1, int)
    for i in range(len(entries)):
        place(i)
        d = depth()
        m = (d < room - 1e-3) & (d < near)
        near[m], owner[m] = d[m], i
    restore()

    # MuJoCo's OpenGL renderer draws at most 8 lights (headlight included) and
    # takes them in order whether or not their diffuse is zero, so a lamp past
    # the eighth never lights its pass. Switch lamps with light_active instead,
    # so every pass carries only the one light it needs.
    lamp_d, lamp_s = model.light_diffuse.copy(), model.light_specular.copy()
    lamp_exp, lamp_cut = model.light_exponent.copy(), model.light_cutoff.copy()
    lamp_on, head_on = model.light_active.copy(), int(model.vis.headlight.active)
    head = (model.vis.headlight.ambient.copy(), model.vis.headlight.diffuse.copy(),
            model.vis.headlight.specular.copy())

    def set_head(g):
        model.vis.headlight.ambient[:] = head[0] * g
        model.vis.headlight.diffuse[:] = head[1] * g
        model.vis.headlight.specular[:] = head[2] * g

    model.light_active[:] = 0
    fill = exposed(set_head)                                    # headlight only, all trios
    set_head(1.0)
    model.vis.headlight.active = 0
    passes, plates, washes = [], [], []
    for i in range(len(entries)):
        model.light_active[:] = 0
        model.light_active[i] = 1

        def set_lamp(g, i=i):
            model.light_diffuse[i] = lamp_d[i] * g
            model.light_specular[i] = lamp_s[i] * g

        if not a.relight:
            place(i)
            passes.append(exposed(set_lamp))
        else:
            # pass and plate share one gain, so their ratio is the shadow alone. The lamp
            # rides on its group body, so the group stays put and the plate only hides
            # the arms (geom groups 2 and 3; hidden geoms cast no shadow)
            hide = mujoco.MjvOption()
            hide.geomgroup[2] = hide.geomgroup[3] = 0

            def bare():
                mujoco.mj_forward(model, data)
                ren.update_scene(data, camera=cam, scene_option=hide)
                return ren.render().astype(np.float32)

            place(i)
            set_lamp(1.0)
            plate = bare()
            k = gain_for(plate)
            if k > 1.05:
                set_lamp(k)
                plate = bare() / k
            passes.append(shot() / k if k > 1.05 else shot())
            plates.append(plate)
            # the wash: the same lamp, broader and softer, with no arm in the way
            model.light_exponent[i], model.light_cutoff[i] = a.env_exponent, a.env_cutoff
            set_lamp(1.0)
            wash = bare()
            kw = gain_for(wash)
            if kw > 1.05:
                set_lamp(kw)
                wash = bare() / kw
            washes.append(wash)
            model.light_exponent[i], model.light_cutoff[i] = lamp_exp[i], lamp_cut[i]
        set_lamp(1.0)
    restore()
    model.light_active[:] = lamp_on
    model.vis.headlight.active = head_on

    roomm = owner < 0
    if a.relight:
        lum = [p.mean(axis=2) for p in washes]
        peaks = [float(np.percentile(l_[roomm], 99.5)) for l_ in lum]
        w = [float(np.mean(peaks)) / max(pk, 1e-3) for pk in peaks]
        env = a.fill_weight * fill + sum(wi * p for wi, p in zip(w, washes))
        trans = np.ones((H, W), np.float32)
        for p, pl in zip(passes, plates):
            lp, ll = p.mean(axis=2), pl.mean(axis=2)
            peak = float(np.percentile(ll, 99.5))
            # near the edge of a lamp's cone both images are a few grey levels, so their
            # ratio is noise that prints as faint panels; fade the shadow in as the plate brightens
            conf = np.clip((ll / max(peak, 1e-6) - 0.05) / 0.20, 0.0, 1.0)
            conf = conf * conf * (3.0 - 2.0 * conf)
            ratio = np.clip(lp / np.maximum(ll, 1e-6), 0.0, 1.0)
            trans *= (1.0 - a.strength * conf * (1.0 - ratio)).astype(np.float32)
        env_blur = getattr(a, "env_blur", 0.08)
        if env_blur > 0:
            # ten broad washes each throw the table edge's shadow onto the strip of wall
            # and floor behind the table at a different place, which sums to stepped
            # panels. Smooth the wash along x within each surface; the letters live in
            # `trans`, so they stay sharp, and no light leaks across surfaces
            import cv2
            gl = ren.scene.camera[0]
            pos, fwd, up = (np.asarray(v, np.float64) for v in (gl.pos, gl.forward, gl.up))
            half_w = gl.frustum_width or 0.5 * (W / H) * (gl.frustum_top - gl.frustum_bottom)
            xn = gl.frustum_center + (2.0 * (np.arange(W) + 0.5) / W - 1.0) * half_w
            yn = gl.frustum_bottom + (1.0 - (np.arange(H) + 0.5) / H) * (gl.frustum_top - gl.frustum_bottom)

            def points(r):
                d = fwd * gl.frustum_near + r * xn[None, :, None] + up * yn[:, None, None]
                return pos + d * (room / gl.frustum_near)[..., None]

            right = np.cross(fwd, up)
            P = points(right)
            cols = [float(np.median(np.nonzero(owner == i)[1])) for i in (0, len(entries) - 1)]
            row_x = np.nanmedian(np.where(roomm, P[..., 0], np.nan), axis=0)
            x0, x1 = (float(np.interp(c, np.arange(W), np.nan_to_num(row_x))) for c in cols)
            if (x0 - x1) * (-entries[0]["lat_path_m"][0] + entries[-1]["lat_path_m"][0]) < 0:
                P = points(-right)
            wall_m = roomm & (P[..., 1] > -0.08) & (P[..., 1] < 0.02) & (P[..., 2] > 0.02)
            floor_m = roomm & (P[..., 2] < 0.02) & ~wall_m
            table_m = roomm & ~wall_m & ~floor_m
            s = env_blur * W
            for mk in (wall_m, floor_m, table_m):
                if not mk.any():
                    continue
                mf = mk.astype(np.float32)
                num = cv2.GaussianBlur(env * mf[..., None], (0, 0), sigmaX=s, sigmaY=0.3)
                den = cv2.GaussianBlur(mf, (0, 0), sigmaX=s, sigmaY=0.3)
                env = np.where(mk[..., None], num / np.maximum(den, 1e-4)[..., None], env)
            print(f"env blur: sigma {s:.0f} px; wall {wall_m.mean():.1%}, floor {floor_m.mean():.1%}, table {table_m.mean():.1%}")
        out = env * trans[..., None]
        # the wall sums several overlapping washes while an arm sees only its own lamp,
        # so lift the arms back to the wash's level
        arm_gain = getattr(a, "arm_gain", 2.5)
        for i in range(len(entries)):
            m = owner == i
            out[m] = a.fill_weight * fill[m] + arm_gain * w[i] * passes[i][m]
        print("relight: wash peaks", [round(pk, 3) for pk in peaks],
              "| plate peaks", [round(float(np.percentile(pl.mean(axis=2), 99.5)), 3) for pl in plates],
              "| fill p99.5", round(float(np.percentile(fill.mean(axis=2)[roomm], 99.5)), 3),
              "| trans min/mean", round(float(trans.min()), 3), round(float(trans[roomm].mean()), 3))
    else:
        if a.balance:
            peaks = [float(np.percentile(p.mean(axis=2)[roomm], 99.5)) for p in passes]
            w = [np.mean(peaks) / max(pk, 1e-3) for pk in peaks]
        else:
            w = [1.0] * len(passes)

        win = [np.ones((H, W), np.float32)] * len(passes)
        if a.window:
            # lateral scene x of every room pixel, back-projected from the empty-room depth
            gl = ren.scene.camera[0]
            pos, fwd, up = (np.asarray(v, np.float64) for v in (gl.pos, gl.forward, gl.up))
            right = np.cross(fwd, up)
            u = (np.arange(W) + 0.5) / W
            v = (np.arange(H) + 0.5) / H
            # a free camera leaves frustum_width at 0; mjr_render derives it from the viewport aspect
            half_w = gl.frustum_width or 0.5 * (W / H) * (gl.frustum_top - gl.frustum_bottom)
            xn = gl.frustum_center + (2.0 * u - 1.0) * half_w
            yn = gl.frustum_bottom + (1.0 - v) * (gl.frustum_top - gl.frustum_bottom)

            def lateral(r):
                d = fwd * gl.frustum_near + r * xn[None, :, None] + up * yn[:, None, None]
                return (pos + d * (room / gl.frustum_near)[..., None])[..., 0]

            X = lateral(right)
            # sanity: the first group has the larger scene x (x = -lat); if its pixels sit on
            # the side where X is smaller, the image is mirrored against `right`
            cols = [float(np.median(np.nonzero(owner == i)[1])) for i in (0, len(passes) - 1)]
            row_x = np.nanmedian(np.where(roomm, X, np.nan), axis=0)
            x0, x1 = (float(np.interp(c, np.arange(W), np.nan_to_num(row_x))) for c in cols)
            if (x0 - x1) * (-entries[0]["lat_path_m"][0] + entries[-1]["lat_path_m"][0]) < 0:
                right = -right
                X = lateral(right)
            centres = np.array([-e["lat_path_m"][0] for e in entries])
            order = np.argsort(centres)
            cs = centres[order]

            def ramp(x, b, f):                              # 0 -> 1 across [b - f/2, b + f/2]
                t = np.clip((x - (b - f / 2)) / max(f, 1e-6), 0.0, 1.0)
                return t * t * (3.0 - 2.0 * t)

            win = [None] * len(passes)
            for k, i in enumerate(order):
                wk = np.ones((H, W), np.float32)
                if k > 0:
                    b, f = (cs[k - 1] + cs[k]) / 2, a.feather * (cs[k] - cs[k - 1])
                    wk *= ramp(X, b, f).astype(np.float32)
                if k < len(cs) - 1:
                    b, f = (cs[k] + cs[k + 1]) / 2, a.feather * (cs[k + 1] - cs[k])
                    wk *= (1.0 - ramp(X, b, f)).astype(np.float32)
                win[i] = wk
            print(f"window: lateral X {float(np.nanmin(X[roomm])):+.2f}..{float(np.nanmax(X[roomm])):+.2f} m, "
                  f"sum of windows {float(np.min(sum(win))):.3f}..{float(np.max(sum(win))):.3f}")
        out = a.fill_weight * fill + sum(wi * wn[..., None] * p for wi, wn, p in zip(w, win, passes))
        for i in range(len(entries)):
            m = owner == i
            out[m] = a.fill_weight * fill[m] + w[i] * passes[i][m]
    ref = float(np.percentile(out.mean(axis=2)[roomm], 99.5))
    out *= a.exposure * 235.0 / max(ref, 1e-3)
    sat = float((out > 255).mean())
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(a.out)
    print(f"linked render {W}x{H} -> {a.out}  (weights {[round(float(x), 2) for x in w]}, saturated {sat:.2%})")


if __name__ == "__main__":
    main()
