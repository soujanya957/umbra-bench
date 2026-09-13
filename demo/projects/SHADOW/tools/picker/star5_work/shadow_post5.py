"""Per-letter shadow layers and a post-produced SHADOW word.

Each letter is rendered ALONE: only its own trio and its own lamp, so no
neighbouring lamp can put a ghost copy on it. The shadow is extracted as
matte = render / plate, where the plate is the same view with the arms hidden
(geom groups 2 and 3). The camera sits between the arms and the wall with a wide
fovy, so no arm is in frame and the flat wall is imaged at uniform scale. Every
letter uses the same camera distance, fovy and resolution, so its pixels per
metre are shared and relative sizes stay true.

A layer only has to trace the silhouette, so its lamp is broad and even with no
fill. The umbra is then black wherever the letter falls. With the scene's
focused spot, the far strokes of H and O came out half-lit. The shape is set by
where the lamp sits relative to the arms, which is unchanged, not by its cone.
How dark the letters look is decided later, in the composite (--darkness).

The letters are laid out on a clean wall with a uniform gap, bottom-aligned by
default. Per-letter dx/dy/scale/darkness can be set in an adjust JSON, and
every letter is also written as an RGBA layer for manual compositing.

  <fleet-shadow python> shadow_post.py --manifest M.json [--clip-dir DIR] [--out DIR]
        [--adjust adjust.json] [--darkness 0.88] [--wide hero.png] [--wide2 left.png]
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageFilter

REPO = Path(r"C:/Users/hexia/Documents/GitHub/umbra-bench")
# PATCHED COPY (star5_work): imports the in-memory n-arm build() instead of
# demo/ensemble_mujoco.py, which is otherwise identical and left untouched.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ensemble_mujoco_n as em  # noqa: E402

ARM_GROUPS = (2, 3)
LAYER_LIGHT = {"cutoff": 70, "exponent": 2, "diffuse": 1.0, "fill": 0.0}
RES = (3840, 2160)
# STAR5 patch: a 5-arm rig reaches to ~0.38 m off the wall (gap 0.25: nearest base
# at 0.70), in front of the letters' 0.6 m camera, so the arms themselves were in
# view. The camera moves inside the rig and FOVY widens so px per metre on the
# wall stays exactly the letters' (0.6 m @ 100 deg = 755.19 px/m at 1080 rows).
LETTER_CAM_Y, LETTER_FOVY = 0.6, 100.0
CAM_Y = 0.25
FOVY = float(2.0 * np.degrees(np.arctan(LETTER_CAM_Y * np.tan(np.radians(LETTER_FOVY / 2)) / CAM_Y)))
LOOK_Z = 0.85


def load_manifest(path, clip_dir):
    man = json.loads(Path(path).read_text(encoding="utf-8"))
    clips = {}
    for e in man["entries"]:
        c = e["clip"]
        p = Path(clip_dir) / f"{c}.json" if clip_dir else None
        if not (p and p.exists()):
            p = em.FSA / "choreographies" / f"{c}.json"
        clips[c] = json.loads(p.read_text(encoding="utf-8"))
    return man, clips


def render_letter(man, clips, e):
    em.LIGHT.update(LAYER_LIGHT)
    model, rigs, _ = em.build(dict(man, entries=[e]))
    data = mujoco.MjData(model)
    robots = sorted(clips[e["clip"]]["robots"], key=lambda r: r["base"]["x"])
    for pfx, r in zip(rigs[0][0], robots):
        for jn, v in zip(em.JOINT_NAMES, r["frames"][0]):
            data.qpos[model.joint(pfx + jn).qposadr[0]] = v
    mujoco.mj_forward(model, data)
    W, H = RES
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    model.vis.global_.fovy = FOVY
    ren = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat = [-e["lat_path_m"][0], 0.0, LOOK_Z]
    cam.azimuth, cam.elevation, cam.distance = -90, 0, CAM_Y
    shown, hidden = mujoco.MjvOption(), mujoco.MjvOption()
    for g in ARM_GROUPS:
        hidden.geomgroup[g] = 0
    ren.update_scene(data, camera=cam, scene_option=shown)
    full = ren.render().astype(np.float32)
    ren.update_scene(data, camera=cam, scene_option=hidden)
    plate = ren.render().astype(np.float32)
    ren.close()
    return full, plate


def matte_of(full, plate):
    lum_f, lum_p = full.mean(2), plate.mean(2)
    m = np.where(lum_p > 8.0, lum_f / np.maximum(lum_p, 1.0), 1.0)
    return np.clip(m, 0.0, 1.0)


def crop_box(m, thr=0.92, pad=40):
    ys, xs = np.where(m < thr)
    if not len(xs):
        return None
    H, W = m.shape
    edge = xs.min() <= 1 or ys.min() <= 1 or xs.max() >= W - 2 or ys.max() >= H - 2
    return (max(0, xs.min() - pad), max(0, ys.min() - pad),
            min(W, xs.max() + pad + 1), min(H, ys.max() + pad + 1)), edge


def wall(h, w, base=222, vignette=0.10):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h * 0.55) / (h * 0.75)) ** 2)
    return np.clip(base * (1.0 - vignette * np.clip(r, 0, 1.5) ** 2), 0, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--clip-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adjust", default=None, help='{"S": {"dx":0,"dy":0,"scale":1,"darkness":1}, ...}')
    ap.add_argument("--darkness", type=float, default=0.88, help="umbra depth for every letter (1 = black)")
    ap.add_argument("--gap", type=float, default=0.22, help="gap between letters, as a fraction of median letter height")
    ap.add_argument("--wide", default=None, help="hero scene render to stitch above the close-up")
    ap.add_argument("--wide2", default=None, help="optional second view stitched beside --wide")
    ap.add_argument("--reuse", action="store_true", help="reuse existing per-letter mattes in --out")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "layers").mkdir(parents=True, exist_ok=True)
    man, clips = load_manifest(a.manifest, a.clip_dir)
    adj = json.loads(Path(a.adjust).read_text(encoding="utf-8")) if a.adjust else {}

    letters = []
    for i, e in enumerate(man["entries"]):
        tag = f"{i}_{e['element']}"
        mp = out / "layers" / f"{tag}_matte16.png"
        if a.reuse and mp.exists():
            m = np.asarray(Image.open(mp), dtype=np.float32) / 65535.0
        else:
            full, plate = render_letter(man, clips, e)
            m = matte_of(full, plate)
            Image.fromarray(full.astype(np.uint8)).save(out / "layers" / f"{tag}_render.jpg", quality=90)
            Image.fromarray((m * 65535).astype(np.uint16)).save(mp)
        got = crop_box(m)
        if got is None:
            print(f"  [!] {tag}: no shadow in frame")
            continue
        (x0, y0, x1, y1), edge = got
        c = m[y0:y1, x0:x1]
        # some letters keep a faintly lit umbra; stretch each layer so its
        # darkest 2% is black and every letter reaches the same depth
        lo = float(np.percentile(c, 2))
        if lo > 0.0:
            c = np.clip((c - lo) / max(1.0 - lo, 1e-3), 0.0, 1.0)
        if edge:
            print(f"  [!] {tag}: shadow touches the frame edge -- widen FOVY or move LOOK_Z")
        alpha = ((1.0 - c) * 255).astype(np.uint8)
        rgba = np.dstack([np.zeros_like(alpha)] * 3 + [alpha])
        Image.fromarray(rgba, "RGBA").save(out / "layers" / f"{tag}_shadow_rgba.png")
        letters.append((e["element"], c))
        print(f"  {tag}: box {x1 - x0}x{y1 - y0} px, umbra {c.min():.2f}, "
              f"umbra p90 {np.percentile(c[c < 0.5], 90) if (c < 0.5).any() else float('nan'):.2f}, edge={edge}")

    # layout: bottom-aligned (or centred), uniform gap, per-letter adjustments;
    # "space_before" (in median letter heights) opens a word gap before a letter
    hs = [c.shape[0] * adj.get(el, {}).get("scale", 1.0) for el, c in letters]
    med_h = float(np.median(hs))
    gap = int(a.gap * med_h)
    placed, x = [], gap
    for el, c in letters:
        k = adj.get(el, {})
        x += int(float(k.get("space_before", 0.0)) * med_h)
        s = float(k.get("scale", 1.0))
        img = Image.fromarray((c * 255).astype(np.uint8))
        if s != 1.0:
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
        placed.append((el, np.asarray(img, dtype=np.float32) / 255.0, x + int(k.get("dx", 0)), int(k.get("dy", 0)),
                       float(k.get("darkness", a.darkness)), k.get("align", "base")))
        x += img.width + gap
    Hc = int(max(p[1].shape[0] for p in placed) * 1.5)
    Wc = x
    canvas = wall(Hc, Wc)
    base_y = int(Hc * 0.82)
    layout = {}
    for el, c, px, dy, dark, align in placed:
        h, w = c.shape
        py = (base_y - int(med_h / 2) - h // 2 if align == "center" else base_y - h) + dy
        py = max(0, min(Hc - h, py))
        region = canvas[py:py + h, px:px + w]
        mm = 1.0 - dark * (1.0 - c)
        canvas[py:py + h, px:px + w] = region * mm
        layout[el] = {"x": px, "y": py, "w": w, "h": h, "darkness": dark, "align": align}
    close = Image.fromarray(canvas.astype(np.uint8)).convert("RGB")
    close = close.filter(ImageFilter.GaussianBlur(0.6))
    tw = 3840
    close = close.resize((tw, round(close.height * tw / close.width)), Image.LANCZOS)
    close.save(out / "SHADOW_close_composite.png")
    (out / "layout.json").write_text(json.dumps({"gap_px": gap, "letters": layout}, indent=1), encoding="utf-8")
    print("close-up composite:", close.size)

    if a.wide:
        panels = [Image.open(a.wide).convert("RGB")]
        if a.wide2:
            panels.append(Image.open(a.wide2).convert("RGB"))
        top_h = round(tw / len(panels) * 9 / 16)
        panels = [p.resize((tw // len(panels), top_h), Image.LANCZOS) for p in panels]
        top = Image.new("RGB", (tw, top_h), (0, 0, 0))
        for j, p in enumerate(panels):
            top.paste(p, (j * (tw // len(panels)), 0))
        sep = 24
        fig = Image.new("RGB", (tw, top.height + sep + close.height), (0, 0, 0))
        fig.paste(top, (0, 0))
        fig.paste(close, (0, top.height + sep))
        fig.save(out / "SHADOW_stitched.png")
        fig.resize((1920, round(fig.height * 1920 / tw)), Image.LANCZOS).save(out / "SHADOW_stitched_preview.jpg", quality=90)
        print("stitched figure:", fig.size)


if __name__ == "__main__":
    main()
