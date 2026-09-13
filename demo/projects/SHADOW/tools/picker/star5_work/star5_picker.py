"""STAR5 pick sheet: 5-arm star solves rendered the way picker.py renders the
letters -- each shape alone, broad even lamp, arms hidden for the plate -- via
patched copies of the renderer whose build() seats entry["n_arms"] arms.

    <fleet-shadow python> star5_picker.py info
    <fleet-shadow python> star5_picker.py diag   --frame star_n5:13
    <fleet-shadow python> star5_picker.py scan   [--runs star_n5,star_ds12_n5,ICRA_scene_01_star_n5]
    <fleet-shadow python> star5_picker.py sheet  --pick "star_n5:2,13;star_ds12_n5:10" [--sanity 5star2]

Nothing here writes outside demo/projects/SHADOW/tools/picker/ (this folder and
its parent). demo/ensemble_mujoco.py and tools/shadow_post.py are read, never
edited: ensemble_mujoco_n.py patches build() in memory and shadow_post5.py is a
copy of shadow_post.py that imports it.

Fidelity check: the solver (fleet-shadow-art/motion-aware-shadow/renderer.py)
casts from a lamp 1.0 m in front of arm 0, 0.30 m above the base plane, onto a
wall 2.0 m behind the last arm, imaged 128 px over +-0.60 m centred on the lamp's
foot point. The post render casts onto a wall at another distance, which only
scales the shadow about the lamp's foot point, so each render is resampled into
the solver's window and compared pixel-for-pixel with best_shadow.png. A wrong
gap, lamp distance, arm order or handedness shows up as a low fidelity IoU.
"""
import argparse
import csv
import importlib.util
import json
import math
import sys
import time

sys.dont_write_bytecode = True  # never drop __pycache__ into fleet-shadow-art (pack imports motion_planner)
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

REPO = Path(r"C:/Users/hexia/Documents/GitHub/umbra-bench")
FSA = REPO.parent / "fleet-shadow-art"
WORK = Path(__file__).resolve().parent          # tools/picker/star5_work
OUT = WORK.parent                               # tools/picker
CLIPS = WORK / "clips"
SCAN = WORK / "scan"
DIAG = WORK / "diag"
sys.path.insert(0, str(WORK))
import shadow_post5 as sp  # noqa: E402
import mujoco  # noqa: E402

_spec = importlib.util.spec_from_file_location("pack", REPO / "demo" / "pack.py")
pack = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, [sys.argv[0]]
_spec.loader.exec_module(pack)
sys.argv = _argv

N_ARMS = 5
SOURCE = "optimizer_n5"
# run dir -> (sequence id, short name, cross-check choreography in fleet-shadow-art)
RUNS = {
    "star_n5": ("star_spinning_scene_01_star", "spin", "star_spinning_scene_01_star__optimizer_n5.json"),
    "star_ds12_n5": ("star_ds12", "ds12", "star_ds12__optimizer_n5.json"),
    "ICRA_scene_01_star_n5": ("ICRA_scene_01_star", "ICRA", "ICRA_scene_01_star__optimizer_n5.json"),
}
SOLVER_LIGHT_TO_FRONT = 1.0   # renderer.py LIGHT_TO_FRONT_ROBOT
SOLVER_BACK_TO_WALL = 2.0     # renderer.py BACK_ROBOT_TO_WALL
SOLVER_HALF = 0.60            # renderer.py WALL_HZ (window centred at WALL_Z = LIGHT_Z = 0.30)
SOLVER_PX = 128


def font(size):
    for p in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


# ---------------------------------------------------------------- run facts
def run_info(run):
    sid, short, choreo = RUNS[run]
    rdir = REPO / "optimized" / run
    qs, _fps, info = pack.clip_poses(sid, source=SOURCE)
    if Path(info["run"]).resolve() != rdir.resolve():
        sys.exit(f"[!] clip_poses({sid}, {SOURCE}) resolved {info['run']}, expected {rdir}")
    ts = info["ts"]
    if ts != pack.latest_ts(rdir):
        sys.exit(f"[!] {run}: recorded ts {ts} is not the latest {pack.latest_ts(rdir)}")
    summ = json.loads((rdir / f"summary_{ts}.json").read_text(encoding="utf-8"))
    n_rob = int(summ["n_robots"])
    gap = float(info["arm_gap_m"])  # pack.run_arm_gap
    cfg = rdir / "CONFIG.json"
    if cfg.exists():
        c = json.loads(cfg.read_text(encoding="utf-8"))
        gap_src = f"CONFIG.json arm_gap_m={c['arm_gap_m']} n_robots={c['n_robots']}"
        if int(c["n_robots"]) != n_rob:
            sys.exit(f"[!] {run}: CONFIG n_robots {c['n_robots']} != summary {n_rob}")
    else:
        gap_src = f"no CONFIG.json -> pack.run_arm_gap LEGACY_ARM_GAP={pack.LEGACY_ARM_GAP}"
    if summ.get("scene", {}).get("arm_gap") is not None:
        gap_src += f"; summary scene.arm_gap={summ['scene']['arm_gap']}"
        if not math.isclose(float(summ["scene"]["arm_gap"]), gap):
            sys.exit(f"[!] {run}: summary scene.arm_gap disagrees with {gap}")
    cj = FSA / "choreographies" / choreo
    chk = None
    if cj.exists():
        c = json.loads(cj.read_text(encoding="utf-8"))
        rb = sorted(c["robots"], key=lambda r: r["base"]["x"])
        xs = [float(r["base"]["x"]) for r in rb]
        steps = np.diff(xs)
        F = np.stack([np.asarray(r["frames"], dtype=float) for r in rb], 1).reshape(len(rb[0]["frames"]), -1)
        kf = max(float(np.abs(F[30 + 6 * k] - qs[k].ravel()).max()) for k in range(len(qs)))
        chk = {"file": cj.name, "n_robots": len(rb), "base_x": xs, "keyframe_max_abs_diff": kf}
        if len(rb) != n_rob or not np.allclose(steps, gap):
            sys.exit(f"[!] {run}: choreography {cj.name} spacing {steps} / {len(rb)} robots disagrees with gap {gap}")
        gap_src += f"; choreography {cj.name} base.x step {float(steps.mean()):.2f} (keyframes == solve, max diff {kf:.1e})"
    else:
        gap_src += f"; choreography {cj.name} not present"
    tc = {}
    mp = REPO / "results" / f"sequence_metrics_{run}.csv"
    if mp.exists():
        with open(mp, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r["sequence_id"] == sid and r["source"] == SOURCE and r["ref"] == "original"
                        and r["frame_idx"] != "" and r["tc_iou"]):
                    tc[int(r["frame_idx"]) + 1] = float(r["tc_iou"])
    frames = {}
    for fr in summ["frames"]:
        n = int(fr["frame"])
        q = np.asarray(qs[n - 1], dtype=float).ravel()
        if q.size != 6 * n_rob:
            sys.exit(f"[!] {run} frame {n}: q has {q.size} values for {n_rob} robots")
        frames[n] = {"target": fr["target"], "iou": float(fr["iou"]), "tc_iou": tc.get(n), "q": q,
                     "best_shadow": rdir / f"frame_{n:02d}_{ts}" / "best_shadow.png"}
    return {"run": run, "sid": sid, "short": short, "ts": ts, "n_robots": n_rob, "gap": gap,
            "gap_source": gap_src, "choreo": chk, "frames": frames, "dir": rdir}


# ---------------------------------------------------------------- rendering
def entry_for(clip, gap, n=N_ARMS, ddepth=None):
    # lamp 1.0 m in front of arm 0; arm 0 sits (n-1)/2*gap in front of the mid
    # arm on the group origin, and the renderer puts the lamp at ly = -ddepth
    # from that origin
    if ddepth is None:
        ddepth = -(SOLVER_LIGHT_TO_FRONT + (n - 1) / 2 * gap)
    return {"element": "star", "clip": clip, "entry": 0, "n_frames": 25, "mag": 2.0, "depth_m": 1.2,
            "n_arms": n, "arm_gap_m": gap, "lat_path_m": [0.0] * 25, "lift_path_m": [0.0] * 25,
            "light": {"ddepth": ddepth, "h": 0.3}}


def camera():
    cam = mujoco.MjvCamera()
    cam.lookat = [0.0, 0.0, sp.LOOK_Z]
    cam.azimuth, cam.elevation, cam.distance = -90, 0, sp.CAM_Y
    return cam


def rig_check(man, clipj, e, scale=4):
    """Arm pixels visible to the post camera (segmentation), plus where build()
    actually put each arm base and the lamp, in metres from the wall."""
    sp.em.LIGHT.update(sp.LAYER_LIGHT)
    model, rigs, _ = sp.em.build(dict(man, entries=[e]))
    data = mujoco.MjData(model)
    robots = sorted(clipj["robots"], key=lambda r: r["base"]["x"])
    for pfx, r in zip(rigs[0][0], robots):
        for jn, v in zip(sp.em.JOINT_NAMES, r["frames"][0]):
            data.qpos[model.joint(pfx + jn).qposadr[0]] = v
    mujoco.mj_forward(model, data)
    roots = {}
    for b in range(model.nbody):
        nm = model.body(b).name
        if nm.startswith("e0r") and nm[3].isdigit():
            roots.setdefault(int(nm[3]), b)
    base_y = [round(float(data.xpos[roots[j]][1]), 4) for j in sorted(roots)]
    lamp = [round(float(v), 4) for v in data.light_xpos[0]]
    W, H = sp.RES[0] // scale, sp.RES[1] // scale
    model.vis.global_.fovy = sp.FOVY
    ren = mujoco.Renderer(model, H, W)
    ren.enable_segmentation_rendering()
    ren.update_scene(data, camera=camera(), scene_option=mujoco.MjvOption())
    seg = ren.render()
    ren.close()
    ids, types = seg[..., 0], seg[..., 1]
    arm_geom = np.array([model.body(int(model.geom_bodyid[g])).name.startswith("e0r") for g in range(model.ngeom)])
    hit = (types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
    hit[hit] = arm_geom[ids[hit]]
    # deepest arm geometry toward the wall (y), to know how much camera room is left
    arm_g = [g for g in range(model.ngeom) if arm_geom[g]]
    min_geom_y = float(min(data.geom_xpos[g][1] for g in arm_g)) if arm_g else None
    return {"arm_px": int(hit.sum()) * scale * scale, "base_y": base_y, "lamp": lamp,
            "lamp_to_arm0": round(lamp[1] - base_y[0], 4) if base_y else None,
            "min_arm_geom_y": round(min_geom_y, 3) if min_geom_y is not None else None}


def render(man, dest, name, q, gap, ddepth=None, rig=True):
    pack.write_choreo(dest, name, 5.0, [q], force=True, arm_gap=gap)
    clipj = json.loads((dest / f"{name}.json").read_text(encoding="utf-8"))
    if len(clipj["robots"]) != N_ARMS or any(len(r["frames"]) != 105 for r in clipj["robots"]):
        sys.exit(f"[!] {name}: clip has {len(clipj['robots'])} robots / frames "
                 f"{[len(r['frames']) for r in clipj['robots']]}")
    e = entry_for(name, gap, ddepth=ddepth)
    full, plate = sp.render_letter(man, {name: clipj}, e)
    m = sp.matte_of(full, plate)
    return m, e, (rig_check(man, clipj, e) if rig else None), full


def solver_frame(m, e, solver_gap):
    """Resample a post matte into the solver's 128 px wall window (bool, shadow=True)."""
    H, W = m.shape
    ppm = (H / 2) / (sp.CAM_Y * math.tan(math.radians(sp.FOVY / 2)))
    t_r = sp.em.SCREEN_Z - e["depth_m"] - e["light"]["ddepth"]           # lamp -> wall, post render
    t_s = SOLVER_LIGHT_TO_FRONT + (e["n_arms"] - 1) * solver_gap + SOLVER_BACK_TO_WALL
    k = t_r / t_s
    half = k * SOLVER_HALF * ppm
    cx = W / 2
    cy = H / 2 - ((sp.em.TABLE_H + e["light"]["h"]) - sp.LOOK_Z) * ppm
    img = Image.fromarray((m * 255).astype(np.uint8)).resize(
        (SOLVER_PX, SOLVER_PX), Image.BOX, box=(cx - half, cy - half, cx + half, cy + half))
    return np.asarray(img, dtype=np.float32) / 255.0 < 0.5


def load_best(p):
    return np.asarray(Image.open(p).convert("L")) > 127


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def shade_crop(m, pad=20):
    got = sp.crop_box(m, pad=pad)
    if not got:
        return None, None, False
    (x0, y0, x1, y1), edge = got
    crop = m[y0:y1, x0:x1]
    shade = np.clip(222 * (1.0 - 0.88 * (1.0 - crop)), 0, 255).astype(np.uint8)
    return Image.fromarray(shade).convert("RGB"), (int(x0), int(y0), int(x1), int(y1)), bool(edge)


def setup():
    sp.RES = (1920, 1080)
    return json.loads((FSA / "ensembles" / "SHADOW_scene_01.json").read_text(encoding="utf-8"))


def settings():
    return {"RES": list(sp.RES), "CAM_Y": sp.CAM_Y, "FOVY": sp.FOVY, "LOOK_Z": sp.LOOK_Z,
            "LAYER_LIGHT": sp.LAYER_LIGHT, "px_per_m": round((sp.RES[1] / 2) / (sp.CAM_Y * math.tan(math.radians(sp.FOVY / 2))), 2)}


# ---------------------------------------------------------------- commands
def cmd_info(a):
    for run in RUNS:
        ri = run_info(run)
        print(f"\n== {run} ({ri['sid']}, {SOURCE})  robots={ri['n_robots']}  gap={ri['gap']}  ts={ri['ts']}")
        print(f"   gap source: {ri['gap_source']}")
        print("   " + "  ".join(f"{n:02d}:{f['iou']:.3f}/{(f['tc_iou'] if f['tc_iou'] is not None else float('nan')):.2f}"
                               for n, f in ri["frames"].items()))


def cmd_diag(a):
    man = setup()
    run, n = a.frame.split(":")
    n = int(n)
    ri = run_info(run)
    fr = ri["frames"][n]
    best = load_best(fr["best_shadow"])
    g = ri["gap"]
    og = 0.25 if math.isclose(g, 0.15) else 0.15
    q = fr["q"]
    qrev = q.reshape(N_ARMS, 6)[::-1].ravel()
    dest = DIAG / "clips"
    variants = [
        ("as solved: gap %.2f, lamp 1.0 before arm0 (ddepth %.3f)" % (g, -(1 + 2 * g)), q, g, None, g),
        ("other gap %.2f (solver assumed %.2f too)" % (og, og), q, og, None, og),
        ("lamp 1.0 before arm0 with 4*gap formula (ddepth %.3f)" % (-(1 + 4 * g)), q, g, -(1 + 4 * g), g),
        ("3-arm formula (ddepth %.3f)" % (-(1 + g)), q, g, -(1 + g), g),
        ("arm order reversed", qrev, g, None, g),
    ]
    print(f"diag {run} frame {n}: solver IoU {fr['iou']}  settings {settings()}")
    for i, (label, qq, gg, dd, sg) in enumerate(variants):
        t0 = time.time()
        m, e, rig, _ = render(man, dest, f"{run}__f{n:02d}__v{i}", qq, gg, ddepth=dd, rig=(i == 0))
        s = solver_frame(m, e, sg)
        _, box, edge = shade_crop(m)
        print(f"  v{i} {label}: fidelity IoU {iou(s, best):.3f}  (mirrored {iou(s[:, ::-1], best):.3f})  "
              f"box={box} edge={edge} {time.time() - t0:.1f}s" + (f"\n      rig {rig}" if rig else ""))
        if i == 0:
            Image.fromarray((np.dstack([best, s, np.zeros_like(s)]) * 255).astype(np.uint8)).resize((384, 384), Image.NEAREST).save(DIAG / f"diag_{run}_f{n:02d}_overlay.png")


def cmd_gapab(a):
    """Render each frame at gap 0.15 and 0.25 and score both against best_shadow.png.
    Diagonal = consistent hypothesis (render gap == solver gap)."""
    man = setup()
    res = []
    for part in a.frames.split(";"):
        run, ns = part.split(":")
        ri = run_info(run)
        for n in [int(x) for x in ns.split(",")]:
            fr = ri["frames"][n]
            best = load_best(fr["best_shadow"])
            row = {}
            for g in (0.15, 0.25):
                m, e, _, _ = render(man, DIAG / "clips", f"{run}__f{n:02d}__g{int(g * 100)}", fr["q"], g, rig=False)
                for sg in (0.15, 0.25):
                    row[(g, sg)] = iou(solver_frame(m, e, sg), best)
            row = {(round(k[0], 2), round(k[1], 2)): v for k, v in row.items()}
            res.append((run, n, ri["gap"], row))
            print(f"  {run} f{n:02d} claimed gap {ri['gap']}: hyp 0.15 -> {row[(0.15, 0.15)]:.3f}   hyp 0.25 -> {row[(0.25, 0.25)]:.3f}   "
                  f"(cross r.15/s.25 {row[(0.15, 0.25)]:.3f}, r.25/s.15 {row[(0.25, 0.15)]:.3f})", flush=True)
    for run in sorted({r[0] for r in res}):
        rr = [r for r in res if r[0] == run]
        print(f"  {run}: mean hyp 0.15 {np.mean([r[3][(0.15, 0.15)] for r in rr]):.4f}  hyp 0.25 {np.mean([r[3][(0.25, 0.25)] for r in rr]):.4f}  "
              f"wins for claimed gap {sum((r[3][(r[2], r[2])] >= r[3][(0.4 - r[2], 0.4 - r[2])]) for r in rr)}/{len(rr)}")


def cmd_scan(a):
    man = setup()
    SCAN.mkdir(parents=True, exist_ok=True)
    idx_p = SCAN / "scan.json"
    idx = json.loads(idx_p.read_text(encoding="utf-8")) if idx_p.exists() else {}
    runs = [r for r in a.runs.split(",") if r]
    for run in runs:
        ri = run_info(run)
        frames = [n for n in ri["frames"] if not (a.max_frame and n > a.max_frame.get(run, 999))]
        for n in frames:
            fr = ri["frames"][n]
            key = f"{run}__f{n:02d}"
            if key in idx and idx[key].get("settings") == settings() and (SCAN / f"{key}.png").exists():
                continue
            t0 = time.time()
            m, e, rig, _ = render(man, SCAN / "clips", key, fr["q"], ri["gap"])
            s = solver_frame(m, e, ri["gap"])
            best = load_best(fr["best_shadow"])
            img, box, edge = shade_crop(m)
            if img is not None:
                img.save(SCAN / f"{key}.png")
            idx[key] = {"run": run, "frame": n, "target": fr["target"], "solver_iou": fr["iou"], "tc_iou": fr["tc_iou"],
                        "gap": ri["gap"], "fidelity_iou": round(iou(s, best), 4),
                        "fidelity_iou_mirrored": round(iou(s[:, ::-1], best), 4),
                        "box": box, "edge": edge, "rig": rig, "settings": settings()}
            idx_p.write_text(json.dumps(idx, indent=1), encoding="utf-8")
            print(f"  {key}: fid {idx[key]['fidelity_iou']:.3f} (mir {idx[key]['fidelity_iou_mirrored']:.3f}) "
                  f"box {box} edge={edge} arm_px={rig['arm_px']} minY={rig['min_arm_geom_y']} {time.time() - t0:.1f}s", flush=True)
        # contact sheet for choosing
        S, cols = 300, 4
        cw, ch = S + 150 + 20, S + 40
        ns = sorted(n for n in frames if f"{run}__f{n:02d}" in idx)
        sheet = Image.new("RGB", (cols * cw, ((len(ns) + cols - 1) // cols) * ch), (255, 255, 255))
        f18 = font(16)
        for k, n in enumerate(ns):
            key = f"{run}__f{n:02d}"
            r = idx[key]
            cell = Image.new("RGB", (cw, ch), (222, 222, 222))
            if (SCAN / f"{key}.png").exists():
                im = Image.open(SCAN / f"{key}.png")
                kk = min(S / im.width, S / im.height)
                im = im.resize((max(1, int(im.width * kk)), max(1, int(im.height * kk))), Image.LANCZOS)
                cell.paste(im, ((S - im.width) // 2, 36 + (S - im.height) // 2))
            bs = Image.open(ri["frames"][n]["best_shadow"]).convert("L").resize((150, 150), Image.NEAREST)
            cell.paste(Image.merge("RGB", [bs] * 3), (S + 10, 36))
            tg = Image.open(REPO / "sequences" / ri["sid"] / r["target"]).convert("L").resize((150, 150), Image.LANCZOS)
            cell.paste(Image.merge("RGB", [tg] * 3), (S + 10, 36 + 152))
            d = ImageDraw.Draw(cell)
            d.text((4, 2), f"{key}  iou {r['solver_iou']:.2f} tc {(r['tc_iou'] or float('nan')):.2f}", fill=(200, 0, 0), font=f18)
            d.text((4, 18), f"fid {r['fidelity_iou']:.2f}  edge {r['edge']}  armpx {r['rig']['arm_px']}", fill=(0, 0, 0), font=f18)
            d.rectangle([0, 0, cw - 1, ch - 1], outline=(150, 150, 150))
            sheet.paste(cell, ((k % cols) * cw, (k // cols) * ch))
        sheet.save(DIAG / f"scan_{run}.png")
        print("scan sheet ->", DIAG / f"scan_{run}.png")


def cmd_sheet(a):
    man = setup()
    CLIPS.mkdir(parents=True, exist_ok=True)
    picks = []
    for part in a.pick.split(";"):
        if not part.strip():
            continue
        run, ns = part.split(":")
        picks += [(run.strip(), int(x)) for x in ns.split(",") if x.strip()]
    notes = json.loads(a.notes) if a.notes else {}
    infos = {run: run_info(run) for run in {p[0] for p in picks}}
    CW, CH, SH = 460, 600, 470
    rows = (len(picks) + 2) // 3
    sheet = Image.new("RGB", (CW * 3, CH * rows), (245, 245, 245))
    big, small = font(34), font(22)
    rows_out, sanity = [], None
    for i, (run, n) in enumerate(picks):
        ri = infos[run]
        fr = ri["frames"][n]
        code = f"5star{i + 1}"
        name = f"{run}__f{n:02d}"
        t0 = time.time()
        m, e, rig, full = render(man, CLIPS, name, fr["q"], ri["gap"])
        s = solver_frame(m, e, ri["gap"])
        best = load_best(fr["best_shadow"])
        fid, fid_m = iou(s, best), iou(s[:, ::-1], best)
        img, box, edge = shade_crop(m)
        cell = Image.new("RGB", (CW, CH), (222, 222, 222))
        if img is not None:
            k = min((CW - 40) / img.width, (SH - 40) / img.height)
            img = img.resize((max(1, int(img.width * k)), max(1, int(img.height * k))), Image.LANCZOS)
            cell.paste(img, ((CW - img.width) // 2, 20 + (SH - 40 - img.height) // 2))
        tgt = Image.open(REPO / "sequences" / ri["sid"] / fr["target"]).convert("L").resize((96, 96), Image.LANCZOS)
        cell.paste(Image.merge("RGB", [tgt] * 3), (CW - 106, 10))
        d = ImageDraw.Draw(cell)
        d.rectangle([0, SH, CW, CH], fill=(250, 250, 250))
        d.text((14, SH + 10), code, fill=(200, 30, 30), font=big)
        tc = fr["tc_iou"]
        info = f"{ri['short']}  fr{n:02d}  gap {ri['gap']:.2f}  IoU {fr['iou']:.2f}" + (f"  tc {tc:.2f}" if tc is not None else "")
        fs = 22
        while d.textlength(info, font=font(fs)) > CW - 28 and fs > 14:
            fs -= 1
        d.text((14, SH + 58), info, fill=(40, 40, 40), font=font(fs))
        d.rectangle([0, 0, CW - 1, CH - 1], outline=(180, 180, 180), width=2)
        sheet.paste(cell, ((i % 3) * CW, (i // 3) * CH))
        row = {"code": code, "run": run, "sequence": ri["sid"], "source": SOURCE, "frame": n,
               "target": fr["target"], "ts": ri["ts"], "n_robots": ri["n_robots"], "arm_gap_m": ri["gap"],
               "gap_source": ri["gap_source"], "solver_iou": fr["iou"], "tc_iou": tc,
               "clip": name, "clip_path": str(CLIPS / f"{name}.json"),
               "best_shadow": str(fr["best_shadow"]),
               "render_vs_best_shadow_iou": round(fid, 4), "render_vs_best_shadow_iou_mirrored": round(fid_m, 4),
               "crop_box_px": box, "edge": edge, "arm_px_in_view": rig["arm_px"],
               "entry": e, "rig": rig, "render_settings": settings(), "note": notes.get(code, "")}
        rows_out.append(row)
        print(f"  {code} {name} gap={ri['gap']} iou={fr['iou']} tc={tc} fid={fid:.3f} (mir {fid_m:.3f}) "
              f"box={box} edge={edge} arm_px={rig['arm_px']} {time.time() - t0:.1f}s", flush=True)
        if a.sanity == code:
            sanity = (code, name, fr, best, s, m, fid, ri)
    stem = a.name
    sheet.save(OUT / f"{stem}_picker_sheet.png")
    sheet.resize((sheet.width * 2 // 3, sheet.height * 2 // 3), Image.LANCZOS).save(OUT / f"{stem}_picker_sheet_preview.jpg", quality=90)
    (OUT / f"{stem}_picker.json").write_text(json.dumps(rows_out, indent=1), encoding="utf-8")
    print("sheet ->", OUT / f"{stem}_picker_sheet.png")
    if sanity:
        code, name, fr, best, s, m, fid, ri = sanity
        P = 384
        pan = [Image.fromarray((best * 255).astype(np.uint8)).resize((P, P), Image.NEAREST),
               Image.fromarray((s * 255).astype(np.uint8)).resize((P, P), Image.NEAREST),
               Image.fromarray(np.dstack([(best & ~s), (best & s), (s & ~best)]).astype(np.uint8) * 255).resize((P, P), Image.NEAREST)]
        img, _, _ = shade_crop(m)
        k = P / img.height
        pan.append(img.resize((max(1, int(img.width * k)), P), Image.LANCZOS))
        W = sum(p.width for p in pan) + 10 * (len(pan) + 1)
        fig = Image.new("RGB", (W, P + 110), (250, 250, 250))
        x = 10
        heads = ["solver best_shadow.png", "post render -> solver window", "overlay: green both, red solver, blue render", "post render (sheet crop)"]
        d = ImageDraw.Draw(fig)
        for p, h in zip(pan, heads):
            fig.paste(p, (x, 40))
            d.text((x, 12), h, fill=(40, 40, 40), font=font(18))
            x += p.width + 10
        d.text((10, P + 52), f"{code} = {name}  ({ri['sid']} {SOURCE}, target {fr['target']})   gap {ri['gap']:.2f}   "
               f"ddepth {-(1 + 2 * ri['gap']):.2f}   render-vs-solver IoU {fid:.3f}", fill=(200, 30, 30), font=font(20))
        fig.save(OUT / f"{stem}_sanity_side_by_side.png")
        print("sanity ->", OUT / f"{stem}_sanity_side_by_side.png")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    p = sub.add_parser("diag")
    p.add_argument("--frame", required=True)
    p = sub.add_parser("gapab")
    p.add_argument("--frames", required=True, help='"star_n5:2,13;ICRA_scene_01_star_n5:6"')
    p = sub.add_parser("scan")
    p.add_argument("--runs", default=",".join(RUNS))
    p.add_argument("--max-frame", dest="max_frame", default=None, help='json {"run": last frame to render}')
    p = sub.add_parser("sheet")
    p.add_argument("--pick", required=True)
    p.add_argument("--name", default="STAR5")
    p.add_argument("--sanity", default=None, help="code to draw the best_shadow-vs-render side-by-side for")
    p.add_argument("--notes", default=None, help='json {"5star1": "note", ...}')
    a = ap.parse_args()
    if a.cmd == "scan" and a.max_frame:
        a.max_frame = json.loads(a.max_frame)
    {"info": cmd_info, "diag": cmd_diag, "gapab": cmd_gapab, "scan": cmd_scan, "sheet": cmd_sheet}[a.cmd](a)


if __name__ == "__main__":
    main()
