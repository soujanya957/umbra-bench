#!/usr/bin/env python3
"""make_ensemble.py — the whole scene on stage at once, as data.

    python demo/make_ensemble.py --project pixar
    python demo/make_ensemble.py --project pixar --scene scene_03

Three arms are one ELEMENT; a scene with six elements needs six trios. The
real fleet has one trio, so the deployable arrangement plays elements in
sequence — but the sim can seat every trio simultaneously and recreate the
composited shot in space. This script emits one manifest per scene into the
robot UI's ensembles/ folder with everything that view needs and the clip
envelope does not carry:

  entry      — the film frame (scene-relative, authored 5 fps) where the
               element enters; before it the trio holds a DEFAULT pose and
               blends into its first solved pose across default_blend_s.
  lat_path_m — per-authored-frame lateral offset of the WHOLE trio, in stage
               metres, derived from the element's authored on-canvas centroid.
               For travelling (stab-solved) elements this makes the group
               glide along the footage's trajectory while the joints hold the
               shape — the trajectory_px idea, embodied as base motion.
               Vertical travel has no base equivalent (arms don't fly) and is
               dropped.

Placement math: canvas x → screen metres via SCREEN_W_FILM (the UI maps the
full canvas width onto SCREEN_W = 1.3*PROJ_SCALE = 2.21 m); arm lateral =
screen lateral / m where m = (screenX-lightX)/(z_mid-lightX) is the point-
light magnification at the trio's mid arm. It assumes each solve casts near
the wall centre (the fit recentres its target); eyeball in the sim and
recalibrate if an element sits visibly off.

Timing note for the UI: choreo clips are the authored 5 fps interpolated to
30 Hz with a 30-frame held head/tail, so authored frame k plays at choreo
frame head_pad + k*interp_factor (single-pose library clips just clamp).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent
FSA = BENCH.parent / "fleet-shadow-art"
OUT = FSA / "ensembles"

# mirrors pack.py's CHOREO_SCENE / the UI's stageProjection constants
SCENE = {"lightX": -1.0, "lightY": 0, "lightH": 0.85, "screenX": 2.4}
SCREEN_W_FILM = 1.3 * 1.7          # metres spanned by the full canvas width
Z_MID = 0.2                        # trio arms sit at depth 0/0.2/0.4
MAGNIFICATION = (SCENE["screenX"] - SCENE["lightX"]) / (Z_MID - SCENE["lightX"])
# The solver's render window is NOT the UI wall: renderer.py frames its
# shadow camera to a 0.6 m half-extent wall, so a best_shadow fraction is a
# fraction of 1.2 m (audit: treating it as 2.21 m made every measured width
# and offset 1.84x too large). Rigs differ too: clip solves ran the default
# arm_gap 0.15 rig, the letter library the 0.2 rig.
SOLVER_SPAN = 1.2
MAG_CLIP_RIG = 3.3 / 1.15
MAG_LIB_RIG = 3.4 / 1.2


def shadow_geometry(clip: str):
    """(off_arm_per_frame, width_arm, note) — the solve's own cast, measured
    AT THE ARM PLANE (silhouette metres), so any ensemble magnification can
    re-project it. Per-frame offsets are the fix for travel double-counting:
    a raw sequence solve keeps the footage's motion in its joints, so its
    blob WALKS across the solver frame — subtracting only frame 0's offset
    replayed that walk on top of the rig glide. A _stab solve's offsets are
    ~constant, so the same code handles both."""
    def measure(path, white):
        g = np.array(Image.open(path).convert("L"))
        m = g > 128
        if not white and m.mean() > 0.5:
            m = ~m
        ys, xs = np.where(m)
        if not len(xs):
            return None
        W = m.shape[1]
        return ((xs.mean() / W) - 0.5, (xs.max() - xs.min() + 1) / W)

    run = BENCH / "optimized" / clip
    js = sorted(run.glob("summary_*.json"))
    if js:                                  # a sequence solve
        ts = js[-1].stem[len("summary_"):]
        note = None
        try:
            tf = (json.loads(js[-1].read_text(encoding="utf-8"))
                  .get("target_fit")) or {}
            if tf.get("touches_edge") or (tf.get("clip_frac") or 0) > 0.02:
                note = (f"{clip}: solve content touches the 1.2 m render "
                        "edge -- measured width underestimates the real "
                        "cast, cropped limbs will reappear on the wall")
        except (OSError, json.JSONDecodeError):
            pass
        ms = [measure(p2, True)
              for p2 in sorted(run.glob(f"frame_*_{ts}/best_shadow.png"))]
        ms = [m for m in ms if m]
        if ms:
            mag = MAG_CLIP_RIG
            offs = [f * SOLVER_SPAN / mag for f, w in ms]
            width = float(np.median([w for f, w in ms])) * SOLVER_SPAN / mag
            return offs, width, note
    rows = {json.loads(l)["id"]: json.loads(l)
            for l in open(BENCH / "metadata.jsonl", encoding="utf-8")}
    if clip in rows:                        # a library id
        rec = rows[clip]
        p2 = (BENCH / "optimized" / "big-budget-grounded" / rec["subset"]
              / Path(rec["target"]).stem
              / (Path(rec["target"]).stem + "_best.png"))
        if p2.exists():
            m = measure(p2, False)
            if m:
                mag = MAG_LIB_RIG
                return ([m[0] * SOLVER_SPAN / mag],
                        m[1] * SOLVER_SPAN / mag, None)
    return [0.0], 0.32, None                # unknown: centred, generic arm


def canvas_centroids(sid: str):
    """Per-frame authored centroid x/y (canvas px), median bbox width px,
    and THIS sequence's canvas width -- pixar footage is 1280 wide, icra
    1640, family 1920; a hard-coded 1920 shoved every pixar trio 1.67 m
    off centre and inflated its screen share (the audit's finding #2)."""
    seq = BENCH / "sequences" / sid
    src = json.loads((seq / "source.json").read_text(encoding="utf-8"))
    crop = src["crop"]
    cw = float((src.get("canvas") or {}).get("w") or 1920)
    side = crop["pad_side"]
    ox = (side - crop["w"]) // 2
    oy = (side - crop["h"]) // 2
    cxs, cys, widths = [], [], []
    last, lasty = cw / 2, 540.0
    for fp in sorted(seq.glob("f*.png")):
        au = np.array(Image.open(fp).convert("L")) < 128
        ys, xs = np.where(au)
        if len(xs):
            k = side / au.shape[0]
            last = crop["x"] + xs.mean() * k - ox
            lasty = crop["y"] + ys.mean() * k - oy
            widths.append((xs.max() - xs.min() + 1) * k)
        cxs.append(last)
        cys.append(lasty)
    return cxs, cys, (float(np.median(widths)) if widths else 100.0), cw


def clip_for(sid: str, ass: dict) -> str | None:
    """Which choreography plays this element — same rules as pack.py."""
    mode = ass.get("mode")
    if mode == "library" and ass.get("library_id"):
        return ass["library_id"]
    if mode == "sequence" and ass.get("donor"):
        return ass["donor"]
    use = sid + "_stab" if (BENCH / "optimized" / (sid + "_stab")).is_dir() else sid
    if (BENCH / "optimized" / use).is_dir():
        return use
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--scene", default=None, help="one scene only (scene_NN)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    ass_all = json.loads((ROOT / "projects" / a.project / "assignments.json")
                         .read_text(encoding="utf-8"))
    scenes: dict[str, list] = {}
    for sid in sorted(ass_all):
        scene = "scene_" + sid.split("_scene_")[1].split("_")[0]
        if a.scene and scene != a.scene:
            continue
        scenes.setdefault(scene, []).append(sid)
    # one trio per ELEMENT: a _stab variant on the assignment board is the
    # same element as its parent (clip_for already resolves the parent to
    # the stab solve) -- both present meant two trios on one spot, twin
    # arms and twin shadows (family scene_06 grew L and L_stab rigs)
    for scene, sids in scenes.items():
        scenes[scene] = [x for x in sids
                         if not (x.endswith("_stab") and x[:-5] in sids)]

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for scene, sids in sorted(scenes.items()):
        entries, id_sets = [], []
        for sid in sids:
            # geometry and timing ALWAYS come from the parent footage: a
            # _stab sid's sequence is stabilised -- travel removed, content
            # centred -- so placing from it would erase the very motion the
            # ensemble exists to restore
            ass_sid = sid
            if sid.endswith("_stab") and (BENCH / "sequences" / sid[:-5]
                                          / "source.json").exists():
                sid = sid[:-5]
            seq = BENCH / "sequences" / sid
            if not (seq / "source.json").is_dir() and not (seq / "source.json").exists():
                print(f"  {sid}: no authored sequence, skipped")
                continue
            src = json.loads((seq / "source.json").read_text(encoding="utf-8"))
            clip = clip_for(sid, ass_all[ass_sid])
            if clip is None:
                print(f"  {sid}: nothing assigned/solved to play, skipped")
                continue
            ids = [int(f[1:]) for f in src["frame_ids"]]
            step = int(src.get("sample_every") or 5)
            id_sets.append((sid, clip, ids, step, src))
        if not id_sets:
            continue
        t0 = min(ids[0] for _, _, ids, _, _ in id_sets)
        # Per-group optics, the owner's design: every trio carries ITS OWN
        # point light rigidly (A_RIG behind the mid arm, 0.30 m above base
        # height), all groups share ONE screen. Sliding the rigid rig toward
        # or away from the wall changes only the magnification -- same rays,
        # longer throw -- so shadow SIZE becomes a solved quantity:
        #   m_i = m_ref * wanted_width / measured_width, depth = f(m_i)
        # and lateral solves the wanted centre given the solve's own offset.
        A_RIG = Z_MID - SCENE["lightX"]
        M_COMFORT = 2.0                  # the median group sits here
        geo = {}
        for sid, clip, ids, step, src in id_sets:
            offs, w_arm, note = shadow_geometry(clip)
            if note:
                print(f"  [!] {note}")
            cxs, cys, wpx, cw = canvas_centroids(sid)
            geo[sid] = (offs, w_arm, wpx, cxs, cys, cw)
        # widen the shared screen until the MEDIAN element's film share
        # equals its shadow at the comfort magnification; relative size
        # differences between elements then live in per-group depth
        w_v = float(np.clip(np.median(
            [w_arm * M_COMFORT * cw / max(wpx, 1.0)
             for offs, w_arm, wpx, cxs, cys, cw in geo.values()]),
            SCREEN_W_FILM, 10.0))
        for sid, clip, ids, step, src in id_sets:
            offs, w_arm, wpx, cxs, cys, cw = geo[sid]
            # airborne moments: translating the rigid rig (arms + lamp)
            # up by h raises the shadow by exactly h, so lift is the
            # authored centroid's rise above its own resting baseline,
            # at the SAME metres-per-pixel as the lateral mapping
            mpp = w_v / cw
            # 90th percentile, not max: one squashed frame (the lamp
            # flattening the I) must not hoist every other frame into
            # the air as false "lift"
            base_y = float(np.percentile(cys, 90))
            lift = [max(0.0, (base_y - cy) * mpp) for cy in cys]
            if max(lift, default=0.0) < 0.04:
                lift = [0.0] * len(cys)     # centroid jitter, not a jump
            want_w = wpx / cw * w_v
            m_i = float(np.clip(want_w / max(w_arm, 1e-6),
                                1.7, 6.0))   # floor 1.7: closer than
                                              # ~0.8 m to the wall reads as
                                              # arms standing IN their shadow
            depth = SCENE["screenX"] + A_RIG - A_RIG * m_i
            entries.append({
                "element": sid.rsplit("_", 1)[1],
                "sequence": sid,
                "clip": clip,
                "entry": (ids[0] - t0) // step,
                "n_frames": len(ids),
                "mag": round(m_i, 3),
                "depth_m": round(depth, 3),
                "lat_path_m": [round(((cx / cw) - 0.5) * w_v
                                     - offs[min(k, len(offs) - 1)] * m_i, 4)
                               for k, cx in enumerate(cxs)],
                "lift_path_m": [round(v, 4) for v in lift],
                "light": {"ddepth": -round(A_RIG, 3), "h": 0.30},
            })
        # spot-light physics: a cone wide enough to cover an arm paints a
        # ~1.8 m pool on the wall, so adjacent rigs closer than that dilute
        # each other's shadow to grey. Stretch the whole lateral layout
        # (sizes/depths untouched) until neighbouring trios sit at least
        # MIN_GAP apart -- the stage is allowed to be wide.
        # NO artificial spacing: the footage's relative geometry IS the
        # choreography (scene_02's lamp trio JUMPS ONTO the I -- their
        # laterals must converge). Wide-cone lights overlap where elements
        # stack; that is the physics of the piece, and stacked pairs are
        # naturally de-collided in DEPTH because their magnifications
        # differ. (An equal-spacing relaxation lived here briefly and
        # flattened the narrative -- owner's call: never again.)
        name = f"{a.project}_{scene}"
        n_film = max(e["entry"] + e["n_frames"] for e in entries)
        doc = {
            "name": name, "kind": "ensemble", "fps": 5.0,
            "n_frames": n_film, "scene": SCENE,
            "screen_w_m": round(w_v, 3),
            "per_group_light": True,
            "choreo_head_pad": 30, "interp_factor": 6,
            "default_blend_s": 1.0,
            "entries": entries,
        }
        (out_dir / f"{name}.json").write_text(json.dumps(doc, indent=1),
                                              encoding="utf-8")
        print(f"  {name}: {len(entries)} trios, {n_film} film frames "
              f"-> {out_dir / (name + '.json')}")
        made += 1
    print(f"{made} ensemble manifest(s)")


if __name__ == "__main__":
    main()
