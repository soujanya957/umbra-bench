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
CANVAS_PX = 1920
Z_MID = 0.2                        # trio arms sit at depth 0/0.2/0.4
MAGNIFICATION = (SCENE["screenX"] - SCENE["lightX"]) / (Z_MID - SCENE["lightX"])


def lat_path(sid: str) -> list[float]:
    """Per-authored-frame trio lateral (m) from the element's canvas centroid."""
    seq = BENCH / "sequences" / sid
    src = json.loads((seq / "source.json").read_text(encoding="utf-8"))
    crop = src["crop"]
    side = crop["pad_side"]
    ox = (side - crop["w"]) // 2
    out, last = [], 0.0
    for fp in sorted(seq.glob("f*.png")):
        au = np.array(Image.open(fp).convert("L")) < 128
        ys, xs = np.where(au)
        if len(xs):
            k = side / au.shape[0]
            cx = crop["x"] + xs.mean() * k - ox
            last = ((cx / CANVAS_PX) - 0.5) * SCREEN_W_FILM / MAGNIFICATION
        out.append(round(last, 4))
    return out


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

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for scene, sids in sorted(scenes.items()):
        entries, id_sets = [], []
        for sid in sids:
            seq = BENCH / "sequences" / sid
            if not (seq / "source.json").is_dir() and not (seq / "source.json").exists():
                print(f"  {sid}: no authored sequence, skipped")
                continue
            src = json.loads((seq / "source.json").read_text(encoding="utf-8"))
            clip = clip_for(sid, ass_all[sid])
            if clip is None:
                print(f"  {sid}: nothing assigned/solved to play, skipped")
                continue
            ids = [int(f[1:]) for f in src["frame_ids"]]
            step = int(src.get("sample_every") or 5)
            id_sets.append((sid, clip, ids, step, src))
        if not id_sets:
            continue
        t0 = min(ids[0] for _, _, ids, _, _ in id_sets)
        for sid, clip, ids, step, src in id_sets:
            entries.append({
                "element": sid.rsplit("_", 1)[1],
                "sequence": sid,
                "clip": clip,
                "entry": (ids[0] - t0) // step,
                "n_frames": len(ids),
                "lat_path_m": lat_path(sid),
            })
        name = f"{a.project}_{scene}"
        n_film = max(e["entry"] + e["n_frames"] for e in entries)
        doc = {
            "name": name, "kind": "ensemble", "fps": 5.0,
            "n_frames": n_film, "scene": SCENE,
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
