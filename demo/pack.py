#!/usr/bin/env python3
"""pack.py — one demo, one folder: images, video, and what the robots need.

    python demo/pack.py --name family \\
        --clip family_ad_scene_06_A --clip family_ad_scene_06_L_stab \\
        --library teleop_candle_01_mask

Assembly is manual by design; what is NOT manual is hunting through
optimized/ and out/ for the pieces. A package collects, per element,
exactly the three things a show is made of:

    elements/<id>/frames/       canvas PNGs (clips) or the silhouette (library)
    elements/<id>/joints.csv    the robot poses -- one row per frame,
                                arm0_q0..arm2_q5, RADIANS, 3 arms x 6 dof
    elements/<id>/meta.json     fps, source frame ids, lane, fit, provenance
    video/                      the composited scene mp4s, if already built
    package.json + README.md    the inventory and the units caveat

Clips come from demo/out/reassembled/ (run 08_reassemble first) and their
joints from optimized/<id>/frame_*/best_q.npy. Library elements -- the point
of the library is reuse, so a static shape that the benchmark already solved
is pulled, never re-solved -- come from optimized/<sweep>/<subset>/<stem>/:
the best run's q_rad and the rendered silhouette.

Joints are solver-space radians against urdf/SO101/so101_new_calib.urdf with
the solve's own stage layout (recorded in meta.json). The physical-rig unit
mapping and base placement are the deploy gate documented in README section C;
the package carries everything that is knowable without the lab.
"""
from __future__ import annotations

import argparse
import csv
import re
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent

# The solver rig behind every solve in this repo (run logs: light_y=1.00
# wall_y=-2.40, three arms 0.2 apart, light 0.30 high, arm 0 nearest the
# light), mapped into the UI frame per Shadow_robot_ui/render_server/
# optimize.py's documented transform: base.x = -depth, lightH = h + tableY
# (0.55). This is what makes the Play preview's shadow geometry match the
# solve instead of whatever stage the UI happens to have open.
CHOREO_SCENE = {"lightX": -1.0, "lightY": 0, "lightH": 0.85, "screenX": 2.4}
CHOREO_BASES = [{"x": 0.0, "y": 0, "z": 0, "yaw": 0},
                {"x": 0.2, "y": 0, "z": 0, "yaw": 0},
                {"x": 0.4, "y": 0, "z": 0, "yaw": 0}]


def latest_ts(run: Path) -> str | None:
    js = sorted(run.glob("summary_*.json"))
    return js[-1].stem[len("summary_"):] if js else None


def write_joints(path: Path, rows: list[np.ndarray]):
    n_arms = rows[0].size // 6
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame"] + [f"arm{a}_q{j}" for a in range(n_arms)
                                for j in range(6)])
        for i, q in enumerate(rows):
            w.writerow([i] + [f"{v:.6f}" for v in np.asarray(q).ravel()])


def pack_clip(sid: str, dest: Path):
    re_dir = ROOT / "out" / "reassembled" / sid
    run = BENCH / "optimized" / sid
    if not (re_dir / "reassembly.json").exists():
        sys.exit(f"[!] {sid}: no reassembly -- run "
                 f"'python demo/08_reassemble.py --sequence {sid}' first")
    ts = latest_ts(run)
    if ts is None:
        sys.exit(f"[!] {sid}: no solve in optimized/{sid}")
    r = json.loads((re_dir / "reassembly.json").read_text(encoding="utf-8"))
    (dest / "frames").mkdir(parents=True)
    for p in sorted(re_dir.glob("f*.png")):
        shutil.copyfile(p, dest / "frames" / p.name)
    qs = [np.load(p) for p in sorted(run.glob(f"frame_*_{ts}/best_q.npy"))]
    if not qs:
        sys.exit(f"[!] {sid}: no best_q.npy under optimized/{sid}")
    # A held clip repeats one pose; a solved clip must match frame for frame.
    if r.get("held_static") and len(qs) == 1:
        qs = qs * r["n_frames"]
    if len(qs) != r["n_frames"]:
        sys.exit(f"[!] {sid}: {len(qs)} poses for {r['n_frames']} frames")
    write_joints(dest / "joints.csv", qs)
    pack_clip.last_qs, pack_clip.last_fps = qs, None
    summ = json.loads(sorted(run.glob("summary_*.json"))[-1]
                      .read_text(encoding="utf-8"))
    (dest / "meta.json").write_text(json.dumps({
        "kind": "clip", "sequence": sid, "n_frames": r["n_frames"],
        "fps": r["fps"], "source_frame_ids": r.get("source_frame_ids"),
        "held_static": r.get("held_static", False),
        "stabilized_from": r.get("stabilized_from"),
        "trajectory_applied": r.get("trajectory_applied", False),
        "fit_applied_by_solver": r.get("fit_applied_by_solver"),
        "avg_iou": summ.get("avg_iou"), "scene": summ.get("scene"),
        "config": summ.get("config"),
        "urdf": summ.get("urdf"), "n_arms": summ.get("n_robots"),
    }, indent=1), encoding="utf-8")
    pack_clip.last_fps = r["fps"]
    return r["n_frames"]


def pack_library(stem: str, sweep: str, dest: Path):
    live = {json.loads(l)["id"]: json.loads(l)
            for l in open(BENCH / "metadata.jsonl", encoding="utf-8")}
    if stem not in live:
        sys.exit(f"[!] {stem}: not in metadata.jsonl -- the library is the "
                 "benchmark, and this id is not a row of it")
    rec = live[stem]
    subset, tstem = rec["subset"], Path(rec["target"]).stem
    tdir = BENCH / "optimized" / sweep / subset / tstem
    rj = tdir / "results.json"
    if not rj.exists():
        sys.exit(f"[!] {stem}: no solve at {tdir} -- solved sweeps are "
                 f"big-budget-grounded / small-budget-grounded")
    res = json.loads(rj.read_text(encoding="utf-8"))
    best = res["runs"][res["best_run"]]
    (dest / "frames").mkdir(parents=True)
    shutil.copyfile(tdir / f"{tstem}_best.png", dest / "frames" / "silhouette.png")
    shutil.copyfile(BENCH / rec["target"], dest / "frames" / "target.png")
    write_joints(dest / "joints.csv", [np.asarray(best["q_rad"])])
    pack_library.last_qs = [np.asarray(best["q_rad"], dtype=float).ravel()]
    (dest / "meta.json").write_text(json.dumps({
        "kind": "library_static", "id": stem, "class": rec.get("class"),
        "subset": subset, "sweep": sweep,
        "best_iou": res.get("best_iou"), "rig": res.get("rig"),
        "optimizer": res.get("optimizer"),
        "fit": res.get("fit"), "prompt": rec.get("prompt"),
    }, indent=1), encoding="utf-8")
    return 1


CHOREO_HZ = 30


def interp_frames(qs, src_hz: float):
    """Linear per-joint interpolation from solve keyframes up to CHOREO_HZ.

    The deploy loop sends each frame raw at 1/hz with no interpolation of its
    own (verified in render_server/deploy.py), so a 5 Hz clip commands up to
    68 deg of travel in one 200 ms tick -- and the Arrange compiler resamples
    nearest-neighbour, making it 33 ms. Shipping 30 Hz with the solver poses
    as keyframes keeps the same motion, smooth on hardware and in Play.
    """
    qs = [np.asarray(q, dtype=float).ravel() for q in qs]
    if len(qs) < 2:
        return qs
    steps = max(1, int(round(CHOREO_HZ / max(src_hz, 0.1))))
    out = []
    for a, b in zip(qs, qs[1:]):
        for t in range(steps):
            out.append(a + (b - a) * (t / steps))
    out.append(qs[-1])
    return out


def write_choreo(dest: Path, name: str, hz: float, qs: list[np.ndarray]):
    """The unified clip envelope Shadow_robot_ui consumes, one file per element.

    Dropping this into fleet-shadow-art/choreographies/ is the whole deploy
    hand-off: Play lists it by filename stem, previews it against the scene
    block, and streams robots[].frames to the arms as-is -- solver radians,
    uncalibrated (the driver applies per-arm visual offsets at send time).
    Name and filename stem must match and may only use [A-Za-z0-9_-].
    """
    src_hz = hz
    qs = interp_frames(qs, src_hz)
    hz = CHOREO_HZ if len(qs) > 1 else hz
    n_arms = np.asarray(qs[0]).size // 6
    robots = []
    for i in range(n_arms):
        robots.append({
            "id": f"SR{101 + i}", "model": "so-101",
            "base": CHOREO_BASES[i] if i < len(CHOREO_BASES) else
                    {"x": 0.2 * i, "y": 0, "z": 0, "yaw": 0},
            # cutout must stay null: with one set, the UI reinterprets q5 as a
            # rod-servo spin and drives only five joints
            "cutout": None, "cutout_id": None, "cutout_rot": 0, "mount": None,
            "start_frame": 0,
            "frames": [[round(float(v), 6) for v in
                        np.asarray(q).ravel()[i * 6:(i + 1) * 6]] for q in qs],
        })
    clip = {"name": name, "stage": None,
            "source": {"method": "umbra-bench demo/pack.py"},
            "hz": int(round(hz)), "n_frames": len(qs), "kind": "clip",
            "scene": CHOREO_SCENE, "robots": robots}
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{name}.json").write_text(json.dumps(clip), encoding="utf-8")


PKG_README = """# demo package: {name}

Self-contained material for one show. Per element:

* `frames/` -- for a clip: 1-bit 1920x1080 canvas frames, dark = shadow,
  already at authored position/scale; align elements by `source_frame_ids`
  in `meta.json`, never by frame index. For a library element: the solver's
  rendered `silhouette.png` (128 px solver frame -- placement is yours) and
  the authored `target.png`.
* `joints.csv` -- one row per frame: `arm0_q0..arm{{A}}_q5`, **radians**,
  6 dof per arm, against the URDF named in `meta.json` and the stage layout
  recorded there. A library element has a single row (a held pose).
* `meta.json` -- fps, provenance, fit, IoU.

`video/` holds the already-composited scene mp4s if they were built.

## Robot deploy: `choreo/`

One `choreo/<element>.json` per element, in the exact "unified clip envelope"
Shadow_robot_ui consumes. To deploy:

1. copy `choreo/*.json` into `fleet-shadow-art/choreographies/`
2. start the UI (`Shadow_robot_ui/start.sh`) and open **Play** -- each element
   lists by filename; preview, arrange on the timeline, deploy.

The scene block carries the solve's own rig (light 1.0 m in front of arm 0,
arms 0.2 m apart in depth, screen 2.4 m behind, mapped into the UI frame), so
the preview's shadow geometry matches the solve. Joints are UNCALIBRATED
solver radians -- the arm driver applies each robot's visual offsets at send
time; do not pre-apply them. Serial ports are an operator setting in the
Robot console, never part of a clip. Robot ids are SR101 (nearest the light),
SR102, SR103; if your stage config names differ, Play remaps positionally and
says so in the status bar.

**Deploy caveat (unchanged from demo/README.md section C):** solver joints are
model-space. Physical playback still needs the lab's base placement, the
Play/render_server stage JSON, and the SR10x unit mapping -- the three
TODO-user facts. Everything knowable without the lab is in this folder.

Combine shadows by union (`np.minimum` on greyscale). Manual assembly is the
intended path.
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--clip", action="append", default=[],
                    help="sequence id (reassembled + solved)")
    ap.add_argument("--library", action="append", default=[],
                    help="metadata.jsonl id of a static library element")
    ap.add_argument("--sweep", default="big-budget-grounded",
                    help="which sweep a library element's solve is pulled from")
    ap.add_argument("--no-video", action="store_true",
                    help="skip copying demo/out/video into the package")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing package of this name")
    a = ap.parse_args()
    if not a.clip and not a.library:
        ap.error("an empty show: give --clip and/or --library")

    pkg = ROOT / "packages" / a.name
    if pkg.exists():
        if not a.force:
            sys.exit(f"[!] {pkg} exists -- --force to overwrite")
        shutil.rmtree(pkg)

    elements = {}
    for sid in a.clip:
        n = pack_clip(sid, pkg / "elements" / sid)
        write_choreo(pkg / "choreo", sid, pack_clip.last_fps or 5.0,
                     pack_clip.last_qs)
        elements[sid] = {"kind": "clip", "n_frames": n}
        print(f"  clip     {sid:<28} {n} frames + choreo")
    # several elements may reuse the same library entry; pack it once
    seen_lib = set()
    a.library = [x for x in a.library
                 if not (x in seen_lib or seen_lib.add(x))]
    for lid in a.library:
        n = pack_library(lid, a.sweep, pkg / "elements" / lid)
        write_choreo(pkg / "choreo", lid, 5.0, pack_library.last_qs)
        elements[lid] = {"kind": "library_static", "sweep": a.sweep}
        print(f"  library  {lid:<28} 1 pose ({a.sweep}) + choreo")

    # Scene replay: one UI arrangement per scene, elements offset by when
    # they enter the shot (source frame ids at the footage's 25 fps). Copy
    # these into fleet-shadow-art/arrangements/ next to the choreo files --
    # Play's Arrange loader compiles them into a single timeline.
    by_scene = {}
    for sid in a.clip:
        mj = pkg / "elements" / sid / "meta.json"
        if not mj.exists():
            continue
        meta = json.loads(mj.read_text(encoding="utf-8"))
        ids = meta.get("source_frame_ids") or []
        m = re.search(r"_(scene_\d+)", sid)
        if not (ids and m):
            continue
        by_scene.setdefault(m.group(1), []).append(
            (sid, int(str(ids[0]).lstrip("f"))))
    import re as _re
    for scene, entries in sorted(by_scene.items()):
        first = min(f for _, f in entries)
        arr = {"name": f"{a.name}_{scene}",
               "segs": [{"name": sid, "offset": round((f - first) / 25.0, 3)}
                        for sid, f in sorted(entries, key=lambda t: t[1])],
               "musicOffset": 0.0}
        d = pkg / "choreo" / "arrangements"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{a.name}_{scene}.json").write_text(json.dumps(arr, indent=1),
                                                  encoding="utf-8")
    if by_scene:
        print(f"  arrange  {len(by_scene)} scene timeline(s) -> choreo/arrangements/")

    vids = [] if a.no_video else sorted((ROOT / "out" / "video").glob("*.mp4"))
    if vids:
        (pkg / "video").mkdir()
        for v in vids:
            shutil.copyfile(v, pkg / "video" / v.name)
        print(f"  video    {len(vids)} mp4(s) from demo/out/video")

    (pkg / "package.json").write_text(json.dumps({
        "name": a.name, "elements": elements,
        "video": [v.name for v in vids],
    }, indent=1), encoding="utf-8")
    (pkg / "README.md").write_text(PKG_README.format(name=a.name),
                                   encoding="utf-8")
    total = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file())
    print(f"\npackage -> {pkg}  ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
