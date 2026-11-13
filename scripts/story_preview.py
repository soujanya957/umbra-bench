"""Preview a solved story sequence in film time: the joints (and LeSlider carriages) are
interpolated linearly between the solved keyframes, the shadow is rendered on the stage for
every 30 fps frame, and the film is shown beside it.

Timeline: the videos a sequence was marked on, end to end, in film order. Before the first
keyframe and after the last the pose holds. Transitions are not collision-checked yet.

    python scripts/story_preview.py c01_left-1 c02_left-1-2 ...
    python scripts/story_preview.py --match 1        # every sequence whose subject has a '1'
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastshadow  # noqa: E402

BENCH = fastshadow.BENCH
SOLVE = os.path.join(BENCH, "results", "icra_story", "solve")
STORY = r"C:\Users\hexia\Downloads\ICRA_story_direction\ICRA_story_direction"
STAGES = {
    "leslider": os.path.join(BENCH, "results", "fastshadow", "pose_library_leslider", "stage.json"),
    "three_in_a_row": os.path.join(BENCH, "results", "fastshadow", "pose_library_row3", "stage.json"),
}
H = 360  # output height per panel


def renderer(stage_name, size):
    fleet = fastshadow.use_fleet()
    os.chdir(fleet)
    from rails import wrap_renderer
    from renderer import ShadowRenderer, build_scene

    stage = json.load(open(STAGES[stage_name], encoding="utf-8"))
    model, data, n = build_scene("urdf/SO101/so101_new_calib.urdf", len(stage["bases"]), stage=stage)
    mj = ShadowRenderer(model, data, size=size, n_robots=n)
    return wrap_renderer(mj, stage, False)


def keyframe_qs(name):
    s = sorted(glob.glob(os.path.join(SOLVE, name, "solve", "summary_*.json")))[-1]
    d = json.load(open(s, encoding="utf-8"))
    qs = []
    for f in d["frames"]:
        qs.append(np.load(os.path.join(f["outdir"], "best_q.npy")).astype(np.float64))
    return qs, [f["iou"] for f in d["frames"]]


def timeline(run):
    """Videos in film order with their frame counts, and each keyframe's global frame."""
    vids = []
    for f in run["frames"]:
        if f["video"] not in vids:
            vids.append(f["video"])
    caps = {v: cv2.VideoCapture(os.path.join(STORY, *v.split("/"))) for v in vids}
    lens = {v: int(caps[v].get(cv2.CAP_PROP_FRAME_COUNT)) for v in vids}
    off, o = {}, 0
    for v in vids:
        off[v] = o
        o += lens[v]
    kt = [off[f["video"]] + f["frame"] for f in run["frames"]]
    return vids, caps, lens, off, o, kt


def interp(qs, kt, t):
    if t <= kt[0]:
        return qs[0]
    if t >= kt[-1]:
        return qs[-1]
    i = int(np.searchsorted(kt, t, side="right")) - 1
    a = (t - kt[i]) / max(1, kt[i + 1] - kt[i])
    return (1 - a) * qs[i] + a * qs[i + 1]


def preview(run, size):
    qs, ious = keyframe_qs(run["name"])
    r = renderer(run["stage"], size)
    vids, caps, lens, off, total, kt = timeline(run)
    out = os.path.join(SOLVE, run["name"], f"preview_{run['name']}.mp4")
    W_src = None
    writer = None
    kset = {t: i for i, t in enumerate(kt)}
    for t in range(total):
        v = next(v for v in reversed(vids) if off[v] <= t)
        ok, src = caps[v].read()
        if not ok:
            src = np.full((H, H, 3), 255, np.uint8)
        src = cv2.resize(src, (int(src.shape[1] * H / src.shape[0]), H))
        if W_src is None:
            W_src = max(src.shape[1], 480)
        src = cv2.copyMakeBorder(src, 0, 0, 0, max(0, W_src - src.shape[1]), cv2.BORDER_CONSTANT, value=(255, 255, 255))[:, :W_src]
        m = np.asarray(r.get_shadow_mask(interp(qs, kt, t)))
        sh = cv2.resize(((1 - (m > 0.5)) * 255).astype(np.uint8), (H, H), interpolation=cv2.INTER_NEAREST)
        sh = cv2.cvtColor(sh, cv2.COLOR_GRAY2BGR)
        if t in kset:  # keyframe: frame the shadow panel
            cv2.rectangle(sh, (1, 1), (H - 2, H - 2), (42, 120, 214), 3)
            cv2.putText(sh, f"keyframe {kset[t] + 1}  IoU {ious[kset[t]]:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (42, 120, 214), 2)
        bar = np.full((28, W_src + 8 + H, 3), 255, np.uint8)
        cv2.putText(bar, f"{run['name']}  ({run['stage']})   film frame {t}/{total - 1}   film | shadow",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        frame = np.vstack([bar, np.hstack([src, np.full((H, 8, 3), 255, np.uint8), sh])])
        if writer is None:
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    writer.release()
    print(f"{run['name']}: {total} frames, {len(kt)} keyframes -> {out}", flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*")
    p.add_argument("--match", default=None, help="every sequence whose subject contains this")
    p.add_argument("--size", type=int, default=256)
    a = p.parse_args()
    runs = json.load(open(os.path.join(SOLVE, "runs.json"), encoding="utf-8"))
    pick = [r for r in runs if r["name"] in a.names or (a.match and a.match in r["subject"])]
    for run in pick:
        preview(run, a.size)
