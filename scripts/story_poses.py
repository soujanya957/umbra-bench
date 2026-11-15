"""Keep the good story poses: one registry, tracked in git (story/best_poses.json).

Each entry is a solved keyframe pose with everything needed to cast it again: the rig (bench
rig: n_robots + arm_gap, or a stage json), the solver settings, the joint angles, the fit
that placed the target, the IoU, and where its target and comparison images are. `approved`
marks the ones Ivy picked; the rest are kept as candidates.

    python scripts/story_poses.py add <run_dir> --name 2_edge --keyframe c02_left-1-2/kf_27 [--approved]
    python scripts/story_poses.py add-quality            # every run under results/icra_story/quality
    python scripts/story_poses.py list
    python scripts/story_poses.py export [--all]        # into the Shadow UI's pose library

export writes the approved poses and the best candidate per keyframe (all with --all) to
fleet-shadow-art/leRobot-control/poses/story_<name>.json, the folder the UI's Record page
(Teleop Â· pose) loads from. Arms carry no id, so they land on the stage's arms in order. These
were solved on the bench rig (3 arms side by side, light centred): on another stage the
joints are the same but the shadow is not.
"""

import argparse
import glob
import json
import os
import re
import shutil

import numpy as np

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(BENCH, "story", "best_poses.json")
IMG = os.path.join(BENCH, "results", "icra_story", "best_poses")
QUALITY = os.path.join(BENCH, "results", "icra_story", "quality")
# results/icra_story/quality/<variant>/<digit> -> its keyframe target (tight crop)
QUALITY_KF = {"2": "c02_left-1-2/kf_27", "3": "c07_3/kf_01", "1_italic": "c01_left-1/kf_00",
              "1_upright": "c03_1/kf_00", "1_lean": "c02_right-1/kf_01", "1_walk": "c10_1/kf_02"}
APPROVED = {("edge", "2"), ("edge", "3")}  # Ivy, 2026-11-14: "edge çš„ 2 å’Œ 3 éƒ½å¯ä»¥"


def load():
    try:
        return json.load(open(REG, encoding="utf-8"))
    except FileNotFoundError:
        return {"about": "Story keyframe poses worth keeping. q_rad per arm, 6 joints each.", "poses": {}}


def save(reg):
    os.makedirs(os.path.dirname(REG), exist_ok=True)
    json.dump(reg, open(REG, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def entry(run_dir, keyframe, approved):
    s = sorted(glob.glob(os.path.join(run_dir, "summary_*.json")))[-1]
    d = json.load(open(s, encoding="utf-8"))
    fr = d["frames"][0]
    log = open(os.path.join(run_dir, "solve.log"), encoding="utf-8", errors="ignore").read()
    cmd = re.search(r"run_sequence\.py.*", log)
    args = json.load(open(s, encoding="utf-8")).get("config", {})
    q = np.load(os.path.join(fr["outdir"], "best_q.npy")).tolist()
    return {
        "keyframe": keyframe,
        "approved": approved,
        "iou_vs_fitted_target": fr["iou"],
        "rig": d.get("scene"),
        "n_robots": d.get("n_robots"),
        "solver_config": args,
        "target_fit": d.get("target_fit"),
        "q_rad": [round(float(x), 6) for x in q],
        "per_arm": [{"label": r["label"], "q_rad": r["q_rad"]} for r in fr.get("robots", [])],
        "carriage_s": fr.get("carriage_s"),
        "run_dir": os.path.relpath(run_dir, BENCH).replace("\\", "/"),
        "comparison_png": os.path.relpath(os.path.join(fr["outdir"], "final_comparison.png"), BENCH).replace("\\", "/"),
        "solved_at": d.get("timestamp"),
    }


def add(reg, name, run_dir, keyframe, approved):
    e = entry(run_dir, keyframe, approved)
    reg["poses"][name] = e
    os.makedirs(os.path.join(IMG, name), exist_ok=True)
    shutil.copy(os.path.join(BENCH, e["comparison_png"]), os.path.join(IMG, name, "comparison.png"))
    print(f"{name:<24} IoU {e['iou_vs_fitted_target']:.3f}  {'approved' if approved else 'candidate'}")


UI_POSES = os.path.join(os.path.dirname(BENCH), "fleet-shadow-art", "leRobot-control", "poses")


def export(reg, everything):
    best = {}
    for n, e in reg["poses"].items():
        k = e["keyframe"]
        if k not in best or e["iou_vs_fitted_target"] > reg["poses"][best[k]]["iou_vs_fitted_target"]:
            best[k] = n
    pick = [n for n, e in reg["poses"].items() if everything or e["approved"] or best[e["keyframe"]] == n]
    os.makedirs(UI_POSES, exist_ok=True)
    for n in sorted(pick):
        e = reg["poses"][n]
        arms = [{"q_rad": a["q_rad"]} for a in e["per_arm"]]
        out = {"source": "umbra-bench story, bench rig (3 arms side by side, arm gap 0.2 m)",
               "note": f"{e['keyframe']} Â· IoU {e['iou_vs_fitted_target']:.2f}", "approved": e["approved"],
               "arms": arms}
        json.dump(out, open(os.path.join(UI_POSES, f"story_{n}.json"), "w", encoding="utf-8"), indent=2)
        print(f"story_{n}.json  {out['note']}{'  approved' if e['approved'] else ''}")
    print(f"{len(pick)} poses -> {UI_POSES}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["add", "add-quality", "list", "export"])
    p.add_argument("--all", action="store_true", help="export: every pose, not just approved + best")
    p.add_argument("run_dir", nargs="?")
    p.add_argument("--name")
    p.add_argument("--keyframe")
    p.add_argument("--approved", action="store_true")
    a = p.parse_args()
    reg = load()
    if a.cmd == "add":
        add(reg, a.name, a.run_dir, a.keyframe, a.approved)
    elif a.cmd == "add-quality":
        for d in sorted(glob.glob(os.path.join(QUALITY, "*", "*"))):
            v, k = os.path.basename(os.path.dirname(d)), os.path.basename(d)
            if k in QUALITY_KF and glob.glob(os.path.join(d, "summary_*.json")):
                fr = json.load(open(sorted(glob.glob(os.path.join(d, "summary_*.json")))[-1]))["frames"][0]
                if fr["iou"] > 0.05:  # skip the broken --distort runs
                    add(reg, f"{k}__{v}", d, QUALITY_KF[k], (v, k) in APPROVED)
    if a.cmd == "export":
        export(reg, a.all)
    for n, e in sorted(reg["poses"].items()):
        if a.cmd == "list":
            print(f"{n:<24} {e['keyframe']:<22} IoU {e['iou_vs_fitted_target']:.3f}  {'approved' if e['approved'] else ''}")
    save(reg)

