#!/usr/bin/env python3
"""Temporal metrics for the sequences track -- S1/S2/S3 from SEQUENCES.md §3.

Why per-frame IoU is not enough: the already-solved `spinning_star` clip in
fleet-shadow-art averages 0.6679 IoU per frame -- indistinguishable from a good
static result -- while requiring a 291° single-joint swing between two of its
frames. Every frame is individually good and the clip cannot be performed.
Averaging per-frame IoU is precisely the statistic that cannot see this, so this
script reports a *set* of numbers, and `mean_frame_iou` is never to be quoted
without `dq_infeasible_frac` beside it.

Three metric groups, reported together rather than collapsed:

  S1  per-frame quality      every metric in metrics.py, aggregated over frames
                             as mean, min, std (and max -- for the metrics where
                             lower is better, the worst frame is the max).
                             `min` is not a footnote: a viewer watching a clip
                             sees the worst frame, not the average.
  S2  transitions            over the N-1 interior steps PLUS the wrap when the
                             sequence loops: dq_max_deg, dq_l2_deg,
                             dq_infeasible_frac, shadow_step_iou, motion_excess.
                             The infeasibility threshold is
                             motion_planner.LARGE_Q_JUMP (1.2 rad), imported
                             from fleet-shadow-art rather than copied here.
  S3  sequence level         assignment_stability, loop_closure,
                             total_path_length.

The wrap matters: if it is not scored on a looping sequence, a solver can unwind
a full rotation between the last frame and the first and pay nothing. Whenever
`target_motion.loop` is true the wrap is in the transition list, labelled
"wrap", and it counts toward `dq_infeasible_frac` like any interior step.

`assignment_stability` is computed in joint space: for each transition, the
Hungarian assignment between the arms' 6-D joint vectors at t and t+1. If the
optimal assignment is not the identity, relabelling the arms explains the motion
better than the arms' own paths do -- the arms traded roles, which on screen
reads as arms crossing and which per-frame IoU is completely blind to.

Output follows compute_metrics.py: one wide CSV in `results/`, rows tagged by
granularity the way that script tags rows by `ref` -- one `row=frame` line per
frame (its S1 metrics, plus the transition *leaving* it) and one `row=aggregate`
line per (sequence, source) with the S1 aggregates, S2 aggregates and S3.

Two input shapes:

    # a fleet-shadow-art clip run (summary_*.csv + frame_*/best_shadow.png):
    python scripts/sequence_metrics.py \
        --run ../fleet-shadow-art/motion-aware-shadow/results/3-robot-runs/spinning_star \
        --sequence star_spin

    # every filled shadows.* slot in sequences.jsonl:
    python scripts/sequence_metrics.py

Optional dependencies degrade to None rather than raising, matching metrics.py:
no joints -> the dq/assignment columns are null and the mask-side columns still
fill; no reachable fleet-shadow-art -> dq_infeasible columns are null and a
warning says so.
"""

import argparse
import ast
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import all_metrics, iou  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)


def _default_repo() -> str:
    """Where fleet-shadow-art lives -- same search order as run_base_optimizer.py."""
    for cand in (
        os.environ.get("FLEET_SHADOW_ART"),
        os.path.join(os.path.dirname(_BENCH), "fleet-shadow-art"),
        os.path.expanduser("~/dev/fleet-shadow-art"),
    ):
        if cand and os.path.isdir(os.path.join(cand, "motion-aware-shadow")):
            return cand
    return os.path.expanduser("~/dev/fleet-shadow-art")


def large_q_jump_deg(repo: str) -> float | None:
    """`motion_planner.LARGE_Q_JUMP` in degrees, read from the repo.

    A real import is tried first so the value can never drift from what the
    planner enforces. motion_planner imports mujoco at module level, which the
    eval env deliberately does not carry (SETUP.md), so on ImportError the
    constant is read from the module's source with `ast` instead -- still the
    repo's value, never a copy kept here.
    """
    ms = os.path.join(repo, "motion-aware-shadow")
    try:
        sys.path.insert(0, ms)
        from motion_planner import LARGE_Q_JUMP  # type: ignore
        return math.degrees(float(LARGE_Q_JUMP))
    except ImportError:
        pass
    finally:
        if ms in sys.path:
            sys.path.remove(ms)
    src = os.path.join(ms, "motion_planner.py")
    try:
        with open(src, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "LARGE_Q_JUMP":
                        return math.degrees(float(ast.literal_eval(node.value)))
    except OSError:
        pass
    return None


# --- io -----------------------------------------------------------------------

def load_mask(path: str, size: int | None = None) -> np.ndarray:
    """A mask as uint8 {0,1}, ink = 1, whichever way the file stores it.

    umbra-bench PNGs are dark shape on white; fleet-shadow-art's best_shadow.png
    is white shadow on black. Decided per file by the corner heuristic
    run_base_optimizer.load_target uses: if three or more corners are "ink", the
    dark region is really the background and the mask is flipped.
    """
    img = Image.open(path).convert("L")
    if size:
        img = img.resize((size, size), Image.NEAREST)
    m = np.array(img) < 128
    corners = int(m[0, 0]) + int(m[0, -1]) + int(m[-1, 0]) + int(m[-1, -1])
    if corners >= 3:
        m = ~m
    return m.astype(np.uint8)


def _read_sequence_index(bench: str, path: str) -> dict:
    out = {}
    fp = os.path.join(bench, path)
    if not os.path.exists(fp):
        return out
    with open(fp, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def _read_mas_run(run_dir: str) -> dict:
    """A fleet-shadow-art clip run: summary_*.csv beside frame_*/best_shadow.png.

    The newest summary wins when there are several. Joints come from the
    q_r{R}_j{J}_deg columns, already in degrees; shadows from each frame's
    best_shadow.png; the recorded per-frame IoU rides along as iou_reported,
    the way compute_metrics.py carries best_iou_reported.
    """
    # Absolute, so the shadow paths recorded in the CSV survive a change of
    # working directory between scoring and payload building.
    run_dir = os.path.abspath(run_dir)
    summaries = sorted(glob.glob(os.path.join(run_dir, "summary_*.csv")))
    if not summaries:
        raise FileNotFoundError(f"no summary_*.csv in {run_dir}")
    summary = summaries[-1]
    ts = re.search(r"summary_(.+)\.csv$", os.path.basename(summary)).group(1)
    # run_sequence.py fits ONE similarity transform per clip; when it did, the
    # recorded per-frame iou is vs the FITTED target while our recomputed iou
    # is vs the authored frames -- both real, and far apart for a big fit, so
    # the transform rides along for the aggregate row.
    fit = None
    sjson = summary[:-4] + ".json"
    if os.path.exists(sjson):
        try:
            with open(sjson, encoding="utf-8") as f:
                fit = json.load(f).get("target_fit")
        except Exception:
            fit = None
    with open(summary, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    qcols = {}
    for c in rows[0]:
        m = re.match(r"^q_r(\d+)_j(\d+)_deg$", c)
        if m:
            qcols[(int(m.group(1)), int(m.group(2)))] = c
    n_arms = 1 + max(r for r, _ in qcols) if qcols else 0
    n_dof = 1 + max(j for _, j in qcols) if qcols else 0
    frames = []
    for row in rows:
        i = int(row["frame"])
        q = None
        if qcols:
            q = np.array([[float(row[qcols[(r, j)]]) for j in range(n_dof)]
                          for r in range(n_arms)])
        shadow = os.path.join(run_dir, f"frame_{i:02d}_{ts}", "best_shadow.png")
        frames.append({
            "target_name": row.get("target"),
            "iou_reported": float(row["iou"]) if row.get("iou") else None,
            "shadow_path": shadow if os.path.exists(shadow) else None,
            "q": q,
        })
    return {"name": os.path.basename(os.path.normpath(run_dir)), "frames": frames,
            "fit": fit}


def _joints_array(joints) -> np.ndarray | None:
    """Normalise a record's per-frame joints to (n_arms, n_dof) degrees.

    Accepts one frame's entry as either a flat list (all arms concatenated,
    6 dof per arm) or a list of per-arm lists.
    """
    if joints is None:
        return None
    a = np.asarray(joints, dtype=float)
    if a.ndim == 1:
        if a.size % 6:
            return None
        a = a.reshape(-1, 6)
    return a


# --- the three groups ---------------------------------------------------------

def _transition_list(n: int, loop: bool) -> list[tuple[int, int, str]]:
    """(from, to, label) for the N-1 interior steps, plus the wrap when looping."""
    t = [(i, i + 1, f"{i}->{i + 1}") for i in range(n - 1)]
    if loop:
        t.append((n - 1, 0, "wrap"))
    return t


def _dq(qa: np.ndarray | None, qb: np.ndarray | None, thr_deg: float | None) -> dict:
    if qa is None or qb is None:
        return {"dq_max_deg": None, "dq_l2_deg": None, "dq_infeasible": None}
    d = np.abs(qb - qa)
    out = {"dq_max_deg": round(float(d.max()), 2),
           "dq_l2_deg": round(float(np.sqrt((d ** 2).sum())), 2)}
    out["dq_infeasible"] = (None if thr_deg is None
                            else int(float(d.max()) > thr_deg))
    return out


def _assignment_stable(qa: np.ndarray | None, qb: np.ndarray | None) -> bool | None:
    """True when the identity arm assignment is already the optimal one."""
    if qa is None or qb is None or len(qa) < 2:
        return None
    cost = np.linalg.norm(qa[:, None, :] - qb[None, :, :], axis=2)
    _, ci = linear_sum_assignment(cost)
    return bool((ci == np.arange(len(qa))).all())


def _agg(vals: list, prefix: str) -> dict:
    """mean/min/max/std of the non-null values, or nulls when there are none."""
    v = np.array([x for x in vals if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return {f"{prefix}_mean": None, f"{prefix}_min": None,
                f"{prefix}_max": None, f"{prefix}_std": None}
    return {f"{prefix}_mean": round(float(v.mean()), 4),
            f"{prefix}_min": round(float(v.min()), 4),
            f"{prefix}_max": round(float(v.max()), 4),
            f"{prefix}_std": round(float(v.std()), 4)}


def score_sequence(seq_id: str, source: str, shadow_paths: list, q_frames: list,
                   target_paths: list, loop: bool, size: int,
                   thr_deg: float | None, iou_reported: list | None = None,
                   fit: dict | None = None) -> list[dict]:
    """All three groups for one (sequence, source) pair -> its CSV rows."""
    n = len(shadow_paths)
    if target_paths and len(target_paths) != n:
        print(f"[!] {seq_id}/{source}: {len(target_paths)} target frames vs "
              f"{n} shadow frames; scoring the first {min(len(target_paths), n)}")
        n = min(len(target_paths), n)

    # Working resolution: the shadows' own, unless --size overrides. Scoring at
    # native size reproduces a run's recorded IoU exactly; downsampling first
    # does not (verified on spinning_star: 0.6468 native vs 0.6485 at 128).
    if not size:
        with Image.open(shadow_paths[0]) as im:
            size = im.size[0]

    shadows = [load_mask(p, size) if p else None for p in shadow_paths[:n]]
    targets = [load_mask(p, size) for p in target_paths[:n]] if target_paths else []
    qs = [(_joints_array(q_frames[i]) if q_frames else None) for i in range(n)]

    stub = {"sequence_id": seq_id, "source": source, "n_frames": n,
            "loop": int(loop), "size": size}

    # S1 -- per-frame quality
    frame_rows, per_metric = [], {}
    for i in range(n):
        row = dict(stub, row="frame", frame_idx=i)
        # The shadow's path rides along so downstream consumers (the atlas
        # sequences payload) can find the frames without re-deriving run-dir
        # layout knowledge the CSV already had at scoring time.
        if shadow_paths[i]:
            row["shadow"] = str(shadow_paths[i]).replace("\\", "/")
        if iou_reported and i < len(iou_reported):
            row["iou_reported"] = iou_reported[i]
        if targets and shadows[i] is not None:
            m = all_metrics(targets[i], shadows[i])
            row.update(m)
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    per_metric.setdefault(k, []).append(v)
        frame_rows.append(row)

    # S2 -- transitions, the wrap included whenever the sequence loops
    transitions = []
    for a, b, label in _transition_list(n, loop):
        t = {"label": label, **_dq(qs[a], qs[b], thr_deg)}
        t["shadow_step_iou"] = (round(iou(shadows[a], shadows[b]), 4)
                                if shadows[a] is not None and shadows[b] is not None
                                else None)
        t["target_step_iou"] = (round(iou(targets[a], targets[b]), 4)
                                if targets else None)
        t["motion_excess"] = (round(t["target_step_iou"] - t["shadow_step_iou"], 4)
                              if t["target_step_iou"] is not None
                              and t["shadow_step_iou"] is not None else None)
        t["assignment_stable"] = _assignment_stable(qs[a], qs[b])
        transitions.append(t)
        # the transition rides on the frame it leaves
        frame_rows[a].update({"step_to": label, **{k: v for k, v in t.items()
                                                   if k not in ("label", "assignment_stable")}})

    # S3 + aggregates -> the one aggregate row
    agg = dict(stub, row="aggregate", n_transitions=len(transitions))
    if iou_reported:
        agg.update(_agg(iou_reported, "iou_reported"))
    for k, vals in per_metric.items():
        agg.update(_agg(vals, k))
    for k in ("dq_max_deg", "dq_l2_deg", "shadow_step_iou",
              "target_step_iou", "motion_excess"):
        agg.update(_agg([t[k] for t in transitions], k))
    feas = [t["dq_infeasible"] for t in transitions if t["dq_infeasible"] is not None]
    agg["dq_infeasible_frac"] = (round(sum(feas) / len(feas), 4) if feas else None)
    agg["dq_infeasible_thr_deg"] = round(thr_deg, 2) if thr_deg is not None else None
    if fit:
        # the clip-level similarity transform the run solved against, when any:
        # it is why iou (vs authored frames) and iou_reported (vs the fitted
        # target) can be far apart, and both belong in the row
        for k in ("scale", "dx", "dy", "clip_frac", "at_bound"):
            if k in fit:
                agg[f"fit_{k}"] = fit[k]

    stable = [t["assignment_stable"] for t in transitions
              if t["assignment_stable"] is not None]
    agg["assignment_stability"] = (round(sum(stable) / len(stable), 4)
                                   if stable else None)
    agg["n_arm_swaps"] = (len(stable) - sum(stable)) if stable else None
    path = [t["dq_l2_deg"] for t in transitions if t["dq_l2_deg"] is not None]
    agg["total_path_length_deg"] = round(sum(path), 2) if path else None
    if loop and transitions and transitions[-1]["label"] == "wrap":
        w = transitions[-1]
        agg["loop_closure_dq_max_deg"] = w["dq_max_deg"]
        agg["loop_closure_shadow_step_iou"] = w["shadow_step_iou"]
        agg["loop_closure_target_step_iou"] = w["target_step_iou"]
    # interior steps as one cell, for eyeballing a clip without the frame rows
    agg["dq_max_deg_by_transition"] = json.dumps(
        [t["dq_max_deg"] for t in transitions if t["label"] != "wrap"])
    return frame_rows + [agg]


# --- drivers ------------------------------------------------------------------

def rows_from_run(a, seq_index: dict, thr_deg: float | None) -> list[dict]:
    run = _read_mas_run(a.run)
    rec = seq_index.get(a.sequence) if a.sequence else None
    if a.sequence and rec is None:
        print(f"[!] sequence '{a.sequence}' not in sequences.jsonl; "
              "scoring without targets")
    target_paths, loop = [], False
    if rec:
        target_paths = [os.path.join(a.bench, p) for p in rec["frames"]]
        loop = bool(rec["target_motion"]["loop"])
    seq_id = a.sequence or run["name"]
    return score_sequence(
        seq_id, a.source,
        [f["shadow_path"] for f in run["frames"]],
        [f["q"] for f in run["frames"]],
        target_paths, loop, a.size, thr_deg,
        iou_reported=[f["iou_reported"] for f in run["frames"]],
        fit=run.get("fit"))


def rows_from_static_sweep(a, seq_index: dict, thr_deg: float | None) -> list[dict]:
    """Score a run_base_optimizer sweep over `targets_sequences/` as clips.

    The frame-independent solve is the baseline the temporal metrics exist to
    condemn: nothing couples consecutive frames, so per-frame IoU comes out
    healthy and the transitions do not. Each frame dir carries results.json
    with the best run's q_rad (flat, 6 dof per arm, radians) -- everything S2
    needs -- and <stem>_best.png for the mask side. Emitted with
    source="optimizer_static" so it sits beside a sequence-aware "optimizer"
    solve of the same clip rather than overwriting it.
    """
    rows = []
    sweep = os.path.abspath(a.static_sweep)   # CWD-independent CSV paths
    for sid, rec in seq_index.items():
        if a.sequence and sid != a.sequence:
            continue
        d = os.path.join(sweep, sid)
        if not os.path.isdir(d):
            continue
        stems = [os.path.splitext(os.path.basename(p))[0] for p in rec["frames"]]
        shadow_paths, q_frames, iou_rep = [], [], []
        for stem in stems:
            fd = os.path.join(d, stem)
            shadow = os.path.join(fd, f"{stem}_best.png")
            shadow_paths.append(shadow if os.path.exists(shadow) else None)
            q = best = None
            rj = os.path.join(fd, "results.json")
            if os.path.exists(rj):
                with open(rj, encoding="utf-8") as f:
                    r = json.load(f)
                best = r.get("best_iou")
                runs = r.get("runs") or []
                if runs:
                    q_rad = runs[r.get("best_run", 0)].get("q_rad")
                    if q_rad is not None:
                        q = [math.degrees(v) for v in q_rad]
            q_frames.append(q)
            iou_rep.append(best)
        if not any(shadow_paths):
            print(f"[!] {sid}: no solved frames under {d}; skipped")
            continue
        rows += score_sequence(
            sid, "optimizer_static", shadow_paths, q_frames,
            [os.path.join(a.bench, p) for p in rec["frames"]],
            bool(rec["target_motion"]["loop"]), a.size, thr_deg,
            iou_reported=iou_rep)
    return rows


def rows_from_index(a, seq_index: dict, thr_deg: float | None) -> list[dict]:
    rows = []
    for sid, rec in seq_index.items():
        for src, cap in (rec.get("shadows") or {}).items():
            if not (cap and cap.get("frames")):
                continue
            rows += score_sequence(
                sid, src,
                [os.path.join(a.bench, p) for p in cap["frames"]],
                cap.get("joints"),
                [os.path.join(a.bench, p) for p in rec["frames"]],
                bool(rec["target_motion"]["loop"]), a.size, thr_deg)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default=_BENCH)
    p.add_argument("--repo", default=_default_repo(),
                   help="fleet-shadow-art checkout; source of LARGE_Q_JUMP")
    p.add_argument("--run", default=None,
                   help="a fleet-shadow-art clip run dir "
                        "(summary_*.csv + frame_*/best_shadow.png)")
    p.add_argument("--static-sweep", default=None,
                   help="a run_base_optimizer sweep over targets_sequences/ "
                        "(<sequence>/<fXX>/); scored as the frame-independent "
                        "baseline, source=optimizer_static")
    p.add_argument("--sequence", default=None,
                   help="sequences.jsonl id the --run solves, for targets + loop")
    p.add_argument("--source", default="optimizer")
    p.add_argument("--seq-index", default="sequences.jsonl")
    p.add_argument("--size", type=int, default=0,
                   help="working resolution; 0 = the shadows' native size")
    p.add_argument("--tag", default=None, help="output filename tag")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    thr_deg = large_q_jump_deg(a.repo)
    if thr_deg is None:
        print(f"[!] LARGE_Q_JUMP unreadable from {a.repo}; "
              "dq_infeasible columns will be null")

    seq_index = _read_sequence_index(a.bench, a.seq_index)
    if a.run:
        rows = rows_from_run(a, seq_index, thr_deg)
    elif a.static_sweep:
        rows = rows_from_static_sweep(a, seq_index, thr_deg)
    else:
        rows = rows_from_index(a, seq_index, thr_deg)
    if not rows:
        print("[!] nothing to score: no --run or --static-sweep given and no "
              "shadows.*.frames filled in sequences.jsonl")
        return

    out_dir = a.out or os.path.join(a.bench, "results")
    os.makedirs(out_dir, exist_ok=True)
    tag = a.tag or (os.path.basename(os.path.normpath(a.run)) if a.run
                    else os.path.basename(os.path.normpath(a.static_sweep))
                    if a.static_sweep else "shadows")
    path = os.path.join(out_dir, f"sequence_metrics_{tag}.csv")

    cols = []
    for r in rows:  # union of keys, first-seen order
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[sequence-metrics] -> {path}  ({len(rows)} rows x {len(cols)} cols)")

    for r in rows:
        if r["row"] != "aggregate":
            continue
        print(f"  [{r['sequence_id']}/{r['source']}]  n_frames={r['n_frames']} "
              f"loop={r['loop']}")
        for k in ("iou_reported_mean", "iou_mean", "iou_min", "dq_max_deg_max",
                  "dq_infeasible_frac", "assignment_stability",
                  "total_path_length_deg"):
            if r.get(k) is not None:
                print(f"    {k:<22} {r[k]}")
        if r.get("dq_max_deg_by_transition"):
            print(f"    {'dq_max_deg (interior)':<22} {r['dq_max_deg_by_transition']}")


if __name__ == "__main__":
    main()
