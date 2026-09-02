#!/usr/bin/env python3
"""Build `results/sequences_payload.json` -- the atlas's sequences-track slot.

Sibling of `_build_browser_payload.py`, same conventions: 1-bit PNG thumbnails
base64'd at a shared `px`, 4-decimal rounds, compact separators. One record per
sequence in `sequences.jsonl`, carrying three kinds of things:

  * the target side, which never depends on a solve: every frame as a thumbnail
    (ordered -- the atlas animates them at the source fps), `target_motion`, and
    the loop label WITH its provenance. `loop_source` matters in the UI: a
    `wrap-test` label is a heuristic and shows its evidence, a `declared` label
    is a fact from source.json and suppresses the ratio as authority --
    demo_01_scene_05_I's wrap (0.991) would pass the test with no loop present.
  * provenance for the demo family: fps, scene, letter, crop -- read from
    source.json, absent for the generated 13.
  * the solve side, joined from `results/sequence_metrics_*.csv` aggregate rows.
    "Not yet solved" is an explicit state (`solved: []`, `solves: {}`), not an
    empty table: every demo clip is in that state today, and the card must say
    so with the command that changes it rather than render a blank.

The headline block enforces the SEQUENCES.md rule at the data level: `iou_mean`
travels WITH `dq_infeasible_frac`, so a card cannot show one without the other.

    python scripts/_build_sequences_payload.py
"""

import argparse
import base64
import csv
import glob
import io
import json
import os

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)

# What the card shows, in display order. The CSV stays the source of truth for
# everything else; this is a curation, not a schema.
S1 = ["iou", "boundary_iou", "nsd", "cldice", "betti_error", "chamfer",
      "hd95", "pw_h1", "tc_iou"]
S2 = ["dq_max_deg", "dq_l2_deg", "shadow_step_iou", "target_step_iou",
      "motion_excess"]
S3 = ["dq_infeasible_frac", "assignment_stability", "n_arm_swaps",
      "total_path_length_deg"]


def _enc(im: Image.Image) -> str:
    b = io.BytesIO()
    im.convert("1").save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode("ascii")


def thumb(path: str, px: int) -> str:
    """Dark-ink-on-white thumbnail, whichever way the file stores it.

    Targets follow the benchmark convention already; a clip run's
    best_shadow.png is white-shadow-on-black, and the overlay compositor in the
    template classifies pixels by darkness, so polarity is normalised here with
    the corner heuristic sequence_metrics.load_mask uses.
    """
    im = Image.open(path).convert("L").resize((px, px), Image.LANCZOS)
    bw = im.point(lambda v: 255 if v > 127 else 0)
    px_ = bw.load()
    w, h = bw.size
    corners = sum(px_[x, y] < 128 for x, y in
                  ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)))
    if corners >= 3:
        bw = bw.point(lambda v: 255 - v)
    return _enc(bw)


def _f(v):
    """CSV cell -> float | None; empty cells are meaningful nulls, never 0."""
    if v is None or v == "":
        return None
    try:
        return round(float(v), 4)
    except ValueError:
        return None


def _stats(row: dict, metric: str):
    out = {k: _f(row.get(f"{metric}_{k}")) for k in ("mean", "min", "max", "std")}
    return out if any(v is not None for v in out.values()) else None


def read_solves(bench: str):
    """(sequence_id, source) -> {aggregate row, frame rows} from every CSV."""
    agg, frames = {}, {}
    for path in sorted(glob.glob(os.path.join(bench, "results",
                                              "sequence_metrics_*.csv"))):
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                key = (row["sequence_id"], row["source"])
                if row["row"] == "aggregate":
                    row["_csv"] = os.path.basename(path)
                    agg[key] = row
                elif row["row"] == "frame":
                    frames.setdefault(key, []).append(row)
    return agg, frames


def solve_block(row: dict, frame_rows: list, bench: str, px: int) -> dict:
    by_tr = None
    if row.get("dq_max_deg_by_transition"):
        try:
            by_tr = json.loads(row["dq_max_deg_by_transition"])
        except json.JSONDecodeError:
            by_tr = None
    loop_scored = row.get("loop") in ("1", "True", "true")
    s3 = {k: _f(row.get(k)) for k in S3}
    s3["loop_closure"] = ({
        "dq_max_deg": _f(row.get("loop_closure_dq_max_deg")),
        "shadow_step_iou": _f(row.get("loop_closure_shadow_step_iou")),
        "target_step_iou": _f(row.get("loop_closure_target_step_iou")),
    } if loop_scored else None)
    s3["dq_max_deg_by_transition"] = by_tr
    out = {
        "csv": row.get("_csv"),
        "solve_size": _f(row.get("size")),
        "loop_scored": loop_scored,
        "n_transitions": _f(row.get("n_transitions")),
        # mean IoU is never quoted without infeasibility beside it (SEQUENCES.md)
        "headline": {
            "dq_infeasible_frac": _f(row.get("dq_infeasible_frac")),
            "dq_infeasible_thr_deg": _f(row.get("dq_infeasible_thr_deg")),
            "iou_mean": _f(row.get("iou_mean")),
            "iou_min": _f(row.get("iou_min")),
        },
        "s1": {m: st for m in S1 if (st := _stats(row, m))},
        "s2": {m: st for m in S2 if (st := _stats(row, m))},
        "s3": s3,
    }
    if frame_rows:
        ordered = sorted(frame_rows, key=lambda r: int(r["frame_idx"]))
        out["frames"] = [{
            "i": int(r["frame_idx"]),
            "iou": _f(r.get("iou")) if r.get("iou") not in (None, "")
                   else _f(r.get("iou_reported")),
            "dq_max_deg": _f(r.get("dq_max_deg")),
            "dq_infeasible": _f(r.get("dq_infeasible")),
            "shadow_step_iou": _f(r.get("shadow_step_iou")),
            "step_to": r.get("step_to") or None,
        } for r in ordered]
        # The cast shadow per frame, for the shadow and overlay plates. The CSV
        # records the path the scorer read; resolve relative paths against the
        # benchmark root (a run dir usually sits in the sibling checkout).
        sf, found = [], 0
        for r in ordered:
            p = r.get("shadow") or ""
            if p and not os.path.isabs(p):
                cand = os.path.normpath(os.path.join(bench, p))
                p = cand if os.path.exists(cand) else p
            if p and os.path.exists(p):
                sf.append(thumb(p, px))
                found += 1
            else:
                sf.append(None)
        if found:
            out["sf"] = sf
        if found < len(ordered):
            print(f"[!] {row.get('sequence_id', '?')}/{row.get('source', '?')}: "
                  f"{len(ordered) - found} shadow frames missing on disk")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default=_BENCH)
    p.add_argument("--px", type=int, default=128,
                   help="thumbnail edge; matches browser_payload.json")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    index = os.path.join(a.bench, "sequences.jsonl")
    if not os.path.exists(index):
        raise SystemExit(f"[!] {index} not found; run build_sequence_metadata.py")

    agg, frames = read_solves(a.bench)
    seqs = []
    for line in open(index, encoding="utf-8"):
        r = json.loads(line)
        sid = r["id"]
        m = r["target_motion"]

        src = {}
        sj = os.path.join(a.bench, "sequences", sid, "source.json")
        if os.path.exists(sj):
            with open(sj, encoding="utf-8") as fh:
                src = json.load(fh)

        solves = {s: solve_block(agg[(i, s)], frames.get((i, s), []), a.bench, a.px)
                  for (i, s) in agg if i == sid}
        rec = {
            "id": sid,
            "cls": r.get("class"),
            "prompt": r.get("prompt"),
            "n_frames": r["n_frames"],
            "frame_size": r["frame_size"],
            "family": "demo" if src else "generated",
            "fps": src.get("fps"),
            "origin": ({"scene": src.get("scene"), "letter": src.get("letter"),
                        "source_fps": src.get("source_fps"),
                        "sample_every": src.get("sample_every")}
                       if src else None),
            "loop": bool(m["loop"]),
            "loop_source": m.get("loop_source", "wrap-test"),
            "motion": {k: m[k] for k in ("mean_step_iou", "min_step_iou",
                                         "max_step_iou", "wrap_iou", "step_iou")},
            "f": [thumb(os.path.join(a.bench, fp), a.px) for fp in r["frames"]],
            "solved": sorted(solves),
            "solves": solves,
        }
        seqs.append(rec)

    for (sid, source) in agg:
        if not any(s["id"] == sid for s in seqs):
            print(f"[!] sequence_metrics row for '{sid}' matches no sequences.jsonl id")

    payload = {
        "px": a.px,
        "n": len(seqs),
        "n_solved": sum(1 for s in seqs if s["solved"]),
        "track": "sequences",
        "metric_groups": {"s1": S1, "s2": S2, "s3": S3},
        "sequences": seqs,
    }
    out = a.out or os.path.join(a.bench, "results", "sequences_payload.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    kb = os.path.getsize(out) / 1024
    nf = sum(s["n_frames"] for s in seqs)
    print(f"[sequences-payload] {len(seqs)} sequences, {nf} frames, "
          f"{payload['n_solved']} solved -> {out}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
