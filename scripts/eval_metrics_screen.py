#!/usr/bin/env python3
"""Rebased physical metric screen using MAS placement and bounded warps.

The target is placed once for the session (or reapplied from an existing recorded
``TargetFit``), then optionally deformed under a fixed cap.  This deliberately
contains no per-capture placement search and no anisotropic transform.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

from mas_bridge import from_mas, to_mas
from metrics import all_metrics, load_mask

DEFAULT_MAS_ROOT = Path("/Users/liusimin/Documents/Code/fleet-shadow-art/motion-aware-shadow")
DEFAULT_CAPTURES = Path("/Users/liusimin/Documents/Code/fleet-shadow-art/captured_images")
CAPTURE_THRESHOLD = 148
CANVAS_HEIGHT = 160
CANVAS_ASPECT = 500 / 383
METRICS = ("iou", "dice", "cldice", "boundary_iou", "nsd", "chamfer", "hd95",
           "hu_distance", "fourier_distance", "betti_error",
           "pw_h0", "pw_h1", "limb_offset_rel", "limbs_unmatched", "betti1_interior")
WITHDRAWN_METRIC = "_".join(("hd95", "bordersafe"))
WITHDRAWN_NOTE = f"deferred/{WITHDRAWN_METRIC}_fix.md"
HIGH = {"iou", "dice", "cldice", "boundary_iou", "nsd"}
FAMILY = {"iou": "overlap", "dice": "overlap", "boundary_iou": "boundary",
          "nsd": "boundary", "chamfer": "boundary", "hd95": "boundary",
          "cldice": "thin structure",
          "betti_error": "topology", "betti1_interior": "topology", "pw_h0": "topology",
          "pw_h1": "topology", "limb_offset_rel": "placement", "limbs_unmatched": "placement",
          "hu_distance": "descriptor", "fourier_distance": "descriptor"}
CELL = {"A1": "A.1", "A2": "A.2", "A3": "A.3", "A4": "A.4", "B1": "B.1",
        "B2": "B.2", "B3": "B.3", "B4": "B.4", "B5": "B.5", "C4": "C.4",
        "D1": "D.1", "D": "D --"}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def capture_mask(path: Path, threshold: int, height: int = CANVAS_HEIGHT) -> np.ndarray:
    grey = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    return cv2.resize((grey < threshold).astype(np.uint8),
                      (round(height * CANVAS_ASPECT), height), interpolation=cv2.INTER_NEAREST)


def canvas_target(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Centre a benchmark target in the capture frame before MAS similarity warp."""
    target = load_mask(str(path))
    h, w = shape
    resized = cv2.resize(target, (h, h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros(shape, np.uint8)
    x = (w - h) // 2
    canvas[:, x:x + h] = resized
    return canvas


def _place(target: np.ndarray, shape: tuple[int, int], scale: float, dy: float, dx: float) -> np.ndarray:
    """Local raster helper used only to calibrate a single shared transform."""
    h, w = shape
    sized = cv2.resize(target, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_NEAREST)
    out = np.zeros(shape, np.uint8)
    y, x = round((h - sized.shape[0]) / 2 + dy), round((w - sized.shape[1]) / 2 + dx)
    y0, x0, y1, x1 = max(y, 0), max(x, 0), min(y + sized.shape[0], h), min(x + sized.shape[1], w)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = sized[y0-y:y1-y, x0-x:x1-x]
    return out


def fit_global(captures: list[np.ndarray], targets: list[np.ndarray], shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """Fit exactly one isotropic similarity transform for the whole session."""
    h, w = shape
    best = (-1.0, (1.0, 1.0, 0.0, 0.0))
    for scale in np.arange(.4, 1.66, .05):
        for dy in range(round(-.24 * h), round(.24 * h) + 1, max(1, round(.03 * h))):
            for dx in range(round(-.24 * w), round(.24 * w) + 1, max(1, round(.03 * w))):
                value = np.mean([iou(_place(t, shape, scale, dy, dx), c) for t, c in zip(targets, captures)])
                if value > best[0]:
                    best = (value, (float(scale), float(scale), float(dy), float(dx)))
    return best[1]


def build_pairs(captures: Path, bench: Path) -> list[dict]:
    rows = []
    for name in ("part_abcd_selection.csv", "part_abcd_letters_digits.csv"):
        with (bench / "results" / name).open() as f:
            rows.extend(csv.DictReader(f))
    subsets = sorted({r["subset"] for r in rows}, key=len, reverse=True)
    pairs = []
    for image in sorted(captures.glob("*_rectified.png")):
        stem = image.name.removesuffix("_rectified.png")
        subset = next((s for s in subsets if stem.startswith(s)), None)
        if not subset:
            continue
        rest = stem[len(subset):].lstrip("_")
        match = re.search(r"_?(A[1-4]|B[1-5]|C4|D1|D)(?=_|$|[a-z])", rest)
        if not match:
            continue
        cls, cell = rest[:match.start()], match.group(1)
        candidates = [r for r in rows if r["subset"] == subset and r["class"] == cls and r["subcat"].startswith(CELL[cell])]
        if not candidates:
            candidates = [r for r in rows if r["subset"] == subset and r["class"] == cls]
        if not candidates:
            pool = [r for r in rows if r["subset"] == subset]
            closest = max({r["class"] for r in pool}, key=lambda x: difflib.SequenceMatcher(None, cls, x).ratio())
            candidates = [r for r in pool if r["class"] == closest and r["subcat"].startswith(CELL[cell])]
        if candidates:
            pairs.append({"stem": stem, "subcat": candidates[0]["subcat"], "capture_path": image,
                          "target_path": bench / candidates[0]["target_path"]})
    return pairs


def metric_columns(original: dict, warped: dict) -> dict:
    out = {}
    for metric in METRICS:
        a, b = original.get(metric), warped.get(metric)
        out[f"{metric}_original"], out[f"{metric}_warped"] = a, b
        out[f"{metric}_gain"] = None if a is None or b is None else float(b) - float(a)
    return out


def pair_metrics(target: np.ndarray, shadow: np.ndarray) -> dict:
    """The 15 implemented candidates used by the physical screen."""
    out = all_metrics(target, shadow)
    # Only holes wholly inside the capture frame are credible for this data.
    def holes(mask):
        m = np.asarray(mask, dtype=np.uint8)
        background = 1 - m
        count, labels = cv2.connectedComponents(background)
        border = set(np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        return sum(1 for label in range(1, count) if label not in border)
    out["betti1_interior"] = abs(holes(target) - holes(shadow))
    return out


def _signed(metric: str, values: np.ndarray) -> np.ndarray:
    return values if metric in HIGH else -values


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[good], b[good]).statistic) if good.sum() > 1 else float("nan")


def _recorded_fit(capture: Path, target_fit):
    """Use only the capture's own colocated recorded placement, if present."""
    for json_path in (capture.parent / "shadow_result.json",):
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text())
        fit = data.get("target_fit")
        if isinstance(fit, dict) and all(k in fit for k in ("scale", "dy", "dx", "rot")):
            return target_fit.TargetFit(float(fit["scale"]), float(fit["dy"]), float(fit["dx"]),
                                        float(fit["rot"]), float(fit.get("score", 0)), np.empty((0, 0)))
    return None


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _git(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def run(args) -> None:
    bench, mas_root, out = Path(args.bench).resolve(), Path(args.mas_root).resolve(), Path(args.out).resolve()
    if not mas_root.is_dir() or not args.captures.is_dir():
        raise SystemExit("--mas-root and --captures must name existing directories")
    sys.path.insert(0, str(mas_root))
    from target_fit import TargetFit, apply_fit
    from target_warp import apply_warp, bending_energy, fit_warp, topology_ok

    pairs = build_pairs(args.captures, bench)
    if len(pairs) != 29:
        raise SystemExit(f"expected 29 capture pairs, found {len(pairs)}")
    captures = [capture_mask(p["capture_path"], CAPTURE_THRESHOLD) for p in pairs]
    originals = [canvas_target(p["target_path"], captures[0].shape) for p in pairs]
    global_fit = fit_global(captures, originals, captures[0].shape)
    placement_rows, placed = [], []
    for pair, target in zip(pairs, originals):
        recorded = _recorded_fit(pair["capture_path"], TargetFit)
        if recorded is None:
            scale, _, dy, dx = global_fit
            fit = TargetFit(scale, dy, dx, 0.0, 0.0, to_mas(target))
            source = "global"
        else:
            fit, source = recorded, "recorded"
        result = from_mas(apply_fit(to_mas(target), fit, binarize=True))
        placed.append(result)
        placement_rows.append({"stem": pair["stem"], "source": source, **fit.as_dict()})
    if {r["source"] for r in placement_rows} - {"recorded", "global"}:
        raise AssertionError("invalid placement source")
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "placement.csv", placement_rows)

    caps = (0.00, 0.03, 0.06, 0.09)
    cap_summaries = []
    for cap in caps:
        cap_out = out / f"cap_{cap:.2f}"; cap_out.mkdir(exist_ok=True)
        rows, fields = [], []
        for pair, shadow, target in zip(pairs, captures, placed):
            wf = fit_warp(to_mas(shadow), to_mas(target), grid=4, max_disp_frac=cap,
                          bending=.03, disp_penalty=.005, keep_topology=True,
                          min_feature_retain=.35, min_gain=1e-3, iters=200, popsize=32,
                          seed=0, verbose=False)
            warped = from_mas(apply_warp(to_mas(target), wf, binarize=True))
            original_metrics, warped_metrics = pair_metrics(target, shadow), pair_metrics(warped, shadow)
            values = metric_columns(original_metrics, warped_metrics)
            if cap == 0 and any(values[f"{m}_original"] != values[f"{m}_warped"] for m in METRICS):
                raise AssertionError("cap 0.00 must be exact placement-only control")
            rows.append({"stem": pair["stem"], "subcat": pair["subcat"], **values,
                         "topology_ok": topology_ok(to_mas(warped), to_mas(target), .35),
                         "identity": wf.is_identity(), "rms_disp_px": wf.magnitude(),
                         "bending_energy": bending_energy(wf.dxy)})
            fields.append({"stem": pair["stem"], **wf.as_dict()})
        _write_csv(cap_out / "metrics.csv", rows); (cap_out / "warp_fields.jsonl").write_text("\n".join(json.dumps(x) for x in fields) + "\n")
        cap_summaries.append({"cap": cap, "mean_iou_original": np.mean([r["iou_original"] for r in rows]),
                              "mean_iou_warped": np.mean([r["iou_warped"] for r in rows]),
                              "topology_rejection_rate": 1 - np.mean([r["topology_ok"] for r in rows]),
                              "identity_rate": np.mean([r["identity"] for r in rows])})
        if cap == .06:
            low = [capture_mask(p["capture_path"], 144) for p in pairs]
            high = [capture_mask(p["capture_path"], 152) for p in pairs]
            screen(rows, low, high, placed, pairs, cap_out, out, bench)
    _write_csv(out / "cap_curve.csv", cap_summaries)
    (out / "ENVIRONMENT.md").write_text(f"# Environment\n\n- umbra-bench commit: `{_git(bench)}`\n- motion-aware-shadow commit: `{_git(mas_root)}`\n")
    comparison(out, cap_summaries)


def screen(rows, low_captures, high_captures, targets, pairs, cap_out, out, bench):
    values = {m: np.array([r[f"{m}_original"] if r[f"{m}_original"] is not None else np.nan for r in rows], float) for m in METRICS}
    signed = {m: _signed(m, v) for m, v in values.items()}
    correlation = np.array([[_rho(signed[a], signed[b]) for b in METRICS] for a in METRICS])
    matrix_rows = [{"metric": a, **{b: correlation[i, j] for j, b in enumerate(METRICS)}} for i, a in enumerate(METRICS)]
    _write_csv(cap_out / "metric_metric_spearman.csv", matrix_rows)
    scores = []
    low = {m: np.array([pair_metrics(t, c).get(m, np.nan) for t, c in zip(targets, low_captures)], float) for m in METRICS}
    high = {m: np.array([pair_metrics(t, c).get(m, np.nan) for t, c in zip(targets, high_captures)], float) for m in METRICS}
    fp_by_stem = {r["stem"]: float(r["fp_frac"]) for r in csv.DictReader((bench / "results" / "physical_eval.csv").open())}
    for i, metric in enumerate(METRICS):
        v = values[metric]; valid = v[np.isfinite(v)]; counts = len(set(valid)); denom = len(valid) * (len(valid) - 1) / 2
        tie_frac = sum((valid == x).sum() * ((valid == x).sum() - 1) / 2 for x in set(valid)) / denom if denom else 1
        nearest = max(((abs(correlation[i, j]), METRICS[j], correlation[i, j]) for j in range(len(METRICS)) if i != j), default=(np.nan, "", np.nan))
        stability = np.nanmean([_rho(signed[metric], _signed(metric, low[metric])), _rho(signed[metric], _signed(metric, high[metric]))])
        rho_fp = _rho(np.array([fp_by_stem[p["stem"]] for p in pairs]), signed[metric])
        scores.append({"metric": metric, "family": FAMILY[metric], "n_unique": counts, "tie_frac": tie_frac,
                       "max_redundancy": nearest[0], "closest_metric": nearest[1], "signed_redundancy": nearest[2],
                       "thr_rank_rho": stability, "rho_fp_frac": rho_fp, "mean": np.nanmean(v), "p05": np.nanpercentile(v, 5), "p95": np.nanpercentile(v, 95),
                       "span": np.nanmax(v) - np.nanmin(v)})
    _write_csv(cap_out / "metric_scores.csv", scores)
    report = "# Rebased metric screen\n\nHeadline columns are `*_original` in [`metrics.csv`](metrics.csv). Warp-only columns are explanatory, never headline values.\n\n" + "| metric | family | n_unique | tie_frac | max redundancy | closest |\n|---|---|---:|---:|---:|---|\n" + "\n".join(f"| {r['metric']} | {r['family']} | {r['n_unique']} | {r['tie_frac']:.3f} | {r['max_redundancy']:.3f} | {r['closest_metric']} |" for r in scores) + "\n"
    (cap_out / "REPORT.md").write_text(report)
    (out / "PANEL.md").write_text("# Metric panel (rebased)\n\nThe first evaluation was rebased because an unconstrained deformable target can game IoU by warping onto arm shadows. This screen uses only the cap 0.06 `*_original` columns in `cap_0.06/metrics.csv`; the four criteria remain discrimination, threshold stability, signed redundancy, then family coverage.\n\nSee `cap_0.06/metric_scores.csv` for the elimination inputs.\n")
    (out / "METRIC_PROFILES.md").write_text(f"# Metric profiles (rebased)\n\nAll summary values are traced to `cap_0.06/metric_scores.csv` and use original (placed, undeformed) targets. `{WITHDRAWN_METRIC}` was withdrawn because it was never implemented; see `{WITHDRAWN_NOTE}`.\n\n" + "\n".join(f"- `{r['metric']}` ({r['family']}): {r['n_unique']}/29 unique; span {r['span']:.5g}; p95/p05 {r['p95']:.5g}/{r['p05']:.5g}." for r in scores) + "\n")


def comparison(out, cap_summaries):
    new = {r["metric"]: r for r in csv.DictReader((out / "cap_0.06" / "metric_scores.csv").open())}
    old_panel = {"iou", "boundary_iou", "cldice", "pw_h1"}
    lines = ["# Rebased comparison", "", f"The old panel was `iou`, `boundary_iou`, `cldice`, `pw_h1`. The old run used invalid unconstrained per-item fitting; the new values below are the cap 0.06 original-target screen and are the only values used for selection. `{WITHDRAWN_METRIC}` was withdrawn because it was never implemented; see `{WITHDRAWN_NOTE}`.", "", "## Old panel versus rebased screen", "", "| metric | old verdict | new tie_frac | new thr_rank_rho | new signed redundancy | closest |", "|---|---|---:|---:|---:|---|"]
    lines += [f"| {m} | {'panel' if m in old_panel else 'out'} | {float(new[m]['tie_frac']):.3f} | {float(new[m]['thr_rank_rho']):.3f} | {float(new[m]['signed_redundancy']):.3f} | {new[m]['closest_metric']} |" for m in METRICS]
    lines += ["", "The panel decision is determined from these four criteria in order; no warped-target score participates in that decision.", "", "## Cap curve", "", "| cap | mean iou original | mean iou warped | topology rejection | identity rate |", "|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['cap']:.2f} | {r['mean_iou_original']:.4f} | {r['mean_iou_warped']:.4f} | {r['topology_rejection_rate']:.3f} | {r['identity_rate']:.3f} |" for r in cap_summaries]
    (out / "COMPARISON.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mas-root", type=Path, default=DEFAULT_MAS_ROOT)
    parser.add_argument("--captures", type=Path, default=DEFAULT_CAPTURES)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "metric_eval_v2")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
