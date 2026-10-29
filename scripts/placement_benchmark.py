#!/usr/bin/env python3
"""How well does the rig cast one shape as a function of WHERE it sits on the wall?

The arms are fixed at the ground centre of the frame. This takes one target, keeps
its size fixed, slides it over a grid of positions -- centred, a little and a lot
to either side, and in rows rising off the ground -- and solves every placement
with the base optimizer. Out come IoU vs distance from the ground-centre origin,
and the same numbers as a heatmap over the frame.

Size is held constant on purpose. `--fit-target` would rescale and re-place the
shape to wherever the rig casts it best, which is exactly the variable this
benchmark measures; so it is never passed, and IoU is against the placed target.

    python scripts/placement_benchmark.py                      # small budget, 7x5, 5 seeds
    python scripts/placement_benchmark.py --budget big --shards 4
    python scripts/placement_benchmark.py --stage plot         # re-plot what is solved
    python scripts/placement_benchmark.py --source targets/digits/7_dejavusans-bold.png

Stages (`--stage all` runs them in order, and every one is safe to re-run):

    targets   targets_placement/<name>/<placement>.png  + placements.csv
    solve     optimized/placement-<budget>/<name>/...   (run_base_optimizer.py)
    plot      results/placement/<name>_<budget>/         csv + three figures

The solve resumes: a placement whose results.json exists is skipped, so an SSH
session that drops mid-run is restarted with the same command. Run it under
tmux / screen, or `Start-Process` on Windows, so it outlives the session.

Coordinates are pixels of the 128 px render with the ORIGIN AT THE GROUND CENTRE
(bottom-centre of the frame), x to the right, `up` upward. A placement is named by
where the bottom-centre of the shape's bounding box sits; "distance" is that
anchor's distance from the origin, so the centred, grounded placement is 0.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
SIZE = 128  # the optimizer's render size; placements are integer pixels on it

# Mirrors the two sweeps' BUDGET.md, so "small" and "big" mean the same thing here.
BUDGETS = {
    "small": [],
    "big": ["--popsize", "48", "--phase1-iters", "16", "--phase2-iters", "16",
            "--final-iters", "30", "--no-adaptive-final"],
}


def load_mask(path: str) -> np.ndarray:
    """Dark = shape, the benchmark convention. Full resolution, bool."""
    return np.array(Image.open(path).convert("L")) < 128


def placed_shape(src: np.ndarray, height_frac: float) -> np.ndarray:
    """Crop to the shape and resize it once, to `height_frac` of the frame.

    Resized from the full-resolution crop and thresholded, rather than warped on
    the 128 grid: every placement is then the SAME bitmap at an integer offset,
    so no position differs from another by resampling noise.
    """
    ys, xs = np.nonzero(src)
    crop = src[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h = max(1, round(height_frac * SIZE))
    w = max(1, round(h * crop.shape[1] / crop.shape[0]))
    if h > SIZE or w > SIZE:
        raise SystemExit(f"--height-frac {height_frac} makes the shape {w}x{h} px, "
                         f"larger than the {SIZE} px frame")
    im = Image.fromarray((crop * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)
    return np.array(im) >= 128


def grid(shape: np.ndarray, nx: int, ny: int, margin: int):
    """Anchor offsets (dx, up) spanning everywhere the shape fits whole.

    dx is symmetric about 0 and always includes it when nx is odd; up starts on
    the ground. The extremes are the far edges of the frame minus `margin`.
    """
    h, w = shape.shape
    max_dx = (SIZE - w) // 2 - margin
    max_up = SIZE - h - margin
    if max_dx < 0 or max_up < 0:
        raise SystemExit("shape leaves no room to move; lower --height-frac")
    dxs = sorted({int(round(v)) for v in np.linspace(-max_dx, max_dx, nx)})
    ups = sorted({int(round(v)) for v in np.linspace(0, max_up, ny)})
    return [(dx, up) for up in ups for dx in dxs]


def place(shape: np.ndarray, dx: int, up: int) -> np.ndarray:
    h, w = shape.shape
    x0 = (SIZE - w) // 2 + dx
    y0 = SIZE - h - up
    out = np.zeros((SIZE, SIZE), bool)
    out[y0:y0 + h, x0:x0 + w] = shape
    return out


def stem_of(dx: int, up: int) -> str:
    return f"x{dx:+04d}_up{up:03d}"


def stage_targets(a) -> None:
    shape = placed_shape(load_mask(a.source), a.height_frac)
    d = os.path.join(BENCH, "targets_placement", a.name)
    os.makedirs(d, exist_ok=True)
    rows = []
    for dx, up in grid(shape, a.nx, a.ny, a.margin):
        m = place(shape, dx, up)
        Image.fromarray(((~m) * 255).astype(np.uint8), "L").convert("1").save(
            os.path.join(d, stem_of(dx, up) + ".png"))
        ys, xs = np.nonzero(m)
        cx = xs.mean() - (SIZE - 1) / 2          # origin = ground centre, y up
        cy = (SIZE - 1) - ys.mean()
        rows.append({"stem": stem_of(dx, up), "dx": dx, "up": up,
                     "w": shape.shape[1], "h": shape.shape[0],
                     "dist": round(float(np.hypot(dx, up)), 3),
                     "cx": round(float(cx), 3), "cy": round(float(cy), 3),
                     "centroid_dist": round(float(np.hypot(cx, cy)), 3)})
    # Stale placements from an earlier grid would be solved and plotted too.
    keep = {r["stem"] + ".png" for r in rows}
    for f in os.listdir(d):
        if f.endswith(".png") and f not in keep:
            os.remove(os.path.join(d, f))
    with open(os.path.join(d, "placements.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[targets] {len(rows)} placements of a {shape.shape[1]}x{shape.shape[0]} px "
          f"shape -> {os.path.relpath(d, BENCH)}")


def stage_solve(a) -> None:
    out = os.path.join(BENCH, "optimized", f"placement-{a.budget}")
    logs = os.path.join(a.results, "logs")
    os.makedirs(logs, exist_ok=True)
    base = [sys.executable, os.path.join(HERE, "run_base_optimizer.py"),
            "--targets-dir", "targets_placement", "--subsets", a.name,
            "--runs", str(a.runs), "--out", out, *BUDGETS[a.budget]]
    if a.repo:
        base += ["--repo", a.repo]
    if a.n_workers is not None:
        base += ["--n-workers", str(a.n_workers)]
    if a.force:
        base.append("--force")
    # run_base_optimizer prints arrows and Greek letters; on Windows a piped
    # stdout is cp1252 and the first one kills the process.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    procs = []
    for i in range(a.shards):
        log = open(os.path.join(logs, f"shard{i}.log"), "w", encoding="utf-8")
        cmd = base + ["--shard", str(i), "--num-shards", str(a.shards)]
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, cwd=BENCH), log))
    print(f"[solve] {a.shards} shard(s), {a.budget} budget, {a.runs} seeds -> "
          f"{os.path.relpath(out, BENCH)}\n        logs: {os.path.relpath(logs, BENCH)}")
    t0, n_total = time.time(), len(read_placements(a))
    while any(p.poll() is None for p, _ in procs):
        time.sleep(30)
        n = sum(1 for r in read_placements(a) if load_result(a, r["stem"]))
        print(f"  {n}/{n_total} placements solved  ({(time.time() - t0) / 60:.1f} min)",
              flush=True)
    bad = [i for i, (p, log) in enumerate(procs) if (log.close() or p.returncode)]
    if bad:
        raise SystemExit(f"[solve] shard(s) {bad} failed -- see {logs}")


def read_placements(a) -> list[dict]:
    p = os.path.join(BENCH, "targets_placement", a.name, "placements.csv")
    if not os.path.exists(p):
        raise SystemExit(f"no {os.path.relpath(p, BENCH)}; run --stage targets first")
    with open(p, encoding="utf-8") as f:
        return [{k: (v if k == "stem" else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]


def run_dir(a, stem: str) -> str:
    return os.path.join(BENCH, "optimized", f"placement-{a.budget}", a.name, stem)


def load_result(a, stem: str) -> dict | None:
    p = os.path.join(run_dir(a, stem), "results.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def stage_plot(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for r in read_placements(a):
        res = load_result(a, r["stem"])
        if not res:
            continue
        ious = np.array([x["iou"] for x in res["runs"]], float)
        rows.append(dict(r, n=len(ious), iou_mean=round(ious.mean(), 4),
                         iou_std=round(ious.std(ddof=1) if len(ious) > 1 else 0.0, 4),
                         iou_best=round(ious.max(), 4)))
    if not rows:
        raise SystemExit("[plot] nothing solved yet")
    os.makedirs(a.results, exist_ok=True)
    with open(os.path.join(a.results, "placement_iou.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    dxs = sorted({r["dx"] for r in rows})
    ups = sorted({r["up"] for r in rows})
    title = f"{a.name} · {a.budget} budget · {rows[0]['n']} seeds"

    # 1. distance vs IoU, one colour per height so left/right and up/down stay
    #    distinguishable even though distance folds them together.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis", max(2, len(ups)))
    for i, up in enumerate(ups):
        sel = sorted((r for r in rows if r["up"] == up), key=lambda r: r["dist"])
        ax.errorbar([r["dist"] for r in sel], [r["iou_mean"] for r in sel],
                    yerr=[r["iou_std"] for r in sel], fmt="o", ms=5, capsize=2,
                    color=cmap(i), label=f"up {int(up)} px")
    ax.set_xlabel("distance of shape anchor from ground centre (px of 128)")
    ax.set_ylabel("IoU (mean ± sd over seeds)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=.3)
    ax.legend(title="height", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.results, "distance_vs_iou.png"), dpi=150)
    plt.close(fig)

    # 2. heatmap over the frame, ground at the bottom like the wall.
    H = np.full((len(ups), len(dxs)), np.nan)
    for r in rows:
        H[ups.index(r["up"]), dxs.index(r["dx"])] = r["iou_mean"]
    fig, ax = plt.subplots(figsize=(1 + 0.9 * len(dxs), 1 + 0.8 * len(ups)))
    im = ax.imshow(H, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(H):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v < .6 else "black")
    ax.set_xticks(range(len(dxs)), [int(d) for d in dxs])
    ax.set_yticks(range(len(ups)), [int(u) for u in ups])
    ax.set_xlabel("dx from centre (px)")
    ax.set_ylabel("up from ground (px)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="mean IoU")
    fig.tight_layout()
    fig.savefig(os.path.join(a.results, "heatmap.png"), dpi=150)
    plt.close(fig)

    # 3. the best shadow at every placement, laid out as on the wall, coloured
    #    like the dashboard's overlay plate: agreement / target missed / spilled.
    tdir = os.path.join(BENCH, "targets_placement", a.name)
    fig, axs = plt.subplots(len(ups), len(dxs), squeeze=False,
                            figsize=(1.4 * len(dxs), 1.4 * len(ups)))
    for ax in axs.flat:
        ax.axis("off")
    for r in rows:
        T = load_mask(os.path.join(tdir, r["stem"] + ".png"))
        S = load_mask(os.path.join(run_dir(a, r["stem"]), r["stem"] + "_best.png"))
        rgb = np.ones(T.shape + (3,))
        rgb[T & S] = (0.1, 0.1, 0.9)
        rgb[T & ~S] = (0.0, 0.85, 0.85)
        rgb[~T & S] = (0.9, 0.0, 0.9)
        ax = axs[len(ups) - 1 - ups.index(r["up"]), dxs.index(r["dx"])]
        ax.imshow(rgb)
        ax.set_title(f"{r['iou_best']:.2f}", fontsize=7, pad=1)
    fig.suptitle(title + " · best seed", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.results, "overlays.png"), dpi=150)
    plt.close(fig)

    print(f"[plot] {len(rows)} placements -> {os.path.relpath(a.results, BENCH)}"
          f"  (placement_iou.csv, distance_vs_iou.png, heatmap.png, overlays.png)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=("all", "targets", "solve", "plot"))
    ap.add_argument("--source", default=os.path.join("targets", "digits",
                                                     "1_dejavusans-bold.png"),
                    help="shape to place, dark on white; relative to the bench root")
    ap.add_argument("--name", default=None,
                    help="label for this benchmark (default: the source's stem)")
    ap.add_argument("--height-frac", type=float, default=0.6,
                    help="shape height as a fraction of the frame, fixed at every "
                         "placement")
    ap.add_argument("--nx", type=int, default=7, help="columns, symmetric about centre")
    ap.add_argument("--ny", type=int, default=5, help="rows, from the ground up")
    ap.add_argument("--margin", type=int, default=2,
                    help="px kept clear of the frame edge at the extremes")
    ap.add_argument("--runs", type=int, default=5, help="seeds per placement")
    ap.add_argument("--budget", default="small", choices=tuple(BUDGETS))
    ap.add_argument("--shards", type=int, default=1,
                    help="parallel solver processes. Measure first: on a machine "
                         "whose render is the bottleneck they do not add up")
    ap.add_argument("--n-workers", type=int, default=None,
                    help="render threads per solver process (run_base_optimizer's "
                         "default: 8 on Linux, 1 on Windows). 1 rules out the "
                         "threaded renderer pool; parallelise with --shards instead")
    ap.add_argument("--repo", default=None, help="fleet-shadow-art checkout, if not "
                    "beside this one (passed through to run_base_optimizer.py)")
    ap.add_argument("--force", action="store_true", help="re-solve finished placements")
    a = ap.parse_args()

    a.source = os.path.join(BENCH, a.source)
    a.name = a.name or os.path.splitext(os.path.basename(a.source))[0]
    a.results = os.path.join(BENCH, "results", "placement", f"{a.name}_{a.budget}")

    if a.stage in ("all", "targets"):
        stage_targets(a)
    if a.stage in ("all", "solve"):
        stage_solve(a)
    if a.stage in ("all", "plot"):
        stage_plot(a)


if __name__ == "__main__":
    main()
