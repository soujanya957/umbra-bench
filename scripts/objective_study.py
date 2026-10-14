"""Which objective should the solver chase? Glyphs only, fast CMA-ES.

Every objective in fastshadow.objectives (plus `current`, the unpatched
1 - IoU + 0.5 DT) solves the same letter/digit targets with the same seeds and
budget. Each solve is re-rendered with MuJoCo and graded on every metric, so an
objective can't win just by being the metric it optimises:

  shape     iou, tc_iou, boundary_iou, nsd, chamfer, cldice, betti_error, holes_filled
  semantic  glyph classifier (rank, p) and the benchmark's CLIP grader
            (ViT-B-32 laion2b, glyph prompts, ranked within the subset)

The `cls` objectives are graded by the same classifier they steer with; CLIP is
steered with ViT-B-16 and graded with ViT-B-32.

    set FLEET_SHADOW_ART=...\\fleet-shadow-art-batched
    python scripts/objective_study.py run --seeds 3
    python scripts/objective_study.py grade
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastshadow  # noqa: E402

BENCH = fastshadow.BENCH
sys.path.insert(0, os.path.join(BENCH, "scripts"))
from run_base_optimizer import load_target, save_mask  # noqa: E402

OUT = os.path.join(BENCH, "results", "fastshadow", "objective_study")
ALL = ["current", "iou", "tversky_fn", "tversky_fp", "dt2", "biou", "cldice", "holes", "tc",
       "cls", "clip", "shape3", "shape3_cls"]


def target_list(src):
    rows = [json.loads(l) for l in open(os.path.join(BENCH, "results", "fastshadow", f"compare_{src}",
                                                     "per_target.jsonl"), encoding="utf-8")]
    return [r["id"] for r in rows]


def all_glyphs(root, studied):
    """One target per character (A-Z, a-z, 0-9). A character already in `studied`
    keeps that target; the rest take the DejaVu Sans Bold one."""
    have = {i.split("/")[1].split("_")[0]: i for i in studied}
    out = []
    for sub in ("letters_upper", "letters_lower", "digits"):
        stems = sorted(s for s in os.listdir(os.path.join(root, sub))
                       if os.path.exists(os.path.join(root, sub, s, f"{s}_shown.png")))
        for ch in sorted({s.split("_")[0] for s in stems}):
            if ch in have and have[ch].startswith(sub):
                out.append(have[ch])
                continue
            pick = [s for s in stems if s.split("_")[0] == ch]
            pref = [s for s in pick if s.endswith("_dejavusans-bold")]
            out.append(f"{sub}/{(pref or pick)[0]}")
    return out


def run(a):
    ms = fastshadow.use_fleet()
    import optimizer as opt
    from optimizer import OptimizerConfig, optimize_staged
    from renderer import ShadowRenderer, build_scene

    assert "batch_shadow_loss" in open(opt.__file__, encoding="utf-8").read(), \
        f"{ms} lacks the batched-render hooks; set FLEET_SHADOW_ART"
    from fastshadow.cma_bridge import BatchedShadowRenderer
    from fastshadow.objectives import TargetCache, use_objective
    from fastshadow.renderer import FastShadowRenderer

    root = os.path.abspath(a.results)
    ids = target_list(a.src)
    if a.all_glyphs:
        ids = all_glyphs(root, ids)
    if a.limit:
        ids = ids[: a.limit]
    first = json.load(open(os.path.join(root, *ids[0].split("/"), "results.json"), encoding="utf-8"))
    rig, ocfg = first["rig"], first["optimizer"]
    model, data, n = build_scene("urdf/SO101/so101_new_calib.urdf", n_robots=rig["n_robots"], arm_gap=rig["arm_gap_m"])
    mj = ShadowRenderer(model, data, size=rig["render_size"], n_robots=n)
    br = BatchedShadowRenderer(mj, FastShadowRenderer(mj))
    os.makedirs(os.path.join(OUT, "masks"), exist_ok=True)
    path = os.path.join(OUT, "solves.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["id"], r["objective"], r["seed"]) for r in map(json.loads, open(path, encoding="utf-8"))}
    objs = a.objectives or ALL
    json.dump({"objectives": objs, "targets": ids, "seeds": a.seeds, "optimizer": ocfg, "rig": rig},
              open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    t_all = time.time()
    for ti, tid in enumerate(ids):
        sub, stem = tid.split("/")
        d = os.path.join(root, sub, stem)
        T = load_target(os.path.join(d, f"{stem}_shown.png"), mj.size)
        label = stem.split("_")[0]
        for s in range(a.seeds):
            for name in objs:
                if (tid, name, s) in done:
                    continue
                use_objective(opt, br, None if name == "current" else name, TargetCache(sub, label))
                t0 = time.perf_counter()
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        res = optimize_staged(br, T, OptimizerConfig(seed=s, **ocfg))
                finally:
                    use_objective(opt, br, None)
                dt = time.perf_counter() - t0
                mask = mj.get_shadow_mask(res.best_q)
                mp = os.path.join("masks", f"{sub}__{stem}__{name}__s{s}.png")
                save_mask(mask, os.path.join(OUT, mp))
                _, _, cross, self_ = mj.eval_q(res.best_q)
                rec = {"id": tid, "subset": sub, "label": label, "objective": name, "seed": s,
                       "seconds": round(dt, 3), "n_evals": int(res.n_evals), "mask": mp,
                       "collision_free": bool(cross <= 0 and self_ <= 0),
                       "q_rad": [round(float(v), 6) for v in res.best_q]}
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
        el = time.time() - t_all
        print(f"[{ti + 1}/{len(ids)}] {tid}  {el / 60:.1f} min", flush=True)


def run_adam(a):
    """The same objectives, solved by Adam on the nvdiffrast soft renderer
    (64 starts × 300 steps, worse half re-seeded per blur stage). Final pick among
    the top 8 starts is by the objective itself on MuJoCo masks, plus MuJoCo's
    floor and contact penalties. Objective-major, CLIP last (it is the slow one)."""
    fastshadow.use_fleet()
    import torch
    from optimizer import floor_penalty_loss
    from renderer import ShadowRenderer, build_scene

    from fastshadow.grad_opt import FnObjective, GradConfig, Objective, Penalties, solve
    from fastshadow.objectives import OBJECTIVES, TargetCache
    from fastshadow.renderer import FastShadowRenderer
    from loss import shadow_loss

    root = os.path.abspath(a.results)
    ids = all_glyphs(root, target_list(a.src)) if a.all_glyphs else target_list(a.src)
    if a.limit:
        ids = ids[: a.limit]
    first = json.load(open(os.path.join(root, *ids[0].split("/"), "results.json"), encoding="utf-8"))
    rig, ocfg = first["rig"], first["optimizer"]
    model, data, n = build_scene("urdf/SO101/so101_new_calib.urdf", n_robots=rig["n_robots"], arm_gap=rig["arm_gap_m"])
    mj = ShadowRenderer(model, data, size=rig["render_size"], n_robots=n)
    fr = FastShadowRenderer(mj)
    picked = set(ids)
    solved = []
    for sub in os.listdir(root):
        d = os.path.join(root, sub)
        if os.path.isdir(d):
            for stem in os.listdir(d):
                f = os.path.join(d, stem, "results.json")
                if f"{sub}/{stem}" not in picked and os.path.exists(f):
                    solved += [r["q_rad"] for r in json.load(open(f, encoding="utf-8"))["runs"]]
    solved = np.array(solved)
    np.random.default_rng(0).shuffle(solved)
    pen = Penalties(fr, solved_q=solved[:3000])
    os.makedirs(os.path.join(OUT, "masks"), exist_ok=True)
    path = os.path.join(OUT, "solves.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["id"], r["objective"], r["seed"]) for r in map(json.loads, open(path, encoding="utf-8"))}
    objs = a.objectives or ALL
    objs = [o for o in objs if o != "clip"] + (["clip"] if "clip" in objs else [])
    json.dump({"optimizer": "adam", "objectives": objs, "targets": ids, "seeds": a.seeds,
               "adam": {"n_restarts": 64, "iters": 300, "reseed_sigma": 0.15, "top_k": 8}, "rig": rig},
              open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    t_all = time.time()
    for name in objs:
        for tid in ids:
            sub, stem = tid.split("/")
            T = load_target(os.path.join(root, sub, stem, f"{stem}_shown.png"), mj.size)
            label = stem.split("_")[0]
            tc = TargetCache(sub, label)
            fn = None if name == "current" else OBJECTIVES[name]
            for s in range(a.seeds):
                if (tid, name, s) in done:
                    continue
                cfg = GradConfig(floor_penalty=ocfg["floor_penalty"], collision_penalty=ocfg["collision_penalty"],
                                 self_collision_penalty=ocfg["self_collision_penalty"], reseed_sigma=0.15, seed=s)
                obj = Objective(fr, pen, T, cfg) if fn is None else FnObjective(fr, pen, T, cfg, fn, tc(T))
                r = solve(fr, pen, T, cfg, obj=obj)
                t0 = time.perf_counter()
                best = None
                for q in r.top_q:  # re-score the finalists on MuJoCo with this objective
                    mask, min_z, cross, self_ = mj.eval_q(q)
                    if fn is None:
                        sh = shadow_loss(mask, T, cfg.alpha, cfg.beta)
                    else:
                        with torch.no_grad():
                            sh = float(fn(torch.tensor(mask, device="cuda")[None], tc(T))[0])
                    tot = sh + floor_penalty_loss(min_z, cfg.floor_z, cfg.floor_penalty, cfg.floor_margin,
                                                  cfg.floor_barrier) + cfg.collision_penalty * cross \
                        + cfg.self_collision_penalty * self_
                    if best is None or tot < best[0]:
                        best = (tot, q, mask, cross, self_)
                dt = r.seconds + time.perf_counter() - t0
                _, q, mask, cross, self_ = best
                mp = os.path.join("masks", f"{sub}__{stem}__{name}__s{s}.png")
                save_mask(mask, os.path.join(OUT, mp))
                rec = {"id": tid, "subset": sub, "label": label, "objective": name, "seed": s,
                       "seconds": round(dt, 3), "mask": mp, "collision_free": bool(cross <= 0 and self_ <= 0),
                       "q_rad": [round(float(v), 6) for v in q]}
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
        print(f"[{name}] done, {(time.time() - t_all) / 60:.1f} min", flush=True)


def grade(a):
    """Metrics for every solve, written to graded.jsonl (recomputed in full)."""
    import torch
    from metrics import betti_error, boundary_iou, chamfer, cldice, iou, nsd, trimmed_centred
    from scipy.ndimage import binary_fill_holes

    from fastshadow.glyph_cls import GlyphClassifier

    sys.path.insert(0, os.path.join(BENCH, "scripts"))
    from ablation_metrics import CHANCE, ClipScorer

    root = os.path.abspath(a.results)
    solves = [json.loads(l) for l in open(os.path.join(OUT, "solves.jsonl"), encoding="utf-8")]
    clf = GlyphClassifier()
    clip = ClipScorer("ViT-B-32", "laion2b_s34b_b79k", "cuda")
    tgt = {}
    rows = []
    for r in solves:
        sub, stem = r["id"].split("/")
        if r["id"] not in tgt:
            d = os.path.join(root, sub, stem)
            T = load_target(os.path.join(d, f"{stem}_shown.png"), 128) > 0.5
            T0 = load_target(os.path.join(d, f"{stem}_original.png"), 128) > 0.5
            holes = binary_fill_holes(T) & ~T
            tgt[r["id"]] = (T, T0, holes)
        T, T0, holes = tgt[r["id"]]
        S = load_target(os.path.join(OUT, r["mask"]), 128) > 0.5
        be = betti_error(T, S)
        g = dict(r)
        g.pop("q_rad", None)
        g.update(iou=iou(T, S), tc_iou=trimmed_centred(T0, S)["tc_iou"], boundary_iou=boundary_iou(T, S),
                 nsd=nsd(T, S), chamfer=chamfer(T, S), cldice=cldice(T, S),
                 betti_error=be.get("betti_error", be.get("total")) if isinstance(be, dict) else be,
                 holes_filled=float((S & holes).sum() / holes.sum()) if holes.sum() else None)
        rk, p = clf.rank(torch.tensor(S[None], dtype=torch.float32), sub, r["label"])
        g.update(cls_rank=int(rk[0]), cls_top1=int(rk[0] == 1), cls_p=float(p[0]))
        rows.append(g)
    by_sub = {}
    for g in rows:
        by_sub.setdefault(g["subset"], []).append(g)
    for sub, gs in by_sub.items():
        res = clip.score(sub, [os.path.join(OUT, g["mask"]) for g in gs], [g["label"] for g in gs])
        ch = CHANCE[sub]["mrr"]
        for g, (rank, p) in zip(gs, res):
            g.update(clip_rank=rank, clip_top1=int(rank == 1), clip_rr=1 / rank,
                     clip_rr_cc=(1 / rank - ch) / (1 - ch), clip_p=p)
    with open(os.path.join(OUT, "graded.jsonl"), "w", encoding="utf-8") as f:
        for g in rows:
            f.write(json.dumps(g) + "\n")
    print(f"graded {len(rows)} solves")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "run_adam", "grade"])
    p.add_argument("--src", default="letters_digits")
    p.add_argument("--results", default=os.path.join(BENCH, "optimized", "big-budget-grounded"))
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--objectives", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--all-glyphs", action="store_true",
                   help="one target per character, A-Z a-z 0-9 (62), instead of the --src sample")
    p.add_argument("--out", default=OUT)
    a = p.parse_args()
    OUT = os.path.abspath(a.out)
    {"run": run, "run_adam": run_adam, "grade": grade}[a.cmd](a)
