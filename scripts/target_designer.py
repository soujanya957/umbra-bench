"""Target designer, letters first (protocol v2, after review: results/fastshadow/target_designer/
protocol_review.json): text -> a target that reads as the character AND that the arms can reach,
so shadows can be verified geometrically against it.

Why: Ivy's reads track overlap with a legible (fitted) target: IoU 0.70-0.75 -> 56% read,
0.75-0.80 -> 84%, 0.80+ -> 90-100% (775 guesses, all on DejaVu targets).

Result (2026-10-28, report + independent audit): own-target IoU 0.737 -> 0.819 on fresh seeds, but
choosing the font by that IoU picks heavy, solid faces with small counters (Gill Sans Ultra Bold
Condensed for 19 of 62) that the arms' thick shadows reach easily. A convex-hull blob already scores
0.79 against those targets (0.67 against the benchmark's), the held-out classifier reads the new
shadows no better, and 11 of 20 holed characters get counters too small to survive. So this is
reach, not readability; the blind read test (make_target_test_page.py) decides. report() writes the
audit's checks as summary.secondary (not pre-registered).

Stages (resumable; outputs in results/fastshadow/target_designer/):
  build       candidate targets: every bold/black/heavy/demi/condensed upright Windows text face,
              the 3 DejaVu faces, and DejaVu Sans Bold dilated by 2/4/6 px (stroke-weight control),
              through the benchmark's own render + grounding pipeline (asserted to reproduce
              targets_grounded). Logs ink area, stroke width, holes, components, small-caps.
  classifier  retrain the glyph classifier with every candidate font held out (DejaVu already is).
  gate        font level: clean top-1 >= 0.97 over the 62 glyphs (held-out classifier).
              glyph level: mean p(correct) over 64 shadow-like augmentations (the same 64 for every
              candidate) >= baseline's p - 0.05; same holes and components as the baseline;
              lowercase that just repeats its uppercase rejected unless DejaVu does too.
              Baselines always pass.
  fit         the benchmark's target fit (fleet-shadow-art target_fit, rig.target_fit settings),
              asserted to reproduce the stored _shown targets.
  solve       staged seeds (dt2, mesh floor): A seed 0 for every gated candidate; B seeds 1-2 for
              the top 5 per character + baselines, pick by the 3-seed mean; C fresh seeds 3-5 for
              the chosen target and both baselines, plus seeds 3-14 for the benchmark baseline
              ("just more seeds" control).
  report      pre-registered geometry endpoints, plus secondary target-independent checks; the
              deciding endpoint is the blind human read test.
  sheet       page/final_sheet.png: examples where the classifier reads the chosen shadow better/worse.

Baselines: "bench" = the benchmark's own target per character (objective_study's list: 43 sans,
11 serif, 8 mono); "sans" = DejaVu Sans Bold everywhere.

    set FLEET_SHADOW_ART=...\\fleet-shadow-art   (main, >= 5cbb9d7: target_fit fields + batched hooks)
    python scripts/target_designer.py build|classifier|gate|fit|solve|report|sheet
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastshadow  # noqa: E402

BENCH = fastshadow.BENCH
sys.path.insert(0, os.path.join(BENCH, "scripts"))
from ground_targets import ground, load_mask, save_mask  # noqa: E402
from make_targets_glyphs import SUBSETS, render_glyph  # noqa: E402
from run_base_optimizer import load_target  # noqa: E402

OUT = os.path.join(BENCH, "results", "fastshadow", "target_designer")
BB = os.path.join(BENCH, "optimized", "big-budget-grounded")
SUB_OF = {c: s for s, cs in SUBSETS.items() for c in cs}
HELDOUT_WEIGHTS = os.path.join(OUT, "glyph_cls_heldout.pt")
WEIGHT_WORDS = ("bold", "black", "heavy", "demi", "condensed", "cond", "extrab", "ultra")
SLANT_WORDS = ("italic", "oblique", "slant")


def P(*a):
    return os.path.join(OUT, *a)


def jload(name, default=None):
    return json.load(open(P(name), encoding="utf-8")) if os.path.exists(P(name)) else default


def jsave(obj, name):
    os.makedirs(OUT, exist_ok=True)
    json.dump(obj, open(P(name), "w", encoding="utf-8"), indent=1)


def bench_targets():
    """character -> benchmark stem (the objective study's target list)."""
    meta = json.load(open(os.path.join(BENCH, "results", "fastshadow", "objective_study", "meta.json"), encoding="utf-8"))
    return {tid.split("/")[1].split("_")[0]: tid.split("/")[1] for tid in meta["targets"]}


def dejavu_fonts():
    import matplotlib

    d = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    return {"dejavusans-bold": os.path.join(d, "DejaVuSans-Bold.ttf"),
            "dejavuserif-bold": os.path.join(d, "DejaVuSerif-Bold.ttf"),
            "dejavusansmono-bold": os.path.join(d, "DejaVuSansMono-Bold.ttf")}


def heavy_text_fonts():
    """Upright bold/black/heavy/demi/condensed Windows faces whose glyphs render as distinct shapes."""
    from PIL import Image, ImageDraw, ImageFont

    from fastshadow.glyph_cls import CLASSES, _fonts

    out, seen = {}, set()
    for f in _fonts():
        try:
            font = ImageFont.truetype(f, 48)
            fam, style = font.getname()
        except Exception:
            continue
        name = f"{fam} {style}".lower()
        if not any(w in name for w in WEIGHT_WORDS) or any(w in name for w in SLANT_WORDS) or name in seen:
            continue
        shapes = []
        for c in CLASSES:
            im = Image.new("L", (64, 64), 0)
            d = ImageDraw.Draw(im)
            try:
                bb = d.textbbox((0, 0), c, font=font)
                d.text((32 - (bb[2] - bb[0]) / 2 - bb[0], 32 - (bb[3] - bb[1]) / 2 - bb[1]), c, fill=255, font=font)
            except Exception:
                shapes = None
                break
            shapes.append(np.array(im) > 127)
        if shapes is None:
            continue
        a = np.stack(shapes).reshape(len(CLASSES), -1)
        if (a.sum(1) < 20).any() or len({r.tobytes() for r in a[:26]}) < 24:
            continue
        seen.add(name)
        out[os.path.splitext(os.path.basename(f))[0].lower()] = (f, f"{fam} {style}")
    return out


def topology(mask):
    """(components, holes) with 8-connected ink and 4-connected background."""
    from scipy import ndimage

    _, comps = ndimage.label(mask, structure=np.ones((3, 3)))
    lab, nb = ndimage.label(~mask)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return int(comps), int(len(set(range(1, nb + 1)) - border))


def stroke_width(mask128):
    from scipy import ndimage

    return float(4 * ndimage.distance_transform_edt(mask128)[mask128].mean()) if mask128.any() else 0.0


def norm_shape(mask, n=48):
    from PIL import Image

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((n, n), bool)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.array(Image.fromarray(crop.astype(np.uint8) * 255).resize((n, n), Image.NEAREST)) > 127


# ── build ──────────────────────────────────────────────────────────────────
def build(a):
    from scipy import ndimage

    dv = dejavu_fonts()
    rep = []
    for sub, chars in SUBSETS.items():
        for ch in chars:
            for name, path in dv.items():
                ref = os.path.join(BENCH, "targets_grounded", sub, f"{ch}_{name}.png")
                if not os.path.exists(ref):
                    continue  # 5 serif glyphs exist only as _v2 in the benchmark
                g, _, _ = ground(np.array(render_glyph(ch, path).convert("L")) < 128)
                r = load_mask(ref)
                rep.append(float((g & r).sum() / max(1, (g | r).sum())))
    print(f"reproduction of benchmark targets: min IoU {min(rep):.4f} over {len(rep)}", flush=True)
    assert min(rep) > 0.999
    fonts = {k: (p, k) for k, p in dv.items()}
    fonts.update(heavy_text_fonts())
    print(f"{len(fonts)} fonts ({len(fonts) - 3} heavy Windows faces + 3 DejaVu)", flush=True)
    cands = []

    def add(name, label, sub, ch, img512):
        g, _, _ = ground(img512)
        if g.sum() < 50:
            return
        fn = P("targets", sub, f"{ch}__{name}.png")
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        save_mask(g, fn)
        m128 = load_target(fn, 128) > 0.5
        comps, holes = topology(g)
        cands.append({"id": f"{sub}/{ch}__{name}", "sub": sub, "ch": ch, "font": name, "label": label,
                      "path": os.path.relpath(fn, BENCH), "area": float(m128.mean()), "stroke": stroke_width(m128),
                      "comps": comps, "holes": holes, "dejavu": name.startswith("dejavu")})

    for name, (path, label) in sorted(fonts.items()):
        for sub, chars in SUBSETS.items():
            for ch in chars:
                try:
                    img = np.array(render_glyph(ch, path).convert("L")) < 128
                except Exception:
                    continue
                add(name, label, sub, ch, img)
                if name == "dejavusans-bold":  # stroke-weight control: the same glyph, thicker
                    for r in (2, 4, 6):
                        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
                        add(f"dejavusans-bold+d{r}", f"DejaVu Sans Bold +{r}px", sub, ch,
                            ndimage.binary_dilation(img, structure=(xx ** 2 + yy ** 2) <= r * r))
    # small caps: a lowercase glyph that is just its own uppercase
    by = {(c["font"], c["ch"]): c for c in cands}
    for c in cands:
        if c["sub"] == "letters_lower" and (c["font"], c["ch"].upper()) in by:
            lo = norm_shape(load_mask(os.path.join(BENCH, c["path"])))
            up = norm_shape(load_mask(os.path.join(BENCH, by[(c["font"], c["ch"].upper())]["path"])))
            c["same_as_upper"] = float((lo & up).sum() / max(1, (lo | up).sum()))
    jsave(cands, "candidates.json")
    print(f"{len(cands)} candidate targets", flush=True)


# ── classifier ─────────────────────────────────────────────────────────────
def classifier(a):
    from fastshadow.glyph_cls import train

    cands = jload("candidates.json")
    held = sorted({c["font"] for c in cands if not c["font"].startswith("dejavu")})
    print(f"retraining the glyph classifier without {len(held)} candidate fonts", flush=True)
    train(exclude=held, weights=HELDOUT_WEIGHTS)


def heldout_classifier():
    from fastshadow.glyph_cls import GlyphClassifier

    return GlyphClassifier("cuda", weights=HELDOUT_WEIGHTS)


# ── gate ───────────────────────────────────────────────────────────────────
def legibility(clf, masks, sub, ch, n_aug=64):
    import torch

    from fastshadow.glyph_cls import augment

    T = torch.tensor(np.asarray(masks, np.float32))[:, None]
    out = []
    for m in T:
        g = torch.Generator(device="cpu").manual_seed(1234)  # the same 64 augmentations for every candidate
        X = augment(m[None].repeat(n_aug, 1, 1, 1).cuda(), g)[:, 0]
        with torch.no_grad():
            out.append(float(clf.prob(X, sub, ch).mean()))
    return out


def gate(a):
    import torch

    cands = jload("candidates.json")
    bt = bench_targets()
    clf = heldout_classifier()
    for c in cands:
        m = load_target(os.path.join(BENCH, c["path"]), 128) > 0.5
        with torch.no_grad():
            r, _ = clf.rank(torch.tensor(m[None], dtype=torch.float32).cuda(), c["sub"], c["ch"])
        c["clean_top1"] = bool(int(r[0]) == 1)
        c["legibility"] = legibility(clf, [m], c["sub"], c["ch"])[0]
    font_acc = defaultdict(list)
    for c in cands:
        font_acc[c["font"]].append(c["clean_top1"])
    base = {}
    for ch, stem in bt.items():  # the benchmark's own target for each character
        sub = SUB_OF[ch]
        T0 = load_target(os.path.join(BENCH, "targets_grounded", sub, f"{stem}.png"), 128) > 0.5
        comps, holes = topology(load_mask(os.path.join(BENCH, "targets_grounded", sub, f"{stem}.png")))
        base[ch] = {"stem": stem, "legibility": legibility(clf, [T0], sub, ch)[0], "comps": comps, "holes": holes}
    dv_same = {c["ch"]: c.get("same_as_upper", 0) for c in cands if c["font"] == "dejavusans-bold"}
    for c in cands:
        b = base[c["ch"]]
        is_base = c["font"] == b["stem"].split("_", 1)[1] or c["font"] == "dejavusans-bold"
        why = []
        if np.mean(font_acc[c["font"]]) < 0.97:
            why.append("font clean top-1 < 0.97")
        if c["legibility"] < b["legibility"] - 0.05:
            why.append("less legible than the baseline")
        if (c["comps"], c["holes"]) != (b["comps"], b["holes"]):
            why.append("topology differs from the baseline")
        if c.get("same_as_upper", 0) > 0.85 and dv_same.get(c["ch"], 0) <= 0.85:
            why.append("lowercase repeats its uppercase")
        c["baseline"] = is_base
        c["gate_fail"] = [] if is_base else why
    jsave(cands, "candidates.json")
    jsave(base, "baselines.json")
    ok = [c for c in cands if not c["gate_fail"]]
    per = defaultdict(int)
    for c in ok:
        per[c["ch"]] += 1
    print(f"{len(ok)} of {len(cands)} candidates pass; per character min {min(per.values())} median "
          f"{int(np.median(list(per.values())))}; fonts passing the font gate "
          f"{sum(np.mean(v) >= 0.97 for v in font_acc.values())} of {len(font_acc)}", flush=True)
    fails = defaultdict(int)
    for c in cands:
        for w in c["gate_fail"]:
            fails[w] += 1
    print("rejections:", dict(fails), flush=True)


# ── fit ────────────────────────────────────────────────────────────────────
_W = {}


def _fit_init():
    ms = fastshadow.use_fleet()
    from reachability import build_reachability_map
    from renderer import ShadowRenderer, build_scene

    model, data, n = build_scene("urdf/SO101/so101_new_calib.urdf", n_robots=3, arm_gap=0.2)
    _W["reach"] = build_reachability_map(ShadowRenderer(model, data, size=128, n_robots=n), n_samples=300, verbose=False)


def _fit_one(job):
    from target_fit import fit_target

    src, dst = job
    T0 = load_target(src, 128)
    fit = fit_target(T0, _W["reach"], scale_range=(0.35, 1.6), n_scales=14, max_shift_frac=0.22, n_shifts=15,
                     scale_penalty=0.0, min_retained=0.98, verbose=False)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    save_mask(fit.target > 0.5, dst)
    return dst, fit.as_dict()


def fit(a):
    from concurrent.futures import ProcessPoolExecutor

    cands = jload("candidates.json")
    srcs = [os.path.join(BENCH, c["path"]) for c in cands if not c["gate_fail"]]
    jobs = [(s, P("fitted", os.path.relpath(s, P("targets")))) for s in srcs]
    todo = [j for j in jobs if not os.path.exists(j[1])]
    print(f"fitting {len(todo)} of {len(jobs)} candidates", flush=True)
    # check first: the fit must reproduce the benchmark's stored shown targets
    _fit_init()
    for ch, stem in list(bench_targets().items())[:6]:
        sub = SUB_OF[ch]
        dst, _ = _fit_one((os.path.join(BENCH, "targets_grounded", sub, f"{stem}.png"), P("check", f"{stem}.png")))
        A = load_target(dst, 128) > 0.5
        B = load_target(os.path.join(BB, sub, stem, f"{stem}_shown.png"), 128) > 0.5
        assert (A ^ B).sum() == 0, f"fit does not reproduce {stem}"
    print("fit reproduces the benchmark's shown targets", flush=True)
    infos = jload("fits.json", {})
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_fit_init) as ex:
        for k, (dst, info) in enumerate(ex.map(_fit_one, todo, chunksize=8)):
            infos[os.path.relpath(dst, BENCH)] = info
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(todo)}", flush=True)
                jsave(infos, "fits.json")
    jsave(infos, "fits.json")


# ── solve ──────────────────────────────────────────────────────────────────
class Solver:
    def __init__(self):
        fastshadow.use_fleet()
        import optimizer as opt
        from renderer import ShadowRenderer, build_scene

        from fastshadow.cma_bridge import BatchedShadowRenderer
        from fastshadow.renderer import FastShadowRenderer

        first = json.load(open(os.path.join(BB, "digits", "0_dejavusans-bold", "results.json"), encoding="utf-8"))
        self.ocfg = first["optimizer"]
        model, data, n = build_scene("urdf/SO101/so101_new_calib.urdf", n_robots=3, arm_gap=0.2)
        self.mj = ShadowRenderer(model, data, size=128, n_robots=n)
        self.fast = FastShadowRenderer(self.mj)
        self.br = BatchedShadowRenderer(self.mj, self.fast, floor="mesh")
        self.opt = opt

    def __call__(self, target_path, sub, ch, seed):
        import contextlib
        import io

        from optimizer import OptimizerConfig, optimize_staged

        from fastshadow.objectives import TargetCache, use_objective

        T = load_target(target_path, 128)
        use_objective(self.opt, self.br, "dt2", TargetCache(sub, ch))
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = optimize_staged(self.br, T, OptimizerConfig(seed=seed, **self.ocfg))
        finally:
            use_objective(self.opt, self.br, None)
        mask = self.mj.get_shadow_mask(res.best_q) > 0.5
        _, _, cross, self_ = self.mj.eval_q(res.best_q)
        Tb = T > 0.5
        return mask, {"iou": float((mask & Tb).sum() / max(1, (mask | Tb).sum())), "seconds": round(time.perf_counter() - t0, 2),
                      "collision_free": bool(cross <= 0 and self_ <= 0),
                      "lowest_mm": round(1000 * float(self.fast.moving_min_z(res.best_q[None])[0]), 2),
                      "q": [round(float(v), 6) for v in res.best_q]}


def solve(a):
    cands = [c for c in jload("candidates.json") if not c["gate_fail"]]
    bt = bench_targets()
    path = P("solves.jsonl")
    done = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            done[(r["id"], r["seed"])] = r
    S = Solver()

    def fitted(c):
        return P("fitted", os.path.relpath(os.path.join(BENCH, c["path"]), P("targets")))

    def run(c, seed, stage):
        if (c["id"], seed) in done:
            return done[(c["id"], seed)]
        tp = c.get("shown") or fitted(c)
        mask, r = S(tp, c["sub"], c["ch"], seed)
        # no ':' in file names: on NTFS "a:b.png" silently becomes a hidden stream of a 0-byte file "a"
        mp = P("masks", f"{c['id'].replace('/', '__').replace(':', '-')}__s{seed}.png")
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        save_mask(mask, mp)
        r.update({"id": c["id"], "ch": c["ch"], "sub": c["sub"], "font": c["font"], "seed": seed, "stage": stage,
                  "target": os.path.relpath(tp, BENCH), "mask": os.path.relpath(mp, BENCH)})
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        done[(c["id"], seed)] = r
        return r

    # the benchmark baseline: its own stored shown target, not a re-render
    bench = {ch: {"id": f"{SUB_OF[ch]}/{ch}__BENCH:{stem}", "sub": SUB_OF[ch], "ch": ch, "font": "BENCH:" + stem.split("_", 1)[1],
                  "shown": os.path.join(BB, SUB_OF[ch], stem, f"{stem}_shown.png")} for ch, stem in bt.items()}
    by_ch = defaultdict(list)
    for c in cands:
        by_ch[c["ch"]].append(c)
    sans = {ch: next(c for c in by_ch[ch] if c["font"] == "dejavusans-bold") for ch in by_ch}
    t0 = time.time()
    # A: seed 0 for every gated candidate and both baselines
    jobs = [c for ch in sorted(by_ch) for c in by_ch[ch]] + list(bench.values())
    for k, c in enumerate(jobs):
        run(c, 0, "A")
        if (k + 1) % 100 == 0:
            print(f"A {k + 1}/{len(jobs)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    # B: seeds 1-2 for the top 5 per character (seed-0 IoU) and both baselines; pick by the 3-seed mean
    picks = {}
    for ch in sorted(by_ch):
        top = sorted(by_ch[ch], key=lambda c: -done[(c["id"], 0)]["iou"])[:5]
        pool = {c["id"]: c for c in top + [sans[ch], bench[ch]]}
        for c in pool.values():
            for s in (1, 2):
                run(c, s, "B")
        mean = {cid: np.mean([done[(cid, s)]["iou"] for s in (0, 1, 2)]) for cid in pool if cid != bench[ch]["id"]}
        best = max(top, key=lambda c: mean[c["id"]])
        picks[ch] = {"chosen": best["id"], "font": best["font"], "mean3": mean[best["id"]], "top5": [c["id"] for c in top]}
    jsave(picks, "picks.json")
    print(f"B done  {(time.time() - t0) / 60:.1f} min", flush=True)
    # C: fresh seeds 3-5 for chosen + both baselines; seeds 3-14 for the benchmark baseline (control)
    for ch in sorted(by_ch):
        chosen = next(c for c in by_ch[ch] if c["id"] == picks[ch]["chosen"])
        for c in (chosen, sans[ch]):
            for s in (3, 4, 5):
                run(c, s, "C")
        for s in range(3, 15):
            run(bench[ch], s, "C")
    print(f"C done  {(time.time() - t0) / 60:.1f} min", flush=True)


def report(a):
    """Pre-registered geometry endpoints (the human read test decides; see make_target_test_page.py).
    (a) paired per-character best-of-3 IoU (fresh seeds 3-5), chosen vs the benchmark's own target:
        Wilcoxon p < 0.05 with a positive median, and the share at IoU >= 0.75 goes up;
    (b) chosen best-of-3 beats the benchmark target given best-of-12 (just more seeds);
    (c) the held-out classifier reads the chosen shadows at least as often (top-1)."""
    import torch
    from scipy.stats import wilcoxon

    rows = [json.loads(l) for l in open(P("solves.jsonl"), encoding="utf-8")]
    picks = jload("picks.json")
    cands = {c["id"]: c for c in jload("candidates.json")}
    by = defaultdict(dict)
    for r in rows:
        by[r["id"]][r["seed"]] = r

    def feasible(r):
        return r["collision_free"] and r["lowest_mm"] >= -5

    def best(cid, seeds):
        rr = [by[cid][s] for s in seeds if s in by[cid]]
        ok = [r for r in rr if feasible(r)] or rr
        return max(ok, key=lambda r: r["iou"]) if ok else None

    clf = heldout_classifier()

    def reads(r):
        m = load_target(os.path.join(BENCH, r["mask"]), 128) > 0.5
        with torch.no_grad():
            rk, p = clf.rank(torch.tensor(m[None], dtype=torch.float32).cuda(), r["sub"], r["ch"])
        return int(rk[0]) == 1, float(p[0])

    per = []
    for ch, pk in sorted(picks.items()):
        bench_id = next(r["id"] for r in rows if r["ch"] == ch and "BENCH:" in r["id"])
        sans_id = next(r["id"] for r in rows if r["ch"] == ch and r["font"] == "dejavusans-bold")
        c3, b3, s3, b12 = best(pk["chosen"], (3, 4, 5)), best(bench_id, (3, 4, 5)), best(sans_id, (3, 4, 5)), best(bench_id, range(3, 15))
        rc, rb = reads(c3), reads(b3)
        cc = cands.get(pk["chosen"], {})
        per.append({"ch": ch, "chosen_font": pk["font"], "chosen_label": cc.get("label"), "chosen_is_bench_font": pk["font"] in bench_id,
                    "chosen": c3["iou"], "bench": b3["iou"], "sans": s3["iou"], "bench12": b12["iou"],
                    "cls_top1_chosen": rc[0], "cls_top1_bench": rb[0], "cls_p_chosen": rc[1], "cls_p_bench": rb[1],
                    "stroke_chosen": cc.get("stroke"), "area_chosen": cc.get("area")})
    d = np.array([x["chosen"] - x["bench"] for x in per])
    d12 = np.array([x["chosen"] - x["bench12"] for x in per])
    g = np.random.default_rng(0)
    boot = [d[g.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    boot12 = [d12[g.integers(0, len(d12), len(d12))].mean() for _ in range(4000)]
    share = {k: float(np.mean([x[k] >= 0.75 for x in per])) for k in ("chosen", "bench", "sans", "bench12")}
    disc = (sum(x["chosen"] >= 0.75 > x["bench"] for x in per), sum(x["bench"] >= 0.75 > x["chosen"] for x in per))
    nz = d[d != 0]
    w = wilcoxon(nz) if len(nz) else None
    fam = defaultdict(list)
    for r in rows:
        if r["seed"] == 0 and "BENCH:" not in r["id"]:
            f = r["font"]
            fam["DejaVu + thickened" if "+d" in f else "DejaVu" if f.startswith("dejavu") else "heavy Windows faces"].append(r["iou"])
    out = {"n_chars": len(per),
           "mean_iou": {k: float(np.mean([x[k] for x in per])) for k in ("chosen", "bench", "sans", "bench12")},
           "share_ge_075": share, "mcnemar_discordant": {"chosen_only": disc[0], "bench_only": disc[1]},
           "paired_diff_mean": float(d.mean()), "paired_diff_median": float(np.median(d)),
           "paired_diff_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "wilcoxon_p": float(w.pvalue) if w else None,
           "vs_bench12_mean": float(d12.mean()), "vs_bench12_ci": [float(np.percentile(boot12, 2.5)), float(np.percentile(boot12, 97.5))],
           "cls_top1": {"chosen": float(np.mean([x["cls_top1_chosen"] for x in per])), "bench": float(np.mean([x["cls_top1_bench"] for x in per]))},
           "chosen_is_bench_font": int(sum(x["chosen_is_bench_font"] for x in per)),
           "chosen_family": dict(Counter("thickened DejaVu" if "+d" in x["chosen_font"] else "DejaVu" if x["chosen_font"].startswith("dejavu")
                                         else "heavy Windows face" for x in per)),
           "seed0_iou_by_family": {k: float(np.mean(v)) for k, v in fam.items()}}
    out["endpoints"] = {
        "(a) paired gain, Wilcoxon p < 0.05 and share >= 0.75 up": bool(out["wilcoxon_p"] is not None and out["wilcoxon_p"] < 0.05
                                                                       and out["paired_diff_median"] > 0 and share["chosen"] > share["bench"]),
        "(b) beats the benchmark target at best-of-12": bool(out["vs_bench12_ci"][0] > 0),
        "(c) held-out classifier reads chosen no worse": bool(out["cls_top1"]["chosen"] >= out["cls_top1"]["bench"])}
    out["secondary"] = secondary(per, picks, rows, by, reads)
    jsave({"summary": out, "per_char": per}, "report.json")
    print(json.dumps(out, indent=1))


def _hull(m):
    from skimage.morphology import convex_hull_image

    return convex_hull_image(m) if m.any() else m


def _counters(m, min_px=4):
    """Enclosed background regions (4-connected) of at least min_px pixels, with their inscribed radius."""
    from scipy import ndimage

    lab, nb = ndimage.label(~m)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    out = []
    for k in range(1, nb + 1):
        if k in border:
            continue
        h = lab == k
        if h.sum() >= min_px:
            out.append((h, float(ndimage.distance_transform_edt(h).max())))
    return out


def secondary(per, picks, rows, by, reads):
    """Checks added after the run (audit 2026-10-28), NOT pre-registered. Endpoints (a)/(b) score each arm
    against its own target, and selecting fonts by that IoU favours heavy, solid glyphs with small counters
    that the arms' thick shadows match easily. These numbers do not depend on which target is easier:
      blob floor   IoU of the target with its own convex hull (a featureless blob covering the letter);
                   shadow IoU minus that floor = how far the shadow is above "any blob";
      solidity     ink / convex-hull area of the shadow (for the target it equals the blob floor);
      counters     target counters >= 4 px and their inscribed radius; kept = an enclosed shadow hole >= 4 px
                   overlaps it; over every fresh seed 3-5 of each arm;
      classifier   held-out classifier over every fresh seed 3-5 (not only the best of 3)."""
    g = np.random.default_rng(0)

    def ci(d):
        d = np.asarray(d, float)
        b = [d[g.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        return [float(np.mean(d)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    def tm(r):
        return load_target(os.path.join(BENCH, r["target"]), 128) > 0.5, load_target(os.path.join(BENCH, r["mask"]), 128) > 0.5

    def iou(a, b):
        return float((a & b).sum() / max((a | b).sum(), 1))

    arms = {}
    for x in per:
        ch = x["ch"]
        bench_id = next(r["id"] for r in rows if r["ch"] == ch and "BENCH:" in r["id"])
        arms[ch] = {"chosen": picks[ch]["chosen"], "bench": bench_id}
    res = defaultdict(dict)
    for ch, ids in arms.items():
        for arm, cid in ids.items():
            fresh = [by[cid][s] for s in (3, 4, 5)]
            t, _ = tm(fresh[0])
            hull_t = iou(t, _hull(t))
            cnt = _counters(t)
            kept, open_px, top1, pc, sol_s, above = [], [], [], [], [], []
            for r in fresh:
                _, m = tm(r)
                sh = [h for h, _ in _counters(m)]
                for h, _r in cnt:
                    kept.append(any((h & s).any() for s in sh))
                    open_px.append(float((h & ~m).sum()))
                k1, p = reads(r)
                top1.append(k1)
                pc.append(p)
                sol_s.append(float(m.sum() / max(_hull(m).sum(), 1)))
                above.append(r["iou"] - hull_t)
            res[ch][arm] = {"hull_iou": hull_t, "solidity_target": float(t.sum() / max(_hull(t).sum(), 1)),
                            "stroke_target": stroke_width(t), "solidity_shadow": float(np.mean(sol_s)),
                            "above_floor": float(np.mean(above)), "n_counters": len(cnt),
                            "min_counter_r": min((r_ for _, r_ in cnt), default=None),
                            "counters_kept": float(np.mean(kept)) if kept else None,
                            "open_counter_px": float(np.mean(open_px)) if open_px else None,
                            "cls_top1_fresh": float(np.mean(top1)), "cls_p_fresh": float(np.mean(pc))}
    chars = sorted(res)
    holed = [ch for ch in chars if res[ch]["bench"]["n_counters"]]

    def pair(key, cs=chars):
        return ci([res[ch]["chosen"][key] - res[ch]["bench"][key] for ch in cs])

    def mean(arm, key, cs=chars):
        return float(np.mean([res[ch][arm][key] for ch in cs]))

    return {
        "note": "added after the run, not pre-registered; see secondary() docstring",
        "blob_floor_iou": {a: mean(a, "hull_iou") for a in ("chosen", "bench")},
        "blob_floor_share_ge_075": {a: float(np.mean([res[ch][a]["hull_iou"] >= 0.75 for ch in chars])) for a in ("chosen", "bench")},
        "shadow_above_floor": {a: mean(a, "above_floor") for a in ("chosen", "bench")},
        "shadow_above_floor_diff_ci": pair("above_floor"),
        "solidity_shadow": {a: mean(a, "solidity_shadow") for a in ("chosen", "bench")},
        "stroke_target_px": {a: mean(a, "stroke_target") for a in ("chosen", "bench")},
        "cls_p_fresh": {a: mean(a, "cls_p_fresh") for a in ("chosen", "bench")},
        "cls_p_fresh_diff_ci": pair("cls_p_fresh"),
        "cls_top1_fresh": {a: mean(a, "cls_top1_fresh") for a in ("chosen", "bench")},
        "cls_top1_fresh_diff_ci": pair("cls_top1_fresh"),
        "holed_chars": "".join(holed),
        "small_counter_chars": "".join(ch for ch in holed if (res[ch]["chosen"]["min_counter_r"] or 0) < 7),
        "small_counter_chars_bench": "".join(ch for ch in holed if (res[ch]["bench"]["min_counter_r"] or 0) < 7),
        "counters_kept": {a: float(np.mean([res[ch][a]["counters_kept"] for ch in holed if res[ch][a]["counters_kept"] is not None]))
                          for a in ("chosen", "bench")},
        "open_counter_px": {a: float(np.mean([res[ch][a]["open_counter_px"] for ch in holed if res[ch][a]["open_counter_px"] is not None]))
                            for a in ("chosen", "bench")},
        "sans_is_bench_chars": int(sum(x["sans"] == x["bench"] for x in per)),
        "top_font": Counter(x["chosen_font"] for x in per).most_common(1)[0],
        "per_char": res}


def sheet(a, n=3):
    """page/final_sheet.png: today's vs the chosen target and best fresh shadow (seeds 3-5), for the n characters
    the held-out classifier reads most better and the n it reads most worse with the chosen font."""
    from PIL import Image, ImageDraw, ImageFont

    rep = jload("report.json")
    rows = [json.loads(l) for l in open(P("solves.jsonl"), encoding="utf-8")]
    picks = jload("picks.json")
    by = defaultdict(dict)
    for r in rows:
        by[r["id"]][r["seed"]] = r

    def best(cid):
        rr = [by[cid][s] for s in (3, 4, 5)]
        ok = [r for r in rr if r["collision_free"] and r["lowest_mm"] >= -5] or rr
        return max(ok, key=lambda r: r["iou"])

    per = sorted(rep["per_char"], key=lambda x: x["cls_p_chosen"] - x["cls_p_bench"])
    chars = [x["ch"] for x in per[::-1][:n]] + [x["ch"] for x in per[:n]]
    wall, ink, tgt = (234, 233, 228), (30, 31, 36), (150, 160, 178)
    cell, pad, lab = 96, 6, 22
    W = 4 * cell + 5 * pad + 40
    H = len(chars) * (cell + lab + pad) + 28
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    try:
        f, fb = ImageFont.truetype("arial.ttf", 13), ImageFont.truetype("arialbd.ttf", 22)
    except OSError:
        f = fb = ImageFont.load_default()
    for k, t in enumerate(("today: target", "shadow", "chosen: target", "shadow")):
        d.text((40 + pad + k * (cell + pad), 6), t, fill=(90, 95, 105), font=f)
    pmap = {x["ch"]: x for x in rep["per_char"]}
    for i, ch in enumerate(chars):
        y = 28 + i * (cell + lab + pad)
        if i == n:
            d.line([(4, y - pad // 2), (W - 4, y - pad // 2)], fill=(200, 200, 205), width=1)
        d.text((8, y + cell // 2 - 12), ch, fill=(0, 0, 0), font=fb)
        bench_id = next(r["id"] for r in rows if r["ch"] == ch and "BENCH:" in r["id"])
        for j, (r, col) in enumerate(((best(bench_id), "bench"), (best(picks[ch]["chosen"]), "chosen"))):
            for q, (path, color) in enumerate(((r["target"], tgt), (r["mask"], ink))):
                m = load_target(os.path.join(BENCH, path), 128) > 0.5
                a_ = np.tile(np.array(wall, np.uint8), m.shape + (1,))
                a_[m] = color
                x = 40 + pad + (2 * j + q) * (cell + pad)
                im.paste(Image.fromarray(a_).resize((cell, cell), Image.NEAREST), (x, y))
            x = 40 + pad + 2 * j * (cell + pad)
            d.text((x, y + cell + 3), f"overlap {r['iou']:.2f} · classifier {100 * pmap[ch]['cls_p_' + col]:.0f}%",
                   fill=(60, 64, 72), font=f)
    os.makedirs(P("page"), exist_ok=True)
    im.save(P("page", "final_sheet.png"), optimize=True)
    print("wrote", P("page", "final_sheet.png"), "chars", "".join(chars))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "classifier", "gate", "fit", "solve", "report", "sheet"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    {"build": build, "classifier": classifier, "gate": gate, "fit": fit, "solve": solve, "report": report, "sheet": sheet}[a.stage](a)


if __name__ == "__main__":
    main()
