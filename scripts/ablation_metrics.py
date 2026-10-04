#!/usr/bin/env python3
"""Per-glyph shape and recognition metrics for ablation-style roots, paired like collect_ablation.py.

    python scripts/ablation_metrics.py optimized/ablation-probe10 --n 3 --no-clip
    <fh_l1>/python.exe scripts/ablation_metrics.py optimized/ablation-probe10      # + CLIP, CUDA
    python scripts/ablation_metrics.py optimized/aimed-start-probe10 --tex
    <fleet-shadow>/python.exe scripts/ablation_metrics.py ROOT --k 3 --no-clip \\
        --rerender --repo ../fleet-shadow-art-main                               # masks for best-of-3

collect_ablation.py reports best-of-seeds IoU, and IoU mostly measures how much
of a fat glyph is filled. A component can leave IoU flat and still move where
the outline sits (NSD), whether the silhouette is the same shape (Hu), or
whether it still reads as the character (CLIP). This scores every (variant,
glyph) cell on all of them and pairs each variant with the reference row
collect_ablation.py uses.

Pairing. Cells are <variant>-n<N>/<subset>/<stem>/results.json. Ablation dirs
follow collect_ablation.ORDER. A build-up row whose predecessor was not run at
this N pairs with the nearest rung below that was, so N=5 `full` reads vs
`flat`, as the paper footer does. A variant ORDER does not name, <base>_no_<x>,
pairs with <base>. Aimed-start dirs are <variant>-<budget>-n<N>, and
"no_*-<budget>" pairs with "full-<budget>". The glyph set is the intersection
across the variants being summarised (--variants narrows both).

Best-of-k. The scored shadow is the run with the highest per-run "iou" among
runs < k. That is the solver's own criterion, the one that set best_run.
Choosing by each reported metric would score a different shadow per column and
inflate every one. k defaults to n_base_runs. Extra restarts ("extra": true)
never count, because they are granted only to targets that scored low, so a
cell that earned them is flagged: collect_ablation.py's best_iou_vs_original
includes them and will not match there. Ties at the stored 4 decimals go to
best_run. Mask sources, in order:
  - <stem>_runKK.png, which exists only on the solving machine;
  - <stem>_best.png, when the chosen run is best_run;
  - a re-render of the run's q_rad (--rerender --repo), cached under
    <out-dir>/renders/<root>/<variant dir>/<subset>/<stem>/<stem>_runKK.png.
    It is kept out of optimized/ so re-renders are never mistaken for
    solver output or committed next to it;
  - otherwise the cell is missing.

Shape (scripts/metrics.py), with compute_metrics.py's conventions: both masks
at the rig's render_size, no alignment. The target goes through
run_base_optimizer.load_target (NEAREST from the 512 px authored PNG, dark =
shape), the same array the solver scored. The shadow is used as written:
1-bit, dark = shadow, already render_size. It stays unaligned because these are
unfitted solves of one placed target, so where the shadow lands is part of what
a component fixes or not. iou is checked against the chosen run's recorded
iou_vs_original, which is also the only proof that a re-render reproduced the
solve. nsd tolerance is 1% of the diagonal (~1.8 px at 128); boundary_iou uses
a 2% band; hu_distance uses 4 invariants, lower is better.

CLIP, as clip_eval_ablation.py: ViT-B-32 laion2b_s34b_b79k, glyph_prompt, the
raw 1-bit mask as the image, ranked inside its subset vocabulary (26 uppercase
letters, 26 lowercase letters or 10 digits; the true class is the stem up to its
first "_"), with p_true = softmax(100 * cosine). The ceiling is the authored
target file. letters_lower is added to VOCAB here, not in clip_eval_ablation.py,
whose uppercase and digit vocabularies are reused unchanged. It needs no prompt of
its own: glyph_prompt already says "a shadow of the lowercase letter a" and keeps
the label's case. semantic_metrics.clip_retrieval reloads the model on
every call and is CPU-only, so the same computation is repeated here with the
model loaded once. TF32 is off, so CUDA agrees with that CPU path.

Chance. Guessing gives MRR H_K/K: 0.148 over 26 letters, 0.293 over 10 digits.
A raw letters+digits mean therefore flatters digits. MRR_cc = (RR - chance) /
(1 - chance) per glyph, then averaged; 0 is chance, 1 is always rank 1. In a
paired delta the chance offset cancels within a glyph but the scale does not:
per glyph dMRR_cc = dMRR / (1 - chance), x1.17 for a letter and x1.41 for a
digit. A glyph's sign is kept, but the pooled mean re-weights digits and can
flip when letters and digits disagree. On aimed-start-probe10 N=3,
no_aimed-small reads dMRR -0.009 and dMRR_cc +0.005. So quote a pooled delta
with its per-subset lines, which the console prints.

Output goes to results/ablation_metrics/<root>[_k<k>].csv and .json (re-renders
under renders/ there). A cell
whose results.json mtime, k, chosen run and mask path are unchanged is reused
(--force recomputes). So a CLIP pass in the torch env can follow a --rerender
pass in the mujoco env without redoing either. Logs go to stderr, so --tex
stdout can be redirected into a .tex file.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fnmatch
import glob
import json
import math
import os
import re
import statistics as st
import sys
import time

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from clip_eval_ablation import VOCAB as _PROBE_VOCAB  # noqa: E402
from collect_ablation import ORDER  # noqa: E402
from metrics import boundary_iou, hu_distance, iou as mask_iou, load_mask, nsd  # noqa: E402
from run_base_optimizer import load_target, save_mask  # noqa: E402

# clip_eval_ablation.py's vocabularies (letters_upper, digits: same classes, same order) plus letters_lower,
# added here rather than there. glyph_prompt (tests/clip_eval.py) already names the case, "a shadow of the
# lowercase letter a" against "... uppercase letter A", and never changes the label's case, so lowercase uses
# it as is and needs no template of its own. letters_lower goes last so the order of the existing subsets,
# and every letters_upper and digits number, stays as before.
VOCAB = {**_PROBE_VOCAB, "letters_lower": [chr(c) for c in range(ord("a"), ord("z") + 1)]}
CHANCE = {s: {"K": len(v), "top1": 1 / len(v), "mrr": sum(1 / i for i in range(1, len(v) + 1)) / len(v)}
          for s, v in VOCAB.items()}
COLS = ["root", "variant_dir", "variant", "N", "subset", "stem", "k", "run",
        "iou", "nsd", "hu_distance", "boundary_iou",
        "clip_rank", "clip_top1", "clip_rr", "clip_rr_cc", "clip_p_true", "mask_source",
        # provenance; the cache is keyed on these
        "iou_reported", "mask_path", "target_path", "rj_mtime", "note"]
SHAPE = ("iou", "nsd", "hu_distance", "boundary_iou")
CLIPK = ("clip_rank", "clip_top1", "clip_rr", "clip_rr_cc", "clip_p_true")
SUMMARY = [  # key, console header, tex header, higher is better, decimals
    ("iou", "IoU", "IoU", True, 3), ("nsd", "NSD", "NSD", True, 3),
    ("hu_distance", "Hu", "Hu", False, 3), ("clip_top1", "top1", "top-1", True, 2),
    ("clip_rr", "MRR", "MRR", True, 3), ("clip_rr_cc", "MRRcc", r"MRR$_\mathrm{cc}$", True, 3),
    ("clip_p_true", "p_true", r"$p_\mathrm{true}$", True, 3)]
BUDGETS = ["tiny", "small", "small-init4", "big"]
IOU_TOL = 1e-4  # recorded IoU is rounded to 4 dp (<= 5e-5); one pixel of drift on a ~6k px union is ~1.7e-4

REF = {v: ref for v, _, ref in ORDER}
LABEL = {v: label for v, label, _ in ORDER}


def _chain(v):
    while v is not None:
        yield v
        v = REF.get(v)


BUILD_UP = set(_chain("full"))  # flat -> ... -> full; leave-one-out rows are not rungs


def log(msg):
    print(msg, file=sys.stderr, flush=True)


class Warnings:
    """Grouped, so 190 cells with the same problem print one line and a few examples."""

    def __init__(self):
        self.by = {}

    def __call__(self, kind, msg):
        self.by.setdefault(kind, []).append(msg)

    def dump(self):
        for kind, msgs in self.by.items():
            info = kind.endswith("(info)")
            show = 1 if info else 4
            log(f"[{'i' if info else '!'}] {kind}: {len(msgs)} cell(s)")
            for m in msgs[:show]:
                log(f"      {m}")
            if len(msgs) > show:
                log(f"      ... {len(msgs) - show} more")


def rel(bench, p):
    try:
        r = os.path.relpath(p, bench)
    except ValueError:  # other drive
        r = os.path.abspath(p)
    return r.replace(os.sep, "/")


def num(x):
    """CSV string / number / None -> float or None."""
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def finite(x):
    return x is not None and math.isfinite(x)


# --- which cells, paired with what ---------------------------------------------

def split_dir(name):
    """'full_no_fd-n3' -> ('full_no_fd', 3); 'no_aimed-small-init4-n5' -> ('no_aimed-small-init4', 5)."""
    m = re.fullmatch(r"(.+)-n(\d+)", name)
    return (m.group(1), int(m.group(2))) if m else None


def reference(v, present):
    """The variant `v` is paired against among those run at this N, or None."""
    if "-" in v:  # aimed-start: <variant>-<budget>
        base, budget = v.split("-", 1)
        ref = f"full-{budget}"
        return ref if base.startswith("no_") and ref in present else None
    if v in REF:
        ref = REF[v]
        if v in BUILD_UP:
            while ref is not None and ref not in present:
                ref = REF.get(ref)
        return ref if ref in present else None
    if "_no_" in v:
        ref = v.rsplit("_no_", 1)[0]
        return ref if ref in present else None
    return None


def row_order(present):
    names = [o[0] for o in ORDER]

    def key(v):
        if v in names:
            return (0, names.index(v), "", "")
        if "-" not in v:
            return (1, 0, v.rsplit("_no_", 1)[0], v)
        base, budget = v.split("-", 1)
        return (2, BUDGETS.index(budget) if budget in BUDGETS else len(BUDGETS), budget,
                f"{int(base != 'full')}{base}")
    return sorted(present, key=key)


def scan(root, ns, patterns):
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        sp = split_dir(os.path.basename(d))
        if not (sp and os.path.isdir(d)):
            continue
        v, n = sp
        if ns and n not in ns:
            continue
        if patterns and not any(fnmatch.fnmatchcase(v, p) for p in patterns):
            continue
        for rj in sorted(glob.glob(os.path.join(d, "*", "*", "results.json"))):
            out.append((os.path.basename(d), v, n, rj))
    return out


def choose(r, k_arg):
    """-> (k, chosen run dict or None, note). Same criterion as run_base_optimizer's best_run."""
    base = [x for x in r["runs"] if not x.get("extra")]  # extras are conditional restarts, not seeds
    k = k_arg or int(r.get("n_base_runs") or len(base))
    cand = [x for x in base if int(x["run"]) < k]
    if len(cand) < k:
        return k, None, f"only {len(cand)} base runs < k={k}"
    top = max(x["iou"] for x in cand)
    tied = [x for x in cand if x["iou"] == top]
    return k, next((x for x in tied if x["run"] == r.get("best_run")), tied[0]), ""


# --- masks ---------------------------------------------------------------------

class Rerenderer:
    """Casts a stored pose again, one scene per rig, built on first use."""

    def __init__(self, repo):
        self.ms = os.path.join(os.path.abspath(repo), "motion-aware-shadow")
        if not os.path.isdir(self.ms):
            raise SystemExit(f"--repo {repo!r} has no motion-aware-shadow/")
        if sys.platform == "win32" and os.environ.get("MUJOCO_GL", "").lower() == "egl":
            del os.environ["MUJOCO_GL"]  # no EGL on Windows; mujoco's default there is WGL
        sys.path.insert(0, self.ms)
        sys.path.insert(0, os.path.join(self.ms, "targets"))
        os.chdir(self.ms)  # URDF meshes resolve relative to the package root, as in run_base_optimizer
        self.scenes = {}

    def mask(self, rig, q):
        key = (int(rig["n_robots"]), rig.get("arm_gap_m"), rig.get("light_to_front_m"),
               rig.get("back_to_wall_m"), int(rig.get("render_size") or 128))
        if key not in self.scenes:
            with contextlib.redirect_stdout(sys.stderr):
                from renderer import ShadowRenderer, build_scene
                model, data, n = build_scene(
                    os.path.join(self.ms, "urdf/SO101/so101_new_calib.urdf"), n_robots=key[0],
                    arm_gap=key[1], light_to_front=key[2], back_to_wall=key[3])
                self.scenes[key] = ShadowRenderer(model, data, size=key[4], n_robots=n)
        return self.scenes[key].get_shadow_mask(np.asarray(q, dtype=np.float64))

    def close(self):
        for r in self.scenes.values():
            r.close()


def plan(a, stubs, cache, warn):
    """results.json -> row stubs with the chosen run and mask; reuse fresh cached metrics."""
    rows, renders = [], []
    for vdir, v, n, rj in stubs:
        cdir = os.path.dirname(rj)
        stem = os.path.basename(cdir)
        try:
            with open(rj, encoding="utf-8") as f:
                r = json.load(f)
        except (OSError, ValueError) as e:
            warn("unreadable results.json", f"{rel(a.bench, rj)}: {e}")
            continue
        subset = r.get("subset") or os.path.basename(os.path.dirname(cdir))
        rig = r.get("rig") or {}
        k, run, note = choose(r, a.k)
        row = dict(root=a.root_rel, variant_dir=vdir, variant=v, N=n, subset=subset, stem=stem, k=k,
                   run="" if run is None else int(run["run"]), mask_source="missing", mask_path="",
                   iou_reported="" if run is None else run.get("iou_vs_original", run.get("iou")),
                   rj_mtime=repr(os.path.getmtime(rj)), note=note, _rig=rig)
        if rig.get("target_fit"):
            warn("fitted solve (scored vs the authored target; _shown.png ignored)", f"{vdir}/{subset}/{stem}")
        if any(x.get("extra") for x in r["runs"]):
            warn("extra restarts excluded from best-of-k (collect_ablation.py's IoU includes them)",
                 f"{vdir}/{subset}/{stem}: {sum(bool(x.get('extra')) for x in r['runs'])} extra")
        tp = os.path.join(a.bench, r["target"]) if r.get("target") else ""
        if not os.path.exists(tp):
            fb = os.path.join(a.bench, "targets_grounded", subset, f"{stem}.png")
            if tp:
                warn("target path not on this machine, using targets_grounded (info)",
                     f"{r['target']} -> {rel(a.bench, fb)}")
            tp = fb
        row["target_path"] = rel(a.bench, tp)
        if not os.path.exists(tp):
            row["note"] = f"no target {row['target_path']}"
            warn("missing target", f"{vdir}/{subset}/{stem}: {row['note']}")
            rows.append(row)
            continue
        if run is None:
            warn("fewer runs than k", f"{vdir}/{subset}/{stem}: {note}")
            rows.append(row)
            continue
        kk = int(run["run"])
        run_png = os.path.join(cdir, f"{stem}_run{kk:02d}.png")
        best_png = os.path.join(cdir, f"{stem}_best.png")
        k_png = os.path.join(a.render_dir, vdir, subset, stem, f"{stem}_run{kk:02d}.png")
        render = False
        if os.path.exists(run_png):
            src, mp = "run_png", run_png
        elif kk == r.get("best_run") and os.path.exists(best_png):
            src, mp = "best_png", best_png
        elif os.path.exists(k_png) and os.path.getmtime(k_png) >= os.path.getmtime(rj):
            src, mp = "rerender", k_png
        elif a.rerender:
            src, mp, render = "rerender", k_png, True
        else:
            row["note"] = (f"run {kk} has no run PNG and no {stem}_best.png" if kk == r.get("best_run")
                           else f"run {kk} is not best_run {r.get('best_run')} and has no run PNG")
            warn("missing mask (pass --rerender --repo)", f"{vdir}/{subset}/{stem}: {row['note']}")
            rows.append(row)
            continue
        row.update(mask_source=src, mask_path=rel(a.bench, mp))
        old = cache.get((vdir, subset, stem))
        fresh = (not a.force and not render and old is not None and old.get("root") == a.root_rel
                 and all(str(old.get(c)) == str(row[c]) for c in ("rj_mtime", "k", "run", "mask_path", "mask_source"))
                 and num(old.get("iou")) is not None)
        if fresh:
            row.update({c: old.get(c, "") for c in SHAPE + CLIPK})
        row["_fresh"] = fresh
        if render:
            renders.append((row, run["q_rad"]))
        rows.append(row)
    return rows, renders


def compute_shape(a, rows, warn):
    targets = {}
    n = 0
    for row in rows:
        if row["mask_source"] == "missing" or row.get("_fresh"):
            continue
        size = int(row["_rig"].get("render_size") or 128)
        tp = os.path.join(a.bench, row["target_path"])
        mp = os.path.join(a.bench, row["mask_path"])
        try:
            if (tp, size) not in targets:
                targets[(tp, size)] = load_target(tp, size).astype(np.uint8)
            T = targets[(tp, size)]
            if Image.open(mp).size != (size, size):
                warn("shadow PNG not at render_size (resized NEAREST)", row["mask_path"])
            S = load_mask(mp, size)
        except (OSError, ValueError) as e:
            row.update(mask_source="missing", note=f"{type(e).__name__}: {e}")
            warn("unreadable mask or target", f"{row['variant_dir']}/{row['subset']}/{row['stem']}: {row['note']}")
            continue
        row.update(iou=round(mask_iou(T, S), 6), nsd=round(nsd(T, S), 6),
                   hu_distance=round(hu_distance(T, S), 6), boundary_iou=round(boundary_iou(T, S), 6),
                   **{c: "" for c in CLIPK})
        n += 1
    return n


# --- CLIP ------------------------------------------------------------------------

class ClipScorer:
    """clip_eval_ablation.score() with the model loaded once and on the GPU."""

    def __init__(self, model_name, pretrained, device):
        import torch
        import open_clip
        sys.path.insert(0, os.path.join(_BENCH, "tests"))
        from clip_eval import glyph_prompt

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch, self.prompt = torch, glyph_prompt
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.text = {}

    def _text(self, subset):
        if subset not in self.text:
            with self.torch.no_grad():
                tok = self.tokenizer([self.prompt(c) for c in VOCAB[subset]]).to(self.device)
                tf = self.model.encode_text(tok)
                self.text[subset] = tf / tf.norm(dim=-1, keepdim=True)
        return self.text[subset]

    def score(self, subset, paths, labels, batch_size=128):
        """-> [(rank, p_true)], ranked within VOCAB[subset]."""
        tf, vocab, out = self._text(subset), VOCAB[subset], []
        with self.torch.no_grad():
            for i in range(0, len(paths), batch_size):
                ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
                f = self.model.encode_image(self.torch.stack([self.preprocess(im) for im in ims]).to(self.device))
                f = f / f.norm(dim=-1, keepdim=True)
                for s, lab in zip((f @ tf.T).float().cpu().numpy(), labels[i:i + batch_size]):
                    t = vocab.index(lab)
                    rank = int(np.where(np.argsort(-s) == t)[0][0]) + 1
                    p = np.exp(100 * s.astype(np.float64))
                    out.append((rank, float(p[t] / p.sum())))
        return out


def compute_clip(a, rows, warn):
    """Fill CLIP columns for rows lacking them, one batch per (variant, subset)."""
    groups = {}
    for row in rows:
        if row["mask_source"] == "missing" or num(row.get("clip_rank")) is not None:
            continue
        if row["subset"] not in VOCAB:
            warn("no CLIP vocabulary for subset", f"{row['variant_dir']}/{row['subset']}/{row['stem']}")
            continue
        if row["stem"].split("_")[0] not in VOCAB[row["subset"]]:
            warn("glyph label not in vocabulary", f"{row['subset']}/{row['stem']}")
            continue
        groups.setdefault((row["variant_dir"], row["subset"]), []).append(row)
    if not groups:
        return 0
    t0 = time.time()
    try:
        sc = ClipScorer(a.model_name, a.pretrained, a.device)
    except ImportError as e:  # e.g. the mujoco env: shape metrics still get written
        warn("CLIP skipped: no torch/open_clip in this env (rerun in fh_l1, or pass --no-clip)",
             f"{type(e).__name__}: {e}")
        return 0
    log(f"[clip] {a.model_name}/{a.pretrained} on {sc.device}, loaded in {time.time() - t0:.1f}s")
    n = 0
    for (_, subset), rs in sorted(groups.items()):
        res = sc.score(subset, [os.path.join(a.bench, r["mask_path"]) for r in rs],
                       [r["stem"].split("_")[0] for r in rs])
        ch = CHANCE[subset]["mrr"]
        for r, (rank, p) in zip(rs, res):
            r.update(clip_rank=rank, clip_top1=int(rank == 1), clip_rr=round(1 / rank, 6),
                     clip_rr_cc=round((1 / rank - ch) / (1 - ch), 6), clip_p_true=round(p, 6))
        n += len(rs)
    log(f"[clip] {n} images in {time.time() - t0:.1f}s")
    return n


def target_rows(a, rows, cache):
    """One 'target' row per glyph: the CLIP ceiling, from the authored file."""
    out = {}
    for row in rows:
        key = (row["subset"], row["stem"])
        if key in out or row["subset"] not in VOCAB or not os.path.exists(os.path.join(a.bench, row["target_path"])):
            continue
        t = dict(root=a.root_rel, variant_dir="target", variant="target", N="", subset=row["subset"],
                 stem=row["stem"], k="", run="", mask_source="target", mask_path=row["target_path"],
                 target_path=row["target_path"], iou_reported="", note="",
                 rj_mtime=repr(os.path.getmtime(os.path.join(a.bench, row["target_path"]))),
                 **{c: "" for c in SHAPE + CLIPK})
        old = cache.get(("target",) + key)
        if (not a.force and old and all(str(old.get(c)) == str(t[c]) for c in ("root", "mask_path", "rj_mtime"))
                and num(old.get("clip_rank")) is not None):
            t.update({c: old[c] for c in CLIPK})
        out[key] = t
    return list(out.values())


# --- summary -----------------------------------------------------------------------

def _mean_sem(d):
    m = st.mean(d)
    sem = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return {"mean": m, "sem": sem, "n": len(d), "beyond_2sem": len(d) > 1 and abs(m) > 2 * sem}


def summarise(rows, trows, n):
    cells = {}
    for r in rows:
        if str(r["N"]) == str(n) and r["mask_source"] != "missing" and num(r.get("iou")) is not None:
            cells.setdefault(r["variant"], {})[(r["subset"], r["stem"])] = r
    if not cells:
        return None
    present = set(cells)
    glyphs = sorted(set.intersection(*(set(c) for c in cells.values())))
    if not glyphs:
        log(f"[!] N={n}: no glyph is common to all {len(cells)} variants; narrow them with --variants")
        return None
    most = max(len(c) for c in cells.values())
    tmap = {(t["subset"], t["stem"]): t for t in trows}
    subsets = sorted({g[0] for g in glyphs}, key=lambda s: (list(VOCAB).index(s) if s in VOCAB else len(VOCAB), s))
    keys = SHAPE + CLIPK[1:]

    def means(get, gs):
        out = {}
        for key in keys:
            vals = [x for x in (num(get(g, key)) for g in gs) if finite(x)]
            out[key] = st.mean(vals) if vals else None
        return out

    def by_subset(get, ref_get=None):
        out = {}
        for s in subsets:
            gs = [g for g in glyphs if g[0] == s]
            rr = [x for x in (num(get(g, "clip_rr")) for g in gs) if finite(x)]
            if not rr:
                continue
            m = means(get, gs)
            e = dict(n=len(rr), K=CHANCE[s]["K"], chance_top1=CHANCE[s]["top1"], chance_mrr=CHANCE[s]["mrr"],
                     top1=m["clip_top1"], mrr=m["clip_rr"], mrr_cc=m["clip_rr_cc"], p_true=m["clip_p_true"])
            if ref_get:
                d = [x - y for x, y in ((num(get(g, "clip_rr")), num(ref_get(g, "clip_rr"))) for g in gs)
                     if finite(x) and finite(y)]
                if d:
                    e["d_mrr"] = _mean_sem(d)
            out[s] = e
        return out

    tget = lambda g, key: (tmap.get(g) or {}).get(key)  # noqa: E731
    out_rows = []
    for v in row_order(present):
        ref = reference(v, present)
        get = lambda g, key, v=v: cells[v][g].get(key)  # noqa: E731
        rget = (lambda g, key, ref=ref: cells[ref][g].get(key)) if ref else None
        row = dict(variant=v, variant_dir=f"{v}-n{n}", ref=ref, order_ref=REF.get(v, "") if v in REF else None,
                   label=LABEL.get(v), n_cells=len(cells[v]),
                   k=sorted({int(cells[v][g]["k"]) for g in glyphs}), mean=means(get, glyphs), delta={})
        if ref:
            for key in keys:
                d = [x - y for x, y in ((num(get(g, key)), num(rget(g, key))) for g in glyphs)
                     if finite(x) and finite(y)]
                if d:
                    row["delta"][key] = _mean_sem(d)
        row["clip_by_subset"] = by_subset(get, rget)
        out_rows.append(row)
    return dict(N=n, n_glyphs=len(glyphs), glyphs=[f"{s}/{t}" for s, t in glyphs],
                n_by_subset={s: sum(g[0] == s for g in glyphs) for s in subsets},
                limited_by={v: len(c) for v, c in cells.items() if len(c) < most} if len(glyphs) < most else {},
                ceiling=dict(mean=means(tget, glyphs), clip_by_subset=by_subset(tget)), rows=out_rows)


def _symbols():
    try:
        "↑↓Δ±".encode(sys.stdout.encoding or "ascii")
        return "↑", "↓", "Δ", "±"
    except (UnicodeEncodeError, LookupError):
        return " ^", " v", "d", "+-"


def print_console(root_rel, S):
    up, down, dl, pm = _symbols()
    W = 24
    width = {key: 2 * dec + 6 + len(pm) for key, _, _, _, dec in SUMMARY}
    ks = {}
    for r in S["rows"]:
        for k in r["k"]:
            ks.setdefault(k, []).append(r["variant"])
    common = max(ks, key=lambda k: len(ks[k]))
    odd = "; ".join(f"k={k} for {', '.join(v)}" for k, v in sorted(ks.items()) if k != common)
    print(f"# {root_rel}  N={S['N']}: {S['n_glyphs']} glyphs common to {len(S['rows'])} variants "
          f"({', '.join(f'{s} {c}' for s, c in S['n_by_subset'].items())}); best-of-k, k = {common}"
          + (f" ({odd})" if odd else ""))
    if S["limited_by"]:
        print("#   intersection limited by: " + ", ".join(f"{v} ({c})" for v, c in sorted(S["limited_by"].items())))
    has_clip = _has_clip(S)
    cols = _columns(has_clip)
    chance = ", ".join(f"{s} K={CHANCE[s]['K']} top1 {CHANCE[s]['top1']:.3f} MRR {CHANCE[s]['mrr']:.3f}"
                       for s in S["n_by_subset"] if s in CHANCE)
    print(("#   CLIP chance: " + chance + "; " if has_clip else "#   ")
          + f"* = |{dl}| > 2 s.e.m. of the paired per-glyph difference")
    print(f"{'variant':<{W}s}" + "".join(f"{h + (up if hi else down):>{width[key]}s}" for key, h, _, hi, _ in cols))

    def mean_line(name, m):
        cells = []
        for key, _, _, _, dec in cols:
            txt = "-" if m.get(key) is None else "%.*f " % (dec, m[key])
            cells.append(txt.rjust(width[key]))
        return name.ljust(W) + "".join(cells)

    def sub_line(by):
        parts = []
        for s, e in by.items():
            txt = (f"{s}({e['n']}) top1 {e['top1']:.2f} MRR {e['mrr']:.3f} cc {e['mrr_cc']:.3f} p {e['p_true']:.3f}")
            if "d_mrr" in e:
                txt += f" {dl}MRR {e['d_mrr']['mean']:+.3f}{pm}{e['d_mrr']['sem']:.3f}"
            parts.append(txt)
        return f"{'':<{W}s}  " + " | ".join(parts) if parts else None

    if has_clip:
        print(mean_line("target (ceiling)", S["ceiling"]["mean"]))
        if sub_line(S["ceiling"]["clip_by_subset"]):
            print(sub_line(S["ceiling"]["clip_by_subset"]))
    for r in S["rows"]:
        print(mean_line(r["variant"], r["mean"]))
        if r["ref"]:
            cells = []
            for key, _, _, _, dec in cols:
                d = r["delta"].get(key)
                if d is None:
                    cells.append(f"{'-':>{width[key]}s}")
                else:
                    sem = "nan" if not math.isfinite(d["sem"]) else f"{d['sem']:.{dec}f}".lstrip("0")
                    cells.append(f"{d['mean']:+.{dec}f}{pm}{sem}{'*' if d['beyond_2sem'] else ' '}".rjust(width[key]))
            print(f"{'  ' + dl + ' vs ' + r['ref']:<{W}s}" + "".join(cells))
        line = sub_line(r["clip_by_subset"])
        if line:
            print(line)
    print()


def _tex_name(r):
    lab = r["label"] or r["variant"].replace("_", r"\_")
    if r["ref"] and r["order_ref"] is not None and r["ref"] != r["order_ref"]:
        lab += r" (vs.\ " + r["ref"].replace("_", r"\_") + ")"
    return lab


def _has_clip(S):
    return (S["ceiling"]["mean"].get("clip_rr") is not None
            or any(r["mean"].get("clip_rr") is not None for r in S["rows"]))


def _columns(has_clip):
    """Drop the CLIP columns when nothing has been CLIP-scored, rather than print a wall of dashes."""
    return [c for c in SUMMARY if has_clip or not c[0].startswith("clip")]


def print_tex(blocks):
    cols = _columns(any(_has_clip(S) for S in blocks))
    ncol = 1 + 2 * len(cols)
    print(r"\begin{tabular}{@{}l" + "rr" * len(cols) + r"@{}}")
    print(r"\toprule")
    print(" & " + " & ".join(r"\multicolumn{2}{c}{" + th + (r" $\uparrow$" if hi else r" $\downarrow$") + "}"
                             for _, _, th, hi, _ in cols) + r" \\")
    print("".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cols))))
    print(" & " + " & ".join([r"mean", r"$\Delta$"] * len(cols)) + r" \\")
    for S in blocks:
        print(r"\midrule")
        print(rf"% N={S['N']}: {S['n_glyphs']} glyphs, bold = paired delta beyond 2 s.e.m.")
        print(r"\multicolumn{" + str(ncol) + r"}{@{}l@{}}{\itshape $N{=}" + str(S["N"]) + "$, "
              + str(S["n_glyphs"]) + r" glyphs} \\")

        def m(v, dec):
            return "--" if v is None else f"${v:.{dec}f}$"

        def dfmt(d, dec):
            if d is None:
                return "---"
            if abs(d["mean"]) < 0.5 * 10 ** -dec:
                return "$0$"
            return (rf"$\mathbf{{{d['mean']:+.{dec}f}}}$" if d["beyond_2sem"] else f"${d['mean']:+.{dec}f}$")

        c, cells = S["ceiling"]["mean"], []
        for key, _, _, _, dec in cols:  # shape metrics are trivial for the target itself
            cells += [m(c.get(key), dec) if key.startswith("clip") else "", ""]
        if any(key.startswith("clip") for key, *_ in cols):
            print("target (ceiling) & " + " & ".join(cells) + r" \\")
        for r in S["rows"]:
            print(_tex_name(r) + " & " + " & ".join(
                f"{m(r['mean'].get(key), dec)} & {dfmt(r['delta'].get(key), dec) if r['ref'] else ''}"
                for key, _, _, _, dec in cols) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# --- io ----------------------------------------------------------------------------

def read_cache(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["variant_dir"], r["subset"], r["stem"]): r for r in csv.DictReader(f)}


def _csv_val(v):
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return v


def _clean(x):
    """NaN -> None, so the JSON stays standard (an s.e.m. over one glyph is undefined)."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def _typed(row):
    out = {}
    for c in COLS:
        v = row.get(c, "")
        if c in ("N", "k", "run", "clip_rank", "clip_top1"):
            out[c] = int(float(v)) if num(v) is not None else None
        elif c in SHAPE + CLIPK[2:] + ("iou_reported", "rj_mtime"):
            out[c] = num(v)
        else:
            out[c] = v
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", help="ablation-style root, e.g. optimized/ablation-probe10 (cwd- or --bench-relative)")
    p.add_argument("--bench", default=_BENCH)
    p.add_argument("--n", default="all", help="fleet sizes to score: 3, 5, 3,5 or all (default)")
    p.add_argument("--k", type=int, default=None,
                   help="best-of-k over runs < k (default: each results.json's n_base_runs)")
    p.add_argument("--variants", nargs="+", default=None, metavar="GLOB",
                   help="only variants matching these globs (dir name minus -n<N>); also narrows the glyph set")
    p.add_argument("--rerender", action="store_true",
                   help="re-cast chosen runs that have no PNG from q_rad; needs --repo and mujoco")
    p.add_argument("--repo", default=None,
                   help="fleet-shadow-art checkout for --rerender: the renderer that solved the root")
    p.add_argument("--no-clip", action="store_true", help="skip CLIP (cached CLIP values are kept)")
    p.add_argument("--model-name", default="ViT-B-32")
    p.add_argument("--pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--device", default=None, help="torch device for CLIP (default: cuda if available)")
    p.add_argument("--out-dir", default=None, help="default: <bench>/results/ablation_metrics")
    p.add_argument("--force", action="store_true", help="recompute every cell")
    p.add_argument("--tex", action="store_true", help="print a LaTeX tabular instead of the console table")
    a = p.parse_args()
    if a.rerender and not a.repo:
        p.error("--rerender needs --repo: a different renderer moves pixels, and the IoU check would only say so afterwards")
    if a.k is not None and a.k < 1:
        p.error("--k must be >= 1")

    a.bench = os.path.abspath(a.bench)
    root = a.root if os.path.isdir(a.root) else os.path.join(a.bench, a.root)
    if not os.path.isdir(root):
        p.error(f"no such root: {a.root}")
    root = os.path.abspath(root)
    a.root_rel = rel(a.bench, root)
    ns = None if a.n == "all" else {int(x) for x in a.n.split(",")}
    out_dir = os.path.abspath(a.out_dir or os.path.join(a.bench, "results", "ablation_metrics"))
    base = os.path.join(out_dir, os.path.basename(root) + (f"_k{a.k}" if a.k else ""))
    a.render_dir = os.path.join(out_dir, "renders", os.path.basename(root))  # per run, so shared across k
    warn = Warnings()

    stubs = scan(root, ns, a.variants)
    if not stubs:
        sys.exit(f"no <variant>-n<N>/<subset>/<stem>/results.json under {root} for --n {a.n}")
    cache = read_cache(base + ".csv")
    rows, renders = plan(a, stubs, cache, warn)

    if renders:
        rr = Rerenderer(a.repo)
        try:
            for i, (row, q) in enumerate(renders, 1):
                mask = rr.mask(row["_rig"], q)
                mp = os.path.join(a.bench, row["mask_path"])
                os.makedirs(os.path.dirname(mp), exist_ok=True)
                save_mask(mask, mp)
                log(f"[render] {i}/{len(renders)} {row['variant_dir']}/{row['subset']}/{row['stem']} run {row['run']}")
        finally:
            rr.close()
    n_shape = compute_shape(a, rows, warn)
    for row in rows:  # also for cached rows: a mismatch should stay loud until it is fixed
        got, want = num(row.get("iou")), num(row.get("iou_reported"))
        if row["mask_source"] != "missing" and finite(got) and want is not None and abs(got - want) > IOU_TOL:
            warn(f"iou differs from results.json by > {IOU_TOL}" + (" (re-render drift?)" if row["mask_source"] == "rerender" else ""),
                 f"{row['variant_dir']}/{row['subset']}/{row['stem']} run {row['run']}: {got:.4f} vs {want:.4f} ({row['mask_source']})")
    trows = target_rows(a, rows, cache)
    n_clip = 0 if a.no_clip else compute_clip(a, rows + trows, warn)
    n_missing = sum(r["mask_source"] == "missing" for r in rows)
    log(f"[ablation_metrics] {len(rows)} cells under {a.root_rel}: {sum(bool(r.get('_fresh')) for r in rows)} cached, "
        f"{n_shape} scored, {len(renders)} re-rendered, {n_clip} CLIP-scored, {n_missing} missing")

    # merge: rows outside this invocation's scope stay as they were in the CSV
    scope = {r["variant_dir"] for r in rows} | {"target"}
    here = {(r["variant_dir"], r["subset"], r["stem"]) for r in rows + trows}
    keep = [r for key, r in cache.items() if key not in here and (key[0] not in scope or key[0] == "target")]
    allrows = sorted(keep + rows + trows, key=lambda r: (r["variant_dir"] == "target", r["variant_dir"], r["subset"], r["stem"]))
    os.makedirs(out_dir, exist_ok=True)
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in allrows:
            w.writerow({c: _csv_val(r.get(c, "")) for c in COLS})

    blocks = [S for S in (summarise(rows, trows, n) for n in sorted({r["N"] for r in rows})) if S]
    old = {}
    if os.path.exists(base + ".json"):
        try:
            with open(base + ".json", encoding="utf-8") as f:
                old = json.load(f).get("summary", {})
        except (OSError, ValueError):
            old = {}
    old.update({str(S["N"]): S for S in blocks})
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"root": a.root_rel, "k": a.k or "n_base_runs", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "chance": CHANCE, "clip": {"model": a.model_name, "pretrained": a.pretrained},
                   "summary": _clean(old), "cells": [_clean(_typed(r)) for r in allrows]}, f, indent=1)
    log(f"[ablation_metrics] -> {base}.csv / .json")
    warn.dump()

    if a.tex:
        print_tex(blocks)
    else:
        for S in blocks:
            print_console(a.root_rel, S)


if __name__ == "__main__":
    main()
