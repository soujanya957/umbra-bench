"""Build the fast-renderer report page (one self-contained HTML).

Sources, all on the same letter/digit targets:
  results/fastshadow/parity.json                     renderer agreement + speed
  results/fastshadow/compare_<tag>/                  target masks, target list
  results/fastshadow/fastcma_<tag>/runs.jsonl        CMA-ES (MuJoCo), CMA-ES (nvdiffrast),
                                                     Adam at N start points
Every IoU there is MuJoCo-measured.

    python scripts/make_fastshadow_report.py --tag letters_digits
"""

import argparse
import base64
import io
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastshadow  # noqa: E402

BENCH = fastshadow.BENCH
FS = os.path.join(BENCH, "results", "fastshadow")

INK = np.array([38, 36, 44])  # shadow on target
EXTRA = np.array([226, 120, 56])  # shadow where the target is empty
MISS = np.array([84, 146, 214])  # target the shadow did not reach
WALL = np.array([247, 245, 240])
ORDER = {"letters_upper": 0, "letters_lower": 1, "digits": 2}
SUBNAME = {"letters_upper": "Uppercase", "letters_lower": "Lowercase", "digits": "Digits"}
ADAM_MAIN = 64  # start points of the Adam configuration shown in the image grid


def load_mask(path):
    return (np.array(Image.open(path).convert("L")) < 128).astype(bool)


def png_uri(rgb):
    b = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def overlay(T, S):
    img = np.tile(WALL, T.shape + (1,))
    img[T & ~S] = MISS
    img[S & ~T] = EXTRA
    img[S & T] = INK
    return img


def plain(S):
    img = np.tile(WALL, S.shape + (1,))
    img[S] = INK
    return img


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def label(tid):
    sub, stem = tid.split("/")
    return stem.split("_")[0], {"letters_upper": "uppercase", "letters_lower": "lowercase", "digits": "digit"}.get(sub, sub)


def parity_examples(ids, best_q):
    """MuJoCo and nvdiffrast render the same stored CMA-ES pose."""
    fastshadow.use_fleet()
    from renderer import ShadowRenderer, build_scene

    from fastshadow.renderer import FastShadowRenderer

    model, data, nr = build_scene("urdf/SO101/so101_new_calib.urdf", n_robots=3, arm_gap=0.2)
    mj = ShadowRenderer(model, data, size=128, n_robots=nr)
    fr = FastShadowRenderer(mj)
    out = []
    for tid in ids:
        q = best_q[tid]
        A = mj.get_shadow_mask(q) > 0.5
        B = fr.get_shadow_mask(q) > 0.5
        diff = np.tile(WALL, A.shape + (1,))
        diff[A & B] = (200, 198, 205)
        diff[A & ~B] = MISS
        diff[B & ~A] = EXTRA
        out.append({"id": tid, "mj": png_uri(plain(A)), "fast": png_uri(plain(B)),
                    "diff": png_uri(diff), "iou": iou(A, B), "px": int((A ^ B).sum())})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="letters_digits")
    p.add_argument("--featured", type=int, default=2, help="featured examples per subset")
    p.add_argument("--out", default=os.path.join(FS, "report", "fastshadow_report.html"))
    a = p.parse_args()
    out = os.path.abspath(a.out)
    cdir = os.path.join(FS, f"compare_{a.tag}")
    fdir = os.path.join(FS, f"fastcma_{a.tag}")
    prow = {r["id"]: r for r in map(json.loads, open(os.path.join(cdir, "per_target.jsonl"), encoding="utf-8"))}
    runs = list(map(json.loads, open(os.path.join(fdir, "runs.jsonl"), encoding="utf-8")))
    parity = json.load(open(os.path.join(FS, "parity.json"), encoding="utf-8"))
    pset = next(s for s in parity["settings"] if s["cluster_mm"] == 2.0 and s["supersample"] == 1)

    by = defaultdict(lambda: defaultdict(list))  # method -> id -> [run]
    for r in runs:
        by[r["method"]][r["id"]].append(r)
    adamN = sorted(int(m.split("_")[1]) for m in by if m.startswith("adam_"))
    n_all = len(by["cma_fast"])
    complete = [N for N in adamN if len(by[f"adam_{N}"]) >= n_all]
    # the configured setting once it has every target, else the largest finished one
    main_m = f"adam_{ADAM_MAIN}" if ADAM_MAIN in complete else (f"adam_{complete[-1]}" if complete else None)
    # targets with every compared method
    need = ["cma", "cma_fast"] + ([main_m] if main_m else [])
    ids = sorted((i for i in prow if all(i in by[m] for m in need)),
                 key=lambda i: (ORDER.get(prow[i]["subset"], 9), i))
    if not ids:
        raise SystemExit("no target has all methods yet")

    def per_target(m, key="iou"):
        return np.array([np.mean([r[key] for r in by[m][i]]) for i in ids])

    def summary(m, sel=None):
        tids = [i for i in ids if i in by[m]] if sel is None else sel
        rs = [r for i in tids for r in by[m][i]]
        return {"iou": float(np.mean([np.mean([r["iou"] for r in by[m][i]]) for i in tids])),
                "sec": float(np.mean([r["seconds"] for r in rs])),
                "cf": float(np.mean([r["collision_free"] for r in rs])),
                "n_t": len(tids), "n_seeds": max(len(by[m][i]) for i in tids)}

    S = {m: summary(m, ids) for m in need}
    sweep = []
    for N in adamN:
        m = f"adam_{N}"
        tids = [i for i in ids if i in by[m]]
        if len(tids) == len(ids):
            s = summary(m, ids)
            sweep.append({"N": N, **s})
    # equal-time view: best-of-k seeds (picked by the solver's own loss) vs seconds
    def best_of(m, k):
        return float(np.mean([min(sorted(by[m][i], key=lambda r: r["seed"])[:k], key=lambda r: r["loss"])["iou"]
                              for i in ids]))

    def kcurve(m):
        ks = range(1, min(len(by[m][i]) for i in ids) + 1)
        sec = summary(m, ids)["sec"]
        return [{"k": k, "sec": k * sec, "iou": best_of(m, k)} for k in ks]

    budget = {"cma": kcurve("cma"), "cma_fast": kcurve("cma_fast"),
              "adam": [{"N": s["N"], "sec": s["sec"], "iou": s["iou"]} for s in sweep]}
    c_m, c_f = per_target("cma"), per_target("cma_fast")
    g = per_target(main_m) if main_m else c_f

    def paired(x, y):
        dl = y - x
        return float(dl.mean()), float(dl.std(ddof=1) / np.sqrt(len(dl))) if len(dl) > 1 else 0.0, \
            int((dl > 0.005).sum()), int((dl < -0.005).sum())

    d_fast = paired(c_m, c_f)
    d_adam = paired(c_f, g)

    featured = []
    for sub in ORDER:
        rs = [i for i in ids if prow[i]["subset"] == sub]
        if rs:
            idx = np.unique(np.linspace(0, len(rs) - 1, min(a.featured, len(rs))).round().astype(int))
            featured += [rs[i] for i in idx]

    def cells(tid):
        sub, stem = tid.split("/")
        T = load_mask(os.path.join(cdir, f"{sub}__{stem}__target.png"))
        base = os.path.join(fdir, f"{sub}__{stem}__")
        c = {"target": png_uri(plain(T))}
        for key, fn in (("cma", "cma0.png"), ("cma_fast", "cma_fast0.png"), ("adam", f"{main_m}.png")):
            Sm = load_mask(base + fn)
            c[key] = (png_uri(plain(Sm)), iou(T, Sm))  # the arms' shadow alone
        return c

    feat = [(i, cells(i)) for i in featured]
    allc = [(i, cells(i)) for i in ids]
    root = os.path.join(BENCH, "optimized", "big-budget-grounded")
    best_q = {}
    for i in featured[:4]:
        res = json.load(open(os.path.join(root, *i.split("/"), "results.json"), encoding="utf-8"))
        best_q[i] = np.array(res["runs"][prow[i]["cma"]["best_loss_run"]]["q_rad"])
    par = parity_examples(featured[:4], best_q)

    subsets = []
    for sub in ORDER:
        sel = [k for k, i in enumerate(ids) if prow[i]["subset"] == sub]
        if sel:
            subsets.append({"name": SUBNAME[sub], "n": len(sel), "cma": float(c_m[sel].mean()),
                            "fast": float(c_f[sel].mean()), "adam": float(g[sel].mean())})

    D = {
        "n": len(ids), "main_N": int(main_m.split("_")[1]) if main_m else None,
        "S": S, "sweep": sweep, "subsets": subsets, "budget": budget,
        "d_fast": d_fast, "d_adam": d_adam,
        "mj_ms": parity["mujoco_ms_per_pose"], "fast_ms": pset["ms_per_pose"],
        "par_iou": pset["solved"]["mean_iou"], "n_par": parity["n_solved"] + parity["n_random"],
        "speedup": parity["mujoco_ms_per_pose"] / pset["ms_per_pose"]["192"],
        "grad_ms": pset["soft_fwd_bwd_ms_per_pose_b48"], "tri": pset["triangles"], "tri_raw": pset["triangles_raw"],
        "scatter": [{"id": i, "c": float(c_f[k]), "g": float(g[k]), "s": prow[i]["subset"]} for k, i in enumerate(ids)],
    }
    from fastshadow.arrangement_report import load as load_arrangement

    D["arr"] = load_arrangement(BENCH)  # robot arrangement + spacing study, if run
    from fastshadow.rail_report import load as load_rail

    D["rail"] = load_rail(BENCH)  # LeSlider rail study (scripts/rail_study.py), if run
    html = render_html(D, feat, allc, par, main_m)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", out, f"({len(html) / 1e6:.2f} MB, {len(ids)} targets, sweep N={[s['N'] for s in sweep]})")


def render_html(D, feat, allc, par, main_m):
    def fig(uri, cap=None):
        c = f'<figcaption>{cap}</figcaption>' if cap is not None else ""
        return f'<figure class="m"><img src="{uri}" alt="" width="128" height="128">{c}</figure>'

    def iou_cap(v, best=False):
        return f'<span class="num{" best" if best else ""}">{v:.3f}</span>'

    frows = []
    for tid, c in feat:
        ch, kind = label(tid)
        top = max(c[k][1] for k in ("cma", "cma_fast", "adam"))
        cols = [fig(c["target"], '<span class="cap">target</span>')]
        cols += [fig(c[k][0], iou_cap(c[k][1], c[k][1] == top)) for k in ("cma", "cma_fast", "adam")]
        frows.append(f'<div class="frow"><div class="glyph"><b>{ch}</b><span>{kind}</span></div>{"".join(cols)}</div>')

    tiles = []
    for tid, c in allc:
        ch, _ = label(tid)
        dg = c["adam"][1] - c["cma_fast"][1]
        cls = "up" if dg > 0.005 else ("down" if dg < -0.005 else "tie")
        tiles.append(
            f'<div class="tile"><div class="trio">{fig(c["cma"][0])}{fig(c["cma_fast"][0])}{fig(c["adam"][0])}</div>'
            f'<div class="tl"><b>{ch}</b><span class="d {cls}">{dg:+.3f}</span></div></div>')

    prow = []
    for p in par:
        ch, kind = label(p["id"])
        dcap = f'<span class="num">{p["iou"]:.3f}</span> <span class="cap">· {p["px"]} px differ</span>'
        prow.append(
            f'<div class="prow"><div class="glyph"><b>{ch}</b><span>{kind}</span></div>'
            f'{fig(p["mj"], "<span class=cap>MuJoCo</span>")}{fig(p["fast"], "<span class=cap>nvdiffrast</span>")}'
            f'{fig(p["diff"], dcap)}</div>')

    return TEMPLATE.replace("/*DATA*/", json.dumps(D)).replace("<!--FEATURED-->", "".join(frows)) \
        .replace("<!--TILES-->", "".join(tiles)).replace("<!--PARITY-->", "".join(prow)) \
        .replace("{N}", str(D["n"])).replace("{AN}", str(D["main_N"]))


TEMPLATE = r"""<title>Shadow Solver Bench</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
/* Layout: a lab sheet. Shadows sit on a fixed "wall" tile in both themes; charts and numbers sit on the page ground. */
:root{
  --bg:#f2f3f5; --panel:#ffffff; --fg:#1d1e24; --muted:#5f6370; --rule:#d9dbe1;
  --cma:#8a8e9a; --fast:#1f9e8f; --grad:#3d5fd6;
  --extra:#e27838; --miss:#5492d6; --ink:#26242c; --wall:#f7f5f0;
  --good:#2f8a57; --bad:#c2473a;
  --f-display:"IBM Plex Sans Condensed","Arial Narrow",sans-serif;
  --f-body:"IBM Plex Sans",system-ui,sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#15161b; --panel:#1d1f26; --fg:#e8e9ee; --muted:#9a9eab; --rule:#2e313b;
  --cma:#9ea2ae; --fast:#3cc7b5; --grad:#7d97ff; --good:#5cc489; --bad:#ef7a6c; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#15161b; --panel:#1d1f26; --fg:#e8e9ee; --muted:#9a9eab; --rule:#2e313b;
  --cma:#9ea2ae; --fast:#3cc7b5; --grad:#7d97ff; --good:#5cc489; --bad:#ef7a6c; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.5 var(--f-body)}
.wrap{max-width:1080px;margin:0 auto;padding-block:28px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:40px}
h1{font:600 clamp(28px,4vw,40px)/1.1 var(--f-display);margin:0;letter-spacing:-.01em;text-wrap:balance}
h2{font:600 22px/1.2 var(--f-display);margin:0;text-wrap:balance}
.sub{color:var(--muted);margin:6px 0 0;max-width:62ch}
.eyebrow{font:500 11px/1 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.num{font-family:var(--f-mono);font-variant-numeric:tabular-nums}
section{display:grid;gap:16px}
.head{display:grid;gap:4px}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.kpi{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:14px 16px;display:grid;gap:4px;min-width:0}
.kpi .v{font:500 28px/1.1 var(--f-mono);font-variant-numeric:tabular-nums}
.kpi .l{color:var(--muted);font-size:13px}
.kpi .v.f{color:var(--fast)} .kpi .v.g{color:var(--grad)}
@media (max-width:760px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
figure.m{margin:0;display:grid;gap:4px;justify-items:center;min-width:0}
figure.m img{width:100%;max-width:128px;height:auto;aspect-ratio:1;image-rendering:pixelated;background:var(--wall);border-radius:3px;display:block}
figure.m figcaption{font-size:12px;color:var(--muted)}
figure.m .best{color:var(--grad);font-weight:500}
.cap{font-size:12px;color:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--muted)}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.grid-head,.frow{display:grid;grid-template-columns:72px repeat(4,minmax(0,128px));gap:14px;align-items:start;min-width:620px}
.grid-head{font:500 12px/1.2 var(--f-mono);letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
.grid-head .f{color:var(--fast)} .grid-head .g{color:var(--grad)}
.frow{padding-block:10px;border-top:1px solid var(--rule)}
.glyph{display:grid;gap:2px;padding-top:6px}
.glyph b{font:600 30px/1 var(--f-display)}
.glyph span{font-size:12px;color:var(--muted)}
.prow{display:grid;grid-template-columns:72px repeat(3,minmax(0,128px));gap:14px;align-items:start;padding-block:10px;border-top:1px solid var(--rule);min-width:480px}
.scroll{overflow-x:auto}
.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
@media (max-width:760px){.charts{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:16px;display:grid;gap:10px;min-width:0}
.card h3{font:600 15px/1.2 var(--f-display);margin:0}
.card .note{font-size:13px;color:var(--muted);margin:0}
svg text{fill:var(--muted);font:11px var(--f-mono)}
svg .lab{fill:var(--fg);font:12px var(--f-body)}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
.tile{display:grid;gap:4px}
.tile .trio{display:grid;grid-template-columns:repeat(3,1fr);gap:3px}
.tile .tl{display:flex;justify-content:space-between;align-items:baseline;font-size:13px}
.tile .tl b{font:600 15px var(--f-display)}
.d{font:12px var(--f-mono)}
.d.up{color:var(--good)} .d.down{color:var(--bad)} .d.tie{color:var(--muted)}
.foot{color:var(--muted);font-size:13px;max-width:70ch;margin:0}
table.cmp{border-collapse:collapse;width:100%;min-width:560px;font-size:14px}
table.cmp th,table.cmp td{text-align:left;padding:8px 10px;border-top:1px solid var(--rule);vertical-align:baseline}
table.cmp thead th{border-top:0;font:500 12px var(--f-mono);letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
table.cmp td.n{font-family:var(--f-mono);font-variant-numeric:tabular-nums}
table.cmp td.f{color:var(--fast);font-weight:500}
table.cmp .grp{color:var(--muted);font-size:12px}
#arr[hidden]{display:none}
#arr_kpis{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}
#arr_kpis .kpi .v{font-size:24px}
.arrgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}
.arrcard{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:12px;display:grid;gap:6px;align-content:start}
.arrcard h4{margin:0;font:600 14px var(--f-display)}
.arrcard p{margin:0;font-size:12px;color:var(--muted)}
.arrcard svg text{font:11px var(--f-mono)}
table.cmp td.best{color:var(--good);font-weight:600}
table.cmp .optc{font:600 11px var(--f-mono);padding:2px 6px;border-radius:3px;color:#fff}
.dgrid{display:grid;gap:10px;width:max-content}
.drow{display:grid;grid-template-columns:70px repeat(var(--nc),190px);gap:10px;align-items:start;padding-block:8px;border-top:1px solid var(--rule)}
.drow.h{font:500 11px var(--f-mono);color:var(--muted);text-transform:uppercase;border-top:0;padding-block:0}
.dlab{display:grid;gap:4px;justify-items:start}
.dlab b{font:600 28px/1 var(--f-display)}
.dlab img{width:64px;height:64px;image-rendering:pixelated;background:var(--wall);border-radius:3px}
.dcell{display:grid;gap:4px;background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:6px}
.dcell .pair{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.dcell img{width:100%;aspect-ratio:1;image-rendering:pixelated;background:var(--wall);border-radius:3px;display:block}
.dcell .cap{font:11px/1.3 var(--f-mono);color:var(--muted)}
.dcell .cap b{font-weight:600}
.dcell .win{outline:2px solid var(--good);outline-offset:-2px}
</style>
<div class="wrap">
  <header class="head">
    <span class="eyebrow">umbra-bench · letters &amp; digits · 3 × SO-101</span>
    <h1>Faster, differentiable shadows</h1>
    <p class="sub">The same shadows without MuJoCo's renderer, on the GPU. Three solvers on the same {N} letters and digits: CMA-ES on MuJoCo, the same CMA-ES on nvdiffrast, and Adam (gradient descent).</p>
  </header>

  <section class="kpis" id="kpis"></section>

  <section class="card">
    <h3>With MuJoCo vs without MuJoCo</h3>
    <div class="scroll"><table class="cmp" id="cmp"></table></div>
  </section>

  <section>
    <div class="head"><span class="eyebrow">1 · same pose, two renderers</span><h2>The new renderer casts the same shadow</h2></div>
    <div class="legend"><span><i style="background:#c8c6cd"></i>both</span><span><i style="background:var(--miss)"></i>MuJoCo only</span><span><i style="background:var(--extra)"></i>nvdiffrast only</span></div>
    <div class="scroll"><!--PARITY--></div>
  </section>

  <section class="charts">
    <div class="card"><h3>Time per pose</h3><p class="note">Poses rendered together in one GPU call. One pose alone is slower than MuJoCo.</p><div id="speed"></div></div>
    <div class="card"><h3>Seconds per solve</h3><p class="note">Same targets, this laptop. CMA-ES on nvdiffrast keeps MuJoCo only for kinematics and contacts.</p><div id="time"></div></div>
  </section>

  <section>
    <div class="head"><span class="eyebrow">2 · same targets, three solvers</span><h2>Shadows cast by each solver</h2></div>
    <div class="legend"><span>shadow cast by the arms, nothing overlaid</span><span class="num">number = IoU, best in blue · seed 0 of each</span></div>
    <div class="scroll">
      <div class="grid-head"><span></span><span>target</span><span>CMA-ES<br>MuJoCo</span><span class="f">CMA-ES<br>nvdiffrast</span><span class="g">Adam<br>{AN} starts</span></div>
      <!--FEATURED-->
    </div>
  </section>

  <section>
    <div class="head"><span class="eyebrow">3 · Adam</span><h2>How many start points does Adam need?</h2></div>
    <div class="charts">
      <div class="card"><h3>IoU per solve</h3><div id="sweep_iou"></div></div>
      <div class="card"><h3>Seconds per solve</h3><div id="sweep_sec"></div></div>
    </div>
    <div class="legend"><span><i style="background:var(--grad)"></i>Adam, N start points</span><span><i style="background:var(--fast)"></i>CMA-ES on nvdiffrast</span><span><i style="background:var(--cma)"></i>CMA-ES on MuJoCo</span></div>
    <div class="card"><h3>Same time budget: IoU vs seconds</h3><p class="note">Adam spends time on more start points. CMA-ES spends it on more seeds and keeps the solve with the lowest loss (1, 2, 3 seeds).</p><div id="budget"></div></div>
  </section>

  <section class="charts">
    <div class="card"><h3>Every target: Adam vs CMA-ES (both on nvdiffrast)</h3><p class="note">Mean IoU per solve. Above the line = Adam better.</p><div id="scatter"></div></div>
    <div class="card"><h3>Mean IoU per solve, by set</h3><div id="subsets"></div></div>
  </section>

  <section>
    <div class="head"><span class="eyebrow">all {N} targets</span><h2>CMA-ES · MuJoCo, CMA-ES · nvdiffrast, Adam</h2></div>
    <div class="legend"><span>number = Adam IoU minus CMA-ES (nvdiffrast) IoU, seed 0</span></div>
    <div class="tiles"><!--TILES--></div>
  </section>

  <section id="arr" hidden>
    <div class="head"><span class="eyebrow">4 · robot arrangement · digits 0–9</span><h2>Let the robots move: arrangement and spacing as variables</h2></div>
    <p class="sub" id="arr_sub"></p>
    <div class="kpis" id="arr_kpis"></div>
    <div class="arrgrid" id="arr_ill"></div>
    <div class="card"><h3>Every arrangement, both optimizers</h3><p class="note" id="arr_tnote"></p><div class="scroll"><table class="cmp" id="arr_table"></table></div></div>
    <div class="charts">
      <div class="card"><h3>Best IoU by arrangement</h3><p class="note">Best of the seeds, mean over the 10 digits. Faded bar: mean IoU of a single solve.</p><div id="arr_bars"></div></div>
      <div class="card"><h3>IoU for the time spent</h3><p class="note">Best-of-k seeds against the seconds those k solves took, mean over digits. Solid: the best arrangement for that optimizer; dashed: the fixed column.</p><div id="arr_budget"></div></div>
    </div>
    <div class="head"><h2 style="font-size:18px">Every digit</h2></div>
    <div class="legend"><span><i style="background:#0f6e66"></i>CMA-ES (best seed)</span><span><i style="background:#3a3fb0"></i>Adam (best seed)</span><span>top-down sketch: optimized base positions of both, light above, wall below</span><span>⚠ = MuJoCo contact</span></div>
    <div class="scroll"><div class="dgrid" id="arr_digits"></div></div>
  </section>

  <section id="rail" hidden>
    <div class="head"><span class="eyebrow">5 · LeSlider · carriages on a rail</span><h2>Slide each arm along the light axis while solving</h2></div>
    <p class="sub" id="rail_sub"></p>
    <div class="kpis" id="rail_kpis"></div>
    <div class="card"><h3>Where the carriages went</h3><p class="note" id="rail_snote"></p><div class="scroll" id="rail_sketch"></div></div>
    <div class="card"><h3>Same targets, carriages fixed vs free</h3><p class="note" id="rail_tnote"></p><div class="scroll"><table class="cmp" id="rail_table"></table></div></div>
    <div class="card" id="rail_geo_card" hidden><h3>Where the rail sits matters more than sliding on it</h3><p class="note" id="rail_gnote"></p><div class="scroll"><table class="cmp" id="rail_geo"></table></div></div>
    <div class="head"><h2 style="font-size:18px">Biggest gains and losses from moving the carriages (CMA-ES, best seed)</h2></div>
    <div class="scroll"><div class="grid-head" style="grid-template-columns:72px repeat(3,minmax(0,128px));min-width:480px"><span></span><span class="cap">target (fitted to the rail)</span><span class="cap">carriages fixed</span><span class="cap">carriages free</span></div><div id="rail_ex"></div></div>
  </section>
  <p class="foot" id="foot"></p>
</div>
<script>
const D = /*DATA*/;
const f3 = v => v.toFixed(3), sg = v => (v >= 0 ? '+' : '') + f3(v);
const C = D.S.cma, F = D.S.cma_fast, A = D.S['adam_' + D.main_N];

document.getElementById('kpis').innerHTML = [
  [f3(D.par_iou), 'IoU between MuJoCo and nvdiffrast shadows of the same pose', ''],
  [D.speedup.toFixed(1) + '×', 'faster per pose than MuJoCo (batch of 192)', ''],
  [(C.sec / F.sec).toFixed(1) + '×', `faster CMA-ES solve (${C.sec.toFixed(1)} s → ${F.sec.toFixed(1)} s), IoU ${sg(D.d_fast[0])}`, 'f'],
  [sg(D.d_adam[0]), `IoU, Adam (${D.main_N} starts) minus CMA-ES on nvdiffrast`, 'g'],
].map(([v, l, c]) => `<div class="kpi"><span class="v ${c}">${v}</span><span class="l">${l}</span></div>`).join('');

document.getElementById('cmp').innerHTML =
  '<thead><tr><th></th><th>with MuJoCo</th><th>without MuJoCo (nvdiffrast)</th></tr></thead><tbody>' +
  [['grp', 'Renderer, same poses'],
   ['Time per pose', D.mj_ms.toFixed(2) + ' ms', D.fast_ms['192'].toFixed(2) + ' ms in a batch of 192 · ' + D.fast_ms['1'].toFixed(2) + ' ms alone'],
   ['Shadow', 'reference', 'IoU ' + f3(D.par_iou) + ' with MuJoCo'],
   ['Gradient', 'none', D.grad_ms.toFixed(2) + ' ms per pose, forward + backward'],
   ['grp', `CMA-ES, same algorithm and budget, ${D.n} targets × ${C.n_seeds} seeds`],
   ['IoU per solve', f3(C.iou), f3(F.iou)],
   ['Seconds per solve', C.sec.toFixed(2) + ' s', F.sec.toFixed(2) + ' s'],
   ['Collision-free', (100 * C.cf).toFixed(0) + '%', (100 * F.cf).toFixed(0) + '%'],
   ['grp', `Adam, ${D.main_N} start points (needs gradients, so nvdiffrast only)`],
   ['IoU per solve', '—', f3(A.iou)],
   ['Seconds per solve', '—', A.sec.toFixed(2) + ' s']]
  .map(r => r[0] === 'grp' ? `<tr><td colspan="3" class="grp">${r[1]}</td></tr>`
    : `<tr><th>${r[0]}</th><td class="n">${r[1]}</td><td class="n f">${r[2]}</td></tr>`).join('') + '</tbody>';

function svg(w, h, body) { return `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img">${body}</svg>`; }
function hbars(id, rows, unit, ticks) {
  const W = 460, L = 190, R = 56, bh = 22, gap = 12, H = rows.length * (bh + gap) + 24;
  const mx = Math.max(...rows.map(r => r[1])) * 1.08, x = v => L + (W - L - R) * v / mx;
  let b = '';
  ticks.filter(t => t <= mx).forEach(t => { b += `<line x1="${x(t)}" x2="${x(t)}" y1="0" y2="${H - 18}" stroke="var(--rule)"/><text x="${x(t)}" y="${H - 4}" text-anchor="middle">${t} ${unit}</text>`; });
  rows.forEach(([n, v, c], i) => { const y = i * (bh + gap);
    b += `<text class="lab" x="${L - 8}" y="${y + bh / 2 + 4}" text-anchor="end">${n}</text><rect x="${L}" y="${y}" width="${Math.max(1, x(v) - L)}" height="${bh}" rx="2" fill="${c}"/><text x="${x(v) + 6}" y="${y + bh / 2 + 4}">${v.toFixed(2)}</text>`; });
  document.getElementById(id).innerHTML = svg(W, H, b);
}
hbars('speed', [['MuJoCo, 1 pose', D.mj_ms, 'var(--cma)'], ['nvdiffrast, 1 pose', D.fast_ms['1'], 'var(--fast)'],
  ['nvdiffrast, batch 48', D.fast_ms['48'], 'var(--fast)'], ['nvdiffrast, batch 192', D.fast_ms['192'], 'var(--fast)'],
  ['gradient step, batch 48', D.grad_ms, 'var(--grad)']], 'ms', [0, 0.5, 1, 1.5]);
hbars('time', [['CMA-ES · MuJoCo', C.sec, 'var(--cma)'], ['CMA-ES · nvdiffrast', F.sec, 'var(--fast)'],
  [`Adam · ${D.main_N} starts`, A.sec, 'var(--grad)']], 's', [0, 1, 2, 3, 4, 5, 6, 8, 10]);

// start-point sweep: line over log2(N) with CMA-ES reference lines
function sweep(id, key, fmt, unit) {
  const P = D.sweep; if (!P.length) { document.getElementById(id).innerHTML = '<p class="note">Sweep still running.</p>'; return; }
  const W = 460, H = 240, L = 44, R = 96, T = 12, B = 30;
  const refs = [[C[key], 'var(--cma)', 'CMA-ES · MuJoCo'], [F[key], 'var(--fast)', 'CMA-ES · nvdiffrast']];
  const vs = P.map(p => p[key]).concat(refs.map(r => r[0]));
  let lo = Math.min(...vs), hi = Math.max(...vs); const pad = (hi - lo) * 0.12 || 0.01; lo -= pad; hi += pad;
  if (key === 'sec') lo = 0;
  const lx = P.map(p => Math.log2(p.N)), x0 = Math.min(...lx), x1 = Math.max(...lx);
  const x = n => L + (W - L - R) * (Math.log2(n) - x0) / Math.max(1e-9, x1 - x0), y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  let b = '';
  const nt = 4; for (let k = 0; k <= nt; k++) { const v = lo + (hi - lo) * k / nt;
    b += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--rule)"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${fmt(v)}</text>`; }
  P.forEach(p => { b += `<text x="${x(p.N)}" y="${H - 10}" text-anchor="middle">${p.N}</text>`; });
  b += `<text x="${(L + W - R) / 2}" y="${H}" text-anchor="middle">start points</text>`;
  refs.forEach(([v, c, n]) => { b += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="${c}" stroke-width="2" stroke-dasharray="5 4"/><text x="${W - R + 6}" y="${y(v) + 4}" style="fill:${c}">${fmt(v)}${unit}</text>`; });
  b += `<polyline fill="none" stroke="var(--grad)" stroke-width="2.5" points="${P.map(p => `${x(p.N)},${y(p[key])}`).join(' ')}"/>`;
  P.forEach(p => { b += `<circle cx="${x(p.N)}" cy="${y(p[key])}" r="4.5" fill="var(--grad)"><title>${p.N} start points: ${fmt(p[key])}${unit}</title></circle>`; });
  const e = P[P.length - 1]; b += `<text x="${x(e.N) + 8}" y="${y(e[key]) - 8}" style="fill:var(--grad)">${fmt(e[key])}${unit}</text>`;
  document.getElementById(id).innerHTML = svg(W, H + 4, b);
}
(function () {
  const Bd = D.budget, W = 920, H = 300, L = 48, R = 24, T = 14, B = 34;
  const all = [...Bd.cma, ...Bd.cma_fast, ...Bd.adam];
  if (!Bd.adam.length) return;
  const xmax = Math.ceil(Math.max(...all.map(p => p.sec)) + 0.5);
  let lo = Math.min(...all.map(p => p.iou)), hi = Math.max(...all.map(p => p.iou)); const pd = (hi - lo) * 0.12; lo -= pd; hi += pd;
  const x = v => L + (W - L - R) * v / xmax, y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  let b = '';
  for (let t = 0; t <= xmax; t += 2) b += `<line x1="${x(t)}" x2="${x(t)}" y1="${T}" y2="${H - B}" stroke="var(--rule)"/><text x="${x(t)}" y="${H - B + 14}" text-anchor="middle">${t} s</text>`;
  for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4; b += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--rule)"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${f3(v)}</text>`; }
  b += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">seconds per target</text>`;
  [[Bd.cma, 'var(--cma)', p => p.k + '×'], [Bd.cma_fast, 'var(--fast)', p => p.k + '×'], [Bd.adam, 'var(--grad)', p => p.N]].forEach(([P, c, lab]) => {
    b += `<polyline fill="none" stroke="${c}" stroke-width="2.5" points="${P.map(p => `${x(p.sec)},${y(p.iou)}`).join(' ')}"/>`;
    P.forEach(p => { b += `<circle cx="${x(p.sec)}" cy="${y(p.iou)}" r="4.5" fill="${c}"><title>${f3(p.iou)} in ${p.sec.toFixed(1)} s</title></circle><text x="${x(p.sec)}" y="${y(p.iou) - 9}" text-anchor="middle" style="fill:${c}">${lab(p)}</text>`; });
  });
  document.getElementById('budget').innerHTML = `<div class="scroll">${svg(W, H, b)}</div>` +
    `<div class="legend"><span><i style="background:var(--fast)"></i>CMA-ES · nvdiffrast, k seeds</span><span><i style="background:var(--cma)"></i>CMA-ES · MuJoCo, k seeds</span><span><i style="background:var(--grad)"></i>Adam, N start points</span><span>higher and further left is better</span></div>`;
})();
sweep('sweep_iou', 'iou', f3, '');
sweep('sweep_sec', 'sec', v => v.toFixed(1), ' s');

(function () {
  const P = D.scatter, W = 460, H = 320, L = 40, B = 30, T = 10, R = 10;
  const vals = P.flatMap(p => [p.c, p.g]); const lo = Math.max(0, Math.floor(Math.min(...vals) * 10) / 10), hi = Math.min(1, Math.ceil(Math.max(...vals) * 10) / 10);
  const x = v => L + (W - L - R) * (v - lo) / (hi - lo), y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  const col = {letters_upper: 'var(--grad)', letters_lower: 'var(--extra)', digits: 'var(--good)'};
  let b = '';
  for (let t = lo; t <= hi + 1e-9; t += 0.1) { b += `<line x1="${x(t)}" x2="${x(t)}" y1="${T}" y2="${H - B}" stroke="var(--rule)"/><line x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}" stroke="var(--rule)"/><text x="${x(t)}" y="${H - B + 14}" text-anchor="middle">${t.toFixed(1)}</text><text x="${L - 6}" y="${y(t) + 4}" text-anchor="end">${t.toFixed(1)}</text>`; }
  b += `<line x1="${x(lo)}" y1="${y(lo)}" x2="${x(hi)}" y2="${y(hi)}" stroke="var(--muted)" stroke-dasharray="4 4"/>`;
  P.forEach(p => { b += `<circle cx="${x(p.c)}" cy="${y(p.g)}" r="4.5" fill="${col[p.s] || 'var(--fg)'}" fill-opacity="0.85"><title>${p.id}: CMA-ES ${f3(p.c)}, Adam ${f3(p.g)}</title></circle>`; });
  b += `<text x="${(L + W - R) / 2}" y="${H - 2}" text-anchor="middle">CMA-ES (nvdiffrast) IoU</text><text x="12" y="${(T + H - B) / 2}" transform="rotate(-90 12 ${(T + H - B) / 2})" text-anchor="middle">Adam IoU</text>`;
  document.getElementById('scatter').innerHTML = svg(W, H, b) +
    `<div class="legend"><span><i style="background:var(--grad)"></i>uppercase</span><span><i style="background:var(--extra)"></i>lowercase</span><span><i style="background:var(--good)"></i>digits</span><span class="num">Adam better on ${D.d_adam[2]} · worse on ${D.d_adam[3]} · of ${D.n}</span></div>`;
})();

(function () {
  const S = D.subsets, W = 460, H = 230, L = 36, B = 30, T = 14;
  const vs = S.flatMap(s => [s.cma, s.fast, s.adam]); const lo = Math.floor(Math.min(...vs) * 10) / 10 - 0.1, hi = Math.min(1, Math.ceil(Math.max(...vs) * 10) / 10);
  const y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  const gw = (W - L - 10) / S.length, bw = Math.min(30, gw / 4);
  let b = '';
  for (let t = lo; t <= hi + 1e-9; t += 0.1) b += `<line x1="${L}" x2="${W}" y1="${y(t)}" y2="${y(t)}" stroke="var(--rule)"/><text x="${L - 6}" y="${y(t) + 4}" text-anchor="end">${t.toFixed(1)}</text>`;
  S.forEach((s, i) => { const cx = L + gw * i + gw / 2;
    [[s.cma, 'var(--cma)'], [s.fast, 'var(--fast)'], [s.adam, 'var(--grad)']].forEach(([v, c], k) => {
      const x0 = cx + (k - 1.5) * (bw + 3);
      b += `<rect x="${x0}" y="${y(v)}" width="${bw}" height="${H - B - y(v)}" rx="2" fill="${c}"><title>${f3(v)}</title></rect><text x="${x0 + bw / 2}" y="${y(v) - 4}" text-anchor="middle" style="font-size:10px">${v.toFixed(2)}</text>`; });
    b += `<text class="lab" x="${cx}" y="${H - 10}" text-anchor="middle">${s.name} (${s.n})</text>`; });
  document.getElementById('subsets').innerHTML = svg(W, H, b) +
    `<div class="legend"><span><i style="background:var(--cma)"></i>CMA-ES · MuJoCo</span><span><i style="background:var(--fast)"></i>CMA-ES · nvdiffrast</span><span><i style="background:var(--grad)"></i>Adam</span></div>`;
})();

// ── 4 · robot arrangement study ────────────────────────────────────────
(function () {
  const A = D.arr; if (!A) return;
  document.getElementById('arr').hidden = false;
  const OPT = {cma: ['CMA-ES', '#0f6e66'], adam: ['Adam', '#3a3fb0']};
  const NAME = {fixed: 'Fixed column (0.2 m)', ...Object.fromEntries(Object.entries(A.meta.arrangements).map(([k, v]) => [k, v.label]))};
  const R2 = Math.SQRT1_2;
  const POS = {
    fixed: () => [[0, 0], [0, -0.2], [0, -0.4]],
    column: (x, y) => [[0, 0], [0, -x], [0, -x - y]],
    row: (x, y) => [[-x, -0.2], [0, -0.2], [y, -0.2]],
    triangle: (x, y) => [[0, 0], [-x / 2, -y], [x / 2, -y]],
    diagonal: (x, y) => [[0, 0], [x * R2, -x * R2], [(x + y) * R2, -(x + y) * R2]],
  };
  const lo = 0.12, hi = 0.45;
  // top-down sketch: x lateral (right), depth y (up toward the light)
  function sketch(sets, opt = {}) {
    const W = opt.w || 200, H = opt.h || 190;
    const X0 = opt.mini ? -0.75 : -0.6, X1 = -X0, Y0 = opt.mini ? 0.2 : 0.15, Y1 = opt.mini ? -1.0 : -0.95;
    const sx = v => 12 + (W - 24) * (v - X0) / (X1 - X0), sy = v => 22 + (H - 44) * (v - Y0) / (Y1 - Y0);
    const k = (W - 24) / (X1 - X0);
    let b = `<rect x="0" y="0" width="${W}" height="${H}" fill="var(--bg)" rx="4"/>`;
    b += `<text x="${W / 2}" y="13" text-anchor="middle" style="fill:var(--muted)">☀ light</text>`;
    b += `<line x1="10" x2="${W - 10}" y1="${H - 9}" y2="${H - 9}" stroke="var(--fg)" stroke-width="3"/><text x="${W / 2}" y="${H - 13}" text-anchor="middle" style="fill:var(--muted)">wall</text>`;
    sets.forEach(({pts, color, dims, labels}) => {
      (dims || []).forEach(([a, c, lab, dashed]) => {
        const [x1, y1] = a, [x2, y2] = c;
        b += `<line x1="${sx(x1)}" y1="${sy(y1)}" x2="${sx(x2)}" y2="${sy(y2)}" stroke="var(--extra)" stroke-width="1.6" ${dashed ? 'stroke-dasharray="3 3"' : ''} marker-start="url(#ar)" marker-end="url(#ar)"/>`;
        const mx = (sx(x1) + sx(x2)) / 2, my = (sy(y1) + sy(y2)) / 2, vert = Math.abs(x1 - x2) < 1e-6;
        const left = vert && dashed;  // a side marker (triangle's y) is labelled on its outer side
        b += `<text x="${mx + (vert ? (left ? -8 : 10) : 0)}" y="${my + (vert ? 4 : -7)}" text-anchor="${vert ? (left ? 'end' : 'start') : 'middle'}" style="fill:var(--extra);font-weight:600">${lab}</text>`;
      });
      pts.forEach(([x, y], i) => {
        const r = Math.max(0.05 * k, opt.mini ? 5 : 7);  // a 10 cm base, but never too small to see
        b += `<circle cx="${sx(x)}" cy="${sy(y)}" r="${r}" fill="${color}" fill-opacity="${opt.mini ? 0.85 : 1}"/>`;
        b += `<path d="M${sx(x) - r * 0.5},${sy(y) + r * 0.2} L${sx(x) + r * 0.5},${sy(y) + r * 0.2} L${sx(x)},${sy(y) + r * 1.1} Z" fill="var(--bg)"/>`;
        if (labels !== false) b += `<text x="${sx(x)}" y="${sy(y) - r - 3}" text-anchor="middle" style="fill:var(--fg)">${i + 1}</text>`;
      });
    });
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img"><defs><marker id="ar" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--extra)"/></marker></defs>${b}</svg>`;
  }
  function dimsFor(c, p) {
    if (c === 'fixed') return [[p[0], p[1], '0.2'], [p[1], p[2], '0.2']];
    if (c === 'triangle') { const xo = p[1][0] - 0.12; return [[p[1], p[2], 'x'], [[xo, p[0][1]], [xo, p[1][1]], 'y', true]]; }
    return [[p[0], p[1], 'x'], [p[1], p[2], 'y']];
  }
  document.getElementById('arr_ill').innerHTML = A.configs.map(c => {
    const p = c === 'fixed' ? POS.fixed() : POS[c](0.28, 0.28);
    const m = A.meta.arrangements[c];
    return `<div class="arrcard"><h4>${NAME[c]}</h4>${sketch([{pts: p, color: 'var(--fg)', dims: dimsFor(c, p)}])}<p>${c === 'fixed' ? 'The rig so far: spacing fixed, only the 18 joints move.' : `x: ${m.x}<br>y: ${m.y}<br>both ${lo}–${hi} m, optimized with the joints`}</p></div>`;
  }).join('');

  const S = A.summary, f3 = v => v.toFixed(3);
  const bestRow = o => S.filter(r => r.opt === o).reduce((a, b) => b.best_iou > a.best_iou ? b : a);
  const fixedRow = o => S.find(r => r.opt === o && r.config === 'fixed');
  const overall = S.reduce((a, b) => b.best_iou > a.best_iou ? b : a);
  const bc = A.opts.includes('cma') ? bestRow('cma') : null, ba = A.opts.includes('adam') ? bestRow('adam') : null;
  document.getElementById('arr_sub').textContent = `Three SO-101s on sliders. Light, wall and camera stay put; the bases move in the table plane. Each arrangement has two spacings, x and y, optimized together with the 18 joint angles. Objective: ${A.meta.objective} (IoU + two-sided distance), the shape objective people preferred. ${A.n} digits; CMA-ES ${A.meta.args.cma_seeds} seeds, Adam ${A.meta.args.adam_seeds} seeds of ${A.meta.args.adam_restarts} starts. Every result re-rendered and contact-checked by MuJoCo with the optimized base positions.`;
  const kp = [[f3(overall.best_iou), `best single arrangement: ${NAME[overall.config]}, ${OPT[overall.opt][0]}`, OPT[overall.opt][1]]];
  if (A.oracle && A.oracle.any) {
    const O = A.oracle.any, fx = fixedRow(overall.opt), wins = Object.entries(O.opts).map(([o, n]) => `${OPT[o][0]} ${n}`).join(', ');
    kp.unshift([f3(O.iou), `each digit in its own best arrangement${fx ? ` (${(O.iou - fx.best_iou >= 0 ? '+' : '') + f3(O.iou - fx.best_iou)} vs fixed column)` : ''}; best per digit: ${wins} of ${A.n}`, 'var(--good)']);
  }
  A.opts.forEach(o => {
    const f = fixedRow(o), b = S.filter(r => r.opt === o && r.config !== 'fixed').reduce((x, y) => y.best_iou > x.best_iou ? y : x, {best_iou: -1});
    if (f && b.config) kp.push([(b.best_iou - f.best_iou >= 0 ? '+' : '') + f3(b.best_iou - f.best_iou), `${OPT[o][0]}: best movable arrangement (${NAME[b.config].split(' (')[0].toLowerCase()}) minus the fixed column`, OPT[o][1]]);
  });
  if (bc && ba) kp.push([`${bc.sec.toFixed(1)} / ${ba.sec.toFixed(1)} s`, 'per solve, CMA-ES / Adam (best arrangements)', 'var(--fg)']);
  document.getElementById('arr_kpis').innerHTML = kp.map(([v, l, c]) => `<div class="kpi"><span class="v" style="color:${c}">${v}</span><span class="l">${l}</span></div>`).join('');

  const cols = [['best_iou', 'best IoU', 1], ['mean_iou', 'IoU / solve', 1], ['best_tc', 'tc_iou', 1], ['best_cls', 'classifier', 1], ['cf', 'collision-free', 1], ['sec', 's / solve', 0], ['sec_target', 's / digit', 0]];
  if (S.every(r => r.judge != null)) cols.splice(4, 0, ['judge', 'Ivy judge', 1], ['judge_pick', 'judge, judge-picked', 1], ['judge_pick_iou', 'IoU, judge-picked', 1]);
  const best = {}; cols.forEach(([k, , hb]) => { const v = S.map(r => r[k]); best[k] = hb ? Math.max(...v) : Math.min(...v); });
  document.getElementById('arr_tnote').textContent = 'Best IoU, tc_iou, classifier and Ivy judge are the best seed per digit (by IoU), averaged over digits; "judge-picked" lets the Ivy judge choose the seed instead (the evaluation layer: P(Ivy marks it good), trained on your picks). IoU / solve, collision-free and seconds average every solve. s / digit = all seeds for one digit. x̄, ȳ: mean optimized spacing of the best seeds; "at 0.12 m" = share of best seeds with a spacing on the lower limit.';
  document.getElementById('arr_table').innerHTML = `<thead><tr><th>arrangement</th><th>optimizer</th>${cols.map(c => `<th>${c[1]}</th>`).join('')}<th>x̄ (m)</th><th>ȳ (m)</th><th>at 0.12 m</th></tr></thead><tbody>` +
    S.map(r => `<tr><th>${NAME[r.config]}</th><td><span class="optc" style="background:${OPT[r.opt][1]}">${OPT[r.opt][0]}</span></td>` +
      cols.map(([k, , hb]) => `<td class="n ${Math.abs(r[k] - best[k]) < 1e-9 ? 'best' : ''}">${k === 'cf' || k === 'best_cls' ? Math.round(100 * r[k]) + '%' : k.startsWith('sec') ? r[k].toFixed(1) : f3(r[k])}</td>`).join('') +
      `<td class="n">${r.config === 'fixed' ? '0.20' : r.gap_x.toFixed(2)}</td><td class="n">${r.config === 'fixed' ? '0.20' : r.gap_y.toFixed(2)}</td><td class="n">${r.at_lo == null ? '–' : Math.round(100 * r.at_lo) + '%'}</td></tr>`).join('') + '</tbody>';

  // grouped bars
  (function () {
    const W = 520, L = 150, R = 60, bh = 13, gap = 3, grp = 12;
    const rows = []; A.configs.forEach(c => A.opts.forEach(o => rows.push(S.find(r => r.config === c && r.opt === o))));
    const vals = rows.flatMap(r => [r.best_iou, r.mean_iou]);
    const x0 = Math.floor(Math.min(...vals) * 10) / 10, x1 = Math.min(1, Math.ceil(Math.max(...vals) * 20) / 20);
    const x = v => L + (W - L - R) * (v - x0) / (x1 - x0);
    let b = '', y = 4;
    A.configs.forEach(c => {
      b += `<text class="lab" x="${L - 8}" y="${y + bh + 4}" text-anchor="end">${NAME[c].split(' (')[0]}</text>`;
      A.opts.forEach(o => { const r = S.find(q => q.config === c && q.opt === o);
        b += `<rect x="${x(x0)}" y="${y}" width="${Math.max(1, x(r.mean_iou) - x(x0))}" height="${bh}" fill="${OPT[o][1]}" opacity=".3"/><rect x="${x(x0)}" y="${y}" width="${Math.max(1, x(r.best_iou) - x(x0))}" height="${bh}" fill="${OPT[o][1]}" opacity=".9"><title>${OPT[o][0]} ${NAME[c]}: best ${f3(r.best_iou)}, per solve ${f3(r.mean_iou)}</title></rect><text x="${x(r.best_iou) + 6}" y="${y + bh - 2}">${f3(r.best_iou)}</text>`;
        y += bh + gap; });
      y += grp;
    });
    for (let t = x0; t <= x1 + 1e-9; t += 0.05) b += `<line x1="${x(t)}" x2="${x(t)}" y1="0" y2="${y - grp}" stroke="var(--rule)"/><text x="${x(t)}" y="${y + 6}" text-anchor="middle">${t.toFixed(2)}</text>`;
    document.getElementById('arr_bars').innerHTML = svg(W, y + 12, b) + `<div class="legend"><span><i style="background:#0f6e66"></i>CMA-ES</span><span><i style="background:#3a3fb0"></i>Adam</span></div>`;
  })();

  // budget curves
  (function () {
    const lines = [];
    A.opts.forEach(o => { const b = bestRow(o); lines.push([o, b.config, false]); if (b.config !== 'fixed' && A.curves['fixed|' + o]) lines.push([o, 'fixed', true]); });
    const pts = lines.flatMap(([o, c]) => A.curves[c + '|' + o]);
    const W = 520, H = 280, L = 48, R = 20, T = 12, B = 34;
    const xmax = Math.max(...pts.map(p => p.sec)) * 1.05;
    let y0 = Math.min(...pts.map(p => p.iou)), y1 = Math.max(...pts.map(p => p.iou)); const pd = (y1 - y0) * 0.1 || 0.01; y0 -= pd; y1 += pd;
    const x = v => L + (W - L - R) * v / xmax, y = v => T + (H - T - B) * (1 - (v - y0) / (y1 - y0));
    let b = '';
    const step = xmax > 100 ? 20 : xmax > 40 ? 10 : 5;
    for (let t = 0; t <= xmax; t += step) b += `<line x1="${x(t)}" x2="${x(t)}" y1="${T}" y2="${H - B}" stroke="var(--rule)"/><text x="${x(t)}" y="${H - B + 14}" text-anchor="middle">${t}s</text>`;
    for (let k = 0; k <= 4; k++) { const v = y0 + (y1 - y0) * k / 4; b += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--rule)"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(3)}</text>`; }
    b += `<text x="${(L + W - R) / 2}" y="${H - 2}" text-anchor="middle">seconds per digit (k solves)</text>`;
    lines.forEach(([o, c, dashed]) => { const P = A.curves[c + '|' + o];
      b += `<polyline fill="none" stroke="${OPT[o][1]}" stroke-width="2.4" ${dashed ? 'stroke-dasharray="6 4"' : ''} points="${P.map(p => `${x(p.sec)},${y(p.iou)}`).join(' ')}"/>`;
      P.forEach(p => { b += `<circle cx="${x(p.sec)}" cy="${y(p.iou)}" r="${dashed ? 2.5 : 3.5}" fill="${OPT[o][1]}"><title>${OPT[o][0]} ${NAME[c]}: best of ${p.k} = ${f3(p.iou)} in ${p.sec.toFixed(1)} s</title></circle>`; });
      const e = P[P.length - 1]; b += `<text x="${Math.min(x(e.sec), W - R - 4)}" y="${y(e.iou) - 8}" text-anchor="end" style="fill:${OPT[o][1]}">${OPT[o][0]} · ${NAME[c].split(' (')[0].toLowerCase()}</text>`; });
    document.getElementById('arr_budget').innerHTML = svg(W, H, b);
  })();

  // digits
  (function () {
    const nc = A.configs.length;
    let h = `<div class="drow h" style="--nc:${nc}"><span>digit</span>${A.configs.map(c => `<span>${NAME[c]}</span>`).join('')}</div>`;
    A.digits.forEach(d => {
      const top = Math.max(...d.cells.flatMap(c => A.opts.map(o => c[o].iou)));
      h += `<div class="drow" style="--nc:${nc}"><div class="dlab"><b>${d.digit}</b><img src="${d.target}" alt="target ${d.digit}"><span class="cap">target</span></div>` +
        d.cells.map(c => {
          const sets = A.opts.map(o => ({pts: c[o].bases, color: OPT[o][1], labels: false}));
          return `<div class="dcell">${sketch(sets, {w: 176, h: 120, mini: true})}<div class="pair">` +
            A.opts.map(o => `<img src="${c[o].img}" alt="${OPT[o][0]} ${d.digit}" class="${Math.abs(c[o].iou - top) < 1e-9 ? 'win' : ''}" loading="lazy">`).join('') + '</div>' +
            A.opts.map(o => `<div class="cap" style="color:${OPT[o][1]}"><b>${OPT[o][0]}</b> ${f3(c[o].iou)} · ${c[o].sec.toFixed(1)}s${c.config === 'fixed' ? '' : ` · x ${c[o].gaps[0].toFixed(2)} y ${c[o].gaps[1].toFixed(2)}`}${c[o].cf ? '' : ' ⚠'}</div>`).join('') + '</div>';
        }).join('') + '</div>';
    });
    document.getElementById('arr_digits').innerHTML = h + `<p class="note" style="margin:6px 0 0">Green frame: best shadow for that digit across all arrangements and both optimizers. Times are the best seed's own solve; x, y in metres.</p>`;
  })();
})();

document.getElementById('foot').textContent =
  `IoU is measured with MuJoCo for every method, against the target the stored CMA-ES solver was shown (big-budget-grounded, --fit-target). ` +
  `CMA-ES: popsize 48, 16 + 16 per-arm iterations, 30 joint iterations, ${C.n_seeds} seeds. Adam: 300 steps on blurred soft IoU plus the CMA-ES penalties, ${A.n_seeds} seeds. ` +
  `Paired differences: CMA-ES nvdiffrast − MuJoCo ${sg(D.d_fast[0])} ± ${f3(D.d_fast[1])}; Adam − CMA-ES nvdiffrast ${sg(D.d_adam[0])} ± ${f3(D.d_adam[1])} (SE over targets). ` +
  `Renderer: ${D.tri.toLocaleString()} triangles after 2 mm vertex clustering (from ${D.tri_raw.toLocaleString()}).`;

// ── LeSlider rail ─────────────────────────────────────────────────────
(function () {
  const R = D.rail; if (!R) return;
  document.getElementById('rail').hidden = false;
  const S = R.summary, C = S.cells, t = R.meta.track, fe = S.free_minus_even;
  const mag = s => (t.light_dist + t.length + t.screen_dist) / (t.light_dist + s);
  const k = C['rail_free/cma'].n_seeds, ka = C['rail_free/adam'].n_seeds;
  document.getElementById('rail_sub').textContent = `${R.meta.n} SO-101s on fleet-shadow-art's 1 m LeSlider (track.leslider_1m): light ${t.light_dist} m before the rail, screen ${t.screen_dist} m after it, base travel ${t.s_min.toFixed(2)}–${t.s_max.toFixed(2)} m from the light end, carriages at least ${t.min_spacing.toFixed(3)} m apart. Each carriage position is one more variable, solved together with the 18 joints on the GPU (CMA-ES populations; Adam gets a gradient through the kinematics). ${R.meta.n_targets} letters and digits, each fitted once to this rail's reach with the carriages evenly spread, the same target for both settings. MuJoCo re-renders and contact-checks every result with the carriages where the solver put them.`;
  const ci = v => `${sg(v[0])} (95% ${sg(v[1])}…${sg(v[2])}, ${v[3]} up / ${v[4]} down)`;
  document.getElementById('rail_kpis').innerHTML = [
    [sg(fe.cma.iou_best[0]), `IoU, best of ${k} CMA-ES seeds: carriages free minus fixed`, 'f'],
    [sg(fe.adam.iou_best[0]), `IoU, best of ${ka} Adam seeds: carriages free minus fixed`, 'g'],
    [C['rail_free/cma'].s_mean.map(v => v.toFixed(2)).join(' · '), `m from the light end, where CMA-ES puts the carriages on average (fixed: ${R.meta.s_even.map(v => v.toFixed(2)).join(' · ')})`, ''],
    [`${mag(t.s_max).toFixed(1)}–${mag(t.s_min).toFixed(1)}×`, 'shadow magnification along this rail (the bench rig: 2.4–3.4×)', ''],
  ].map(([v, l, c]) => `<div class="kpi"><span class="v ${c}">${v}</span><span class="l">${l}</span></div>`).join('');
  // top-down sketch: light, rail, screen; fixed carriages; every free solve's carriages
  (function () {
    const W = 920, H = 160, L = 60, Rr = 60, y0 = 46;
    const total = t.light_dist + t.length + t.screen_dist;
    const x = d => L + (W - L - Rr) * d / total;  // d = metres from the light
    let b = `<circle cx="${x(0)}" cy="${y0}" r="9" fill="none" stroke="var(--ink)" stroke-width="2.5"/><text x="${x(0)}" y="${y0 - 16}" text-anchor="middle">light</text>`;
    b += `<line x1="${x(total)}" x2="${x(total)}" y1="${y0 - 26}" y2="${y0 + 26}" stroke="var(--ink)" stroke-width="4"/><text x="${x(total)}" y="${y0 - 32}" text-anchor="middle">screen</text>`;
    b += `<rect x="${x(t.light_dist)}" y="${y0 - 5}" width="${x(t.light_dist + t.length) - x(t.light_dist)}" height="10" rx="3" fill="var(--rule)"/>`;
    b += `<line x1="${x(t.light_dist + t.s_min)}" x2="${x(t.light_dist + t.s_max)}" y1="${y0 + 14}" y2="${y0 + 14}" stroke="var(--muted)" stroke-dasharray="3 3"/><text x="${x(t.light_dist + (t.s_min + t.s_max) / 2)}" y="${y0 + 28}" text-anchor="middle">base travel</text>`;
    R.meta.s_even.forEach(s => { b += `<rect x="${x(t.light_dist + s) - 6}" y="${y0 - 12}" width="12" height="24" rx="2" fill="none" stroke="var(--muted)" stroke-width="2"><title>fixed carriage at ${s.toFixed(3)} m</title></rect>`; });
    [['cma', 'var(--fast)', 100], ['adam', 'var(--grad)', 134]].forEach(([o, c, yy]) => {
      b += `<text x="${L - 8}" y="${yy + 4}" text-anchor="end" style="fill:${c}">${o === 'cma' ? 'CMA-ES' : 'Adam'}</text>`;
      R.spots[o].forEach((ss, j) => ss.forEach((s, i) => { b += `<circle cx="${x(t.light_dist + s)}" cy="${yy + ((j % 5) - 2) * 3}" r="2.6" fill="${c}" opacity="${0.35 + 0.2 * i}"/>`; }));
    });
    document.getElementById('rail_sketch').innerHTML = svg(W, H, b);
    document.getElementById('rail_snote').textContent = `Top-down, to scale. Hollow boxes: the fixed setting (evenly spread). Dots: every free solve's three carriages (darker = further from the light). Closer to the light = bigger shadow.`;
  })();
  const lab = {rail_even: 'carriages fixed (evenly spread)', rail_free: 'carriages free'};
  const rows = Object.entries(C).map(([key, v]) => { const [c, o] = key.split('/');
    return `<tr><th>${lab[c]} · ${o === 'cma' ? 'CMA-ES' : 'Adam'}</th><td class="n">${f3(v.iou)}</td><td class="n">${f3(v.iou_best)}</td><td class="n">${f3(v.tc_iou)}</td><td class="n">${(100 * v.cls_top1).toFixed(0)}%</td><td class="n">${(100 * v.collision_free).toFixed(0)}% · ${(100 * v.floor_ok).toFixed(0)}%</td><td class="n">${v.seconds.toFixed(1)} s</td></tr>`; });
  const o = S.old_rig;
  rows.push(`<tr><td colspan="7" class="grp">The bench rig (fixed 0.2 m column, light 1 m before, screen 3.4 m from the light): its IoU is against a different fitted target, so only shape and reading compare</td></tr>`);
  rows.push(`<tr><th>bench rig · CMA-ES (nvdiffrast)</th><td class="n">—</td><td class="n">—</td><td class="n">${f3(o.cma_fast.tc_iou)}</td><td class="n">${(100 * o.cma_fast.cls_top1).toFixed(0)}%</td><td class="n">—</td><td class="n">—</td></tr>`);
  rows.push(`<tr><th>bench rig · Adam, 64 starts</th><td class="n">—</td><td class="n">—</td><td class="n">${f3(o.adam_64.tc_iou)}</td><td class="n">—</td><td class="n">—</td><td class="n">—</td></tr>`);
  document.getElementById('rail_table').innerHTML = `<thead><tr><th></th><th>IoU per solve</th><th>best of seeds</th><th>tc_iou</th><th>classifier</th><th>contact-free · floor ok</th><th>time</th></tr></thead><tbody>${rows.join('')}</tbody>`;
  const rmo = S.rail_minus_old_tc || {};
  document.getElementById('rail_tnote').textContent = `Mean over ${R.meta.n_targets} targets; CMA-ES ${k} seeds, Adam ${ka} seeds of 64 starts. Free minus fixed, paired by target: CMA-ES IoU ${ci(fe.cma.iou)}, best seed ${ci(fe.cma.iou_best)}; Adam IoU ${ci(fe.adam.iou)}, best seed ${ci(fe.adam.iou_best)}. tc_iou is the shape against the original glyph after centring, comparable across rigs: on this rail it is ${sg(rmo['rail_free/cma'][0])} below the bench rig for CMA-ES with free carriages (95% ${sg(rmo['rail_free/cma'][1])}…${sg(rmo['rail_free/cma'][2])}), because the arms sit 1.13–1.95 m from the light here (bench rig 1.0–1.4 m) and the screen is closer, so their shadows are smaller. The rail's placement matters more than sliding on it.`;
  if (R.variants && R.variants.length > 1) {
    document.getElementById('rail_geo_card').hidden = false;
    document.getElementById('rail_geo').innerHTML = '<thead><tr><th>rail placement</th><th>magnification</th><th>solver</th><th>best IoU, fixed → free</th><th>tc_iou, free</th><th>tc_iou vs bench rig</th><th>classifier, free</th></tr></thead><tbody>' +
      R.variants.map(v => ['cma', 'adam'].map((o, j) => { const fc = v.cells['rail_free/' + o], ec = v.cells['rail_even/' + o], d = v.rail_minus_old_tc['rail_free/' + o];
        return `<tr>${j ? '<th></th><td></td>' : `<th>light ${v.light_dist} m before, screen ${v.screen_dist} m after</th><td class="n">${v.mag[0].toFixed(1)}–${v.mag[1].toFixed(1)}×</td>`}<td>${o === 'cma' ? 'CMA-ES' : 'Adam'}</td><td class="n">${f3(ec.iou_best)} → ${f3(fc.iou_best)}</td><td class="n">${f3(fc.tc_iou)}</td><td class="n">${d ? sg(d[0]) + ' (' + sg(d[1]) + '…' + sg(d[2]) + ')' : '—'}</td><td class="n">${(100 * fc.cls_top1).toFixed(0)}%</td></tr>`; }).join('')).join('') + '</tbody>';
    document.getElementById('rail_gnote').textContent = 'Same 30 characters, targets fitted to each placement; best of the seeds (CMA-ES 3, Adam 2). tc_iou against the bench rig (fixed 0.2 m column, 2.4–3.4×, CMA-ES 3 seeds or Adam 64 starts) with its 95% range, paired by character. Bench rig classifier: 53% (CMA-ES). Light close to the rail and the screen far (bigger shadows) brings the rail level with the bench rig; sliding the carriages helps only when the shadows are small.';
  }
  document.getElementById('rail_ex').innerHTML = R.examples.map(e => `<div class="frow" style="grid-template-columns:72px repeat(3,minmax(0,128px));min-width:480px"><div class="glyph"><b>${e.ch}</b><span>${sg(e.gain)}</span></div>` +
    `<figure class="m"><img src="${e.target}" alt="target ${e.ch}"><figcaption>target</figcaption></figure>` +
    ['rail_even', 'rail_free'].map(c => `<figure class="m"><img src="${e[c].img}" alt="${c} ${e.ch}"><figcaption><span class="num">${f3(e[c].iou)}</span> · ${e[c].s.map(v => v.toFixed(2)).join('/')} m</figcaption></figure>`).join('') + `</div>`).join('');
})();
</script>
"""

if __name__ == "__main__":
    main()
