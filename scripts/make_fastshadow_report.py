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
            c[key] = (png_uri(overlay(T, Sm)), iou(T, Sm))
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
    <div class="legend"><span><i style="background:var(--ink)"></i>shadow on target</span><span><i style="background:var(--extra)"></i>extra shadow</span><span><i style="background:var(--miss)"></i>missed target</span><span class="num">number = IoU, best in blue · seed 0 of each</span></div>
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

document.getElementById('foot').textContent =
  `IoU is measured with MuJoCo for every method, against the target the stored CMA-ES solver was shown (big-budget-grounded, --fit-target). ` +
  `CMA-ES: popsize 48, 16 + 16 per-arm iterations, 30 joint iterations, ${C.n_seeds} seeds. Adam: 300 steps on blurred soft IoU plus the CMA-ES penalties, ${A.n_seeds} seeds. ` +
  `Paired differences: CMA-ES nvdiffrast − MuJoCo ${sg(D.d_fast[0])} ± ${f3(D.d_fast[1])}; Adam − CMA-ES nvdiffrast ${sg(D.d_adam[0])} ± ${f3(D.d_adam[1])} (SE over targets). ` +
  `Renderer: ${D.tri.toLocaleString()} triangles after 2 mm vertex clustering (from ${D.tri_raw.toLocaleString()}).`;
</script>
"""

if __name__ == "__main__":
    main()
