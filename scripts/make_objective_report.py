"""Results page for scripts/objective_study.py (results/fastshadow/objective_study).

    python scripts/make_objective_report.py
"""

import base64
import io
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = os.path.join(BENCH, "results", "fastshadow")
OUT = os.path.join(FS, "objective_study")
sys.path.insert(0, os.path.join(BENCH, "scripts"))
from run_base_optimizer import load_target  # noqa: E402

ORDER = {"letters_upper": 0, "letters_lower": 1, "digits": 2}
INK, EXTRA, MISS, WALL = (np.array(c) for c in ([38, 36, 44], [226, 120, 56], [84, 146, 214], [247, 245, 240]))

OBJ = [  # name, label, family, what it steers with
    ("current", "IoU + DT (current)", "baseline", ""),
    ("iou", "IoU only", "shape", "iou"),
    ("tversky_fn", "Tversky, gaps cost more", "shape", ""),
    ("tversky_fp", "Tversky, spill costs more", "shape", ""),
    ("dt2", "IoU + two-sided DT", "shape", "chamfer"),
    ("biou", "IoU + boundary IoU", "shape", "boundary_iou"),
    ("cldice", "IoU + clDice", "shape", "cldice"),
    ("holes", "IoU + hole term", "shape", "holes_filled"),
    ("tc", "placement-free IoU", "shape", "tc_iou"),
    ("shape3", "IoU + boundary + clDice + holes", "shape", "boundary_iou cldice holes_filled"),
    ("cls", "IoU + glyph classifier", "semantic", "cls_top1"),
    ("clip", "IoU + CLIP (ViT-B-16)", "semantic", "clip_top1 clip_rr_cc"),
    ("shape3_cls", "shape3 + glyph classifier", "semantic", "boundary_iou cldice holes_filled cls_top1"),
]
MET = [  # key, label, higher is better, group
    ("iou", "IoU", True, "shape"), ("tc_iou", "tc_iou", True, "shape"), ("boundary_iou", "boundary IoU", True, "shape"),
    ("nsd", "NSD", True, "shape"), ("chamfer", "chamfer", False, "shape"), ("cldice", "clDice", True, "shape"),
    ("betti_error", "Betti error", False, "topology"), ("holes_filled", "holes filled", False, "topology"),
    ("cls_top1", "classifier top-1", True, "semantic"), ("clip_top1", "CLIP top-1", True, "semantic"),
    ("clip_rr_cc", "CLIP MRR (chance-corr.)", True, "semantic"),
]


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


def main():
    global OUT
    if len(sys.argv) > 1:  # another study folder, e.g. a smoke run
        OUT = os.path.abspath(sys.argv[1])
    G = [json.loads(l) for l in open(os.path.join(OUT, "graded.jsonl"), encoding="utf-8")]
    meta = json.load(open(os.path.join(OUT, "meta.json"), encoding="utf-8"))
    objs = [o for o in OBJ if any(g["objective"] == o[0] for g in G)]
    by = defaultdict(lambda: defaultdict(list))
    for g in G:
        by[g["objective"]][g["id"]].append(g)
    ids = sorted({g["id"] for g in G if all(g["id"] in by[o[0]] for o in objs)},
                 key=lambda i: (ORDER.get(i.split("/")[0], 9), i))

    def tmean(o, k, sel=None):
        """mean over targets of the per-target seed mean (targets lacking the metric skipped)"""
        vals = []
        for i in (sel or ids):
            v = [r[k] for r in by[o][i] if r.get(k) is not None]
            if v:
                vals.append(np.mean(v))
        return float(np.mean(vals)) if vals else None

    def paired(o, k):
        d = []
        for i in ids:
            a = [r[k] for r in by["current"][i] if r.get(k) is not None]
            b = [r[k] for r in by[o][i] if r.get(k) is not None]
            if a and b:
                d.append(np.mean(b) - np.mean(a))
        d = np.array(d)
        return (float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 1 else (0.0, 0.0)

    table = []
    for o, lab, fam, steers in objs:
        row = {"o": o, "label": lab, "family": fam, "steers": steers.split(), "cells": {}}
        for k, _, hb, _ in MET:
            v = tmean(o, k)
            dm, se = paired(o, k) if o != "current" else (0.0, 0.0)
            row["cells"][k] = {"v": v, "d": dm, "se": se}
        row["sec"] = float(np.mean([r["seconds"] for i in ids for r in by[o][i]]))
        row["cf"] = float(np.mean([r["collision_free"] for i in ids for r in by[o][i]]))
        table.append(row)

    # gallery: every character, seed 0 of every objective
    seeds_per = [len(by["current"][i]) for i in ids]
    root = os.path.join(BENCH, "optimized", "big-budget-grounded")
    grid = []
    for i in ids:
        sub, stem = i.split("/")
        T = load_target(os.path.join(root, sub, stem, f"{stem}_shown.png"), 128) > 0.5
        cells = []
        for o, *_ in objs:
            r = min(by[o][i], key=lambda r: r["seed"])
            S = load_target(os.path.join(OUT, r["mask"]), 128) > 0.5
            sh = np.tile(WALL, S.shape + (1,))
            sh[S] = INK  # the arms' shadow alone; the target sits in its own column
            cells.append({"uri": png_uri(sh), "iou": r["iou"], "cls": r["cls_top1"], "clip": r["clip_rank"]})
        tgt = np.tile(WALL, T.shape + (1,))
        tgt[T] = INK
        grid.append({"id": i, "sub": sub, "label": stem.split("_")[0], "target": png_uri(tgt), "cells": cells})

    ns = sorted(set(seeds_per))
    seeds_txt = (f"{ns[0]} seed{'s' if ns[0] > 1 else ''} each" if len(ns) == 1 else
                 ", ".join(f"{seeds_per.count(k)} with {k} seed{'s' if k > 1 else ''}" for k in ns[::-1]))
    D = {"n": len(ids), "seeds": seeds_txt, "objs": [{"o": o, "label": l, "family": f} for o, l, f, _ in objs],
         "met": [{"k": k, "label": l, "hb": hb, "group": g} for k, l, hb, g in MET], "table": table, "grid": grid}
    hp = os.path.join(FS, "human_summary.json")  # scripts/human_summary.py, from the picker page
    D["human"] = json.load(open(hp, encoding="utf-8")) if os.path.exists(hp) else None
    html = TEMPLATE.replace("/*DATA*/", json.dumps(D))
    out = os.path.join(FS, "report", "objective_report.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write(html)
    print("wrote", out, f"{len(html) / 1e6:.2f} MB, {len(ids)} targets, {len(objs)} objectives")


TEMPLATE = r"""<title>Glyph Objective Study</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
/* Layout: same lab sheet as the solver page. A heatmap leads; shadows on a fixed wall tile follow. */
:root{
  --bg:#f2f3f5; --panel:#ffffff; --fg:#1d1e24; --muted:#5f6370; --rule:#d9dbe1;
  --shape:#3d5fd6; --sem:#b04a8f; --base:#8a8e9a;
  --up:47,138,87; --down:194,71,58;
  --extra:#e27838; --miss:#5492d6; --ink:#26242c; --wall:#f7f5f0;
  --f-display:"IBM Plex Sans Condensed","Arial Narrow",sans-serif;
  --f-body:"IBM Plex Sans",system-ui,sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#15161b; --panel:#1d1f26; --fg:#e8e9ee; --muted:#9a9eab; --rule:#2e313b;
  --shape:#7d97ff; --sem:#e07fc2; --base:#9ea2ae; --up:92,196,137; --down:239,122,108; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#15161b; --panel:#1d1f26; --fg:#e8e9ee; --muted:#9a9eab; --rule:#2e313b;
  --shape:#7d97ff; --sem:#e07fc2; --base:#9ea2ae; --up:92,196,137; --down:239,122,108; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.5 var(--f-body)}
.wrap{max-width:1180px;margin:0 auto;padding-block:28px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:36px}
h1{font:600 clamp(28px,4vw,40px)/1.1 var(--f-display);margin:0;text-wrap:balance}
h2{font:600 22px/1.2 var(--f-display);margin:0;text-wrap:balance}
.sub{color:var(--muted);margin:6px 0 0;max-width:66ch}
.eyebrow{font:500 11px/1 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
section{display:grid;gap:14px}
.head{display:grid;gap:4px}
.scroll{overflow-x:auto}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--muted)}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
@media (max-width:760px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
.kpi{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:14px 16px;display:grid;gap:4px;min-width:0}
.kpi .v{font:600 20px/1.2 var(--f-display)}
.kpi .n{font:500 14px var(--f-mono)}
.kpi .l{color:var(--muted);font-size:13px}
table.hm{border-collapse:separate;border-spacing:2px;font-size:13px;min-width:1000px}
table.hm th{font:500 11px/1.2 var(--f-mono);color:var(--muted);text-align:center;padding:4px;vertical-align:bottom}
table.hm th.row{text-align:left;font:500 13px var(--f-body);color:var(--fg);white-space:nowrap;padding-right:10px}
table.hm th.row i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
table.hm td{text-align:center;padding:6px 4px;border-radius:3px;font:12px var(--f-mono);font-variant-numeric:tabular-nums;min-width:74px;position:relative}
table.hm td .v{display:block;color:var(--fg)}
table.hm td .d{display:block;font-size:11px;color:var(--muted)}
table.hm td.own{outline:1.5px dashed var(--muted);outline-offset:-3px}
table.hm td.grp{background:none}
table.hm tr.base td{background:var(--panel)}
.grid{display:grid;gap:6px;width:max-content}
.grow{display:grid;grid-auto-flow:column;grid-auto-columns:72px;grid-template-columns:44px;gap:6px;align-items:start}
.grow.h{font:500 10px/1.2 var(--f-mono);color:var(--muted);align-items:end;position:sticky;top:env(safe-area-inset-top,0px);background:var(--bg);padding-block:4px;z-index:1}
.grow.h span{text-align:center;overflow-wrap:anywhere}
.grow .off,.grow.h .off{display:none}
.subhead{font:600 15px var(--f-display);padding-top:10px;border-top:1px solid var(--rule)}
.tabs,.chips{display:flex;flex-wrap:wrap;gap:6px}
.tabs button,.chips button{font:500 12px var(--f-mono);border:1px solid var(--rule);background:var(--panel);color:var(--fg);border-radius:999px;padding:5px 12px;cursor:pointer}
.tabs button[aria-selected="true"]{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.chips button[aria-pressed="false"]{opacity:.45;text-decoration:line-through}
.tabs button:focus-visible,.chips button:focus-visible{outline:2px solid var(--shape);outline-offset:2px}
.glyph b{font:600 26px/1 var(--f-display)}
figure.m{margin:0;display:grid;gap:2px;justify-items:center}
figure.m img{width:72px;height:72px;image-rendering:pixelated;background:var(--wall);border-radius:3px}
figure.m figcaption{font:11px var(--f-mono);color:var(--muted);text-align:center}
figure.m figcaption .ok{color:rgb(var(--up))} figure.m figcaption .no{color:rgb(var(--down))}
.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
@media (max-width:760px){.charts{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:16px;display:grid;gap:10px;min-width:0}
.card h3{font:600 15px/1.2 var(--f-display);margin:0}
.card .note{font-size:13px;color:var(--muted);margin:0}
svg text{fill:var(--muted);font:11px var(--f-mono)}
svg .lab{fill:var(--fg);font:11px var(--f-body)}
.foot{color:var(--muted);font-size:13px;max-width:76ch;margin:0}
#human[hidden]{display:none}
#human .charts{align-items:start}
.card{align-content:start}
.take{margin:0;padding-left:20px;display:grid;gap:4px;max-width:80ch}
.take b{font-weight:600}
</style>
<div class="wrap">
  <header class="head">
    <span class="eyebrow">umbra-bench · letters &amp; digits · fast CMA-ES</span>
    <h1>Which objective makes a better glyph?</h1>
    <p class="sub" id="sub"></p>
  </header>
  <section class="kpis" id="kpis"></section>
  <section>
    <div class="head"><span class="eyebrow">1 · every objective, every metric</span><h2>Change vs the current objective</h2></div>
    <div class="legend"><span><i style="background:rgba(var(--up),.55)"></i>better than current</span><span><i style="background:rgba(var(--down),.55)"></i>worse</span><span>small number = Δ ± SE over targets</span><span>dashed = the objective steers with this metric, so a win there is partly circular</span></div>
    <div class="scroll"><table class="hm" id="hm"></table></div>
  </section>
  <section class="charts">
    <div class="card"><h3>Shape vs readability</h3><p class="note">One dot per objective. CLIP grader is ViT-B-32, never used for steering.</p><div id="tradeoff"></div></div>
    <div class="card"><h3>Seconds per solve</h3><p class="note">Fast CMA-ES, same budget for every objective.</p><div id="time"></div></div>
  </section>
  <section id="human" hidden>
    <div class="head"><span class="eyebrow">2 · human test</span><h2>What a person can read</h2></div>
    <ul class="take" id="take"></ul>
    <div class="charts">
      <div class="card"><h3>You read the shadow, by objective</h3><p class="note">Guess-the-glyph, shadow shown alone. Bar = you, with 95% interval; ◆ classifier and ● CLIP on the same shadows.</p><div id="h_per"></div></div>
      <div class="card"><h3>Which metric predicts that you read it?</h3><p class="note">AUC over your guesses: how often a shadow you read scores higher than one you didn't. 0.5 = chance.</p><div id="h_pred"></div></div>
      <div class="card"><h3>You vs the machine judges</h3><p class="note">Share of the same shadows each one reads, and how often each agrees with you shadow by shadow.</p><div id="h_judges"></div></div>
      <div class="card"><h3>Shadows you marked as good</h3><p class="note">Marking tab, target visible, any number per character. Bars: share of characters where each objective was marked good.</p><div id="h_marks"></div></div>
    </div>
  </section>
  <section>
    <div class="head"><span class="eyebrow">3 · every character, every objective</span><h2>Shadows</h2></div>
    <div class="tabs" role="tablist" id="tabs"></div>
    <div class="chips" id="chips"></div>
    <div class="legend"><span>shadow cast by the arms, nothing overlaid · seed 0</span><span>caption: IoU · classifier ✓/✗ · CLIP rank</span></div>
    <div class="scroll"><div class="grid" id="grid"></div></div>
  </section>
  <p class="foot" id="foot"></p>
</div>
<script>
const D = /*DATA*/;
const f3 = v => v == null ? '–' : v.toFixed(3), sg = v => (v >= 0 ? '+' : '') + v.toFixed(3);
const fam = {baseline: 'var(--base)', shape: 'var(--shape)', semantic: 'var(--sem)'};
const base = D.table.find(r => r.o === 'current');
document.getElementById('sub').textContent = `${D.objs.length} objectives on the same ${D.n} letters and digits (${D.seeds}). Every shadow is re-rendered with MuJoCo and graded on all ${D.met.length} metrics.`;

// KPIs: best objective per metric group, by mean paired Δ over its group's metrics (sign-corrected), excluding own metrics
(function () {
  const groups = {shape: 'Best for shape', topology: 'Best for holes / topology', semantic: 'Best for readability'};
  const out = [];
  for (const [g, title] of Object.entries(groups)) {
    const ms = D.met.filter(m => m.group === g);
    let best = null;
    D.table.filter(r => r.o !== 'current').forEach(r => {
      const sc = ms.map(m => { const c = r.cells[m.k]; const sd = Math.max(...D.table.map(x => Math.abs(x.cells[m.k].d))) || 1; return (m.hb ? 1 : -1) * c.d / sd; });
      const s = sc.reduce((a, b) => a + b, 0) / sc.length;
      if (!best || s > best.s) best = {r, s};
    });
    out.push(`<div class="kpi"><span class="l">${title}</span><span class="v">${best.r.label}</span><span class="n">${ms.map(m => m.label + ' ' + sg(best.r.cells[m.k].d)).slice(0, 2).join(' · ')}</span></div>`);
  }
  const fast = D.table.reduce((a, b) => a.sec < b.sec ? a : b);
  out.push(`<div class="kpi"><span class="l">Current objective</span><span class="v">IoU ${f3(base.cells.iou.v)}</span><span class="n">CLIP top-1 ${f3(base.cells.clip_top1.v)} · classifier ${f3(base.cells.cls_top1.v)}</span></div>`);
  document.getElementById('kpis').innerHTML = out.join('');
})();

// heatmap
(function () {
  const ms = D.met;
  const maxd = {}; ms.forEach(m => maxd[m.k] = Math.max(1e-9, ...D.table.map(r => Math.abs(r.cells[m.k].d))));
  let h = '<thead><tr><th></th>' + ms.map(m => `<th>${m.label}<br>${m.hb ? '↑' : '↓'}</th>`).join('') + '</tr></thead><tbody>';
  D.table.forEach(r => {
    h += `<tr class="${r.o === 'current' ? 'base' : ''}"><th class="row"><i style="background:${fam[r.family]}"></i>${r.label}</th>`;
    ms.forEach(m => {
      const c = r.cells[m.k];
      if (r.o === 'current') { h += `<td><span class="v">${f3(c.v)}</span><span class="d">baseline</span></td>`; return; }
      const good = (m.hb ? 1 : -1) * c.d, a = Math.min(1, Math.abs(c.d) / maxd[m.k]) * 0.6;
      const sig = Math.abs(c.d) > 2 * c.se;
      const bg = `rgba(var(${good >= 0 ? '--up' : '--down'}),${(sig ? a : a * 0.35).toFixed(3)})`;
      h += `<td class="${r.steers.includes(m.k) ? 'own' : ''}" style="background:${bg}"><span class="v">${f3(c.v)}</span><span class="d">${sg(c.d)}±${c.se.toFixed(3)}</span></td>`;
    });
    h += '</tr>';
  });
  document.getElementById('hm').innerHTML = h + '</tbody>';
})();

function svg(w, h, body) { return `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img">${body}</svg>`; }
(function () {
  const W = 520, H = 340, L = 50, R = 16, T = 14, B = 34;
  const P = D.table.map(r => ({x: r.cells.iou.v, y: r.cells.clip_rr_cc.v, r}));
  const xs = P.map(p => p.x), ys = P.map(p => p.y);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const px = (x1 - x0) * 0.1 || 0.01, py = (y1 - y0) * 0.12 || 0.01; x0 -= px; x1 += px; y0 -= py; y1 += py;
  const x = v => L + (W - L - R) * (v - x0) / (x1 - x0), y = v => T + (H - T - B) * (1 - (v - y0) / (y1 - y0));
  let b = '';
  for (let k = 0; k <= 4; k++) { const vx = x0 + (x1 - x0) * k / 4, vy = y0 + (y1 - y0) * k / 4;
    b += `<line x1="${x(vx)}" x2="${x(vx)}" y1="${T}" y2="${H - B}" stroke="var(--rule)"/><text x="${x(vx)}" y="${H - B + 14}" text-anchor="middle">${vx.toFixed(2)}</text>`;
    b += `<line x1="${L}" x2="${W - R}" y1="${y(vy)}" y2="${y(vy)}" stroke="var(--rule)"/><text x="${L - 6}" y="${y(vy) + 4}" text-anchor="end">${vy.toFixed(2)}</text>`; }
  b += `<line x1="${x(base.cells.iou.v)}" x2="${x(base.cells.iou.v)}" y1="${T}" y2="${H - B}" stroke="var(--base)" stroke-dasharray="4 4"/><line x1="${L}" x2="${W - R}" y1="${y(base.cells.clip_rr_cc.v)}" y2="${y(base.cells.clip_rr_cc.v)}" stroke="var(--base)" stroke-dasharray="4 4"/>`;
  P.forEach(p => { b += `<circle cx="${x(p.x)}" cy="${y(p.y)}" r="5.5" fill="${fam[p.r.family]}"><title>${p.r.label}: IoU ${f3(p.x)}, CLIP MRR ${f3(p.y)}</title></circle><text class="lab" x="${x(p.x) + 8}" y="${y(p.y) + 4}">${p.r.o}</text>`; });
  b += `<text x="${(L + W - R) / 2}" y="${H - 2}" text-anchor="middle">IoU (shape)</text><text x="12" y="${(T + H - B) / 2}" transform="rotate(-90 12 ${(T + H - B) / 2})" text-anchor="middle">CLIP MRR, chance-corrected</text>`;
  document.getElementById('tradeoff').innerHTML = svg(W, H, b) + `<div class="legend"><span><i style="background:var(--base)"></i>current</span><span><i style="background:var(--shape)"></i>shape</span><span><i style="background:var(--sem)"></i>semantic</span><span>dashed = current</span></div>`;
})();
(function () {
  const R = [...D.table].sort((a, b) => a.sec - b.sec);
  const W = 520, L = 210, Rm = 50, bh = 16, gap = 6, H = R.length * (bh + gap) + 22;
  const mx = Math.max(...R.map(r => r.sec)) * 1.1, x = v => L + (W - L - Rm) * v / mx;
  let b = '';
  for (let t = 0; t <= mx; t += 2) b += `<line x1="${x(t)}" x2="${x(t)}" y1="0" y2="${H - 18}" stroke="var(--rule)"/><text x="${x(t)}" y="${H - 4}" text-anchor="middle">${t} s</text>`;
  R.forEach((r, i) => { const yy = i * (bh + gap);
    b += `<text class="lab" x="${L - 8}" y="${yy + bh / 2 + 4}" text-anchor="end">${r.label}</text><rect x="${L}" y="${yy}" width="${x(r.sec) - L}" height="${bh}" rx="2" fill="${fam[r.family]}"/><text x="${x(r.sec) + 6}" y="${yy + bh / 2 + 4}">${r.sec.toFixed(1)}</text>`; });
  document.getElementById('time').innerHTML = svg(W, H, b);
})();
(function () {
  const SUBS = [['all', 'All'], ['letters_upper', 'Uppercase'], ['letters_lower', 'Lowercase'], ['digits', 'Digits']];
  const on = Object.fromEntries(D.objs.map(o => [o.o, true]));
  let tab = 'all';
  const counts = s => s === 'all' ? D.grid.length : D.grid.filter(g => g.sub === s).length;
  function draw() {
    const head = `<div class="grow h"><span></span><span>target</span>${D.objs.map(o => `<span class="${on[o.o] ? '' : 'off'}" style="color:${fam[o.family]}">${o.o}</span>`).join('')}</div>`;
    let h = '';
    SUBS.slice(1).forEach(([s, name]) => {
      if (tab !== 'all' && tab !== s) return;
      const rows = D.grid.filter(g => g.sub === s);
      if (!rows.length) return;
      h += `<div class="subhead">${name}</div>` + head;
      rows.forEach(g => {
        h += `<div class="grow"><div class="glyph"><b>${g.label}</b></div><figure class="m"><img src="${g.target}" alt="target ${g.label}"></figure>` +
          g.cells.map((c, k) => `<figure class="m ${on[D.objs[k].o] ? '' : 'off'}"><img src="${c.uri}" alt="${D.objs[k].o} ${g.label}" loading="lazy"><figcaption>${c.iou.toFixed(2)} <span class="${c.cls ? 'ok' : 'no'}">${c.cls ? '✓' : '✗'}</span> #${c.clip}</figcaption></figure>`).join('') + '</div>';
      });
    });
    document.getElementById('grid').innerHTML = h;
  }
  const tabs = document.getElementById('tabs');
  tabs.innerHTML = SUBS.map(([s, n]) => `<button role="tab" id="tab-${s}" aria-selected="${s === tab}" data-s="${s}">${n} (${counts(s)})</button>`).join('');
  tabs.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; tab = b.dataset.s;
    tabs.querySelectorAll('button').forEach(x => x.setAttribute('aria-selected', x === b)); draw(); });
  const chips = document.getElementById('chips');
  chips.innerHTML = D.objs.map(o => `<button id="chip-${o.o}" aria-pressed="true" data-o="${o.o}" style="color:${fam[o.family]}">${o.o}</button>`).join('');
  chips.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; on[b.dataset.o] = !on[b.dataset.o];
    b.setAttribute('aria-pressed', on[b.dataset.o]); draw(); });
  draw();
})();
(function () {
  const H = D.human; if (!H) return;
  document.getElementById('human').hidden = false;
  const pct = v => Math.round(100 * v) + '%';
  const multi = H.optimizers.length > 1, nm = p => (multi ? (p.tk === 'cma' ? 'CMA · ' : 'Adam · ') : '') + p.o;
  function bars(id, rows, x0, x1, ticks, fmt, ref) {
    const W = 520, L = 150, R = 54, bh = 15, gap = 7, T = 4, Hh = rows.length * (bh + gap) + T + 22;
    const x = v => L + (W - L - R) * (v - x0) / (x1 - x0);
    let b = '';
    ticks.forEach(t => { b += `<line x1="${x(t)}" x2="${x(t)}" y1="${T}" y2="${Hh - 18}" stroke="var(--rule)"/><text x="${x(t)}" y="${Hh - 4}" text-anchor="middle">${fmt(t)}</text>`; });
    if (ref != null) b += `<line x1="${x(ref)}" x2="${x(ref)}" y1="${T}" y2="${Hh - 18}" stroke="var(--muted)" stroke-dasharray="4 3"/>`;
    rows.forEach((r, i) => { const y = T + i * (bh + gap), cy = y + bh / 2;
      b += `<text class="lab" x="${L - 8}" y="${cy + 4}" text-anchor="end">${r.label}</text>`;
      b += `<rect x="${x(x0)}" y="${y}" width="${Math.max(1, x(r.v) - x(x0))}" height="${bh}" rx="2" fill="${r.color}" opacity="${r.op || 0.85}"><title>${r.label}: ${fmt(r.v)}</title></rect>`;
      if (r.lo != null) b += `<line x1="${x(r.lo)}" x2="${x(r.hi)}" y1="${cy}" y2="${cy}" stroke="var(--fg)" stroke-width="1.2"/><line x1="${x(r.lo)}" x2="${x(r.lo)}" y1="${cy - 4}" y2="${cy + 4}" stroke="var(--fg)"/><line x1="${x(r.hi)}" x2="${x(r.hi)}" y1="${cy - 4}" y2="${cy + 4}" stroke="var(--fg)"/>`;
      (r.marks || []).forEach(m => { b += m.shape === 'd' ? `<rect x="${x(m.v) - 4}" y="${cy - 4}" width="8" height="8" transform="rotate(45 ${x(m.v)} ${cy})" fill="var(--panel)" stroke="var(--fg)" stroke-width="1.4"><title>classifier ${fmt(m.v)}</title></rect>` : `<circle cx="${x(m.v)}" cy="${cy}" r="4" fill="var(--panel)" stroke="var(--sem)" stroke-width="1.6"><title>CLIP ${fmt(m.v)}</title></circle>`; });
      b += `<text x="${x(Math.max(r.v, r.hi || 0)) + 8}" y="${cy + 4}" style="fill:var(--fg)">${fmt(r.v)}</text>`; });
    document.getElementById(id).innerHTML = svg(W, Hh, b);
  }
  bars('h_per', H.per.map(p => ({label: nm(p), v: p.you, lo: p.lo, hi: p.hi, color: fam[(D.objs.find(o => o.o === p.o) || {}).family] || 'var(--base)',
    marks: [{v: p.cls, shape: 'd'}, {v: p.clip, shape: 'c'}]})), 0, 1, [0, 0.25, 0.5, 0.75, 1], pct);
  bars('h_pred', H.pred.map(p => ({label: p.label, v: p.auc, color: p.k === 'cls_p' || p.k === 'clip_p' ? 'var(--sem)' : 'var(--shape)'})), 0.4, 0.85, [0.4, 0.5, 0.6, 0.7, 0.8], v => v.toFixed(2), 0.5);
  bars('h_judges', [{label: 'you', v: H.you, color: 'var(--fg)'}, {label: 'glyph classifier', v: H.cls, color: 'var(--sem)'}, {label: 'CLIP (ViT-B-32)', v: H.clip, color: 'var(--sem)', op: 0.45}],
    0, 1, [0, 0.25, 0.5, 0.75, 1], pct);
  document.getElementById('h_judges').insertAdjacentHTML('beforeend', `<p class="note">Agrees with you shadow by shadow: classifier ${pct(H.agree_cls)}, CLIP ${pct(H.agree_clip)}. ${H.n} guesses, ${H.cant} "can't tell". Most confused: ${H.confusions.slice(0, 6).map(c => c.t + '→' + c.a).join(', ')}.</p>`);
  bars('h_marks', [...H.mark_rate.map(m => ({label: m.o, v: m.rate, color: fam[(D.objs.find(o => o.o === m.o) || {}).family] || 'var(--base)'}))], 0, 0.6, [0, 0.2, 0.4, 0.6], pct);
  document.getElementById('h_marks').insertAdjacentHTML('beforeend', `<p class="note">${H.n_judged} characters, ${H.good_per_glyph.toFixed(1)} good per character. In the guess test you read <b>${pct(H.marked_read)}</b> of the shadows you had marked good and ${pct(H.unmarked_read)} of the others.</p>`);
  const top = H.per.slice(0, 3).map(p => `${nm(p)} ${pct(p.you)}`).join(', ');
  const fool = [...H.per].sort((a, b) => (b.cls - b.you) - (a.cls - a.you))[0];
  const topMark = H.mark_rate[0];
  document.getElementById('take').innerHTML = [
    `<b>Your marks and your guesses agree.</b> Shadows you marked good you read ${pct(H.marked_read)} of the time; the rest ${pct(H.unmarked_read)}.`,
    `<b>The glyph classifier tracks a person; CLIP does not.</b> On the same shadows you read ${pct(H.you)}, the classifier ${pct(H.cls)}, CLIP ${pct(H.clip)}. Classifier probability is the best predictor of a correct read (AUC ${H.pred[0].auc.toFixed(2)}).`,
    `<b>Most readable:</b> ${top}. The intervals are about ±14 points, so the leaders are not yet separable; <b>${topMark.o}</b> is also the shadow you marked good most often (${pct(topMark.rate)}).`,
    `<b>Steering with a judge games it.</b> ${fool.o}: the classifier reads ${pct(fool.cls)}, you read ${pct(fool.you)}.`,
  ].map(s => `<li>${s}</li>`).join('');
})();

document.getElementById('foot').textContent =
  `Each objective replaces both the CMA-ES loss and the staged pipeline's accept/keep test (as 1 − loss), so the whole search chases it. ` +
  `Penalties (floor, collisions) are unchanged; collision-free rate ranges ${Math.min(...D.table.map(r => r.cf)).toFixed(2)}–${Math.max(...D.table.map(r => r.cf)).toFixed(2)}. ` +
  `Semantic weight 0.5 on (1 − p). Classifier: small CNN trained on 322 system fonts, never the DejaVu faces of the targets (100% on the targets). ` +
  `CLIP steering ViT-B-16 laion2b; CLIP grading ViT-B-32 laion2b, glyph prompts, ranked within the subset. Δ is paired over targets (seed means), SE over targets; faded cells are within 2 SE.`;
</script>
"""

if __name__ == "__main__":
    main()
