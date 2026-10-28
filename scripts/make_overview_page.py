"""The big-picture page: how do we make robot-arm shadows that people read? One map of the
pipeline (what each stage is, what we settled, what is open) and small diagrams of how every
claim is tested. Experiment pages (objective study, solver bench, picker) are linked, not
repeated. Numbers are read from the result files where they exist.

    python scripts/make_overview_page.py
"""

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
sys.path.insert(0, os.path.join(BENCH, "scripts"))
from run_base_optimizer import load_target  # noqa: E402

FS = os.path.join(BENCH, "results", "fastshadow")
TD = os.path.join(FS, "target_designer")
OUT = os.path.join(FS, "overview", "shadow_roadmap.html")
WALL, INK, TGT = np.array([234, 233, 228]), np.array([30, 31, 36]), np.array([150, 160, 178])


def load(*p):
    f = os.path.join(FS, *p)
    return json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None


def png(path, color):
    m = load_target(os.path.join(BENCH, path), 128) > 0.5
    a = np.tile(WALL, m.shape + (1,))
    a[m] = color
    b = io.BytesIO()
    Image.fromarray(a.astype(np.uint8)).save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def designer():
    """Target-designer result on fresh seeds 3-5 (report.json, incl. the audit's secondary checks), with the
    characters the held-out classifier reads most better (first 3) and most worse (last 3) with the chosen font."""
    rep = load("target_designer", "report.json")
    if not rep or "secondary" not in rep["summary"]:
        return None
    rows = [json.loads(l) for l in open(os.path.join(TD, "solves.jsonl"), encoding="utf-8")]
    picks = json.load(open(os.path.join(TD, "picks.json"), encoding="utf-8"))
    cands = {c["id"]: c for c in json.load(open(os.path.join(TD, "candidates.json"), encoding="utf-8"))}
    by = defaultdict(list)
    for r in rows:
        by[r["id"]].append(r)

    def best(cid):
        rr = [r for r in by[cid] if r["seed"] in (3, 4, 5)]
        ok = [r for r in rr if r["collision_free"] and r["lowest_mm"] >= -5] or rr
        return max(ok, key=lambda r: r["iou"])

    per = sorted(rep["per_char"], key=lambda x: x["cls_p_chosen"] - x["cls_p_bench"])
    ex = []
    for x in per[::-1][:3] + per[:3]:
        ch = x["ch"]
        bid = next(r["id"] for r in rows if r["ch"] == ch and "BENCH:" in r["id"])
        b, c = best(bid), best(picks[ch]["chosen"])
        ex.append({"ch": ch, "font": cands[picks[ch]["chosen"]]["label"].replace(" Regular", ""),
                   "bt": png(b["target"], TGT), "bs": png(b["mask"], INK), "bi": b["iou"], "bp": x["cls_p_bench"],
                   "ct": png(c["target"], TGT), "cs": png(c["mask"], INK), "ci": c["iou"], "cp": x["cls_p_chosen"]})
    s_, x_ = rep["summary"], rep["summary"]["secondary"]
    top = x_["top_font"]
    return {"examples": ex, "n_candidates": sum(1 for c in cands.values() if not c.get("gate_fail")),
            "n_fonts": len({c["font"] for c in cands.values()}),
            "mean": s_["mean_iou"], "share": s_["share_ge_075"], "vs12": s_["vs_bench12_mean"], "cls": s_["cls_top1"],
            "blob": x_["blob_floor_iou"], "above": x_["shadow_above_floor"], "cls_fresh": x_["cls_top1_fresh"],
            "small_holes": x_["small_counter_chars"], "n_holed": len(x_["holed_chars"]),
            "top_font": [next((c["label"] for c in cands.values() if c["font"] == top[0]), top[0]).replace(" Regular", ""), top[1]]}


def main():
    ivy = load("ivy_reader", "report.json")
    D = {"designer": designer(), "judge": ivy["cv"]["linear"] if ivy else None,
         "judge_base": ivy["baselines"]["iou"] if ivy else None}
    html = TEMPLATE.replace("/*DATA*/", json.dumps(D))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(html)
    print("wrote", OUT, f"{len(html) / 1e3:.0f} KB")


TEMPLATE = r"""<title>Umbra Shadow Roadmap</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@500&display=swap">
<style>
/* Layout: one light path read left to right (text -> target -> arms -> wall -> reader), then a wall of small test diagrams. */
:root{
  --bg:#eef0f2; --wall:#e6e5df; --panel:#fbfbfc; --ink:#1e1f24; --muted:#5d6370; --rule:#d3d7de;
  --lamp:#d98a1c; --lamp-soft:#f3dcb4;
  --done:#2c7a5b; --no:#b24a3f; --run:#2f63a8; --open:#7a7f8c;
  --f-display:"Bricolage Grotesque","Segoe UI",system-ui,sans-serif;
  --f-body:"Source Sans 3","Segoe UI",system-ui,sans-serif;
  --f-num:"JetBrains Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#141619; --wall:#1d2024; --panel:#1a1c20; --ink:#eceef2; --muted:#a2a8b4; --rule:#2e323a;
  --lamp:#f0a53c; --lamp-soft:#4a3a20; --done:#5cc196; --no:#ec7c6f; --run:#7fa9e6; --open:#9aa0ad; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#141619; --wall:#1d2024; --panel:#1a1c20; --ink:#eceef2; --muted:#a2a8b4; --rule:#2e323a;
  --lamp:#f0a53c; --lamp-soft:#4a3a20; --done:#5cc196; --no:#ec7c6f; --run:#7fa9e6; --open:#9aa0ad; color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:16px/1.5 var(--f-body)}
.wrap{max-width:1120px;margin:0 auto;padding-block:32px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:44px}
h1{font:800 clamp(30px,5vw,48px)/1.05 var(--f-display);margin:0;text-wrap:balance;letter-spacing:-.01em}
h2{font:800 clamp(22px,3vw,28px)/1.15 var(--f-display);margin:0;text-wrap:balance}
h3{font:600 17px/1.25 var(--f-display);margin:0}
p{margin:0;max-width:64ch}
.eyebrow{font:500 12px/1 var(--f-num);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.lede{font-size:18px;color:var(--muted)}
section{display:grid;gap:18px}
.wrap>*,.hero>*,section>*{min-width:0}
svg text{font-family:var(--f-body)}
.hero{display:grid;gap:16px}
.path{background:var(--wall);border-radius:14px;padding:16px;overflow-x:auto}
.path svg{display:block;min-width:720px;width:100%}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:14px;color:var(--muted)}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:0}
.stages{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
@media (max-width:900px){.stages{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:560px){.stages{grid-template-columns:1fr}}
.stage{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:16px;display:grid;gap:10px;align-content:start;min-width:0}
.stage .top{display:flex;justify-content:space-between;gap:10px;align-items:baseline}
.pill{font:500 11px/1 var(--f-num);text-transform:uppercase;letter-spacing:.05em;padding:4px 7px;border-radius:99px;color:var(--panel);white-space:nowrap}
.pill.done{background:var(--done)} .pill.no{background:var(--no)} .pill.run{background:var(--run)} .pill.open{background:var(--open)}
.q{color:var(--muted);font-size:15px}
.facts{margin:0;padding:0;list-style:none;display:grid;gap:6px}
.facts li{display:grid;grid-template-columns:auto minmax(0,1fr);gap:8px;font-size:15px}
.facts b{font:500 14px var(--f-num);font-variant-numeric:tabular-nums;white-space:nowrap}
.tests{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:14px}
.test{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:16px;display:grid;gap:10px;align-content:start;min-width:0}
.test svg{width:100%;height:auto;display:block}
.cap{font-size:14px;color:var(--muted)}
.ex{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,160px),1fr));gap:12px}
.excard{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:12px;display:grid;gap:8px;min-width:0}
.excard .row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px}
.excard img{width:100%;aspect-ratio:1;image-rendering:pixelated;border-radius:4px;display:block}
.excard .lab{display:flex;justify-content:space-between;gap:8px;font:500 12px var(--f-num);color:var(--muted)}
.excard .lab b{font:800 20px/1 var(--f-display)}
.excard .lab .fn{text-align:right;min-width:0}
.excard .head{min-height:4em;align-items:start}
.excard .lab b{color:var(--ink)}
.next{display:grid;gap:10px;margin:0;padding:0;list-style:none}
.next li{display:grid;grid-template-columns:28px minmax(0,1fr);gap:10px;align-items:start;background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:12px 14px}
.next .n{font:800 18px/1.2 var(--f-display);color:var(--lamp)}
.links{display:flex;flex-wrap:wrap;gap:10px}
.links a{color:var(--ink);text-decoration:none;border:1px solid var(--rule);background:var(--panel);border-radius:8px;padding:8px 12px;font-size:15px}
.links a:hover,.links a:focus-visible{border-color:var(--lamp);outline:none}
.links a span{color:var(--muted)}
.foot{font-size:13px;color:var(--muted)}
</style>
<div class="wrap">
  <header class="hero">
    <span class="eyebrow">Umbra · three arms, one lamp, one wall</span>
    <h1>How do we make shadows people can read?</h1>
    <p class="lede">Every shadow goes through the same path. Each stage below has one question. Most are settled; the open ones are where the next gains are.</p>
    <div class="path" role="img" aria-label="The path from a word to a readable shadow: text, target shape, search for joint angles with a score, pick one, check it, run it on the real arms.">
      <svg viewBox="0 0 1060 210" id="pathsvg"></svg>
    </div>
    <div class="legend"><span><i class="dot" style="background:var(--done)"></i>settled</span><span><i class="dot" style="background:var(--open)"></i>open</span></div>
  </header>

  <section>
    <div><span class="eyebrow">The stages</span><h2>What we know at each step</h2></div>
    <div class="stages" id="stages"></div>
  </section>

  <section>
    <div><span class="eyebrow">Latest experiment</span><h2>Let the font be free</h2>
      <p class="lede" id="td_lede"></p></div>
    <div class="ex" id="examples"></div>
    <p class="cap" id="td_cap"></p>
  </section>

  <section>
    <div><span class="eyebrow">How we test</span><h2>Six checks behind every claim</h2></div>
    <div class="tests">
      <div class="test"><h3>Overlap, and why 0.75</h3><svg viewBox="0 0 340 190" id="t_iou"></svg>
        <p class="cap">Overlap (IoU) is the shared area divided by the combined area. Ivy's 775 blind guesses show where it starts to matter: below 0.75 about half are read; above it, nearly all.</p></div>
      <div class="test"><h3>Three human tests</h3><svg viewBox="0 0 340 190" id="t_human"></svg>
        <p class="cap">Marking (target shown), guessing (shadow alone), and blind side-by-side. Only the guessing and blind tests leave the target out, so they decide.</p></div>
      <div class="test"><h3>Never grade on what it learned</h3><svg viewBox="0 0 340 190" id="t_folds"></svg>
        <p class="cap">Anything trained on Ivy's labels (the Ivy judge, the fine-tuned SDS) is scored only on characters held out of its training, five rounds so each is held out once.</p></div>
      <div class="test"><h3>Re-test the winner</h3><svg viewBox="0 0 340 190" id="t_curse"></svg>
        <p class="cap">One solve is a noisy draw. Picking the best of many inflates it, so every winner is solved again with fresh random starts, and compared with simply running the old way more times.</p></div>
      <div class="test"><h3>Compare with a blob</h3><svg viewBox="0 0 340 190" id="t_blob"></svg>
        <p class="cap">Overlap only means something above what a featureless blob would score. A heavy letter is almost its own outline, so a blob already reaches 0.79 on the chosen fonts, against 0.67 on today's.</p></div>
      <div class="test"><h3>Check for gaming</h3><svg viewBox="0 0 340 190" id="t_game"></svg>
        <p class="cap">A score that the search optimizes is graded by something it never saw: a separate reader and, in the end, people. Optimizing the glyph classifier made it right 93% of the time while Ivy read 43%.</p></div>
    </div>
  </section>

  <section>
    <div><span class="eyebrow">Still needed</span><h2>What would move us forward</h2></div>
    <ol class="next" id="next"></ol>
  </section>

  <section>
    <div><span class="eyebrow">Experiment pages</span><h2>The detail lives here</h2></div>
    <div class="links">
      <a href="https://claude.ai/artifact/ChM2otn7JGKKN1sy2yJsY8">Glyph Objective Study <span>· objectives, Ivy judge, overnight research</span></a>
      <a href="https://claude.ai/artifact/HU5HFUjpvTf9BTnkTveK7i">Solver bench <span>· GPU renderer, CMA-ES vs Adam, arm layouts</span></a>
      <a href="https://claude.ai/artifact/DTjJuNUhPWfJN5hRrTWmSX">Shadow picker <span>· where Ivy's labels come from</span></a>
      <a href="https://claude.ai/artifact/YRfwV5BKayNk9H4aS9e3wu">Shadow reading test <span>· blind test of the font experiment</span></a>
    </div>
    <p class="foot">Letters and digits only so far: 62 characters, three SO-101 arms, shadows rendered at 128 px and checked in MuJoCo.</p>
  </section>
</div>
<script>
const D = /*DATA*/;
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent, text) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; parent.appendChild(e); return e; }
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const f2 = v => v == null ? '–' : v.toFixed(2), f3 = v => v == null ? '–' : v.toFixed(3), pc = v => Math.round(100 * v) + '%';
const T = D.designer;

const STAGES = [
  {k: 'target', name: 'Target', q: 'What shape should the shadow aim for?', st: 'open',
   facts: T ? [[f3(T.mean.bench) + ' → ' + f3(T.mean.chosen), 'overlap when each character picks its font'],
               ['but', 'heavy fonts are just easier to reach; a blob scores ' + f2(T.blob.chosen)],
               ['next', 'Ivy\u2019s blind reading test decides']] : [['–', 'not started']]},
  {k: 'search', name: 'Search', q: 'How do we find joint angles?', st: 'done',
   facts: [['CMA-ES', 'on the GPU renderer, 1.2 s per solve (MuJoCo: 3.3 s)'], ['0.705 → 0.754', 'overlap from 1 try to the best of 12'], ['Adam', 'behind at equal time']]},
  {k: 'score', name: 'Score', q: 'What does the search maximize?', st: 'done',
   facts: [['dt2', 'overlap plus distance penalties, the safe default of 16 tried'], ['gamed', 'classifier, CLIP and judge scores when optimized'], ['SDS', 'cannot read letters, even fine-tuned (0.49)']]},
  {k: 'pick', name: 'Pick', q: 'Which of the tries do we keep?', st: 'done',
   facts: [[D.judge ? f2(D.judge.mark_auc) : '0.81', 'Ivy judge agrees with Ivy’s marks (plain overlap ' + (D.judge_base ? f2(D.judge_base.mark_auc) : '0.76') + ')'], ['rule', 'physically possible first, then overlap; the judge only reports'], ['limit', 'glyphs only: it reads a letter classifier']]},
  {k: 'check', name: 'Check', q: 'Is it readable and safe?', st: 'done',
   facts: [['≥ 0.75', 'overlap → Ivy reads 84–100% (DejaVu targets)'], ['10.7% → 0%', 'solves with the gripper in the table, after the floor fix'], ['1°', 'joint error costs 0.013 overlap; 2° costs 0.035']]},
  {k: 'rig', name: 'Real arms & motion', q: 'Does it hold on hardware and in animations?', st: 'open',
   facts: [['rail', 'arms only in a column, ≥ 0.18 m apart (layout adds +0.006)'], ['animation', 'plan all frames together: designed, not tested'], ['photos', 'real-rig shadows vs simulation: not done']]},
];
const STATUS = {done: 'settled', run: 'running', no: 'ruled out', open: 'open'};

// the light path
(function () {
  const s = document.getElementById('pathsvg');
  const ink = css('--ink'), muted = css('--muted'), lamp = css('--lamp'), rule = css('--rule'), panel = css('--panel');
  const col = {done: css('--done'), run: css('--run'), no: css('--no'), open: css('--open')};
  el('circle', {cx: 34, cy: 105, r: 18, fill: lamp}, s);
  el('text', {x: 34, y: 150, 'text-anchor': 'middle', 'font-size': 13, fill: muted}, s, 'text "A"');
  const xs = [110, 255, 400, 545, 690, 835], w = 128;
  STAGES.forEach((st, i) => {
    const x = xs[i];
    el('rect', {x, y: 62, width: w, height: 86, rx: 10, fill: panel, stroke: col[st.st], 'stroke-width': 2}, s);
    el('circle', {cx: x + 16, cy: 80, r: 6, fill: col[st.st]}, s);
    el('text', {x: x + 28, y: 85, 'font-size': 16, 'font-weight': 700, fill: ink}, s, st.name.split(' ')[0]);
    const words = st.q.split(' '); let line = '', ly = 108;
    words.forEach(wd => { if ((line + wd).length > 18) { el('text', {x: x + 12, y: ly, 'font-size': 12, fill: muted}, s, line.trim()); line = ''; ly += 15; } line += wd + ' '; });
    el('text', {x: x + 12, y: ly, 'font-size': 12, fill: muted}, s, line.trim());
    if (i < STAGES.length - 1) el('path', {d: `M${x + w + 2} 105 L${xs[i + 1] - 4} 105`, stroke: rule, 'stroke-width': 2, 'marker-end': 'url(#arr)'}, s);
  });
  const defs = el('defs', {}, s); const m = el('marker', {id: 'arr', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto'}, defs);
  el('path', {d: 'M0 0 L10 5 L0 10 z', fill: rule}, m);
  el('path', {d: 'M54 105 L106 105', stroke: lamp, 'stroke-width': 2, 'stroke-dasharray': '4 4'}, s);
  el('rect', {x: 980, y: 40, width: 60, height: 130, rx: 4, fill: css('--wall'), stroke: rule}, s);
  el('text', {x: 1010, y: 122, 'text-anchor': 'middle', 'font-size': 54, 'font-weight': 800, fill: ink}, s, 'A');
  el('text', {x: 1010, y: 190, 'text-anchor': 'middle', 'font-size': 13, fill: muted}, s, 'a person reads it');
  el('path', {d: `M${835 + w + 2} 105 L976 105`, stroke: rule, 'stroke-width': 2, 'marker-end': 'url(#arr)'}, s);
  [[0, 0, 'what to aim for'], [1, 2, 'how to get there'], [3, 4, 'how to know it worked'], [5, 5, 'does it survive reality']].forEach(([i, j, t]) => {
    const a = xs[i] + 4, z = xs[j] + w - 4;
    el('path', {d: `M${a} 50 L${a} 44 L${z} 44 L${z} 50`, fill: 'none', stroke: muted, 'stroke-width': 1}, s);
    el('text', {x: (a + z) / 2, y: 36, 'text-anchor': 'middle', 'font-size': 13, fill: muted}, s, t); });
})();

document.getElementById('stages').innerHTML = STAGES.map(st => `<article class="stage"><div class="top"><h3>${st.name}</h3><span class="pill ${st.st}">${STATUS[st.st]}</span></div>
  <p class="q">${st.q}</p><ul class="facts">${st.facts.map(([a, b]) => `<li><b>${a}</b><span>${b}</span></li>`).join('')}</ul></article>`).join('');

// font-free examples
if (T) {
  document.getElementById('td_lede').textContent = `Each character picked, from ${T.n_candidates.toLocaleString()} legible targets in ${T.n_fonts} heavy fonts, the one the arms match best. Overlap rose from ${f3(T.mean.bench)} to ${f3(T.mean.chosen)}, but mostly because heavy, solid letters are easy to reach: ${T.top_font[0]} won ${T.top_font[1]} of 62. The glyph classifier reads the new shadows no better (${pc(T.cls_fresh.chosen)} vs ${pc(T.cls_fresh.bench)}).`;
  document.getElementById('examples').innerHTML = T.examples.map(e => `<div class="excard"><div class="lab head"><b>${e.ch}</b><span class="fn">${e.font}</span></div>
    <div class="row"><img src="${e.bt}" alt="benchmark target ${e.ch}"><img src="${e.bs}" alt="shadow for the benchmark target ${e.ch}"></div>
    <div class="lab"><span>today</span><span>${f2(e.bi)} · ${pc(e.bp)}</span></div>
    <div class="row"><img src="${e.ct}" alt="chosen target ${e.ch}"><img src="${e.cs}" alt="shadow for the chosen target ${e.ch}"></div>
    <div class="lab"><span>chosen</span><span>${f2(e.ci)} · ${pc(e.cp)}</span></div></div>`).join('');
  document.getElementById('td_cap').textContent = `Grey: target. Dark: best fresh shadow. Numbers: overlap with its own target · how sure the glyph classifier is. First three: the classifier reads the new shadow better; last three: worse. ${T.small_holes.length} of the ${T.n_holed} characters with holes got holes too small for a shadow to keep.`;
}

// test diagrams
(function iou() {
  const s = document.getElementById('t_iou'), ink = css('--ink'), muted = css('--muted'), lamp = css('--lamp'), rule = css('--rule');
  el('rect', {x: 14, y: 24, width: 70, height: 70, fill: css('--open'), opacity: .35}, s);
  el('rect', {x: 44, y: 44, width: 70, height: 70, fill: ink, opacity: .55}, s);
  el('text', {x: 14, y: 16, 'font-size': 12, fill: muted}, s, 'target');
  el('text', {x: 72, y: 132, 'font-size': 12, fill: muted}, s, 'shadow');
  el('text', {x: 14, y: 160, 'font-size': 13, fill: ink}, s, 'IoU = shared / combined');
  const bins = [['<0.70', 0.23, 0.43], ['0.70–0.75', 0.56, 0.56], ['0.75–0.80', 0.84, 0.84], ['0.80+', 0.90, 1.0]];
  const x0 = 150, w = 40, H = 120, y0 = 140;
  bins.forEach(([l, lo, hi], i) => { const x = x0 + i * (w + 6), h = H * hi, c = i >= 2 ? lamp : css('--open');
    el('rect', {x, y: y0 - h, width: w, height: h, rx: 3, fill: c, opacity: lo < hi ? .45 : 1}, s);
    if (lo < hi) el('rect', {x, y: y0 - H * lo, width: w, height: H * lo, rx: 3, fill: c}, s);
    el('text', {x: x + w / 2, y: y0 - h - 4, 'text-anchor': 'middle', 'font-size': 11, fill: ink}, s, lo < hi ? Math.round(100 * lo) + '–' + pc(hi) : pc(hi));
    el('text', {x: x + w / 2, y: y0 + 14, 'text-anchor': 'middle', 'font-size': 10, fill: muted}, s, l); });
  el('line', {x1: x0 - 4, x2: x0 + 4 * (w + 6), y1: y0, y2: y0, stroke: rule}, s);
  el('text', {x: x0, y: 178, 'font-size': 12, fill: muted}, s, 'Ivy reads it, by overlap');
})();
(function human() {
  const s = document.getElementById('t_human'), ink = css('--ink'), muted = css('--muted'), lamp = css('--lamp'), wall = css('--wall'), rule = css('--rule');
  const box = (x, y, w, h) => el('rect', {x, y, width: w, height: h, rx: 6, fill: wall, stroke: rule}, s);
  const items = [['mark', 'target shown', '1,612'], ['guess', 'shadow alone', '775'], ['blind A/B', 'two shadows', '139']];
  items.forEach(([n, d, c], i) => { const x = 10 + i * 112;
    box(x, 14, 100, 100);
    if (i === 0) { el('text', {x: x + 30, y: 82, 'text-anchor': 'middle', 'font-size': 44, 'font-weight': 800, fill: css('--open')}, s, 'A'); el('rect', {x: x + 56, y: 40, width: 30, height: 50, rx: 4, fill: ink, opacity: .8}, s); el('text', {x: x + 70, y: 106, 'text-anchor': 'middle', 'font-size': 14, fill: css('--done')}, s, '✓ good'); }
    if (i === 1) { el('rect', {x: x + 35, y: 30, width: 30, height: 56, rx: 4, fill: ink, opacity: .8}, s); el('text', {x: x + 50, y: 106, 'text-anchor': 'middle', 'font-size': 14, fill: lamp}, s, '“A?”'); }
    if (i === 2) { el('rect', {x: x + 12, y: 34, width: 30, height: 50, rx: 4, fill: ink, opacity: .8}, s); el('rect', {x: x + 58, y: 34, width: 30, height: 50, rx: 4, fill: ink, opacity: .5}, s); el('text', {x: x + 50, y: 106, 'text-anchor': 'middle', 'font-size': 14, fill: lamp}, s, '← or →'); }
    el('text', {x: x + 50, y: 136, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 700, fill: ink}, s, n);
    el('text', {x: x + 50, y: 154, 'text-anchor': 'middle', 'font-size': 12, fill: muted}, s, d);
    el('text', {x: x + 50, y: 176, 'text-anchor': 'middle', 'font-size': 12, fill: muted, 'font-family': 'monospace'}, s, c + ' answers'); });
})();
(function folds() {
  const s = document.getElementById('t_folds'), ink = css('--ink'), muted = css('--muted'), lamp = css('--lamp'), rule = css('--rule');
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
  for (let r = 0; r < 5; r++) {
    el('text', {x: 4, y: 36 + r * 28, 'font-size': 11, fill: muted}, s, 'round ' + (r + 1));
    chars.forEach((c, i) => { const held = i % 5 === r, x = 52 + i * 10.8;
      el('rect', {x, y: 24 + r * 28, width: 9, height: 16, rx: 2, fill: held ? lamp : css('--open'), opacity: held ? 1 : .35}, s); });
  }
  el('rect', {x: 52, y: 170, width: 9, height: 10, rx: 2, fill: css('--open'), opacity: .35}, s);
  el('text', {x: 66, y: 179, 'font-size': 11, fill: muted}, s, 'trained on');
  el('rect', {x: 150, y: 170, width: 9, height: 10, rx: 2, fill: lamp}, s);
  el('text', {x: 164, y: 179, 'font-size': 11, fill: muted}, s, 'graded on (never seen)');
})();
(function curse() {
  const s = document.getElementById('t_curse'), ink = css('--ink'), muted = css('--muted'), lamp = css('--lamp'), rule = css('--rule'), done = css('--done');
  const y = v => 160 - (v - 0.6) * 500;
  el('line', {x1: 30, x2: 330, y1: y(0.6), y2: y(0.6), stroke: rule}, s);
  [0.65, 0.7, 0.75, 0.8].forEach(v => { el('line', {x1: 30, x2: 330, y1: y(v), y2: y(v), stroke: rule, 'stroke-dasharray': '2 4'}, s); el('text', {x: 2, y: y(v) + 4, 'font-size': 10, fill: muted}, s, v.toFixed(2)); });
  const pts = [0.66, 0.69, 0.71, 0.68, 0.73, 0.70, 0.67, 0.72, 0.795, 0.69, 0.71, 0.70];
  pts.forEach((v, i) => el('circle', {cx: 44 + i * 13, cy: y(v), r: 4, fill: i === 8 ? lamp : css('--open')}, s));
  el('text', {x: 44, y: 20, 'font-size': 12, fill: muted}, s, '12 one-off tries: pick the best');
  [0.725, 0.735, 0.715].forEach((v, i) => el('circle', {cx: 250 + i * 18, cy: y(v), r: 5, fill: done}, s));
  el('path', {d: `M${44 + 8 * 13 + 6} ${y(0.795)} C 200 ${y(0.79)}, 220 ${y(0.74)}, 244 ${y(0.73)}`, fill: 'none', stroke: lamp, 'stroke-width': 1.5, 'stroke-dasharray': '4 3'}, s);
  el('text', {x: 236, y: 20, 'font-size': 12, fill: muted}, s, 'fresh tries: real level');
  el('text', {x: 44, y: 182, 'font-size': 11, fill: muted}, s, 'illustration; the real numbers are on the experiment pages');
})();
(function game() {
  const s = document.getElementById('t_game'), ink = css('--ink'), muted = css('--muted'), lamp = css('--lamp'), rule = css('--rule'), no = css('--no'), done = css('--done');
  const node = (x, y, t, c) => { el('rect', {x, y, width: 92, height: 40, rx: 8, fill: css('--wall'), stroke: c, 'stroke-width': 1.5}, s); el('text', {x: x + 46, y: y + 25, 'text-anchor': 'middle', 'font-size': 13, fill: ink}, s, t); };
  node(10, 20, 'search', rule); node(124, 20, 'score A', lamp); node(238, 20, 'shadow', rule);
  el('path', {d: 'M102 40 L122 40', stroke: rule, 'stroke-width': 2}, s); el('path', {d: 'M216 40 L236 40', stroke: rule, 'stroke-width': 2}, s);
  node(160, 104, 'reader B', done); node(40, 104, 'people', done);
  el('path', {d: 'M284 62 L220 102', stroke: rule, 'stroke-width': 2}, s); el('path', {d: 'M284 62 C 200 80, 120 86, 90 102', stroke: rule, 'stroke-width': 2, fill: 'none'}, s);
  el('text', {x: 10, y: 172, 'font-size': 13, fill: ink}, s, 'classifier steering:');
  el('text', {x: 140, y: 172, 'font-size': 13, fill: no}, s, 'classifier 93%');
  el('text', {x: 248, y: 172, 'font-size': 13, fill: done}, s, 'Ivy 43%');
})();

(function blob() {
  const s = document.getElementById('t_blob'), ink = css('--ink'), muted = css('--muted'), no = css('--no'), open = css('--open');
  const bl = T ? T.blob : {bench: 0.667, chosen: 0.789};
  const glyph = (x, w, label, v, c) => {
    el('rect', {x: x - 6, y: 18, width: 100, height: 112, rx: 50, fill: c, opacity: .22}, s);
    el('text', {x: x + 44, y: 112, 'text-anchor': 'middle', 'font-size': 112, 'font-weight': w, fill: open, 'font-family': 'Arial, sans-serif'}, s, 'o');
    el('text', {x: x + 44, y: 150, 'text-anchor': 'middle', 'font-size': 13, fill: ink}, s, label);
    el('text', {x: x + 44, y: 170, 'text-anchor': 'middle', 'font-size': 13, fill: c, 'font-family': 'monospace'}, s, 'blob ' + v.toFixed(2)); };
  glyph(40, 400, "today's target", bl.bench, open);
  glyph(206, 900, 'chosen font', bl.chosen, no);
  el('text', {x: 170, y: 186, 'text-anchor': 'middle', 'font-size': 11, fill: muted}, s, 'shaded: the blob, its overlap with the letter');
})();

document.getElementById('next').innerHTML = [
  ['A blind reading test of the new targets', 'Ready (link below). Decides whether choosing fonts makes shadows easier to read, not only to match. Ivy plus one or two people new to the project, so it counts beyond one pair of eyes.'],
  ['Choose fonts by reading, not reach', 'Holes at least 14 px across after fitting (changes 11 picks), stroke weight near DejaVu\'s, or a score that does not depend on the target, so the search stops rewarding blobs.'],
  ['Any shape, not only letters', 'The same path with targets drawn from text by an image model, and a reader for animals and objects. Needs a few hundred labelled shadows.'],
  ['Animation as one plan', 'Keep a few good poses per frame and choose the smoothest path through all frames, with the arms\' real speed limits.'],
  ['Photos of the real wall', 'Check that a shadow that reads in simulation reads on the wall, with real light and calibration.'],
].map(([h, d], i) => `<li><span class="n">${i + 1}</span><div><h3>${h}</h3><p class="cap">${d}</p></div></li>`).join('');
</script>
"""

if __name__ == "__main__":
    main()
