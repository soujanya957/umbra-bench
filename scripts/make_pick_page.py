"""Blind human picking page for the glyph objective study, one tab per optimizer.

For every character, each objective's seed-0 shadow is shown in a fixed shuffled
order with names hidden; the viewer marks every shadow that looks like the target
(any number), or "none". Picks live in the artifact's db at `picks/<viewer id>`
as {cma: {glyph: {good: [...]}} , adam: {...}}; a v1 document ({picks: {glyph:
{best, second}}}) is migrated into `cma` on first load and kept as it was. The page
computes each objective's good-rate and how well each machine metric agrees with
the marks (top-1 precision and AUC).

    python scripts/make_pick_page.py
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
sys.path.insert(0, os.path.join(BENCH, "scripts"))
from make_objective_report import OBJ  # noqa: E402
from run_base_optimizer import load_target  # noqa: E402

STUDIES = [("cma", "CMA-ES", "objective_study"), ("adam", "Adam", "objective_study_adam")]
ORDER = {"letters_upper": 0, "letters_lower": 1, "digits": 2}
METRICS = [  # key, label, higher is better (agreement chart)
    ("iou", "IoU", True), ("tc_iou", "tc_iou", True), ("boundary_iou", "boundary IoU", True), ("nsd", "NSD", True),
    ("chamfer", "chamfer", False), ("cldice", "clDice", True), ("betti_error", "Betti error", False),
    ("holes_filled", "holes filled", False), ("cls_p", "glyph classifier p", True), ("clip_p", "CLIP p (ViT-B-32)", True),
]
SUMMARY = [  # key, label, higher is better (summary table)
    ("iou", "IoU", True), ("tc_iou", "tc_iou", True), ("boundary_iou", "boundary IoU", True), ("cldice", "clDice", True),
    ("holes_filled", "holes filled", False), ("betti_error", "Betti err", False),
    ("cls_top1", "classifier top-1", True), ("clip_top1", "CLIP top-1", True), ("seconds", "s / solve", False),
]
INK, WALL = np.array([38, 36, 44]), np.array([247, 245, 240])


def uri(mask):
    img = np.tile(WALL, mask.shape + (1,))
    img[mask] = INK
    b = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def study(dirname):
    path = os.path.join(FS, dirname, "graded.jsonl")
    if not os.path.exists(path):
        return None
    G = [json.loads(l) for l in open(path, encoding="utf-8")]
    s0 = defaultdict(dict)
    for g in G:
        if g["seed"] == 0:
            s0[g["id"]][g["objective"]] = g
    objs = [o for o, *_ in OBJ if any(o in v for v in s0.values())]
    ids = sorted((i for i in s0 if all(o in s0[i] for o in objs)), key=lambda i: (ORDER.get(i.split("/")[0], 9), i))
    root = os.path.join(BENCH, "optimized", "big-budget-grounded")
    glyphs = []
    for i in ids:
        sub, stem = i.split("/")
        T = load_target(os.path.join(root, sub, stem, f"{stem}_shown.png"), 128) > 0.5
        cands = []
        for o in objs:
            r = s0[i][o]
            S = load_target(os.path.join(FS, dirname, r["mask"]), 128) > 0.5
            m = {k: r.get(k) for k, *_ in METRICS}
            m.update(cls_top1=r.get("cls_top1"), clip_top1=r.get("clip_top1"))
            cands.append({"o": o, "img": uri(S), "m": m})
        glyphs.append({"key": i.replace("/", "__"), "sub": sub, "label": stem.split("_")[0], "target": uri(T), "cands": cands})
    summary = []
    for o in objs:
        row = {"o": o}
        for k, *_ in SUMMARY:
            v = [s0[i][o][k] for i in ids if s0[i][o].get(k) is not None]
            row[k] = float(np.mean(v)) if v else None
        summary.append(row)
    return {"objs": objs, "glyphs": glyphs, "summary": summary}


def main():
    tabs = []
    for key, label, d in STUDIES:
        s = study(d)
        if s and s["glyphs"]:
            tabs.append({"key": key, "label": label, **s})
            print(f"{label}: {len(s['glyphs'])} glyphs × {len(s['objs'])} objectives")
    D = {"objs": [{"o": o, "label": l, "family": f} for o, l, f, _ in OBJ],
         "metrics": [{"k": k, "label": l, "hb": hb} for k, l, hb in METRICS],
         "summary_cols": [{"k": k, "label": l, "hb": hb} for k, l, hb in SUMMARY], "tabs": tabs}
    out = os.path.join(FS, "report", "pick_page.html")
    open(out, "w", encoding="utf-8").write(TEMPLATE.replace("/*DATA*/", json.dumps(D)))
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.2f} MB")


TEMPLATE = r"""<title>Shadow Glyph Picker</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
/* Layout: a judging desk. Optimizer tabs on top, target left, the blind line-up right, tallies and the summary below. */
:root{
  --bg:#f2f3f5; --panel:#ffffff; --fg:#1d1e24; --muted:#5f6370; --rule:#d9dbe1;
  --good:#2f8a57; --none:#c2473a; --shape:#3d5fd6; --sem:#b04a8f; --base:#8a8e9a; --accent:#3d5fd6;
  --ink:#26242c; --wall:#f7f5f0;
  --f-display:"IBM Plex Sans Condensed","Arial Narrow",sans-serif;
  --f-body:"IBM Plex Sans",system-ui,sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#15161b; --panel:#1d1f26; --fg:#e8e9ee; --muted:#9a9eab; --rule:#2e313b;
  --good:#5cc489; --none:#ef7a6c; --shape:#7d97ff; --sem:#e07fc2; --base:#9ea2ae; --accent:#7d97ff; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#15161b; --panel:#1d1f26; --fg:#e8e9ee; --muted:#9a9eab; --rule:#2e313b;
  --good:#5cc489; --none:#ef7a6c; --shape:#7d97ff; --sem:#e07fc2; --base:#9ea2ae; --accent:#7d97ff; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.5 var(--f-body)}
.wrap{max-width:1180px;margin:0 auto;padding-block:24px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:24px}
h1{font:600 clamp(26px,4vw,36px)/1.1 var(--f-display);margin:0}
h2{font:600 20px/1.2 var(--f-display);margin:0}
.sub{color:var(--muted);margin:4px 0 0;max-width:72ch}
.eyebrow{font:500 11px/1 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between}
.prog{font:500 13px var(--f-mono);color:var(--muted)} .prog b{color:var(--fg)}
.status{font-size:13px;color:var(--muted)}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--rule)}
.tabs button{font:600 15px var(--f-display);border:0;border-bottom:3px solid transparent;background:none;color:var(--muted);padding:8px 16px;cursor:pointer;border-radius:0}
.tabs button[aria-selected="true"]{color:var(--fg);border-bottom-color:var(--accent)}
.tabs button .n{font:12px var(--f-mono);color:var(--muted);margin-left:6px}
.desk{display:grid;grid-template-columns:minmax(0,250px) minmax(0,1fr);gap:20px;align-items:start}
@media (max-width:820px){.desk{grid-template-columns:1fr}}
.tgt{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:14px;display:grid;gap:10px;justify-items:center;position:sticky;top:calc(env(safe-area-inset-top,0px) + 8px)}
.tgt .ch{font:600 44px/1 var(--f-display)}
.tgt .kind{font-size:12px;color:var(--muted)}
.tgt img{width:100%;max-width:220px;aspect-ratio:1;image-rendering:pixelated;background:var(--wall);border-radius:4px}
.nav{display:flex;gap:6px;flex-wrap:wrap;justify-content:center}
button{font:500 13px var(--f-body);border:1px solid var(--rule);background:var(--panel);color:var(--fg);border-radius:6px;padding:6px 12px;cursor:pointer}
button:hover{border-color:var(--muted)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button.none[aria-pressed="true"]{background:var(--none);color:#fff;border-color:var(--none)}
label.sw{display:flex;gap:6px;align-items:center;font-size:13px;color:var(--muted);cursor:pointer}
.line{display:grid;grid-template-columns:repeat(auto-fill,minmax(126px,1fr));gap:12px}
.cand{position:relative;display:grid;gap:4px;padding:6px;border:2px solid transparent;border-radius:8px;background:var(--panel);cursor:pointer;text-align:left}
.cand img{width:100%;aspect-ratio:1;image-rendering:pixelated;background:var(--wall);border-radius:4px;display:block}
.cand .k{font:600 13px var(--f-mono)}
.cand .who{font:12px var(--f-mono);min-height:1.4em}
.cand[aria-pressed="true"]{border-color:var(--good)}
.cand .tag{position:absolute;top:10px;right:10px;font:600 11px var(--f-mono);padding:2px 6px;border-radius:4px;color:#fff;background:var(--good)}
.help{font-size:13px;color:var(--muted)}
.help kbd{font:12px var(--f-mono);border:1px solid var(--rule);border-radius:3px;padding:0 4px;background:var(--panel)}
.strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(34px,1fr));gap:4px}
.strip button{padding:4px 0;font:600 14px var(--f-display);border-radius:4px}
.strip button.done{background:var(--good);color:#fff;border-color:var(--good)}
.strip button.nonepick{background:var(--none);color:#fff;border-color:var(--none)}
.strip button.cur{outline:2px solid var(--fg);outline-offset:1px}
.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;align-items:start}
@media (max-width:820px){.charts{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:16px;display:grid;gap:8px;min-width:0;align-content:start}
.card h3{font:600 15px/1.2 var(--f-display);margin:0}
.card .note{font-size:13px;color:var(--muted);margin:0}
svg text{fill:var(--muted);font:11px var(--f-mono)}
svg .lab{fill:var(--fg);font:12px var(--f-body)}
.empty{font-size:13px;color:var(--muted);padding:12px 0}
.scroll{overflow-x:auto}
table.sum{border-collapse:collapse;width:100%;min-width:900px;font-size:13px}
table.sum th,table.sum td{padding:6px 8px;border-top:1px solid var(--rule);text-align:right;font-variant-numeric:tabular-nums}
table.sum thead th{border-top:0;font:500 11px/1.2 var(--f-mono);color:var(--muted);vertical-align:bottom}
table.sum th.o,table.sum td.o{text-align:left;white-space:nowrap}
table.sum td{font-family:var(--f-mono)}
table.sum td.best{color:var(--good);font-weight:600}
table.sum td.you{background:color-mix(in srgb,var(--good) 10%,transparent)}
table.sum i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
#markview{display:grid;gap:24px}
#markview[hidden],#guessview[hidden],#abview[hidden]{display:none}
.abdesk{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,190px) minmax(0,1fr);gap:16px;align-items:end;max-width:900px;margin:0 auto;width:100%}
.abdesk figure{margin:0;display:grid;gap:6px;justify-items:center;background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:12px}
.abdesk img{width:100%;aspect-ratio:1;image-rendering:pixelated;background:var(--wall);border-radius:4px}
.abside img{max-width:300px} .abtgt img{max-width:170px}
.abdesk figcaption{font:500 12px var(--f-mono);color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.abbtns{display:flex;flex-wrap:wrap;gap:8px;justify-content:center}
.abbtns button{padding:10px 16px;font-size:14px}
.abbtns kbd{font:12px var(--f-mono);border:1px solid var(--rule);border-radius:3px;padding:0 4px}
@media (max-width:640px){.abdesk{grid-template-columns:1fr 1fr}.abtgt{grid-column:1/-1;order:-1}}
.gv{display:grid;gap:18px}
.gdesk{display:grid;grid-template-columns:minmax(0,340px) minmax(0,1fr);gap:24px;align-items:start}
@media (max-width:820px){.gdesk{grid-template-columns:1fr}}
.gshow{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:16px;display:grid;gap:10px;justify-items:center}
.gshow img{width:100%;max-width:300px;aspect-ratio:1;image-rendering:pixelated;background:var(--wall);border-radius:4px}
.gcat{font:500 12px var(--f-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.gfb{min-height:2.2em;font:600 18px var(--f-display);text-align:center}
.gfb .ok{color:var(--good)} .gfb .no{color:var(--none)}
.gpad{display:grid;gap:12px}
.keys{display:grid;grid-template-columns:repeat(auto-fill,minmax(46px,1fr));gap:6px}
.keys button{font:600 20px var(--f-display);padding:8px 0;border-radius:6px}
.keys button.right{background:var(--good);color:#fff;border-color:var(--good)}
.keys button.wrong{background:var(--none);color:#fff;border-color:var(--none)}
#gopts label{display:flex;gap:6px;align-items:center;font-size:13px;color:var(--muted);cursor:pointer}
</style>
<div class="wrap">
  <header>
    <span class="eyebrow">umbra-bench · glyph objective study · blind</span>
    <h1>Can you read the shadow?</h1>
    <p class="sub">Guess the glyph: a shadow appears alone, you say which character it is. That measures readability without anyone judging similarity. The marking tabs show the target and let you mark every shadow that looks like it.</p>
  </header>
  <div class="tabs" role="tablist" id="tabs"></div>
  <section id="guessview" class="gv">
    <div class="bar"><span class="prog" id="gprog"></span>
      <span style="display:flex;gap:14px;align-items:center;flex-wrap:wrap" id="gopts"></span></div>
    <div class="gdesk">
      <div class="gshow">
        <span class="gcat" id="gcat"></span>
        <img id="gimg" alt="shadow to identify">
        <div class="gfb" id="gfb" aria-live="polite"></div>
      </div>
      <div class="gpad">
        <div class="keys" id="keys"></div>
        <div class="nav"><button id="gskip" title="can't tell (space)">Can't tell</button><button id="gnext" title="next (enter)" hidden>Next →</button></div>
        <div class="help">Type the character or click it · <kbd>space</kbd> can't tell · <kbd>enter</kbd> next. Every objective gets the same number of questions; the same character never comes twice in a row.</div>
      </div>
    </div>
    <div class="card">
      <h3>Readability by objective: you vs the machine judges</h3>
      <p class="note">Your accuracy on the shadows you saw, with a 95% interval; the classifier's and CLIP's top-1 on exactly the same shadows. Chance: 1/26 for letters, 1/10 for digits.</p>
      <div class="scroll"><table class="sum" id="gsum"></table></div>
    </div>
  </section>
  <section id="abview" class="gv" hidden>
    <div class="bar"><span class="prog" id="abprog"></span><span class="help">Blind: the two sides are the same objective under the two optimizers, side chosen at random, same ink. Nothing is revealed per question.</span></div>
    <div class="abdesk">
      <figure class="abside"><img id="abL" alt="left shadow"><figcaption>left</figcaption></figure>
      <figure class="abtgt"><span class="gcat" id="abcat"></span><img id="abT" alt="target"><figcaption>target</figcaption></figure>
      <figure class="abside"><img id="abR" alt="right shadow"><figcaption>right</figcaption></figure>
    </div>
    <div class="abbtns">
      <button id="ab-left" data-c="L"><kbd>←</kbd> Left looks more like it</button>
      <button id="ab-tie" data-c="tie"><kbd>↓</kbd> About the same</button>
      <button id="ab-right" data-c="R">Right looks more like it <kbd>→</kbd></button>
      <button id="ab-bad" data-c="bad"><kbd>4</kbd> Neither looks like it</button>
    </div>
    <div class="charts">
      <div class="card"><h3>Which optimizer you preferred</h3><p class="note">Ties and "neither" left out of the test; p is a two-sided sign test on the decided pairs.</p><div id="ab_all"></div></div>
      <div class="card"><h3>By objective</h3><p class="note">Bar = share of decided pairs where Adam won (50% = no preference). Number of decided pairs on the right.</p><div id="ab_obj"></div></div>
    </div>
  </section>
  <div id="markview" hidden>
  <div class="bar"><span class="prog" id="prog"></span>
    <span style="display:flex;gap:16px;align-items:center"><label class="sw"><input type="checkbox" id="names"> show names</label><span class="status" id="status">Loading saved picks…</span></span></div>
  <section class="desk">
    <div class="tgt">
      <span class="ch" id="ch"></span><span class="kind" id="kind"></span>
      <img id="timg" alt="target">
      <div class="nav">
        <button id="prev" title="previous (←)">← Prev</button>
        <button id="none" class="none" aria-pressed="false" title="none of them (0)">None look like it</button>
        <button id="next" title="next (→)">Next →</button>
      </div>
      <div class="help"><kbd>a</kbd>–<kbd>m</kbd> mark / unmark · <kbd>0</kbd> none · <kbd>←</kbd><kbd>→</kbd> move</div>
    </div>
    <div class="line" id="line"></div>
  </section>
  <section>
    <div class="bar"><h2>All characters</h2><span class="help">green = at least one marked good, red = none look like it</span></div>
    <div class="strip" id="strip"></div>
  </section>
  <section class="card">
    <h3>Average over all characters</h3>
    <p class="note">Seed 0, MuJoCo-measured. Green = best in column. "you: good" = share of the characters you judged where you marked this objective's shadow good.</p>
    <div class="scroll"><table class="sum" id="sum"></table></div>
  </section>
  <section class="charts">
    <div class="card"><h3>How often each objective is good</h3><p class="note">Share of judged characters where you marked it good.</p><div id="tally"></div></div>
    <div class="card"><h3>Which metric agrees with you?</h3><p class="note">Solid: the metric's top shadow is one you marked good. Faded: AUC, the chance the metric ranks a good shadow above a not-good one (0.5 = no better than chance).</p><div id="agree"></div></div>
  </section>
  </div>
</div>
<script>
const D = /*DATA*/;
const famOf = Object.fromEntries(D.objs.map(o => [o.o, o.family]));
const fam = {baseline: 'var(--base)', shape: 'var(--shape)', semantic: 'var(--sem)'};
const KEYS = 'abcdefghijklmnopqrstuvwxyz';
const $ = id => document.getElementById(id);
let tab = D.tabs[0].key, mode = 'guess';
const cur = Object.fromEntries(D.tabs.map(t => [t.key, 0]));
let store = {guesses: [], ab: []}; for (const t of D.tabs) store[t.key] = {};
let legacy = null, docRef = null, writing = Promise.resolve(), dirty = false, moved = false;
['pointerdown', 'keydown'].forEach(ev => document.addEventListener(ev, () => { moved = true; }, {once: true}));
const T = () => D.tabs.find(t => t.key === tab);
const P = () => store[tab];
const judged = p => !!(p && (p.none || (p.good && p.good.length)));

function order(gi) {  // fixed shuffle per glyph so a reload shows the same line-up
  let s = (gi + 1) * 2654435761 >>> 0;
  const rnd = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const a = T().glyphs[gi].cands.map((_, i) => i);
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

const HAS_AB = ['cma', 'adam'].every(k => D.tabs.some(t => t.key === k));
function renderTabs() {
  $('tabs').innerHTML = `<button role="tab" id="tab-guess" aria-selected="${mode === 'guess'}" data-m="guess">Guess the glyph<span class="n">${store.guesses.length}</span></button>` +
    (HAS_AB ? `<button role="tab" id="tab-ab" aria-selected="${mode === 'ab'}" data-m="ab">A/B · CMA-ES vs Adam<span class="n">${store.ab.length}</span></button>` : '') +
    D.tabs.map(t => `<button role="tab" id="tab-${t.key}" aria-selected="${mode === 'mark' && t.key === tab}" data-m="mark" data-t="${t.key}">Mark good · ${t.label}<span class="n">${t.glyphs.filter(g => judged(store[t.key][g.key])).length}/${t.glyphs.length}</span></button>`).join('');
}
function render() {
  $('guessview').hidden = mode !== 'guess';
  $('markview').hidden = mode !== 'mark';
  $('abview').hidden = mode !== 'ab';
  if (mode === 'guess') { renderTabs(); renderGuess(); } else if (mode === 'ab') { renderTabs(); renderAB(); } else renderDesk();
}

// ── A/B: CMA-ES vs Adam, blind ──────────────────────────────────────────
// a record is [glyph key, objective, choice: 'cma' | 'adam' | 'tie' | 'bad' | 'identical', side of CMA-ES: 'L' | 'R', ms]
let ab = null;
function abPair(gk, o) {
  const c = D.tabs.find(t => t.key === 'cma').glyphs.find(g => g.key === gk);
  const a = D.tabs.find(t => t.key === 'adam').glyphs.find(g => g.key === gk);
  if (!c || !a) return null;
  const cc = c.cands.find(x => x.o === o), ac = a.cands.find(x => x.o === o);
  return cc && ac ? {g: c, cma: cc, adam: ac} : null;
}
function nextAB() {
  const cg = D.tabs.find(t => t.key === 'cma'), objs = cg.objs;
  const done = new Set(store.ab.map(r => r[0] + '|' + r[1]));
  const count = {}; store.ab.forEach(r => { count[r[1]] = (count[r[1]] || 0) + 1; });
  const recent = new Set(store.ab.slice(-6).map(r => r[0]));
  const open = objs.filter(o => cg.glyphs.some(g => !done.has(g.key + '|' + o)));
  if (!open.length) { ab = null; return; }
  const minN = Math.min(...open.map(o => count[o] || 0));
  const pool = open.filter(o => (count[o] || 0) === minN);
  const o = pool[Math.floor(Math.random() * pool.length)];
  let gs = cg.glyphs.filter(g => !done.has(g.key + '|' + o) && !recent.has(g.key));
  if (!gs.length) gs = cg.glyphs.filter(g => !done.has(g.key + '|' + o));
  const g = gs[Math.floor(Math.random() * gs.length)];
  const p = abPair(g.key, o);
  if (!p || p.cma.img === p.adam.img) { store.ab.push([g.key, o, 'identical', 'L', 0]); return nextAB(); }  // nothing to judge
  ab = {...p, o, cmaSide: Math.random() < 0.5 ? 'L' : 'R', t0: performance.now()};
}
function answerAB(c) {
  if (!ab) return;
  const choice = c === 'tie' || c === 'bad' ? c : (c === ab.cmaSide ? 'cma' : 'adam');
  store.ab.push([ab.g.key, ab.o, choice, ab.cmaSide, Math.round(performance.now() - ab.t0)]);
  save(); nextAB(); renderAB();
}
function signP(a, b) {  // two-sided exact sign test
  const n = a + b, k = Math.min(a, b);
  if (!n) return 1;
  let s = 0, c = 1; for (let i = 0; i <= k; i++) { if (i) c = c * (n - i + 1) / i; s += c; }
  return Math.min(1, 2 * s / Math.pow(2, n));
}
function renderAB() {
  if (!ab) nextAB();
  const total = D.tabs.find(t => t.key === 'cma').glyphs.length * D.tabs.find(t => t.key === 'cma').objs.length;
  $('abprog').innerHTML = `<b>${store.ab.length}</b> / ${total} pairs judged`;
  if (ab) {
    const L = ab.cmaSide === 'L' ? ab.cma : ab.adam, R = ab.cmaSide === 'L' ? ab.adam : ab.cma;
    $('abL').src = L.img; $('abR').src = R.img; $('abT').src = ab.g.target;
    $('abcat').textContent = ab.g.label + ' · ' + {letters_upper: 'uppercase', letters_lower: 'lowercase', digits: 'digit'}[ab.g.sub];
  } else { $('abprog').innerHTML += ' · all done, thank you'; }
  document.querySelectorAll('.abbtns button').forEach(b => { b.disabled = !ab; });
  renderTabs(); renderABSummary();
}
function renderABSummary() {
  const R = store.ab.filter(r => r[2] !== 'identical');  // identical images were never shown
  if (!R.length) { $('ab_all').innerHTML = $('ab_obj').innerHTML = '<p class="empty">Judge a few pairs and the results appear here.</p>'; return; }
  const n = c => R.filter(r => r[2] === c).length;
  const a = n('adam'), c = n('cma'), t = n('tie'), b = n('bad'), p = signP(a, c);
  const rows = [{label: 'Adam better', v: a / R.length, color: '#3a3fb0'}, {label: 'CMA-ES better', v: c / R.length, color: '#0f6e66'},
    {label: 'about the same', v: t / R.length, color: 'var(--base)'}, {label: 'neither', v: b / R.length, color: 'var(--none)'}];
  $('ab_all').innerHTML = hbars(rows, r => Math.round(100 * r.v) + '%') +
    `<p class="note">${R.length} pairs. Of the ${a + c} decided, Adam won ${a + c ? Math.round(100 * a / (a + c)) : 0}% (p = ${p < 0.001 ? p.toExponential(1) : p.toFixed(3)}).</p>`;
  const by = {};
  R.forEach(r => { const x = by[r[1]] = by[r[1]] || {a: 0, c: 0}; if (r[2] === 'adam') x.a++; if (r[2] === 'cma') x.c++; });
  const orow = Object.entries(by).filter(([, x]) => x.a + x.c > 0).map(([o, x]) => ({label: o, v: x.a / (x.a + x.c), n: x.a + x.c, color: fam[famOf[o]]}))
    .sort((x, y) => y.v - x.v);
  $('ab_obj').innerHTML = orow.length ? hbars(orow, r => `${Math.round(100 * r.v)}% · ${r.n}`) : '<p class="empty">No decided pairs yet.</p>';
}
document.querySelector('.abbtns').addEventListener('click', e => { const b = e.target.closest('button'); if (b && !b.disabled) answerAB(b.dataset.c); });

// ── Guess the glyph ─────────────────────────────────────────────────────
const SUBS = {letters_upper: [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'], letters_lower: [...'abcdefghijklmnopqrstuvwxyz'], digits: [...'0123456789']};
const CAT = {letters_upper: 'uppercase letter', letters_lower: 'lowercase letter', digits: 'digit'};
const useTab = Object.fromEntries(D.tabs.map(t => [t.key, true]));
let trial = null, answered = null, advanceTimer = null;
const condName = (tk, o) => (D.tabs.length > 1 ? D.tabs.find(t => t.key === tk).label + ' · ' : '') + o;
function nextTrial() {
  clearTimeout(advanceTimer);
  const conds = D.tabs.filter(t => useTab[t.key]).flatMap(t => t.objs.map(o => [t.key, o]));
  if (!conds.length) { trial = null; return; }
  const count = {}, seen = {};
  store.guesses.forEach(([g, c]) => { count[c] = (count[c] || 0) + 1; (seen[c] = seen[c] || new Set()).add(g); });
  const recent = new Set(store.guesses.slice(-6).map(x => x[0]));
  const minN = Math.min(...conds.map(([tk, o]) => count[tk + ':' + o] || 0));
  const pool = conds.filter(([tk, o]) => (count[tk + ':' + o] || 0) === minN);
  const [tk, o] = pool[Math.floor(Math.random() * pool.length)];
  const t = D.tabs.find(x => x.key === tk), c = tk + ':' + o;
  let cand = t.glyphs.filter(g => !(seen[c] && seen[c].has(g.key)) && !recent.has(g.key));
  if (!cand.length) cand = t.glyphs.filter(g => !recent.has(g.key));
  if (!cand.length) cand = t.glyphs;
  const g = cand[Math.floor(Math.random() * cand.length)];
  trial = {tk, o, c, g, ci: g.cands.findIndex(x => x.o === o), t0: performance.now()};
  answered = null;
}
function answer(ch) {
  if (!trial || answered) return;
  const ok = ch === trial.g.label;
  answered = {ch, ok};
  store.guesses.push([trial.g.key, trial.c, ch, ok ? 1 : 0, Math.round(performance.now() - trial.t0)]);
  save(); renderGuess();
  if (ok) advanceTimer = setTimeout(() => { nextTrial(); renderGuess(); }, 650);
}
function renderGuess() {
  if (!trial) nextTrial();
  $('gopts').innerHTML = D.tabs.length > 1 ? D.tabs.map(t => `<label><input type="checkbox" id="use-${t.key}" data-t="${t.key}" ${useTab[t.key] ? 'checked' : ''}> ${t.label}</label>`).join('') : '';
  const n = store.guesses.length, k = store.guesses.filter(x => x[3]).length;
  $('gprog').innerHTML = `<b>${n}</b> answered · ${n ? Math.round(100 * k / n) : 0}% right`;
  if (!trial) { $('gimg').removeAttribute('src'); $('keys').innerHTML = '<p class="empty">Pick at least one optimizer above.</p>'; renderGuessSummary(); return; }
  const g = trial.g, c = g.cands[trial.ci];
  $('gcat').textContent = CAT[g.sub];
  $('gimg').src = c.img;
  $('keys').innerHTML = SUBS[g.sub].map(ch => {
    const cls = answered ? (ch === g.label ? 'right' : ch === answered.ch ? 'wrong' : '') : '';
    return `<button data-ch="${ch}" class="${cls}" aria-label="${ch}">${ch}</button>`;
  }).join('');
  $('gfb').innerHTML = !answered ? '' : answered.ok ? `<span class="ok">✓ ${g.label}</span>` :
    `<span class="no">✗ it was ${g.label}${answered.ch === '?' ? '' : ` (you said ${answered.ch})`}</span>`;
  $('gnext').hidden = !answered || answered.ok;
  $('gskip').hidden = !!answered;
  renderTabs(); renderGuessSummary();
}
function wilson(k, n) {
  if (!n) return [0, 0];
  const z = 1.96, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}
function renderGuessSummary() {
  const by = {};
  const look = {};
  D.tabs.forEach(t => t.glyphs.forEach(g => g.cands.forEach(c => { look[t.key + ':' + c.o + '|' + g.key] = c.m; })));
  store.guesses.forEach(([gk, c, , ok]) => {
    const m = look[c + '|' + gk]; if (!m) return;
    const b = by[c] = by[c] || {n: 0, k: 0, cls: 0, clip: 0};
    b.n += 1; b.k += ok; b.cls += m.cls_top1 || 0; b.clip += m.clip_top1 || 0;
  });
  const rows = Object.entries(by).map(([c, b]) => ({c, ...b, acc: b.k / b.n})).sort((a, b) => b.acc - a.acc || b.n - a.n);
  if (!rows.length) { $('gsum').innerHTML = '<tbody><tr><td class="o">Answer a few and the table fills in.</td></tr></tbody>'; return; }
  const tot = rows.reduce((s, r) => ({n: s.n + r.n, k: s.k + r.k, cls: s.cls + r.cls, clip: s.clip + r.clip}), {n: 0, k: 0, cls: 0, clip: 0});
  const pct = v => Math.round(100 * v) + '%';
  const best = Math.max(...rows.filter(r => r.n >= 5).map(r => r.acc), -1);
  $('gsum').innerHTML = '<thead><tr><th class="o">objective</th><th>shown</th><th>you ↑</th><th>95% interval</th><th>classifier ↑</th><th>CLIP ↑</th></tr></thead><tbody>' +
    rows.map(r => { const [lo, hi] = wilson(r.k, r.n), o = r.c.split(':')[1];
      return `<tr><td class="o"><i style="background:${fam[famOf[o]]}"></i>${condName(r.c.split(':')[0], o)}</td><td>${r.n}</td><td class="you ${r.n >= 5 && r.acc === best ? 'best' : ''}">${pct(r.acc)}</td><td>${pct(lo)}–${pct(hi)}</td><td>${pct(r.cls / r.n)}</td><td>${pct(r.clip / r.n)}</td></tr>`; }).join('') +
    `<tr><td class="o"><b>all</b></td><td>${tot.n}</td><td class="you">${pct(tot.k / tot.n)}</td><td></td><td>${pct(tot.cls / tot.n)}</td><td>${pct(tot.clip / tot.n)}</td></tr></tbody>`;
}
$('keys').addEventListener('click', e => { const b = e.target.closest('button'); if (b && b.dataset.ch) answer(b.dataset.ch); });
$('gskip').onclick = () => answer('?');
$('gnext').onclick = () => { nextTrial(); renderGuess(); };
$('gopts').addEventListener('change', e => { const t = e.target.dataset.t; if (t) { useTab[t] = e.target.checked; nextTrial(); renderGuess(); } });

function renderDesk() {
  const t = T(), gi = cur[tab], g = t.glyphs[gi], p = P()[g.key] || {};
  const show = $('names').checked, good = new Set(p.good || []);
  $('ch').textContent = g.label;
  $('kind').textContent = {letters_upper: 'uppercase', letters_lower: 'lowercase', digits: 'digit'}[g.sub];
  $('timg').src = g.target;
  $('none').setAttribute('aria-pressed', !!p.none);
  $('line').innerHTML = order(gi).map((ci, pos) => {
    const c = g.cands[ci], on = good.has(c.o);
    return `<button class="cand" data-o="${c.o}" aria-pressed="${on}" aria-label="candidate ${KEYS[pos]}">
      <span class="k">${KEYS[pos]}</span><img src="${c.img}" alt="">${on ? '<span class="tag">GOOD</span>' : ''}
      <span class="who" style="color:${show ? fam[famOf[c.o]] : 'transparent'}">${show ? c.o : '·'}</span></button>`;
  }).join('');
  const n = t.glyphs.filter(x => judged(P()[x.key])).length;
  $('prog').innerHTML = `<b>${n}</b> / ${t.glyphs.length} judged · ${g.label} (${gi + 1})`;
  $('strip').innerHTML = t.glyphs.map((x, i) => {
    const q = P()[x.key] || {};
    return `<button class="${q.good && q.good.length ? 'done' : q.none ? 'nonepick' : ''} ${i === gi ? 'cur' : ''}" data-i="${i}" aria-label="${x.label}">${x.label}</button>`;
  }).join('');
  renderTabs(); renderSummary(); renderStats();
}

function toggle(o) {
  const g = T().glyphs[cur[tab]], p = {...(P()[g.key] || {})};
  const s = new Set(p.good || []);
  s.has(o) ? s.delete(o) : s.add(o);
  P()[g.key] = s.size ? {good: [...s]} : {};
  save(); renderDesk();
}
function setNone() {
  const g = T().glyphs[cur[tab]];
  P()[g.key] = P()[g.key] && P()[g.key].none ? {} : {none: true};
  save(); renderDesk();
}
function go(d) { const n = T().glyphs.length; cur[tab] = (cur[tab] + d + n) % n; renderDesk(); window.scrollTo({top: 0}); }

function save() {  // one write at a time, coalesced; the whole state is one document
  if (!docRef) { $('status').textContent = 'Not saved: this view has no storage. Marks last until you close the page.'; return; }
  dirty = true;
  writing = writing.then(async () => {
    if (!dirty) return;
    dirty = false;
    const body = {...store, updated: new Date().toISOString()};
    if (legacy) body.picks = legacy;  // the v1 picks stay as they were
    try { await docRef.set(body); $('status').textContent = 'Saved'; }
    catch (e) {
      $('status').textContent = e && e.code === 'invalid_argument' ? 'You can view this page but not save marks.' : 'Could not save. Your marks are still on this page; the next mark retries.';
      dirty = true;
    }
  });
}

function renderSummary() {
  const t = T(), cols = D.summary_cols, jd = t.glyphs.filter(g => judged(P()[g.key]));
  const rate = o => jd.length ? jd.filter(g => (P()[g.key].good || []).includes(o)).length / jd.length : null;
  const best = {};
  cols.forEach(c => { const v = t.summary.map(r => r[c.k]).filter(v => v != null); best[c.k] = v.length ? (c.hb ? Math.max(...v) : Math.min(...v)) : null; });
  const rates = t.summary.map(r => rate(r.o)), bestRate = Math.max(...rates.filter(v => v != null), -1);
  const fmt = (k, v) => v == null ? '–' : k === 'seconds' ? v.toFixed(1) : v.toFixed(3);
  $('sum').innerHTML = `<thead><tr><th class="o">objective</th>${cols.map(c => `<th>${c.label} ${c.hb ? '↑' : '↓'}</th>`).join('')}<th>you: good ↑</th></tr></thead><tbody>` +
    t.summary.map((r, i) => `<tr><td class="o"><i style="background:${fam[famOf[r.o]]}"></i>${r.o}</td>` +
      cols.map(c => `<td class="${r[c.k] != null && Math.abs(r[c.k] - best[c.k]) < 1e-9 ? 'best' : ''}">${fmt(c.k, r[c.k])}</td>`).join('') +
      `<td class="you ${rates[i] != null && rates[i] === bestRate ? 'best' : ''}">${rates[i] == null ? '–' : Math.round(100 * rates[i]) + '%'}</td></tr>`).join('') + '</tbody>';
}

function svg(w, h, b) { return `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img">${b}</svg>`; }
function hbars(rows, fmt) {
  const W = 520, L = 170, R = 96, bh = 16, gap = 6, H = rows.length * (bh + gap) + 4, x = v => L + (W - L - R) * v;
  return svg(W, H, rows.map((r, i) => { const y = i * (bh + gap);
    return `<text class="lab" x="${L - 8}" y="${y + 12}" text-anchor="end">${r.label}</text>` +
      (r.bg != null ? `<rect x="${L}" y="${y}" width="${Math.max(0, x(r.bg) - L)}" height="${bh}" rx="2" fill="${r.color}" opacity=".22"/>` : '') +
      `<rect x="${L}" y="${y}" width="${Math.max(1, x(r.v) - L)}" height="${bh}" rx="2" fill="${r.color}"/><text x="${x(Math.max(r.v, r.bg || 0)) + 6}" y="${y + 12}">${fmt(r)}</text>`; }).join(''));
}
function renderStats() {
  const t = T(), jd = t.glyphs.filter(g => judged(P()[g.key]));
  if (!jd.length) { $('tally').innerHTML = $('agree').innerHTML = '<p class="empty">Mark a few characters and the tallies appear here.</p>'; return; }
  const rows = t.objs.map(o => ({label: o, v: jd.filter(g => (P()[g.key].good || []).includes(o)).length / jd.length, color: fam[famOf[o]]})).sort((a, b) => b.v - a.v);
  const nNone = jd.filter(g => P()[g.key].none).length;
  $('tally').innerHTML = hbars(rows, r => Math.round(100 * r.v) + '%') + `<p class="note">${jd.length} judged${nNone ? `, ${nNone} with none good` : ''}</p>`;
  const ag = D.metrics.map(m => {
    let hit = 0, nh = 0, auc = 0, na = 0;
    jd.forEach(g => {
      const good = new Set(P()[g.key].good || []);
      if (!good.size) return;
      const sc = g.cands.map(c => c.m[m.k] == null ? null : (m.hb ? 1 : -1) * c.m[m.k]);
      if (sc.some(v => v == null)) return;
      const top = Math.max(...sc), tops = g.cands.filter((c, i) => sc[i] >= top - 1e-9);
      nh += 1; hit += tops.filter(c => good.has(c.o)).length / tops.length;  // ties share the credit
      const G = g.cands.map((c, i) => [good.has(c.o), sc[i]]), pos = G.filter(x => x[0]), neg = G.filter(x => !x[0]);
      if (pos.length && neg.length) {
        let s = 0; pos.forEach(a => neg.forEach(b => { s += a[1] > b[1] + 1e-9 ? 1 : Math.abs(a[1] - b[1]) <= 1e-9 ? 0.5 : 0; }));
        auc += s / (pos.length * neg.length); na += 1;
      }
    });
    return {label: m.label, v: nh ? hit / nh : 0, bg: na ? auc / na : null, color: 'var(--accent)'};
  }).sort((a, b) => (b.bg || 0) - (a.bg || 0));
  $('agree').innerHTML = hbars(ag, r => `${Math.round(100 * r.v)}% · AUC ${r.bg == null ? '–' : r.bg.toFixed(2)}`);
}

$('tabs').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  mode = b.dataset.m; if (b.dataset.t) tab = b.dataset.t;
  render();
});
$('line').addEventListener('click', e => { const b = e.target.closest('.cand'); if (b) toggle(b.dataset.o); });
$('strip').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { cur[tab] = +b.dataset.i; renderDesk(); } });
$('prev').onclick = () => go(-1); $('next').onclick = () => go(1); $('none').onclick = setNone;
$('names').addEventListener('change', render);
document.addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey || e.target.tagName === 'INPUT') return;
  if (mode === 'ab') {
    const c = {ArrowLeft: 'L', '1': 'L', ArrowDown: 'tie', ' ': 'tie', '2': 'tie', ArrowRight: 'R', '3': 'R', '4': 'bad'}[e.key];
    if (c) { e.preventDefault(); answerAB(c); }
    return;
  }
  if (mode === 'guess') {
    if (e.key === 'Enter' && answered) { e.preventDefault(); nextTrial(); renderGuess(); return; }
    if (!trial || answered) return;
    if (e.key === ' ' || e.key === '?') { e.preventDefault(); answer('?'); return; }
    const sub = trial.g.sub, ch = sub === 'letters_upper' ? e.key.toUpperCase() : sub === 'letters_lower' ? e.key.toLowerCase() : e.key;
    if (e.key.length === 1 && SUBS[sub].includes(ch)) answer(ch);
    return;
  }
  const k = e.key.toLowerCase();
  if (k === 'arrowright') go(1); else if (k === 'arrowleft') go(-1); else if (k === '0') setNone();
  else if (k.length === 1 && KEYS.includes(k)) { const pos = KEYS.indexOf(k), ord = order(cur[tab]); if (pos < ord.length) toggle(T().glyphs[cur[tab]].cands[ord[pos]].o); }
});
render();

(async () => {
  try {
    const [db, user] = await Promise.all([window.claude?.use?.('db') ?? null, window.claude?.use?.('user') ?? null]);
    const uid = user ? await user.id() : null;
    if (!db || !uid) { $('status').textContent = 'Not saved: this view has no storage. Marks last until you close the page.'; return; }
    docRef = db.doc('picks/' + uid);
    const snap = await docRef.get();
    const data = snap.exists ? (snap.data() || {}) : {};
    legacy = data.picks || null;
    for (const t of D.tabs) store[t.key] = {...(data[t.key] || {}), ...store[t.key]};
    for (const k of Object.keys(data)) if (!(k in store) && k !== 'picks' && k !== 'updated') store[k] = data[k];  // tabs not built yet (e.g. adam) keep their marks
    store.guesses = [...(Array.isArray(data.guesses) ? data.guesses : []), ...store.guesses];
    store.ab = [...(Array.isArray(data.ab) ? data.ab : []), ...store.ab];
    let migrated = 0;
    if (legacy && !data.cma && store.cma) {  // v1: best/second → good set
      for (const [k, v] of Object.entries(legacy)) {
        if (store.cma[k]) continue;
        store.cma[k] = v.none ? {none: true} : {good: [v.best, v.second].filter(Boolean)};
        migrated += 1;
      }
    }
    if (!moved) for (const t of D.tabs) { const f = t.glyphs.findIndex(g => !judged(store[t.key][g.key])); if (f >= 0) cur[t.key] = f; }
    $('status').textContent = migrated ? `Loaded your earlier ${migrated} picks as "good" marks` : snap.exists ? 'Loaded saved marks' : 'Marks save as you go';
    if (migrated) save();
    trial = null; ab = null; render();
  } catch (e) { $('status').textContent = 'Could not load saved marks.'; }
})();
</script>
"""

if __name__ == "__main__":
    main()
