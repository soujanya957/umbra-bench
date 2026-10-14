"""Blind human picking page for the glyph objective study.

For every character, the 13 objectives' seed-0 shadows are shown in a fixed
shuffled order with their names hidden; the viewer picks the one that looks most
like the target (and optionally a second). Picks live in the artifact's db at
`picks/<viewer id>` and are read back with ArtifactData. The page itself
computes which objective wins most often and how often each machine metric's
favourite matches the human pick.

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
OUT = os.path.join(BENCH, "results", "fastshadow", "objective_study")
sys.path.insert(0, os.path.join(BENCH, "scripts"))
from make_objective_report import OBJ  # noqa: E402
from run_base_optimizer import load_target  # noqa: E402

ORDER = {"letters_upper": 0, "letters_lower": 1, "digits": 2}
METRICS = [  # key, label, higher is better
    ("iou", "IoU", True), ("tc_iou", "tc_iou", True), ("boundary_iou", "boundary IoU", True), ("nsd", "NSD", True),
    ("chamfer", "chamfer", False), ("cldice", "clDice", True), ("betti_error", "Betti error", False),
    ("holes_filled", "holes filled", False), ("cls_p", "glyph classifier p", True), ("clip_p", "CLIP p (ViT-B-32)", True),
]
INK, WALL = np.array([38, 36, 44]), np.array([247, 245, 240])


def uri(mask):
    img = np.tile(WALL, mask.shape + (1,))
    img[mask] = INK
    b = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def main():
    G = [json.loads(l) for l in open(os.path.join(OUT, "graded.jsonl"), encoding="utf-8")]
    s0 = defaultdict(dict)
    for g in G:
        if g["seed"] == 0:
            s0[g["id"]][g["objective"]] = g
    objs = [o for o, *_ in OBJ if all(o in v for v in s0.values())]
    ids = sorted(s0, key=lambda i: (ORDER.get(i.split("/")[0], 9), i))
    root = os.path.join(BENCH, "optimized", "big-budget-grounded")
    glyphs = []
    for i in ids:
        sub, stem = i.split("/")
        T = load_target(os.path.join(root, sub, stem, f"{stem}_shown.png"), 128) > 0.5
        cands = []
        for o in objs:
            r = s0[i][o]
            S = load_target(os.path.join(OUT, r["mask"]), 128) > 0.5
            cands.append({"o": o, "img": uri(S), "m": {k: r.get(k) for k, *_ in METRICS}})
        glyphs.append({"key": i.replace("/", "__"), "sub": sub, "label": stem.split("_")[0],
                       "target": uri(T), "cands": cands})
    D = {"objs": [{"o": o, "label": l, "family": f} for o, l, f, _ in OBJ if o in objs],
         "metrics": [{"k": k, "label": l, "hb": hb} for k, l, hb in METRICS], "glyphs": glyphs}
    out = os.path.join(BENCH, "results", "fastshadow", "report", "pick_page.html")
    open(out, "w", encoding="utf-8").write(TEMPLATE.replace("/*DATA*/", json.dumps(D)))
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.2f} MB, {len(glyphs)} glyphs × {len(objs)} objectives")


TEMPLATE = r"""<title>Shadow Glyph Picker</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
/* Layout: a judging desk. Target on the left, the blind line-up on the right, tallies below. */
:root{
  --bg:#f2f3f5; --panel:#ffffff; --fg:#1d1e24; --muted:#5f6370; --rule:#d9dbe1;
  --pick:#3d5fd6; --second:#1f9e8f; --none:#c2473a; --shape:#3d5fd6; --sem:#b04a8f; --base:#8a8e9a;
  --ink:#26242c; --wall:#f7f5f0;
  --f-display:"IBM Plex Sans Condensed","Arial Narrow",sans-serif;
  --f-body:"IBM Plex Sans",system-ui,sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#15161b; --panel:#1d1f26; --fg:#e8e9ee; --muted:#9a9eab; --rule:#2e313b;
  --pick:#7d97ff; --second:#3cc7b5; --none:#ef7a6c; --shape:#7d97ff; --sem:#e07fc2; --base:#9ea2ae; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#15161b; --panel:#1d1f26; --fg:#e8e9ee; --muted:#9a9eab; --rule:#2e313b;
  --pick:#7d97ff; --second:#3cc7b5; --none:#ef7a6c; --shape:#7d97ff; --sem:#e07fc2; --base:#9ea2ae; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.5 var(--f-body)}
.wrap{max-width:1180px;margin:0 auto;padding-block:24px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:28px}
h1{font:600 clamp(26px,4vw,36px)/1.1 var(--f-display);margin:0}
h2{font:600 20px/1.2 var(--f-display);margin:0}
.sub{color:var(--muted);margin:4px 0 0;max-width:70ch}
.eyebrow{font:500 11px/1 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between}
.prog{font:500 13px var(--f-mono);color:var(--muted)}
.prog b{color:var(--fg)}
.status{font-size:13px;color:var(--muted)}
.desk{display:grid;grid-template-columns:minmax(0,260px) minmax(0,1fr);gap:20px;align-items:start}
@media (max-width:820px){.desk{grid-template-columns:1fr}}
.tgt{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:14px;display:grid;gap:10px;justify-items:center;position:sticky;top:calc(env(safe-area-inset-top,0px) + 8px)}
.tgt .ch{font:600 44px/1 var(--f-display)}
.tgt .kind{font-size:12px;color:var(--muted)}
.tgt img{width:100%;max-width:224px;aspect-ratio:1;image-rendering:pixelated;background:var(--wall);border-radius:4px}
.nav{display:flex;gap:6px;flex-wrap:wrap;justify-content:center}
button{font:500 13px var(--f-body);border:1px solid var(--rule);background:var(--panel);color:var(--fg);border-radius:6px;padding:6px 12px;cursor:pointer}
button:hover{border-color:var(--muted)}
button:focus-visible{outline:2px solid var(--pick);outline-offset:2px}
button.none[aria-pressed="true"]{background:var(--none);color:#fff;border-color:var(--none)}
.line{display:grid;grid-template-columns:repeat(auto-fill,minmax(128px,1fr));gap:12px}
.cand{position:relative;display:grid;gap:4px;padding:6px;border:2px solid transparent;border-radius:8px;background:var(--panel);cursor:pointer;text-align:left}
.cand img{width:100%;aspect-ratio:1;image-rendering:pixelated;background:var(--wall);border-radius:4px;display:block}
.cand .k{font:600 13px var(--f-mono)}
.cand .who{font:12px var(--f-mono);color:var(--muted);min-height:1.4em}
.cand[data-state="best"]{border-color:var(--pick)}
.cand[data-state="second"]{border-color:var(--second)}
.cand .tag{position:absolute;top:10px;right:10px;font:600 11px var(--f-mono);padding:2px 6px;border-radius:4px;color:#fff}
.cand[data-state="best"] .tag{background:var(--pick)}
.cand[data-state="second"] .tag{background:var(--second)}
.help{font-size:13px;color:var(--muted)}
.help kbd{font:12px var(--f-mono);border:1px solid var(--rule);border-radius:3px;padding:0 4px;background:var(--panel)}
.strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(34px,1fr));gap:4px}
.strip button{padding:4px 0;font:600 14px var(--f-display);border-radius:4px}
.strip button.done{background:var(--pick);color:#fff;border-color:var(--pick)}
.strip button.nonepick{background:var(--none);color:#fff;border-color:var(--none)}
.strip button.cur{outline:2px solid var(--fg);outline-offset:1px}
.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
@media (max-width:820px){.charts{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:16px;display:grid;gap:8px;min-width:0}
.card h3{font:600 15px/1.2 var(--f-display);margin:0}
.card .note{font-size:13px;color:var(--muted);margin:0}
svg text{fill:var(--muted);font:11px var(--f-mono)}
svg .lab{fill:var(--fg);font:12px var(--f-body)}
.empty{font-size:13px;color:var(--muted);padding:12px 0}
</style>
<div class="wrap">
  <header>
    <span class="eyebrow">umbra-bench · glyph objective study · blind</span>
    <h1>Which shadow looks most like the target?</h1>
    <p class="sub">Each card is a shadow cast by the three arms under a different objective, in a shuffled order with names hidden. Pick the one most like the target; a second pick is optional. Names appear after you pick.</p>
  </header>
  <div class="bar"><span class="prog" id="prog"></span><span class="status" id="status">Loading saved picks…</span></div>
  <section class="desk">
    <div class="tgt">
      <span class="ch" id="ch"></span><span class="kind" id="kind"></span>
      <img id="timg" alt="target">
      <div class="nav">
        <button id="prev" title="previous (←)">← Prev</button>
        <button id="none" class="none" aria-pressed="false" title="none of them (0)">None look like it</button>
        <button id="next" title="next (→)">Next →</button>
      </div>
      <div class="help"><kbd>a</kbd>–<kbd>m</kbd> pick · again for 2nd · <kbd>0</kbd> none · <kbd>←</kbd><kbd>→</kbd> move</div>
    </div>
    <div class="line" id="line"></div>
  </section>
  <section>
    <div class="bar"><h2>All characters</h2><span class="help">blue = picked, red = none look like it</span></div>
    <div class="strip" id="strip"></div>
  </section>
  <section class="charts">
    <div class="card"><h3>Your best picks, by objective</h3><p class="note">Second picks count half.</p><div id="tally"></div></div>
    <div class="card"><h3>Which metric agrees with you?</h3><p class="note">Of your picks, how often the metric also ranks your pick best of the 13 (ties count). Faded bar: mean percentile of your pick under the metric.</p><div id="agree"></div></div>
  </section>
</div>
<script>
const D = /*DATA*/;
const fam = {baseline: 'var(--base)', shape: 'var(--shape)', semantic: 'var(--sem)'};
const famOf = Object.fromEntries(D.objs.map(o => [o.o, o.family]));
const KEYS = 'abcdefghijklm';
const $ = id => document.getElementById(id);
let picks = {};   // glyph key -> {best, second, none}
let cur = 0, db = null, docRef = null, writing = Promise.resolve(), dirty = false, moved = false;
document.addEventListener('pointerdown', () => { moved = true; }, {once: true});
document.addEventListener('keydown', () => { moved = true; }, {once: true});

// fixed shuffle per glyph so a reload shows the same line-up
function order(gi) {
  let s = (gi + 1) * 2654435761 >>> 0;
  const rnd = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const a = D.glyphs[gi].cands.map((_, i) => i);
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

function renderDesk() {
  const g = D.glyphs[cur], p = picks[g.key] || {};
  $('ch').textContent = g.label;
  $('kind').textContent = {letters_upper: 'uppercase', letters_lower: 'lowercase', digits: 'digit'}[g.sub];
  $('timg').src = g.target;
  $('none').setAttribute('aria-pressed', !!p.none);
  const revealed = !!(p.best || p.none);
  $('line').innerHTML = order(cur).map((ci, pos) => {
    const c = g.cands[ci], st = p.best === c.o ? 'best' : p.second === c.o ? 'second' : '';
    return `<button class="cand" data-o="${c.o}" data-state="${st}" aria-label="candidate ${KEYS[pos]}">
      <span class="k">${KEYS[pos]}</span><img src="${c.img}" alt="">${st ? `<span class="tag">${st === 'best' ? 'BEST' : '2ND'}</span>` : ''}
      <span class="who" style="color:${revealed ? fam[famOf[c.o]] : 'transparent'}">${revealed ? c.o : '·'}</span></button>`;
  }).join('');
  renderProgress();
}
function renderProgress() {
  const n = D.glyphs.filter(g => picks[g.key] && (picks[g.key].best || picks[g.key].none)).length;
  $('prog').innerHTML = `<b>${n}</b> / ${D.glyphs.length} picked · ${D.glyphs[cur].label} (${cur + 1})`;
  $('strip').innerHTML = D.glyphs.map((g, i) => {
    const p = picks[g.key] || {};
    return `<button class="${p.best ? 'done' : p.none ? 'nonepick' : ''} ${i === cur ? 'cur' : ''}" data-i="${i}" aria-label="${g.label}">${g.label}</button>`;
  }).join('');
  renderStats();
}

function choose(o) {
  const g = D.glyphs[cur], p = {...(picks[g.key] || {})};
  delete p.none;
  if (p.best === o) { p.best = p.second || null; p.second = null; }       // unpick best: second moves up
  else if (p.second === o) p.second = null;
  else if (!p.best) p.best = o;
  else p.second = o;
  if (!p.best) delete p.best;
  if (!p.second) delete p.second;
  picks[g.key] = p;
  save(); renderDesk();
}
function setNone() {
  const g = D.glyphs[cur];
  picks[g.key] = picks[g.key] && picks[g.key].none ? {} : {none: true};
  save(); renderDesk();
}
function go(d) { cur = (cur + d + D.glyphs.length) % D.glyphs.length; renderDesk(); window.scrollTo({top: 0}); }

// one write at a time, coalesced: the whole picks map lives in one document
function save() {
  if (!docRef) { $('status').textContent = 'Not saved: this view has no storage. Picks last until you close the page.'; return; }
  dirty = true;
  writing = writing.then(async () => {
    if (!dirty) return;
    dirty = false;
    try {
      await docRef.set({picks, updated: new Date().toISOString()});
      $('status').textContent = 'Saved';
    } catch (e) {
      $('status').textContent = e && e.code === 'invalid_argument' ? 'You can view this page but not save picks.' : 'Could not save. Your picks are still on this page; try another pick to retry.';
      dirty = true;
    }
  });
}

function svg(w, h, b) { return `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img">${b}</svg>`; }
function hbars(rows, fmt, max) {
  const W = 520, L = 170, R = 60, bh = 16, gap = 6, H = rows.length * (bh + gap) + 4;
  const x = v => L + (W - L - R) * v / (max || 1);
  return svg(W, H, rows.map((r, i) => { const y = i * (bh + gap);
    return `<text class="lab" x="${L - 8}" y="${y + 12}" text-anchor="end">${r.label}</text>` +
      (r.bg != null ? `<rect x="${L}" y="${y}" width="${Math.max(0, x(r.bg) - L)}" height="${bh}" rx="2" fill="${r.color}" opacity=".22"/>` : '') +
      `<rect x="${L}" y="${y}" width="${Math.max(1, x(r.v) - L)}" height="${bh}" rx="2" fill="${r.color}"/><text x="${x(Math.max(r.v, r.bg || 0)) + 6}" y="${y + 12}">${fmt(r)}</text>`; }).join(''));
}
function renderStats() {
  const done = D.glyphs.filter(g => picks[g.key] && picks[g.key].best);
  const nNone = D.glyphs.filter(g => picks[g.key] && picks[g.key].none).length;
  if (!done.length) { $('tally').innerHTML = $('agree').innerHTML = '<p class="empty">Pick a few glyphs and the tallies appear here.</p>'; return; }
  const t = Object.fromEntries(D.objs.map(o => [o.o, 0]));
  done.forEach(g => { t[picks[g.key].best] += 1; if (picks[g.key].second) t[picks[g.key].second] += 0.5; });
  const rows = D.objs.map(o => ({label: o.o, v: t[o.o], color: fam[o.family]})).sort((a, b) => b.v - a.v);
  $('tally').innerHTML = hbars(rows, r => r.v, Math.max(...rows.map(r => r.v))) +
    `<p class="note">${done.length} picked${nNone ? ` · ${nNone} marked none` : ''}</p>`;
  const ag = D.metrics.map(m => {
    let hit = 0, n = 0, pct = 0;
    done.forEach(g => {
      const vals = g.cands.map(c => c.m[m.k]);
      if (vals.some(v => v == null)) return;
      const sgn = m.hb ? 1 : -1, pv = sgn * g.cands.find(c => c.o === picks[g.key].best).m[m.k];
      const best = Math.max(...vals.map(v => sgn * v));
      n += 1; if (pv >= best - 1e-9) hit += 1;
      const worse = vals.filter(v => sgn * v < pv - 1e-9).length, ties = vals.filter(v => Math.abs(sgn * v - pv) <= 1e-9).length - 1;
      pct += (worse + ties / 2) / (vals.length - 1);
    });
    return {label: m.label, v: n ? hit / n : 0, bg: n ? pct / n : 0, n, color: 'var(--pick)'};
  }).sort((a, b) => b.v - a.v || b.bg - a.bg);
  $('agree').innerHTML = hbars(ag, r => `${Math.round(100 * r.v)}% · ${Math.round(100 * r.bg)}th pct` + (r.n < done.length ? ` (${r.n})` : ''), 1) +
    `<p class="note">Chance for a single random metric: ${Math.round(100 / D.objs.length)}% top-1, 50th percentile.</p>`;
}

$('line').addEventListener('click', e => { const b = e.target.closest('.cand'); if (b) choose(b.dataset.o); });
$('strip').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { cur = +b.dataset.i; renderDesk(); } });
$('prev').onclick = () => go(-1); $('next').onclick = () => go(1); $('none').onclick = setNone;
document.addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === 'arrowright') go(1); else if (k === 'arrowleft') go(-1); else if (k === '0') setNone();
  else if (KEYS.includes(k) && k.length === 1) { const pos = KEYS.indexOf(k), o = order(cur)[pos]; if (o != null) choose(D.glyphs[cur].cands[o].o); }
});
renderDesk();

(async () => {
  try {
    const [d, user] = await Promise.all([window.claude?.use?.('db') ?? null, window.claude?.use?.('user') ?? null]);
    const uid = user ? await user.id() : null;
    if (!d || !uid) { $('status').textContent = 'Not saved: this view has no storage. Picks last until you close the page.'; return; }
    db = d; docRef = db.doc('picks/' + uid);
    const snap = await docRef.get();
    if (snap.exists) {
      const saved = (snap.data() || {}).picks || {};
      picks = {...saved, ...picks};
      const first = D.glyphs.findIndex(g => !(picks[g.key] && (picks[g.key].best || picks[g.key].none)));
      if (first >= 0 && !moved) cur = first;  // resume at the first unpicked glyph
      $('status').textContent = `Loaded ${Object.keys(saved).length} saved picks`;
    } else $('status').textContent = 'Picks save as you go';
    renderDesk();
  } catch (e) { $('status').textContent = 'Could not load saved picks.'; }
})();
</script>
"""

if __name__ == "__main__":
    main()
