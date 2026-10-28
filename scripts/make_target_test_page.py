"""Blind reading test for the target designer (scripts/target_designer.py): does a shadow made for
the chosen font read better than one made for the benchmark's font?

For every character: the best of 3 fresh-seed solves (seeds 3-5) for the chosen target and for
the benchmark's own target, both shown as plain shadows (same ink, no target, no font name).
  1. Reading, two blocks: every character appears once per block, in one version per block;
     which version comes first is counterbalanced per rater (half the characters each way).
     The category (uppercase / lowercase / digit) is given, nothing else; no feedback.
  2. Side by side: "which reads more like X?", each pair twice with sides swapped, plus
     "can't tell". Characters where both versions are the same shadow are left out here and
     shown once in the reading part.
Answers go to the artifact's db, one document per rater (tests/<id>), readable by the owner.
Publish with capabilities {"user": {}, "db": {"rules": [{"path": "tests", "read": "admin", "write": "owner"},
{"path": "tests/{self}", "read": "interact", "write": "interact"}]}}: without "user" every rater gets
"Sign in needed" (no id); without the rules every rater can read and overwrite every other rater's answers.
Raters need Contributor access, which only people in the owner's organization (Team/Enterprise plan)
can hold; outside raters can only be email-invited Editors (no public link), and Editors pass the
"admin" read rule, so they could read other raters' answers.

    python scripts/make_target_test_page.py
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

TD = os.path.join(BENCH, "results", "fastshadow", "target_designer")
OUT = os.path.join(TD, "page", "shadow_reading_test.html")
WALL, INK = np.array([236, 233, 226]), np.array([34, 33, 38])
ORDER = {"letters_upper": 0, "letters_lower": 1, "digits": 2}


def uri(mask):
    img = np.tile(WALL, mask.shape + (1,))
    img[mask] = INK
    b = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def main():
    picks = json.load(open(os.path.join(TD, "picks.json"), encoding="utf-8"))
    rows = [json.loads(l) for l in open(os.path.join(TD, "solves.jsonl"), encoding="utf-8")]
    by = defaultdict(list)
    for r in rows:
        if r["seed"] in (3, 4, 5):
            by[r["id"]].append(r)
    items = []
    for ch, p in sorted(picks.items(), key=lambda kv: (ORDER[[r for r in rows if r["ch"] == kv[0]][0]["sub"]], kv[0])):
        sub = next(r["sub"] for r in rows if r["ch"] == ch)
        bench_id = next(r["id"] for r in rows if r["ch"] == ch and "BENCH:" in r["id"])
        best = {}
        for arm, cid in (("chosen", p["chosen"]), ("bench", bench_id)):
            rr = [r for r in by[cid] if r["collision_free"] and r["lowest_mm"] >= -5] or by[cid]
            b = max(rr, key=lambda r: r["iou"])
            best[arm] = (load_target(os.path.join(BENCH, b["mask"]), 128) > 0.5, b)
        same = bool((best["chosen"][0] ^ best["bench"][0]).sum() == 0)
        items.append({"ch": ch, "sub": sub, "same": same,
                      "chosen": uri(best["chosen"][0]), "bench": None if same else uri(best["bench"][0])})
    D = {"items": items, "version": 1}
    html = TEMPLATE.replace("/*DATA*/", json.dumps(D))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(html)
    n_same = sum(i["same"] for i in items)
    print("wrote", OUT, f"{len(html) / 1e3:.0f} KB; {len(items)} characters, {n_same} where both versions are the same shadow")


TEMPLATE = r"""<title>Shadow Reading Test</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=IBM+Plex+Mono:wght@500&display=swap">
<style>
/* Layout: one task at a time on a wall-coloured stage; the shadow is the only large object. */
:root{
  --bg:#f3f1ec; --stage:#e6e2d9; --panel:#fbfaf7; --fg:#1f1e24; --muted:#67646d; --rule:#d6d1c6;
  --accent:#2e5f8a; --accent-fg:#ffffff; --ok:#2f7a55;
  --f-ui:"Atkinson Hyperlegible",system-ui,sans-serif;
  --f-num:"IBM Plex Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#17181c; --stage:#22242a; --panel:#1d1f24; --fg:#ecebe8; --muted:#a3a1aa; --rule:#33353d;
  --accent:#7fb0dc; --accent-fg:#101318; --ok:#6cc59a; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#17181c; --stage:#22242a; --panel:#1d1f24; --fg:#ecebe8; --muted:#a3a1aa; --rule:#33353d;
  --accent:#7fb0dc; --accent-fg:#101318; --ok:#6cc59a; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:16px/1.5 var(--f-ui)}
.wrap{max-width:760px;margin:0 auto;padding-block:24px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:18px}
h1{font:700 clamp(24px,4vw,32px)/1.15 var(--f-ui);margin:0;text-wrap:balance}
p{margin:0;max-width:62ch}
.muted{color:var(--muted)}
.bar{height:6px;background:var(--rule);border-radius:3px;overflow:hidden}
.bar i{display:block;height:100%;width:0;background:var(--accent);transition:width .2s}
.meta{display:flex;justify-content:space-between;gap:12px;font:500 13px var(--f-num);color:var(--muted);font-variant-numeric:tabular-nums}
.stage{background:var(--stage);border-radius:10px;padding:20px;display:grid;gap:16px;justify-items:center}
.shadow{width:min(72vw,300px);aspect-ratio:1;max-width:100%;image-rendering:pixelated;border-radius:6px;display:block}
.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;width:100%}
.pair figure{margin:0;display:grid;gap:6px;justify-items:center}
.pair .shadow{width:100%}
.pair figcaption{font:500 13px var(--f-num);color:var(--muted)}
.chip{font:700 13px var(--f-ui);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.ask{font:700 20px var(--f-ui);text-align:center}
.ask b{font-size:28px;padding:0 6px;border:1px solid var(--rule);border-radius:6px;background:var(--panel)}
form{display:flex;gap:10px;justify-content:center;flex-wrap:wrap}
input[type=text]{font:700 28px var(--f-ui);width:3.2em;text-align:center;padding:6px;border:2px solid var(--rule);border-radius:8px;background:var(--panel);color:var(--fg)}
input[type=text]:focus-visible{outline:none;border-color:var(--accent)}
button{font:700 16px var(--f-ui);border:0;border-radius:8px;padding:10px 18px;background:var(--accent);color:var(--accent-fg);cursor:pointer}
button.ghost{background:var(--panel);color:var(--fg);border:1px solid var(--rule)}
button:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
button:disabled{opacity:.45;cursor:default}
.row{display:flex;gap:10px;flex-wrap:wrap;justify-content:center}
.note{font-size:14px;color:var(--muted)}
.done{color:var(--ok);font-weight:700}
kbd{font:500 12px var(--f-num);border:1px solid var(--rule);border-radius:4px;padding:1px 5px;background:var(--panel)}
@media (prefers-reduced-motion: reduce){.bar i{transition:none}}
</style>
<div class="wrap">
  <header style="display:grid;gap:6px">
    <span class="chip">Robot-arm shadows</span>
    <h1>Shadow Reading Test</h1>
  </header>
  <div class="bar" aria-hidden="true"><i id="bar"></i></div>
  <div class="meta"><span id="phase">Loading</span><span id="count"></span></div>
  <main id="main" class="stage" aria-live="polite">
    <p class="muted">Loading the test…</p>
  </main>
  <p class="note" id="foot">Your answers save as you go, so you can stop and come back. There is no right-or-wrong feedback until the end of the study.</p>
</div>
<script>
const D = /*DATA*/;
const KIND = {letters_upper: 'Capital letter', letters_lower: 'Lowercase letter', digits: 'Digit'};
let db = null, ref = null, state = {reads: {}, ab: {}}, plan = null, saving = Promise.resolve(), exists = false, blocked = false, lastOk = true;
const $ = id => document.getElementById(id);
const FOOT = $('foot').textContent;

function hash(s) { let h = 2166136261; for (const c of s) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return h >>> 0; }
function rng(seed) { let s = seed || 1; return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return ((s >>> 0) % 1e6) / 1e6; }; }
function shuffle(a, r) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }

function makePlan(uid) {
  const r = rng(hash(uid));
  const items = D.items.map((it, k) => ({...it, k}));
  // which version each character shows first: half each way, balanced within each category
  const firstChosen = {};
  for (const sub of Object.keys(KIND)) {
    const chars = shuffle(items.filter(i => i.sub === sub && !i.same), r);
    chars.forEach((it, n) => { firstChosen[it.k] = n % 2 === 0; });
  }
  const b1 = [], b2 = [];
  for (const it of items) {
    if (it.same) { b1.push({key: `${it.k}:same`, it, arm: 'same', block: 1}); continue; }
    const fc = firstChosen[it.k];
    b1.push({key: `${it.k}:${fc ? 'chosen' : 'bench'}`, it, arm: fc ? 'chosen' : 'bench', block: 1});
    b2.push({key: `${it.k}:${fc ? 'bench' : 'chosen'}`, it, arm: fc ? 'bench' : 'chosen', block: 2});
  }
  const reads = [...shuffle(b1, r), ...shuffle(b2, r)];
  const ab = [];
  for (const it of items) if (!it.same) {
    const left = r() < 0.5;  // one draw per character, so the second showing really has the sides swapped
    for (const rep of [0, 1]) ab.push({key: `${it.k}:r${rep}`, it, rep, chosenLeft: left !== (rep === 1)});
  }
  return {reads, ab: shuffle(ab, r)};
}

const NO_ACCESS = 'Your answers cannot be saved with your current access to this page. Ask the owner for Contributor access.';
function stop(phase, msg) { blocked = true; $('phase').textContent = phase; $('main').innerHTML = `<p>${msg}</p>`; }

async function save() {
  // one write at a time to this rater's document; each write carries every answer so far, so an
  // answer whose write failed goes out again with the next one (update merges nested objects)
  saving = saving.then(async () => {
    for (let t = 0; t < 2 && !blocked; t++) {
      try {
        if (!exists) { if (!(await ref.get()).exists) await ref.set({version: D.version, reads: {}, ab: {}}); exists = true; }
        await ref.update({reads: state.reads, ab: state.ab});
        $('foot').textContent = FOOT;
        lastOk = true;
        return;
      } catch (e) {
        if (e && e.code === 'invalid_argument') return stop('Cannot save', NO_ACCESS);
        if (t === 0) await new Promise(res => setTimeout(res, 500 + 1000 * Math.random()));
        else { lastOk = false; $('foot').textContent = 'An answer did not save. Check your connection; the next answer will try again.'; }
      }
    }
  });
  return saving;
}

function progress() {
  const nR = plan.reads.length, nA = plan.ab.length;
  const dR = plan.reads.filter(x => state.reads[x.key]).length, dA = plan.ab.filter(x => state.ab[x.key]).length;
  $('bar').style.width = `${100 * (dR + dA) / (nR + nA)}%`;
  $('count').textContent = `${dR + dA} / ${nR + nA}`;
  return {dR, dA, nR, nA};
}

function next() {
  if (blocked) return;
  const p = progress();
  const r = plan.reads.find(x => !state.reads[x.key]);
  if (r) return showRead(r, p);
  const a = plan.ab.find(x => !state.ab[x.key]);
  if (a) return showAB(a, p);
  $('phase').textContent = 'Finished';
  $('main').innerHTML = '<p class="done">All done. Thank you!</p><p class="muted" id="fin">Saving your last answers…</p>';
  saving.then(() => { if (!blocked && $('fin')) $('fin').textContent = lastOk ? 'Your answers are saved. You can close this page.'
    : 'Your last answers did not save. Keep this page open, check your connection and reload to try again.'; });
}

function showRead(x, p) {
  const block = x.block === 1 || x.arm === 'same' ? 1 : 2;
  $('phase').textContent = `Part 1 of 2 · reading · block ${block}`;
  const t0 = performance.now();
  $('main').innerHTML = `<span class="chip">${KIND[x.it.sub]}</span>
    <img class="shadow" src="${x.arm === 'bench' ? x.it.bench : x.it.chosen}" alt="A shadow to read">
    <form id="f" autocomplete="off"><label for="ans" class="ask">Which character is this?</label>
      <input id="ans" type="text" maxlength="1" autocapitalize="off" autocorrect="off" spellcheck="false" aria-label="Your answer, one character" required>
      <button type="submit">Next</button>
      <button type="button" class="ghost" id="skip">Can't tell</button></form>`;
  const go = async ans => {
    const rec = {answer: ans, arm: x.arm, ch: x.it.ch, sub: x.it.sub, block: x.block, ms: Math.round(performance.now() - t0), n: p.dR + p.dA};
    state.reads[x.key] = rec;
    const s = save();
    next();
    await s;
  };
  $('f').addEventListener('submit', e => { e.preventDefault(); const v = $('ans').value.trim(); if (v) go(v); });
  $('skip').addEventListener('click', () => go(''));
  $('ans').focus();
}

function showAB(x, p) {
  $('phase').textContent = 'Part 2 of 2 · side by side';
  const L = x.chosenLeft ? x.it.chosen : x.it.bench, R = x.chosenLeft ? x.it.bench : x.it.chosen;
  const t0 = performance.now();
  $('main').innerHTML = `<p class="ask">Which reads more like <b>${x.it.ch}</b>?</p>
    <div class="pair"><figure><img class="shadow" src="${L}" alt="Left shadow"><figcaption>left <kbd>←</kbd></figcaption></figure>
    <figure><img class="shadow" src="${R}" alt="Right shadow"><figcaption>right <kbd>→</kbd></figcaption></figure></div>
    <div class="row"><button id="bl">Left</button><button class="ghost" id="bt">Can't tell <kbd>↓</kbd></button><button id="br">Right</button></div>`;
  const go = async side => {
    const pick = side === 'tie' ? 'tie' : ((side === 'left') === x.chosenLeft ? 'chosen' : 'bench');
    const rec = {side, pick, ch: x.it.ch, sub: x.it.sub, rep: x.rep, chosenLeft: x.chosenLeft, ms: Math.round(performance.now() - t0), n: p.dR + p.dA};
    state.ab[x.key] = rec;
    document.removeEventListener('keydown', onKey);
    const s = save();
    next();
    await s;
  };
  const onKey = e => { if (e.repeat) return; if (e.key === 'ArrowLeft') go('left'); else if (e.key === 'ArrowRight') go('right'); else if (e.key === 'ArrowDown') go('tie'); };
  document.addEventListener('keydown', onKey);
  $('bl').addEventListener('click', () => go('left'));
  $('br').addEventListener('click', () => go('right'));
  $('bt').addEventListener('click', () => go('tie'));
}

(async () => {
  const user = await window.claude?.use?.('user');
  db = await window.claude?.use?.('db');
  const uid = user ? await user.id() : null;
  if (!db || !uid) {
    $('phase').textContent = 'Sign in needed';
    $('main').innerHTML = '<p>This test saves each person\'s answers, so it needs you to be signed in to claude.ai with access to this page.</p>';
    return;
  }
  if ((await user.can?.('data.write')) === false) return stop('Cannot save', NO_ACCESS);
  ref = db.collection('tests').doc(uid);
  plan = makePlan(uid);
  let snap = null;
  for (let t = 0; t < 2 && !snap; t++) {
    try { snap = await ref.get(); } catch (e) { if (t === 0) await new Promise(res => setTimeout(res, 500 + 1000 * Math.random())); }
  }
  // a missing document is not an error, so a failed read is a real failure: starting fresh would ask
  // the same items again and overwrite the earlier answers
  if (!snap) return stop('Could not load', 'Your earlier answers could not be loaded. Reload the page to try again.');
  exists = snap.exists;
  if (snap.exists) { const d = snap.data(); state = {reads: {...(d.reads || {})}, ab: {...(d.ab || {})}}; }
  const {dR, dA} = progress();
  if (dR + dA === 0) {
    $('phase').textContent = 'Before you start';
    $('main').innerHTML = `<p>You will see shadows cast by three robot arms. Each is meant to be a letter or a digit.</p>
      <p><b>Part 1:</b> type the character you see (you are told whether it is a capital, lowercase or digit). Some characters come up twice. Answer with your first impression; use "Can't tell" if you have no idea.</p>
      <p><b>Part 2:</b> pick which of two shadows reads more like a given character.</p>
      <p class="muted">About 20 minutes. Your answers save as you go.</p>
      <button id="start">Start</button>`;
    $('start').addEventListener('click', next);
  } else next();
})();
</script>
"""

if __name__ == "__main__":
    main()
