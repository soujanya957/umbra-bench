"""Train the Ivy reader (fastshadow/ivy_reader.py) on the picker page's data.

Cross-validated by character (5 folds, whole characters held out), so a score is
"how well does it judge characters it never saw". Writes fastshadow/ivy_reader.pt
(trained on everything) and results/fastshadow/ivy_reader/report.json.

    python scripts/train_ivy_reader.py <picks.json>
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastshadow  # noqa: E402

BENCH = fastshadow.BENCH
sys.path.insert(0, os.path.join(BENCH, "scripts"))
from run_base_optimizer import load_target  # noqa: E402

STUDY = {"cma": "objective_study", "adam": "objective_study_adam"}
OUT = os.path.join(BENCH, "results", "fastshadow", "ivy_reader")


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    r = np.arange(1, len(allv) + 1, dtype=float)
    while i < len(allv):  # average ranks for ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2
        i = j + 1
    ranks[order] = r
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def build(doc):
    import torch

    from fastshadow.ivy_reader import FeatureMaker
    from fastshadow.objectives import Target

    fm = FeatureMaker()
    root = os.path.join(BENCH, "optimized", "big-budget-grounded")
    items, feats = [], []
    for tk, d in STUDY.items():
        rows = defaultdict(dict)
        for l in open(os.path.join(BENCH, "results", "fastshadow", d, "graded.jsonl"), encoding="utf-8"):
            g = json.loads(l)
            if g["seed"] == 0:
                rows[g["id"]][g["objective"]] = g
        for gid, objs in rows.items():
            sub, stem = gid.split("/")
            T = load_target(os.path.join(root, sub, stem, f"{stem}_shown.png"), 128)
            t = Target(T, sub, stem.split("_")[0])
            names = sorted(objs)
            S = torch.tensor(np.stack([load_target(os.path.join(BENCH, "results", "fastshadow", d, objs[o]["mask"]), 128)
                                       for o in names]), device="cuda")
            with torch.no_grad():
                F = fm(S, t).cpu().numpy()
            for o, f in zip(names, F):
                items.append((tk, gid.replace("/", "__"), o))
                feats.append(f)
    index = {k: i for i, k in enumerate(items)}
    X = np.array(feats, dtype=np.float32)
    marks = []  # (item, y)
    for tk in STUDY:
        for g, v in doc.get(tk, {}).items():
            if not (v.get("good") or v.get("none")):
                continue
            good = set(v.get("good", []))
            for (tk2, g2, o), i in index.items():
                if tk2 == tk and g2 == g:
                    marks.append((i, int(o in good)))
    guesses = [(index[(c.split(":")[0], g, c.split(":")[1])], ok) for g, c, ans, ok, ms in doc.get("guesses", [])
               if (c.split(":")[0], g, c.split(":")[1]) in index]
    pairs = []  # (i_a, i_b, y=1 if a preferred); a = adam, b = cma
    for g, o, ch, side, ms in doc.get("ab", []):
        if ch in ("adam", "cma") and ("adam", g, o) in index and ("cma", g, o) in index:
            pairs.append((index[("adam", g, o)], index[("cma", g, o)], int(ch == "adam")))
    glyph_of = np.array([g for _, g, _ in items])
    return items, X, marks, guesses, pairs, glyph_of


def fit(X, marks, guesses, pairs, hidden, steps=1500, l2=1e-3, seed=0):
    import torch

    from fastshadow.ivy_reader import IvyReader

    torch.manual_seed(seed)
    m = IvyReader(hidden=hidden)
    used = sorted({i for i, _ in marks} | {i for i, _ in guesses} | {a for a, _, _ in pairs} | {b for _, b, _ in pairs})
    Xt = torch.tensor(X)
    m.mu.copy_(Xt[used].mean(0))
    m.sd.copy_(Xt[used].std(0).clamp_min(1e-4))
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    mi = torch.tensor([i for i, _ in marks]); my = torch.tensor([y for _, y in marks], dtype=torch.float32)
    gi = torch.tensor([i for i, _ in guesses]); gy = torch.tensor([y for _, y in guesses], dtype=torch.float32)
    pa = torch.tensor([a for a, _, _ in pairs]); pb = torch.tensor([b for _, b, _ in pairs])
    py = torch.tensor([y for _, _, y in pairs], dtype=torch.float32)
    for _ in range(steps):
        s = m.score(Xt)
        loss = 0
        if len(mi):
            loss = loss + bce(m.mark[0] * s[mi] + m.mark[1], my)
        if len(gi):
            loss = loss + bce(m.guess[0] * s[gi] + m.guess[1], gy)
        if len(pa):
            loss = loss + bce(s[pa] - s[pb], py)
        loss = loss + l2 * sum((p ** 2).sum() for n, p in m.net.named_parameters() if "weight" in n)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    return m


def evaluate(score, marks, guesses, pairs):
    out = {"mark_auc": auc([y for _, y in marks], [score[i] for i, _ in marks]) if marks else float("nan"),
           "guess_auc": auc([y for _, y in guesses], [score[i] for i, _ in guesses]) if guesses else float("nan")}
    if pairs:
        out["ab_acc"] = float(np.mean([(score[a] > score[b]) == bool(y) if score[a] != score[b] else 0.5 for a, b, y in pairs]))
    else:
        out["ab_acc"] = float("nan")
    return out


def main():
    import torch

    from fastshadow.ivy_reader import FEATURES, WEIGHTS

    doc = json.load(open(sys.argv[1], encoding="utf-8"))
    doc = doc.get("data", doc)
    os.makedirs(OUT, exist_ok=True)
    items, X, marks, guesses, pairs, glyph_of = build(doc)
    print(f"{len(items)} shadows, {len(marks)} marks, {len(guesses)} guesses, {len(pairs)} decided A/B pairs")
    glyphs = sorted(set(glyph_of))
    rng = np.random.default_rng(0)
    perm = rng.permutation(glyphs)
    folds = [set(perm[k::5]) for k in range(5)]

    def split(fold):
        te = lambda i: glyph_of[i] in fold  # noqa: E731
        return ([x for x in marks if not te(x[0])], [x for x in guesses if not te(x[0])], [x for x in pairs if not te(x[0])],
                [x for x in marks if te(x[0])], [x for x in guesses if te(x[0])], [x for x in pairs if te(x[0])])

    results = {}
    for name, hidden in (("linear", 0), ("mlp16", 16)):
        sc = np.full(len(items), np.nan)
        for fold in folds:
            trm, trg, trp, *_ = split(fold)
            m = fit(X, trm, trg, trp, hidden)
            with torch.no_grad():
                s = m.score(torch.tensor(X)).numpy()
            for i in range(len(items)):
                if glyph_of[i] in fold:
                    sc[i] = s[i]
        results[name] = evaluate(sc, marks, guesses, pairs)
        print(f"{name:8s} held-out characters: {results[name]}")
    base = {}
    for j, f in enumerate(FEATURES):
        sgn = 1 if np.corrcoef(X[[i for i, _ in marks], j], [y for _, y in marks])[0, 1] >= 0 else -1  # direction from all marks
        base[f] = evaluate(sgn * X[:, j], marks, guesses, pairs)
    for f, r in sorted(base.items(), key=lambda kv: -kv[1]["mark_auc"]):
        print(f"  baseline {f:10s} {r}")
    best = max(results, key=lambda k: results[k]["mark_auc"] + results[k]["guess_auc"] + results[k]["ab_acc"])
    hidden = 0 if best == "linear" else 16
    m = fit(X, marks, guesses, pairs, hidden)
    torch.save({"state": m.state_dict(), "hidden": hidden, "features": FEATURES}, WEIGHTS)
    # importance: drop in training mark AUC when a feature is shuffled
    Xt = torch.tensor(X)
    with torch.no_grad():
        s0 = m.score(Xt).numpy()
    a0 = evaluate(s0, marks, guesses, pairs)
    imp = {}
    for j, f in enumerate(FEATURES):
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        with torch.no_grad():
            sp = m.score(torch.tensor(Xp)).numpy()
        ap = evaluate(sp, marks, guesses, pairs)
        imp[f] = float((a0["mark_auc"] - ap["mark_auc"]) + (a0["guess_auc"] - ap["guess_auc"]))
    lin = None
    if hidden == 0:
        lin = {f: float(w) for f, w in zip(FEATURES, m.net.weight.detach()[0].numpy())}
    rep = {"n": {"shadows": len(items), "marks": len(marks), "guesses": len(guesses), "pairs": len(pairs), "glyphs": len(glyphs)},
           "cv": results, "chosen": best, "baselines": base, "importance": imp, "linear_weights": lin, "train_fit": a0}
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=2)
    print("chosen", best, "importance", {k: round(v, 3) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])})
    print("saved", WEIGHTS)


if __name__ == "__main__":
    main()
