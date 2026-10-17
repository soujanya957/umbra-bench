"""SDS done properly, steps 1-2 (protocol v2, after review: results/fastshadow/sds/protocol_review.json).

Question: can a fine-tuned Stable Diffusion 1.5 tell (a) which character a shadow is and (b) how
good it is, on characters whose shadows it never trained on? Only then is it worth using SDS
to steer the joints (step 3).

Training, per fold (the Ivy judge's 5 character folds, rng seed 0):
  - clean glyphs: 30 non-DejaVu system fonts x 62 characters, 'a printed {kind} <ch_X> on a
    white wall'. 62 new tokens <ch_X> (the tokenizer maps "A", "a" and the article "a" to one
    token) are the only text-encoder weights trained (textual-inversion style).
  - every marked shadow of the training characters: 'a clear robot-arm shadow of the {kind}
    <ch_X> on a white wall' if Ivy marked it good, 'an unreadable robot-arm shadow ...' if not;
    15% of captions dropped to the neutral one. 50/50 glyphs and shadows per batch.
  - UNet attention LoRA rank 8, lr 1e-4, 3000 steps, batch 4, min-SNR 5; small affine
    augmentation (scale 10%, rotation 5 deg, shift 8 px, no mirroring).
Scores on the held-out characters (fixed noise, t = 200..800 step 100, 4 draws, every
(t, draw) error stored):
  - quality: err(unreadable c) - err(clear c)                                 -> marks
  - reading: err(clear x) for every x of the subset (2 of the 4 draws), each prompt calibrated
    by its mean over the fold; margin = min_{x != c} err(x) - err(c)              -> reads
  - clean-letter test (fold-0 model; every fold sees the same fonts): DejaVu targets, all
    characters of the subset, per-prompt calibration, 'printed' caption.
Pre-registered GO for step 3 (all of them, on held-out characters):
  G0 reliability: within-character split-half Spearman of the quality score >= 0.8
  G1 within-character marks AUC of the quality score >= 0.72 (IoU: 0.67)
  G2 IoU + quality score (logistic, nested over folds) beats IoU alone by >= 0.03
     within-character marks AUC, character-bootstrap 95% interval above 0
  G3 reads AUC of the reading margin >= 0.75
  G4 clean top-1 >= 0.50 for letters and >= 0.70 for digits
The Ivy judge refit on the same folds is reported alongside.

    python scripts/sds_lora.py <picks.json>            (add --smoke for an API check)
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastshadow  # noqa: E402

BENCH = fastshadow.BENCH
sys.path.insert(0, os.path.join(BENCH, "scripts"))
from run_base_optimizer import load_target  # noqa: E402
from sds_score import KIND, MODELS, mask_paths  # noqa: E402
from train_ivy_reader import auc, build, fit  # noqa: E402

OUT = os.path.join(BENCH, "results", "fastshadow", "sds", "lora_v2")
ROOT = os.path.join(BENCH, "optimized", "big-budget-grounded")
TS = (200, 300, 400, 500, 600, 700, 800)
N_DRAW = 4
NEUTRAL_SHADOW = "a robot-arm shadow on a white wall"
NEUTRAL_PRINT = "a printed character on a white wall"


def tok(ch):
    return f"<ch_{ch}>" if ch.isdigit() or ch.isupper() else f"<ch_{ch}l>"  # tokenizer lower-cases; keep a/A apart


def cap_shadow(sub, ch, good):
    return f"{'a clear' if good else 'an unreadable'} robot-arm shadow of the {KIND[sub]} {tok(ch)} on a white wall"


def cap_print(sub, ch):
    return f"a printed {KIND[sub]} {tok(ch)} on a white wall"


def sub_of(ch):
    return "digits" if ch.isdigit() else ("letters_upper" if ch.isupper() else "letters_lower")


class Model:
    """SD 1.5 fp16 + new character tokens + UNet LoRA; errors at fixed (t, draw)."""

    def __init__(self, chars, seed=0):
        import torch
        from diffusers import DDPMScheduler, StableDiffusionPipeline

        rid = MODELS["sd15"]
        self.pipe = StableDiffusionPipeline.from_pretrained(rid, torch_dtype=torch.float16, variant="fp16",
                                                            safety_checker=None).to("cuda")
        self.unet, self.vae, self.te, self.tok = self.pipe.unet, self.pipe.vae, self.pipe.text_encoder, self.pipe.tokenizer
        self.te.to(torch.float32)
        n0 = len(self.tok)
        new = [tok(c) for c in chars]
        self.tok.add_tokens(new)
        self.te.resize_token_embeddings(len(self.tok))
        E = self.te.get_input_embeddings().weight
        with torch.no_grad():  # start each token at the embedding of its character
            for c, t in zip(chars, new):
                src = self.tok(c, add_special_tokens=False).input_ids[0]
                E[self.tok.convert_tokens_to_ids(t)] = E[src]
        self.n0 = n0
        self.abar = DDPMScheduler.from_pretrained(rid, subfolder="scheduler").alphas_cumprod.cuda()
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.ts = torch.tensor(TS, device="cuda").repeat_interleave(N_DRAW)  # (K,), t-major
        self.noise = torch.randn(len(self.ts), 4, 64, 64, generator=g).cuda().half()
        self._emb = {}

    def latents(self, masks, aug=None):
        import torch
        import torch.nn.functional as F

        m = torch.as_tensor(np.asarray(masks, np.float32), device="cuda")[:, None]
        if aug is not None:
            B = m.shape[0]
            g = aug
            r = lambda: torch.rand(B, generator=g, device="cuda")  # noqa: E731
            ang = (r() - 0.5) * np.deg2rad(10)
            sc = 1 + (r() - 0.5) * 0.2
            tx, ty = (r() - 0.5) * 16 / 64, (r() - 0.5) * 16 / 64  # +-8 px of 128 in [-1, 1] units
            A = torch.stack([torch.stack([torch.cos(ang) / sc, -torch.sin(ang) / sc, tx], 1),
                             torch.stack([torch.sin(ang) / sc, torch.cos(ang) / sc, ty], 1)], 1)
            grid = F.affine_grid(A, m.shape, align_corners=False)
            m = F.grid_sample(m, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        m = F.interpolate(m, size=512, mode="bilinear", align_corners=False)
        img = ((0.94 - 0.80 * m).repeat(1, 3, 1, 1) * 2 - 1).half()
        with torch.no_grad():
            return torch.cat([self.vae.encode(img[s:s + 8]).latent_dist.mean * self.vae.config.scaling_factor
                              for s in range(0, len(img), 8)])

    def encode(self, prompts):
        ids = self.tok(prompts, padding="max_length", max_length=self.tok.model_max_length, truncation=True,
                       return_tensors="pt").input_ids.cuda()
        return self.te(ids)[0]

    def cond(self, p):
        import torch

        if p not in self._emb:
            with torch.no_grad():
                self._emb[p] = self.encode([p]).half()
        return self._emb[p]

    def errors(self, z, prompts, draws=N_DRAW):
        """(N latents) x prompts -> (N, P, K) squared errors at the fixed (t, draw); `draws` < N_DRAW
        uses only the first draws of every timestep (cheaper, for the many-prompt tests)."""
        import torch

        sel = torch.tensor([ti * N_DRAW + d for ti in range(len(TS)) for d in range(draws)], device="cuda")
        ts, noise = self.ts[sel], self.noise[sel]
        K = len(ts)
        a = self.abar[ts].view(K, 1, 1, 1).half()
        out = np.zeros((len(z), len(prompts), K), np.float32)
        with torch.no_grad():
            for n in range(len(z)):
                zt = a.sqrt() * z[n:n + 1] + (1 - a).sqrt() * noise
                for j, p in enumerate(prompts):
                    pred = self.unet(zt, ts, encoder_hidden_states=self.cond(p).expand(K, -1, -1)).sample
                    out[n, j] = ((pred.float() - noise.float()) ** 2).mean((1, 2, 3)).cpu().numpy()
        return out

    def train(self, samples, steps, seed, log, lr=1e-4, lr_emb=5e-4, rank=8, snr_gamma=5.0):
        """samples: list of (mask 128x128 bool, caption, neutral caption, kind)."""
        import torch
        from peft import LoraConfig

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.te.requires_grad_(False)
        self.unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                         target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
        lora = [p for p in self.unet.parameters() if p.requires_grad]
        for p in lora:
            p.data = p.data.float()
        E = self.te.get_input_embeddings().weight
        E.requires_grad_(True)
        n0 = self.n0
        E.register_hook(lambda g: torch.cat([torch.zeros_like(g[:n0]), g[n0:]]))  # only the new tokens learn
        opt = torch.optim.AdamW([{"params": lora, "lr": lr, "weight_decay": 1e-2},
                                 {"params": [E], "lr": lr_emb, "weight_decay": 0.0}])
        scaler = torch.amp.GradScaler("cuda")
        g = torch.Generator(device="cuda").manual_seed(seed)
        gi = np.random.default_rng(seed)
        shadows = [s for s in samples if s[3] == "shadow"]
        prints = [s for s in samples if s[3] == "print"]
        snr = self.abar / (1 - self.abar)
        self.unet.train()
        t0 = time.perf_counter()
        for step in range(steps):
            batch = [shadows[i] for i in gi.integers(0, len(shadows), 2)] + [prints[i] for i in gi.integers(0, len(prints), 2)]
            caps = [s[2] if gi.random() < 0.15 else s[1] for s in batch]
            z = self.latents([s[0] for s in batch], aug=g).float()
            noise = torch.randn(z.shape, device="cuda", generator=g)
            t = torch.randint(0, 1000, (len(batch),), device="cuda", generator=g)
            a = self.abar[t].view(-1, 1, 1, 1)
            zt = a.sqrt() * z + (1 - a).sqrt() * noise
            with torch.autocast("cuda", dtype=torch.float16):
                ehs = self.encode(caps)
                pred = self.unet(zt.half(), t, encoder_hidden_states=ehs.half()).sample
            w = torch.clamp(snr[t], max=snr_gamma) / snr[t]  # min-SNR weighting (eps prediction)
            loss = (w * ((pred.float() - noise) ** 2).mean((1, 2, 3))).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(lora, 1.0)
            scaler.step(opt)
            scaler.update()
            if (step + 1) % 500 == 0 or step + 1 == steps:
                log.append({"step": step + 1, "loss": float(loss)})
                print(f"  step {step + 1}/{steps} loss {float(loss):.4f}  {time.perf_counter() - t0:.0f} s", flush=True)
        self.unet.eval()
        E.requires_grad_(False)
        self._emb = {}


def font_bank(n_fonts=30, seed=0):
    import torch
    import torch.nn.functional as F

    from fastshadow.glyph_cls import CLASSES, render_bank

    X, _ = render_bank(px=96)
    nf = len(X) // len(CLASSES)
    pick = np.random.default_rng(seed).choice(nf, min(n_fonts, nf), replace=False)
    X = X.reshape(nf, len(CLASSES), 96, 96)[pick]
    out = []
    for f in range(len(pick)):
        m = F.pad(torch.tensor(X[f], dtype=torch.float32), (16, 16, 16, 16))  # 96 -> 128, glyph about target size
        for c, ch in enumerate(CLASSES):
            out.append((m[c].numpy() > 0.5, ch))
    return out, nf


def logistic_fit(Z, y):
    """Plain L2-regularised logistic regression (standardised inputs) -> decision function."""
    from scipy.optimize import minimize

    Z, y = np.asarray(Z, float), np.asarray(y, float)
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    Zs = (Z - mu) / sd

    def f(w):
        s = Zs @ w[1:] + w[0]
        return np.mean(np.logaddexp(0, s) - y * s) + 1e-3 * np.sum(w[1:] ** 2)

    w = minimize(f, np.zeros(Z.shape[1] + 1), method="L-BFGS-B").x
    return lambda Q: ((np.asarray(Q, float) - mu) / sd) @ w[1:] + w[0]


def within_auc(score, marks, glyph_of):
    """Pooled AUC of per-character-centred scores, and the mean of per-character AUCs."""
    s = np.asarray(score, float).copy()
    for g in np.unique(glyph_of):
        sel = glyph_of == g
        if np.any(~np.isnan(s[sel])):
            s[sel] -= np.nanmean(s[sel])
    pooled = auc([y for _, y in marks], [s[i] for i, _ in marks])
    per = defaultdict(list)
    for i, y in marks:
        per[glyph_of[i]].append((y, score[i]))
    vals = [auc([y for y, _ in v], [x for _, x in v]) for v in per.values() if 0 < sum(y for y, _ in v) < len(v)]
    return float(pooled), float(np.nanmean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4], help="folds to run this time")
    ap.add_argument("--resume", action="store_true", help="keep the scores and fold records already in the output folder")
    a = ap.parse_args()
    import torch
    from scipy.stats import spearmanr

    from fastshadow.glyph_cls import CLASSES
    from fastshadow.ivy_reader import FEATURES

    od = OUT + ("_smoke" if a.smoke else "")
    os.makedirs(od, exist_ok=True)
    steps = 30 if a.smoke else a.steps
    doc = json.load(open(a.picks, encoding="utf-8"))
    items, X, marks, guesses, pairs, glyph_of = build(doc)
    paths = mask_paths()
    meta = {r["id"].replace("/", "__"): (r["subset"], r["label"]) for _, r in paths.values()}
    glyphs = sorted(set(glyph_of))
    perm = np.random.default_rng(0).permutation(glyphs)  # the judge's folds (train_ivy_reader.main)
    folds = [set(perm[k::5]) for k in range(5)]
    fold_of = np.array([next(k for k in range(5) if glyph_of[i] in folds[k]) for i in range(len(items))])
    marked = {i: y for i, y in marks}
    guessed = {i for i, _ in guesses}
    fonts, n_fonts_avail = font_bank()
    prints = [(m, cap_print(sub_of(ch), ch), NEUTRAL_PRINT, "print") for m, ch in fonts]
    print(f"{len(items)} shadows, {len(marks)} marks; {len(prints)} clean glyphs from 30 of {n_fonts_avail} fonts", flush=True)

    K = len(TS) * N_DRAW
    quality = np.full((len(items), K), np.nan, np.float32)  # err(unreadable) - err(clear) per (t, draw)
    margin = np.full(len(items), np.nan)
    judge = np.full(len(items), np.nan)
    fj = os.path.join(od, "folds.json")
    done = []
    if a.resume and os.path.exists(fj):
        quality = np.load(os.path.join(od, "quality.npy"))
        margin = np.load(os.path.join(od, "margin.npy"))
        done = json.load(open(fj))
        print("resuming with folds", [f["fold"] for f in done], flush=True)
    rep = {"protocol": "v2", "ts": TS, "n_draw": N_DRAW, "steps": steps, "folds": done}
    for k in ([0] if a.smoke else [k for k in a.folds if k not in {f["fold"] for f in done}]):
        print(f"\n== fold {k}: {len(folds[k])} held-out characters", flush=True)
        train_items = [i for i in marked if fold_of[i] != k]
        samples = prints + [(load_target(paths[items[i]][0], 128) > 0.5, cap_shadow(*meta[items[i][1]], marked[i]),
                             NEUTRAL_SHADOW, "shadow") for i in train_items]
        mdl = Model(CLASSES, seed=0)
        log = []
        mdl.train(samples, steps, seed=k, log=log)
        test = [i for i in range(len(items)) if fold_of[i] == k][: 24 if a.smoke else None]
        t0 = time.perf_counter()
        for s in range(0, len(test), 16):
            chunk = test[s:s + 16]
            z = mdl.latents([load_target(paths[items[i]][0], 128) > 0.5 for i in chunk])
            for n, i in enumerate(chunk):
                sub, ch = meta[items[i][1]]
                E = mdl.errors(z[n:n + 1], [cap_shadow(sub, ch, True), cap_shadow(sub, ch, False)])[0]
                quality[i] = E[1] - E[0]
            if s % 160 == 0:
                print(f"  quality {s + len(chunk)}/{len(test)}  {time.perf_counter() - t0:.0f} s", flush=True)
        gt = [i for i in test if i in guessed]
        bysub = defaultdict(list)
        for i in gt:
            bysub[meta[items[i][1]][0]].append(i)
        for sub, ids in bysub.items():
            chars = [c for c in CLASSES if sub_of(c) == sub]
            z = mdl.latents([load_target(paths[items[i]][0], 128) > 0.5 for i in ids])
            E = mdl.errors(z, [cap_shadow(sub, c, True) for c in chars], draws=2).mean(2)  # (n, chars)
            E = E - E.mean(0, keepdims=True)  # per-prompt calibration over this fold's shadows
            for n, i in enumerate(ids):
                c = chars.index(meta[items[i][1]][1])
                margin[i] = np.delete(E[n], c).min() - E[n, c]
        print(f"  reading margins for {len(gt)} guessed shadows  {time.perf_counter() - t0:.0f} s", flush=True)
        clean = None
        if k == 0:  # every fold trains on the same fonts; one clean test is enough
            clean = {}
            for sub in KIND:
                rows = [(stem.split("_")[0], load_target(os.path.join(ROOT, sub, stem, f"{stem}_shown.png"), 128) > 0.5)
                        for stem in sorted(os.listdir(os.path.join(ROOT, sub)))][: 4 if a.smoke else None]
                chars = [c for c in CLASSES if sub_of(c) == sub]
                E = mdl.errors(mdl.latents([m for _, m in rows]), [cap_print(sub, c) for c in chars], draws=2).mean(2)
                E = E - E.mean(0, keepdims=True)
                hits = sum(int(np.argmin(E[n]) == chars.index(c)) for n, (c, _) in enumerate(rows))
                clean[sub] = {"hits": hits, "n": len(rows), "chance": len(rows) / len(chars)}
                print(f"  clean {sub:14} top-1 {hits}/{len(rows)} (chance {len(rows) / len(chars):.1f})", flush=True)
        torch.save({"lora": {n: p.detach().cpu() for n, p in mdl.unet.named_parameters() if "lora" in n},
                    "tokens": mdl.te.get_input_embeddings().weight[mdl.n0:].detach().cpu()}, os.path.join(od, f"fold{k}.pt"))
        rep["folds"].append({"fold": k, "n_train_shadows": len(train_items), "loss_log": log, "clean": clean})
        del mdl
        torch.cuda.empty_cache()
        np.save(os.path.join(od, "quality.npy"), quality)
        np.save(os.path.join(od, "margin.npy"), margin)
        json.dump(sorted(rep["folds"], key=lambda f: f["fold"]), open(fj, "w"), indent=1)
    if not a.smoke and len({f["fold"] for f in rep["folds"]}) < 5:
        print("folds done:", sorted(f["fold"] for f in rep["folds"]), "- final report after all 5", flush=True)
        return
    rep["folds"] = sorted(rep["folds"], key=lambda f: f["fold"])
    for k in range(5):  # the Ivy judge refit without each fold's characters (CPU, seconds)
        m = fit(X, [x for x in marks if fold_of[x[0]] != k], [x for x in guesses if fold_of[x[0]] != k],
                [p for p in pairs if fold_of[p[0]] != k], 0, FEATURES)
        with torch.no_grad():
            sj = m.score(torch.tensor(X)).numpy()
        judge[fold_of == k] = sj[fold_of == k]

    ok = ~np.isnan(quality[:, 0])
    q = np.where(ok, np.nanmean(np.where(np.isnan(quality), 0, quality), 1), np.nan)
    mk = [x for x in marks if ok[x[0]]]
    gs = [x for x in guesses if ok[x[0]] and not np.isnan(margin[x[0]])]
    iou = X[:, FEATURES.index("iou")].astype(float)
    Q3 = quality.reshape(len(items), len(TS), N_DRAW)
    qa, qb = Q3[:, :, :2].mean((1, 2)), Q3[:, :, 2:].mean((1, 2))

    def cen(v):
        v = v.copy()
        for g in np.unique(glyph_of):
            sel = (glyph_of == g) & ok
            if sel.any():
                v[sel] -= v[sel].mean()
        return v

    rel = float(spearmanr(cen(qa)[ok], cen(qb)[ok])[0])
    res = {name: {"marks_within": within_auc(s, mk, glyph_of), "marks_pooled": auc([y for _, y in mk], [s[i] for i, _ in mk])}
           for name, s in (("SDS quality", q), ("IoU", iou), ("Ivy judge (same folds)", judge))}
    res["SDS reading margin"] = {"reads_auc": auc([y for _, y in gs], [margin[i] for i, _ in gs]) if gs else float("nan")}
    for name, s in (("IoU", iou), ("Ivy judge (same folds)", judge)):
        res[name]["reads_auc"] = auc([y for _, y in gs], [s[i] for i, _ in gs]) if gs else float("nan")
    comb = np.full(len(items), np.nan)
    Z = np.c_[iou, np.nan_to_num(q)]
    for k in range(5):
        tr = [i for i, _ in mk if fold_of[i] != k]
        te = [i for i in range(len(items)) if fold_of[i] == k and ok[i]]
        if len({marked[i] for i in tr}) < 2 or not te:
            continue
        comb[te] = logistic_fit(Z[tr], [marked[i] for i in tr])(Z[te])
    have = ~np.isnan(comb)
    mkc = [x for x in mk if have[x[0]]]
    gain = within_auc(comb, mkc, glyph_of)[0] - within_auc(iou, mkc, glyph_of)[0] if mkc else float("nan")
    rng = np.random.default_rng(0)
    gl = sorted({glyph_of[i] for i, _ in mkc})
    boots = []
    for _ in range(0 if a.smoke else 1000):
        pick = list(rng.choice(gl, len(gl)))
        sub = [x for g in pick for x in mkc if glyph_of[x[0]] == g]
        boots.append(within_auc(comb, sub, glyph_of)[0] - within_auc(iou, sub, glyph_of)[0])
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
    cl = rep["folds"][0]["clean"]
    letters = (cl["letters_upper"]["hits"] + cl["letters_lower"]["hits"]) / (cl["letters_upper"]["n"] + cl["letters_lower"]["n"])
    digits = cl["digits"]["hits"] / cl["digits"]["n"]
    gates = {"G0 reliability >= 0.8": bool(rel >= 0.8),
             "G1 within-character marks AUC >= 0.72": bool(res["SDS quality"]["marks_within"][0] >= 0.72),
             "G2 IoU+SDS gain >= 0.03, CI > 0": bool(gain >= 0.03 and lo > 0),
             "G3 reads AUC >= 0.75": bool(res["SDS reading margin"]["reads_auc"] >= 0.75),
             "G4 clean top-1 letters >= 0.5, digits >= 0.7": bool(letters >= 0.5 and digits >= 0.7)}
    rep.update({"reliability": rel, "eval": res, "g2_gain": gain, "g2_ci": [float(lo), float(hi)],
                "clean_top1": {"letters": letters, "digits": digits}, "gates": gates, "go": all(gates.values())})
    json.dump(rep, open(os.path.join(od, "report.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in rep.items() if k != "folds"}, indent=1, default=float))
    print("STEP 3:", "GO" if rep["go"] else "STOP")


if __name__ == "__main__":
    main()
