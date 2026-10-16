"""SDS as a judge: does a text-to-image diffusion model's opinion of a shadow agree with Ivy's?

Score Distillation Sampling pushes a rendered image along eps(x_t, prompt) - eps(x_t, reference).
Before optimizing with it, check that the quantity it climbs ranks shadows the way a person
does. That quantity is the diffusion model's implicit classifier,

    score = E_{t,eps} ||eps - eps(x_t, ref)||^2 - ||eps - eps(x_t, glyph prompt)||^2

large when naming the character helps the model explain the image. Estimated with the same
fixed timesteps and noise for every shadow (common random numbers), so differences between
shadows are not sampling noise. Three variants: reference = a neutral shadow prompt
(sds_rel), the empty prompt (sds_rel_u), and the raw error with the glyph prompt (sds_abs).

A sanity check comes first: on the clean target glyphs, is the right character's prompt the
best explanation among the subset's characters? A model that cannot do that cannot judge a
shadow's legibility, and --sanity-only stops there.

Nothing is fitted on the human data, so the numbers compare directly with the Ivy judge's
cross-validated ones (scripts/train_ivy_reader.py) and with single metrics.

    python scripts/sds_score.py results/fastshadow/picks_v10/picks/<export>.json [--model sdxl]
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
from train_ivy_reader import STUDY, build, evaluate  # noqa: E402

MODELS = {"sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5", "sdxl": "stabilityai/stable-diffusion-xl-base-1.0"}
OUT = os.path.join(BENCH, "results", "fastshadow", "sds")
KIND = {"letters_upper": "capital letter", "letters_lower": "lowercase letter", "digits": "digit"}
NEUTRAL = "a black shadow on a white wall"


def glyph_prompt(sub, ch):
    return f'a black shadow of the {KIND[sub]} "{ch}" on a white wall'


class SDSScorer:
    def __init__(self, model="sd15", timesteps=(100, 200, 300, 400, 500, 600, 700, 800), n_noise=2, batch=16, seed=0):
        import torch
        from diffusers import DDPMScheduler

        self.kind, rid = model, MODELS[model]
        if model == "sd15":
            from diffusers import StableDiffusionPipeline as P
        else:
            from diffusers import StableDiffusionXLPipeline as P
        self.pipe = P.from_pretrained(rid, torch_dtype=torch.float16, variant="fp16", safety_checker=None) \
            if model == "sd15" else P.from_pretrained(rid, torch_dtype=torch.float16, variant="fp16")
        self.pipe.to("cuda")
        self.unet, self.vae = self.pipe.unet.eval(), self.pipe.vae.eval()
        if model == "sdxl":
            self.vae.to(torch.float32)  # the SDXL VAE overflows in fp16
        self.res = 512 if model == "sd15" else 1024
        self.abar = DDPMScheduler.from_pretrained(rid, subfolder="scheduler").alphas_cumprod.cuda()
        self.ts = torch.tensor(timesteps, device="cuda")
        g = torch.Generator(device="cpu").manual_seed(seed)
        # one fixed noise per (timestep, draw), shared by every image and prompt
        lat = self.res // 8
        self.noise = torch.randn(len(timesteps) * n_noise, 4, lat, lat, generator=g).cuda().half()
        self.t_all = self.ts.repeat_interleave(n_noise)
        self.batch = batch
        self._emb = {}

    def cond(self, prompt):
        """Text conditioning for the UNet, cached per prompt."""
        import torch

        if prompt not in self._emb:
            with torch.no_grad():
                if self.kind == "sd15":
                    pe, _ = self.pipe.encode_prompt(prompt, "cuda", 1, False)
                    self._emb[prompt] = (pe, None)
                else:
                    pe, _, pooled, _ = self.pipe.encode_prompt(prompt, device="cuda", num_images_per_prompt=1,
                                                               do_classifier_free_guidance=False)
                    tid = torch.tensor([[self.res, self.res, 0, 0, self.res, self.res]], device="cuda", dtype=pe.dtype)
                    self._emb[prompt] = (pe, {"text_embeds": pooled, "time_ids": tid})
        return self._emb[prompt]

    def latents(self, masks):
        """(N,128,128) bool shadow masks -> latents of the shadow as a photo: dark ink on a light wall."""
        import torch
        import torch.nn.functional as F

        m = torch.as_tensor(np.asarray(masks, np.float32), device="cuda")[:, None]
        m = F.interpolate(m, size=self.res, mode="bilinear", align_corners=False)
        img = (0.94 - 0.80 * m).repeat(1, 3, 1, 1) * 2 - 1  # wall 0.94, ink 0.14, to [-1,1]
        out = []
        with torch.no_grad():
            for s in range(0, len(img), 4):
                x = img[s:s + 4].to(self.vae.dtype)
                out.append((self.vae.encode(x).latent_dist.mean * self.vae.config.scaling_factor).half())
        return torch.cat(out)

    def errors(self, z, prompts):
        """Mean denoising error of each latent under each prompt: (N, len(prompts))."""
        import torch

        K = len(self.t_all)
        a = self.abar[self.t_all].view(K, 1, 1, 1).half()
        res = np.zeros((len(z), len(prompts)))
        for n in range(len(z)):
            zt = a.sqrt() * z[n:n + 1] + (1 - a).sqrt() * self.noise  # (K,4,h,w)
            for j, p in enumerate(prompts):
                pe, add = self.cond(p)
                err = []
                with torch.no_grad():
                    for s in range(0, K, self.batch):
                        b = len(zt[s:s + self.batch])
                        kw = {}
                        if add is not None:
                            kw["added_cond_kwargs"] = {k: v.expand(b, *v.shape[1:]) for k, v in add.items()}
                        pred = self.unet(zt[s:s + b], self.t_all[s:s + b], encoder_hidden_states=pe.expand(b, -1, -1),
                                         **kw).sample
                        err.append(((pred.float() - self.noise[s:s + b].float()) ** 2).mean((1, 2, 3)))
                res[n, j] = float(torch.cat(err).mean())
        return res


def mask_paths():
    """(tk, glyph id with __, objective) -> (mask file, graded row), the items train_ivy_reader.build sees."""
    out = {}
    for tk, d in STUDY.items():
        for l in open(os.path.join(BENCH, "results", "fastshadow", d, "graded.jsonl"), encoding="utf-8"):
            g = json.loads(l)
            if g["seed"] == 0:
                out[(tk, g["id"].replace("/", "__"), g["objective"])] = (
                    os.path.join(BENCH, "results", "fastshadow", d, g["mask"]), g)
    return out


def sanity(sc, fonts=3):
    """Clean target glyphs: rank of the right character's prompt among the subset's characters.
    `fonts` caps the target images per character (the benchmark has 3 fonts each)."""
    root = os.path.join(BENCH, "optimized", "big-budget-grounded")
    out = {}
    for sub in KIND:
        rows = [(stem.split("_")[0], load_target(os.path.join(root, sub, stem, f"{stem}_shown.png"), 128) > 0.5)
                for stem in sorted(os.listdir(os.path.join(root, sub)))]
        seen = defaultdict(int)
        rows = [r for r in rows if (seen.__setitem__(r[0], seen[r[0]] + 1) or seen[r[0]]) <= fonts]
        chars = sorted({c for c, _ in rows})
        E = sc.errors(sc.latents([m for _, m in rows]), [glyph_prompt(sub, c) for c in chars])
        ranks = [int((E[i] < E[i, chars.index(c)]).sum()) + 1 for i, (c, _) in enumerate(rows)]
        out[sub] = {"top1": float(np.mean([r == 1 for r in ranks])), "mean_rank": float(np.mean(ranks)),
                    "n_chars": len(chars), "chance_top1": 1 / len(chars), "chance_mean_rank": (len(chars) + 1) / 2,
                    "n_images": len(rows)}
        print(f"  {sub:14} top-1 {out[sub]['top1']:.2f} (chance {1 / len(chars):.2f})  mean rank "
              f"{out[sub]['mean_rank']:.1f} of {len(chars)} (chance {(len(chars) + 1) / 2:.1f})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picks")
    ap.add_argument("--model", choices=sorted(MODELS), default="sd15")
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--sanity-only", action="store_true")
    ap.add_argument("--n-noise", type=int, default=2, help="noise draws per timestep (8 timesteps)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fonts", type=int, default=3, help="sanity: target images per character")
    a = ap.parse_args()
    out = os.path.join(OUT, a.model)
    os.makedirs(out, exist_ok=True)
    sc = SDSScorer(a.model, n_noise=a.n_noise, batch=a.batch)
    rp = os.path.join(out, "report.json")
    report = json.load(open(rp)) if os.path.exists(rp) else {}
    report.update({"model": MODELS[a.model], "timesteps": sc.ts.tolist(), "n_noise": len(sc.t_all) // len(sc.ts),
                   "prompt": glyph_prompt("digits", "3"), "neutral": NEUTRAL})
    if not a.skip_sanity:
        print("sanity: clean targets, right prompt vs the subset's other characters", flush=True)
        report["sanity"] = sanity(sc, a.fonts)
        report["sanity_settings"] = {"fonts": a.fonts, "n_noise": a.n_noise}
        json.dump(report, open(rp, "w"), indent=2)
    if a.sanity_only:
        return

    doc = json.load(open(a.picks, encoding="utf-8"))
    items, X, marks, guesses, pairs, glyph_of = build(doc)
    paths = mask_paths()
    print(f"{len(items)} shadows, {len(marks)} marks, {len(guesses)} guesses, {len(pairs)} A/B pairs", flush=True)
    cache = os.path.join(out, "errors.json")
    done = json.load(open(cache)) if os.path.exists(cache) else {}
    t0 = time.perf_counter()
    todo = [it for it in items if "|".join(it) not in done]
    for k in range(0, len(todo), 16):
        chunk = todo[k:k + 16]
        z = sc.latents([load_target(paths[it][0], 128) > 0.5 for it in chunk])
        for n, it in enumerate(chunk):
            g = paths[it][1]
            E = sc.errors(z[n:n + 1], [glyph_prompt(g["subset"], g["label"]), NEUTRAL, ""])[0]
            done["|".join(it)] = E.tolist()
        json.dump(done, open(cache, "w"))
        print(f"  {k + len(chunk)}/{len(todo)} shadows, {time.perf_counter() - t0:.0f} s", flush=True)

    E = np.array([done["|".join(it)] for it in items])
    scores = {"sds_rel (vs neutral prompt)": E[:, 1] - E[:, 0], "sds_rel_u (vs empty prompt)": E[:, 2] - E[:, 0],
              "sds_abs (-error, glyph prompt)": -E[:, 0]}
    # marks are decided within a character: also try removing each character's offset
    for name in list(scores):
        s = scores[name].copy()
        for g in np.unique(glyph_of):
            s[glyph_of == g] -= s[glyph_of == g].mean()
        scores[name + ", per-glyph centred"] = s
    res = {k: evaluate(v, marks, guesses, pairs) for k, v in scores.items()}
    judge = json.load(open(os.path.join(BENCH, "results", "fastshadow", "ivy_reader", "report.json"), encoding="utf-8"))
    res["Ivy judge (held-out characters)"] = judge["cv"]["linear"]
    for f, v in judge["baselines"].items():
        res[f"single metric: {f}"] = v
    print(f"\n{'score':48} {'marks AUC':>9} {'read AUC':>9} {'A/B':>6}")
    for k, v in res.items():
        print(f"{k:48} {v['mark_auc']:9.3f} {v['guess_auc']:9.3f} {v['ab_acc']:6.3f}")
    from scipy.stats import spearmanr

    ij = np.array([paths[it][1].get("ivy_judge", np.nan) for it in items])
    iou = np.array([paths[it][1]["iou"] for it in items])
    s = scores["sds_rel (vs neutral prompt)"]
    corr = {"sds_rel vs ivy_judge": float(spearmanr(s, ij, nan_policy="omit")[0]),
            "sds_rel vs iou": float(spearmanr(s, iou)[0])}
    print("\nSpearman:", {k: round(v, 3) for k, v in corr.items()})
    report.update({"eval": res, "spearman": corr, "n_items": len(items)})
    json.dump(report, open(rp, "w"), indent=2)
    print("wrote", rp)


if __name__ == "__main__":
    main()
