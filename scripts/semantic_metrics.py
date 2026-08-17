"""Recognizability metrics: can anything -- a model or a person -- tell what it is.

Every metric in `metrics.py` asks whether the shadow matches the target's geometry.
None of them ask whether it reads as a horse. On this benchmark those come apart
badly: the highest-IoU targets are `hcircle` and `jar`, and `abstract` exists
precisely as a subset with no name to get right. A shadow-art system that maximises
geometric agreement and produces unreadable shadows has optimised the wrong thing,
and no amount of boundary or topology metric will say so.

Three protocols here, cheapest first. They are ordered as a ladder: run the cheap
ones on all 546, use them to choose the ~100 items worth paying humans for.

  1. clip_retrieval   -- N-way class retrieval with CLIP. Zero marginal cost.
  2. vlm_naming       -- open-ended "what is this?" from a VLM, then synonym match.
                         ~$0.001/image; the closest cheap proxy for human naming.
  3. human study      -- not code. `build_human_study()` emits the item manifest;
                         see METRICS.md Part C for the protocol.

Two design rules apply to all three and are easy to get wrong:

**Always score the target too.** CLIP is weak on 1-bit silhouettes -- they are far
outside its training distribution. Without the target as a ceiling, a low score is
unattributable: bad shadow, or blind judge? Report the ratio
`recognizability(shadow) / recognizability(target)`, and treat the `abstract` subset
as the null control -- its ceiling should sit near chance, and if it doesn't, the
class list is leaking.

**Render, don't feed masks.** A 1-bit mask is out of distribution for every model
here. `render_shadow()` produces a warm-wall render with a soft edge, which is both
closer to the training distribution and closer to what a human viewer sees.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
from PIL import Image

# --- rendering ----------------------------------------------------------------

def render_shadow(mask: np.ndarray, size: int = 384, blur_frac: float = 0.012,
                  wall=(232, 220, 200), shadow=(38, 32, 30)) -> Image.Image:
    """Turn a binary mask into a photo-like shadow-on-a-wall image.

    Feeding a raw 1-bit mask to CLIP or a VLM measures how well that model handles
    an out-of-distribution input as much as it measures the shadow. A soft-edged
    dark shape on a warm wall is both what the models were trained on and what a
    human rater in the study will actually be shown, which keeps the automatic
    metrics and the human ceiling on the same footing.
    """
    m = np.asarray(mask)
    m = (m > 127) if m.max() > 1 else (m > 0)
    m = cv2.resize(m.astype(np.uint8) * 255, (size, size), interpolation=cv2.INTER_AREA)
    k = int(blur_frac * size) | 1
    m = cv2.GaussianBlur(m, (k, k), 0).astype(np.float32) / 255.0
    img = (np.array(wall, np.float32)[None, None] * (1 - m[..., None])
           + np.array(shadow, np.float32)[None, None] * m[..., None])
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


# --- 1. CLIP ------------------------------------------------------------------

def clip_retrieval(images, class_names, true_idx, model_name="ViT-B-32",
                   pretrained="laion2b_s34b_b79k", prompt_tmpl="a shadow of a {}",
                   batch_size=32):
    """N-way retrieval accuracy over the benchmark's own class list.

    Deliberately *not* raw image-text cosine similarity. A bare CLIP score has no
    interpretable scale -- 0.24 is neither good nor bad -- and it drifts with prompt
    wording, so it cannot be compared across subsets. Ranking the true class against
    the benchmark's other classes gives top-1 / top-5 / MRR, which have a known
    chance level and are directly comparable to the human N-AFC task.

    Returns per-item rank plus aggregate top1/top5/MRR. Requires `open_clip_torch`.
    """
    import torch
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    with torch.no_grad():
        txt = tokenizer([prompt_tmpl.format(c) for c in class_names])
        tf = model.encode_text(txt)
        tf /= tf.norm(dim=-1, keepdim=True)

        ranks = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([preprocess(im) for im in images[i:i + batch_size]])
            f = model.encode_image(batch)
            f /= f.norm(dim=-1, keepdim=True)
            sims = (f @ tf.T).numpy()
            for j, s in enumerate(sims):
                order = np.argsort(-s)
                ranks.append(int(np.where(order == true_idx[i + j])[0][0]) + 1)

    ranks = np.array(ranks)
    return {
        "rank": ranks.tolist(),
        "top1": float((ranks == 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "chance_top1": 1.0 / len(class_names),
    }


def recognizability_ratio(shadow_result: dict, target_result: dict, key="top1") -> float:
    """shadow score / target score -- the number to actually report.

    An absolute CLIP accuracy of 0.31 on shadows is uninterpretable on its own,
    because nobody knows what CLIP scores on the *targets*. If the targets score
    0.34, the shadows are carrying nearly all the recognizability the silhouettes
    ever had and the optimizer is close to its ceiling. If the targets score 0.90,
    the same 0.31 means two thirds of the identity was lost in the solve.
    """
    t = target_result[key]
    return float(shadow_result[key] / t) if t > 0 else float("nan")


# --- 2. VLM -------------------------------------------------------------------

NAMING_PROMPT = (
    "This is a shadow cast on a wall. In two words or fewer, what object or "
    "character does it look like? If it does not resemble anything identifiable, "
    "answer exactly: nothing. Answer with the name only."
)

LEGIBILITY_PROMPT = (
    "Rate how clearly this shadow reads as a recognisable object, 1 to 5, where "
    "1 = an unidentifiable blob and 5 = instantly recognisable. Answer with the "
    "digit only."
)


def vlm_naming(image_paths, ask_fn, n_votes: int = 5) -> list[dict]:
    """Open-ended naming by a VLM, sampled `n_votes` times per image.

    Closer to the human task than CLIP is: CLIP is handed the candidate list and
    only has to rank it, whereas naming has to be produced from nothing, which is
    what the human study will measure. It also degrades gracefully on `abstract`
    -- a forced-choice model must pick something, while a VLM can answer "nothing",
    which is the correct answer there.

    Vote spread across the `n_votes` samples doubles as a confidence estimate: five
    agreeing answers and five different ones are very different results that a
    single sample cannot distinguish.

    `ask_fn(image_path, prompt) -> str` is supplied by the caller so this module
    stays free of any particular API client.
    """
    out = []
    for p in image_paths:
        votes = [ask_fn(p, NAMING_PROMPT).strip().lower() for _ in range(n_votes)]
        top = max(set(votes), key=votes.count)
        out.append({"path": p, "votes": votes, "modal_answer": top,
                    "agreement": votes.count(top) / len(votes)})
    return out


def match_answer(answer: str, class_name: str, synonyms: dict | None = None) -> bool:
    """Free-text answer vs ground-truth class, via a synonym table.

    Naming accuracy is only as good as its grading: "pony"/"horse" and "cup"/"mug"
    are correct, and exact string matching scores them wrong. Keep the synonym
    table in the repo and version it -- it is part of the metric definition, not an
    implementation detail, and results are not reproducible without it.
    """
    a = answer.strip().lower()
    c = class_name.strip().lower()
    if a in ("nothing", "", "unclear", "unknown"):
        return False
    pool = {c} | set((synonyms or {}).get(c, []))
    return any(a == s or a in s or s in a for s in pool)


# --- 3. human study manifest --------------------------------------------------

def build_human_study(items, out_dir: str, seed: int = 0,
                      include_targets: bool = True, n_foils: int = 3):
    """Write the rendered stimuli + a trial manifest for the human study.

    `items` = [{id, subset, class, target_path, shadow_path}]. Emits, per trial, the
    task type the subset calls for:

      semantic subsets -> `naming` (free response) and `nafc` (true class + foils)
      `abstract`       -> `match2afc` (this shadow, two candidate targets, which one)

    The split is not cosmetic. `abstract` shapes have no name, so a naming task is
    undefined there -- but it is the control group that separates geometric matching
    from recognizability, and dropping it would waste the design. Matching a shadow
    back to its target is well-defined without semantics and measures exactly the
    fidelity half.

    `include_targets` adds the targets themselves as trials. That upper-bound
    condition is what makes every shadow number interpretable, and it is the piece
    people skip.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "stimuli"), exist_ok=True)
    classes = sorted({it["class"] for it in items})
    trials = []

    def emit(it, kind, src, cond):
        name = f"{it['id']}__{cond}.png"
        render_shadow(np.array(Image.open(src).convert("L")) < 128).save(
            os.path.join(out_dir, "stimuli", name))
        t = {"trial_id": f"{it['id']}__{cond}", "stimulus": f"stimuli/{name}",
             "task": kind, "condition": cond, "subset": it["subset"],
             "true_class": it["class"], "item_id": it["id"]}
        if kind == "nafc":
            foils = [c for c in classes if c != it["class"]]
            opts = [it["class"]] + list(rng.choice(foils, n_foils, replace=False))
            rng.shuffle(opts)
            t["options"] = opts
        elif kind == "match2afc":
            others = [o for o in items if o["id"] != it["id"]]
            foil = others[int(rng.integers(len(others)))]
            opts = [it["target_path"], foil["target_path"]]
            rng.shuffle(opts)
            t["options"] = opts
            t["correct"] = it["target_path"]
        trials.append(t)

    for it in items:
        kinds = ["match2afc"] if it["subset"] == "abstract" else ["naming", "nafc"]
        for k in kinds:
            emit(it, k, it["shadow_path"], "shadow")
            if include_targets:
                emit(it, k, it["target_path"], "target")

    rng.shuffle(trials)
    with open(os.path.join(out_dir, "trials.json"), "w") as f:
        json.dump(trials, f, indent=1)
    return trials
