#!/usr/bin/env python3
"""09_clip_score.py — does the cast shadow read as the letter?

    python 09_clip_score.py
    python 09_clip_score.py --sequence demo_01_scene_06_A --source optimizer_fit022

IoU is the wrong question for a demo video. It measures overlap with a
silhouette the clip fit was free to move; whether a viewer reads the shape as an
F is a different question, and the review sheet shows the two disagreeing --
scene_05_M scores 0.681 and reads as a blob, scene_06_Y scores 0.747 and is a
clean Y in frame 1 that has collapsed to a T by frame 3.

So ask the legibility question with the instrument the benchmark already uses:
retrieval rank over the benchmark's own class set, as top-1 and MRR. Not raw
image-text cosine -- 0.24 is neither good nor bad and it moves with prompt
wording. Rank has a known chance level, printed beside it.

Three things are borrowed rather than reinvented, because each has a reason
recorded elsewhere in the repo:

* `tests/clip_eval.py`'s class set and alias map. I is scored as "1", since
  I/l/1 are the same picture; a thin vertical bar read as "l" is an I read
  correctly, not a miss. Inventing a 26-letter list would score it as a miss.
* The authored frames, scored through the identical path in the same run, as
  the ceiling. CLIP is half-blind to 1-bit silhouettes, so a low absolute score
  cannot separate "the shadow is poor" from "the judge cannot see either", and
  only the pair does. That is `recognizability_ratio`, called not rewritten.
* Content-cropping before scoring. A reassembled frame is a 1920x1080 canvas
  holding a fifty-pixel letter; handed to CLIP whole it is a picture of white.

Per frame, not only per clip. A clip that starts legible and degrades is a
temporal failure, which is the whole reason this track exists, and a clip mean
hides it exactly.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from semantic_metrics import clip_retrieval, recognizability_ratio  # noqa: E402
from clip_eval import ALL_LABELS, canonicalize_label, glyph_prompt  # noqa: E402


def framed(a: np.ndarray, size: int = 224, margin: float = 0.18) -> Image.Image:
    """Content-cropped, squared, on white -- what a viewer would be shown."""
    ys, xs = np.where(a)
    if not len(ys):
        return Image.new("RGB", (size, size), "white")
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0 + 1, x1 - x0 + 1
    side = max(h, w) + 2 * int(margin * max(h, w))
    tile = np.zeros((side, side), bool)
    oy, ox = (side - h) // 2, (side - w) // 2
    tile[oy:oy + h, ox:ox + w] = a[y0:y1 + 1, x0:x1 + 1]
    return (Image.fromarray(np.where(tile, 0, 255).astype(np.uint8), "L")
            .resize((size, size), Image.LANCZOS).convert("RGB"))


def masks(d: Path):
    return [np.asarray(Image.open(p).convert("L")) < 128
            for p in sorted(d.glob("f*.png"))]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reassembled", default=str(ROOT / "results" / "demo_reassembled"))
    ap.add_argument("--sequence", action="append", default=[])
    ap.add_argument("--source", default="optimizer",
                    help="which solve pass these shadows came from; carried "
                         "into the CSV so two passes can sit side by side")
    ap.add_argument("--out", default=str(ROOT / "results" / "demo_clip_legibility.csv"))
    ap.add_argument("--model-name", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    re_root = Path(a.reassembled)
    names = a.sequence or sorted(
        p.name for p in re_root.iterdir()
        if p.is_dir() and (ROOT / "sequences" / p.name).is_dir())

    imgs, meta = [], []
    for n in names:
        raw = n.split("_")[-1]
        cls = canonicalize_label(raw)
        if cls is None or cls not in ALL_LABELS:
            print(f"  {n}: {raw!r} is not a scoring class, skipped")
            continue
        ti = ALL_LABELS.index(cls)
        for cond, src in (("reassembled", re_root / n),
                          ("authored", ROOT / "sequences" / n)):
            for k, f in enumerate(masks(src)):
                imgs.append(framed(f))
                meta.append({"sequence_id": n, "letter": raw, "cls": cls,
                             "true_idx": ti, "input": cond, "frame_idx": k})
    if not imgs:
        raise SystemExit("no scorable frames")

    nc = len(ALL_LABELS)
    chance = 1.0 / nc
    print(f"{len(imgs)} frames over {len(names)} clips, {nc}-way retrieval "
          f"(chance top-1 {chance:.4f})")
    # glyph_prompt, not the default "a shadow of a {}". A bare class name makes
    # the prompt "a shadow of a A", which does not say the label is a letter and
    # cannot separate letter O from digit 0 -- with it, a clean authored A
    # scored top-1 0.000 while an illegible L fragment scored 1.000, i.e. the
    # ranking was noise. glyph_prompt states case and letter-vs-digit.
    res = clip_retrieval(imgs, ALL_LABELS, [m["true_idx"] for m in meta],
                         model_name=a.model_name, pretrained=a.pretrained,
                         prompt_tmpl=glyph_prompt,
                         batch_size=a.batch_size, top_k=5)
    for m, r in zip(meta, res["rank"]):
        m["clip_rank"] = int(r)
        m["clip_rr"] = round(1.0 / int(r), 4)
        m["clip_top1"] = int(int(r) == 1)

    def agg(rows):
        if not rows:
            return None
        rk = np.array([r["clip_rank"] for r in rows], float)
        return {"n": len(rows), "top1": float((rk == 1).mean()),
                "mrr": float((1 / rk).mean()), "rank": float(rk.mean())}

    rows, unscoreable = [], []
    print(f"\n{'clip':<24}{'as':>4}{'top1':>8}{'mrr':>7}{'rank':>7}"
          f" |{'ceiling top1':>13}{'mrr':>7} |{'ratio1':>8}{'ratioM':>8}")
    print("-" * 96)
    for n in names:
        sel = [m for m in meta if m["sequence_id"] == n]
        c = agg([m for m in sel if m["input"] == "reassembled"])
        t = agg([m for m in sel if m["input"] == "authored"])
        if not c or not t:
            continue
        r1 = recognizability_ratio(c, t, "top1")
        rm = recognizability_ratio(c, t, "mrr")
        # A zero ceiling is not a bad shadow, it is an unusable measurement:
        # CLIP could not read the authored frame either, so there is nothing to
        # be a fraction of. Printing nan invites it to be read as 0.
        blind = t["top1"] == 0.0
        if blind:
            unscoreable.append(n)
        print(f"{n:<24}{sel[0]['cls']:>4}{c['top1']:>8.3f}{c['mrr']:>7.3f}"
              f"{c['rank']:>7.1f} |{t['top1']:>13.3f}{t['mrr']:>7.3f}"
              f" |{'    --  ' if blind else f'{r1:>8.3f}'}{rm:>8.3f}")
        for m in sel:                                        # one row per frame
            rows.append({"sequence_id": n, "frame_idx": m["frame_idx"],
                         "source": a.source if m["input"] == "reassembled" else "",
                         "input": m["input"], "clip_rank": m["clip_rank"],
                         "clip_rr": m["clip_rr"], "clip_top1": m["clip_top1"],
                         "n_classes": nc, "chance_top1": round(chance, 4)})
        for cond, g in (("reassembled", c), ("authored", t)):   # clip aggregate
            rows.append({"sequence_id": n, "frame_idx": "",
                         "source": a.source if cond == "reassembled" else "",
                         "input": cond, "clip_rank": round(g["rank"], 3),
                         "clip_rr": round(g["mrr"], 4),
                         "clip_top1": round(g["top1"], 4),
                         "n_classes": nc, "chance_top1": round(chance, 4)})

    ca = agg([m for m in meta if m["input"] == "reassembled"])
    ta = agg([m for m in meta if m["input"] == "authored"])
    print("-" * 96)
    print(f"{'all frames':<24}{'':>4}{ca['top1']:>8.3f}{ca['mrr']:>7.3f}{ca['rank']:>7.1f}"
          f" |{ta['top1']:>13.3f}{ta['mrr']:>7.3f}"
          f" |{recognizability_ratio(ca, ta, 'top1'):>8.3f}"
          f"{recognizability_ratio(ca, ta, 'mrr'):>8.3f}")
    print(f"\nchance top-1 {chance:.4f} over {nc} classes. 'ceiling' is the authored")
    print("frame through the identical path: CLIP is half-blind to 1-bit")
    print("silhouettes, so the ratio is the quotable number, not the raw score.")
    if unscoreable:
        print(f"\n{len(unscoreable)} clip(s) have a zero ceiling and so no top-1 "
              f"ratio: {', '.join(unscoreable)}.")
        print("The cause is the class design, not the solve. tests/clip_eval.py folds")
        print("I/l/1 into one class because they are the same picture, and glyph_prompt")
        print("then asks for 'the digit 1'. A serif capital I from a title card is not a")
        print("digit 1 to CLIP, so the authored frame misses too and the ratio has no")
        print("denominator. Their MRR ratios near 1.0 say shadow and target are equally")
        print("unreadable here, not that the shadow is good.")

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out}  ({len(rows)} rows: per frame, plus one aggregate "
          f"per clip and condition with frame_idx blank)")


if __name__ == "__main__":
    main()
