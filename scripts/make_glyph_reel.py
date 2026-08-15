#!/usr/bin/env python3
"""Assemble specific glyphs from a sweep into a word strip or an alphabet GIF.

The benchmark solves every glyph in isolation, which answers "can the rig cast an R"
but not "can it spell". This picks named glyphs back out of a finished sweep and lines
them up, which is the form the result has to take to be shown to anyone.

Two modes:

  --word rO8OT     one strip, one shadow per character, left to right. A character's
                   case chooses its subset, so the spec is just the string you want:
                   lowercase -> letters_lower, uppercase -> letters_upper,
                   digit -> digits. That is what makes glyph substitution expressible
                   -- `8` in the b slot is a digit standing in for a letter, and the
                   spec says so on its face.

  --alphabet       a GIF running a, A, b, B, ... z, Z, so each letter is immediately
                   next to its own other case rather than 26 frames away.

Each glyph ships in several fonts. Unless --font pins one, the highest-IoU variant
wins, since a strip meant to be looked at should show the best the rig can do; the
chosen font is printed under every panel so the mix is never hidden.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from make_best_runs_gif import add_banner  # noqa: E402

SUBSET_FOR = {"lower": "letters_lower", "upper": "letters_upper", "digit": "digits"}


def subset_for(ch: str) -> str:
    """The subset a character lives in, decided by the character itself."""
    if ch.isdigit():
        return SUBSET_FOR["digit"]
    if ch.isupper():
        return SUBSET_FOR["upper"]
    if ch.islower():
        return SUBSET_FOR["lower"]
    raise SystemExit(f"[!] {ch!r} is not a letter or digit — no subset holds it")


def load_glyphs(root: str) -> dict:
    """Every solved glyph, keyed (subset, character) -> list of font variants."""
    out = {}
    for sub in SUBSET_FOR.values():
        sd = os.path.join(root, sub)
        if not os.path.isdir(sd):
            continue
        for stem in sorted(os.listdir(sd)):
            rj = os.path.join(sd, stem, "results.json")
            if not os.path.exists(rj):
                continue
            with open(rj) as f:
                r = json.load(f)
            # "A_dejavuserif-bold" -> character "A", font "dejavuserif-bold"
            ch, _, font = stem.partition("_")
            out.setdefault((sub, ch), []).append(
                {
                    "font": font,
                    "stem": stem,
                    "iou": r["best_iou"],
                    "shadow": os.path.join(sd, stem, stem + "_best.png"),
                    "target": os.path.join(_BENCH, r["target"]),
                }
            )
    return out


def pick(glyphs: dict, ch: str, font: str | None):
    sub = subset_for(ch)
    variants = glyphs.get((sub, ch))
    if not variants:
        raise SystemExit(f"[!] no solved glyph for {ch!r} in {sub} — sweep incomplete?")
    if font:
        hit = [v for v in variants if v["font"] == font]
        if not hit:
            have = ", ".join(sorted(v["font"] for v in variants))
            raise SystemExit(f"[!] {ch!r} has no font {font!r}; have: {have}")
        return hit[0]
    return max(variants, key=lambda v: v["iou"])


def load_mask(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return (np.array(img) < 128).astype(np.float32)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def word_strip(picks, word, label, out, size, dpi):
    """One row of shadows, read left to right as the word."""
    plt = _plt()
    n = len(picks)
    fig, axes = plt.subplots(1, n, figsize=(1.9 * n, 2.35), squeeze=False)
    for ax, ch, v in zip(axes[0], word, picks):
        ax.imshow(load_mask(v["shadow"], size), cmap="gray_r")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"‘{ch}’   IoU {v['iou']:.3f}", fontsize=8)
        ax.set_xlabel(v["font"], fontsize=6, color="0.45")
    mean = float(np.mean([v["iou"] for v in picks]))
    fig.suptitle(
        f"{label} — cast by 3× SO-101 arm shadows   "
        f"(glyphs {' '.join(word)}, mean IoU {mean:.3f})",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[word] {label}: {' '.join(word)}  mean IoU {mean:.3f} → {out}")


def glyph_frame(ch, v, size):
    """target | shadow | overlay for one glyph, as a PIL image."""
    plt = _plt()
    T, S = load_mask(v["target"], size), load_mask(v["shadow"], size)
    fig, ax = plt.subplots(1, 3, figsize=(5.4, 2.15))
    ax[0].imshow(T, cmap="gray_r")
    ax[0].set_title("target", fontsize=8)
    ax[1].imshow(S, cmap="gray_r")
    ax[1].set_title("robot silhouette", fontsize=8)
    ov = np.zeros((*T.shape, 3), np.float32)
    ov[..., 0], ov[..., 1] = T, S
    ax[2].imshow(1.0 - ov)
    ax[2].set_title("cyan missed, magenta spill", fontsize=7)
    for x in ax:
        x.set_xticks([])
        x.set_yticks([])
    fig.suptitle(f"‘{ch}’   IoU {v['iou']:.3f}   {v['font']}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    # buffer_rgba, not the tostring_rgb this used to call — that was dropped in
    # matplotlib 3.10, and the box runs a newer matplotlib than the Mac.
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="sweep dir, e.g. .../big-budget")
    p.add_argument("--out", default=None, help="default <results>/reels")
    p.add_argument("--word", default=None, help="glyph spec, e.g. rO8OT")
    p.add_argument("--label", default=None, help="word as written, e.g. ROBOT")
    p.add_argument("--alphabet", action="store_true", help="a, A, b, B, ... z, Z GIF")
    p.add_argument("--font", default=None, help="pin one font instead of best-IoU")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--seconds", type=float, default=1.0, help="GIF seconds per frame")
    a = p.parse_args()

    if not a.word and not a.alphabet:
        raise SystemExit("[!] nothing to do — pass --word and/or --alphabet")

    glyphs = load_glyphs(a.results)
    if not glyphs:
        raise SystemExit(f"[!] no solved glyphs under {a.results}")
    out = a.out or os.path.join(a.results, "reels")
    os.makedirs(out, exist_ok=True)

    if a.word:
        picks = [pick(glyphs, ch, a.font) for ch in a.word]
        label = a.label or a.word
        word_strip(
            picks, a.word, label,
            os.path.join(out, f"word_{label.lower()}.png"), a.size, a.dpi,
        )

    if a.alphabet:
        # Lowercase then its uppercase, so a letter's two cases are adjacent frames.
        order = [c for ch in "abcdefghijklmnopqrstuvwxyz" for c in (ch, ch.upper())]
        picks = [(c, pick(glyphs, c, a.font)) for c in order]
        frames = [glyph_frame(c, v, a.size) for c, v in picks]

        lines = [
            "UMBRA — every letter, lower case then upper",
            f"{len(frames)} glyphs · best-of-N IoU "
            f"{np.mean([v['iou'] for _, v in picks]):.3f} mean, "
            f"{min(v['iou'] for _, v in picks):.3f}–{max(v['iou'] for _, v in picks):.3f}",
            f"3× SO-101 · {'font ' + a.font if a.font else 'best font per letter'}"
            f" · {os.path.basename(os.path.normpath(a.results))}",
        ]
        banner = add_banner(Image.new("RGB", (frames[0].width, 1)), lines)
        stacked = []
        for f in frames:
            im = Image.new("RGB", (f.width, f.height + banner.height - 1))
            im.paste(banner, (0, 0))
            im.paste(f, (0, banner.height - 1))
            stacked.append(im)

        master = stacked[0].quantize(colors=64, method=Image.MEDIANCUT)
        q = [s.quantize(palette=master, dither=Image.NONE) for s in stacked]
        gp = os.path.join(out, "alphabet.gif")
        q[0].save(
            gp, save_all=True, append_images=q[1:],
            duration=int(round(a.seconds * 1000)), loop=0, optimize=True, disposal=2,
        )
        print(
            f"[alphabet] {len(q)} frames @ {a.seconds:g}s "
            f"({os.path.getsize(gp) / 1e6:.1f} MB) → {gp}"
        )
        worst = sorted(picks, key=lambda kv: kv[1]["iou"])[:5]
        print("  weakest: " + ", ".join(f"{c} {v['iou']:.3f}" for c, v in worst))


if __name__ == "__main__":
    main()
