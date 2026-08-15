#!/usr/bin/env python3
"""Spell a phrase out of the sweep's glyph solves, as two animated GIFs.

The benchmark solves each glyph in isolation, which is the right unit for scoring and
the wrong unit for showing anyone what the fleet can do. A phrase is the demo: the same
three arms, re-posed once per character, casting `ICrA 2027` on a wall. This assembles
that from solves that already exist — nothing is re-optimized, so it is instant to
re-run and stays honest about what the sweep actually achieved.

Two GIFs, because they answer different questions:

    <phrase>-shadows.gif    what the wall sees — the cast shadow, one character at a
                            time, with the full phrase as a filmstrip underneath so
                            the word stays readable while a single glyph is on screen
    <phrase>-overlay.gif    how close it got — target | robot shadow | overlay, the
                            same triptych and colour convention as the best-runs sheets

One font family for the whole phrase by default. Picking each character's best-scoring
variant would flatter the mean IoU and produce a phrase in three different typefaces,
which is a worse demo and a dishonest one; `--best-font` is there when the number
matters more than the look.

The budget line is read back out of the sweep's own `results.json` rather than typed,
for the reason `make_best_runs_gif.py` gives: a GIF outlives the folder it came from.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)

sys.path.insert(0, _HERE)
from make_best_runs_gif import _font, budget_caption  # noqa: E402

BG = (24, 24, 28)
PANEL_BG = (255, 255, 255)
INK = (17, 17, 20)
ACCENT = (168, 214, 255)
MUTED = (150, 150, 160)

# Same convention as the best-runs sheets: cyan = target the shadow missed,
# magenta = shadow that spilled outside it, blue = agreement.
OV_HIT = (0, 0, 255)
OV_MISS = (0, 255, 255)
OV_SPILL = (255, 0, 255)


def subset_for(ch: str) -> str:
    if ch.isdigit():
        return "digits"
    if ch.isupper():
        return "letters_upper"
    if ch.islower():
        return "letters_lower"
    raise SystemExit(f"[!] no benchmark subset holds {ch!r} — letters and digits only")


def glyph_solves(results: str, ch: str) -> list[tuple[str, dict]]:
    """Every finished solve for one character in one sweep, as (stem, results.json)."""
    d = os.path.join(results, subset_for(ch))
    if not os.path.isdir(d):
        return []
    out = []
    for stem in sorted(s for s in os.listdir(d) if s.split("_", 1)[0] == ch):
        rj = os.path.join(d, stem, "results.json")
        if os.path.exists(rj):
            with open(rj) as f:
                out.append((stem, json.load(f)))
    return out


def covers(results: str, phrase: str, font: str, best_font: bool) -> bool:
    """Whether a sweep can spell the phrase in one typeface.

    A sweep that has solved five of six letters cannot spell the word, and one that has
    the sixth only in a different face can spell it only by mixing typefaces — which is
    what `--best-font` is for and not what a default should silently do.
    """
    for ch in set(phrase.replace(" ", "")):
        solves = glyph_solves(results, ch)
        if not solves or (not best_font and not any(s == f"{ch}_{font}" for s, _ in solves)):
            return False
    return True


def choose_results(prefer: list[str], phrase: str, font: str, best_font: bool) -> str:
    """The first sweep in `prefer` that can spell the phrase.

    The sweeps fill in over days, so which one can spell a given word is a fact about
    this morning, not about the phrase. Asking each in preference order means the same
    command picks up the wider budget the day it finishes that letter, with no edit.
    """
    for results in prefer:
        if not os.path.isdir(results):
            continue
        if covers(results, phrase, font, best_font):
            return results
        print(f"[skip] {os.path.basename(results)} cannot spell “{phrase}” in {font} yet")
    raise SystemExit(f"[!] no sweep in {prefer} has every glyph of “{phrase}”")


def find_glyph(results: str, ch: str, font: str, best_font: bool) -> dict:
    """Locate one character's solve, preferring `font` unless asked for the best.

    Falls back to whatever variant the sweep does have: an explicit `--results` on a
    half-finished sweep should cost a warning, not the whole GIF.
    """
    sub = subset_for(ch)
    d = os.path.join(results, sub)
    cands = glyph_solves(results, ch)
    if not cands:
        raise SystemExit(f"[!] {results} has no solve for {ch!r} ({sub}/{ch}_*)")

    want = f"{ch}_{font}"
    exact = [c for c in cands if c[0] == want]
    if best_font or not exact:
        if not best_font:
            print(f"[warn] {want} not solved here — falling back to best available")
        stem, r = max(cands, key=lambda c: c[1]["best_iou"])
    else:
        stem, r = exact[0]

    return {
        "char": ch,
        "subset": sub,
        "stem": stem,
        "iou": r["best_iou"],
        "shadow": os.path.join(d, stem, f"{stem}_best.png"),
        "target": os.path.join(_BENCH, r["target"]),
    }


def load_mask(path: str, size: int) -> np.ndarray:
    """1-bit PNGs, dark = shape. NEAREST so a 128 px shadow enlarges honestly.

    A smooth filter would invent grey where the renderer only ever produced lit or
    unlit pixels, softening exactly the ragged edges this is meant to show.
    """
    img = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return np.array(img) < 128


def panel(mask: np.ndarray, fg=INK, bg=PANEL_BG) -> Image.Image:
    rgb = np.where(mask[..., None], np.array(fg, np.uint8), np.array(bg, np.uint8))
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def overlay_panel(t: np.ndarray, s: np.ndarray) -> Image.Image:
    rgb = np.full((*t.shape, 3), 255, np.uint8)
    rgb[t & s] = OV_HIT
    rgb[t & ~s] = OV_MISS
    rgb[~t & s] = OV_SPILL
    return Image.fromarray(rgb, "RGB")


def dim(img: Image.Image, amount: float) -> Image.Image:
    """Blend a panel toward the page background — used to push the filmstrip back."""
    return Image.blend(img, Image.new("RGB", img.size, BG), amount)


def framed(img: Image.Image, color, width: int) -> Image.Image:
    out = Image.new("RGB", (img.width + 2 * width, img.height + 2 * width), color)
    out.paste(img, (width, width))
    return out


def caption_lines(results: str, phrase: str, glyphs: list[dict], font: str, best_font: bool):
    mean = sum(g["iou"] for g in glyphs) / len(glyphs)
    face = "best-scoring variant per glyph" if best_font else font
    return [
        f"UMBRA — “{phrase}” cast by 3× SO-101",
        budget_caption(results)[1],
        f"{len(glyphs)} glyphs · {face} · mean best-of-N IoU {mean:.3f}",
    ]


def banner(width: int, lines: list[str]) -> Image.Image:
    """The header strip, rendered once and pasted onto every frame.

    Type is sized off the frame width so `--size` does not change the proportions.
    """
    size = max(9, round(width * 0.018))
    pad = max(4, size // 2)
    fonts = [_font(round(size * 1.3)), _font(size), _font(size)]
    heights = [round(f.size * 1.5) for f in fonts]
    img = Image.new("RGB", (width, sum(heights) + 2 * pad), BG)
    d = ImageDraw.Draw(img)
    y = pad
    for text, f, h, color in zip(lines, fonts, heights, [(255, 255, 255), ACCENT, MUTED]):
        d.text((pad * 2, y), text, font=f, fill=color)
        y += h
    return img


def filmstrip(items: list, panels: dict, cell: int, current: int | None = None) -> Image.Image:
    """One row of glyphs at a given cell size, optionally with one of them lit.

    `items` is the phrase as panel keys with `None` for each space, so a caller can hand
    over the whole phrase or a single word without the renderer knowing the difference.
    Panels are resized here rather than by the caller, so one cached set of renders
    serves the filmstrip and the enlarged block both.

    Keeping the word on screen while a single glyph is blown up is the point: a viewer
    who joins the loop midway can still read what is being spelled.
    """
    border = max(2, cell // 32)
    gap = max(4, cell // 4)
    step = cell + 2 * border + gap
    space = step // 2
    width = sum(space if it is None else step for it in items) - gap
    img = Image.new("RGB", (width, cell + 2 * border), BG)
    x = 0
    for it in items:
        if it is None:
            x += space
            continue
        t = panels[it].resize((cell, cell), Image.NEAREST)
        # Dimming exists to make one cell stand out; with nothing to stand out from, it
        # would only mute the row — and on the overlay panels it would mute the colours
        # the overlay is entirely about.
        lit = current is None or current == it
        img.paste(framed(t if lit else dim(t, 0.5), ACCENT if current == it else BG, border), (x, 0))
        x += step
    return img


def phrase_block(items: list, panels: dict, max_w: int, max_h: int) -> Image.Image:
    """The phrase set as large as the space allows, one row per word.

    A nine-cell row inside a slide-width frame leaves every glyph barely larger than the
    filmstrip it is supposed to be the payoff for. Breaking at the spaces — where the
    phrase already wants to break — roughly doubles the cell.

    The cell size is found by trying sizes rather than solved for: inverting the layout
    would mean restating the border and gap rules and keeping the two copies in step
    forever, while walking down from the ceiling cannot disagree with the renderer.
    """
    rows, cur = [], []
    for it in items:
        if it is None:
            rows.append(cur) if cur else None
            cur = []
        else:
            cur.append(it)
    if cur:
        rows.append(cur)

    for cell in range(max_h, 15, -2):
        imgs = [filmstrip(r, panels, cell) for r in rows]
        gap = max(6, cell // 6)
        w = max(i.width for i in imgs)
        h = sum(i.height for i in imgs) + gap * (len(imgs) - 1)
        if w <= max_w and h <= max_h:
            out = Image.new("RGB", (w, h), BG)
            y = 0
            for i in imgs:
                out.paste(i, ((w - i.width) // 2, y))
                y += i.height + gap
            return out
    return filmstrip(items, panels, 16)


def compose(size: tuple[int, int], head: Image.Image, body: list[Image.Image],
            strip: Image.Image, caption: str, pad: int) -> Image.Image:
    """Stack banner / panel row / caption / filmstrip on a canvas of a fixed size.

    Every frame is composed at the same size, and the banner, caption and filmstrip sit
    at the same offsets in all of them: a GIF frame that changes height gets cropped to
    the first one, and anything that drifts by a few pixels between frames reads as a
    flicker in a loop.
    """
    width, height = size
    f = _font(max(11, round(width * 0.017)))
    line = round(f.size * 1.6)
    img = Image.new("RGB", size, BG)
    img.paste(head, (0, 0))

    strip_y = height - strip.height - pad
    cap_y = strip_y - line
    img.paste(strip, ((width - strip.width) // 2, strip_y))
    ImageDraw.Draw(img).text((width // 2, cap_y), caption, font=f, fill=MUTED, anchor="ma")

    row_h = max(b.height for b in body)
    x = (width - (sum(b.width for b in body) + pad * (len(body) - 1))) // 2
    y = head.height + (cap_y - head.height - row_h) // 2
    for b in body:
        img.paste(b, (x, y + (row_h - b.height) // 2))
        x += b.width + pad
    return img


def save_gif(frames: list[Image.Image], out: str, seconds: float, hold: float, colors: int):
    """One shared palette: these frames use a handful of colours between them, and a
    per-frame palette would cost bytes without changing what anyone sees."""
    master = frames[0].quantize(colors=colors, method=Image.MEDIANCUT)
    quant = [f.quantize(palette=master, dither=Image.NONE) for f in frames]
    durations = [int(seconds * 1000)] * (len(quant) - 1) + [int(hold * 1000)]
    quant[0].save(out, save_all=True, append_images=quant[1:], duration=durations,
                  loop=0, optimize=True, disposal=2)
    print(f"[gif] {len(quant)} frames  {os.path.getsize(out) / 1e6:.1f} MB → {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phrase", nargs="?", default="ICrA 2027")
    p.add_argument("--results", default=None,
                   help="one sweep to read; default is the first of --prefer that "
                        "can spell the phrase")
    p.add_argument("--prefer", nargs="+", default=["big-budget", "small-budget"],
                   metavar="SWEEP",
                   help="sweep names under optimized/base-optimizer, best budget first")
    p.add_argument("--out-dir", default=None, help="default <results>/phrases")
    p.add_argument("--font", default="dejavusans-bold", help="glyph variant to spell with")
    p.add_argument("--best-font", action="store_true",
                   help="highest-IoU variant per character instead of one family")
    p.add_argument("--size", type=int, default=448, help="px of the enlarged panel")
    p.add_argument("--cell", type=int, default=96, help="px of a filmstrip thumbnail")
    p.add_argument("--seconds", type=float, default=0.9, help="seconds per character")
    p.add_argument("--hold", type=float, default=3.0, help="seconds on the final frame")
    p.add_argument("--colors", type=int, default=64)
    p.add_argument("--only", choices=["shadows", "overlay"], default=None)
    a = p.parse_args()

    root = os.path.join(_BENCH, "optimized", "base-optimizer")
    if a.results:
        results = a.results
    else:
        results = choose_results([os.path.join(root, s) for s in a.prefer],
                                 a.phrase, a.font, a.best_font)
        print(f"[sweep] {os.path.relpath(results, _BENCH)}")
    out_dir = a.out_dir or os.path.join(results, "phrases")
    os.makedirs(out_dir, exist_ok=True)
    slug = "".join(c if c.isalnum() else "-" for c in a.phrase).strip("-").lower()

    glyphs = [find_glyph(results, ch, a.font, a.best_font) for ch in a.phrase if ch != " "]
    idx = [i for i, ch in enumerate(a.phrase) if ch != " "]
    for g in glyphs:
        print(f"[glyph] {g['char']!r}  {g['subset']}/{g['stem']}  IoU {g['iou']:.3f}")

    head_lines = caption_lines(results, a.phrase, glyphs, a.font, a.best_font)
    S, C = a.size, a.cell

    # Thumbnails are rendered once at panel resolution and downsampled per strip, so a
    # frame costs a resize rather than a re-read of the source PNGs.
    items = [None if ch == " " else i for i, ch in enumerate(a.phrase)]
    masks = {i: (load_mask(g["target"], C), load_mask(g["shadow"], C)) for i, g in zip(idx, glyphs)}
    shadows = {i: panel(s) for i, (_, s) in masks.items()}
    targets = {i: panel(t) for i, (t, _) in masks.items()}
    overlays = {i: overlay_panel(t, s) for i, (t, s) in masks.items()}
    pad = max(12, S // 24)

    # Both GIFs share a canvas width so the pair sits side by side in a slide without
    # one of them being scaled down to meet the other.
    P = S * 3 // 4
    strip = filmstrip(items, shadows, C)
    width = max(strip.width + 2 * pad, 3 * P + 4 * pad, 900)
    head = banner(width, head_lines)
    line = round(max(11, round(width * 0.017)) * 1.6)
    label_h = round(max(11, round(width * 0.016)) * 1.7)
    mean = sum(g["iou"] for g in glyphs) / len(glyphs)

    if a.only != "overlay":
        size = (width, head.height + pad + S + pad + line + strip.height + pad)
        frames = []
        for i, g in zip(idx, glyphs):
            big = panel(load_mask(g["shadow"], S))
            cap = f"{g['subset']}/{g['stem']}      IoU {g['iou']:.3f}"
            frames.append(compose(size, head, [big], filmstrip(items, shadows, C, i), cap, pad))
        # Payoff frame, held long enough to read: the demo is the phrase, and until here
        # it has only been seen a glyph at a time. Targets in the strip below it, so the
        # frame anyone screenshots carries the comparison rather than repeating itself.
        frames.append(compose(size, head, [phrase_block(items, shadows, width - 2 * pad, S)],
                              filmstrip(items, targets, C),
                              f"the whole phrase — mean IoU {mean:.3f}   (targets below)", pad))
        save_gif(frames, os.path.join(out_dir, f"{slug}-shadows.gif"), a.seconds, a.hold, a.colors)

    if a.only != "shadows":
        lf = _font(max(11, round(width * 0.016)))
        body_h = P + label_h
        size = (width, head.height + pad + body_h + pad + line + strip.height + pad)
        frames = []
        for i, g in zip(idx, glyphs):
            T, Sh = load_mask(g["target"], P), load_mask(g["shadow"], P)
            body = []
            for img, text in zip([panel(T), panel(Sh), overlay_panel(T, Sh)],
                                 ["target", "robot shadow", "overlay"]):
                b = Image.new("RGB", (img.width, img.height + label_h), BG)
                ImageDraw.Draw(b).text((img.width // 2, 0), text, font=lf, fill=ACCENT, anchor="ma")
                b.paste(img, (0, label_h))
                body.append(b)
            # Recomputed at panel resolution rather than reusing the stored score: these
            # panels are resampled, and a caption should describe the pixels above it.
            iou = (T & Sh).sum() / max((T | Sh).sum(), 1)
            cap = (f"{g['subset']}/{g['stem']}      IoU {iou:.3f}"
                   "      cyan missed · magenta spill · blue agreement")
            frames.append(compose(size, head, body, filmstrip(items, targets, C, i), cap, pad))
        frames.append(compose(size, head, [phrase_block(items, overlays, width - 2 * pad, body_h)],
                              filmstrip(items, targets, C),
                              f"every glyph overlaid — mean IoU {mean:.3f}", pad))
        save_gif(frames, os.path.join(out_dir, f"{slug}-overlay.gif"), a.seconds, a.hold, a.colors)


if __name__ == "__main__":
    main()
