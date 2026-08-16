#!/usr/bin/env python3
"""Walk the alphabet a letter at a time, lowercase beside uppercase.

The two letter subsets are scored separately and read separately, which hides the one
comparison a shadow rig makes anyone curious: the same letter in both cases is the same
*name* and a completely different shape problem. `a` is a closed bowl with a stem, `A`
is two struts and a crossbar. Putting them in one frame, 26 frames in a row, turns
`letters_lower` vs `letters_upper` from two numbers into something you can watch.

Two GIFs, mirroring `make_phrase_gif.py`:

    alphabet-shadows.gif    the two cast shadows, side by side, with an index strip
    alphabet-overlay.gif    both cases as target | robot shadow | overlay, two rows

Every panel, palette and layout helper is imported from `make_phrase_gif` rather than
restated — these are the same figure with a different sequence through the same sweep,
and two copies of the overlay convention would eventually disagree about which colour
means spill.
"""

import argparse
import os
import string

from PIL import Image, ImageDraw

import make_phrase_gif as P
from make_phrase_gif import (BG, ACCENT, MUTED, _font, banner, choose_results, compose,
                             filmstrip, find_glyph, load_mask, overlay_panel, panel,
                             resolve_sweep, save_gif)

_BENCH = P._BENCH


def label_over(img: Image.Image, text: str, font, color=ACCENT) -> Image.Image:
    """Caption a panel by growing it upward, so type never covers a shadow."""
    h = round(font.size * 1.7)
    out = Image.new("RGB", (img.width, img.height + h), BG)
    ImageDraw.Draw(out).text((img.width // 2, 0), text, font=font, fill=color, anchor="ma")
    out.paste(img, (0, h))
    return out


def row(images: list[Image.Image], label: str, font, label_w: int, gap: int) -> Image.Image:
    """One case's panels in a row, with the case named in a fixed-width left gutter.

    The gutter is a fixed width rather than measured per label, so the panels sit at the
    same x in both rows and in every frame — a column that shifts between `a` and `A`
    reads as the image jittering rather than as the letter changing.
    """
    h = max(i.height for i in images)
    out = Image.new("RGB", (label_w + sum(i.width for i in images) + gap * len(images), h), BG)
    ImageDraw.Draw(out).text((label_w - gap, h // 2), label, font=font, fill=ACCENT, anchor="rm")
    x = label_w
    for i in images:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


def stack(images: list[Image.Image], gap: int) -> Image.Image:
    w = max(i.width for i in images)
    out = Image.new("RGB", (w, sum(i.height for i in images) + gap * (len(images) - 1)), BG)
    y = 0
    for i in images:
        out.paste(i, ((w - i.width) // 2, y))
        y += i.height + gap
    return out


def grid(panels: list[Image.Image], cols: int, gap: int) -> Image.Image:
    cw, ch = panels[0].width, panels[0].height
    rows = (len(panels) + cols - 1) // cols
    out = Image.new("RGB", (cols * cw + (cols - 1) * gap, rows * ch + (rows - 1) * gap), BG)
    for i, p in enumerate(panels):
        out.paste(p, ((i % cols) * (cw + gap), (i // cols) * (ch + gap)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--letters", default=string.ascii_lowercase,
                   help="which letters to walk, in order")
    p.add_argument("--results", default=None, help="one sweep; default the first of --prefer")
    p.add_argument("--prefer", nargs="+", default=["big-budget", "small-budget"], metavar="SWEEP")
    p.add_argument("--out-dir", default=None, help="default <results>/phrases")
    p.add_argument("--font", default="dejavusans-bold")
    p.add_argument("--best-font", action="store_true")
    p.add_argument("--size", type=int, default=352, help="px of a shadows-GIF panel")
    p.add_argument("--cell", type=int, default=34, help="px of an index-strip cell")
    p.add_argument("--seconds", type=float, default=0.8)
    p.add_argument("--hold", type=float, default=3.0)
    p.add_argument("--colors", type=int, default=64)
    p.add_argument("--only", choices=["shadows", "overlay"], default=None)
    a = p.parse_args()

    letters = a.letters
    if a.results:
        results = resolve_sweep(a.results)
    else:
        # Both cases of every letter must be present, so the probe phrase is both cases.
        results = choose_results([resolve_sweep(s) for s in a.prefer],
                                 letters + letters.upper(), a.font, a.best_font)
    print(f"[sweep] {os.path.relpath(results, _BENCH)}")
    out_dir = a.out_dir or os.path.join(results, "phrases")
    os.makedirs(out_dir, exist_ok=True)

    pairs = [(find_glyph(results, ch, a.font, a.best_font),
              find_glyph(results, ch.upper(), a.font, a.best_font)) for ch in letters]
    lo = sum(g["iou"] for g, _ in pairs) / len(pairs)
    up = sum(g["iou"] for _, g in pairs) / len(pairs)
    for ch, (l, u) in zip(letters, pairs):
        print(f"[pair] {ch}{ch.upper()}  {l['iou']:.3f} / {u['iou']:.3f}")
    print(f"[mean] lowercase {lo:.3f}  uppercase {up:.3f}")

    flat = [g for pair in pairs for g in pair]
    fitted = any(g["fitted"] for g in flat)
    third = (f"{2 * len(pairs)} glyphs · {a.font} · mean best-of-N IoU"
             f" — lowercase {lo:.3f} · uppercase {up:.3f}")
    if fitted:
        # Same caveat the phrase banner carries: on a fitted sweep the number is against
        # a target that was moved, and the move is the experiment.
        s = [g["scale"] for g in flat if g["scale"]]
        third += f" · targets placed into reach (×{min(s):.2f}–×{max(s):.2f})"
    head_lines = [
        f"UMBRA — {letters[0]}{letters[0].upper()} … {letters[-1]}{letters[-1].upper()},"
        " every letter in both cases, cast by 3× SO-101",
        P.budget_caption(results)[1],
        third,
    ]

    # The index strip is the alphabet as targets, not as shadows: it is there to say
    # which letter is on screen, and a legible letterform does that better than the
    # shadow whose readability is the question being asked.
    idx = {i: panel(load_mask(u["target"], a.cell * 2)) for i, (_, u) in enumerate(pairs)}
    items = list(range(len(pairs)))
    pad = max(12, a.size // 20)
    strip = filmstrip(items, idx, a.cell)
    width = max(strip.width + 2 * pad, 1080)
    head = banner(width, head_lines)
    lf = _font(max(11, round(width * 0.016)))
    cf = _font(max(11, round(width * 0.017)))
    line = round(cf.size * 1.6)

    if a.only != "overlay":
        S = a.size
        body_h = S + round(lf.size * 1.7)
        size = (width, head.height + pad + body_h + pad + line + strip.height + pad)
        frames = []
        for i, (l, u) in enumerate(pairs):
            body = [label_over(panel(load_mask(g["shadow"], S)),
                               f"{g['char']}      IoU {g['iou']:.3f}", lf)
                    for g in (l, u)]
            cap = f"{l['subset']}/{l['stem']}      ·      {u['subset']}/{u['stem']}"
            frames.append(compose(size, head, body, filmstrip(items, idx, a.cell, i), cap, pad))
        # Closer: the whole alphabet as shadows, lowercase above uppercase. The frames
        # before it show two letters at a time; this is the only view of the set.
        cell = min(body_h // 4 - 6, (width - 2 * pad) // 13 - 8)
        sheet = stack([grid([panel(load_mask(g["shadow"], cell)) for g in row_], 13, 8)
                       for row_ in ([l for l, _ in pairs], [u for _, u in pairs])], 10)
        frames.append(compose(size, head, [sheet], filmstrip(items, idx, a.cell),
                              f"all {2 * len(pairs)} shadows — lowercase above uppercase", pad))
        save_gif(frames, os.path.join(out_dir, "alphabet-shadows.gif"), a.seconds, a.hold, a.colors)

    if a.only != "shadows":
        Q = a.size * 3 // 4
        gap = pad
        # Measured, not guessed: the gutter has to hold the widest of the 52 labels, and
        # a fraction-of-width guess clips the letter off the front of every one of them.
        labels = {g["char"]: f"{g['char']}  {g['iou']:.3f}" for pair in pairs for g in pair}
        label_w = round(max(cf.getlength(s) for s in labels.values())) + 2 * gap
        head_h = round(lf.size * 1.7)
        body_h = 2 * Q + head_h + gap
        size = (width, head.height + pad + body_h + pad + line + strip.height + pad)
        frames = []
        for i, (l, u) in enumerate(pairs):
            rows = []
            for g in (l, u):
                T, Sh = load_mask(g["target"], Q), load_mask(g["shadow"], Q)
                panels = [panel(T), panel(Sh), overlay_panel(T, Sh)]
                if not rows:  # column headers belong on the top row only
                    panels = [label_over(im, t, lf) for im, t in
                              zip(panels, ["target", "robot shadow", "overlay"])]
                rows.append(row(panels, labels[g["char"]], cf, label_w, gap))
            placed = f"      placed ×{l['scale']:.2f} / ×{u['scale']:.2f}" if fitted else ""
            cap = (f"{l['subset']}/{l['stem']}   ·   {u['subset']}/{u['stem']}{placed}"
                   "      cyan missed · magenta spill · blue agreement")
            frames.append(compose(size, head, [stack(rows, gap)],
                                  filmstrip(items, idx, a.cell, i), cap, pad))
        cell = min((body_h - 10) // 4, (width - 2 * pad) // 13 - 8)
        sheet = stack([grid([overlay_panel(load_mask(g["target"], cell),
                                           load_mask(g["shadow"], cell)) for g in row_], 13, 8)
                       for row_ in ([l for l, _ in pairs], [u for _, u in pairs])], 10)
        frames.append(compose(size, head, [sheet], filmstrip(items, idx, a.cell),
                              f"every letter overlaid — lowercase {lo:.3f} · uppercase {up:.3f}",
                              pad))
        save_gif(frames, os.path.join(out_dir, "alphabet-overlay.gif"), a.seconds, a.hold, a.colors)


if __name__ == "__main__":
    main()
