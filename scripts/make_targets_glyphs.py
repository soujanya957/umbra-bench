"""Generate glyph targets: digits, uppercase and lowercase letters.

Deterministic: BLACK glyph on WHITE (shadow convention), 1-bit PNG,
512x512, centered, ~10% margin.

Subsets are separate directories (digits/, letters_upper/, letters_lower/) because
macOS's default filesystem is case-insensitive — "A_font.png" and "a_font.png"
would collide in one directory.

Fonts are looked up from common system paths; edit FONTS to add more.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = 512
MARGIN = 0.10
FONTS = {
    "dejavusans-bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "dejavuserif-bold": "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "dejavusansmono-bold": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
}
SUBSETS = {
    "digits": "0123456789",
    "letters_upper": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "letters_lower": "abcdefghijklmnopqrstuvwxyz",
}
TARGETS = Path(__file__).resolve().parent.parent / "targets"


def render_glyph(ch: str, font_path: str) -> Image.Image:
    # probe at a fixed size, then re-render scaled so every glyph fits the margin box
    probe = ImageFont.truetype(font_path, 400)
    img = Image.new("L", (1024, 1024), 0)
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), ch, font=probe)
    gw, gh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    target = SIZE * (1 - 2 * MARGIN)
    scale = target / max(gw, gh)
    font = ImageFont.truetype(font_path, int(400 * scale))
    img = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), ch, font=font)
    x = (SIZE - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = (SIZE - (bbox[3] - bbox[1])) / 2 - bbox[1]
    d.text((x, y), ch, fill=255, font=font)
    # canonical: black shape on white background (shadow on screen), 1-bit
    return img.point(lambda p: 0 if p > 127 else 255).convert("1")


def main() -> None:
    n = 0
    for subset, chars in SUBSETS.items():
        out = TARGETS / subset
        out.mkdir(parents=True, exist_ok=True)
        for name, path in FONTS.items():
            if not Path(path).exists():
                print(f"skip font (not found): {name} -> {path}")
                continue
            for ch in chars:
                img = render_glyph(ch, path)
                assert not np.asarray(img).all(), f"empty render: {ch} {name}"
                img.save(out / f"{ch}_{name}.png")
                n += 1
    print(f"wrote {n} glyph targets under {TARGETS}")


if __name__ == "__main__":
    main()
