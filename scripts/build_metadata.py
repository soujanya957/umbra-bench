"""Build/refresh metadata.jsonl from targets/ and shadows/.

- New targets get a fresh record (attributes auto-computed, shadows null).
- Existing records keep hand-edited capture metadata (operator, rig, notes...).
- Attributes are always recomputed; shadow paths are linked when files exist.

Usage: python scripts/build_metadata.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from shape_attributes import compute_attributes

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "metadata.jsonl"
SOURCES = ("hand", "teleop", "optimizer")

PROMPTS = {
    "digits": "cast a shadow of the digit {cls}",
    "letters_upper": "cast a shadow of the uppercase letter {cls}",
    "letters_lower": "cast a shadow of the lowercase letter {cls}",
    "animals": "cast a shadow of a {cls}",
    "objects": "cast a shadow of a {cls}",
    "vehicles": "cast a shadow of a {cls}",
    "figures": "cast a shadow of a {cls}",
    "abstract": "cast a shadow matching the target shape",  # no semantic prior — on purpose
    "hand_shadow": "cast a hand shadow of a {cls}",
}

EMPTY_SHADOWS = {
    "hand": {"path": None, "captured_at": None, "operator": None, "notes": None},
    "teleop": {"path": None, "captured_at": None, "operator": None, "n_arms": None, "notes": None},
    "optimizer": {"path": None, "captured_at": None, "run_id": None, "config": None, "notes": None},
}


def load_existing() -> dict[str, dict]:
    if not META.exists():
        return {}
    return {r["id"]: r for r in (json.loads(line) for line in META.read_text().splitlines() if line.strip())}


def main() -> None:
    existing = load_existing()
    records = []
    for png in sorted((ROOT / "targets").rglob("*.png")):
        if png.stem.startswith("_"):
            continue
        subset = png.parent.name
        stem = png.stem  # e.g. "7_dejavusans-bold" or "swan_mpeg7-01"
        cls = stem.split("_")[0]
        sid = f"{subset}_{stem}"

        # provenance: curation scripts write targets/<subset>/_sources.json
        sources_file = png.parent / "_sources.json"
        curated_src = None
        if sources_file.exists():
            curated_src = json.loads(sources_file.read_text()).get(stem)

        # canonical targets are black-shape-on-white; attributes expect shape=white
        mask = 255 - np.asarray(Image.open(png).convert("L"))
        rec = existing.get(sid, {})
        shadows = rec.get("shadows") or json.loads(json.dumps(EMPTY_SHADOWS))
        for src in SOURCES:
            p = ROOT / "shadows" / sid / f"{src}.png"
            shadows[src]["path"] = str(p.relative_to(ROOT)) if p.exists() else None

        records.append({
            "id": sid,
            "subset": subset,
            "class": cls,
            "prompt": rec.get("prompt") or PROMPTS.get(subset, "cast a shadow of {cls}").format(cls=cls),
            "target": str(png.relative_to(ROOT)),
            "target_source": curated_src
            or rec.get("target_source")
            or ("generated" if subset in ("digits", "letters_upper", "letters_lower") else "curated"),
            "attributes": compute_attributes(mask),
            "shadows": shadows,
            "rig": rec.get("rig") or {"light": None, "screen_distance_m": None, "camera": None},
        })

    with META.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    captured = sum(any(r["shadows"][s]["path"] for s in SOURCES) for r in records)
    print(f"metadata.jsonl: {len(records)} samples ({captured} with >=1 shadow)")


if __name__ == "__main__":
    main()
