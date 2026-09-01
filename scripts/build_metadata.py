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
TELEOP_IDX: dict[str, tuple[str, str]] = {}

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
    # teleop targets are shapes a human teleoperator actually cast, re-used as
    # targets. The shape is a digit / a device / whatever the capture was posed
    # against, so the prompt is built from the ORIGINATING subset's wording via
    # _teleop_index() rather than from this fallback.
    "teleop": "cast a shadow matching the target shape",
}

EMPTY_SHADOWS = {
    "hand": {"path": None, "captured_at": None, "operator": None, "notes": None},
    "teleop": {"path": None, "captured_at": None, "operator": None, "n_arms": None, "notes": None},
    "optimizer": {"path": None, "captured_at": None, "run_id": None, "config": None, "notes": None},
}


def _teleop_index() -> dict[str, tuple[str, str]]:
    """mask stem -> (originating subset, class), from the capture manifest.

    A teleop target's filename is `<subset><class>_<part>_<slug>_mask`, which
    `stem.split("_")[0]` reads as a single garbage token ("abstract" for
    abstract_device3_A1_..., "digits1" for digits1_B2_...). The manifest already
    records the real subset and class per capture, so use it and fall back only
    when a mask is not in it.
    """
    man = ROOT / "Teleops" / "masks" / "manifest.json"
    if not man.exists():
        return {}
    recs = json.loads(man.read_text(encoding="utf-8")).get("records", [])
    out = {}
    for r in recs:
        mask = r.get("mask") or ""
        stem = Path(mask).stem
        if stem and r.get("class"):
            out[stem] = (r.get("subset") or "", r["class"])
    return out


def load_existing() -> dict[str, dict]:
    if not META.exists():
        return {}
    return {r["id"]: r for r in (json.loads(line) for line in META.read_text(encoding="utf-8").splitlines() if line.strip())}


def main() -> None:
    existing = load_existing()
    global TELEOP_IDX
    TELEOP_IDX = _teleop_index()
    records = []
    for png in sorted((ROOT / "targets").rglob("*.png")):
        if png.stem.startswith("_"):
            continue
        subset = png.parent.name
        stem = png.stem  # e.g. "7_dejavusans-bold" or "swan_mpeg7-01"
        cls = stem.split("_")[0]
        origin = None
        if subset == "teleop" and stem in TELEOP_IDX:
            origin, cls = TELEOP_IDX[stem]
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
            if p.exists():
                shadows[src]["path"] = p.relative_to(ROOT).as_posix()
                continue
            # Only clear a link whose file is actually gone. This used to assign
            # None unconditionally, which wiped what link_teleop.py had written
            # on all 28 captured rows -- those point at Teleops/masks/..., not at
            # shadows/<sid>/teleop.png, so "not in the canonical place" was being
            # read as "does not exist".
            cur = shadows[src].get("path")
            if cur and not (ROOT / cur).exists():
                shadows[src]["path"] = None

        merged = {
            "id": sid,
            "subset": subset,
            "class": cls,
            "prompt": rec.get("prompt")
            or PROMPTS.get(origin or subset, "cast a shadow of {cls}").format(cls=cls),
            # as_posix(): str(Path) is backslashed on Windows, and metadata.jsonl
            # is committed -- a Windows run otherwise rewrites every row and
            # breaks consumers that match on "targets/".
            "target": png.relative_to(ROOT).as_posix(),
            "target_source": curated_src
            or rec.get("target_source")
            or ("generated" if subset in ("digits", "letters_upper", "letters_lower") else "curated"),
            "attributes": compute_attributes(mask),
            "shadows": shadows,
            "rig": rec.get("rig") or {"light": None, "screen_distance_m": None, "camera": None},
        }
        # Anything the row already carried that this script does not compute --
        # `version`/`rescue` from the quarantine pass (64 rows), `tags` from
        # link_teleop.py (28) -- is not this script's to drop. Building each
        # record from a fixed key set silently discarded them on every run.
        for k, v in rec.items():
            merged.setdefault(k, v)
        records.append(merged)

    with META.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    captured = sum(any(r["shadows"][s]["path"] for s in SOURCES) for r in records)
    print(f"metadata.jsonl: {len(records)} samples ({captured} with >=1 shadow)")


if __name__ == "__main__":
    main()
