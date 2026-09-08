#!/usr/bin/env python3
"""Publish atlas sequences into the UI's Targets shelf, so demos can be collected there.

The plumbing already exists on both ends; this only moves the data.

`TargetAsset` in `Shadow_robot_ui/src/types/dataModel.ts` was built for exactly
this -- its own comment reads "many = a target SEQUENCE (the shadow-space
counterpart of a motion clip)". It lands in `library/models.json` under `targets`,
which `MeshLibraryLoader` hydrates on open and auto-saves back, and `TargetsPanel`
renders as one thumbnail per clip that flips through its frames on click.

What is NOT wired is the way in: the panel's mask-file import is commented out
("Import Masks SHELVED"), so there is no button that does this. Writing the records
into `models.json` is the supported route, not a workaround -- the GET/POST pair at
`/api/library-models` is the same path the app itself persists through.

Frames come from `results/sequences_payload.json`, already base64'd at a shared px
by `_build_sequences_payload.py`, so nothing is re-encoded and nothing can drift
from what the atlas shows. All 54 sequences are 0.31 MB of base64 in total -- size
is not a reason to be selective here.

Ids are `umbra-<sequence id>`: stable, so re-running updates in place rather than
piling up duplicates (the loader merges by id), and prefixed, so this script can
tell its own records from ones drawn in the UI and never touches the latter.

    python scripts/export_to_shape_library.py --dry-run
    python scripts/export_to_shape_library.py --family demo --solved-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
PREFIX = "umbra-"


def _default_ui() -> str:
    for cand in (
        os.environ.get("SHADOW_ROBOT_UI"),
        os.path.join(os.path.dirname(_BENCH), "fleet-shadow-art", "Shadow_robot_ui"),
        os.path.join(os.path.dirname(_BENCH), "Shadow_robot_ui"),
    ):
        if cand and os.path.isdir(os.path.join(cand, "library")):
            return cand
    return ""


def to_target(s: dict) -> dict:
    """One payload record -> one TargetAsset."""
    # `source` drives the Library UI's provenance label. The demo family was cut
    # from footage (it carries fps and an origin); the generated 13 were rasterised
    # from parametric generators. Calling both "image" would throw that away.
    src = "video" if (s.get("fps") or s.get("origin")) else "image"
    tags = [t for t in (s.get("family"), s.get("cls")) if t]
    tags.append("loop" if s.get("loop") else "open")
    tags.append("solved" if s.get("solved") else "unsolved")
    return {
        "id": f"{PREFIX}{s['id']}",
        "name": s["id"],
        "source": src,
        "masks": [f"data:image/png;base64,{f}" for f in s["f"]],
        # hz is what makes the shelf play the clip at the rate it was cut at.
        # 6 matches the panel's own default for a multi-frame import.
        "hz": s.get("fps") or 6,
        "tags": tags,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", default=_BENCH)
    p.add_argument("--ui-repo", default=_default_ui())
    p.add_argument("--payload", default="results/sequences_payload.json")
    p.add_argument("--only", nargs="+", default=None, help="sequence ids")
    p.add_argument("--family", choices=["demo", "generated"], default=None)
    p.add_argument("--solved-only", action="store_true")
    p.add_argument("--prune", action="store_true",
                   help="drop previously exported records that this run excludes")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if not a.ui_repo or not os.path.isdir(os.path.join(a.ui_repo, "library")):
        p.error("could not find Shadow_robot_ui; pass --ui-repo or set SHADOW_ROBOT_UI")

    pl = os.path.join(a.bench, a.payload)
    if not os.path.isfile(pl):
        p.error(f"{pl} not found; run scripts/_build_sequences_payload.py first")
    seqs = json.load(open(pl, encoding="utf-8"))["sequences"]

    sel = seqs
    if a.only:
        sel = [s for s in sel if s["id"] in set(a.only)]
    if a.family:
        sel = [s for s in sel if (s.get("family") or "generated") == a.family]
    if a.solved_only:
        sel = [s for s in sel if s.get("solved")]
    if not sel:
        p.error("selection is empty")

    new = [to_target(s) for s in sel]

    dest = os.path.join(a.ui_repo, "library", "models.json")
    try:
        lib = json.load(open(dest, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        lib = {}
    lib.setdefault("assets", [])
    existing = lib.get("targets") or []

    # Hand-made records are never touched; ours are replaced by id.
    mine = {t["id"] for t in new}
    kept = [t for t in existing
            if not (t.get("id", "").startswith(PREFIX)
                    and (a.prune or t["id"] in mine))]
    lib["targets"] = kept + new

    frames = sum(len(t["masks"]) for t in new)
    mb = len(json.dumps(lib, separators=(",", ":"))) / 1e6
    print(f"[export] {len(new)} sequences, {frames} frames -> {dest}")
    print(f"[export] kept {len(kept)} existing target(s); models.json will be {mb:.2f} MB")
    for t in new[:8]:
        print(f"    {t['name']:<32} {len(t['masks']):3d} frames  {t['hz']}Hz  "
              f"[{', '.join(t['tags'])}]")
    if len(new) > 8:
        print(f"    ... and {len(new) - 8} more")

    if a.dry_run:
        print("\n[export] dry run, nothing written")
        return

    if os.path.isfile(dest):
        bak = dest + f".bak_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copyfile(dest, bak)
        print(f"[export] backed up -> {os.path.basename(bak)}")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(lib, f, separators=(",", ":"))

    print("\n[export] done. Open the Shape page -> Library -> Targets tab.")
    print("[export] note: the loader merges by id and LOCAL wins, so a browser that")
    print("         already holds an edited copy of one of these keeps its own until")
    print("         that target is removed in the UI.")


if __name__ == "__main__":
    main()
