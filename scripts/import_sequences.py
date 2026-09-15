#!/usr/bin/env python3
"""Import the animated targets from fleet-shadow-art into the sequences track.

umbra-bench is entirely single-frame: all 542 targets are one silhouette solved
once, so the benchmark can ask "can the rig make this shape" and has no way to ask
"and can it then make the next one". That gap is not hypothetical. The existing
`spinning_star` run in fleet-shadow-art reports **mean per-frame IoU 0.6679** --
indistinguishable from a good static result -- while its recorded joint angles move
up to **291 degrees on a single joint between consecutive frames**, against the
repo's own feasibility bound `motion_planner.LARGE_Q_JUMP = 1.2 rad = 68.8 deg`.
Every one of its four transitions breaks that bound. Averaging per-frame IoU is
exactly the statistic that cannot see this, which is why the sequences track exists
as its own thing with its own metrics rather than as a tenth subset.

Sources, all 1-bit black-on-white like `targets/` (verified, no polarity flip):

    motion-aware-shadow/showcase/targets/{star_spin,wiper,triangle,flower}_NN.png
    motion-aware-shadow/showcase/targets/anim_n3/<motion>_NN.png    3-arm rig
    motion-aware-shadow/showcase/targets/anim_n5/<motion>_NN.png    5-arm rig
    plant-demo/targets/plant_NNNN.png

`anim_n3` and `anim_n5` overlap by name but not always by content: `bird` and
`stick_wave` are byte-identical across the two, `reeds` and `windmill` are not.
Identical pairs are imported once under the bare motion name; differing pairs are
kept as separate sequences suffixed `_n3` / `_n5`, because a target authored for a
wider rig is a different target. The decision is made by hashing, not assumed.

    python scripts/import_sequences.py
    python scripts/import_sequences.py --repo ../fleet-shadow-art --dry-run
"""

import argparse
import hashlib
import os
import re
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)


def _default_repo() -> str:
    for cand in (
        os.environ.get("FLEET_SHADOW_ART"),
        os.path.join(os.path.dirname(_BENCH), "fleet-shadow-art"),
        os.path.expanduser("~/dev/fleet-shadow-art"),
    ):
        if cand and os.path.isdir(os.path.join(cand, "motion-aware-shadow")):
            return cand
    return os.path.expanduser("~/dev/fleet-shadow-art")


def _md5(p: str) -> str:
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def _group(paths: list[str]) -> dict[str, list[str]]:
    """Split `<motion>_<index>.png` paths into ordered per-motion frame lists."""
    out: dict[str, list[tuple[int, str]]] = {}
    for p in paths:
        m = re.match(r"^(.*)_(\d+)\.png$", os.path.basename(p))
        if not m:
            continue
        out.setdefault(m.group(1), []).append((int(m.group(2)), p))
    return {k: [p for _, p in sorted(v)] for k, v in out.items()}


def collect(repo: str) -> dict[str, dict]:
    """-> {seq_id: {frames, source_glob, generator, rig_variant}}"""
    ms = os.path.join(repo, "motion-aware-shadow")
    show = os.path.join(ms, "showcase", "targets")
    seqs: dict[str, dict] = {}

    flat = _group([os.path.join(show, f) for f in os.listdir(show) if f.endswith(".png")])
    for motion, frames in flat.items():
        if len(frames) < 2:            # text_00_A etc are stills, not sequences
            continue
        seqs[motion] = dict(frames=frames, rig_variant=None,
                            source=os.path.relpath(os.path.join(show, f"{motion}_NN.png"), repo),
                            generator="motion-aware-shadow/targets/generate_targets.py")

    n3 = _group([os.path.join(show, "anim_n3", f) for f in os.listdir(os.path.join(show, "anim_n3"))])
    n5 = _group([os.path.join(show, "anim_n5", f) for f in os.listdir(os.path.join(show, "anim_n5"))])
    for motion in sorted(set(n3) | set(n5)):
        a, b = n3.get(motion), n5.get(motion)
        same = (a and b and len(a) == len(b)
                and all(_md5(x) == _md5(y) for x, y in zip(a, b)))
        gen = "motion-aware-shadow/targets/generate_rig_targets.py"
        if same:                       # one artefact, imported once
            seqs[motion] = dict(frames=a, rig_variant="n3,n5 (identical)",
                                source=f"motion-aware-shadow/showcase/targets/anim_n{{3,5}}/{motion}_NN.png",
                                generator=gen)
        else:
            for tag, fr in (("n3", a), ("n5", b)):
                if fr:
                    seqs[f"{motion}_{tag}"] = dict(
                        frames=fr, rig_variant=tag,
                        source=f"motion-aware-shadow/showcase/targets/anim_{tag}/{motion}_NN.png",
                        generator=gen)

    plant_dir = os.path.join(repo, "plant-demo", "targets")
    if os.path.isdir(plant_dir):
        pg = _group([os.path.join(plant_dir, f) for f in os.listdir(plant_dir) if f.endswith(".png")])
        for motion, frames in pg.items():
            if len(frames) >= 2:
                seqs[motion] = dict(frames=frames, rig_variant=None,
                                    source=f"plant-demo/targets/{motion}_NNNN.png",
                                    generator="plant-demo")
    return seqs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default=_BENCH)
    p.add_argument("--repo", default=_default_repo())
    p.add_argument("--out", default="sequences", help="destination tree, repo-relative")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if not os.path.isdir(os.path.join(a.repo, "motion-aware-shadow")):
        p.error(f"--repo {a.repo!r} contains no motion-aware-shadow/")

    seqs = collect(a.repo)
    out_root = os.path.join(a.bench, a.out)
    total = 0
    print(f"[import] {len(seqs)} sequences from {a.repo}\n")
    for sid in sorted(seqs):
        s = seqs[sid]
        dst_dir = os.path.join(out_root, sid)
        if os.path.isdir(dst_dir) and not a.force:
            print(f"  {sid:<18} exists, skipped (--force to redo)")
            continue
        if not a.dry_run:
            os.makedirs(dst_dir, exist_ok=True)
            for i, src in enumerate(s["frames"]):
                shutil.copyfile(src, os.path.join(dst_dir, f"f{i:02d}.png"))
        total += len(s["frames"])
        tag = f"  [{s['rig_variant']}]" if s["rig_variant"] else ""
        print(f"  {sid:<18} {len(s['frames']):3d} frames  <- {s['source']}{tag}")

    print(f"\n[import] {total} frames"
          f"{' (dry run, nothing written)' if a.dry_run else f' -> {out_root}'}")
    print("[import] next: python scripts/build_sequence_metadata.py")


if __name__ == "__main__":
    main()
