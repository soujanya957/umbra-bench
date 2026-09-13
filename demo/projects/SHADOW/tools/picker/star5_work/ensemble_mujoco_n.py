"""In-memory patched copy of demo/ensemble_mujoco.py: build() seats
entry["n_arms"] arms (default 3) instead of a hard-coded trio.

demo/ensemble_mujoco.py itself is NOT touched (another agent is using it).
Its text is read, exactly one block is replaced, and the result is exec'd into
a fresh module whose __file__ is the REAL path, so BENCH / FSA / SO101_XML
resolve exactly as they do in the original. Importing this file returns that
module (sys.modules entry swapped during import).

Patch (the only change):
    for j in range(3):
        frame = grp.add_frame(pos=[0, gap - j * gap, 0])
  ->
    n_arms = int(e.get("n_arms", 3))
    for j in range(n_arms):
        frame = grp.add_frame(pos=[0, ((n_arms - 1) / 2 - j) * gap, 0])
For n_arms=3 this is the original placement; for any n the mid arm stays on the
group origin and arm 0 (SR101, base.x 0) is the one nearest the lamp.
"""
import sys
import types
from pathlib import Path

REAL = Path(r"C:/Users/hexia/Documents/GitHub/umbra-bench/demo/ensemble_mujoco.py")

OLD = (
    '        gap = float(e.get("arm_gap_m", 0.2))\n'
    '        for j in range(3):\n'
    '            # SR101 (base.x 0) nearest the light in the manifest frame;\n'
    '            # z -> y flips depth order, same rig\n'
    '            frame = grp.add_frame(pos=[0, gap - j * gap, 0])\n'
)
NEW = (
    '        gap = float(e.get("arm_gap_m", 0.2))\n'
    '        n_arms = int(e.get("n_arms", 3))\n'
    '        for j in range(n_arms):\n'
    '            # SR101 (base.x 0) nearest the light in the manifest frame;\n'
    '            # z -> y flips depth order, same rig; mid arm on the group origin\n'
    '            frame = grp.add_frame(pos=[0, ((n_arms - 1) / 2 - j) * gap, 0])\n'
)


def patched_source() -> str:
    src = REAL.read_text(encoding="utf-8")
    k = src.count(OLD)
    if k != 1:
        raise RuntimeError(f"ensemble_mujoco.py patch anchor found {k}x (expected 1) -- "
                           "the original changed; re-derive the patch")
    return src.replace(OLD, NEW)


_mod = types.ModuleType("ensemble_mujoco_n")
_mod.__file__ = str(REAL)
_mod.PATCHED_FROM = str(REAL)
exec(compile(patched_source(), str(REAL), "exec"), _mod.__dict__)
sys.modules[__name__] = _mod
