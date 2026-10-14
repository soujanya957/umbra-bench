"""MuJoCo-free shadow rendering on nvdiffrast. See renderer.py."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(_HERE)


def fleet_repo() -> str:
    """Same resolution order as scripts/run_base_optimizer._default_repo."""
    for cand in (
        os.environ.get("FLEET_SHADOW_ART"),
        os.path.join(os.path.dirname(BENCH), "fleet-shadow-art-main"),
        os.path.join(os.path.dirname(BENCH), "fleet-shadow-art"),
        os.path.expanduser("~/dev/fleet-shadow-art"),
    ):
        if cand and os.path.isdir(os.path.join(cand, "motion-aware-shadow")):
            return cand
    raise FileNotFoundError("fleet-shadow-art not found; set FLEET_SHADOW_ART")


def use_fleet(repo: str | None = None) -> str:
    """Put motion-aware-shadow on sys.path and chdir there (URDF meshes resolve
    relative to it). Returns the package dir."""
    ms = os.path.join(repo or fleet_repo(), "motion-aware-shadow")
    for p in (ms, os.path.join(ms, "targets")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(ms)
    return ms
