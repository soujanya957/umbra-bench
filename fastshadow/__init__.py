"""MuJoCo-free shadow rendering on nvdiffrast. See renderer.py."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(_HERE)


def _has_batched_hooks(repo: str) -> bool:
    p = os.path.join(repo, "motion-aware-shadow", "optimizer.py")
    return os.path.exists(p) and "eval_q_batch" in open(p, encoding="utf-8").read()


def fleet_repo() -> str:
    """FLEET_SHADOW_ART if set, else the first checkout beside the bench whose optimizer has the batched
    hooks (eval_q_batch / batch_shadow_loss, fleet-shadow-art main since 2026-10-15). Unlike
    scripts/run_base_optimizer._default_repo this skips the fleet-shadow-art-main worktree, which stays
    at the ablation provenance commit b86634c (tag provenance/ablation-2026-09) and has no hooks."""
    env = os.environ.get("FLEET_SHADOW_ART")
    if env and os.path.isdir(os.path.join(env, "motion-aware-shadow")):
        return env
    for cand in (
        os.path.join(os.path.dirname(BENCH), "fleet-shadow-art"),
        os.path.join(os.path.dirname(BENCH), "fleet-shadow-art-main"),
        os.path.expanduser("~/dev/fleet-shadow-art"),
    ):
        if _has_batched_hooks(cand):
            return cand
    raise FileNotFoundError("no fleet-shadow-art checkout with the batched optimizer hooks; set FLEET_SHADOW_ART")


def use_fleet(repo: str | None = None) -> str:
    """Put motion-aware-shadow on sys.path and chdir there (URDF meshes resolve
    relative to it). Returns the package dir."""
    ms = os.path.join(repo or fleet_repo(), "motion-aware-shadow")
    for p in (ms, os.path.join(ms, "targets")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(ms)
    return ms
