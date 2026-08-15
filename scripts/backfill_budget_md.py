#!/usr/bin/env python3
"""Write a BUDGET.md for a sweep that finished before the runner emitted one.

Everything here is read back out of the sweep's own `results.json` files, which record
the optimizer config and rig they were solved with — so the document describes what was
actually run, not what someone remembers launching. Refuses to guess: if the sweep's
results disagree about their config, that is reported rather than averaged away.

Shares `write_budget_md` with the runner, so backfilled and live budget sheets stay
formatted identically and remain comparable side by side.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from run_base_optimizer import write_budget_md  # noqa: E402

# Defaults for fields added to the runner after this sweep ran. A results.json without
# them was produced by the older code path, whose behaviour these values describe.
_LEGACY = {"adaptive_final": True}

# Settings that change how fast a solve runs but not what it finds: worker threads only
# fan out the population evaluation, which `executor.map` reassembles in order, so a
# seed produces the same result at any width. A sweep that changed only these is still
# one budget, and saying otherwise would block a document that is perfectly accurate.
_EXECUTION_ONLY = {"n_workers"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", help="sweep directory, e.g. optimized/base-optimizer/small-budget")
    p.add_argument("--bench", default=_BENCH)
    p.add_argument("--repo", default=os.path.expanduser("~/dev/fleet-shadow-art"))
    p.add_argument("--note", default=None, help="provenance line; default explains the backfill")
    a = p.parse_args()

    records = []
    for dirpath, _, files in os.walk(a.results):
        if "results.json" in files:
            with open(os.path.join(dirpath, "results.json")) as f:
                records.append(json.load(f))
    if not records:
        sys.exit(f"no results.json under {a.results}")

    def budget_key(r):
        return json.dumps(
            {k: v for k, v in r["optimizer"].items() if k not in _EXECUTION_ONLY},
            sort_keys=True,
        )

    cfgs = {budget_key(r) for r in records}
    rigs = {json.dumps(r["rig"], sort_keys=True) for r in records}
    if len(cfgs) > 1 or len(rigs) > 1:
        sys.exit(
            f"[!] {a.results} mixes {len(cfgs)} search budgets and {len(rigs)} rigs — "
            "one BUDGET.md cannot describe it. Split the directory first."
        )

    # Execution-only settings may legitimately vary (a sweep resumed at a different
    # width). Record the spread instead of pretending it was uniform.
    varied = {
        k: sorted({r["optimizer"].get(k) for r in records})
        for k in _EXECUTION_ONLY
        if len({r["optimizer"].get(k) for r in records}) > 1
    }

    cfg = dict(records[0]["optimizer"])
    rig = records[0]["rig"]
    for k, v in _LEGACY.items():
        cfg.setdefault(k, v)

    runs = max(r.get("n_base_runs", r["n_runs"]) for r in records)
    extra = max(r.get("n_extra_runs", 0) for r in records)
    below = next((r.get("extra_below") for r in records if r.get("extra_below")), 0.0)
    subsets = sorted({r["subset"] for r in records})

    ns = SimpleNamespace(
        phase1_iters=cfg["phase1_iters"],
        phase2_iters=cfg["phase2_iters"],
        final_iters=cfg["final_iters"],
        adaptive_final=cfg["adaptive_final"],
        floor_penalty=cfg["floor_penalty"],
        collision_penalty=cfg["collision_penalty"],
        self_collision_penalty=cfg["self_collision_penalty"],
        n_workers=cfg["n_workers"],
        runs=runs,
        extra_runs=extra,
        extra_below=below,
        subsets=subsets,
    )

    note = a.note
    if note is None:
        note = (
            "\n> Backfilled after the fact from this sweep's `results.json` files —\n"
            "> the configuration below is what the solves actually recorded.\n"
        )
        for k, vals in varied.items():
            counts = ", ".join(
                f"{v} ({sum(1 for r in records if r['optimizer'].get(k) == v)} targets)"
                for v in vals
            )
            note += (
                f">\n> `{k}` varied across this sweep: {counts}. It sets how wide the\n"
                f"> population evaluation fans out, not what the search explores, so the\n"
                f"> budget below applies to every target regardless.\n"
            )

    out = os.path.join(a.results, "BUDGET.md")
    write_budget_md(out, ns, cfg, rig, len(records), a.repo, note=note)
    print(f"[budget] {len(records)} targets, {len(subsets)} subsets → {out}")


if __name__ == "__main__":
    main()
