#!/usr/bin/env python3
"""Concatenate the per-sweep metric CSVs into results/master_table.csv.

`compute_metrics.py` writes one `results/metrics_<tag>.csv` per sweep. Four
consumers -- `_build_browser_payload.py`, `_rescue_common.py`,
`make_corr_figures.py`, `quarantine_dropped.py` -- read a single
`results/master_table.csv` with a `sweep` column instead, and nothing in the
repo built it. It was assembled by hand somewhere else, which is why a fresh
checkout can compute every metric and still not render the dashboard.

The table is the concatenation and nothing more: one row per (sample, ref) per
sweep, with `sweep` set from the CSV's tag. Columns are the union across inputs,
so a sweep missing a column reads as NaN rather than dropping the sweep.

    python scripts/make_master_table.py                  # every metrics_*.csv
    python scripts/make_master_table.py --only small-budget-grounded
    python scripts/make_master_table.py --list           # what would go in
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BENCH, "results")
OUT = os.path.join(RESULTS, "master_table.csv")

# metrics_<tag>.csv -> sweep tag. The tag is the sweep's directory name under
# optimized/, so a row's `sweep` value is the thing you can `ls`.
PREFIX = "metrics_"


def discover(only: list[str] | None) -> list[tuple[str, str]]:
    """-> [(tag, path)] sorted by tag."""
    found = []
    for p in sorted(glob.glob(os.path.join(RESULTS, f"{PREFIX}*.csv"))):
        tag = os.path.basename(p)[len(PREFIX):-len(".csv")]
        if only and tag not in only:
            continue
        found.append((tag, p))
    return found


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None,
                    help="sweep tags to include (default: all metrics_*.csv)")
    ap.add_argument("--list", action="store_true", help="show inputs and exit")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    inputs = discover(a.only)
    if not inputs:
        raise SystemExit(
            f"no {PREFIX}*.csv in {RESULTS}\n"
            "run scripts/compute_metrics.py first -- it writes one per sweep.")

    if a.list:
        for tag, p in inputs:
            print(f"  {tag:<26} {os.path.getsize(p)/1e6:6.2f} MB  {p}")
        return

    frames = []
    for tag, p in inputs:
        df = pd.read_csv(p, low_memory=False)
        if "sweep" in df.columns:
            # A CSV that already carries the column is passed through unchanged;
            # overwriting it would silently relabel someone else's table.
            print(f"  {tag:<26} {len(df):>6} rows  (kept its own `sweep` column)")
        else:
            # assign-then-reorder rather than insert(): insert() on a 130-column
            # frame reallocates and pandas warns about fragmentation.
            df["sweep"] = tag
            df = df[["sweep"] + [c for c in df.columns if c != "sweep"]]
            print(f"  {tag:<26} {len(df):>6} rows")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True, sort=False)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False, encoding="utf-8")

    print(f"\nmaster_table: {len(out)} rows x {len(out.columns)} cols -> {a.out}")
    print(f"  sweeps: {', '.join(sorted(out['sweep'].unique()))}")
    print(f"  refs:   {', '.join(sorted(out['ref'].dropna().unique()))}")


if __name__ == "__main__":
    main()
