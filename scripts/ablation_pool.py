#!/usr/bin/env python3
"""Replicate a reference solver study onto a larger glyph set with a pool of one-glyph solves.

    python scripts/ablation_pool.py --reference optimized/ablation-probe10 --glyphs dvsb36 --dry-run
    python scripts/ablation_pool.py --reference optimized/ablation-probe10 --glyphs dvsb36 --procs 22
    python scripts/ablation_pool.py --reference optimized/aimed-start-probe10 --glyphs dvsb36 \\
        --specs full-tiny-n3 no_aimed-tiny-n3 no_zones-tiny-n3

Why a pool instead of ablation_probe.sh: on Windows the optimizer clamps every process
to one render thread (WGL contexts are not thread-safe), so parallelism has to come from
many single-thread processes. Each process solves one (variant, glyph) cell, picked with
run_base_optimizer.py's own --shard/--num-shards split over the target tree, so the solve,
the output layout and results.json are exactly what the shell scripts produce and
collect_ablation.py / make_ablation_table.py / clip_eval_ablation.py read them unchanged.

Why configs are read back from the reference instead of retyped: a new glyph only
pairs with the reference glyphs if it was solved under the same flags. Each variant's
CLI is rebuilt from its results.json, pushed back through the bookkeeping
run_base_optimizer.py does (cfg overrides, --flat's injected fields, the variant
string), and compared field by field with EVERY results.json in that variant dir, so a
reconstruction that would record anything different fails before a solve is launched.

Why glyph-major order: under a deadline the run may be stopped at any point. Finishing
every variant of one glyph before the next glyph starts means whatever exists is a set
of complete, paired glyphs, which is what the paired deltas need. Tiers come first (the
compact table's N=3 main block, then every cell its footer rows read, then ORDER's other
N=5 variants, then the rest), so the paper table fills before the extras.

Reference glyphs are copied in rather than re-solved (--seed-from-reference), and
optimized/<out-sub>/PROVENANCE.md records which cells came from which machine.
optimized/<out-sub>/POOL.lock keeps a second driver (or a restart while a killed driver's
children still run) off the same folder.
"""
from __future__ import annotations

import argparse
import atexit
import collections
import dataclasses
import glob
import heapq
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collect_ablation import ORDER  # noqa: E402  the paper table's variant order

SUBSETS = ["letters_upper", "digits"]
FONT = "dejavusans-bold"
PRESETS = {
    "probe10": [("letters_upper", f"{c}_{FONT}") for c in "EHKUW"]
    + [("digits", f"{c}_{FONT}") for c in "23479"],
    "dvsb36": [("letters_upper", f"{chr(c)}_{FONT}") for c in range(65, 91)]
    + [("digits", f"{i}_{FONT}") for i in range(10)],
    "all": None,  # every PNG in targets_grounded/<SUBSETS>, listed when resolved
}
ORDER_NAMES = [v for v, _, _ in ORDER]
# Every cell make_ablation_table.py compact()'s footer rows read. Its main block is ORDER at
# N=3; ORDER's other N=5 variants (full_no_icp-n5, ...) sit in no row of that table, so they
# must not hold up these under a deadline.
FOOTER = ["flat-n5", "full-n5",                                    # N=5 row
          "full-tiny-n3", "no_zones-tiny-n3", "no_aimed-tiny-n3",  # quarter-budget rows
          "full_long-n3", "full_long_no_icp-n3"]                   # sequence-budget row
# run_base_optimizer.py flags that land in results.json "optimizer", in its argparse order.
OPT_FLAGS = [
    ("popsize", "--popsize", int),
    ("phase1_iters", "--phase1-iters", int),
    ("phase2_iters", "--phase2-iters", int),
    ("final_iters", "--final-iters", int),
    ("floor_penalty", "--floor-penalty", float),
    ("collision_penalty", "--collision-penalty", float),
    ("self_collision_penalty", "--self-collision-penalty", float),
]
# --flat writes these itself, after --cfg; passing them as --cfg would change nothing
# in the record but would change the variant string, so they are never passed.
FLAT_INJECTED = ("max_iter", "use_hungarian", "use_voronoi_warmstart", "use_icp_restarts", "fd_iters")
# Execution details, not conditions: the win32 clamp makes n_workers differ by machine.
IGNORED = ("n_workers",)
CORE_SOLVER = ("optimizer.py", "renderer.py", "loss.py")

_BENCH = os.path.dirname(HERE)
_DRIVER_LOG = None


def die(msg):
    raise SystemExit(f"[pool] ERROR: {msg}")


def say(msg=""):
    print(msg, flush=True)
    if _DRIVER_LOG:
        try:
            with open(_DRIVER_LOG, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def rel(path):
    try:
        r = os.path.relpath(path, _BENCH)
    except ValueError:  # another drive
        return path
    return path if r.startswith("..") else r.replace(os.sep, "/")


def cmdline(cmd):
    return subprocess.list2cmdline(cmd) if sys.platform == "win32" else shlex.join(cmd)


def git(repo, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", repo, *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def is_lfs_pointer(path):
    """An unsmudged git-lfs file: copying it would plant a text stub where a PNG belongs."""
    try:
        if os.path.getsize(path) > 1024:
            return False
        with open(path, "rb") as fh:
            return fh.read(24).startswith(b"version https://git-lfs")
    except OSError:
        return False


def same_bytes(a, b):
    try:
        with open(a, "rb") as fa, open(b, "rb") as fb:
            return fa.read() == fb.read()
    except OSError:
        return False


def fmt_num(x):
    return repr(x) if isinstance(x, float) and not isinstance(x, bool) else str(x)


# ── Reading the reference ────────────────────────────────────────────────────


def read_budget(path):
    """Who ran a sweep, and the flag values a --cfg override hides from results.json.

    run_base_optimizer.py records cfg_common AFTER --cfg is applied, so e.g.
    full_no_backward's --phase2-iters 16 survives only here, as the table row.
    """
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as fh:
        txt = fh.read()

    def grab(pat, cast=str):
        m = re.search(pat, txt, re.M)
        return cast(m.group(1)) if m else None

    def opt_num(s):
        return None if s == "None" else float(s)

    b = {}
    m = re.search(r"^Generated (.+?) on `([^`]*)`", txt, re.M)
    if m:
        b["generated"], b["machine"] = m.group(1), m.group(2)
    b["sha"] = grab(r"fleet-shadow-art @ `([^`]*)`")
    b["popsize"] = grab(r"^\| popsize \| (\d+) \|", int)
    b["phase1_iters"] = grab(r"^\| phase1_iters [^|]*\| (\d+) \|", int)
    b["phase2_iters"] = grab(r"^\| phase2_iters [^|]*\| (\d+) \|", int)
    b["final_iters"] = grab(r"^\| final_iters [^|]*\| (\d+) \|", int)
    b["adaptive_final"] = grab(r"^\| adaptive_final \| (True|False) \|", lambda s: s == "True")
    m = re.search(r"^\| floor / collision / self-collision penalty \| (\S+) / (\S+) / (\S+) \|", txt, re.M)
    if m:
        b["floor_penalty"], b["collision_penalty"], b["self_collision_penalty"] = map(float, m.groups())
    b["runs"] = grab(r"^- (\d+) independent solves per target", int)
    m = re.search(r"below \*\*IoU (\S+)\*\* get \*\*(\d+) extra\*\*", txt)
    if m:
        b["extra_below"], b["extra_runs"] = float(m.group(1)), int(m.group(2))
    b["n_robots"] = grab(r"^\| robots \| (\d+)", int)
    b["arm_gap"] = grab(r"^\| arm gap \| (\S+) m \|", float)
    m = re.search(r"^\| light-to-front / back-to-wall \| (\S+) / (\S+) ", txt, re.M)
    if m:
        b["light"], b["back"] = opt_num(m.group(1)), opt_num(m.group(2))
    b["size"] = grab(r"^\| render size \| (\d+) px \|", int)
    b["distortion"] = grab(r"^\| target deformation[^|]*\| (.*?) \|\s*$", lambda s: s != "False")
    b["target_fit"] = grab(r"^\| target fit[^|]*\| (.*?) \|\s*$", lambda s: s != "False")
    return {k: v for k, v in b.items() if v is not None}


def log_variant(bench, ref_sub, name):
    """The reference's own launch log for a variant that has no results.json yet.

    A log is matched to its variant dir by the BUDGET.md path it prints, not by file
    name: the shell scripts name logs `${v}_${b}_n${N}.log`, which does not invert.
    """
    pat = re.compile(
        r"[\\/]optimized[\\/]" + re.escape(ref_sub) + r"[\\/]" + re.escape(name) + r"[\\/]BUDGET\.md"
    )
    for lp in sorted(glob.glob(os.path.join(bench, "logs", ref_sub, "*.log"))):
        try:
            with open(lp, encoding="utf-8", errors="replace") as fh:
                head = fh.read(16000)
        except OSError:
            continue
        v = re.search(r"\] variant=(\S+)", head)
        if pat.search(head) and v:
            runs = re.search(r"\] \d+ targets \S+ (\d+) runs", head)
            gap = re.search(r"arm_gap=(\S+)", head)
            return {
                "log": lp,
                "variant": v.group(1),
                "runs": int(runs.group(1)) if runs else None,
                "arm_gap": float(gap.group(1)) if gap else None,
            }
    return None


def scan_reference(ref_root):
    """variant dir -> {(subset, stem): (glyph dir, results record)}"""
    out = {}
    for d in sorted(os.listdir(ref_root)):
        vdir = os.path.join(ref_root, d)
        if not os.path.isdir(vdir):
            continue
        cells = {}
        for f in sorted(glob.glob(os.path.join(vdir, "*", "*", "results.json"))):
            gdir = os.path.dirname(f)
            key = (os.path.basename(os.path.dirname(gdir)), os.path.basename(gdir))
            try:
                with open(f, encoding="utf-8") as fh:
                    r = json.load(fh)
            except (OSError, ValueError) as e:
                die(f"{rel(f)}: unreadable results.json ({e})")
            if r.get("id") != f"{key[0]}_{key[1]}":
                die(f"{rel(f)}: id {r.get('id')!r} does not match its folder {key[0]}/{key[1]}")
            cells[key] = (gdir, r)
        out[d] = cells
    return out


def parse_variant(variant, where):
    if variant == "flat":
        return True, []
    if variant == "staged":
        return False, []
    if isinstance(variant, str) and variant.startswith("staged+"):
        items = variant[len("staged+"):].split(",")
        if all(re.fullmatch(r"[A-Za-z_]\w*=[^,]*", x) for x in items):
            return False, items
    die(f"{where}: cannot parse variant {variant!r} back into --flat / --cfg")


def guess(v):
    """A --cfg value typed the way _parse_cfg_overrides would, near enough to compare records."""
    vl = v.strip().lower()
    if vl == "none":
        return None
    if vl in ("true", "yes", "on"):
        return True
    if vl in ("false", "no", "off"):
        return False
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def fmt_cfg(v):
    return "none" if v is None else ("true" if v is True else "false" if v is False else fmt_num(v))


def cli_flags(c, runs=None):
    """run_base_optimizer.py flags for a condition (everything but job, paths and threads)."""
    f = ["--n-robots", str(c["n_robots"]), "--arm-gap", fmt_num(c["arm_gap"])]
    if c["light"] is not None:
        f += ["--light-to-front", fmt_num(c["light"])]
    if c["back"] is not None:
        f += ["--back-to-wall", fmt_num(c["back"])]
    f += ["--size", str(c["size"])]
    for key, flag, _ in OPT_FLAGS:
        f += [flag, fmt_num(c[key])]
    if not c["adaptive_final"]:
        f.append("--no-adaptive-final")
    if c["flat"]:
        f.append("--flat")
    for item in c["cfg"]:
        f += ["--cfg", item]
    if c["extra_runs"]:
        f += ["--extra-runs", str(c["extra_runs"]), "--extra-below", fmt_num(c["extra_below"])]
    return f + ["--runs", str(c["runs"] if runs is None else runs)]


def expected_record(c):
    """The condition-defining fields run_base_optimizer.py would write for this CLI,
    following its cfg_common / --flat / variant code line for line."""
    opt = {k: c[k] for k, _, _ in OPT_FLAGS}
    opt["adaptive_final"] = c["adaptive_final"]
    opt.update({k: guess(v) for k, v in (x.split("=", 1) for x in c["cfg"])})
    if c["flat"]:
        ph_pop = max(32, 4 + 3 * math.ceil(math.log(6)))
        staged = (
            c["n_robots"] * (c["phase1_iters"] + c["phase2_iters"]) * ph_pop
            + c["final_iters"] * c["popsize"]
        )
        opt.update(
            max_iter=max(1, round(staged / c["popsize"])),
            use_hungarian=False,
            use_voronoi_warmstart=False,
            use_icp_restarts=False,
            fd_iters=0,
        )
    for k in IGNORED:
        opt.pop(k, None)
    return {
        "method": "base-optimizer",
        "fit": None,
        "rig": {
            "n_robots": c["n_robots"], "arm_gap_m": c["arm_gap"], "light_to_front_m": c["light"],
            "back_to_wall_m": c["back"], "render_size": c["size"], "distortion": False,
            "target_fit": False,
        },
        "optimizer": opt,
        "variant": "flat" if c["flat"] else ("staged+" + ",".join(c["cfg"]) if c["cfg"] else "staged"),
        "n_base_runs": c["runs"],
        "extra_below": c["extra_below"] if c["extra_runs"] else None,
    }


def recorded(r):
    return {
        "method": r.get("method"),
        "fit": r.get("fit"),
        "rig": {k: v for k, v in (r.get("rig") or {}).items() if k != "targets_dir"},
        "optimizer": {k: v for k, v in (r.get("optimizer") or {}).items() if k not in IGNORED},
        "variant": r.get("variant"),
        "n_base_runs": r.get("n_base_runs"),
        "extra_below": r.get("extra_below"),
    }


def diff(want, got, path=""):
    if isinstance(want, dict) and isinstance(got, dict):
        out = []
        for k in sorted(set(want) | set(got), key=str):
            out += diff(want.get(k, "<absent>"), got.get(k, "<absent>"), f"{path}.{k}" if path else k)
        return out
    return [] if want == got else [f"{path}: rebuilt {want!r}, recorded {got!r}"]


def from_results(name, cells, budget, notes):
    recs = [(os.path.join(g, "results.json"), r) for g, r in cells.values()]
    f0, r = recs[0]
    rig, opt = r.get("rig") or {}, r.get("optimizer") or {}
    flat, cfg = parse_variant(r.get("variant"), rel(f0))
    base = {k for k, _, _ in OPT_FLAGS} | {"adaptive_final"} | set(IGNORED)
    if flat:
        # "flat" hides any --cfg that came with it; recover them from the record itself.
        hidden = [k for k in opt if k not in base and k not in FLAT_INJECTED]
        cfg = [f"{k}={fmt_cfg(opt[k])}" for k in hidden]
        if cfg:
            notes.append("flat record carries extra optimizer fields; passed back as --cfg " + " ".join(cfg))
    cfg_keys = [x.split("=", 1)[0] for x in cfg]
    c = {
        "n_robots": rig.get("n_robots"), "arm_gap": rig.get("arm_gap_m"),
        "light": rig.get("light_to_front_m"), "back": rig.get("back_to_wall_m"),
        "size": rig.get("render_size"), "flat": flat, "cfg": cfg, "runs": r.get("n_base_runs"),
        "extra_runs": 0, "extra_below": None,
    }
    if None in (c["n_robots"], c["arm_gap"], c["size"], c["runs"]):
        die(f"{rel(f0)}: rig/n_base_runs incomplete, cannot rebuild the CLI")
    for key, flag, cast in OPT_FLAGS + [("adaptive_final", "--no-adaptive-final", bool)]:
        if key not in opt:
            die(f"{rel(f0)}: optimizer.{key} missing")
        if key in cfg_keys:
            if budget and key in budget:
                c[key] = budget[key]
            else:
                c[key] = cast(opt[key])
                notes.append(
                    f"{flag} is shadowed by --cfg {key} and no BUDGET.md value survives; passing "
                    "the overridden value (identical solve, BUDGET.md reads differently)"
                )
        else:
            c[key] = opt[key]
    if r.get("extra_below") is not None:
        k = max(x.get("n_extra_runs") or 0 for _, x in recs) or (budget or {}).get("extra_runs")
        if not k:
            die(f"{name}: reference granted extra runs below IoU {r['extra_below']}, but how many "
                "is recorded nowhere (no glyph triggered them and no BUDGET.md)")
        c["extra_runs"], c["extra_below"] = k, r["extra_below"]
    if not flat:
        stray = [k for k in opt if k not in base and k not in cfg_keys]
        if stray:
            die(f"{rel(f0)}: optimizer fields {stray} are not explained by its variant string {r['variant']!r}")
    want = expected_record(c)
    for f, x in recs:
        d = diff(want, recorded(x))
        if d:
            die(f"{name}: {rel(f)} disagrees with the CLI rebuilt from {rel(f0)}:\n    "
                + "\n    ".join(d) + "\n  rebuilt: " + " ".join(cli_flags(c)))
    if budget:
        pairs = [(k, c[k]) for k, _, _ in OPT_FLAGS if k not in cfg_keys]
        pairs += [("adaptive_final", c["adaptive_final"])] if "adaptive_final" not in cfg_keys else []
        pairs += [("runs", c["runs"]), ("n_robots", c["n_robots"]), ("arm_gap", c["arm_gap"]), ("size", c["size"])]
        bad = [f"{k} {budget[k]!r} vs {v!r}" for k, v in pairs if k in budget and budget[k] != v]
        if bad:
            notes.append("BUDGET.md disagrees with results.json (results.json wins; the last "
                         "shard-0 launch rewrites BUDGET.md): " + "; ".join(bad))
    return c


def from_budget(name, budget, lv, notes):
    need = [k for k, _, _ in OPT_FLAGS] + ["adaptive_final", "runs", "n_robots", "arm_gap", "size"]
    if not budget:
        return None, "no results.json and no BUDGET.md"
    miss = [k for k in need if k not in budget]
    if miss:
        return None, f"no results.json; BUDGET.md lacks {miss}"
    if not lv:
        return None, "no results.json; BUDGET.md has no --cfg list and no reference log names this dir"
    if budget.get("distortion") or budget.get("target_fit"):
        die(f"{name}: BUDGET.md reports a target warp/fit, which this pool does not reproduce")
    flat, cfg = parse_variant(lv["variant"], rel(lv["log"]))
    c = {k: budget[k] for k in need}
    c.update(light=budget.get("light"), back=budget.get("back"), flat=flat, cfg=cfg,
             extra_runs=budget.get("extra_runs", 0), extra_below=budget.get("extra_below"))
    for k, v in (("runs", lv.get("runs")), ("arm_gap", lv.get("arm_gap"))):
        if v is not None and v != c[k]:
            die(f"{name}: BUDGET.md says {k}={c[k]!r} but {rel(lv['log'])} says {v!r}")
    notes.append(f"UNVERIFIED: no results.json in the reference; flags from BUDGET.md, --cfg from "
                 f"the 'variant=' line of {rel(lv['log'])}")
    return c, None


def sibling_seconds(c, scan):
    """Mean reference seconds of any variant on the same budget and fleet, for ETA only."""
    xs = []
    for cells in scan.values():
        for _, r in cells.values():
            o, g = r.get("optimizer") or {}, r.get("rig") or {}
            if (g.get("n_robots"), o.get("popsize"), o.get("phase1_iters"), o.get("final_iters"),
                    r.get("n_base_runs")) == (c["n_robots"], c["popsize"], c["phase1_iters"],
                                              c["final_iters"], c["runs"]):
                xs.append(r["seconds"])
    return sum(xs) / len(xs) if xs else None


# ── Specs and tiers ──────────────────────────────────────────────────────────


@dataclasses.dataclass
class Spec:
    name: str
    ref_dir: str
    cli: dict
    how: str
    notes: list
    ref_seconds: float
    budget: dict
    ref_cells: dict
    tier_key: tuple
    tier_label: str
    seed_cells: set = dataclasses.field(default_factory=set)


def tier_of(name):
    """Paper-table priority: the compact table's N=3 main block, then the cells its footer
    rows read, then the remaining ORDER ladders by N, then anything else."""
    if name in FOOTER:
        return (1, 0), "table footer"
    m = re.fullmatch(r"(.+)-n(\d+)", name)
    if m and m.group(1) in ORDER_NAMES:
        n = int(m.group(2))
        return ((0, n), "N=3 main block") if n == 3 else ((2, n), f"N={n} extras")
    return (3, 0), "other"


def default_order(names):
    """Tiers in tier_of order; inside a tier collect_ablation.ORDER per N, then FOOTER, then the rest."""
    ns = sorted({int(m.group(2)) for m in (re.fullmatch(r"(.+)-n(\d+)", x) for x in names)
                 if m and m.group(1) in ORDER_NAMES})
    out = [f"{v}-n{n}" for n in ns for v in ORDER_NAMES if f"{v}-n{n}" in names]
    out += [x for x in FOOTER if x in names and x not in out]
    out += sorted(x for x in names if x not in out)
    return sorted(out, key=lambda x: tier_of(x)[0])  # stable: the order above survives inside a tier


def build_specs(a, bench, ref_root, ref_sub, scan):
    wanted = [x for item in (a.specs or []) for x in item.split(",") if x]
    unknown = [x for x in wanted if x not in scan]
    if unknown:
        die(f"--specs {unknown} not under {rel(ref_root)}; have: {' '.join(scan)}")
    specs, skipped = [], []
    all_secs = [r["seconds"] for cells in scan.values() for _, r in cells.values()]
    for name in list(dict.fromkeys(wanted)) or default_order(list(scan)):
        cells = scan[name]
        budget = read_budget(os.path.join(ref_root, name, "BUDGET.md"))
        notes = []
        if cells:
            c = from_results(name, cells, budget, notes)
            how = f"results.json x{len(cells)}, all agree"
            secs = sum(r["seconds"] for _, r in cells.values()) / len(cells)
        else:
            c, why = from_budget(name, budget, log_variant(bench, ref_sub, name), notes)
            if c is None:
                if wanted:
                    die(f"{name}: cannot rebuild its CLI ({why})")
                skipped.append((name, why))
                continue
            how = "BUDGET.md + reference log"
            secs = sibling_seconds(c, scan)
            if secs is None:
                secs = sum(all_secs) / len(all_secs) if all_secs else 100.0
                notes.append(f"ETA uses the mean over the whole reference ({secs:.0f}s): no sibling on this budget")
            else:
                notes.append(f"ETA uses {secs:.0f}s, the reference mean on the same budget and N")
        tk, tl = tier_of(name)
        specs.append(Spec(name, os.path.join(ref_root, name), c, how, notes, secs, budget or {},
                          {k: g for k, (g, _) in cells.items()}, tk, tl))
    return specs, skipped


def group_tiers(specs):
    """[(label, [spec])], tiers in order of first appearance, specs in the given order."""
    tiers, index = [], {}
    for s in specs:
        if s.tier_key not in index:
            index[s.tier_key] = len(tiers)
            tiers.append((s.tier_label, []))
        tiers[index[s.tier_key]][1].append(s)
    return tiers


# ── Glyphs and the target tree ───────────────────────────────────────────────


def resolve_glyphs(items, bench):
    tg = os.path.join(bench, "targets_grounded")
    out, presets_only, tokens = [], True, []
    for item in items:
        for tok in (t.strip() for t in item.split(",")):
            if not tok:
                continue
            tokens.append(tok)
            if tok in PRESETS:
                gs = PRESETS[tok]
                if gs is None:
                    gs = [(s, os.path.splitext(fn)[0]) for s in SUBSETS if os.path.isdir(os.path.join(tg, s))
                          for fn in sorted(os.listdir(os.path.join(tg, s))) if fn.lower().endswith(".png")]
            else:
                presets_only = False
                parts = tok.replace("\\", "/").strip("/").split("/")
                if len(parts) != 2:
                    die(f"--glyphs {tok!r}: expected a preset ({', '.join(PRESETS)}) or subset/stem")
                stem = parts[1][:-4] if parts[1].lower().endswith(".png") else parts[1]
                gs = [(parts[0], stem)]
            out += [g for g in gs if g not in out]
    if not out:
        die("--glyphs selected nothing")
    missing = [f"{s}/{t}" for s, t in out if not os.path.isfile(os.path.join(tg, s, t + ".png"))]
    if missing:
        die(f"not in targets_grounded: {' '.join(missing)}")
    label = "+".join(dict.fromkeys(tokens)) if presets_only else f"custom{len(out)}"
    return out, label


def order_glyphs(glyphs, subsets, how):
    """`stratified` interleaves subsets in proportion to their size, so a run stopped early
    holds letters and digits in the full set's ratio instead of A..P and no digits."""
    if how == "listed":
        return list(glyphs)
    by = collections.defaultdict(list)
    for g in glyphs:
        by[g[0]].append(g)
    keyed = [((i + 0.5) / len(gs), subsets.index(s), g) for s, gs in by.items() for i, g in enumerate(gs)]
    return [g for _, _, g in sorted(keyed, key=lambda t: t[:2])]


def target_jobs(tdir, subsets, add=()):
    """run_base_optimizer.py's job list, replicated: subsets in CLI order, each dir's
    sorted(os.listdir) filtered to .png. `add` = (subset, filename) about to be copied in,
    so --dry-run sees the tree the run will see."""
    jobs = []
    for sub in subsets:
        d = os.path.join(tdir, sub)
        names = os.listdir(d) if os.path.isdir(d) else []
        extra = [fn for s, fn in add if s == sub and fn not in names]
        if not os.path.isdir(d) and not extra:
            continue
        for fn in sorted(list(names) + extra):
            if fn.lower().endswith(".png"):
                jobs.append((sub, os.path.splitext(fn)[0]))
    return jobs


def crosscheck_shards(run_base, bench, tdir_rel, subsets, jobs):
    """Self-check against the source, not against this file's copy of it: exec
    run_base_optimizer.py's own job-collection lines for every shard on the tree and
    require each to select exactly the one glyph scheduled for it."""
    with open(run_base, encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(
        r"^([ \t]*)jobs = \[\]\n.*?^\1jobs = \[j for i, j in enumerate\(jobs\) if i % a\.num_shards == a\.shard\]\n",
        src, re.M | re.S,
    )
    if not m:
        return "SKIPPED: job-collection block not found in run_base_optimizer.py (shard math unverified)"
    code = compile(textwrap.dedent(m.group(0)), run_base, "exec")
    n = len(jobs)
    for i, want in enumerate(jobs):
        ns = {"os": os, "print": lambda *x, **k: None,
              "a": types.SimpleNamespace(subsets=list(subsets), bench=bench, targets_dir=tdir_rel,
                                         num_shards=n, shard=i)}
        exec(code, ns)
        got = [(s, os.path.splitext(os.path.basename(t))[0]) for s, t in ns["jobs"]]
        if got != [want]:
            die(f"shard self-check: run_base_optimizer.py --shard {i} --num-shards {n} selects {got}, "
                f"scheduled {want}")
    return f"passed: run_base_optimizer.py's own job-collection code selects exactly the scheduled glyph for all {n} shards"


def crosscheck_virtual(run_base, tdir, tdir_rel, subsets, add, jobs):
    """--dry-run writes nothing to the bench, so mirror the would-be tree (empty files) in a temp dir."""
    with tempfile.TemporaryDirectory(prefix="ablation_pool_") as tmp:
        for sub in subsets:
            d, vd = os.path.join(tdir, sub), os.path.join(tmp, tdir_rel, sub)
            names = os.listdir(d) if os.path.isdir(d) else []
            if not names and not any(s == sub for s, _ in add) and not os.path.isdir(d):
                continue
            os.makedirs(vd, exist_ok=True)
            for fn in set(names) | {fn for s, fn in add if s == sub}:
                if os.path.isdir(os.path.join(d, fn)):
                    os.makedirs(os.path.join(vd, fn), exist_ok=True)
                else:
                    open(os.path.join(vd, fn), "wb").close()
        return crosscheck_shards(run_base, tmp, tdir_rel, subsets, jobs)


# ── Seeding and provenance ───────────────────────────────────────────────────


def plan_seed(spec, out_root, glyphs):
    """-> (cells, copies, pointers). Never replaces real data: a glyph folder is filled only
    while its results.json is absent or byte-identical to the reference's (i.e. seeded
    before), and only missing files or LFS stubs are written. A folder with no results.json
    but a file the reference lacks or holds differently is a killed local solve: seeding
    into it would pair the reference's results.json with that solve's _best.png/_runKK.png,
    so it is left to be re-solved instead. results.json goes last so a half-copied folder
    never looks finished."""
    cells, copies, pointers = set(), [], []
    vdir = os.path.join(out_root, spec.name)
    for g in glyphs:
        src = spec.ref_cells.get(g)
        if not src:
            continue
        dst = os.path.join(vdir, g[0], g[1])
        rd = os.path.join(dst, "results.json")
        names = sorted(n for n in os.listdir(src)
                       if not n.startswith(".") and os.path.isfile(os.path.join(src, n)))
        if os.path.exists(rd):
            if not same_bytes(os.path.join(src, "results.json"), rd):
                continue
        elif os.path.isdir(dst) and any(
                os.path.isfile(os.path.join(dst, n)) and not (
                    n in names and (is_lfs_pointer(os.path.join(dst, n))
                                    or same_bytes(os.path.join(src, n), os.path.join(dst, n))))
                for n in os.listdir(dst)):
            continue
        cells.add(g)
        names.sort(key=lambda n: n == "results.json")
        for n in names:
            s, d = os.path.join(src, n), os.path.join(dst, n)
            if is_lfs_pointer(s):
                pointers.append(s)
            elif not os.path.exists(d) or is_lfs_pointer(d):
                copies.append((s, d))
    budget = os.path.join(spec.ref_dir, "BUDGET.md")
    if cells and os.path.isfile(budget) and not os.path.exists(os.path.join(vdir, "BUDGET.md")):
        copies.append((budget, os.path.join(vdir, "BUDGET.md")))
    return cells, copies, pointers


def write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def pid_alive(pid):
    """Is `pid` a live python process? os.kill(pid, 0) is no probe on Windows (it calls
    TerminateProcess), so ask the kernel; a pid reused by a non-python program reads dead."""
    if sys.platform != "win32":
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except (PermissionError, OSError):
            return True
        return True
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    k32.QueryFullProcessImageNameW.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                               ctypes.POINTER(wintypes.DWORD))
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ctypes.get_last_error() == 5  # access denied: it exists
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != 259:  # STILL_ACTIVE
            return False
        buf, n = ctypes.create_unicode_buffer(1024), wintypes.DWORD(1024)
        if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)):
            return "python" in os.path.basename(buf.value).lower()
        return True
    finally:
        k32.CloseHandle(h)


class PoolLock:
    """One driver per output folder. Two drivers on one --out-sub launch the same cells twice,
    and a driver killed past its cleanup (Task Manager, a stopped background task) leaves its
    children solving where the next driver cannot see them. The lock names the driver and its
    live children, so a second start refuses instead of doubling up."""

    def __init__(self, path):
        self.path, self.mine = path, False
        self.state = {"node": platform.node(), "driver_pid": os.getpid(), "started": stamp(), "children": []}

    def acquire(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        for attempt in (0, 1):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if attempt:
                    die(f"{rel(self.path)} reappeared while replacing a stale lock; another pool is starting")
                self._clear_stale()
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.state, fh)
            self.mine = True
            atexit.register(self.release)
            return

    def _clear_stale(self):
        try:
            with open(self.path, encoding="utf-8") as fh:
                old = json.load(fh)
        except (OSError, ValueError):
            old = {}
        if old.get("node") == platform.node():
            pids = [old.get("driver_pid")] + list(old.get("children") or [])
            live = [p for p in pids if p and pid_alive(p)]
            if live:
                die(f"{rel(self.path)}: the pool started {old.get('started')} (driver pid {old.get('driver_pid')}) "
                    f"still has live python process(es) {live}. Let it finish, or stop them "
                    f"(taskkill /F /PID <pid>) and re-run; delete the lock only if those pids are unrelated.")
        say(f"[pool] note: replacing stale {rel(self.path)} (node {old.get('node')}, driver pid "
            f"{old.get('driver_pid')}, started {old.get('started')})")
        os.remove(self.path)

    def children(self, pids):
        # Windows refuses os.replace onto a file another process has open (a second driver
        # reading it), so retry briefly rather than leave a live child out of the lock.
        self.state["children"] = sorted(pids)
        for _ in range(20):
            try:
                write_atomic(self.path, json.dumps(self.state))
                return
            except OSError:
                time.sleep(0.05)

    def release(self):
        # Children that would not die keep the lock: the next start must see them.
        if not self.mine or self.state["children"]:
            return
        self.mine = False
        try:
            with open(self.path, encoding="utf-8") as fh:
                if json.load(fh).get("driver_pid") != os.getpid():
                    return
            os.remove(self.path)
        except (OSError, ValueError):
            pass


def disk_cells(out_root):
    out = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(out_root, "*", "*", "*", "results.json")):
        gdir = os.path.dirname(f)
        out[os.path.basename(os.path.dirname(os.path.dirname(gdir)))][
            f"{os.path.basename(os.path.dirname(gdir))}/{os.path.basename(gdir)}"] = f
    return out


def render_provenance(led, out_root, out_sub, subsets):
    esc = lambda s: str(s).replace("|", "\\|")  # noqa: E731
    disk = disk_cells(out_root)
    variants = list(dict.fromkeys(led["variants"] + sorted(disk)))
    glyphs = sorted({g for v in variants for g in list(led["cells"].get(v, {})) + list(disk.get(v, {}))},
                    key=lambda g: (subsets.index(g.split("/")[0]) if g.split("/")[0] in subsets else 99, g))
    L = [
        f"# Provenance - `{out_sub}`", "",
        f"Replicates `{led['reference']}` on more glyphs with `scripts/ablation_pool.py`. A cell is",
        "one variant x glyph folder. `R` cells were copied from the reference study: solved on the",
        "reference machine listed below and not re-solved here. `L<k>` cells were solved by pool",
        "session k below, one single-thread process per cell. Per-run PNGs exist only where a cell",
        "was solved. A variant dir's BUDGET.md is the reference's copy until a local solve of the",
        "first PNG in the target tree (shard 0) rewrites it with this machine's budget.", "",
        f"Regenerated {stamp()}.", "",
        "## Reference variants", "",
        "| variant | machine | fleet-shadow-art | generated | CLI rebuilt from | flags |",
        "|---|---|---|---|---|---|",
    ]
    for v in led["variants"]:
        b = led["reference_budgets"].get(v, {})
        L.append(f"| {v} | {esc(b.get('machine', '?'))} | {esc(b.get('sha', '?'))} | {esc(b.get('generated', '?'))} "
                 f"| {esc(b.get('how', '?'))} | `{esc(b.get('flags', ''))}` |")
    L += ["", "## Pool sessions", "",
          "| k | started | machine | platform | fleet-shadow-art | solver vs reference | procs | command |",
          "|---|---|---|---|---|---|---|---|"]
    for s in led["sessions"]:
        L.append(f"| {s['id']} | {s['started']} | {esc(s['node'])} | {esc(s['platform'])} | {esc(s['sha'])} "
                 f"| {esc(s['solver_vs_reference'])} | {s['procs']} | `{esc(s['command'])}` |")
    L += ["", "## Cells", "",
          "`R` reference copy, `L<k>` solved in pool session k, `?` on disk but not recorded by the pool, "
          "`.` missing.", "",
          "| glyph | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
    count = collections.Counter()
    for g in glyphs:
        row = []
        for v in variants:
            cell = led["cells"].get(v, {}).get(g)
            on_disk = g in disk.get(v, {})
            sym = "." if not on_disk else ("?" if not cell else ("R" if cell["source"] == "reference" else f"L{cell['session']}"))
            count[sym[0]] += 1
            row.append(sym)
        L.append(f"| {g} | " + " | ".join(row) + " |")
    L += ["", f"Totals: {count['R']} reference, {count['L']} local, {count['?']} unrecorded, {count['.']} missing.", ""]
    if led.get("adopted_provenance"):
        L += ["## Earlier PROVENANCE.md", "",
              "Found here before this pool first ran and not written by it; its cells show as `?` above.", "",
              "```text", led["adopted_provenance"].rstrip("\n"), "```", ""]
    return "\n".join(L)


# ── Estimates ────────────────────────────────────────────────────────────────


def makespan(durations, procs, busy=()):
    """Finish time of each job when jobs start in order on the first free of `procs` slots."""
    slots = list(busy)[:procs] + [0.0] * max(0, procs - len(busy))
    heapq.heapify(slots)
    fin = []
    for d in durations:
        t = heapq.heappop(slots) + d
        heapq.heappush(slots, t)
        fin.append(t)
    return fin


@dataclasses.dataclass
class Job:
    tier: int
    spec: Spec
    g: tuple
    rjson: str
    log: str
    ref_s: float
    shard: int


# ── Pool ─────────────────────────────────────────────────────────────────────


class Pool:
    def __init__(self, a, jobs, ctx):
        self.a, self.ctx = a, ctx
        self.queue = collections.deque(jobs)
        self.total = len(jobs)
        self.running = {}  # Popen -> (job, t0, log offset)
        self.ok = self.failed = self.skipped = self.launched = 0
        self.local_s = self.ref_s = 0.0
        self.failures, self.interrupted_jobs = [], []
        self.consec = 0
        self.stop = None
        self.interrupted = False
        self.tier_left = collections.Counter(j.tier for j in jobs)
        self.tier_ok = collections.Counter()
        self.tier_n = collections.Counter(j.tier for j in jobs)
        self.t0 = time.time()

    def ratio(self):
        if self.ok and self.ref_s > 0:
            r = self.local_s / self.ref_s
            return r, f"x{r:.2f} of reference, measured on {self.ok}"
        return self.a.speed, f"x{self.a.speed:.2f} of reference, --speed"

    def eta(self):
        r, _ = self.ratio()
        now = time.time()
        busy = [(j, max(0.0, j.ref_s * r - (now - t))) for j, t, _ in self.running.values()]
        queued = [] if self.stop else list(self.queue)
        if self.a.limit is not None:
            queued = queued[: max(0, self.a.limit - self.launched)]
        fin = makespan([j.ref_s * r for j in queued], self.a.procs, [b for _, b in busy])
        per = collections.defaultdict(float)
        for j, t in busy + list(zip(queued, fin)):
            per[j.tier] = max(per[j.tier], t)
        return max([0.0] + list(per.values())), per

    def progress(self, what):
        eta, _ = self.eta()
        _, basis = self.ratio()
        say(f"[{time.strftime('%H:%M:%S')}] {self.ok + self.failed + self.skipped}/{self.total}"
            f"  ok {self.ok}  fail {self.failed}  run {len(self.running)}"
            f"  elapsed {(time.time() - self.t0) / 3600:.2f}h  eta {eta / 3600:.2f}h ({basis})  {what}")

    def status(self):
        _, per = self.eta()
        parts = []
        for ti, label in enumerate(self.ctx.tier_labels):
            if self.tier_n[ti]:
                s = f"{label} {self.tier_n[ti] - self.tier_left[ti]}/{self.tier_n[ti]}"
                parts.append(s + (f" eta {per[ti] / 3600:.2f}h" if self.tier_left[ti] and ti in per else ""))
        self.progress("status")
        say("           tiers: " + " | ".join(parts) + (f"  [no new launches: {self.stop}]" if self.stop else ""))

    def launch(self, job):
        c = self.ctx
        cmd = [c.py, c.run_base, "--repo", c.repo, "--bench", c.bench, "--targets-dir", c.tdir_rel,
               "--subsets", *c.subsets, "--out", os.path.join(c.out_root, job.spec.name),
               "--shard", str(job.shard), "--num-shards", str(c.n_shards), "--n-workers", str(c.nw),
               *cli_flags(job.spec.cli, c.runs_for(job.spec))]
        os.makedirs(os.path.dirname(job.log), exist_ok=True)
        with open(job.log, "ab") as fh:
            fh.write(f"# {stamp()} ablation_pool on {platform.node()}: {cmdline(cmd)}\n".encode("utf-8"))
            fh.flush()
            offset = fh.tell()
            kw = dict(stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, env=c.env, cwd=c.bench)
            # Own process group: Ctrl+C reaches only the driver, which then stops every child itself.
            if sys.platform == "win32":
                kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kw["start_new_session"] = True
            self.running[subprocess.Popen(cmd, **kw)] = (job, time.time(), offset)
        c.children_changed([p.pid for p in self.running])
        self.launched += 1
        c.session_started()

    def finish(self, p, rc):
        job, t, offset = self.running.pop(p)
        self.ctx.children_changed([q.pid for q in self.running])
        wall = time.time() - t
        self.tier_left[job.tier] -= 1
        g = f"{job.g[0]}/{job.g[1]}"
        why, r = None, None
        if rc != 0:
            why = f"exit code {rc}"
        elif not os.path.exists(job.rjson):
            why = "exited 0 without writing results.json"
        else:
            try:
                with open(job.rjson, encoding="utf-8") as fh:
                    r = json.load(fh)
                if r.get("id") != f"{job.g[0]}_{job.g[1]}":
                    why = f"results.json id {r.get('id')!r}"
            except (OSError, ValueError) as e:
                why = f"unreadable results.json ({e})"
        sl = shard_line(job.log, offset)
        if sl and (sl[2] != 1 or sl[:2] != (job.shard, self.ctx.n_shards)):
            why = why or "wrong shard selection"
            self.stop = self.stop or (f"shard mapping broken: {rel(job.log)} reports {sl[2]} target(s) "
                                      f"for shard {sl[0]}/{sl[1]}, scheduled {job.shard}/{self.ctx.n_shards} = {g}")
            say(f"[pool] ERROR {self.stop}")
        if why is None:
            self.ok += 1
            self.consec = 0
            self.tier_ok[job.tier] += 1
            self.local_s += wall
            self.ref_s += job.ref_s
            self.ctx.record_local(job, wall, r)
            self.progress(f"ok   {job.spec.name} {g} best={r.get('best_iou_vs_original', float('nan')):.3f} {wall:.0f}s")
        else:
            self.failed += 1
            self.consec += 1
            self.failures.append((job, why))
            self.progress(f"FAIL {job.spec.name} {g}: {why} after {wall:.0f}s, log {rel(job.log)}")
            for line in tail(job.log, 15):
                say(f"           | {line}")
            if (self.a.max_consecutive_failures and self.consec >= self.a.max_consecutive_failures
                    and not self.stop):
                self.stop = f"{self.consec} consecutive failures"
                say(f"[pool] no new launches ({self.stop}); running jobs finish, then the driver exits")
        self.tier_done(job.tier)

    def skip(self, job, why):
        self.skipped += 1
        self.tier_left[job.tier] -= 1
        say(f"[pool] skip {job.spec.name} {job.g[0]}/{job.g[1]}: {why}")
        self.tier_done(job.tier)

    def tier_done(self, ti):
        if self.tier_left[ti] == 0 and self.tier_n[ti]:
            say(f"[pool] tier '{self.ctx.tier_labels[ti]}' finished: {self.tier_ok[ti]}/{self.tier_n[ti]} ok")

    def loop(self):
        next_launch, last_status = 0.0, time.time()
        try:
            while True:
                if (self.queue and self.stop is None and len(self.running) < self.a.procs
                        and time.time() >= next_launch):
                    if self.a.limit is not None and self.launched >= self.a.limit:
                        self.stop = f"--limit {self.a.limit} reached"
                    else:
                        job = self.queue.popleft()
                        if os.path.exists(job.rjson):
                            self.skip(job, "results.json appeared since scheduling")
                            continue
                        if self.ctx.tree_changed():
                            self.queue.appendleft(job)
                            self.stop = (f"{self.ctx.tdir_rel} changed after scheduling; shard indices "
                                         "would now address other glyphs")
                            say(f"[pool] ERROR {self.stop}")
                        else:
                            self.launch(job)
                            next_launch = time.time() + self.a.stagger
                            continue
                for p in list(self.running):
                    rc = p.poll()
                    if rc is not None:
                        self.finish(p, rc)
                if not self.running and (self.stop is not None or not self.queue):
                    break
                if time.time() - last_status >= self.a.status_every:
                    self.status()
                    last_status = time.time()
                room = self.queue and self.stop is None and len(self.running) < self.a.procs
                time.sleep(0.2 if room else 1.0)
        except KeyboardInterrupt:
            self.interrupted = True
            say(f"\n[pool] interrupted")
        finally:
            self.shutdown()

    def shutdown(self):
        if not self.running:
            return
        prev = signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl+C must not strand children
        try:
            say(f"[pool] terminating {len(self.running)} solver process(es)")
            for p in self.running:
                try:
                    p.terminate()
                except OSError:
                    pass
            deadline = time.time() + 20
            stuck = []
            for p in self.running:
                try:
                    p.wait(timeout=max(0.1, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    p.kill()
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        stuck.append(p.pid)
                        say(f"[pool] pid {p.pid} did not exit")
            self.ctx.children_changed(stuck)
            for p, (job, t, _) in self.running.items():
                self.interrupted_jobs.append(job)
                try:
                    with open(job.log, "ab") as fh:
                        fh.write(f"\n# {stamp()} terminated by ablation_pool after {time.time() - t:.0f}s; "
                                 "re-run the driver to redo this cell\n".encode("utf-8"))
                except OSError:
                    pass
            self.running.clear()
        finally:
            signal.signal(signal.SIGINT, prev)


def shard_line(log, offset):
    try:
        with open(log, "rb") as fh:
            fh.seek(offset)
            head = fh.read(8192).decode("utf-8", "replace")
    except OSError:
        return None
    m = re.search(r"\[shard (\d+)/(\d+)\] (\d+) targets", head)
    return tuple(int(x) for x in m.groups()) if m else None


def tail(path, n):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 16000))
            return fh.read().decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []


# ── Main ─────────────────────────────────────────────────────────────────────


def default_repo(bench):
    """The clean fleet-shadow-art-main worktree beside the bench if present, else
    run_base_optimizer.py's own discovery (FLEET_SHADOW_ART, sibling clone, ~/dev)."""
    for cand in (
        os.path.join(os.path.dirname(bench), "fleet-shadow-art-main"),
        os.environ.get("FLEET_SHADOW_ART"),
        os.path.join(os.path.dirname(bench), "fleet-shadow-art"),
        os.path.expanduser("~/dev/fleet-shadow-art"),
    ):
        if cand and os.path.isdir(os.path.join(cand, "motion-aware-shadow")):
            return cand
    return os.path.expanduser("~/dev/fleet-shadow-art")


def solver_vs(repo, ref_sha):
    if not ref_sha:
        return "reference sha unknown"
    out = git(repo, "diff", "--name-only", ref_sha, "--", "motion-aware-shadow")
    if out is None:
        return f"{ref_sha} not in this clone"
    files = [f for f in out.splitlines() if f.strip()]
    if not files:
        return f"motion-aware-shadow identical to {ref_sha}"
    py = [f for f in files if f.endswith(".py")]
    core = [os.path.basename(f) for f in files if os.path.basename(f) in CORE_SOLVER]
    return (f"vs {ref_sha}: {len(files)} file(s) differ, {len(py)} .py"
            + (f", CORE CHANGED: {' '.join(core)}" if core else f", {'/'.join(CORE_SOLVER)} identical"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True,
                   help="reference study root, e.g. optimized/ablation-probe10 (relative to --bench or cwd)")
    p.add_argument("--glyphs", nargs="+", default=["dvsb36"],
                   help="presets probe10 | dvsb36 | all, and/or subset/stem items (default dvsb36)")
    p.add_argument("--glyph-order", choices=("stratified", "listed"), default="stratified",
                   help="order glyphs are completed in; stratified interleaves letters and digits so an "
                   "early stop keeps the full set's mix")
    p.add_argument("--specs", nargs="+", default=None,
                   help="variant dirs to run, in this order (default: every rebuildable dir, in paper order)")
    p.add_argument("--out-sub", default=None,
                   help="output under optimized/ (default: reference name with -probe10 replaced by the glyph set)")
    p.add_argument("--seed-from-reference", action=argparse.BooleanOptionalAction, default=True,
                   help="copy the reference's own glyph folders in instead of re-solving them")
    p.add_argument("--runs", type=int, default=None,
                   help="override seeds per glyph; mixes best-of-k with the seeded reference cells")
    p.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    p.add_argument("--speed", type=float, default=1.0,
                   help="local wall seconds per reference second, for estimates until jobs finish")
    p.add_argument("--n-workers", type=int, default=1,
                   help="optimizer threads per process off Windows (win32 always gets 1)")
    p.add_argument("--py", default=sys.executable, help="python that runs the solves")
    p.add_argument("--repo", default=None, help="fleet-shadow-art checkout")
    p.add_argument("--bench", default=_BENCH)
    p.add_argument("--limit", type=int, default=None, help="launch at most this many jobs this session")
    p.add_argument("--stagger", type=float, default=0.5, help="seconds between launches")
    p.add_argument("--status-every", type=float, default=600, help="seconds between status lines")
    p.add_argument("--max-consecutive-failures", type=int, default=5,
                   help="stop launching after this many failures in a row (0 = never); a broken env "
                   "otherwise burns through every job")
    p.add_argument("--dry-run", action="store_true", help="print the plan and estimates; write and launch nothing")
    return p.parse_args(argv)


def main(argv=None):
    global _BENCH, _DRIVER_LOG
    a = parse_args(argv)
    if a.procs < 1 or a.speed <= 0 or (a.runs is not None and a.runs < 1):
        die("--procs and --runs must be >= 1 and --speed > 0")
    bench = _BENCH = os.path.abspath(a.bench)
    run_base = os.path.join(bench, "scripts", "run_base_optimizer.py")
    if not os.path.isfile(run_base):
        die(f"no scripts/run_base_optimizer.py under --bench {bench}")
    ref_root = a.reference if os.path.isabs(a.reference) else (
        os.path.join(bench, a.reference) if os.path.isdir(os.path.join(bench, a.reference)) else a.reference)
    ref_root = os.path.abspath(ref_root).rstrip("\\/")
    if not os.path.isdir(ref_root):
        die(f"reference {a.reference!r} is not a directory (check `git ls-tree HEAD -- {a.reference}`)")
    ref_sub = os.path.basename(ref_root)

    glyphs, glabel = resolve_glyphs(a.glyphs, bench)
    subsets = SUBSETS + [s for s in dict.fromkeys(g[0] for g in glyphs) if s not in SUBSETS]
    glyphs = order_glyphs(glyphs, subsets, a.glyph_order)
    out_sub = a.out_sub or f"{re.sub(r'-probe10$', '', ref_sub)}-{glabel}"
    if not re.fullmatch(r"[\w.+-]+", out_sub) or out_sub in (".", ".."):
        die(f"--out-sub {out_sub!r} must be a plain folder name")
    out_root = os.path.join(bench, "optimized", out_sub)
    if os.path.normcase(os.path.abspath(out_root)) == os.path.normcase(ref_root):
        die(f"output optimized/{out_sub} is the reference itself; pass a different --out-sub")
    # Checked before anything is written: a folder built from another reference must not
    # receive this one's target copies and seeded cells first and refuse afterwards.
    led_path = os.path.join(out_root, "PROVENANCE.json")
    md_path = os.path.join(out_root, "PROVENANCE.md")
    led = None
    if os.path.isfile(led_path):
        with open(led_path, encoding="utf-8") as fh:
            led = json.load(fh)
        if led.get("reference") != rel(ref_root):
            die(f"optimized/{out_sub} was built from {led.get('reference')!r}, not {rel(ref_root)!r}; "
                "pick another --out-sub")
    tdir_rel = f"targets_{out_sub}"
    tdir = os.path.join(bench, tdir_rel)
    logs_root = os.path.join(bench, "logs", out_sub)
    repo = os.path.abspath(a.repo or default_repo(bench))
    if not os.path.isdir(os.path.join(repo, "motion-aware-shadow")):
        die(f"--repo {repo} has no motion-aware-shadow/")

    scan = scan_reference(ref_root)
    specs, skipped = build_specs(a, bench, ref_root, ref_sub, scan)
    if not specs:
        die(f"nothing rebuildable under {rel(ref_root)}")
    tiers = group_tiers(specs)
    runs_for = lambda s: a.runs if a.runs is not None else s.cli["runs"]  # noqa: E731

    # Children: one render thread each, BLAS pinned so 22 processes do not each spawn 24 threads.
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    if sys.platform == "win32":
        env.pop("MUJOCO_GL", None)  # egl does not exist on Windows; WGL is the default
        env.pop("PYOPENGL_PLATFORM", None)  # same: an egl/osmesa value left from a Linux habit breaks WGL
        nw = 1
    else:
        env["MUJOCO_GL"] = "egl"
        nw = a.n_workers
    sha = git(repo, "rev-parse", "--short", "HEAD") or "unknown"
    if git(repo, "status", "--porcelain", "--untracked-files=no", "--", "motion-aware-shadow"):
        sha += "-dirty"
    ref_shas = sorted({s.budget.get("sha") for s in specs if s.budget.get("sha")})
    solver_note = "; ".join(solver_vs(repo, x) for x in ref_shas) or "reference sha unknown"

    # ── Target tree ──
    add, bad = [], []
    for sub, stem in glyphs:
        src = os.path.join(bench, "targets_grounded", sub, stem + ".png")
        dst = os.path.join(tdir, sub, stem + ".png")
        if is_lfs_pointer(src):
            bad.append(f"{rel(src)} is an unsmudged git-lfs pointer (git lfs pull --include=\"{rel(src)}\")")
        elif not os.path.exists(dst):
            add.append((sub, stem + ".png"))
        elif not same_bytes(src, dst):
            bad.append(f"{rel(dst)} differs from {rel(src)}; cells may already be solved on it, so it is "
                       "not replaced")
    if bad:
        die("target tree:\n  " + "\n  ".join(bad))
    tree = target_jobs(tdir, subsets, add)
    n_shards = len(tree)
    shard = {g: i for i, g in enumerate(tree)}
    if len(shard) != n_shards:
        die(f"{tdir_rel} lists a target twice: {[g for g in tree if tree.count(g) > 1]}")
    strays = [f"{s}/{t}" for s, t in tree if (s, t) not in set(glyphs)]

    # ── Seeding plan and cell states ──
    copies, pointers = [], []
    for s in specs:
        if a.seed_from_reference:
            s.seed_cells, cp, pt = plan_seed(s, out_root, glyphs)
            copies += cp
            pointers += pt
    jobs, table, mism, runs_seen, corrupt = [], [], collections.defaultdict(list), collections.defaultdict(set), []
    for ti, (label, tspecs) in enumerate(tiers):
        row = dict(label=label, variants=len(tspecs), cells=0, disk=0, seeded=0, todo=0, ref_s=0.0)
        for g in glyphs:
            for s in tspecs:
                row["cells"] += 1
                rj = os.path.join(out_root, s.name, g[0], g[1], "results.json")
                if os.path.exists(rj):
                    try:
                        with open(rj, encoding="utf-8") as fh:
                            r = json.load(fh)
                    except (OSError, ValueError):
                        corrupt.append(rj)
                    else:
                        want, got = expected_record(s.cli), recorded(r)
                        runs_seen[s.name].add(got.pop("n_base_runs"))
                        want.pop("n_base_runs")
                        if diff(want, got):
                            mism[s.name].append(f"{g[0]}/{g[1]}: " + "; ".join(diff(want, got)[:3]))
                        row["disk"] += 1
                        continue
                elif g in s.seed_cells:
                    row["seeded"] += 1
                    runs_seen[s.name].add(s.cli["runs"])
                    continue
                ref_s = s.ref_seconds * runs_for(s) / s.cli["runs"]
                jobs.append(Job(ti, s, g, rj, os.path.join(logs_root, s.name, f"{g[0]}_{g[1]}.log"), ref_s, shard[g]))
                row["todo"] += 1
                row["ref_s"] += ref_s
        table.append(row)

    # ── Report ──
    if not a.dry_run:
        os.makedirs(logs_root, exist_ok=True)
        _DRIVER_LOG = os.path.join(logs_root, "pool.log")
        say(f"\n# {stamp()} {cmdline([sys.executable] + sys.argv)}")
    n_by_sub = collections.Counter(g[0] for g in glyphs)
    say(f"[pool] reference  {rel(ref_root)}: {len(scan)} variant dirs, {len(specs)} scheduled, {len(skipped)} skipped")
    say(f"[pool] glyphs     {glabel}: {len(glyphs)} (" + ", ".join(f"{k} {v}" for k, v in n_by_sub.items())
        + f"), {a.glyph_order} order: " + " ".join(g[1].split("_")[0] for g in glyphs[:40])
        + (" ..." if len(glyphs) > 40 else ""))
    say(f"[pool] output     optimized/{out_sub}   targets {tdir_rel}   logs logs/{out_sub}")
    say(f"[pool] solver     {a.py}")
    say(f"[pool]            fleet-shadow-art {repo} @ {sha} ({solver_note})")
    say(f"[pool] children   {a.procs} processes, --n-workers {nw}, "
        + ("MUJOCO_GL unset (win32)" if sys.platform == "win32" else "MUJOCO_GL=egl")
        + ", OMP/MKL/OPENBLAS_NUM_THREADS=1")
    differ = [f"{s.name} (best-of-{s.cli['runs']})" for s in specs if a.runs is not None and a.runs != s.cli["runs"]]
    if differ:
        say(f"[pool] WARNING    --runs {a.runs}: new cells are best-of-{a.runs} but the reference cells seeded next to "
            f"them are not, in {', '.join(differ)}. Equalise with the metrics script's --k.")
    say("")
    say("variants (reference glyphs, mean reference seconds per glyph, reference machine@sha, flags):")
    for label, tspecs in tiers:
        say(f"  [{label}]")
        for s in tspecs:
            who = f"{s.budget.get('machine', '?')}@{s.budget.get('sha', '?')}"
            say(f"    {s.name:<24s} {len(s.ref_cells):>3d} ref {s.ref_seconds:5.0f}s  {who:<18s} "
                + " ".join(cli_flags(s.cli, runs_for(s))))
            if s.how != f"results.json x{len(s.ref_cells)}, all agree" or s.notes:
                say(f"      rebuilt from {s.how}")
            for n in s.notes:
                say(f"      note: {n}")
    for name, why in skipped:
        say(f"  SKIPPED {name}: {why}")
    say("")
    check = (crosscheck_virtual(run_base, tdir, tdir_rel, subsets, add, tree) if a.dry_run
             else "deferred until the target tree is built")
    say(f"shard index = position in run_base_optimizer's job list over {tdir_rel} "
        f"(--subsets {' '.join(subsets)}; sorted .png), --num-shards {n_shards}")
    say(f"  self-check {check}")
    for g in list(dict.fromkeys(glyphs[:4] + glyphs[-2:])):
        say(f"  {g[0] + '/' + g[1]:<42s} --shard {shard[g]:>3d} --num-shards {n_shards}")
    if strays:
        say(f"  note: {len(strays)} PNG(s) in {tdir_rel} are not in this glyph set (counted in --num-shards, "
            f"never scheduled): {' '.join(strays[:6])}{' ...' if len(strays) > 6 else ''}")
    say("")
    if a.seed_from_reference:
        n_seed = sum(r["seeded"] for r in table)
        say(f"seed from reference: {n_seed} new cells, {len(copies)} files to copy"
            + (f"; {len(pointers)} reference PNG(s) are unsmudged LFS pointers, so a real run REFUSES to start "
               f"until: git lfs pull --include=\"{rel(ref_root)}/**\"" if pointers else ""))
    for name, lines in mism.items():
        say(f"WARNING {name}: {len(lines)} existing results.json in optimized/{out_sub} do not match the "
            f"rebuilt condition (kept, not re-solved): {lines[0]}")
    for name, rs in runs_seen.items():
        if len(rs) > 1:
            say(f"WARNING {name}: cells mix best-of-{sorted(rs)}; equalise with the metrics script's --k")
    if corrupt:
        say(f"WARNING {len(corrupt)} unreadable results.json "
            + ("would be set aside and re-solved" if a.dry_run else "set aside as results.json.corrupt and re-solved")
            + f": {rel(corrupt[0])}{' ...' if len(corrupt) > 1 else ''}")

    fin = makespan([j.ref_s * a.speed for j in jobs], a.procs)
    done_by, t = [], 0.0
    for ti in range(len(tiers)):
        t = max([t] + [f for j, f in zip(jobs, fin) if j.tier == ti])
        done_by.append(t)
    say(f"{'tier':<24s} {'variants':>8s} {'cells':>6s} {'on disk':>7s} {'seeded':>6s} {'to solve':>8s} "
        f"{'ref-h':>7s} {'pool-h':>7s} {'done by':>8s}")
    for row, db in zip(table, done_by):
        say(f"{row['label']:<24s} {row['variants']:>8d} {row['cells']:>6d} {row['disk']:>7d} {row['seeded']:>6d} "
            f"{row['todo']:>8d} {row['ref_s'] / 3600:>7.2f} {row['ref_s'] * a.speed / a.procs / 3600:>7.2f} {db / 3600:>7.2f}h")
    tot = {k: sum(r[k] for r in table) for k in ("variants", "cells", "disk", "seeded", "todo", "ref_s")}
    say(f"{'total':<24s} {tot['variants']:>8d} {tot['cells']:>6d} {tot['disk']:>7d} {tot['seeded']:>6d} "
        f"{tot['todo']:>8d} {tot['ref_s'] / 3600:>7.2f} {tot['ref_s'] * a.speed / a.procs / 3600:>7.2f} "
        f"{(done_by[-1] if done_by else 0) / 3600:>7.2f}h")
    say(f"  ref-h = reference seconds of the cells to solve; pool-h = ref-h x --speed {a.speed:g} / --procs {a.procs}; "
        "done by = the pool simulated in job order (tiers in sequence, glyph-major inside each)")
    if a.limit is not None:
        say(f"  --limit {a.limit}: only the first {min(a.limit, len(jobs))} of these jobs launch this session")
    if jobs:
        j = jobs[0]
        example = [a.py, run_base, "--repo", repo, "--bench", bench, "--targets-dir", tdir_rel, "--subsets",
                   *subsets, "--out", os.path.join(out_root, j.spec.name), "--shard", str(j.shard),
                   "--num-shards", str(n_shards), "--n-workers", str(nw), *cli_flags(j.spec.cli, runs_for(j.spec))]
        say(f"first job: {j.spec.name} {j.g[0]}/{j.g[1]} -> {rel(j.log)}\n  {cmdline(example)}")
    if a.dry_run:
        say("\n[pool] dry run: nothing written, nothing launched")
        return 0

    # ── Real run ──
    if pointers:
        die(f"{len(pointers)} reference file(s) are unsmudged git-lfs pointers (e.g. {rel(pointers[0])}). Seeding "
            "would copy results.json without its PNG and every metrics script would count that cell as done. "
            f"Run: git lfs pull --include=\"{rel(ref_root)}/**\"")
    lock = PoolLock(os.path.join(out_root, "POOL.lock"))
    lock.acquire()
    # With the children's env and cwd, importing the solver modules themselves: a MUJOCO_GL=egl
    # left in this shell is removed for them and must not fail this, a missing scipy must.
    pre = subprocess.run([a.py, "-c", "import cma, PIL, scipy.ndimage, mujoco, loss, renderer, optimizer"],
                         capture_output=True, text=True, timeout=300, env=env,
                         cwd=os.path.join(repo, "motion-aware-shadow"))
    if pre.returncode:
        die(f"--py {a.py} cannot import the solver's dependencies:\n{pre.stderr[-1500:]}")
    for sub in subsets:
        os.makedirs(os.path.join(tdir, sub), exist_ok=True)
    for sub, fn in add:
        shutil.copy2(os.path.join(bench, "targets_grounded", sub, fn), os.path.join(tdir, sub, fn))
    if target_jobs(tdir, subsets) != tree:
        die(f"{tdir_rel} does not list as planned after copying; another process is editing it")
    say(f"shard self-check {crosscheck_shards(run_base, bench, tdir_rel, subsets, tree)}")
    for rj in corrupt:
        os.replace(rj, rj + ".corrupt")
    for s_, d_ in copies:
        os.makedirs(os.path.dirname(d_), exist_ok=True)
        shutil.copy2(s_, d_)

    ref_rel = rel(ref_root)
    if led is None:
        led = {"reference": ref_rel, "variants": [], "reference_budgets": {}, "sessions": [], "cells": {}}
        if os.path.isfile(md_path):
            # Written by some other launcher: its record is the only one those cells have, so it is
            # carried verbatim into the regenerated file instead of being overwritten.
            with open(md_path, encoding="utf-8", errors="replace") as fh:
                led["adopted_provenance"] = fh.read()
            say(f"[pool] note: {rel(md_path)} was not written by this pool; kept verbatim at its end")
    led["variants"] = list(dict.fromkeys(led["variants"] + [s.name for s in specs]))
    for s in specs:
        led["reference_budgets"][s.name] = {
            "machine": s.budget.get("machine"), "sha": s.budget.get("sha"), "generated": s.budget.get("generated"),
            "how": s.how, "flags": " ".join(cli_flags(s.cli)), "reference_cells": len(s.ref_cells)}
        cells = led["cells"].setdefault(s.name, {})
        for g in s.seed_cells:
            key = f"{g[0]}/{g[1]}"
            if cells.get(key, {}).get("source") != "local":
                cells.setdefault(key, {"source": "reference", "machine": s.budget.get("machine"),
                                       "sha": s.budget.get("sha"), "generated": s.budget.get("generated"),
                                       "copied": stamp()})
    session = {"id": len(led["sessions"]) + 1, "started": stamp(), "node": platform.node(),
               "platform": f"{sys.platform} {platform.platform()}", "repo": repo, "sha": sha,
               "solver_vs_reference": solver_note, "python": a.py, "procs": a.procs,
               "command": cmdline([sys.executable] + sys.argv)}
    warned = []

    def save():
        try:
            os.makedirs(out_root, exist_ok=True)
            write_atomic(led_path, json.dumps(led, indent=1))
            write_atomic(md_path, render_provenance(led, out_root, out_sub, subsets))
        except OSError as e:
            if not warned:
                warned.append(e)
                say(f"[pool] WARNING could not write provenance ({e}); solves continue")

    def session_started():
        if session not in led["sessions"]:
            led["sessions"].append(session)
            save()

    def record_local(job, wall, r):
        led["cells"].setdefault(job.spec.name, {})[f"{job.g[0]}/{job.g[1]}"] = {
            "source": "local", "session": session["id"], "finished": stamp(), "wall_s": round(wall, 1),
            "solve_s": r.get("seconds"), "best_iou_vs_original": r.get("best_iou_vs_original")}
        save()

    save()
    ctx = types.SimpleNamespace(
        py=a.py, run_base=run_base, repo=repo, bench=bench, tdir_rel=tdir_rel, subsets=subsets,
        out_root=out_root, n_shards=n_shards, nw=nw, env=env, runs_for=runs_for,
        tier_labels=[label for label, _ in tiers], record_local=record_local,
        session_started=session_started, tree_changed=lambda: target_jobs(tdir, subsets) != tree,
        children_changed=lock.children)
    def interrupt(*_):
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGBREAK"):  # treat like Ctrl+C so children are stopped, not orphaned
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupt)
    say(f"\n[pool] {stamp()} launching {len(jobs)} job(s) on {a.procs} process slot(s); Ctrl+C stops all cleanly")
    pool = Pool(a, jobs, ctx)
    pool.loop()
    lock.release()  # children are stopped by now; atexit still covers the die() paths above
    save()
    left = len(pool.queue)
    say(f"\n[pool] {stamp()} ok {pool.ok}, failed {pool.failed}, skipped {pool.skipped}, "
        f"interrupted {len(pool.interrupted_jobs)}, not launched {left}"
        + (f" ({pool.stop})" if pool.stop and left else "")
        + f"; {(time.time() - pool.t0) / 3600:.2f}h; {pool.ratio()[1]}")
    for job, why in pool.failures:
        say(f"  FAILED {job.spec.name} {job.g[0]}/{job.g[1]}: {why}  ({rel(job.log)}); re-run the driver to retry")
    say(f"[pool] provenance: {rel(md_path)}")
    return 130 if pool.interrupted else (1 if pool.failed or (pool.stop and left and "limit" not in pool.stop) else 0)


if __name__ == "__main__":
    sys.exit(main())
