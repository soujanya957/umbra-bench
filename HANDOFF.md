# Handoff — overnight sweep on `distort_target`

Written 2026-09-01. Both repos are on branch `distort_target`, nothing pushed.

**Pre-flight steps 1–3 are DONE (2026-09-01 00:30, machine `ivy`), and the
small-budget sweep is running.** They are kept below as the record of what was
done and why. Jump to §4 for the run that is in flight and §5 for the morning.

## 1. Cherry-pick the renderer-clone leak fix — DONE

`d964173` cherry-picked onto `fleet-shadow-art/distort_target` as **`bccd8c1`**.
It auto-merged with `adfda3a`, which had touched the same file; both survived and
were verified by hand:

- `_RendererPool._clones` / `close()` present, `shutdown(wait=True)` at both call
  sites (the FD path and the staged path);
- the win32 `n_workers` clamp still present on both paths.

Note the two interact: on win32 the clamp sets `_n_workers = 1`, so `_pool` is
never built and the leak fix is inert for *this* run. Both `close()` call sites
are `if ... is not None`-guarded, so that is safe rather than a crash — verified
before launching. The fix still matters for any EGL run off this branch.

## 2. Stage the grounded targets — DONE

Committed as **`cdb65e8`**, 542 LFS pointers. git-lfs 3.7.1 is present on `ivy`.
Re-verified before the run: 40/40 sampled targets rest on the bottom frame row.

## 3. Repo hygiene — DONE

`core.autocrlf=true` set on `umbra-bench` (status went from ~12,900 phantom
modifications to clean); `_stale_locks/` removed from both repos.

A stale `index.lock` (fleet-shadow-art) and `HEAD.lock` (umbra-bench) both had to
be cleared during this session — 0 bytes, no live `git.exe`. **GitHub Desktop is
running on this machine and polls both repos**, which is the likely source. If a
git command reports a lock, check `ls -la .git/*.lock` and the process list
before removing anything.

## 3b. Windows blockers found by smoke-testing — FIXED, commit `b112986`

A one-target smoke run was done before committing the night to a sweep. It hit
three failures in a row, each of which aborted in the first seconds — the night
would have been lost to any one of them:

| failure | cause | fix |
| --- | --- | --- |
| `UnicodeEncodeError: '→'` | solver prints `→`/`×`; console is cp1252 | `PYTHONIOENCODING=utf-8` |
| `AttributeError: os.uname` | posix-only, in `write_budget_md` | `platform.node()` |
| `UnicodeEncodeError: 'σ'` | `BUDGET.md` holds `σ`; bare `open(...,"w")` is cp1252 | explicit `encoding="utf-8"` |

The last two are fixed in this repo (`b112986`, which also fixes the three reads
and writes in `compute_metrics.py`). The first cannot be fixed from here — the
prints are in `fleet-shadow-art` — so **any caller must export
`PYTHONIOENCODING=utf-8`**. Setting `PYTHONUTF8=1` as well is belt-and-braces: it
makes `open()` default to UTF-8 process-wide and covers the ~25 other bare
`open()` calls in `scripts/` that this run did not happen to touch.

## 4. The sweep — LAUNCHED 2026-09-01 00:27, 12 shards

Running now as 12 detached processes on `ivy`. Launcher:
`scratchpad/launch_sweep.ps1` (copy it into `scripts/` if it earns its keep).

```powershell
powershell -File launch_sweep.ps1 -NumShards 12 `
  -Out "optimized/small-budget-grounded" -Tag small
```

which runs, per shard, the §4 command as originally written, plus
`--num-shards 12 --shard $s`, with `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1`
exported (see §3b) and a 400 ms stagger so the shards do not race writing the
reach-map cache.

- logs: `results/sweep_logs/small-shard{0..11}.{log,err}` (`results/` is gitignored)
- progress: `grep -h best= results/sweep_logs/small-shard*.log | wc -l` out of 542
- to stop: `Get-Process python | Where-Object {$_.Path -like "*fleet-shadow*"} | Stop-Process`

Four things that are easy to get wrong here:

- **`--n-workers 0`, never 2.** `BUDGET.md` for the existing sweeps records 2,
  which was true of the EGL machine that produced them. On Windows wgl contexts
  are not thread-safe and the failure is silent — renders corrupt mid-run and IoU
  collapses to 0. Confirmed correct for this run: the freshly written
  `optimized/small-budget-grounded/BUDGET.md` reads `0 requested / 1 effective`.
- **Export `PYTHONIOENCODING=utf-8`** or the run dies in the first second (§3b).
- **`--targets-dir targets_grounded` must be repeated at metrics time** (§5),
  or `ref=original` scores against the centred tree and the numbers are nonsense.
- **`--fit-target` requires its own `--out`.** Sharing a folder with a no-fit run
  makes the resume check skip targets it should solve.

Resume is automatic: a target with a `results.json` is skipped unless `--force`,
so an interrupted run continues where it stopped, and re-running the launcher is
the way to pick up after a crash.

### Time budget — measured on `ivy`, replacing the earlier estimate

The estimate below §4 in the previous revision (65 s/target, ~4.9 h at 4 shards)
came from `seconds` in sweeps produced on a slower machine. Measured here:

| | measured |
| --- | --- |
| per target, per shard, 12 shards contending | **73 s** |
| small budget, 542 targets, 12 shards | **~0.9 h** |
| big budget, extrapolated at 2.55x | ~2.4 h |

**The GPU is the bottleneck, not the core count.** At 12 shards `nvidia-smi`
reads 95% utilisation and only 5.7 GB of 16.3 GB. Adding shards past ~12 buys
little; the total is ~10,800 GPU-seconds however it is divided. `ivy` has 24
cores, so cores are not the constraint and the old advice to set `--num-shards`
to the core count does not apply.

Both budgets therefore fit in one night, which the earlier estimate said was
impossible.

## 4b. Result — small-budget-grounded vs small-budget-fitted, 478 shared targets

The sweep finished 2026-09-01 01:24, 542/542, in **0.92 h** on 12 shards. Both
CSVs are in `results/`. Paired on the 478 target ids the two sweeps share
(grounded has 542; the extra 64 are the `_v2` targets the centred sweep never
ran):

| ref | grounded | centred | delta | better |
| --- | --- | --- | --- | --- |
| `original` | **0.3504** | 0.1999 | **+0.1506** | **441/478** |
| `shown` | 0.7523 | 0.7345 | +0.0179 | 285/478 |

**+75% relative on IoU against the authored target, improving 92% of targets.**

Every subset improves, and the spread is itself informative:

| subset | n | grounded | centred | delta | better |
| --- | --- | --- | --- | --- | --- |
| vehicles | 30 | 0.5261 | 0.1144 | **+0.4117** | 30/30 |
| objects | 95 | 0.4076 | 0.1886 | +0.2190 | 89/95 |
| animals | 76 | 0.3914 | 0.1856 | +0.2058 | 69/76 |
| hand_shadow | 10 | 0.2825 | 0.1326 | +0.1499 | 10/10 |
| letters_lower | 77 | 0.2988 | 0.2010 | +0.0978 | 74/77 |
| abstract | 76 | 0.3391 | 0.2596 | +0.0795 | 65/76 |
| letters_upper | 74 | 0.2709 | 0.1918 | +0.0792 | 69/74 |
| digits | 30 | 0.2482 | 0.1916 | +0.0566 | 26/30 |
| figures | 10 | 0.4160 | 0.3604 | +0.0556 | 9/10 |

`vehicles` gains most and wins 30/30 — consistent with the earlier reach-map
finding that vehicles are the subset that wants to move *up* (−11 px). Centred
placement pushed them hardest in the wrong direction, so grounding recovers the
most there. `digits` and `figures` gain least, and digits wanted +17 px, the
opposite sign. The effect tracks the per-subset placement remainder rather than
being uniform, which is what a placement effect should look like.

**`ref=shown` moved too, by +0.0179.** An earlier partial on `abstract` alone had
it at −0.006, i.e. flat, which made a tidier story: gain entirely in placement,
solver untouched. On the full set that is not quite right — the solver also does
marginally better on grounded targets. The attribution still holds, because the
`ref=original` gain is ~8x larger, but it should be stated as "dominated by
placement", not "purely placement".

### Two things to resolve before this is written up

1. **`at_bound` came in at 14.2%**, against the **1.1%** predicted by the
   empirical reach-map sweep over the same (scale, dy) grid, and 60% on the
   centred sweeps. The direction is right and the improvement is large, but the
   reach-map estimate was optimistic by ~13x. It is a more permissive proxy than
   an actual solve; worth understanding before either figure is quoted.
2. `betti_error` is **0.40 grounded vs 0.49 centred** and `cldice` 0.517 vs
   0.299 — the topology metrics move the same way as IoU, so the result is not
   an IoU artefact. Worth a line in the writeup.

## 5. Metrics

```bash
conda activate umbra-bench          # see SETUP.md
python scripts/compute_metrics.py \
  --results optimized/small-budget-grounded \
  --targets-dir targets_grounded
```

New columns on this branch, from the fit: `clip_frac`, `touches_edge`, `at_bound`.
Check `at_bound` first — it should now be near 0%, against 60% on the centred
sweeps. **Ignore `touches_edge` for grounded targets**: they rest on the bottom by
construction, so it reads ~70% and means nothing here. `clip_frac` is the one that
carries information in this condition.

## 6. CLIP / recognizability — not written yet

`semantic_metrics.py` is a library: no `__main__`, no argparse, and nothing in the
repo calls `clip_retrieval`. A runner still has to be written, and it carries real
design decisions rather than just plumbing:

- **Per-subset class lists, not one global N-way.** `figures` has 2 classes and
  `letters_upper` has 26; top-1 across those is not comparable untransformed.
- **`abstract` is the null control.** Its labels are `device0`…`device9`, `bone`,
  `comma`. Its ceiling should sit near chance — if it does not, the class list is
  leaking information.
- **Render, never feed 1-bit masks.** Use `render_shadow()`; a raw mask is far
  outside CLIP's training distribution.
- **Score the target too, and report the ratio.** Without the target as a ceiling
  a low shadow score is unattributable: bad shadow, or blind judge?

This is independent of the sweep and can be written while it runs.
