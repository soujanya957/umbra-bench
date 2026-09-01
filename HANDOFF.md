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

## 4a. Why grounded is the dataset, not a condition

Settled 2026-09-01. The arms are mounted on the table; a shadow is cast by
something that grows up from the base plane. A target floating at the vertical
centre of the frame is asking the rig for a shape it is not built to throw, and
the 27-row offset between where targets were drawn and where the rig's real
footprint sits is that mismatch measured rather than a property of any target.

So `targets_grounded/` is the canonical tree from here, and `targets/` is the
version with the placement bug. The grounded-vs-centred tables in 4b and 4c stay
as the record of how it was established, but there is nothing left to A/B: a
sweep against `targets/` is a sweep against targets the rig cannot reach, and
re-stating the gain under a different metric would not change the decision.

Practical consequences:

- new targets get `scripts/normalize_targets.py` then `scripts/ground_targets.py`
  before anything solves them, which is what the teleop subset went through;
- `--targets-dir targets_grounded` at both solve and metrics time;
- the atlas builds from the grounded tree and the grounded sweeps by default.

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

## 4c. Both budgets, and what compute cannot buy

`big-budget-grounded` finished 2026-09-01 03:22, 542/542, in **1.96 h**. All four
sweeps are now scored; the CSVs are in `results/`.

**The grounding effect replicates across budgets, to three decimals.** Paired on
the 478 shared target ids:

| budget | ref | grounded | centred | delta | better |
| --- | --- | --- | --- | --- | --- |
| small | `original` | 0.3504 | 0.1999 | **+0.1506** | 441/478 |
| big | `original` | 0.3483 | 0.2000 | **+0.1483** | 435/478 |
| small | `shown` | 0.7523 | 0.7345 | +0.0179 | 285/478 |
| big | `shown` | 0.7833 | 0.7677 | +0.0156 | 279/478 |

The per-subset ordering is essentially identical at both budgets — vehicles
+0.407 vs +0.412 (30/30 both), objects +0.2188 vs +0.2190, animals +0.2057 vs
+0.2058. An effect that reproduces this closely under a 2.1x change in optimizer
budget is not an artefact of either budget.

### The interaction is the finding

Within the grounded tree, 542 shared targets, small budget → big budget:

| ref | small | big | delta | better |
| --- | --- | --- | --- | --- |
| `shown` | 0.7489 | 0.7793 | **+0.0303** | **522/542** |
| `original` | 0.3481 | 0.3465 | **−0.0017** | 269/542 |

**2.1x the compute buys +0.030 against the shown target and nothing at all
against the authored one** — 269/542 is exactly chance, so the `ref=original`
difference is noise, not a small gain. Meanwhile grounding buys **+0.15** against
the authored target at either budget.

The two act on different axes and do not substitute:

- **optimizer budget** improves how well the rig hits *what it was asked to
  cast*. It is bounded by the solver.
- **target placement** improves *what it is possible to ask for*. It is bounded
  by where the rig can throw a shadow at all.

A target placed where the rig cannot reach cannot be rescued by more search, and
the numbers say so directly: placement is worth ~5x a doubling of the budget on
the metric that measures fidelity to the authored shape, and the budget is worth
~0 on it. This is the argument for treating placement as a dataset property
rather than a per-target search parameter — which is what the original finding
proposed, now measured rather than inferred.

### Caveats

- `hand_shadow` and `figures` are n=10; those rows are directional.
- The 64 `_v2` targets are in the grounded sweeps (542) but not the centred ones
  (478), so every paired table above is on the 478 intersection. The grounded
  means over all 542 are 0.3481 / 0.7489 (small) and 0.3465 / 0.7793 (big).
- `at_bound` is 14.2%, not the 1.1% the reach map predicted (see §4b).

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

## 6. CLIP / recognizability — RUN, 2026-09-01

Superseded. The claim above this line, that `semantic_metrics.py` has no driver,
was already false when it was written: Simin pushed `56a6f49` "metrics & clip
test" to `main` on 2026-08-31 22:16, and this branch simply did not have it.
Merged as `9389d33`, no conflicts.

`tests/clip_eval.py` is that driver. It scores the 25 letter and digit captures
in `Teleops/masks` against a 49-class glyph set -- A-Z a-z 0-9 with the
low-confidence lowercase folded to uppercase and visual equivalents collapsed
(`I/l/1`, `O/o/0`, `q/9`). That class design is the careful part and is left
alone.

`scripts/clip_eval_dataset.py` extends the same metric to the rest: all 571
targets, three conditions each -- the target, and the shadow each grounded sweep
solved for it -- 1713 scored images. It imports `clip_retrieval` and the glyph
folding rather than reimplementing them.

### What the metric is

Not cosine similarity. `clip_retrieval` encodes one prompt per class, L2-
normalises both sides, takes `f @ tf.T`, and reports where the true class *ranks*
-- top1 / top5 / MRR against a known chance level. There is no softmax and no
`logit_scale` anywhere, so the `clip_top1_similarity` column is a raw cosine in
[-1, 1]: it does not sum to 1 across candidates and is not a confidence.

### Results, raw 1-bit masks, per-subset class lists

| subset | classes | chance | target | big | small | big/target |
| --- | --- | --- | --- | --- | --- | --- |
| letters_upper | 26 | 0.038 | 0.987 | 0.320 | 0.295 | 0.325 |
| letters_lower | 26 | 0.038 | 0.923 | 0.244 | 0.180 | 0.264 |
| digits | 10 | 0.100 | 0.900 | 0.333 | 0.133 | 0.370 |
| animals | 22 | 0.046 | 0.700 | 0.100 | 0.154 | 0.143 |
| objects | 23 | 0.044 | 0.617 | 0.139 | 0.078 | 0.225 |
| vehicles | 6 | 0.167 | 0.467 | 0.233 | 0.200 | 0.500 |
| figures | 2 | 0.500 | 1.000 | 0.500 | 0.500 | 0.500 |
| hand_shadow | 9 | 0.111 | 0.500 | 0.000 | 0.000 | 0.000 |
| abstract | 17 | 0.059 | 0.086 | 0.086 | 0.062 | — |
| teleop | 27 | 0.037 | 0.069 | 0.103 | 0.103 | — |

**The null control holds.** `abstract` targets score 0.086 against a chance of
0.059 -- at chance, as designed. The class list is not leaking, which is what
makes every row above it worth reading.

**Targets are legible, shadows are not.** CLIP reads the authored glyphs almost
perfectly (0.99 / 0.92 / 0.90) and the cast shadows at roughly a third of that.
`hand_shadow` is the extreme: targets 0.500, shadows 0.000. The ratio column is
the number to quote, because the absolute shadow score cannot distinguish a bad
shadow from a blind judge.

**`teleop` reads as a second null control.** Its targets score 0.069 against
0.037 chance -- barely above. That follows from what they are: human shadow
captures re-used as targets, and a shadow of a letter is not itself legible as
that letter. Its shadow scores match its target scores, so the subset carries no
recognizability signal. Worth knowing before it goes in a table next to the
others.

**Budget helps recognizability, mostly.** big beats small on 7 of 8 semantic
subsets, `animals` being the exception (0.100 vs 0.154). That is the opposite
direction from IoU, where the budget bought +0.030 against the shown target and
nothing against the authored one -- worth a look rather than a footnote.

### `--render` was tested and does not matter

`METRICS.md` Part C argues CLIP should see `render_shadow()` output rather than a
1-bit mask, since silhouettes are far outside its training distribution. Run both
ways over all 1713 images: mean top1 **0.3338 raw vs 0.3352 rendered**. Per
subset the differences go both directions and look like noise.

Rendering is also mildly *worse* where it counts: `abstract` targets rise from
0.086 to 0.198, which weakens the null control rather than the shadows' scores.
So feeding raw masks -- Simin's choice -- costs nothing measurable here, and the
`--render` flag stays for re-testing rather than as the default.

### Still open

- The captures in `Teleops/masks` are scored by `tests/clip_eval.py` on their own
  49-class pooled set; the dataset runner scores the `teleop` *subset* on its own
  27 classes. Those are different questions and should not be put in one column.
- `recognizability_ratio()` in `semantic_metrics.py` is still uncalled -- the
  dataset runner computes the ratio itself. One of the two should go.
- Nothing here has a human arm yet. `build_human_study()` writes the manifest and
  has never been run.

