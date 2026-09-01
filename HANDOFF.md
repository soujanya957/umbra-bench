# Handoff — overnight sweep on `distort_target`

Written 2026-09-01. Both repos are on branch `distort_target`, nothing pushed.
Do these in order; steps 1–3 are pre-flight and take a couple of minutes.

## 1. Cherry-pick the renderer-clone leak fix — do not skip this one

```bash
cd fleet-shadow-art          # on distort_target
git cherry-pick d964173
```

`d964173` is a single unmerged commit on `origin/renderer-clone-leak` that closes
renderer clones when a parallel section ends. `main` has never had it, and every
existing sweep in `optimized/` was produced *with* it.

It matters most for exactly the run being planned. Its commit message records a
sweep deadlocking after **~800 solves** with ~25k leaked EGL contexts and 12 GB of
GPU memory held by no live process. A 542-target sweep at `--runs 10` is **5,420
solves**. Without this, the overnight run is likely to hang partway and the night
is lost.

## 2. Stage the grounded targets (needs git-lfs)

```bash
cd umbra-bench
git add targets_grounded/    # 542 PNGs, LFS-tracked
git commit -m "targets: add the grounded target tree"
```

They are on disk but unstaged: `*.png` is LFS-tracked and git-lfs was not
available on the machine that produced them.

## 3. Two small repo hygiene fixes

```bash
git -C umbra-bench config core.autocrlf true    # else every text file reads as modified
rm -rf umbra-bench/.git/_stale_locks fleet-shadow-art/.git/_stale_locks
```

## 4. The sweep

**Run the small budget first.** It is the matched comparison against
`optimized/small-budget-fitted`, it fits in one night, and if something about the
grounded targets is wrong you find out after one night instead of three.

```bash
cd umbra-bench
python scripts/run_base_optimizer.py \
  --repo ../fleet-shadow-art \
  --targets-dir targets_grounded \
  --out optimized/small-budget-grounded --fit-target \
  --subsets abstract animals digits figures hand_shadow \
            letters_lower letters_upper objects vehicles \
  --runs 10 \
  --popsize 32 --phase1-iters 8 --phase2-iters 8 --final-iters 10 \
  --n-robots 3 --arm-gap 0.2 --size 128 \
  --fit-scale-min 0.35 --fit-scale-max 1.6 --fit-n-scales 14 \
  --fit-n-shifts 15 --fit-max-shift 0.22 --reach-samples 300 \
  --n-workers 0 --num-shards 4 --shard 0
```

Then `--shard 1`, `2`, `3` in their own terminals. Set `--num-shards` to the core
count; four is a placeholder.

Three things that are easy to get wrong here:

- **`--n-workers 0`, never 2.** `BUDGET.md` for the existing sweeps records 2,
  which was true of the EGL machine that produced them. On Windows wgl contexts
  are not thread-safe and the failure is silent — renders corrupt mid-run and IoU
  collapses to 0. The clamp added on this branch now overrides an explicit 2
  anyway, but do not ask for it.
- **`--targets-dir targets_grounded` must be repeated at metrics time** (step 5),
  or `ref=original` scores against the centred tree and the numbers are nonsense.
- **`--fit-target` requires its own `--out`.** Sharing a folder with a no-fit run
  makes the resume check skip targets it should solve.

Resume is automatic: a target with a `results.json` is skipped unless `--force`,
so an interrupted run continues where it stopped.

### Time budget

From `seconds` in the existing sweeps' `results.json`, scaled to 542 targets,
4 shards, and doubled for win32 single-threaded rendering:

| sweep | per target | 542 targets, 4 shards, win32 |
| --- | --- | --- |
| small budget | 65s | **~4.9 h** — one night |
| big budget | 166s | **~12.5 h** — more than one night at 4 shards |
| just the 64 missing `_v2`, big budget | | ~1.5 h |

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
