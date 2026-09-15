# Running the evaluation locally

Verified 2026-09-01 on Windows (`C:\Users\hexia\Documents\GitHub`). Three lanes,
in dependency order. Only lane 1 is needed to score a sweep that already exists.

## 0. Environment

A dedicated env, separate from `fleet-shadow`. The two share MuJoCo but not much
else, and lane 3 pulls torch, which is the half of the tree most likely to break
an env you also need for solving.

```bash
conda create -n umbra-bench python=3.11
conda activate umbra-bench
pip install -r requirements-eval.txt
```

Do not `conda install pillow` into this env. The conda build's `_imaging` DLL
conflicts with pip-installed torch on Windows (`DLL load failed while importing
_imaging`); `requirements-eval.txt` takes Pillow from pip for that reason, and
`fleet-shadow-art/environment.yml` carries the same note.

Check the install actually gave you the optional metrics rather than silently
degrading — `metrics.py` returns `None` instead of raising when `gudhi` or `POT`
is missing, so a broken install looks like a working one:

```bash
python -c "import gudhi, ot, cv2, skimage; print('topology metrics available')"
```

## 1. Metrics — scoring a sweep that already exists

```bash
python scripts/compute_metrics.py --results optimized/big-budget-fitted
```

Writes `results/metrics_big-budget-fitted.csv`, one row per (sample, reference).
Fitted sweeps emit two rows per sample: `ref=shown` (what the solver was asked to
cast) and `ref=original` (the target as authored). They are far apart — mean
0.768 vs 0.228 across this sweep — so a mean IoU quoted without its `ref` is not
a number anyone can interpret.

Roughly 0.07 s per pair, parallel over `--workers` (default: cores − 1). Add
`--align` only when you specifically want `aligned_iou`; it is ~40x slower.

To score the dataset's own captures instead of a sweep:

```bash
python scripts/compute_metrics.py --shadows --sources hand teleop optimizer
```

This reads the `shadows` slots in `metadata.jsonl`. They are still `null`, so it
prints `nothing to do` until captures land.

## 2. Optimizer sweep — `run_base_optimizer.py`

Needs `motion-aware-shadow` from the other repo. Four things bite here.

**`--repo` defaults to `~/dev/fleet-shadow-art`, which does not exist on this
machine.** Always pass it:

```bash
--repo C:/Users/hexia/Documents/GitHub/fleet-shadow-art
```

**`--n-workers` must stay 0 or 1 on Windows.** `optimizer.py` forces 1 worker on
win32 because wgl contexts are not safe across threads — parallel clone renders
corrupt frames mid-run and IoU collapses to 0 — but an explicit `--n-workers 2`
overrides that guard and reintroduces the bug. `BUDGET.md` for the existing
sweeps records `n_workers = 2`; those ran on `dutchman` under EGL. Do not copy
that number onto Windows. Use processes instead:

```bash
--num-shards 4 --shard 0   # ... and 1, 2, 3 in their own terminals
```

**The existing sweeps were built at `fleet-shadow-art @ d964173`, which is not on
`main`.** It is a single commit on `origin/renderer-clone-leak`, and it holds the
fix that closes renderer clones when a parallel section ends. Without it a long
sweep leaks EGL contexts — the commit message records a run deadlocking after
~800 solves with ~25k leaked contexts. Merge or cherry-pick it before any sweep
long enough to matter, or results will not be comparable to what is already in
`optimized/`.

**`--fit-target` demands an explicit `--out`.** It is a separate experimental
condition; sharing an output folder with a no-fit run makes the resume check skip
targets it should have solved.

Reproducing `big-budget-fitted` exactly (from its `BUDGET.md`):

```bash
python scripts/run_base_optimizer.py \
  --repo C:/Users/hexia/Documents/GitHub/fleet-shadow-art \
  --out optimized/big-budget-fitted --fit-target \
  --subsets abstract animals digits figures hand_shadow \
            letters_lower letters_upper objects vehicles \
  --runs 10 --extra-runs 5 --extra-below 0.5 \
  --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 \
  --no-adaptive-final --n-robots 3 --arm-gap 0.2 --size 128 \
  --fit-scale-min 0.35 --fit-scale-max 1.6 --fit-n-scales 14 \
  --fit-n-shifts 15 --fit-max-shift 0.22 --reach-samples 300 \
  --n-workers 0
```

Targets that already have a `results.json` are skipped unless `--force`, so this
same command resumes an interrupted sweep and, run against the current
`targets/`, solves only what is missing.

Then aggregate:

```bash
python scripts/summarize_base_optimizer.py --results optimized/big-budget-fitted
```

## 3. Recognizability — `semantic_metrics.py`

`clip_retrieval()` downloads ViT-B-32 / `laion2b_s34b_b79k` on first call. On an
NVIDIA machine install torch from the CUDA index before the rest, or pip will
resolve the CPU wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install open_clip_torch
```

Score the targets as well as the shadows every time. CLIP is weak on 1-bit
silhouettes, so a low shadow score is unattributable without the target's own
score as the ceiling; the reported number is the ratio. `abstract` is the null
control — its ceiling should sit near chance, and if it does not, the class list
is leaking.

## Layout note

`compute_metrics.py` writes only to `results/`, never back into the dataset.
`metadata.jsonl` holds attributes of *targets* only.
