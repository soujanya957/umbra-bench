# atlas — the benchmark dashboard

One self-contained HTML file. Every thumbnail, metric table and teleop frame is
embedded, so it opens from disk, a URL or an email attachment with nothing
installed and no server running.

```
atlas/
  atlas.html              ← open this. the only openable file here
  build_atlas.py
  README.md
  src/
    atlas.tpl.html        ← edit this
    atlas.fragment.html   ← generated, for publishing only
```

```
python3 atlas/build_atlas.py          # -> atlas/atlas.html
python3 atlas/build_atlas.py --bare   # -> atlas/src/atlas.fragment.html
python3 atlas/build_atlas.py --check  # is the committed build current?
```

**Do not edit `atlas.html`.** It is generated. Edit `src/atlas.tpl.html` (markup,
CSS and all the view logic) and rebuild — a 6 MB document edited by hand is how a
dashboard and the data it claims to show come apart without anyone noticing.

## Why two outputs, and why the template hides in src/

`src/atlas.tpl.html` is a *fragment*: no doctype, no `<head>`. The artifact host
supplies those at publish time, so `--bare` hands it exactly what it expects.
Opened straight off disk, though, that same fragment lands in quirks mode with the
character encoding merely guessed, which turns the arrows and em-dashes in the
copy into mojibake. The default build wraps the fragment in a real document so the
committed file is one a browser can open as it stands. Both come from the same
template, so they cannot drift.

The two files that must not be opened live in `src/` because they used to sit
beside the built one, and the template — a perfectly ordinary-looking `.html` —
got opened instead, failing as `Unexpected token '_'` in the console and a blank
page. Opening the template now renders a short note telling you what to run; the
directory layout is what stops you needing it.

## Inputs

| slot | file | built by |
| --- | --- | --- |
| `__PAYLOAD__` | `results/browser_payload.json` | `scripts/_build_browser_payload.py` |
| `__TELEOP__` | `results/teleop_payload.json` | `scripts/_build_teleop_payload.py` |
| `__SEQUENCES__` | `results/sequences_payload.json` | `scripts/_build_sequences_payload.py` |

Both are parsed before splicing. A truncated payload is otherwise a syntax error
in someone's browser hours later, with no line number worth having. A missing one
is a hard failure rather than a placeholder: the build refuses instead of shipping
a page whose teleop tab is quietly empty.

`results/` is gitignored, so a fresh checkout has neither. The chain from solved
sweeps to an openable page is five commands:

```bash
python scripts/compute_metrics.py --results optimized/big-budget-grounded --targets-dir targets_grounded
python scripts/make_master_table.py        # metrics_*.csv -> master_table.csv
python scripts/_build_browser_payload.py   # + metadata.jsonl, + the target tree
python scripts/_build_sequences_payload.py # sequences.jsonl + sequence_metrics_*.csv
python atlas/build_atlas.py
```

`make_master_table.py` is the step most easily missed. `compute_metrics.py` writes
one `results/metrics_<sweep>.csv` per sweep, while the payload builder reads a
single `results/master_table.csv` with a `sweep` column, and for a long time
nothing in the repo produced it — so a checkout could compute every metric and
still not render a dashboard.

`_build_browser_payload.py --list` prints the sweeps and target trees on disk
without building anything, which is the quickest way to see what the payload could
be pointed at. It defaults to the grounded tree and the grounded sweeps;
`--targets-dir`, `--big`, `--small` and `--ref` move it.

## Plates, and the frame they are drawn in

The plate row is **overlay / shadow / target / all 3**, plus **human**. All of
them draw the target *the optimizer was actually given*, which on a
`--fit-target` sweep is not the target as authored: the fit scales it (0.832 on
average, and not once exactly 1.0 across 571 solves) and shifts it by ~14px in a
128px frame before anyone solves for it.

That matters because it is invisible and it looks like a bug. Drawing the shadow
against the authored target makes every card read as mis-registered — shapes
plainly similar, size and position plainly off — and none of that is solver
error. Worse, letting `target` show the authored shape while `overlay` compared
against the fitted one put the two plates in different coordinate systems, so
`all 3` was three views of two frames.

So `tgt = e.w || s.t`: the fitted target where one was recorded, falling back to
the authored target on an unfitted sweep, where it is exactly what was cast. The
card metrics follow the same choice — `--ref shown` is the default — so the
picture and the number are always the same comparison.

For the end-to-end number instead, with the fit counted as error, build the
payload with `--ref original`. It will not line up with the pictures, and that is
the honest result rather than a rendering fault.

`tc_iou` is the third question: trim both to their ink, match on height, centre,
compare. Position and size removed, so it is about the shape alone — 0.687, where
plain IoU against the authored target reads 0.350. It has no plate of its own; it
is in the sort menu as **shape IoU**, with **aspect error** beside it.

## Views

- **atlas** — every target with its best solved shadow, sortable by any of the
  four panel metrics, with the distribution of whatever you sorted by across the
  top. Star frames to build a sequence.
- **benchmark** — the subset table and the metric-vs-metric figures.
- **teleop** — the *pipeline*, not a second copy of the dataset: six steps from
  rig to rebuild, each with the command to run. The interactive segmenter lives
  inside step 4, for redoing a mask that came out wrong.
- **sequences** — the animated track. Previews play at the source fps and wrap
  around only when the sequence actually loops; the loop badge carries its
  provenance (`declared` is a fact from source.json, `wrap-test` a heuristic
  with its evidence beside it). A solve's mean frame IoU is only ever shown
  next to `dq_infeasible_frac`, per SEQUENCES.md; unsolved clips say so and
  print the command that changes it.
- **guide** — how to use the dashboard, plus the metric reference: range,
  direction, what each number answers, and the ways it lies when quoted alone.
  Content is checked against the metric code; when they disagree, the code is
  right.

### The `hand-cast` badge

The small badge in the corner of a card means **somebody posed the arms by hand
and cast this shape for real**, and the card is showing the robot's attempt at the
same target. Switch the plate to **human** to see the mask they produced, or use
`hand-cast only` to filter to them. 28 of the 571 cards carry it.

It is a marker, not a score. Whether the human did better is not answerable from
what was recorded: the pose was never captured, so their mask and the target share
no alignment, and a number comparing the two would be measuring the photograph
rather than the performance. The badge says a human reference exists, nothing more.

**The badge and the `teleop` subset are different things, and they do not
overlap.** The badge marks an ordinary target — a digit, a letter, an mpeg7 shape
— that happens to have a human capture attached in `shadows.teleop`. The `teleop`
subset is those captured masks re-used as targets in their own right, so the rig is
asked to cast a shape a human once cast. Every badged card sits outside the teleop
subset and every teleop-subset card is unbadged; the overlap is exactly zero.

Two captures were posed against targets the v1→v2 rescue later replaced.
`link_teleop.py` re-points those to the surviving `_v2` id, and
`_build_teleop_payload.py` now does the same. Until it did, their `sample_id`
named a target that no longer existed, so two real captures silently lost their
badge and the count read 26 instead of 28.

In the **benchmark** the captures appear as a reference-set block with per-subset
coverage and the open alignment question. A capture is attached to a target, so
that is where it belongs; a separate tab made it look like a rival dataset.

The rescue view is gone. It reviewed the sub-floor targets one at a time, and once
the v1→v2 replacement was applied it had nothing left to decide; the originals and
their old sweep results are preserved in `dropped/`, so the comparison is still
reconstructable from the data rather than from a UI.

## Saved state

The sidebar persists to `localStorage` under `uba:*` — view, sweep, plate, subset
filters, sort, starred frames. Saved state outlives the code that wrote it, so it
is reconciled with the payload on load rather than trusted:

- `st.page` is validated against `PAGES`, and `st.view` against the live plate
  list. Removing either without the guard leaves anyone who had it selected
  looking at a blank page — which is what happened when the rescue view went, and
  would have happened again when the `fit` and `centred` plates folded back into
  `overlay`.
- The subset chips are the subtler case. `uba:subs` records only what is
  *selected*, which cannot distinguish "the reader turned this off" from "this did
  not exist yet", so a newly added subset restores unpressed and its samples
  vanish — teleop looking missing rather than deselected. `uba:subsKnown` records
  what the payload offered at the time, which separates the two: deselections are
  kept, genuinely new subsets are opted into.

If a page still looks stale after a rebuild, `localStorage.clear()` in the console
is the blunt fix.

## Staleness

The atlas and benchmark views show whatever sweep produced `browser_payload.json`.
After a re-run, rebuild the payload first, then the atlas — `--check` will tell you
whether the committed `atlas.html` is behind the template and payloads it came
from.
