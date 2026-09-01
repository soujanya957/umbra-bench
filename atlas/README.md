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

Both are parsed before splicing. A truncated payload is otherwise a syntax error
in someone's browser hours later, with no line number worth having.

## Views

- **atlas** — every target with its best solved shadow, sortable by any of the
  four panel metrics, with the distribution of whatever you sorted by across the
  top. Star frames to build a sequence.
- **benchmark** — the subset table and the metric-vs-metric figures.
- **teleop** — the *pipeline*, not a second copy of the dataset: six steps from
  rig to rebuild, each with the command to run. The interactive segmenter lives
  inside step 4, for redoing a mask that came out wrong.

The hand-cast captures themselves surface where the targets are, not in a tab of
their own. In the **atlas** a target somebody posed by hand carries a `hand-cast`
badge, the **human** plate shows the mask they produced, and `hand-cast only`
filters to them. In the **benchmark** they appear as a reference-set block with
per-subset coverage and the open alignment question. A capture is attached to a
target, so that is where it belongs; a separate tab made it look like a rival
dataset.

The rescue view is gone. It reviewed the sub-floor targets one at a time, and once
the v1→v2 replacement was applied it had nothing left to decide; the originals and
their old sweep results are preserved in `dropped/`, so the comparison is still
reconstructable from the data rather than from a UI.

## Saved state

The sidebar persists to `localStorage` under `uba:*` — view, sweep, plate, subset
filters, sort, starred frames. Saved state outlives the code that wrote it, so
`st.page` is validated against `PAGES` on load and anything unrecognised falls
back to the atlas. Removing a view without that guard leaves anyone who had it
open looking at a blank page, which is exactly what happened when the rescue view
went away.

## Staleness

The atlas and benchmark views show whatever sweep produced `browser_payload.json`.
After a re-run, rebuild the payload first, then the atlas — `--check` will tell you
whether the committed `atlas.html` is behind the template and payloads it came
from.
