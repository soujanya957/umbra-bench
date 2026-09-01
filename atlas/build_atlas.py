#!/usr/bin/env python3
"""Splice the payloads into the atlas template and write the single-file dashboard.

The atlas is one self-contained HTML file: the base64 thumbnails, the metric
tables and the teleop frames all live inside it, so it opens from a URL, a file,
or an email attachment with nothing else installed. That only works if the file
is *built* rather than edited -- the payloads run to megabytes and hand-editing a
6 MB document is how a dashboard and the data it claims to show drift apart.

    python atlas/build_atlas.py                 # -> atlas/atlas.html
    python atlas/build_atlas.py --bare          # -> atlas/src/atlas.fragment.html
    python atlas/build_atlas.py --check         # verify, write nothing

Two shapes, and the difference is not cosmetic. The template is a *fragment* --
no doctype, no <head> -- because the artifact host wraps it in one at publish
time. Opened straight off disk that same fragment lands in quirks mode with the
character set merely guessed, which is how the arrows and em-dashes in the copy
turn into mojibake. `--bare` writes the fragment for publishing; the default
wraps it so the committed file is a document a browser can open as it stands.

Layout follows from that: `atlas/atlas.html` is the only openable file at the
root, and the two that must not be opened -- the template and the publish
fragment -- live in `atlas/src/`. They had all three sat together, and the
template, being a perfectly ordinary-looking .html, got opened instead.

Inputs, all produced by scripts/ and all committed:

    results/browser_payload.json    scripts/_build_browser_payload.py
    results/teleop_payload.json     scripts/_build_teleop_payload.py

`teleop_payload.json` is also emitted as `tff_0..7.json` chunks elsewhere in the
pipeline because a single upload above ~1.5 MB fails; the chunks are that
transport's problem, not this one, and the whole file is read here.
"""
from __future__ import annotations

import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)

SRC = os.path.join(HERE, "src")
TPL = os.path.join(SRC, "atlas.tpl.html")
OUT = os.path.join(HERE, "atlas.html")
BARE = os.path.join(SRC, "atlas.fragment.html")

# Mirrors what the artifact host prepends, so the file on disk and the published
# page agree on box model, scaling and encoding rather than only appearing to.
SKELETON = ("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{color-scheme:light dark}
  body{margin:0;font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
  img{max-width:100%%}
  [hidden]{display:none!important}
</style>
%s
</head>
<body>
%s
</body>
</html>
""")
SLOTS = {
    "__PAYLOAD__": os.path.join(BENCH, "results", "browser_payload.json"),
    "__TELEOP__": os.path.join(BENCH, "results", "teleop_payload.json"),
}


def wrap(fragment: str) -> str:
    """Split the fragment at the end of its stylesheet and build a document.

    Everything up to the first `</style>` is head material -- the title, the font
    links, the whole token sheet -- and the rest is markup. A `<title>` left in
    the body is ignored by some browsers, which is the failure this avoids.
    """
    cut = fragment.index("</style>") + len("</style>")
    return SKELETON % (fragment[:cut], fragment[cut:].lstrip("\n"))


def load(path: str) -> str:
    """Read as text, but parse first: a truncated payload is a syntax error in
    the browser, hours after the build, with no line number worth having."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    json.loads(raw)
    return raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="validate inputs and the existing build, write nothing")
    ap.add_argument("--bare", action="store_true",
                    help="emit the head-less fragment the artifact host expects")
    ap.add_argument("--out")
    a = ap.parse_args()
    if not a.out:
        a.out = BARE if a.bare else OUT

    if not os.path.exists(TPL):
        print(f"no template at {os.path.relpath(TPL, BENCH)}", file=sys.stderr)
        return 1
    with open(TPL, encoding="utf-8") as f:
        html = f.read()

    missing = [p for p in SLOTS.values() if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"missing payload: {os.path.relpath(p, BENCH)}", file=sys.stderr)
        return 1

    for slot, path in SLOTS.items():
        if slot not in html:
            print(f"template has no {slot} slot", file=sys.stderr)
            return 1
        blob = load(path)
        html = html.replace(slot, blob, 1)
        print(f"  {slot:12} <- {os.path.relpath(path, BENCH):38} {len(blob)/1e6:6.2f} MB")

    left = [s for s in ("__PAYLOAD__", "__TELEOP__", "__RESCUE__") if s in html]
    if left:
        print(f"unfilled slots remain: {left}", file=sys.stderr)
        return 1

    if not a.bare:
        html = wrap(html)

    if a.check:
        if not os.path.exists(a.out):
            print("no existing build to check", file=sys.stderr)
            return 1
        with open(a.out, encoding="utf-8") as f:
            cur = f.read()
        same = cur == html
        print(f"{os.path.relpath(a.out, BENCH)} is "
              + ("up to date" if same else "STALE — re-run without --check"))
        return 0 if same else 1

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {os.path.relpath(a.out, BENCH)}  {len(html)/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
