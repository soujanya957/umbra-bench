#!/usr/bin/env python3
"""Serve the atlas AND close the labeling loop from inside it.

`python -m http.server` shows the dashboard; this shows the same dashboard and
additionally accepts what the teleop view produces, so clicking points is no
longer half of a workflow whose other half lives in a terminal:

    POST /api/points/<set>   merge the exported points into <set>/points.json
    POST /api/rerun/<set>    same merge, then teleop_pipeline.py on exactly the
                             named captures, then payload + atlas rebuild
    GET  /api/ping           how the page discovers it is being served by the
                             studio rather than a plain static server

The dashboard feature-detects the ping: served statically, its save/rerun
buttons never appear and copy-points still works. Points merge per capture and
never delete -- an entry someone wants gone is removed in the file, where the
removal is reviewable. Set names are validated against Teleops/source/ and
capture stems against that set's manifest, so a request can only ever touch a
folder the repo already calls a teleop set.

Run it with the eval env (SAM2 must be importable for the rerun):

    ~/miniconda3/envs/umbra-bench/python.exe scripts/teleop_studio.py
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
STEM_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

# One pipeline run at a time: a second click while SAM2 is grinding gets a 409
# instead of a second process racing the first over manifest.json.
RERUN_LOCK = threading.Lock()


def discover_sets():
    sets = {}
    for mp in sorted(glob.glob(os.path.join(BENCH, "Teleops", "source", "*",
                                            "masks", "manifest.json"))):
        root = os.path.dirname(os.path.dirname(mp))
        sets[os.path.basename(root)] = root
    return sets


def merge_points(root, body):
    """Update per-capture entries; keep everything else (other captures, notes)."""
    pp = os.path.join(root, "points.json")
    cur = {}
    if os.path.exists(pp):
        with open(pp, encoding="utf-8") as f:
            cur = json.load(f)
    cur.setdefault("captures", {})
    if "ff_window" in body:
        cur["ff_window"] = body["ff_window"]
    cur["captures"].update(body["captures"])
    with open(pp, "w", encoding="utf-8") as f:
        json.dump(cur, f, indent=1)
    return pp, sorted(body["captures"])


def run(cmd):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    r = subprocess.run(cmd, cwd=BENCH, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:])
    if r.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[1])} exited {r.returncode}:\n{tail}")
    return tail


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kw):
        super().__init__(*args, directory=os.path.join(BENCH, "atlas"), **kw)

    def log_message(self, fmt, *args):
        sys.stderr.write("[studio] %s\n" % (fmt % args))

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/ping":
            return self._json(200, {"ok": True, "sets": sorted(SETS)})
        return super().do_GET()

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        if not 0 < n <= 4_000_000:
            raise ValueError("bad Content-Length")
        return json.loads(self.rfile.read(n))

    def do_POST(self):
        m = re.match(r"^/api/(points|rerun)/([A-Za-z0-9_\-]+)$", self.path)
        if not m:
            return self._json(404, {"error": "unknown endpoint"})
        action, name = m.groups()
        root = SETS.get(name)
        if root is None:
            return self._json(404, {"error": f"unknown set {name!r}"})
        try:
            body = self._body()
            caps = body.get("captures") or {}
            with open(os.path.join(root, "masks", "manifest.json"),
                      encoding="utf-8") as f:
                known = {r["capture"] for r in json.load(f)["records"]}
            bad = [c for c in caps if not STEM_RE.match(c) or c not in known]
            if bad or not caps:
                return self._json(400, {"error": f"unknown captures: {bad}"
                                        if bad else "no captures in body"})
            pp, stems = merge_points(root, body)
            if action == "points":
                return self._json(200, {"ok": True, "saved": stems,
                                        "path": os.path.relpath(pp, BENCH)})
            if not RERUN_LOCK.acquire(blocking=False):
                return self._json(409, {"error": "a rerun is already in progress"})
            try:
                rel = os.path.relpath(root, BENCH)
                log = run([sys.executable, os.path.join(HERE, "teleop_pipeline.py"),
                           "--teleop-root", rel,
                           "--images"] + [os.path.join(rel, "raw", s + ".png")
                                          for s in stems])
                log += "\n" + run([sys.executable,
                                   os.path.join(HERE, "_build_teleop_payload.py")])
                log += "\n" + run([sys.executable,
                                   os.path.join(BENCH, "atlas", "build_atlas.py")])
            finally:
                RERUN_LOCK.release()
            return self._json(200, {"ok": True, "reran": stems, "log": log})
        except Exception as e:                            # surfaced to the page
            return self._json(500, {"error": str(e)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8479)
    a = ap.parse_args()
    global SETS
    SETS = discover_sets()
    if not SETS:
        raise SystemExit("[!] no sets under Teleops/source/*/masks/manifest.json")
    print(f"[studio] sets: {', '.join(sorted(SETS))}")
    print(f"[studio] http://localhost:{a.port}/atlas.html")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
