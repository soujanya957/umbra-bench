#!/usr/bin/env python3
"""doctor.py -- can this machine run the studio, and can it deploy?

    python demo/doctor.py            # check everything
    python demo/doctor.py --deploy   # only what deploy_to_robot needs

Answers the questions a fresh box makes you answer by crashing: which
interpreter each lane actually resolved to, whether that interpreter can
import what its stages need, whether the viewer has mjpython, whether the
serial ports in the fleet configs are plugged in right now, and whether the
robot UI's deploy server is listening.

Nothing here touches the arms. It only reports.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("studio", ROOT / "studio.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "
_rc = {"bad": 0, "warn": 0}


def say(level: str, what: str, detail: str = "") -> None:
    if level is BAD:
        _rc["bad"] += 1
    elif level is WARN:
        _rc["warn"] += 1
    print(f"[{level}] {what}" + (f"\n            {detail}" if detail else ""))


def head(t: str) -> None:
    print(f"\n--- {t} " + "-" * max(0, 66 - len(t)))


def probe(py: str, mods: list[str]) -> dict:
    """Import each module in THAT interpreter and report what came back."""
    code = ("import json,sys;r={'py':sys.version.split()[0]}\n"
            "for m in %r:\n"
            "    try:\n"
            "        __import__(m); r[m]='ok'\n"
            "    except Exception as e: r[m]=type(e).__name__+': '+str(e)[:60]\n"
            "print(json.dumps(r))" % mods)
    try:
        p = subprocess.run([py, "-c", code], capture_output=True, text=True,
                           timeout=120, env=S.env_for(py))
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:                       # noqa: BLE001 -- report, never raise
        return {"error": f"{type(e).__name__}: {e}"}


def lane(name: str, py: str, mods: list[str]) -> None:
    print(f"\n  {name}: {py}")
    if py == sys.executable:
        say(WARN, f"{name} fell back to the interpreter running doctor.py",
            "no conda env of that name was found -- fine if this box has one "
            "env, but the stage will only run if that env has the imports")
    if not Path(py).exists():
        return say(BAD, f"{name}: interpreter does not exist")
    r = probe(py, mods)
    if "error" in r:
        return say(BAD, f"{name}: could not run it", r["error"])
    print(f"        python {r.pop('py', '?')}")
    for m, v in r.items():
        say(OK if v == "ok" else BAD, f"{name}: import {m}",
            "" if v == "ok" else v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", action="store_true",
                    help="only the deploy path (fleet env, ports, server)")
    a = ap.parse_args()

    head("machine")
    print(f"  platform   {sys.platform}   ({os.name})")
    print(f"  studio.py  {sys.executable}")
    roots = S.conda_roots()
    print(f"  conda roots{'':1} " + (", ".join(str(r) for r in roots) or "none found"))
    if not roots:
        say(WARN, "no conda root found", "every lane will use this interpreter")
    if (ROOT / "studio_env.json").exists():
        say(OK, "studio_env.json overrides in effect", json.dumps(S._OVERRIDE))

    head("interpreters")
    lane("fleet", S.PY_FLEET, ["mujoco", "websockets", "numpy"])
    if not a.deploy:
        lane("eval", S.PY_EVAL, ["cv2", "numpy", "PIL", "scipy", "skimage"])
        lane("gpu", S.PY_GPU, ["torch", "numpy"])

    head("mujoco viewer")
    vp = S.viewer_python(S.PY_FLEET)
    if sys.platform != "darwin":
        say(OK, "not macOS -- the viewer runs under plain python")
    elif Path(vp).name == "mjpython":
        say(OK, f"mjpython found: {vp}")
    else:
        say(BAD, "no mjpython next to the fleet interpreter",
            "MuJoCo's passive viewer must own the main thread on macOS; "
            "`ensemble (mujoco)` will abort. Fix: conda install -n <fleet env> "
            "mujoco  (mjpython ships with the pip/conda mujoco package)")

    if not a.deploy:
        head("ffmpeg")
        if S.FFMPEG_DIR:
            say(OK, f"lerobot ffmpeg: {S.FFMPEG_DIR}")
        else:
            from shutil import which
            w = which("ffmpeg")
            say(OK if w else WARN, f"system ffmpeg: {w or 'NOT on PATH'}",
                "" if w else "trim/scenes/compose need it: brew install ffmpeg")

    head("robot UI checkout")
    for p, what in ((S.FSA, "fleet-shadow-art"),
                    (S.CHOREO_DIR, "choreographies/"),
                    (S.FSA / "arrangements", "arrangements/"),
                    (S.FLEET_CFG_DIR, "leRobot-control/configs/")):
        say(OK if p.is_dir() else BAD, f"{what}: {p}")
    names = S.choreo_names()
    say(OK if names else WARN, f"{len(names)} clip(s) ready to deploy",
        ", ".join(names[:8]) + (" ..." if len(names) > 8 else ""))

    head("fleet")
    arms = S.fleet_default_arms()
    if not arms:
        say(BAD, "no default --arms could be built",
            f"expected {S.FLEET_STAGE}.json in {S.FLEET_CFG_DIR}")
    else:
        print(f"  default --arms:\n    {arms}")
        live = set()
        for pat in ("/dev/tty.usbmodem*", "/dev/tty.usbserial*", "/dev/cu.usb*",
                    "/dev/ttyACM*", "/dev/ttyUSB*"):
            live |= set(glob.glob(pat))
        for entry in arms.split(","):
            rid, port, _model, _conn = (entry.split("|") + ["", "", "", ""])[:4]
            if not port:
                say(WARN, f"{rid}: no port in any config",
                    "plug it in and save a config from the robot UI, or type "
                    "the port into the deploy prompt")
            elif os.name == "nt":
                say(OK, f"{rid}: {port} (not checkable on Windows)")
            elif Path(port).exists():
                say(OK, f"{rid}: {port} is plugged in")
            else:
                say(WARN, f"{rid}: {port} is NOT present right now",
                    ("unplugged, or the serial number changed. Ports visible "
                     "now: " + ", ".join(sorted(live))) if live
                    else "no USB serial devices visible at all")

    head("what is deployable right now")
    import re as _re
    B = ROOT.parent
    solved_lib = 0
    try:
        for line in (B / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if (B / "optimized" / "big-budget-grounded" / r["subset"]
                    / Path(r["target"]).stem / "results.json").exists():
                solved_lib += 1
    except OSError as e:
        say(WARN, "could not read metadata.jsonl", str(e))
    say(OK if solved_lib else WARN,
        f"{solved_lib} library shape(s) solved -- one click each",
        "pick one in the rail, then 'deploy this'. Its choreography is "
        "written on demand; nothing else has to run first.")

    seqs = sorted(d.name for d in (B / "sequences").glob("*_scene_*"))
    solved = [s for s in seqs
              if list((B / "optimized" / s).glob("summary_*.json"))]
    ready = []
    for sid in solved:
        proj = sid.split("_scene_")[0]
        if ((ROOT / "projects" / proj / "out" / "reassembled" / sid
             / "reassembly.json").exists()
                or (ROOT / "out" / "reassembled" / sid
                    / "reassembly.json").exists()):
            ready.append(sid)
    print(f"  sequences: {len(seqs)} cut, {len(solved)} solved, "
          f"{len(ready)} reassembled")
    if ready:
        say(OK, f"{len(ready)} sequence(s) deployable",
            ", ".join(ready[:6]) + (" ..." if len(ready) > 6 else ""))
    elif solved:
        say(WARN, "no sequence is deployable yet",
            f"{len(solved)} are solved but none is REASSEMBLED, and a clip's "
            "choreography is built from reassembly.json. Run the studio's "
            "'re-mount' button (demo/08_reassemble.py --all) once -- library "
            "shapes deploy without it, sequences do not.")
    else:
        say(WARN, "no solved sequences", "solve a scene first (studio step 6)")

    head("deploy server")
    hostport = ("127.0.0.1", 8001)
    s = socket.socket()
    s.settimeout(1.5)
    try:
        s.connect(hostport)
        say(OK, "render_server is listening on ws://127.0.0.1:8001")
    except OSError as e:
        say(BAD, "nothing listening on 127.0.0.1:8001", f"{e} -- start the "
            "robot UI's render_server first; deploy speaks its /ws/deploy")
    finally:
        s.close()

    print()
    if _rc["bad"]:
        print(f"{_rc['bad']} blocker(s), {_rc['warn']} warning(s).")
    elif _rc["warn"]:
        print(f"no blockers, {_rc['warn']} warning(s).")
    else:
        print("all clear.")
    return 1 if _rc["bad"] else 0


if __name__ == "__main__":
    sys.exit(main())
