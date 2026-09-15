#!/usr/bin/env python3
"""deploy_to_robot.py — stream choreographies to the physical fleet.

    fleet-shadow python demo/deploy_to_robot.py \\
        --arms "SR101|COM3|so-101|serial,SR102|COM4|so-101|serial,SR103|COM5|so-101|serial" \\
        --clip pixar_scene_03_lamp_stab [--clip letters_upper_A_...]

Speaks the robot UI's own deploy protocol to the running render_server
(ws://127.0.0.1:8001/ws/deploy) — the same path the Play page uses, so the
server-side safety gate (velocity ceiling, joint clamping, ramp floor) and
its E-stop semantics apply unchanged. Clips play in the order given, each
with its own start ramp from the fleet's current pose.

SAFETY: closing this window (or Ctrl+C) disconnects the socket, and the
server's contract for a disconnect is E-STOP — the fleet freezes in place
with torque held. The Play page's E-STOP button also remains available.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.parse
from pathlib import Path

FSA = Path(__file__).resolve().parent.parent.parent / "fleet-shadow-art"
CHOREO = FSA / "choreographies"


async def deploy_clip(name: str, arms: str, hz: int, ramp_s: float,
                      backend: str) -> bool:
    import websockets
    clip = json.loads((CHOREO / f"{name}.json").read_text(encoding="utf-8"))
    fleet_ids = [a.split("|")[0] for a in arms.split(",") if a]
    # The Play page's remap, reproduced exactly (mapClipRobots): pass 1 is an
    # EXACT id match, pass 2 assigns remaining clip lanes IN CLIP-ARRAY ORDER
    # onto free fleet arms in roster order. (An earlier base.x sort here could
    # swap lanes between physical arms whenever clip ids and stage order
    # disagreed -- clip_01A.json does.)
    lanes = [r for r in clip["robots"] if r.get("frames")]
    free = [f for f in fleet_ids if f not in {r["id"] for r in lanes}]
    mapping, unmapped = [], []
    for r in lanes:
        if r["id"] in fleet_ids:
            mapping.append((r["id"], r))
        elif free:
            mapping.append((free.pop(0), r))
        else:
            unmapped.append(r["id"])
    if unmapped:
        print(f"  {name}: NO ARM for clip lane(s) {', '.join(unmapped)} -- "
              f"the fleet has {len(fleet_ids)} arm(s), the clip needs "
              f"{len(lanes)}. Refusing a partial performance; add the "
              "missing arms to --arms or deploy a smaller clip.")
        return False
    payload = [{"id": fid, "start_frame": r.get("start_frame", 0),
                "frames": r["frames"]} for fid, r in mapping]
    if not payload:
        print(f"  {name}: clip carries no frames, skipped")
        return True
    use_hz = hz if hz is not None else int(clip.get("hz", 30))
    url = (f"{backend}/ws/deploy?arms={urllib.parse.quote(arms)}"
           f"&hz={use_hz}")
    n = clip.get("n_frames", "?")
    print(chr(10) + f"=== {name}  ({len(payload)} arms, {n} frames "
          f"@ {use_hz} Hz, ramp {ramp_s}s) ===")
    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        while True:
            try:
                # a wrong port can leave the server stuck opening serial
                # before it says anything -- do not sit silent forever
                raw = await asyncio.wait_for(ws.recv(), timeout=25)
            except asyncio.TimeoutError:
                print("  no response in 25 s -- check the ports and that "
                      "the arms are plugged in; disconnecting (fleet is "
                      "not moving).")
                return False
            m = json.loads(raw)
            t = m.get("type")
            if t == "connected":
                await ws.send(json.dumps({"type": "start", "ramp_s": ramp_s,
                                          "loop": False, "robots": payload}))
                print("  connected — ramping to start pose")
            elif t == "progress":
                if m.get("phase") == "play":
                    fr = m.get("frame", 0)
                    print(f"\r  playing frame {fr}/{m.get('n', '?')}   ",
                          end="", flush=True)
            elif t == "done":
                print("\n  done")
                return True
            elif t == "estopped":
                print("\n  E-STOP — fleet frozen in place (torque held)")
                return False
            elif t == "error":
                print(f"\n  ERROR from deploy server: {m.get('data')}")
                return False
    print("\n  connection closed by server")
    return False


async def main_async(a) -> int:
    ids = []
    for entry in a.arms.split(","):
        parts = entry.split("|")
        if len(parts) != 4 or not all(pt.strip() for pt in parts):
            print(f"[!] bad --arms entry {entry!r}: need ID|PORT|MODEL|CONN")
            return 1
        ids.append(parts[0])
    dups = {i2 for i2 in ids if ids.count(i2) > 1}
    if dups:
        print(f"[!] duplicated fleet id(s) in --arms: {', '.join(sorted(dups))}"
              " -- one lane would silently never play and its serial port "
              "would stay claimed; fix the list.")
        return 1
    for i, name in enumerate(a.clip):
        if not (CHOREO / f"{name}.json").exists():
            print(f"[!] {name}: not in {CHOREO}")
            return 1
    for i, name in enumerate(a.clip):
        ok = await deploy_clip(name, a.arms, a.hz, a.ramp, a.backend)
        if not ok:
            print("stopping the sequence here.")
            return 1
        if i + 1 < len(a.clip):
            await asyncio.sleep(0.5)
    print("\nall clips played.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", required=True,
                    help="ID|PORT|MODEL|CONN,... e.g. SR101|COM3|so-101|serial")
    ap.add_argument("--clip", action="append", required=True,
                    help="choreography name; repeat for a sequence of clips")
    ap.add_argument("--hz", type=int, default=None,
                    help="override the clip's own hz (default: play as "
                         "authored -- every packed clip is 30)")
    ap.add_argument("--ramp", type=float, default=2.0)
    ap.add_argument("--backend", default="ws://127.0.0.1:8001")
    a = ap.parse_args()
    print("Close this window or Ctrl+C at ANY time = E-STOP "
          "(fleet freezes in place, torque held).")
    try:
        rc = asyncio.run(main_async(a))
    except KeyboardInterrupt:
        print("\ninterrupted — socket closed, fleet frozen in place.")
        rc = 1
    try:
        input(chr(10) + ("finished — " if rc == 0 else "")
              + "press Enter to close.")
    except EOFError:
        pass
    sys.exit(rc)


if __name__ == "__main__":
    main()
