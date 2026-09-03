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
    # clip robot ids map positionally onto the fleet, ordered by base.x —
    # the same remap the Play page does on load
    robots = sorted(clip["robots"], key=lambda r: r["base"]["x"])
    payload = [{"id": fid, "start_frame": r.get("start_frame", 0),
                "frames": r["frames"]}
               for fid, r in zip(fleet_ids, robots) if r.get("frames")]
    if not payload:
        print(f"  {name}: no frames for the given arms, skipped")
        return True
    url = (f"{backend}/ws/deploy?arms={urllib.parse.quote(arms)}"
           f"&hz={int(clip.get('hz', hz))}")
    n = clip.get("n_frames", "?")
    print(f"\n=== {name}  ({len(payload)} arms, {n} frames "
          f"@ {clip.get('hz', hz)} Hz, ramp {ramp_s}s) ===")
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
    ap.add_argument("--hz", type=int, default=30)
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
