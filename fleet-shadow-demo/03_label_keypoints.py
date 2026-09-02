#!/usr/bin/env python3
"""
03_label_keypoints.py — click SAM point prompts onto the frames.

    python3 03_label_keypoints.py                      # all scenes
    python3 03_label_keypoints.py --scenes scene_03    # just one
    python3 03_label_keypoints.py --review             # only unlabelled frames

Writes `keypoints.json`, which 04_sam_segment.py turns into masks on a GPU box.
Saves after every edit, so closing the window never loses work, and reopening
picks up exactly where you left off.

Controls
--------
  left click        add a POSITIVE point for the current label
  right click       add a NEGATIVE point (tells SAM "not this")
  a-z / 0-9         set the current label (F, A, M, I, L, Y, …)
  TAB               cycle through labels already used in this frame
  u                 undo last point in this frame
  d                 delete every point of the current label in this frame
  D                 clear the whole frame
  → / space         next frame          ← / b   previous frame
  n                 next frame with no points yet
  c                 copy the previous frame's points into this one
  h                 toggle this help
  q                 quit (everything is already saved)

Negative points are optional. They earn their keep where a glyph sits against
something similar — the doorway in the room shots, the trees in the park — and
one negative click there is usually worth more than three extra positives.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------- data model
IMG_EXT = (".png", ".jpg", ".jpeg")


def frame_id_of(p: Path) -> str:
    return p.stem


def find_frames(root: Path, scenes: list[str] | None) -> list[tuple[str, Path]]:
    """-> [(scene, path)] sorted by frame id."""
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if scenes and d.name not in scenes:
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                out.append((d.name, p))
    out.sort(key=lambda t: int(re.sub(r"\D", "", t[1].stem) or 0))
    return out


class Store:
    """keypoints.json, kept on disk after every edit.

    Written atomically (temp file + replace) because the whole point of saving
    constantly is that a crash mid-write must not be able to eat the file.
    """

    def __init__(self, path: Path):
        self.path = path
        self.data = {"meta": {}, "frames": {}}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
                self.data.setdefault("meta", {})
                self.data.setdefault("frames", {})
            except (json.JSONDecodeError, OSError) as e:
                sys.exit(f"{path} exists but could not be read ({e}). "
                         "Move it aside if you want to start over.")

    def entry(self, fid: str, scene: str, file: str) -> dict:
        f = self.data["frames"].setdefault(
            fid, {"file": file, "scene": scene, "objects": {}})
        f["file"], f["scene"] = file, scene
        return f

    def points(self, fid: str, label: str) -> dict:
        objs = self.data["frames"][fid]["objects"]
        return objs.setdefault(label, {"points": [], "labels": []})

    def add(self, fid: str, label: str, xy, positive: bool) -> None:
        o = self.points(fid, label)
        o["points"].append([round(float(xy[0]), 1), round(float(xy[1]), 1)])
        o["labels"].append(1 if positive else 0)

    def undo(self, fid: str) -> str | None:
        """Remove the most recently added point anywhere in this frame."""
        objs = self.data["frames"].get(fid, {}).get("objects", {})
        best, best_n = None, -1
        for lab, o in objs.items():
            if len(o["points"]) > best_n:
                best, best_n = lab, len(o["points"])
        if not best or best_n == 0:
            return None
        objs[best]["points"].pop()
        objs[best]["labels"].pop()
        if not objs[best]["points"]:
            del objs[best]
        return best

    def drop_label(self, fid: str, label: str) -> None:
        self.data["frames"].get(fid, {}).get("objects", {}).pop(label, None)

    def clear(self, fid: str) -> None:
        self.data["frames"].get(fid, {})["objects"] = {}

    def n_points(self, fid: str) -> int:
        objs = self.data["frames"].get(fid, {}).get("objects", {})
        return sum(len(o["points"]) for o in objs.values())

    def save(self) -> None:
        self.data["meta"]["updated"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
        os.replace(tmp, self.path)


# ---------------------------------------------------------------- the window
PALETTE = ["#ffd400", "#ff4fa3", "#33e06a", "#4fc3ff", "#ff8a3d",
           "#b06bff", "#00d9c0", "#ff5c5c", "#9bd400", "#ff9ecb"]


def color_for(label: str) -> str:
    return PALETTE[sum(map(ord, label)) % len(PALETTE)]


def run_gui(frames, store: Store, start_label: str) -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    if matplotlib.get_backend().lower() == "agg":
        sys.exit("matplotlib has no interactive backend (got 'agg').\n"
                 "  pip install pyobjc  # macOS native window\n"
                 "or run with: MPLBACKEND=TkAgg python3 03_label_keypoints.py")

    # Matplotlib's own shortcuts would otherwise steal the letter keys —
    # 's' saves a figure, 'l' toggles log scale, 'q' quits, and so on.
    for k in list(plt.rcParams):
        if k.startswith("keymap."):
            plt.rcParams[k] = []

    state = {"i": 0, "label": start_label, "help": True, "img": None}

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.canvas.manager.set_window_title("keypoints — h for help")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.06)

    HELP = ("left=positive  right=negative   a-z/0-9=label  TAB=cycle\n"
            "u=undo  d=drop label  D=clear frame  c=copy prev\n"
            "→/space=next  ←/b=prev  n=next empty  h=help  q=quit")

    def draw():
        i = state["i"]
        scene, path = frames[i]
        fid = frame_id_of(path)
        store.entry(fid, scene, str(path))

        ax.clear()
        if state["img"] is None or state["img"][0] != path:
            state["img"] = (path, mpimg.imread(str(path)))
        ax.imshow(state["img"][1])
        ax.set_xticks([]); ax.set_yticks([])

        objs = store.data["frames"][fid]["objects"]
        for lab, o in sorted(objs.items()):
            c = color_for(lab)
            for (x, y), pos in zip(o["points"], o["labels"]):
                ax.plot(x, y, marker="o" if pos else "X", ms=11,
                        mfc=c if pos else "none", mec="black" if pos else c,
                        mew=1.8 if pos else 2.6, zorder=5)
                ax.annotate(lab, (x, y), xytext=(9, 9),
                            textcoords="offset points", color=c, fontsize=11,
                            fontweight="bold", zorder=6,
                            path_effects=None)

        done = sum(1 for _, p in frames if store.n_points(frame_id_of(p)))
        summary = "  ".join(
            f"{l}:{len(o['points'])}" for l, o in sorted(objs.items())) or "—"
        ax.set_title(
            f"[{i+1}/{len(frames)}]  {fid}  ({scene})     "
            f"label→ {state['label']}     this frame: {summary}     "
            f"labelled {done}/{len(frames)}",
            fontsize=11, loc="left")
        if state["help"]:
            ax.text(0.5, -0.035, HELP, transform=ax.transAxes, ha="center",
                    va="top", fontsize=9, family="monospace", alpha=0.75)
        fig.canvas.draw_idle()

    def cur_fid() -> str:
        return frame_id_of(frames[state["i"]][1])

    def go(delta: int):
        state["i"] = max(0, min(len(frames) - 1, state["i"] + delta))
        draw()

    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None:
            return
        if ev.button not in (1, 3):
            return
        store.add(cur_fid(), state["label"], (ev.xdata, ev.ydata),
                  positive=(ev.button == 1))
        store.save()
        draw()

    def on_key(ev):
        k = ev.key or ""
        fid = cur_fid()

        if k in ("right", " ", "space"):
            go(1); return
        if k in ("left", "b"):
            go(-1); return
        if k == "n":
            for j in range(state["i"] + 1, len(frames)):
                if store.n_points(frame_id_of(frames[j][1])) == 0:
                    state["i"] = j; draw(); return
            print("no empty frames after this one")
            return
        if k == "q":
            store.save(); plt.close(fig); return
        if k == "h":
            state["help"] = not state["help"]; draw(); return
        if k == "u":
            store.undo(fid); store.save(); draw(); return
        if k == "d":
            store.drop_label(fid, state["label"]); store.save(); draw(); return
        if k == "D":
            store.clear(fid); store.save(); draw(); return
        if k == "c":
            if state["i"] > 0:
                prev = frame_id_of(frames[state["i"] - 1][1])
                src = store.data["frames"].get(prev, {}).get("objects", {})
                store.data["frames"][fid]["objects"] = json.loads(
                    json.dumps(src))
                store.save(); draw()
            return
        if k == "tab":
            labs = sorted(store.data["frames"][fid]["objects"])
            if labs:
                nxt = (labs.index(state["label"]) + 1) % len(labs) \
                    if state["label"] in labs else 0
                state["label"] = labs[nxt]; draw()
            return
        if len(k) == 1 and (k.isalpha() or k.isdigit()):
            state["label"] = k.upper(); draw(); return

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()
    store.save()


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Click SAM point prompts onto the frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--frames-root", default="scenes")
    ap.add_argument("--scenes", nargs="*",
                    help="limit to these scene folder names")
    ap.add_argument("--out", default="keypoints.json")
    ap.add_argument("--label", default="I", help="starting label")
    ap.add_argument("--review", action="store_true",
                    help="only show frames that have no points yet")
    ap.add_argument("--stats", action="store_true",
                    help="print progress and exit, no window")
    args = ap.parse_args()

    os.chdir(Path(__file__).resolve().parent)
    frames = find_frames(Path(args.frames_root), args.scenes)
    if not frames:
        sys.exit(f"no frames under {args.frames_root}/")
    store = Store(Path(args.out))
    store.data["meta"].setdefault("frames_root", args.frames_root)

    if args.review:
        frames = [(s, p) for s, p in frames
                  if store.n_points(frame_id_of(p)) == 0]
        if not frames:
            print("every frame already has points."); return

    if args.stats:
        per_scene = {}
        for s, p in frames:
            n = store.n_points(frame_id_of(p))
            d, t = per_scene.get(s, (0, 0))
            per_scene[s] = (d + (1 if n else 0), t + 1)
        total_done = sum(d for d, _ in per_scene.values())
        for s in sorted(per_scene):
            d, t = per_scene[s]
            print(f"  {s}: {d}/{t} frames labelled")
        print(f"  TOTAL: {total_done}/{len(frames)}")
        objs = store.data["frames"]
        labs = {}
        for f in objs.values():
            for lab, o in f.get("objects", {}).items():
                labs[lab] = labs.get(lab, 0) + len(o["points"])
        if labs:
            print("  points per label: " +
                  "  ".join(f"{k}={v}" for k, v in sorted(labs.items())))
        return

    print(f"{len(frames)} frame(s). Saving to {args.out} after every edit.")
    print("Press h in the window for the key list.")
    run_gui(frames, store, args.label.upper())
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
