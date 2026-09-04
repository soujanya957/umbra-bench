#!/usr/bin/env python3
"""Capture a teleop shadow: RealSense frame -> AprilTag rectify -> named pair on disk.

This is the step that was missing. The first 29 captures were made by hand with a
script that never reached the repo, which meant the rectification could not be
reproduced and a second batch could not be guaranteed comparable to the first.
The parameters below were recovered by fitting against those 29 pairs, so
anything captured here lands in the same frame as what already exists.

What was recovered, and why each part is not a guess:

  * The markers are **AprilTag 36h11, ids 0-3**, one per corner of the projection
    area. All 29 existing pairs detect all four, so this is the rig's convention
    rather than an inference from one photo.
  * The projection quad is each tag's **inner corner** — the one nearest the
    centroid of the four. Warping that quad onto the stored rectified image
    correlates at 0.983; tag centres score 0.588 and outer corners 0.328, so the
    choice is not ambiguous.
  * Output is **500 x 383**. The existing files are 500x383 or 500x384 depending
    on where the quad's aspect ratio rounded; pinning it removes a one-pixel
    difference that has no meaning and would otherwise persist forever.

Ordering matters: tags must be placed so that ids 0,1,2,3 run clockwise from the
top-left of the projection area. The script draws the detected quad in the live
view, so a mis-placed tag is visible before anything is saved, and it refuses to
save a frame where fewer than four tags are found — a capture that cannot be
rectified is not a capture.

    python scripts/capture_teleop.py --name letters_upperK_A3_topology
    python scripts/capture_teleop.py --from-images 'Teleops/*.png'   # no camera
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time

import cv2
import numpy as np

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEOP = os.path.join(BENCH, "Teleops")
OUT_W, OUT_H = 500, 383
TAG_IDS = (0, 1, 2, 3)


def detector():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())


def find_quad(gray, det):
    """The four inner corners, ordered by tag id. None if any tag is missing."""
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return None, set()
    ids = ids.flatten()
    by = {int(i): corners[k][0] for k, i in enumerate(ids)}
    have = set(by) & set(TAG_IDS)
    if have != set(TAG_IDS):
        return None, have
    centroid = np.mean([by[i].mean(0) for i in TAG_IDS], axis=0)
    quad = [by[i][int(np.argmin(np.linalg.norm(by[i] - centroid, axis=1)))] for i in TAG_IDS]
    return np.float32(quad), have


def rectify(img, quad):
    dst = np.float32([[0, 0], [OUT_W, 0], [OUT_W, OUT_H], [0, OUT_H]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(quad, dst), (OUT_W, OUT_H))


def read_pose(repo):
    """Joint angles at the moment of capture, if leRobot-control is importable."""
    try:
        sys.path.insert(0, os.path.join(repo, "leRobot-control"))
        from so100_arm import SO100Arm  # noqa
        import json as _j
        cfg = os.path.join(repo, "leRobot-control", "configs", "lab_default.json")
        ports = _j.load(open(cfg))["arms"] if os.path.exists(cfg) else []
        out = {}
        for a in ports:
            arm = SO100Arm(a["port"], robot_id=a["id"])
            arm.connect()
            out[a["id"]] = [float(v) for v in arm.read_qpos_rad()]
            arm.disconnect()
        return out or None
    except Exception as e:                                     # noqa: BLE001
        print(f"  [pose] skipped ({type(e).__name__}: {e})")
        return None


def save_pair(name, raw, quad, pose=None):
    os.makedirs(TELEOP, exist_ok=True)
    raw_p = os.path.join(TELEOP, name + ".png")
    rect_p = os.path.join(TELEOP, name + "_rectified.png")
    cv2.imwrite(raw_p, raw)
    cv2.imwrite(rect_p, rectify(raw, quad))
    meta = {"name": name, "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tag_family": "apriltag_36h11", "tag_ids": list(TAG_IDS),
            "quad_inner_corners": quad.tolist(), "out_size": [OUT_W, OUT_H],
            "raw_size": [raw.shape[1], raw.shape[0]]}
    if pose:
        meta["joint_angles_rad"] = pose
    with open(os.path.join(TELEOP, name + "_capture.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"  saved {name}.png + _rectified.png + _capture.json")


def from_images(pattern, det):
    n_ok = n_bad = 0
    for p in sorted(glob.glob(os.path.join(BENCH, pattern))):
        if p.endswith("_rectified.png"):
            continue
        img = cv2.imread(p)
        if img is None:
            continue
        quad, have = find_quad(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), det)
        name = os.path.splitext(os.path.basename(p))[0]
        if quad is None:
            print(f"  !! {name}: found tags {sorted(have) or 'none'} — need {list(TAG_IDS)}")
            n_bad += 1
            continue
        cv2.imwrite(os.path.join(TELEOP, name + "_rectified.png"), rectify(img, quad))
        n_ok += 1
    print(f"\nrectified {n_ok}, failed {n_bad}")


def live(a, det):
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, a.width, a.height, rs.format.bgr8, a.fps)
    pipe.start(cfg)
    print("SPACE capture · N rename · Q quit    (a frame without 4 tags cannot be saved)")
    name, i = a.name or "capture", 0
    try:
        while True:
            f = pipe.wait_for_frames().get_color_frame()
            if not f:
                continue
            img = np.asanyarray(f.get_data())
            quad, have = find_quad(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), det)
            view = img.copy()
            ok = quad is not None
            if ok:
                cv2.polylines(view, [quad.astype(int)], True, (0, 220, 0), 2)
                for k, (x, y) in enumerate(quad):
                    cv2.circle(view, (int(x), int(y)), 6, (0, 220, 0), -1)
                    cv2.putText(view, str(k), (int(x) + 8, int(y)), 0, 0.6, (0, 220, 0), 2)
            msg = (f"{name}_{i:02d}  READY" if ok
                   else f"need tags {sorted(set(TAG_IDS) - have)}")
            cv2.putText(view, msg, (12, 30), 0, 0.8, (0, 220, 0) if ok else (0, 90, 240), 2)
            cv2.imshow("teleop capture", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("n"):
                cv2.destroyAllWindows()
                name = input("name: ").strip() or name
                i = 0
            if k == 32:
                if not ok:
                    print("  refused — fewer than four tags in frame")
                    continue
                save_pair(f"{name}_{i:02d}" if a.numbered else name, img, quad,
                          read_pose(a.repo) if a.record_pose else None)
                i += 1
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="base name for the capture, e.g. letters_upperK_A3_topology")
    ap.add_argument("--numbered", action="store_true", help="append _00, _01, … per shot")
    ap.add_argument("--from-images", metavar="GLOB",
                    help="re-rectify existing photos instead of opening a camera")
    ap.add_argument("--record-pose", action="store_true",
                    help="also read the arms' joint angles at capture time")
    ap.add_argument("--repo", default=os.path.expanduser("~/GitHub/fleet-shadow-art"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    det = detector()
    if a.from_images:
        from_images(a.from_images, det)
    else:
        live(a, det)


if __name__ == "__main__":
    main()
