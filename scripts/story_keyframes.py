"""Keyframe targets for the ICRA story film: one mask per (keyframe, subject), all on one scale.

Input: the story folder and the keyframes.json exported by scripts/keyframe_annotator.html.
For every keyframe the frame is read, side bars are cut, the subject's silhouette is picked
out, and it is brought back to the scale of the original film:

  * full-frame videos (1880x1080) are the reference: digits ~710 px tall standing on the ground
    line (y ~ 897). Their scale is 1.
  * cropped videos were often zoomed and lost their top or bottom, so their zoom is estimated
    from stroke width (2x the median of the distance transform on the ridge), which a crop does
    not change: zoom = file stroke / reference stroke of the same digit in a full-frame video.
  * every subject stands on the ground: full frames keep their own ground line; a cropped
    video's bottom edge is taken as the ground (its elements touch it).

Output (results/icra_story/keyframes/): per keyframe and subject a centred mask on a
film-sized canvas (1880x1080, film px), plus targets.json with the subject's
horizontal position in the film where it is known (full-frame videos), the zoom used and the
ground offset; review.jpg puts each source frame next to its normalised mask, with the
reference digit height drawn, so sizes can be checked by eye.

    python scripts/story_keyframes.py --story <folder> --keyframes <keyframes.json>
"""

import argparse
import json
import os
import re

import cv2
import numpy as np

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BENCH, "results", "icra_story", "keyframes")
FILM_H = 1080        # film frame height (px)
DIGIT_H = 710        # reference digit height in the film (px)
FILM_GROUND = 898    # top row of the film's ground line (rows 898-909)
CANVAS_H, CANVAS_W = 1080, 1880   # output canvas, in film px: the film frame itself

# Videos that hold a single element: their subject is the file's, whatever was selected
# while marking (the annotator kept the clip's first subject active across videos).
FILE_SUBJECT = {
    "right_1_lean_back": "right 1",
    "number_4_full_intro": "4",
    "only_1_full_intro": "1",
    "number_3_walking_left": "3",
    "number_7_lean": "7",
    "left_1_transform_2": "left 1→2",
}
# Reference stroke width (px, film scale) of the digit each cropped video starts from,
# measured on full-frame videos where the digit is whole and unzoomed.
REF_STROKE = {"1": 134.0, "3": 118.7, "7": 122.0}
# '+' is thinner than the digits; its cropped video is sized by its full height against the '+'
# in the full-frame clip 6 (312 px), using the tallest frame (it rises in from the bottom).
REF_HEIGHT = {"+": 312.0}


def subject_of_file(path):
    for k, s in FILE_SUBJECT.items():
        if k in path.replace("\\", "/"):
            return s
    return None


def black(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 110


def cut_bars(b):
    """Drop fully black columns at both sides (letterbox of a cropped export)."""
    col = b.mean(0)
    l, r = 0, b.shape[1]
    while l < r and col[l] > 0.97:
        l += 1
    while r > l and col[r - 1] > 0.97:
        r -= 1
    return b[:, l:r], l


def ground_line(b):
    """(top, bottom) rows of the ground line, or None: a run of at most 16 rows (the film's is 12), mostly black
    across the frame, in its lower 30%. The base of a '1' in a narrow crop is as wide but much
    thicker, so it is not taken for the line."""
    H = b.shape[0]
    rows = np.nonzero(b.mean(1) > 0.8)[0]
    rows = rows[rows >= 0.7 * H]
    if not len(rows):
        return None
    runs = [r for r in np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1) if len(r) <= 16]
    if not runs:
        return None
    run = max(runs, key=lambda r: r[-1])
    return int(run[0]), int(run[-1])


def without_ground(b):
    """b with the ground line cleared, and the line (or None)."""
    g = ground_line(b)
    if g:
        b = b.copy()
        y0, y1 = max(0, g[0] - 2), min(b.shape[0], g[1] + 3)
        # a shape crossing the line (a squashed walking '3') stays whole: refill the band where
        # it is black both just above and just below the line
        through = b[max(0, y0 - 1)] & b[min(b.shape[0] - 1, y1)]
        b[y0:y1] = through[None]
    return b, g


def stroke(m):
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    ridge = dt[(dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3))) - 1e-6)]
    return 2 * float(np.median(ridge)) if len(ridge) else 0.0


def components(b, min_frac=0.0003):
    n, lab, st, cen = cv2.connectedComponentsWithStats(b.astype(np.uint8))
    keep = [i for i in range(1, n) if st[i, 4] >= min_frac * b.size]
    return lab, [(i, st[i], cen[i]) for i in keep]


def pick_subject(b, subject, subjects):
    """Mask of one subject in a frame holding several: small blobs are the dot, the rest are
    matched to the clip's subjects (minus the dot) from left to right."""
    lab, comps = components(b)
    if not comps:
        return np.zeros_like(b)
    big = max(c[1][4] for c in comps)
    dots = [c for c in comps if c[1][4] < 0.08 * big]
    main = sorted([c for c in comps if c[1][4] >= 0.08 * big], key=lambda c: c[2][0])
    names = [s for s in subjects if s != "dot"]
    if subject == "dot":
        sel = dots
    elif len(main) == len(names):
        sel = [main[names.index(subject)]]
    else:  # touching or merged elements: keep everything big (flagged in targets.json)
        sel = main
    return np.isin(lab, [c[0] for c in sel])


def read_frame(path, k):
    c = cv2.VideoCapture(path)
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    c.set(cv2.CAP_PROP_POS_FRAMES, min(k, n - 1))
    ok, im = c.read()
    c.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {k} of {path}")
    return im


def file_zoom(path, subject):
    """Zoom of a cropped video against the film, from stroke width over the whole video."""
    if subject is None:
        return 1.0, None
    key = subject.split("→")[0].split(" ")[-1]
    ref = REF_STROKE.get(key)
    if not ref and key not in REF_HEIGHT:
        return 1.0, None
    c = cv2.VideoCapture(path)
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    s = []
    for k in range(0, n, max(1, n // 15)):
        c.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, im = c.read()
        if ok:
            b, _ = without_ground(cut_bars(black(im))[0])
            if b.any():
                if ref:
                    s.append(stroke(b))
                else:
                    ys = np.nonzero(b.any(1))[0]
                    s.append(float(ys.max() - ys.min() + 1))
    c.release()
    if not ref:
        h = max(s)
        return h / REF_HEIGHT[key], h
    st = float(np.median(s))
    return st / ref, st


def main(a):
    kf = json.load(open(a.keyframes, encoding="utf-8"))
    os.makedirs(OUT, exist_ok=True)
    targets, review, fixes = [], [], []
    zoom_cache = {}
    for clip, cd in kf["clips"].items():
        subjects = cd["subjects"]
        for rel, vd in cd["videos"].items():
            path = os.path.join(a.story, *rel.split("/"))
            fsub = subject_of_file(rel)
            for m in vd["keyframes"]:
                subj = m["subject"]
                if fsub and subj != fsub:
                    fixes.append((rel.split("/")[-1], m["frame"], subj, fsub))
                    subj = fsub
                im = read_frame(path, m["frame"])
                b0 = black(im)
                b, left = cut_bars(b0)
                full = b.shape[1] >= 0.95 * FILM_H * 1880 / 1080 and b.shape[0] == FILM_H
                b, g = without_ground(b)
                ground_y = g[0] if g else b.shape[0]
                # a crop that kept the film's ground line at the film's height was not zoomed
                film_line = bool(g and b.shape[0] == FILM_H and abs(g[0] - FILM_GROUND) <= 15)
                mask = pick_subject(b, subj, subjects) if (len(subjects) > 1 and not fsub) else b.copy()
                _, comps = components(mask)
                if comps:  # drop specks
                    lab, _ = components(mask)
                    mask = np.isin(lab, [c[0] for c in comps])
                if full or film_line:
                    zoom, st = 1.0, None
                else:
                    if rel not in zoom_cache:
                        zoom_cache[rel] = file_zoom(path, fsub)
                    zoom, st = zoom_cache[rel]
                if not mask.any():
                    print("empty mask:", rel, m["frame"], subj)
                    continue
                ys, xs = np.nonzero(mask)
                # to film scale, standing on the ground, centred horizontally on the canvas
                s = 1.0 / zoom
                sub = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
                h, w = sub.shape
                hs, ws = max(1, round(h * s)), max(1, round(w * s))
                sub = cv2.resize(sub, (ws, hs), interpolation=cv2.INTER_AREA) > 127
                lift = (ground_y - 1 - ys.max()) * s           # gap between subject and ground, film px
                canvas = np.zeros((CANVAS_H, CANVAS_W), bool)
                gy = CANVAS_H - 1                                # ground = canvas bottom row
                y1 = gy; y0 = y1 - hs + 1          # pinned: every subject stands on the ground, centred
                                                   # (its own lift is kept in targets.json for later)
                x0 = (CANVAS_W - ws) // 2
                ya, yb = max(0, y0), min(CANVAS_H, y1 + 1)
                xa, xb = max(0, x0), min(CANVAS_W, x0 + ws)
                if ya >= yb or xa >= xb:
                    print(f"off canvas, skipped: {rel} {m['frame']} {subj} (h {hs}, lift {lift:.0f})")
                    continue
                canvas[ya:yb, xa:xb] = sub[ya - y0:yb - y0, xa - x0:xb - x0]
                cid = re.match(r"clip_(\d+)", clip).group(1)
                name = f"c{int(cid):02d}_{os.path.basename(rel)[:-4]}_f{m['frame']:03d}_{re.sub(r'[^0-9A-Za-z+]+', '-', subj)}"
                cv2.imwrite(os.path.join(OUT, name + ".png"), (~canvas).astype(np.uint8) * 255)
                cx = (left + (xs.min() + xs.max()) / 2) if full else None
                targets.append({"id": name, "clip": clip, "video": rel, "frame": m["frame"], "subject": subj,
                                "note": m.get("note", ""), "zoom": round(zoom, 3), "stroke_px": st and round(st, 1),
                                "height_film_px": hs, "width_film_px": ws, "lift_film_px": round(lift, 1),
                                "x_center_film_px": cx, "full_frame": bool(full),
                                "merged": bool(len(subjects) > 1 and not fsub and subj != "dot"
                                               and len(components(b)[1]) < len(subjects))})
                T = 200
                small = cv2.resize(im, (int(im.shape[1] * T / im.shape[0]), T), interpolation=cv2.INTER_AREA)
                tw = int(T * CANVAS_W / CANVAS_H)
                thumb = cv2.resize((~canvas).astype(np.uint8) * 255, (tw, T), interpolation=cv2.INTER_AREA)
                review.append((name, small, thumb, zoom, subj))
                print(f"{name:<70} zoom {zoom:.2f}  h {hs:4d}  lift {lift:6.1f}", flush=True)
    json.dump({"film_height_px": FILM_H, "digit_height_px": DIGIT_H, "canvas_px": [CANVAS_W, CANVAS_H],
               "ground": "canvas bottom row", "targets": targets,
               "subject_fixes": [{"video": v, "frame": f, "was": w, "now": n} for v, f, w, n in fixes]},
              open(os.path.join(OUT, "targets.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # review sheet: source frame | normalised target with the reference digit height marked
    T = 200
    rows = []
    for name, im, canvas, zoom, subj in review:
        src = cv2.copyMakeBorder(im, 0, 0, 0, max(0, 360 - im.shape[1]), cv2.BORDER_CONSTANT, value=(255, 255, 255))[:, :360]
        TW = canvas.shape[1]
        tgt = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        ref = int(round(T - DIGIT_H * T / CANVAS_H))
        cv2.line(tgt, (0, ref), (TW - 1, ref), (214, 120, 42), 1)      # reference digit height
        cv2.line(tgt, (0, T - 1), (TW - 1, T - 1), (120, 120, 120), 1)  # ground
        lab = np.full((T, 330, 3), 255, np.uint8)
        for i, t in enumerate([name[:40], name[40:80], f"subject: {subj}", f"zoom {zoom:.2f}"]):
            cv2.putText(lab, t, (6, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        rows.append(np.hstack([lab, src, np.full((T, 6, 3), 255, np.uint8), tgt]))
    sheet = np.vstack([np.vstack([r, np.full((3, r.shape[1], 3), 200, np.uint8)]) for r in rows])
    for i in range(0, len(sheet), 4000):
        cv2.imwrite(os.path.join(OUT, f"review_{i // 4000}.jpg"), sheet[i:i + 4000], [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"{len(targets)} targets; {len(fixes)} subject labels corrected")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--story", default=r"C:\Users\hexia\Downloads\ICRA_story_direction\ICRA_story_direction")
    p.add_argument("--keyframes", required=True)
    main(p.parse_args())
