"""Assemble the "sHAdOW * art" scene: sHAdOW's six letters, a star, then a/r/t.

    <fleet-shadow python> build_shadow_art.py --star-clip <choreography name> --star-n-arms 5 --star-gap 0.25
        [--star-clip-file scratch.json] [--star-offset 0.0] [--star-scale 1.0] [--force] [--dry-run]
    <fleet-shadow python> build_shadow_art.py --no-star --scene scene_02 [--word-gap 0.6]

Wall layout: keep sHAdOW's own wall centres and pitch, leave 1.5 pitches before and
after the star, set a/r/t at the sHAdOW pitch, and centre the whole phrase.
Letters keep make_ensemble's placement (lat = wall centre - offset*mag). The star
is a sequence frame, so it has no library offset; it sits on its slot with
--star-offset (0 by default) and carries its own arm count and spacing. Its lamp
stays 1.0 m in front of arm 0, as in the solver's rig.

--no-star spells plain "sHAdOW art": a/r/t follow W after one pitch plus
--word-gap pitches of word space, and the phrase is re-centred.

Writes fleet-shadow-art/ensembles/SHADOW_ART_<scene>.json (refuses to overwrite
without --force), the studio scaffold demo/projects/SHADOW_ART/{assignments.json,
scenes/<scene>/f0001.png}, and an adjust file for shadow_post.py that opens the
word gap and, with a star, centres it vertically. scene_01 writes adjust.json;
any other scene writes adjust_<scene>.json.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(r"C:/Users/hexia/Documents/GitHub/umbra-bench")
sys.path.insert(0, str(REPO / "demo"))
import make_ensemble as me  # noqa: E402

ART = [("a", "letters_lower_a_dejavusansmono-bold"), ("r", "letters_lower_r_dejavuserif-bold"),
       ("t", "letters_lower_t_dejavuserif-bold")]
PROJECT = REPO / "demo" / "projects" / "SHADOW_ART"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene_01", help="scene id; the manifest is SHADOW_ART_<scene>")
    ap.add_argument("--no-star", action="store_true", help='spell "sHAdOW art" with no star between the words')
    ap.add_argument("--word-gap", type=float, default=0.6, help="--no-star: extra pitches of space between the words")
    ap.add_argument("--star-clip", default=None, help="choreography name in fleet-shadow-art/choreographies")
    ap.add_argument("--star-clip-file", default=None, help="scratch clip JSON to install under --star-clip first")
    ap.add_argument("--star-n-arms", type=int, default=3)
    ap.add_argument("--star-gap", type=float, default=0.2)
    ap.add_argument("--star-offset", type=float, default=0.0, help="arm-plane lateral offset of the star's cast (m)")
    ap.add_argument("--star-scale", type=float, default=1.0, help="post composite scale for the star")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.no_star and not a.star_clip:
        ap.error("--star-clip is required unless --no-star")
    name = f"SHADOW_ART_{a.scene}"

    src = json.loads((me.OUT / "SHADOW_scene_01.json").read_text(encoding="utf-8"))
    n = src["n_frames"]
    letters = [dict(e) for e in src["entries"]]
    walls = [e["lat_path_m"][0] + me.shadow_geometry(e["clip"])[0][0] * e["mag"] for e in letters]
    pitch = (walls[-1] - walls[0]) / (len(walls) - 1)
    if a.no_star:
        star_wall = None
        art_walls = [walls[-1] + (1.0 + a.word_gap + k) * pitch for k in range(len(ART))]
    else:
        star_wall = walls[-1] + 1.5 * pitch
        art_walls = [star_wall + (1.5 + k) * pitch for k in range(len(ART))]
    shift = -(walls[0] + art_walls[-1]) / 2

    entries = []
    for e, w in zip(letters, walls):
        off = me.shadow_geometry(e["clip"])[0][0]
        entries.append(dict(e, lat_path_m=[round(w + shift - off * e["mag"], 4)] * n))
    lamp = 1.0 + (a.star_n_arms - 1) / 2 * a.star_gap
    if not a.no_star:
        entries.append({"element": "star", "clip": a.star_clip, "entry": 0, "n_frames": n, "mag": 2.0,
                        "depth_m": 1.2, "n_arms": a.star_n_arms, "arm_gap_m": a.star_gap,
                        "lat_path_m": [round(star_wall + shift - a.star_offset * 2.0, 4)] * n,
                        "lift_path_m": [0.0] * n, "light": {"ddepth": -round(lamp, 3), "h": 0.3}})
    for (el, lid), w in zip(ART, art_walls):
        off = me.shadow_geometry(lid)[0][0]
        entries.append({"element": el, "clip": lid, "entry": 0, "n_frames": n, "mag": 2.0, "depth_m": 1.2,
                         "lat_path_m": [round(w + shift - off * 2.0, 4)] * n, "lift_path_m": [0.0] * n,
                         "light": {"ddepth": -1.2, "h": 0.3}})
    doc = dict(src, name=name, screen_w_m=round(art_walls[-1] - walls[0] + 1.0, 3), entries=entries)

    print(f"{name}: pitch {pitch:.3f} m, "
          + (f"star wall centre {star_wall + shift:+.3f}, star lamp {lamp:.2f} m behind mid arm, " if star_wall is not None
             else f"word gap {1.0 + a.word_gap:.2f} pitches, ")
          + f"phrase span {walls[0] + shift:+.2f} .. {art_walls[-1] + shift:+.2f} m")
    for e in entries:
        print(f"  {e['element']:4s} {e['clip']:44s} lat {e['lat_path_m'][0]:+.4f}"
              + (f"  n_arms {e['n_arms']} gap {e['arm_gap_m']}" if "n_arms" in e else ""))
    if a.dry_run:
        return

    dest = me.OUT / f"{name}.json"
    if dest.exists() and not a.force:
        sys.exit(f"[!] {dest} exists; pass --force to overwrite")
    if not a.no_star:
        clip_path = me.FSA / "choreographies" / f"{a.star_clip}.json"
        if a.star_clip_file:
            shutil.copyfile(a.star_clip_file, clip_path)
        if not clip_path.exists():
            sys.exit(f"[!] no choreography {clip_path}")
        robots = len(json.loads(clip_path.read_text(encoding="utf-8"))["robots"])
        if robots != a.star_n_arms:
            sys.exit(f"[!] {a.star_clip} has {robots} robots, --star-n-arms says {a.star_n_arms}")
    dest.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print("wrote", dest)

    (PROJECT / "scenes" / a.scene).mkdir(parents=True, exist_ok=True)
    ass = PROJECT / "assignments.json"
    if not ass.exists():
        ass.write_text("{}\n", encoding="utf-8")
    if a.no_star:
        adjust = {"a": {"space_before": 0.35 + a.word_gap * 0.4}}
    else:
        adjust = {"star": {"space_before": 0.35, "align": "center", "scale": a.star_scale},
                  "a": {"space_before": 0.35}}
    adj_path = PROJECT / ("adjust.json" if a.scene == "scene_01" else f"adjust_{a.scene}.json")
    adj_path.write_text(json.dumps(adjust, indent=1), encoding="utf-8")
    print("studio scaffold +", adj_path.name, "in", PROJECT, "| placeholder ->", placeholder(a.scene, not a.no_star))


def placeholder(scene="scene_01", star=True):
    """Studio's scene chip only needs one frame; spell the letters from their solver renders."""
    stems = [("letters_lower", "s_dejavuserif-bold"), ("letters_upper", "H_dejavusansmono-bold"),
             ("letters_upper", "A_dejavusans-bold"), ("letters_lower", "d_dejavuserif-bold"),
             ("letters_upper", "O_dejavusans-bold"), ("letters_upper", "W_dejavusansmono-bold"), None,
             ("letters_lower", "a_dejavusansmono-bold"), ("letters_lower", "r_dejavuserif-bold"),
             ("letters_lower", "t_dejavuserif-bold")]
    if not star:
        stems[6] = "gap"
    tiles = []
    for s in stems:
        if s is None or s == "gap":
            tiles.append(np.full((128, 128 if s is None else 64), 255, np.uint8))
        else:
            p = REPO / "optimized" / "big-budget-grounded" / s[0] / s[1] / f"{s[1]}_best.png"
            tiles.append(np.asarray(Image.open(p).convert("L")))
        tiles.append(np.full((128, 12), 255, np.uint8))
    row = np.hstack(tiles[:-1])
    canvas = np.full((168, row.shape[1] + 40), 255, np.uint8)
    canvas[20:148, 20:20 + row.shape[1]] = row
    out = PROJECT / "scenes" / scene / "f0001.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--placeholder-only":
        print("placeholder ->", placeholder())
    else:
        main()
