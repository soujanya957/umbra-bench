"""Pick sheet: benchmark letter solves rendered the way the post composite
renders letters (alone, broad lamp, arms hidden for the plate).

    <fleet-shadow python> picker.py --letters S,s,D,d --name SD
    <fleet-shadow python> picker.py --letters A,a,R,r,T,t --name ART --used letters_upper_A_dejavusans-bold

Each letter gets one row of big-budget-grounded best runs, in font order sans /
mono / serif, coded <letter><1-3>. Uppercase letters come from letters_upper and
lowercase from letters_lower. Clips are written to a scratch folder with
demo/pack.py's write_choreo, so nothing lands in fleet-shadow-art until the user
has chosen. Each cell shows the target glyph inset and the benchmark scores
underneath (tc_iou, CLIP rank of 26).
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(r"C:/Users/hexia/Documents/GitHub/umbra-bench")
FSA = REPO.parent / "fleet-shadow-art"
HERE = Path(__file__).resolve().parent
OUT = HERE / "picker"
CLIPS = OUT / "clips"
sys.path.insert(0, str(HERE))
import shadow_post as sp  # noqa: E402

_spec = importlib.util.spec_from_file_location("pack", REPO / "demo" / "pack.py")
pack = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, [sys.argv[0]]
_spec.loader.exec_module(pack)
sys.argv = _argv

FONTS = [("dejavusans-bold", "sans"), ("dejavusansmono-bold", "mono"), ("dejavuserif-bold", "serif")]


def scores():
    tc = {}
    with open(REPO / "results" / "metrics_big-budget-grounded.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["tc_iou"]:
                tc.setdefault(r["sample_id"], []).append(float(r["tc_iou"]))
    clip = {}
    with open(REPO / "results" / "clip_dataset" / "clip_per_image.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["condition"] == "big":
                clip[r["id"]] = (int(r["rank"]), int(r["n_classes"]))
    return {k: float(np.mean(v)) for k, v in tc.items()}, clip


def font(size):
    for p in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", default="S,s,D,d")
    ap.add_argument("--name", default="SD")
    ap.add_argument("--used", default="", help="comma-separated library ids to mark as already in use")
    a = ap.parse_args()
    letters = [x for x in a.letters.split(",") if x]
    used = set(x for x in a.used.split(",") if x)
    CLIPS.mkdir(parents=True, exist_ok=True)
    sp.RES = (1920, 1080)
    man = json.loads((FSA / "ensembles" / "SHADOW_scene_01.json").read_text(encoding="utf-8"))
    tc, clip = scores()
    CW, CH, SH = 460, 600, 470
    sheet = Image.new("RGB", (CW * 3, CH * len(letters)), (245, 245, 245))
    big, small = font(34), font(22)
    rows_out = []
    for ri, L in enumerate(letters):
        case = "upper" if L.isupper() else "lower"
        for fi, (fname, fshort) in enumerate(FONTS):
            lid = f"letters_{case}_{L}_{fname}"
            stem = f"{L}_{fname}"
            rj = REPO / "optimized" / "big-budget-grounded" / f"letters_{case}" / stem / "results.json"
            code = f"{L}{fi + 1}"
            cell = Image.new("RGB", (CW, CH), (222, 222, 222))
            d = ImageDraw.Draw(cell)
            if not rj.exists():
                d.text((14, SH + 10), code + "  (no solve)", fill=(200, 30, 30), font=big)
                sheet.paste(cell, (fi * CW, ri * CH))
                continue
            res = json.loads(rj.read_text(encoding="utf-8"))
            q = np.asarray(res["runs"][res["best_run"]]["q_rad"], dtype=float).ravel()
            gap = float(res["rig"]["arm_gap_m"])
            pack.write_choreo(CLIPS, lid, 5.0, [q], force=True, arm_gap=gap)
            clipj = json.loads((CLIPS / f"{lid}.json").read_text(encoding="utf-8"))
            entry = {"element": L, "clip": lid, "entry": 0, "n_frames": 25, "mag": 2.0, "depth_m": 1.2,
                     "arm_gap_m": gap, "lat_path_m": [0.0] * 25, "lift_path_m": [0.0] * 25,
                     "light": {"ddepth": -(1.0 + gap), "h": 0.3}}
            full, plate = sp.render_letter(man, {lid: clipj}, entry)
            m = sp.matte_of(full, plate)
            got = sp.crop_box(m, pad=20)
            if got:
                (x0, y0, x1, y1), edge = got
                crop = m[y0:y1, x0:x1]
                shade = np.clip(222 * (1.0 - 0.88 * (1.0 - crop)), 0, 255).astype(np.uint8)
                img = Image.fromarray(shade).convert("RGB")
                k = min((CW - 40) / img.width, (SH - 40) / img.height)
                img = img.resize((max(1, int(img.width * k)), max(1, int(img.height * k))), Image.LANCZOS)
                cell.paste(img, ((CW - img.width) // 2, 20 + (SH - 40 - img.height) // 2))
            tgt = Image.open(REPO / "targets_grounded" / f"letters_{case}" / f"{stem}.png").convert("L").resize((96, 96), Image.LANCZOS)
            cell.paste(Image.merge("RGB", [tgt] * 3), (CW - 106, 10))
            d = ImageDraw.Draw(cell)
            d.rectangle([0, SH, CW, CH], fill=(250, 250, 250))
            d.text((14, SH + 10), code + ("  (in sHAdOW)" if lid in used else ""), fill=(200, 30, 30), font=big)
            rk = clip.get(lid)
            d.text((14, SH + 58), f"{fshort}   tc_iou {tc.get(lid, float('nan')):.2f}   CLIP {rk[0]}/{rk[1]}" if rk else fshort,
                   fill=(40, 40, 40), font=small)
            d.rectangle([0, 0, CW - 1, CH - 1], outline=(180, 180, 180), width=2)
            sheet.paste(cell, (fi * CW, ri * CH))
            rows_out.append({"code": code, "id": lid, "font": fshort, "arm_gap_m": gap,
                             "tc_iou": round(tc.get(lid, float("nan")), 3),
                             "clip_rank": rk[0] if rk else None, "best_run": res["best_run"],
                             "edge": bool(got and got[1])})
            print(f"  {code} {lid} gap={gap} tc={tc.get(lid, float('nan')):.2f} clip={rk} edge={bool(got and got[1])}")
    sheet.save(OUT / f"{a.name}_picker_sheet.png")
    sheet.resize((sheet.width * 2 // 3, sheet.height * 2 // 3), Image.LANCZOS).save(OUT / f"{a.name}_picker_sheet_preview.jpg", quality=90)
    (OUT / f"{a.name}_picker.json").write_text(json.dumps(rows_out, indent=1), encoding="utf-8")
    print("sheet ->", OUT / f"{a.name}_picker_sheet.png")


if __name__ == "__main__":
    main()
