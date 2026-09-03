#!/usr/bin/env python3
"""studio.py — the demo production line as buttons.

    ~/miniconda3/envs/umbra-bench/python.exe demo/studio.py
    -> http://localhost:8478/

One page per THE pipeline every demo video walks: pick a video (its filename
stem becomes the project name, so naming can never be fumbled), then run each
stage from the browser — trim, scenes, label (opens the click window on this
machine), segment, clean, sequences, index, route, per-clip solve, reassemble,
score, compose, pack. Each stage runs the same script the terminal would, with
the same arguments run_demo.py uses, under the interpreter its dependencies
need:

    eval   umbra-bench   cv2/scipy/skimage stages, metrics, payloads
    gpu    fh_l1         SAM2 segmentation and CLIP scoring (CUDA -- the
                         user's standing instruction is to prioritise it)
    fleet  fleet-shadow  run_sequence solves (mujoco)

ffmpeg's binary comes from the lerobot env's Library/bin, prepended to every
subprocess PATH (fh_l1 has the ffmpeg-python wrapper, which resolves the
binary through that same PATH).

One job at a time; the page tails the log live. The footage workspace
(scenes/, keypoints.json, letters_*) holds ONE project at a time — the page
says which — while sequences/, optimized/ and out/ are multi-project.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent
HOME = Path.home()

PY_EVAL = str(HOME / "miniconda3/envs/umbra-bench/python.exe")
PY_GPU = str(HOME / "miniconda3/envs/fh_l1/python.exe")
PY_FLEET = str(HOME / "miniconda3/envs/fleet-shadow/python.exe")
# The ffmpeg BINARY lives in the lerobot env; fh_l1 carries only the
# ffmpeg-python wrapper, which finds the binary through PATH.
FFMPEG_DIR = str(HOME / "miniconda3/envs/lerobot/Library/bin")
MAS = str(BENCH.parent / "fleet-shadow-art" / "motion-aware-shadow")
SAM_SMALL = ("C:/Users/hexia/Documents/GitHub/animal_inspired_BC/thirdparty/"
             "sam2/checkpoints/sam2.1_hiera_small.pt",
             "configs/sam2.1/sam2.1_hiera_s.yaml")

LOG_DIR = ROOT / "out" / "studio_logs"
STATE_F = ROOT / "out" / "studio_state.json"
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

JOB = {"running": False, "step": None, "log": None, "rc": None, "t0": None}
LOCK = threading.Lock()


def sanitize(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "", stem)


def project_of(stem: str) -> str:
    """pixar_01, pixar_02 ... are trims of ONE project: strip the trim number
    (the user's rule), so their sequences share the pixar_ prefix and their
    scene numbers continue instead of colliding."""
    return re.sub(r"_" + chr(92) + "d+$", "", sanitize(stem))


def pdir(project: str) -> Path:
    """demo/projects/<name>/ — every project owns its whole workspace, so
    nothing is ever overwritten by switching. Raw videos stay in demo/."""
    d = ROOT / "projects" / project
    d.mkdir(parents=True, exist_ok=True)
    return d


def video_registry(project: str) -> dict:
    f = pdir(project) / "videos.json"
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_registry(project: str, reg: dict) -> None:
    (pdir(project) / "videos.json").write_text(json.dumps(reg, indent=1),
                                               encoding="utf-8")


def assignments(project: str) -> dict:
    f = pdir(project) / "assignments.json"
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_assignments(project: str, a: dict) -> None:
    (pdir(project) / "assignments.json").write_text(
        json.dumps(a, indent=1, sort_keys=True), encoding="utf-8")


def re_dir_of(sid: str) -> Path:
    """A sequence's reassembly directory: its project's out/, else legacy."""
    proj = sid.split("_scene_")[0]
    d = pdir(proj) / "out" / "reassembled" / sid
    return d if d.is_dir() else ROOT / "out" / "reassembled" / sid


def seq_frame_paths(stem: str):
    """(paths, invert) for a sequence's motion preview: the solve's
    best_shadow frames when present (white-on-black, so invert), else the
    authored frames (already dark-on-white)."""
    run = BENCH / "optimized" / stem
    js = sorted(run.glob("summary_*.json"))
    if js:
        ts = js[-1].stem[len("summary_"):]
        sh = sorted(run.glob(f"frame_*_{ts}/best_shadow.png"))
        if sh:
            return sh, True
    return sorted((BENCH / "sequences" / stem).glob("f*.png")), False


def seq_thumb(stem: str, i: int = 0) -> bytes | None:
    key = f"seqanim/{stem}/{i}"
    if key in THUMBS:
        return THUMBS[key]
    paths, inv = seq_frame_paths(stem)
    if not paths:
        return None
    from PIL import Image, ImageOps
    import io
    im = Image.open(paths[i % len(paths)]).convert("L").resize(
        (96, 96), Image.NEAREST)
    if inv:
        im = ImageOps.invert(im)
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    THUMBS[key] = b.getvalue()
    return THUMBS[key]


def active_project() -> str | None:
    try:
        return json.loads(STATE_F.read_text(encoding="utf-8"))["project"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def set_active(project: str, video: str) -> None:
    STATE_F.parent.mkdir(parents=True, exist_ok=True)
    STATE_F.write_text(json.dumps({"project": project, "video": video}),
                       encoding="utf-8")


def env_for(py: str) -> dict:
    """Activation-equivalent environment for a conda env, from its python.exe.

    On Windows a bare interpreter path is not enough: torch/cv2/ffmpeg resolve
    DLLs through the directories `conda activate` prepends. Build that PATH
    set explicitly so every subprocess behaves as if the env were activated
    -- the user's instruction, verbatim: "you need to activate fh_l1".
    """
    root = Path(py).parent
    pre = [root, root / "Library" / "mingw-w64" / "bin",
           root / "Library" / "usr" / "bin", root / "Library" / "bin",
           root / "Scripts", root / "bin", Path(FFMPEG_DIR)]
    e = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
             CONDA_PREFIX=str(root), CONDA_DEFAULT_ENV=root.name)
    e["PATH"] = os.pathsep.join(str(x) for x in pre) + os.pathsep + e.get("PATH", "")
    return e


def night_solve_cmd(targets: list[str], outdir: str) -> list[str]:
    """The demo family's solve config — the same explicit flag set every
    family_ad solve used (omitting a flag means silently inheriting a
    different default, the config-drift bug this repo already paid for)."""
    return ([PY_FLEET, "scripts/run_sequence.py",
             "--urdf", "urdf/SO101/so101_new_calib.urdf", "--targets"]
            + targets +
            ["--n-robots", "3",
             "--alpha", "1.0", "--beta", "0.3", "--gamma", "0.0",
             "--final-gamma", "0.0", "--delta", "0.0",
             "--popsize", "192", "--sigma0", "0.4",
             "--phase1-iters", "128", "--phase2-iters", "128",
             "--final-iters", "256",
             "--fit-target", "--fit-scale-min", "0.35",
             "--fit-scale-max", "1.6", "--fit-max-shift", "0.45",
             "--reach-samples", "300",
             "--outdir", str(BENCH / "optimized" / Path(outdir).name)])


def seq_frames(sid: str) -> list[str]:
    return sorted(str(p) for p in (BENCH / "sequences" / sid).glob("f*.png"))


def routing() -> dict:
    try:
        return json.loads((ROOT / "out" / "motion_routing.json")
                          .read_text(encoding="utf-8"))["routing"]
    except (OSError, KeyError, json.JSONDecodeError):
        return {}


def solve_jobs(sid: str) -> list:
    """The lane-aware solve chain for one element (same as the solve button)."""
    lane = routing().get(sid, {}).get("lane", "dynamic")
    jobs = []
    if lane == "translation" and not sid.endswith("_stab"):
        jobs.append([PY_EVAL, str(BENCH / "scripts" / "stabilize_sequence.py"),
                     "--ids", sid])
        solve_id = sid + "_stab"
        jobs.append(("SOLVE_LATER", solve_id))
    else:
        solve_id = sid
        fr = seq_frames(sid)
        if lane == "static":
            fr = fr[:1]
        jobs.append(night_solve_cmd(fr, solve_id))
    jobs.append([PY_EVAL, str(BENCH / "scripts" / "sequence_metrics.py"),
                 "--run", str(BENCH / "optimized" / solve_id),
                 "--sequence", solve_id])
    return jobs


def mount_jobs(P):
    """Reassemble everything solved + cast the library assignments."""
    jobs = [[PY_EVAL, "08_reassemble.py", "--all"],
            [PY_EVAL, "08_reassemble.py"] + sum(
                [["--sequence", d.name] for d in
                 sorted((BENCH / "sequences").glob("*_stab"))
                 if (BENCH / "optimized" / d.name).is_dir()], [])]
    if P:
        for sid, ass in sorted(assignments(P).items()):
            if ass.get("mode") == "library" and ass.get("library_id"):
                jobs.append([PY_EVAL, str(ROOT / "08_place_library.py"),
                             "--sequence", sid,
                             "--library-id", ass["library_id"]])
            elif ass.get("mode") == "sequence" and ass.get("donor"):
                jobs.append([PY_EVAL, str(ROOT / "08_place_sequence.py"),
                             "--sequence", sid, "--donor", ass["donor"]])
    return jobs


def build_job(step: str, arg: str | None):
    """-> (argv|list-of-argv, cwd, detach) or an error string."""
    P = active_project()
    if step == "trim":
        v = ROOT / (arg or "")
        if not v.is_file():
            return "pick a video first"
        return ([[PY_EVAL, "00_trim.py", str(v),
                  "--out-dir", str(ROOT / "clips" / sanitize(v.stem))]],
                ROOT, False)
    if step == "scenes":
        v = ROOT / (arg or "")
        if not v.is_file():
            return "pick a video first"
        proj = project_of(v.stem)
        JOB["pending_active"] = (proj, arg)
        wd = pdir(proj)
        reg = video_registry(proj)
        vname = v.name
        if vname in reg:
            # RE-splitting a video this project already split: replace, not
            # append -- delete its old scene dirs and reuse its numbering so
            # frame ids reproduce and every label keyed on them survives.
            import shutil as _sh
            r = reg[vname]
            for k in range(r["scene_start"], r["scene_start"] + r.get("n_scenes", 0)):
                _sh.rmtree(wd / "scenes" / f"scene_{k:02d}", ignore_errors=True)
            start, fstart = r["scene_start"], r["frame_start"]
            print(f"[studio] re-split of {vname}: replacing scenes "
                  f"{r['scene_start']:02d}.. with the same numbering")
        else:
            taken = [rr["scene_start"] + rr.get("n_scenes", 1) - 1
                     for rr in reg.values()]
            for d in (wd / "scenes").glob("scene_*"):
                m = re.match(r"scene_(" + chr(92) + "d+)$", d.name)
                if m:
                    taken.append(int(m.group(1)))
            start = max(taken, default=0) + 1
            fstart = 0
            for f in (wd / "scenes").glob("scene_*/f*.png"):
                m = re.match(r"f(" + chr(92) + "d+)$", f.stem)
                if m:
                    fstart = max(fstart, int(m.group(1)))
        JOB["pending_register"] = (proj, vname, start, fstart)
        return ([[PY_EVAL, str(ROOT / "01_split_scenes.py"), str(v),
                  "--workdir", str(wd),
                  "--sample", "5", "--scene-start", str(start),
                  "--frame-start", str(fstart)]],
                wd, False)
    if step == "label":
        if not P:
            return "no active project"
        return ([[PY_EVAL, str(ROOT / "03_label_keypoints.py"),
                  "--workdir", str(pdir(P))]], pdir(P), True)
    if step == "segment":
        if not P:
            return "no active project"
        # SAM2 video propagation is clean enough on its own (the user's
        # call); 06_clean_masks stays available from the terminal for
        # footage that needs it
        # segment, then group into sequences and index -- one batch, no
        # separate build button to remember
        return ([[PY_GPU, str(ROOT / "04_video_segment.py"),
                  "--workdir", str(pdir(P)), "--device", "cuda",
                  "--out", "letters_sam2_small"],
                 [PY_EVAL, str(ROOT / "07_make_sequences.py"),
                  "--in", "letters_sam2_small", "--keypoints", "keypoints.json",
                  "--prefix", f"{P}_"],
                 [PY_EVAL, str(BENCH / "scripts" / "build_sequence_metadata.py")]],
                pdir(P), False)
    if step == "clean":
        if not P:
            return "no active project"
        return ([[PY_EVAL, str(ROOT / "06_clean_masks.py"),
                  "--in", "letters_sam2_small", "--out", "letters_clean",
                  "--keypoints", "keypoints.json", "--sigma", "3.0"]],
                pdir(P), False)
    if step == "sequences":
        if not P:
            return "no active project — run 'scenes' with a video first"
        # one batch step: group into sequences, then index the track
        return ([[PY_EVAL, str(ROOT / "07_make_sequences.py"),
                  "--in", "letters_sam2_small", "--keypoints", "keypoints.json",
                  "--prefix", f"{P}_"],
                 [PY_EVAL, str(BENCH / "scripts" / "build_sequence_metadata.py")]],
                pdir(P), False)
    if step == "index":
        return ([[PY_EVAL, str(BENCH / "scripts" /
                               "build_sequence_metadata.py")]], BENCH, False)
    if step == "route":
        return [[PY_EVAL, "route_motion.py"]], ROOT, False
    if step == "solve":
        sid = arg or ""
        if not NAME_RE.match(sid) or not (BENCH / "sequences" / sid).is_dir():
            return f"unknown sequence {sid!r}"
        return solve_jobs(sid), MAS, False
    if step == "solve_all":
        if not P:
            return "no active project"
        pending = []
        for sid, ass in sorted(assignments(P).items()):
            if ass.get("mode") != "solve":
                continue
            use = sid + "_stab" if (BENCH / "optimized" / (sid + "_stab")
                                    ).is_dir() else sid
            if not list((BENCH / "optimized" / use).glob("summary_*.json")):
                pending.append(sid)
        if not pending:
            return "nothing marked 'to solve' is still unsolved"
        jobs = []
        for sid in pending:
            jobs += solve_jobs(sid)
        jobs += mount_jobs(P)
        return jobs, MAS, False
    if step == "solve_scene":
        if not P:
            return "no active project"
        scene = arg or ""
        if not re.match(r"^scene_\d+$", scene):
            return f"bad scene {scene!r}"
        pending = []
        for sid, ass in sorted(assignments(P).items()):
            if ass.get("mode") != "solve" or f"_{scene}_" not in sid + "_":
                continue
            use = sid + "_stab" if (BENCH / "optimized" / (sid + "_stab")
                                    ).is_dir() else sid
            if not list((BENCH / "optimized" / use).glob("summary_*.json")):
                pending.append(sid)
        if not pending:
            return f"nothing in {scene} is marked 'to solve' and unsolved"
        jobs = []
        for sid in pending:
            jobs += solve_jobs(sid)
        jobs += mount_jobs(P)
        return jobs, MAS, False
    if step == "reassemble":
        return mount_jobs(P), ROOT, False
    if step == "score":
        return [[PY_GPU, "09_clip_score.py"]], ROOT, False
    if step == "compose":
        return [[PY_EVAL, "10_compose_video.py"]], ROOT, False
    if step == "pack":
        if not P:
            return "no active project"
        clips = []
        for d in sorted((BENCH / "sequences").glob(f"{P}_*")):
            sid = d.name
            if sid.endswith("_stab"):
                continue
            use = sid + "_stab" if (BENCH / "optimized" / (sid + "_stab")
                                    ).is_dir() else sid
            if (BENCH / "optimized" / use).is_dir() and \
                    re_dir_of(use).is_dir():
                clips += ["--clip", use]
        libs = []
        for sid, ass in sorted(assignments(P).items()):
            if ass.get("mode") == "library" and ass.get("library_id"):
                libs += ["--library", ass["library_id"]]
        if not clips and not libs:
            return "nothing solved+reassembled for this project yet"
        return ([[PY_EVAL, "pack.py", "--name", P, "--force"] + clips + libs],
                ROOT, False)
    if step == "atlas":
        return ([[PY_EVAL, str(BENCH / "scripts" /
                               "_build_sequences_payload.py")],
                 [PY_EVAL, str(BENCH / "atlas" / "build_atlas.py")]],
                BENCH, False)
    return f"unknown step {step!r}"


def run_jobs(step: str, jobs, cwd, log_path: Path):
    rc = 0
    with open(log_path, "a", encoding="utf-8") as log:
        for j in jobs:
            if isinstance(j, tuple) and j[0] == "SOLVE_LATER":
                j = night_solve_cmd(seq_frames(j[1]), j[1])
            log.write(f"\n$ {' '.join(map(str, j))}\n")
            log.flush()
            r = subprocess.run(j, cwd=cwd, env=env_for(str(j[0])),
                               stdout=log, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")
            rc = r.returncode
            if rc:
                log.write(f"\n[exit {rc}] chain stopped\n")
                break
        else:
            log.write("\n[done]\n")
    if rc == 0:
        pa = JOB.pop("pending_active", None)
        if pa:
            set_active(*pa)
        pr = JOB.pop("pending_register", None)
        if pr:
            proj, vname, start, fstart = pr
            n = sum(1 for d in (pdir(proj) / "scenes").glob("scene_*")
                    if (m := re.match(r"scene_(" + chr(92) + "d+)$", d.name))
                    and int(m.group(1)) >= start)
            reg = video_registry(proj)
            reg[vname] = {"scene_start": start, "frame_start": fstart,
                          "n_scenes": n}
            save_registry(proj, reg)
    else:
        JOB.pop("pending_active", None)
        JOB.pop("pending_register", None)
    JOB.update(running=False, rc=rc)


def state() -> dict:
    P = active_project()
    wd = pdir(P) if P else ROOT
    n = lambda pat, root=None: len(glob.glob(str((root or wd) / pat)))
    kp = {}
    try:
        kp = json.loads((ROOT / "keypoints.json")
                        .read_text(encoding="utf-8")).get("frames", {})
    except (OSError, json.JSONDecodeError):
        pass
    # Every film-cut sequence, tagged by its own project -- the active
    # project's prefix and a video's numbered trims (pixar vs pixar_01)
    # need not agree, and a filter that assumes they do shows an empty table.
    seqs = []
    rt = routing()
    ass_all = assignments(P) if P else {}
    for d in sorted((BENCH / "sequences").glob("*_scene_*")):
        sid = d.name
        run = BENCH / "optimized" / sid
        ass = ass_all.get(sid, {})
        mode = ass.get("mode")
        solved = bool(list(run.glob("summary_*.json")))
        seqs.append({
            "id": sid,
            "project": sid.split("_scene_")[0],
            "lane": rt.get(sid, {}).get("lane"),
            "solved": solved,
            "assign": mode,
            "library_id": ass.get("library_id"),
            "donor": ass.get("donor"),
            "resolved": (mode == "solve" and solved) or
                        (mode == "library" and bool(ass.get("library_id"))) or
                        (mode == "sequence" and bool(ass.get("donor"))),
            "reassembled": (re_dir_of(sid) / "reassembly.json").exists(),
        })
    return {
        "project": P,
        "videos": sorted(str(p.relative_to(ROOT)).replace(os.sep, "/")
                         for pat in ("*.mp4", "clips/*/*.mp4")
                         for p in ROOT.glob(pat)),
        "workspace": {
            "scenes": n("scenes/*/*.png"),
            "labelled": len(kp),
            "masks": n("letters_sam2_small/by_frame/*/*_mask.png"),
            "clean": n("letters_clean/*_mask.png"),
            "routed": (ROOT / "out" / "motion_routing.json").exists(),
        },
        "sequences": seqs,
        "outputs": {
            "video": sorted(p.name for p in
                            (pdir(P) / "out" / "video").glob("*.mp4")) if P else [],
            "package": bool(P and (ROOT / "packages" / P).is_dir()),
        },
        "job": {k: JOB[k] for k in ("running", "step", "rc")} |
               {"elapsed": round(time.time() - JOB["t0"]) if JOB["t0"] and
                JOB["running"] else None},
    }


KP_LOCK = threading.Lock()


def kp_path() -> Path:
    P = active_project() or "_none"
    return pdir(P) / "keypoints.json"


def kp_load() -> dict:
    try:
        d = json.loads(kp_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        d = {}
    d.setdefault("meta", {})
    d.setdefault("frames", {})
    d.setdefault("decisions", {})
    return d


def kp_save(d: dict) -> None:
    """Atomic, exactly like 03's Store: a crash mid-write must not be able
    to eat hand labour."""
    d["meta"]["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    kp = kp_path()
    tmp = kp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, kp)


MASKTHUMB: dict[str, tuple[float, bytes]] = {}


def label_rgb(label: str) -> tuple[int, int, int]:
    """The same hue hash the page uses, so gallery and canvas agree."""
    import colorsys
    h = (sum(ord(c) for c in label) * 47 % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(h, 0.55, 0.9)
    return int(r * 255), int(g * 255), int(b * 255)


def mask_thumb(project: str, fid: str) -> bytes | None:
    """One tile per frame: every label's mask tinted onto white, 170px."""
    import io
    import numpy as np
    from PIL import Image
    d = pdir(project) / "letters_sam2_small" / "by_frame" / fid
    files = sorted(d.glob(f"{fid}_*_mask.png"))
    if not files:
        return None
    key = f"{project}/{fid}"
    mt = sum(f.stat().st_mtime for f in files)
    hit = MASKTHUMB.get(key)
    if hit and hit[0] == mt:
        return hit[1]
    base = None
    for f in files:
        m = np.array(Image.open(f).convert("L")) < 128
        if base is None:
            base = np.full(m.shape + (3,), 255, np.uint8)
        lab = f.stem[len(fid) + 1:-5] if f.stem.endswith("_mask")             else f.stem[len(fid) + 1:]
        base[m] = label_rgb(lab)
    im = Image.fromarray(base)
    im.thumbnail((170, 170), Image.NEAREST)
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    MASKTHUMB[key] = (mt, b.getvalue())
    return MASKTHUMB[key][1]


LIB_CACHE: list | None = None
THUMBS: dict[str, bytes] = {}


def library() -> list:
    """The static library, once: id, class, subset, solved flag."""
    global LIB_CACHE
    if LIB_CACHE is not None:
        return LIB_CACHE
    rows = []
    for line in (BENCH / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        stem = Path(r["target"]).stem
        rows.append({"id": r["id"], "cls": str(r.get("class", "")),
                     "subset": r["subset"], "stem": stem,
                     "solved": (BENCH / "optimized" / "big-budget-grounded" /
                                r["subset"] / stem / "results.json").exists()})
    # sequences are library citizens too: a solved clip can stand in for a
    # whole element (the board's "seq" assignment), timing re-aligned
    try:
        for line in (BENCH / "sequences.jsonl").read_text(
                encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({"id": r["id"], "cls": str(r.get("class", r["id"])),
                         "subset": "sequence", "stem": r["id"], "kind": "seq",
                         "nf": len(list((BENCH / "sequences" / r["id"])
                                        .glob("f*.png"))),
                         "solved": bool(list((BENCH / "optimized" / r["id"])
                                             .glob("summary_*.json")))})
    except OSError:
        pass
    LIB_CACHE = rows
    return rows


def thumb(subset: str, stem: str) -> bytes | None:
    key = f"{subset}/{stem}"
    if key in THUMBS:
        return THUMBS[key]
    # The solved SHADOW, not the authored target: the library exists to pick
    # what the rig can actually cast, and the best-render is that answer.
    # Unsolved rows fall back to the target (they render dimmed in the UI).
    if subset == "sequence":
        return seq_thumb(stem, 0)
    p = (BENCH / "optimized" / "big-budget-grounded" / subset / stem /
         f"{stem}_best.png")
    if not p.exists():
        p = BENCH / "targets_grounded" / subset / f"{stem}.png"
    if not p.exists():
        p = BENCH / "targets" / subset / f"{stem}.png"
    if not p.exists():
        return None
    from PIL import Image
    import io
    im = Image.open(p).convert("L").resize((96, 96), Image.NEAREST)
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    THUMBS[key] = b.getvalue()
    return THUMBS[key]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):
        sys.stderr.write("[studio] %s\n" % (fmt % args))

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/studio.html"):
            # no-store: the page is the tool, and a browser serving a stale
            # cached copy of it makes every fix look like it didn't happen
            body = (ROOT / "studio.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/state":
            return self._json(200, state())
        if self.path.startswith("/api/masks/"):
            fid = self.path.rsplit("/", 1)[1]
            if not NAME_RE.match(fid):
                return self._json(400, {"error": "bad fid"})
            P = active_project()
            if not P:
                return self._json(200, {"masks": []})
            wd, pbase = pdir(P), f"projects/{P}"
            ov = wd / "letters_sam2_small" / "overlay" / f"{fid}.jpg"
            def entry(p: Path, base: str):
                lab = p.stem[len(fid) + 1:]
                if lab.endswith("_mask"):
                    lab = lab[:-5]
                return {"label": lab, "file": f"{base}/{p.name}"}
            seg = {e["label"]: e for p in sorted(
                       (wd / "letters_sam2_small" / "by_frame" / fid)
                       .glob(f"{fid}_*_mask.png"))
                   for e in [entry(p, f"{pbase}/letters_sam2_small/by_frame/{fid}")]}
            return self._json(200, {
                "masks": list(seg.values()),
                "overlay": f"{pbase}/letters_sam2_small/overlay/{fid}.jpg"
                           if ov.exists() else None})
        if self.path == "/api/gallery":
            P = active_project()
            if not P:
                return self._json(200, [])
            wd = pdir(P)
            kp = kp_load()
            scene_of = {fid: r.get("scene", "")
                        for fid, r in kp.get("frames", {}).items()}
            items = []
            for d in sorted((wd / "letters_sam2_small" / "by_frame").glob("f*")):
                fid = d.name
                labs = sorted(q.stem[len(fid) + 1:-5]
                              for q in d.glob(f"{fid}_*_mask.png"))
                if labs:
                    items.append({"fid": fid, "labels": labs,
                                  "scene": scene_of.get(fid, "")})
            return self._json(200, items)
        if self.path.startswith("/maskthumb/"):
            fid = self.path.rsplit("/", 1)[1].split("?")[0]
            P = active_project()
            data = mask_thumb(P, fid) if P and NAME_RE.match(fid) else None
            if data is None:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/api/output":
            P = active_project() or ""
            po = (pdir(P) / "out") if P else None
            vids = [{"name": v.name, "mb": round(v.stat().st_size / 1e6, 1)}
                    for v in sorted((po / "video").glob("*.mp4"))] if po else []
            re_dirs = {}
            for d in (sorted((po / "reassembled").glob(f"{P}_*"))
                      if po else []):
                fr = sorted(f.name for f in d.glob("f*.png"))
                if fr:
                    rj = {}
                    try:
                        rj = json.loads((d / "reassembly.json")
                                        .read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        pass
                    re_dirs[d.name] = {"frames": fr,
                                       "fps": rj.get("fps", 5)}
            choreo = sorted(c.name for c in
                            (ROOT / "packages" / P / "choreo").glob("*.json"))                 if P else []
            return self._json(200, {"videos": vids, "reassembled": re_dirs,
                                    "project": P,
                                    "package": P if choreo else None,
                                    "choreo": choreo})
        if self.path == "/api/frames":
            P = active_project()
            if not P:
                return self._json(200, [])
            wd, pbase = pdir(P), f"projects/{P}"
            kp = kp_load()
            frames = []
            for d in sorted((wd / "scenes").glob("scene_*")):
                for f in sorted(d.glob("f*.png")):
                    fid = f.stem
                    rec = kp["frames"].get(fid, {})
                    frames.append({
                        "fid": fid, "scene": d.name,
                        "file": f"{pbase}/scenes/{d.name}/{f.name}",
                        "labelled": bool(rec.get("objects")),
                        "masked": (wd / "letters_sam2_small" / "by_frame" /
                                   fid).is_dir(),
                    })
            return self._json(200, frames)
        if self.path == "/api/keypoints":
            return self._json(200, kp_load())
        if self.path == "/api/library":
            return self._json(200, library())
        if self.path.startswith("/seqanim/"):
            parts = self.path[len("/seqanim/"):].split("/")
            data = None
            if (len(parts) == 2 and NAME_RE.match(parts[0])
                    and parts[1].isdigit()):
                data = seq_thumb(parts[0], int(parts[1]))
            if data is None:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "max-age=600")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path.startswith("/thumb/"):
            parts = self.path[len("/thumb/"):].split("/")
            data = thumb(*parts) if len(parts) == 2 and all(
                NAME_RE.match(x.replace(".", "")) or NAME_RE.match(x)
                for x in parts) else None
            if data is None:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "max-age=3600")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/api/log":
            p = JOB.get("log")
            txt = ""
            if p and Path(p).exists():
                txt = Path(p).read_text(encoding="utf-8",
                                        errors="replace")[-8000:]
            return self._json(200, {"log": txt})
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/dropframe":
            try:
                body = json.loads(self.rfile.read(
                    int(self.headers.get("Content-Length", 0))))
            except (ValueError, json.JSONDecodeError):
                return self._json(400, {"error": "bad body"})
            P = active_project()
            fid = str(body.get("fid", ""))
            if not P or not NAME_RE.match(fid):
                return self._json(400, {"error": "bad fid"})
            import shutil as _sh
            wd = pdir(P)
            removed = []
            for f in wd.glob(f"scenes/scene_*/{fid}.png"):
                f.unlink()
                removed.append(str(f.name))
            _sh.rmtree(wd / "letters_sam2_small" / "by_frame" / fid,
                       ignore_errors=True)
            ovf = wd / "letters_sam2_small" / "overlay" / f"{fid}.jpg"
            ovf.unlink(missing_ok=True)
            with KP_LOCK:
                d = kp_load()
                d["frames"].pop(fid, None)
                kp_save(d)
            MASKTHUMB.pop(f"{P}/{fid}", None)
            return self._json(200, {"ok": True, "removed": removed,
                                    "note": "re-run 3 to rebuild sequences "
                                            "without this frame"})
        if self.path == "/api/assign":
            try:
                body = json.loads(self.rfile.read(
                    int(self.headers.get("Content-Length", 0))))
            except (ValueError, json.JSONDecodeError):
                return self._json(400, {"error": "bad body"})
            P = active_project()
            sid = str(body.get("seq", ""))
            if not P or not NAME_RE.match(sid) or                     not (BENCH / "sequences" / sid).is_dir():
                return self._json(400, {"error": f"unknown sequence {sid!r}"})
            mode = body.get("mode")
            a = assignments(P)
            if mode not in ("solve", "library", "sequence", None):
                return self._json(400, {"error": "mode: solve|library|sequence|null"})
            if mode is None:
                a.pop(sid, None)
            elif mode == "sequence":
                don = str(body.get("donor", ""))
                if not NAME_RE.match(don) or                         not (BENCH / "sequences" / don).is_dir():
                    return self._json(400, {"error": f"{don!r} is not a sequence"})
                if don == sid:
                    return self._json(400, {"error": "an element cannot stand in for itself"})
                a[sid] = {"mode": "sequence", "donor": don}
            elif mode == "library":
                lid = str(body.get("library_id", ""))
                if not any(json.loads(l)["id"] == lid for l in
                           (BENCH / "metadata.jsonl").read_text(
                               encoding="utf-8").splitlines() if l.strip()):
                    return self._json(400, {"error": f"{lid!r} not in library"})
                a[sid] = {"mode": "library", "library_id": lid}
            else:
                a[sid] = {"mode": "solve"}
            save_assignments(P, a)
            return self._json(200, {"ok": True})
        if self.path in ("/api/keypoints", "/api/decision"):
            try:
                body = json.loads(self.rfile.read(
                    int(self.headers.get("Content-Length", 0))))
            except (ValueError, json.JSONDecodeError):
                return self._json(400, {"error": "bad body"})
            with KP_LOCK:
                d = kp_load()
                if self.path == "/api/keypoints":
                    fid = str(body.get("fid", ""))
                    if not fid:
                        return self._json(400, {"error": "no fid"})
                    d["frames"][fid] = {
                        "file": str(body.get("file", "")),
                        "scene": str(body.get("scene", "")),
                        "objects": body.get("objects", {}),
                    }
                else:
                    sc, lab = str(body.get("scene", "")), str(body.get("label", ""))
                    reuse = body.get("reuse")
                    dd = d["decisions"].setdefault(sc, {})
                    if reuse:
                        dd[lab] = {"reuse": str(reuse)}
                    else:
                        dd.pop(lab, None)
                        if not dd:
                            d["decisions"].pop(sc, None)
                kp_save(d)
            return self._json(200, {"ok": True})
        if self.path != "/api/run":
            return self._json(404, {"error": "unknown endpoint"})
        try:
            body = json.loads(self.rfile.read(
                int(self.headers.get("Content-Length", 0))))
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "bad body"})
        step, arg = body.get("step", ""), body.get("arg")
        built = build_job(step, arg)
        if isinstance(built, str):
            return self._json(400, {"error": built})
        jobs, cwd, detach = built
        if detach:
            subprocess.Popen(jobs[0], cwd=cwd, env=env_for(str(jobs[0][0])),
                             creationflags=subprocess.CREATE_NEW_CONSOLE
                             if os.name == "nt" else 0)
            return self._json(200, {"ok": True,
                                    "note": "window opened on this machine"})
        with LOCK:
            if JOB["running"]:
                return self._json(409, {"error": f"busy: {JOB['step']}"})
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            lp = LOG_DIR / f"{time.strftime('%H%M%S')}_{step}.log"
            JOB.update(running=True, step=step, log=str(lp), rc=None,
                       t0=time.time())
        threading.Thread(target=run_jobs, args=(step, jobs, cwd, lp),
                         daemon=True).start()
        return self._json(200, {"ok": True, "log": str(lp)})


def main():
    port = int(sys.argv[sys.argv.index("--port") + 1]) \
        if "--port" in sys.argv else 8478
    print(f"[studio] project: {active_project() or '(none — pick a video)'}")
    print(f"[studio] http://localhost:{port}/")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
