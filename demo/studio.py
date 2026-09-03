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
        set_active(sanitize(v.stem), arg)
        return [[PY_EVAL, "01_split_scenes.py", str(v)]], ROOT, False
    if step == "label":
        return [[PY_EVAL, "03_label_keypoints.py"]], ROOT, True
    if step == "segment":
        ckpt, cfg = SAM_SMALL
        return ([[PY_GPU, "04_sam_segment.py", "--backend", "sam2",
                  "--device", "cuda", "--sam2-checkpoint", ckpt,
                  "--sam2-config", cfg, "--out", "letters_sam2_small"]],
                ROOT, False)
    if step == "clean":
        return ([[PY_EVAL, "06_clean_masks.py", "--in", "letters_sam2_small",
                  "--out", "letters_clean", "--sigma", "3.0"]], ROOT, False)
    if step == "sequences":
        if not P:
            return "no active project — run 'scenes' with a video first"
        return ([[PY_EVAL, "07_make_sequences.py", "--in", "letters_clean",
                  "--prefix", f"{P}_"]], ROOT, False)
    if step == "index":
        return ([[PY_EVAL, str(BENCH / "scripts" /
                               "build_sequence_metadata.py")]], BENCH, False)
    if step == "route":
        return [[PY_EVAL, "route_motion.py"]], ROOT, False
    if step == "solve":
        sid = arg or ""
        if not NAME_RE.match(sid) or not (BENCH / "sequences" / sid).is_dir():
            return f"unknown sequence {sid!r}"
        lane = routing().get(sid, {}).get("lane", "dynamic")
        jobs = []
        if lane == "translation" and not sid.endswith("_stab"):
            jobs.append([PY_EVAL, str(BENCH / "scripts" /
                                      "stabilize_sequence.py"), "--ids", sid])
            solve_id = sid + "_stab"
            # frames won't exist until the stabilizer ran; resolved at runtime
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
        return jobs, MAS, False
    if step == "reassemble":
        return ([[PY_EVAL, "08_reassemble.py", "--all"],
                 [PY_EVAL, "08_reassemble.py"] + sum(
                     [["--sequence", d.name] for d in
                      sorted((BENCH / "sequences").glob("*_stab"))
                      if (BENCH / "optimized" / d.name).is_dir()], [])],
                ROOT, False)
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
                    (ROOT / "out" / "reassembled" / use).is_dir():
                clips += ["--clip", use]
        if not clips:
            return "nothing solved+reassembled for this project yet"
        return ([[PY_EVAL, "pack.py", "--name", P, "--force"] + clips],
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
    JOB.update(running=False, rc=rc)


def state() -> dict:
    P = active_project()
    n = lambda pat, root=ROOT: len(glob.glob(str(root / pat)))
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
    for d in sorted((BENCH / "sequences").glob("*_scene_*")):
        sid = d.name
        run = BENCH / "optimized" / sid
        seqs.append({
            "id": sid,
            "project": sid.split("_scene_")[0],
            "lane": rt.get(sid, {}).get("lane"),
            "solved": bool(list(run.glob("summary_*.json"))),
            "reassembled": (ROOT / "out" / "reassembled" / sid /
                            "reassembly.json").exists(),
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
                            (ROOT / "out" / "video").glob(f"{P}*.mp4")) if P else [],
            "package": bool(P and (ROOT / "packages" / P).is_dir()),
        },
        "job": {k: JOB[k] for k in ("running", "step", "rc")} |
               {"elapsed": round(time.time() - JOB["t0"]) if JOB["t0"] and
                JOB["running"] else None},
    }


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
    LIB_CACHE = rows
    return rows


def thumb(subset: str, stem: str) -> bytes | None:
    key = f"{subset}/{stem}"
    if key in THUMBS:
        return THUMBS[key]
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
        if self.path == "/":
            self.path = "/studio.html"
            return super().do_GET()
        if self.path == "/api/state":
            return self._json(200, state())
        if self.path == "/api/library":
            return self._json(200, library())
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
