# fleet-shadow-demo — a film clip becomes a sequences-track dataset

Video in, animated shadow targets out, with exactly one human step in the middle.

```bash
python run_demo.py --video FAMILY_trimmed.mp4 --demo-id 02
```

It runs every stage that can be automated, stops at the one that cannot, and
resumes when you run it again. Stages whose output already exists are skipped, so
re-running after fixing a few labels costs only the stages downstream of them.

| | stage | script | |
| --- | --- | --- | --- |
| 1 | split scenes, extract frames | `01_split_scenes.py` | auto |
| 2 | **label one object per glyph** | `03_label_keypoints.py` | **you** |
| 3 | segment from the keypoints | `04_sam_segment.py` | auto |
| 4 | clean the masks | `06_clean_masks.py` | auto |
| 5 | group into sequences, crop per clip | `07_make_sequences.py` | auto |
| 6 | index the track | `../scripts/build_sequence_metadata.py` | auto |
| 7 | *(after solving)* put the shadows back on the source canvas | `08_reassemble.py` | auto |
| 8 | score whether the shadow reads as the letter | `09_clip_score.py` | auto |
| 9 | composite the scenes back into video | `10_compose_video.py` | auto |

Step 2 is manual because deciding *which* shape in a frame is the subject is not
something the pixels answer. Everything else is.

Output lands in the repo's **sequences track**, not in `targets/`. SEQUENCES.md is
explicit that an animation is "a separate track, not a tenth subset": dropping
frames into `targets/` scores them as unrelated static targets and discards the
temporal information silently.

```
sequences/demo_01_scene_04_M/f00.png …    1-bit, dark = shape
sequences/demo_01_scene_04_M/source.json  crop box, source frame ids, fps, loop
sequences.jsonl                           the index, built by stage 6
results/demo_review/_all.png              every sequence, one glance
results/demo_reassembled/<seq>/f00.png …  stage 7, on the 1920x1080 canvas
```

---

## The manual step

```bash
python 03_label_keypoints.py
```

**Label each glyph as its own named object** — `F`, `A`, `M`, `I`, `L`, `Y` — not
several points inside one object.

This is the single most consequential thing you do here, and getting it wrong is
silent. SAM answers the question it is asked. Points for six glyphs inside one
object asks for "the thing containing all of these", and it returns exactly that:
one 38 000-pixel mask spanning x 1189→1794, the whole word as a single blob. No
larger checkpoint fixes it — `sam2.1_hiera_large` already is the largest — because
nothing is failing.

With one object per glyph, `04_sam_segment.py` prompts SAM once per object and the
letters come out separate.

If you have already labelled a clip the other way, `05_split_objects.py` recovers
it: it prompts once per point with the siblings as negatives, then merges the
results that turn out to be the same object. The merge rule is read off the masks
rather than guessed — two clicks refining one figure produce masks overlapping
97.4%, two clicks on different glyphs produce 0.3%. It is a fallback; label
properly and you never need it.

---

## What the defaults encode

Each of these was measured on this footage. They are the reason the stages have
the shape they do.

### SAM2 **small**, not large

Over the same 159 masks:

| | interior holes | detached fragments | roughness | time |
| --- | --- | --- | --- | --- |
| `hiera_large` (898 MB) | 866 | 707 | 2.02 | ~3 min |
| **`hiera_small` (184 MB)** | **213** | **54** | **1.96** | **66 s** |

Four times fewer holes, thirteen times fewer fragments, smoother, and three times
faster. The large model resolves the glyph's own dark outline strokes and its
anti-aliased edge and excludes them — precisely the detail a silhouette target
does not want. Bigger is worse here. `--sam-model large` re-checks it on new
footage.

### One crop per sequence, never per frame

A glyph travels 39–329 px horizontally and 39–233 px vertically inside its scene,
and that motion **is** the content. Cropping each frame to its own bounding box
would centre every frame identically and play back as a shape twitching in place.

Stage 5 computes one box over the union of each sequence and applies it unchanged
to every frame, so relative position and relative size survive.

The box is not squared before cropping. `scene_06`'s word is 1638×272; a square
about its centre starts 564 px above the top of a 1080-line frame. The crop is the
union rectangle clamped to the frame, and the squaring happens by padding
afterwards — identically for every frame, so the padding cannot introduce motion
of its own.

### Every target is one connected shadow

A rig cannot cast a floating piece. Stage 4, in this order:

1. **keep parts** — anything ≥3% of the largest survives; specks go. The cane in
   `scene_03` is 3.2–3.7k px against a 63k body, so "largest only" deleted it from
   every frame while those same frames carry 280–390 single-pixel specks.
2. **bridge** — whatever is still detached is joined to the body along the
   shortest gap.
3. **fill holes** — the dark outline strokes read as interior holes.
4. **smooth** — Gaussian on the binary field, thresholded at 0.5. Symmetric, where
   a morphological close alone grows the shape and an open alone shrinks it.
5. **sweep again** — smoothing can shed a two-pixel speck off a thin tip.

Order matters: filling before dropping would keep the specks, smoothing before
either would lock them into the shape.

`--sigma` is the smoothing radius. 3 is the default; measured across 0–8, the
letters keep their corners up to about 4 and visibly round past 6.

### The outline stays

`06_clean_masks.py --colour` shrinks each mask to the glyph's own hue (S ≥ 150,
H 15–40, measured). It is **off by default**: the dark border is part of the
shape, and removing it also severs the cane at the stroke where it meets the hand,
opening a 5–8 px gap that then has to be bridged back. The flag remains for
footage where the drop shadow really is inside the mask.

---

## Solving

The targets are a sequence, so solve them as one. `run_sequence.py` keeps the
renderer, the shadow forward model and the previous frame's pose across frames;
`run.py` in a loop throws all of that away between frames.

Measured across all 26 clips against a per-frame-independent baseline, the
chained solve wins on both axes at once: 0.629 mean frame IoU against 0.564,
and 2% of transitions infeasible against 97%. The expectation that independent
per-frame solving buys frame quality at the cost of playability did not
survive contact — it lost on frame quality too, in 19 of the 26.

**The exception is worth knowing before you trust the default.** On `wiper`,
the fastest-stepping clip in the set — consecutive targets overlap at IoU 0.21
— the independent baseline scores 0.472 per frame against the chain's 0.327.
Three separate mechanisms pull each frame toward its predecessor: the ICP
warm-start from `prev_q`, the reachability hinge against `prev_q`, and
`skip_greedy` when that warm start is accepted. On a slowly-moving clip all
three are right. On a target that genuinely moves a long way each frame they
are three reasons not to follow it, and the clip under-moves. The same clip is
the one where closing the loop pushed the previous transition to 131.7 deg:
where the frames really do demand large joint travel, the continuity anchor and
the target are pulling against each other, and that shows up in both
measurements. If a clip steps that fast, check the per-frame arm before
assuming the chain.

```bash
cd ../../fleet-shadow-art/motion-aware-shadow
python scripts/run_sequence.py --urdf urdf/SO101/so101_new_calib.urdf --targets ../../umbra-bench/sequences/demo_01_scene_04_M/f*.png --n-robots 3 --arm-gap 0.2 --size 128 --fit-target --fit-max-shift 0.45 --reach-samples 300 --prior ../../umbra-bench/optimized/big-budget-grounded/letters_upper/M_dejavuserif-bold_v2/results.json --popsize 192 --phase1-iters 128 --phase2-iters 128 --final-iters 256 --beta 0.3 --gamma 0 --delta 0 --reachability-penalty 100 --outdir ../../umbra-bench/optimized/demo_01_scene_04_M
```

Four flags carry most of the result:

**`--arm-gap 0.2`** — not optional and not defaulted. `renderer.py` falls back to
0.15, which is a different rig: the throw becomes 3.30 m instead of 3.40 and the
shadow comes out the wrong size. Nothing errors; `summary.json` simply records
`arm_gap: null`.

**`--fit-target`** — one similarity transform for the whole clip. Without it the
glyph spans rows 9–118 of a 128-row frame while the rig can only cast rows 81–127,
so the solver is asked for something unreachable and more search does not help:
the same clip scores 0.563 at popsize 192 and 0.605 at popsize 16. With it,
`uncastable` drops 12.3% → 2.9% and IoU reaches 0.75–0.82.

**`--prior`** — seeds frame 1 from the benchmark's solve for that letter. Frames
2+ already chain off their predecessor, so only the first starts cold, and the
benchmark has already solved every uppercase letter well (I 0.887, Y 0.829,
L 0.798, A 0.797, F 0.787, M 0.756 under
`optimized/big-budget-grounded/letters_upper/`).

**`--reachability-penalty 100`** — the temporal barrier. Set it to 0 for the
per-frame arm, where each frame is solved for its own best appearance and the
transitions are allowed to be whatever they are.

**`--loop-close`** — for a clip that loops, and only then. The frame loop is a
forward pass, so every frame is anchored to its predecessor and the last-to-first
transition is anchored to nothing — which is the transition that fails: in the
two archive runs whose only infeasible step was the wrap, it jumped 95.5° and
71.2° against a 68.75° bound. The flag re-solves the final frame against frame
1's pose and keeps the result only if it is reachable from *both* neighbours, so
the wrap can improve or stay as it was but never quietly get worse; `loop_anchor`
in the summary records which happened. The demo clips are cuts from a film and do
not loop, so they do not want it. `--export-clip-loop` is a different thing — it
adds a closing morph at bake time, which makes an unplayable wrap look smooth.

### Back onto the source canvas

```bash
python 08_reassemble.py --all
```

Solving is not the last step, and the step after it is not a paste.

`source.json` ties each sequence to a rectangle on the 1920x1080 source, so a
solved shadow can go back where the glyph was. But `--fit-target` applied a
similarity transform to the whole clip before solving, so the shadow lives in
fitted coordinates -- scale 0.64-1.02, shift 8-41 px in a 128 px frame -- and
pasted through the crop unchanged it lands visibly displaced and resized
against the footage it came from.

The fit is a similarity transform, so it inverts exactly. Inverting it per frame
before the crop recovers the placement in full:

| clip | pasted as solved | inverted first | the solve, vs its fitted target |
| --- | --- | --- | --- |
| `scene_06_A` | 0.187 | **0.825** | 0.828 |
| `scene_06_L` | 0.000 | **0.561** | 0.548 |

So the gap between scoring against the authored frame and against the fitted one
is placement, not solve quality, and it does not reach the video. It also keeps
two things apart that the pipeline had run together: the rig casts wherever it
can reach, and the letter still lands where the ad put it.

`--check` (on by default) round-trips an authored frame through the fit and its
inverse instead of trusting the derivation -- 0.988 on `scene_06_A`, 0.951 on
the thinner `scene_06_L`, the difference being resampling.

### Is it legible?

```bash
python 09_clip_score.py
```

IoU is not the demo's quality number and this set shows why. `scene_05_M`
scores 0.681 and CLIP cannot name it; `scene_06_A` scores 0.828 and is read
correctly as often as the authored frame is. IoU measures overlap with a
silhouette the clip fit was free to move — legibility is a different question.

Retrieval rank over the benchmark's own 49 classes, top-1 and MRR, chance
0.0204. The class set, the alias map and `recognizability_ratio` all come from
`tests/clip_eval.py` and `scripts/semantic_metrics.py` rather than being
rewritten here, and the authored frames are scored through the identical path
in the same run as the ceiling — CLIP is half-blind to 1-bit silhouettes, so a
low absolute score cannot separate "the shadow is poor" from "the judge cannot
see either". **The ratio is the quotable number, not the raw score.**

Passing the repo's `glyph_prompt` is not a detail. With the default template
the prompt reads "a shadow of a A", which never says the label is a letter, and
the ranking is noise: a clean authored A scored top-1 0.000 while an illegible
four-pixel L fragment scored 1.000.

The three I clips print `--` rather than a ratio. I/l/1 fold to one class
because they are the same picture, and the prompt then asks for "the digit 1",
which a serif capital I from a title card is not — so the authored frame misses
too and there is no denominator. That is the class design, not the solve.

### The video

```bash
python 10_compose_video.py
```

The sequences track holds one letter per clip, but the ad does not: scene_06 is
F A M I L Y being spelled out, with L entering at source frame 0712 and Y at
0717. Stage 7 put every shadow back at its true position on the 1920x1080
canvas, which is what makes reassembling the shot possible — the letters go
back into the same frame and the word assembles.

**Alignment is by source frame id, never by index.** A letter is only tracked
while it is on screen, so scene_06 has four frames of L against ten of A;
pairing f03 with f03 would put different moments of the ad in one picture.
`reassembly.json` carries `source_frame_ids` and that is the key.

Written with OpenCV rather than ffmpeg, which is not installed here. One mp4 per
scene plus `demo_01.mp4` for the whole cut, 92 frames. The rate is the
sequences' own 5 fps — the shot was sampled every 5th frame of 25 fps footage,
so that is real time, and playing it faster would show motion nothing was
solved for.

```
results/demo_video/scene_01.mp4 … scene_06.mp4
results/demo_video/demo_01.mp4
```

### Scoring

```bash
cd ../../umbra-bench
python scripts/sequence_metrics.py --run optimized/demo_01_scene_04_M --sequence demo_01_scene_04_M --tag demo_01_scene_04_M
python scripts/_build_sequences_payload.py
python atlas/build_atlas.py
```

Two IoUs appear and they disagree on purpose: `summary.json`'s is against the
*fitted* target, `sequence_metrics.py` recomputes against the authored frames in
`sequences.jsonl`. That is the temporal form of the `ref=shown` / `ref=original`
split in METRICS.md. Both are in the CSV.

**`mean_frame_iou` is never quoted without `dq_infeasible_frac` beside it.**
SEQUENCES.md §0 records why: a 5-frame clip scored 0.6679 per frame — a
respectable static number — while 4 of its 4 transitions demanded a 291° swing
from a single joint, against a 68.8° feasibility bound. Every frame was good and
the clip was unplayable.

---

## Files

```
run_demo.py               the driver — start here
01_split_scenes.py        video → scenes/<scene>/f####.png + frames_manifest.csv
01_extract_frames.sh      the un-split variant, every frame at source fps
02_segment_letters.py     HSV + hue/area segmentation. Superseded by SAM2 but kept:
                          it is faster, needs no checkpoint, and separates letters
                          by colour where SAM needs a prompt per object.
03_label_keypoints.py     the manual step
04_sam_segment.py         SAM2 (or HF SAM 1 via --backend hf), one prompt per object
05_split_objects.py       FALLBACK for keypoints that lumped several glyphs into one
06_clean_masks.py         specks, bridges, holes, smoothing — one connected shadow
07_make_sequences.py      per-sequence crop → the sequences track
08_reassemble.py          solved shadows → the 1920x1080 canvas, fit inverted
09_clip_score.py          retrieval rank: does the shadow read as the letter
10_compose_video.py       scenes recomposited from the per-letter clips -> mp4
scenes/                   extracted frames
letters_sam2_small/       SAM output
letters_clean/            final masks, flat, one directory
review/                   per-scene contact sheets from stage 1
```

`keypoints.json` is the only file worth backing up by hand — everything else is
regenerable from it and the video.


---

## Deploying the demo

Two deliverables, two paths. "Build the atlas" covers the first one only.

### A. The dashboard (atlas) — yes, one build and it's there

Every solve's metrics CSV is committed, so on this machine the board rebuilds
from two commands (eval env, see SETUP.md):

```bash
python scripts/_build_sequences_payload.py
python atlas/build_atlas.py            # -> atlas/atlas.html, open it
```

That page IS the deployed dashboard: all 26 sequences with animated
target/shadow/overlay plates, both IoU references, legibility, and the
guide. To put it at the shared URL, publish `atlas/src/atlas.fragment.html`
(built by `python atlas/build_atlas.py --bare`) to the existing artifact —
from Claude Code, `/artifacts` lists it; republishing the fragment to the
same URL updates it in place.

Fresh-checkout caveat: `results/` is gitignored and the atlas also needs
`browser_payload.json` and `teleop_payload.json`, which are NOT committed —
a new machine runs the five-command chain in atlas/README.md once first.

### B. The demo video — the atlas does not produce this

The video is its own two commands, downstream of the solves in `optimized/`.
That directory is **not** gitignored -- most of it is committed, 15849 files of
earlier sweeps -- but the demo solves in it are untracked, so a fresh checkout
has the sweeps and not these, and re-solves via ## Solving. (`results/` *is*
gitignored, line 7; the CSVs under it are committed anyway because a tracked
file overrides the rule.)

```bash
python fleet-shadow-demo/08_reassemble.py --all     # fit-inverse, back onto the 1920x1080 canvas
python fleet-shadow-demo/10_compose_video.py        # -> results/demo_video/*.mp4, one per scene + full cut
```

Stage 7 (`08_reassemble.py` — the file numbers and the stage numbers differ,
since stages 1-6 use scripts 01-07) applies each clip's recorded fit inverse so
the letters land where the ad drew them: 0.187 -> 0.825 IoU against the
authored frames on scene_06_A, against the solver's own 0.828, so the recovery
is complete to within resampling. `--check` round-trips the transform rather
than trusting the derivation. Stage 9 (`10_compose_video.py`) composites
letters that shared a shot back into one frame, aligned by source frame id —
scene_06 is F A M I L Y assembling, not six separate clips. Output is real-time 5 fps by construction; it looks
slow because it is.

Decisions already taken (see results/OVERNIGHT_NOTES.md): the demo is a
rendered composite (size/position adjustable in post — which is exactly what
stage 8 does), the wide-fit solves are the ship set, `star_spin` ships at
fit floor 0.85 and must NOT get a loop bake.


### C. Deploying a sequence to the robots

> **GATE — verify the physical base layout before executing anything.**
> Every solve in this repo assumes the renderer's default rig: three arms in
> a single line along the light→screen depth axis at x=0 —
> `SR-A (0.0, 0.0) · SR-B (0.0, −0.2) · SR-C (0.0, −0.4)`, light_y +1.0,
> wall −2.4 (verified against `renderer._default_base_positions`). The
> per-arm magnification spread of that line is what makes the shadows work.
> **No committed config describes this layout** — `lab_default` spreads the
> arms 0.6 m laterally, `test_stage` differs again, `old_man` rotates two
> bases. A pose executed on any of those casts a different shape. Before
> deploying: either arrange the rig as above and write that config, or
> re-solve against the real base positions (`--base-positions`).
>
> **TODO (user, lab facts no session can supply):** which config matches the
> current physical stage; how Play/render_server is started; which SR10x
> units map to the three solved arms.

The solved clips are already in the robot pipeline's native currency: every
`optimized/<clip>/frame_NN_<ts>/shadow_result.json` is a SCHEMA.md
"Solution" (per-robot `q_rad` keyframe). Two ways to run one on hardware,
both in `fleet-shadow-art/motion-aware-shadow/deploy/`:

**Bake a unified clip, then play it** (recommended — Play does zero conversion):

```bash
python deploy/export_clip.py --urdf urdf/SO101/so101_new_calib.urdf     --ts <TIMESTAMP> --out ../leRobot-control/recordings/<name>.json
```

`--ts` is the timestamp shared by the frame dirs (e.g. `20260902_030833`
for star_spin). This runs `motion_planner.plan_all_transitions` — the same
collision-safe staggered/quintic machinery the dq metrics validate against —
and bakes hold → morph → hold into the clip envelope (hz, per-robot frames,
stage config embedded). The clip then loads in Play / Shadow_robot_ui like
any recording; the physical stage config (ports, base placement — see
`leRobot-control/configs/`) rides inside it.

**Or drive the keyframes live**:

```bash
python deploy/deploy_animation.py --urdf urdf/SO101/so101_new_calib.urdf     --ts <TIMESTAMP> --hold 1.0 --morph 1.5 --dry-run   # plan + print first
```

Drop `--dry-run` to execute. Hardware notes, all from tonight's measurements:

- **star_spin: never `--loop`.** Its wrap is 151.7° — accepted for the video
  precisely because it is not performed; a looped export would ask the arms
  to perform it.
- **Two transitions sit inside the planner model's error margin** (star_spin
  68.6°, flower 68.4°, against the 68.75° direct-connect bound). First
  hardware pass: lengthen `--morph`, watch those steps.
- The planner can return a path level marked unsafe rather than refusing;
  `--dry-run` shows the plan — read it before the arms do.
- On hardware the rig casts at the FITTED position — the fit inverse is a
  video-compositing step and does not exist physically.
