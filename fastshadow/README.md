# fastshadow

GPU shadow rendering, gradient and batched solvers, objectives and a learned human judge for
the umbra-bench shadow rig (three SO-101 arms, a point light, a flat wall).

The rig's shadow is the arms' triangles projected from the light onto the wall plane, a
linear map in homogeneous coordinates, so a whole CMA-ES population or a batch of Adam
restarts renders in one nvdiffrast call. MuJoCo is read once for geometry; it is still used
for contact checks and as the reference renderer every reported number is measured with.

## Modules

| file | what |
|---|---|
| `renderer.py` | `FastShadowRenderer`: batched FK (level-order, CPU), 2 mm vertex clustering, hard masks and antialiased (differentiable) masks; optional movable bases (`arrangement=`). Matches MuJoCo at IoU 0.995. |
| `cma_bridge.py` | `BatchedShadowRenderer`: lets the staged CMA-ES render populations on nvdiffrast (MuJoCo still does kinematics and contacts). Needs the fleet-shadow-art optimizer hooks, see *Setup*. |
| `grad_opt.py` | Adam solver on the soft renderer: multi-start, blurred-to-sharp stages, re-seeding, sphere collision proxy calibrated on contact-free solves, MuJoCo re-scoring of the finalists. |
| `objectives.py` | Shadow objectives batched on the GPU (IoU, Tversky, two-sided DT `dt2`, boundary IoU, clDice, holes, placement-free IoU, classifier, CLIP, combinations, Ivy reader), and `use_objective` to swap one into the whole CMA-ES pipeline. |
| `spacing.py` | Robot arrangements (column, row, triangle, diagonal) with two spacings optimized alongside the joints (the LeSlider study); `SpacedMJ` keeps MuJoCo's contact check on the moved bases. |
| `glyph_cls.py` + `.pt` | Small A-Z / a-z / 0-9 classifier trained on system fonts only (never the DejaVu targets); 100% on the benchmark targets. |
| `ivy_reader.py` + `.pt` | The Ivy judge: a logistic regression on 11 shadow measurements (8 shape, 3 from the glyph classifier, so letters and digits only), fitted to Ivy's marks, reads and A/B picks. For evaluation and selection only; optimizing it is gamed. `ivy_mix` (half judge, half IoU) is the one judge-based objective kept in the study. |
| `arrangement_report.py` | Aggregation for the arrangement section of the report. |

## Scripts (in `scripts/`)

| script | what |
|---|---|
| `fastshadow_parity.py` | nvdiffrast vs MuJoCo masks and timing |
| `compare_grad.py` | Adam vs stored CMA-ES solves on letters + digits (writes `results/fastshadow/compare_<tag>`) |
| `compare_fast_cma.py` | CMA-ES on MuJoCo vs on nvdiffrast, and Adam's start-point sweep |
| `objective_study.py` | `run` (CMA-ES), `run_adam`, `grade`: every objective on every glyph, MuJoCo-graded on 11 metrics |
| `arrangement_study.py` | arrangements x optimizers on digits, spacing optimized with the joints |
| `train_ivy_reader.py` | trains the judge from a picker export, cross-validated by held-out characters |
| `ivy_judge.py` | adds the judge's score to every graded result |
| `analyze_guesses.py`, `analyze_ab.py`, `compare_marks.py`, `human_summary.py` | human-test analyses |
| `make_fastshadow_report.py`, `make_objective_report.py`, `make_pick_page.py` | the three report / picker pages |

Results go to `results/fastshadow/` (gitignored). Human data comes from the picker page's
database; export it with ArtifactData into `results/fastshadow/picks_v*/`.

## Setup

* Env: `fh_l1` (torch cu128). nvdiffrast is built from source: `tools/build_nvdiffrast_windows.bat`.
* fleet-shadow-art: CMA-ES on nvdiffrast and the objective swap need two optimizer hooks
  (`eval_q_batch`, `batch_shadow_loss`). They are in `patches/` until they land upstream:
  `git am fastshadow/patches/*.patch` on a fleet-shadow-art checkout, then point
  `FLEET_SHADOW_ART` at it. A plain `ShadowRenderer` takes the old per-pose path.

## Findings (letters and digits, details in the reports)

* Renderer: same shadow as MuJoCo (IoU 0.995), 0.10 vs 0.74 ms per pose batched; CMA-ES 1.2 vs 3.3 s per solve at the same IoU.
* Adam improves with start points (0.68 at 8 to 0.73 at 256) but at equal time best-of-k fast CMA-ES is ahead on IoU. People, blind, preferred Adam's shadows (A/B 61%, p = 0.011; read 55% vs 45%, p = 0.044).
* Objectives: nothing beats IoU-type objectives on IoU; semantic objectives (classifier, CLIP, the Ivy judge) get gamed when optimized. `dt2` is the safe default.
* Arrangements (digits): the column stays best; optimizing its spacing adds about 0.006 IoU, a per-digit choice of arrangement 0.018. CMA-ES beat Adam on all 10 digits and pushes spacing to the 0.12 m floor.
* The Ivy judge predicts held-out human marks / reads / A/B choices at AUC 0.81 / 0.82 / 76%, above every single metric.
* Floor: the solver's floor term was pinned by the static base at -1.83 cm, so 10.7% of stored solves put the gripper or jaw more than 5 mm into the table (all flagged collision-free). `BatchedShadowRenderer(floor="mesh")` / `objective_study.py --floor mesh` uses the moving bodies' real lowest point: penetration 10% to 0% on 30 glyphs x 3 seeds, best-of-3 IoU -0.003 (95% -0.015..+0.007), 1.32 to 1.47 s per solve.
* More seeds beat more objectives: best of 12 dt2 seeds 0.754 IoU vs best of the 13-objective menu 0.740 (+0.015, 95% +0.005..+0.025), tc_iou 0.687 vs 0.652.
* Joint error: 1 degree costs 0.013 IoU on average, 2 degrees 0.035; the best solve is almost always also the most robust, so robustness need not be a selection term (`scripts/robustness_check.py`).
* SDS does not work here: SD 1.5 and SDXL cannot tell clean letters apart (top-1 at chance) and SD 1.5's score agrees with human marks at AUC 0.47 (`scripts/sds_score.py`).

Adam tuning (8 glyphs, 1 seed, mean IoU vs CMA-ES 0.744): 32 starts x 200 steps 0.727 (2.8 s); 64 x 300 0.744 (5.2 s); 128 x 300 0.760 (7.9 s); lr 0.1 0.717; 64 x 300 with re-seeding 0.777 (5.4 s, chosen); 128 x 200 0.748; 128 x 200 re-seeded 0.769.

Report pages: solver bench https://claude.ai/artifact/HU5HFUjpvTf9BTnkTveK7i ·
objective study https://claude.ai/artifact/ChM2otn7JGKKN1sy2yJsY8 ·
picker https://claude.ai/artifact/DTjJuNUhPWfJN5hRrTWmSX
