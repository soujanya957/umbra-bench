# umbra-bench — the sequences track

26 animated targets, 276 frames, in two families: 13 generated motions
(128×128, from the parametric generators) and 13 film-cut letters
(512×512, extracted from the demo ad by `fleet-shadow-demo/07_make_sequences.py`
— one letter in one scene is one sequence). A separate track, not a tenth
subset, and the reason is a measurement rather than a preference.

## 0. Why per-frame IoU is not enough

`fleet-shadow-art/motion-aware-shadow/results/3-robot-runs/spinning_star/` is an
already-solved 5-frame clip. Its recorded numbers:

| | |
| --- | --- |
| mean per-frame IoU | **0.6679** — indistinguishable from a good static result |
| worst single-joint step between consecutive frames | **291°** |
| `motion_planner.LARGE_Q_JUMP`, the repo's own feasibility bound | **1.2 rad = 68.8°** |
| transitions that break that bound | **4 of 4** |

Every frame is individually good and the clip cannot be performed: the arms would
have to swing 291° between two frames. Averaging per-frame IoU is precisely the
statistic that cannot see this — it is the temporal version of the problem
[METRICS.md](METRICS.md) §0 already documents for static IoU. So the track reports
a *set* of numbers, and the rule is that **`mean_frame_iou` is never quoted without
`dq_infeasible_frac` beside it**.

Dropping these frames into `targets/` as a subset would have scored them as 119
unrelated static targets and discarded the temporal information silently, which is
worse than not adding them.

## 1. Layout

```
sequences/<id>/f00.png f01.png …      1-bit, dark = shape, same convention as targets/
sequences.jsonl                        the index — one JSON line per sequence
```

| id | frames | loop | mean step IoU | source |
| --- | --- | --- | --- | --- |
| `bird` | 8 | yes | 0.509 | `anim_n3` ≡ `anim_n5` |
| `cheer_n5` | 8 | yes | 0.693 | `anim_n5` |
| `flower` | 8 | yes | 0.590 | showcase |
| `plant` | 20 | **no** | 0.908 | `plant-demo` |
| `reeds_n3` / `reeds_n5` | 8 / 8 | yes | 0.362 / 0.399 | differ per rig |
| `star_spin` | 5 | yes | 0.564 | showcase |
| `stick_wave` | 8 | yes | 0.854 | `anim_n3` ≡ `anim_n5` |
| `triangle` | 10 | **no** | 0.585 | showcase |
| `two_arm_wave_n5` | 8 | yes | 0.761 | `anim_n5` |
| `windmill_n3` / `windmill_n5` | 8 / 8 | yes | 0.234 / 0.425 | differ per rig |
| `wiper` | 12 | yes | 0.210 | showcase |

`anim_n3` and `anim_n5` overlap by name but not always by content — `bird` and
`stick_wave` are byte-identical across the two and are imported once; `reeds` and
`windmill` differ and are kept as separate sequences. The importer decides by
hashing rather than by assumption.

The demo family, `demo_01_scene_XX_<letter>`, is 3–24 frames per sequence at
5 fps (every 5th source frame), each with a `source.json` beside the frames:
the crop that puts a solved shadow back on the 1920×1080 canvas, the source
frame ids, the fps, the glyph — and `loop: false`, which §2 explains. Their
`class` is the bare letter, the same convention as `letters_upper`, so the CLIP
recognizability machinery can score these frames against the same glyph set.

**The generated frames are 128×128 (imported as-is), the demo frames 512×512,
while `targets/` is 512×512.** The generated family comes from parametric
generators (`generate_targets.py`, `generate_rig_targets.py`) that take
`--size` and a continuous phase parameter, so regenerating it at 512 with a
controlled frame count is the natural next step — see §4.

## 2. Schema — `sequences.jsonl`

```json
{
  "id": "star_spin",
  "track": "sequences",
  "class": "star_spin",
  "prompt": "a five-pointed star, rotating",
  "n_frames": 5,
  "frame_size": [128, 128],
  "frames": ["sequences/star_spin/f00.png", "…"],
  "target_motion": {
    "mean_step_iou": 0.564, "min_step_iou": 0.558, "max_step_iou": 0.573,
    "wrap_iou": 0.565, "loop": true, "step_iou": [0.565, 0.558, 0.573, 0.558]
  },
  "frame_attributes": [ { "…33 attributes per frame…" } ],
  "shadows": {
    "hand":      {"frames": null, "joints": null, "captured_at": null, "…": null},
    "teleop":    {"frames": null, "joints": null, "…": null},
    "optimizer": {"frames": null, "joints": null, "run_id": null, "config": null, "…": null}
  },
  "rig": {"n_arms": null, "light": null, "screen_distance_m": null, "camera": null}
}
```

Three fields carry the weight:

**`frames` is ordered and authoritative.** Frame order is data, not filesystem luck.

**`target_motion` is the denominator.** Every shadow-side temporal number is
uninterpretable without it: `wiper` changes by IoU 0.21 per step and `plant` by
0.91, so the same amount of shadow movement means opposite things in the two.

**`loop` changes what must be scored, and it is declared where the source knows
and detected only where it doesn't.** A loop returns to the same *appearance*,
which for a rotationally symmetric shape happens before the frames repeat —
`star_spin`'s last frame is nowhere near identical to its first (IoU 0.565),
yet its wrap step is the same size as every interior step (0.558–0.573). The
detection test is `wrap_iou ≥ 0.9 × mean_step_iou`, not `last == first`.
**If the wrap is not scored, a solver can unwind an entire rotation in the gap
between the last frame and the first and pay nothing.**

The test cannot tell a slow shot from a loop, and it failed exactly that way on
the demo family: `demo_01_scene_05_I` moves so little (mean step IoU 0.994)
that its wrap (0.991) passes the ratio with no loop present, and
`demo_01_scene_06_I` genuinely returns to its starting appearance (wrap 0.912
against steps of 0.447) — a loop by any appearance-based test, except that the
film it was cut from runs once and no last-to-first transition is ever
performed. Scoring a wrap that never happens is as wrong as skipping one that
does. So an importer that knows the answer writes `loop` into `source.json`
and the declaration wins; the wrap test remains for sequences with no
declaration (the generated 13, whose labels it gets right). `target_motion.
loop_source` records which path produced the label — `declared` or
`wrap-test` — so a reader can tell a fact from a heuristic.

`shadows.*.joints` holds the pose per frame. Joint-space continuity is the thing
per-frame IoU cannot see and it is unrecoverable from the masks alone, so the
capture has to carry it.

## 3. Metrics

Three groups. Report them together; do not collapse to one number.

### S1 — per-frame quality

Every metric in `metrics.py` (`iou`, `boundary_iou`, `nsd`, `cldice`,
`betti_error`, …), aggregated over frames as **mean, min, std**.

`min` is not a footnote. A viewer watching a clip sees the worst frame, not the
average — one broken frame in a 12-frame loop is what registers.

### S2 — transitions

Over the N−1 interior steps, **plus the wrap when `loop` is true**.

| metric | what it answers |
| --- | --- |
| `dq_max_deg` | largest single-joint step between consecutive frames |
| `dq_l2_deg` | ‖Δq‖ per transition, over all 6·n_arms joints |
| **`dq_infeasible_frac`** | fraction of transitions where any joint exceeds `LARGE_Q_JUMP` |
| `shadow_step_iou` | IoU(Sₜ, Sₜ₊₁) — how much the shadow actually moved |
| `motion_excess` | `target_step_iou − shadow_step_iou`; positive = the rig moved more than the animation asked for |

`dq_infeasible_frac` is the headline. It is a hard physical bound rather than a
smoothness preference, and `optimizer.py` already carries the constant and a
`reachability_penalty` that fires on it — the penalty is simply not applied when
frames are solved independently, which is how `spinning_star` reached 4/4.

`motion_excess` is the normalised form of "is it thrashing", which is why
`target_motion` is in the schema at all.

### S3 — sequence level

| metric | what it answers |
| --- | --- |
| `assignment_stability` | does each arm keep covering the same part of the shape, or do arms swap roles mid-clip? |
| `loop_closure` | for `loop: true`, the wrap transition scored like any other |
| `total_path_length` | Σ `dq_l2_deg` — what performing the clip costs |

Arm swaps deserve their own line: the Hungarian assignment is per-solve, so
nothing currently stops two arms trading regions between frames. On screen that
reads as the arms crossing, and per-frame IoU is completely blind to it.
`sequence_metrics.py` measures it in joint space: a transition where relabelling
the arms (Hungarian on the per-arm joint vectors) beats the identity assignment
is a swap. On `spinning_star` that is every transition — `assignment_stability`
0.0 on a clip whose mean frame IoU says nothing is wrong.

### Running it

All three groups are `scripts/sequence_metrics.py`. Two input shapes:

```bash
# a fleet-shadow-art clip run (summary_*.csv + frame_*/best_shadow.png beside it);
# --sequence links it to this index for the target frames and the loop flag
python scripts/sequence_metrics.py \
    --run ../fleet-shadow-art/motion-aware-shadow/results/3-robot-runs/spinning_star \
    --sequence star_spin

# every shadows.<source>.frames slot filled in sequences.jsonl
python scripts/sequence_metrics.py
```

Output is `results/sequence_metrics_<tag>.csv`, one wide CSV in the
compute_metrics.py mould: one `row=frame` line per frame (its S1 metrics plus
the transition leaving it) and one `row=aggregate` line per (sequence, source).
The joint columns need the run's `q_r*_j*_deg` columns (or `shadows.*.joints`)
and degrade to null without them; `LARGE_Q_JUMP` is read from the
fleet-shadow-art checkout (`--repo`, auto-found beside this repo), never copied.

New sequences — the demo letter clips included — join the track the same way
the first thirteen did: frames into `sequences/<id>/f00.png …`, then
`build_sequence_metadata.py` in the eval env (so `frame_attributes` fill and
`loop` is detected), and the id is scoreable. Verified against the recorded
`spinning_star` numbers: mean frame IoU 0.6679, per-transition dq_max
[160.9, 291.0, 116.6, 157.6]°, infeasible 5/5 with the wrap.

## 4. Not done yet

1. **Regenerate at 512 with controlled frame counts.** These 13 are hand-authored
   phases of parametric generators. `make_star(n_points, outer_frac, inner_frac,
   rotation_deg)` alone spans limb count, thinness and phase — the same
   parametric-and-controlled construction the `digits` and `letters` subsets use.
   13 motions is a pilot, not a benchmark; a generated family is how it becomes one.
2. **The `hand` source is partly undefined here.** A person can hand-cast a static
   silhouette; whether they can hand-cast a 12-frame loop is a different question
   per sequence. Expect the slot to stay sparse and say so rather than reporting a
   mean over whatever happens to be filled.
3. **Recognizability of *motion*.** A spinning star should read as spinning. The
   Part C ladder in [METRICS.md](METRICS.md) scores stills; the temporal analogue
   is open.
