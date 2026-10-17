"""Compose results/fastshadow/research/research.json: the "overnight research" section of the
objective report, in plain words, with every number read from the result files.

    python scripts/make_research_notes.py && python scripts/make_objective_report.py --viz-subset digits
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastshadow  # noqa: E402

FS = os.path.join(fastshadow.BENCH, "results", "fastshadow")
OUT = os.path.join(FS, "research", "research.json")

GLOSSARY = [
    ("Overlap (IoU)", "the area shadow and target share, divided by the area they cover together; 1 is perfect"),
    ("tc_iou", "the same, after moving and resizing the shadow onto the target, so it measures shape only"),
    ("dt2", "our default objective: overlap plus a penalty for stray shadow far outside the letter and for missing "
            "parts deep inside it"),
    ("CMA-ES", "our search method: try a cloud of poses, move the cloud toward the better ones, repeat"),
    ("Adam", "the gradient method: follow the slope of the score"),
    ("A try", "one complete solve from a different random start"),
    ("AUC", "take one shadow you marked good and one you did not; how often a score ranks the good one higher "
            "(0.5 = coin flip, 1 = always)"),
    ("95% range", "where the true value most likely lies, given how many glyphs we tested"),
    ("Gamed", "the solver learns to please a score with shadows people cannot read"),
]


def load(*p):
    f = os.path.join(FS, *p)
    return json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None


def png_uri(path, w=180):
    import base64
    import io

    from PIL import Image

    im = Image.open(path).convert("RGB")
    im.thumbnail((w * 2, w * 2))
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def frac(s):
    """'192/1772 = 10.8%' -> 0.108"""
    return float(s.split("=")[-1].strip().rstrip("%")) / 100


def plan_item():
    v = load("research", "verify_claims.json")
    s = load("research", "seeds_vs_objectives.json")
    x = load("research", "extra_numbers.json")
    if not (v and s and x):
        return None
    bk, ab = v["best_of_k"][0]["numbers"], v["ab_follows_iou"][0]["numbers"]
    curve = [float(c) for c in bk["fixed_allsubsets_k1_2_4_8_16"].split("/")]
    g36 = [float(c) for c in bk["glyph36_full_k1_to_k5"].split("@")[0].split("/")]
    g36_long = float(bk["glyph36_full_long_k1_to_k3"].split("/")[0])
    kc = s["k_curve"]
    p0_ci = json.loads(ab["P0_wald_ci"])
    ns_ci = json.loads(ab["nonsemantic_P0_glyphboot_ci"])
    n_adam = round(ab["adam_raw_share"] * ab["decided_pairs"])
    return {"title": "What is the most sensible way to optimize?", "verdict": "plan",
            "what": "We had 15 AI agents work through it in stages. Four read the code and every result file. Five each "
                    "designed a complete scheme from a different angle: smarter search, a better objective, reusing old "
                    "solutions, changing the rig, or generating many shadows and picking. Three judges scored all five. One "
                    "agent merged them into a plan, a critic attacked the plan by checking its numbers against the files, and "
                    "a last agent revised it. Then 10 more agents tried to prove the plan wrong (next card).",
            "why": "We have many partial results: 16 objectives, CMA-ES vs Adam, the Ivy judge (a score fitted to your marks), "
                   "arm arrangements, and SDS. We needed one plan that agrees with all of them, instead of another one-off "
                   "experiment.",
            "expect": "That the answer would be a better objective or a smarter search method.",
            "result": "Simpler than expected:<ol>"
                      "<li><b>One search method: CMA-ES.</b> At equal time Adam's shadows overlap the target less. In your blind "
                      f"side-by-side test you still picked Adam's shadow in {n_adam} of {ab['decided_pairs']} pairs "
                      f"({100 * ab['adam_raw_share']:.0f}%). Part of that is overlap, but not all of it: for pairs with equal "
                      f"overlap Adam still wins {100 * ab['P_adam_at_dIoU0']:.0f}% (95% range {100 * p0_ci[0]:.0f}-{100 * p0_ci[1]:.0f}%), "
                      f"and {100 * ab['nonsemantic_P0']:.0f}% ({100 * ns_ci[0]:.0f}-{100 * ns_ci[1]:.0f}%) on the 10 objectives that "
                      "do not use a classifier. So this is not settled; a test with new people decides it.</li>"
                      "<li><b>Spend extra time on more compute, and keep the best result.</b> "
                      f"On 30 letters and digits, the best of 1 try has overlap {kc['1']['same_objective']:.3f}, of 4 tries "
                      f"{kc['4']['same_objective']:.3f}, of 12 tries {kc['12']['same_objective']:.3f} (chart: 10 digits in a longer "
                      f"setting, same trend). One solve given about 3 times the effort did as well as the best of 3 tries "
                      f"({g36_long:.3f} vs {g36[2]:.3f} on 36 glyphs), so the gain is the compute, and more tries is just an easy "
                      "way to spend it: they run in parallel and give several poses to choose a safe one from. Longer solves "
                      "vs more tries is still to test.</li>"
                      "<li><b>No learned score may steer the search or choose the result.</b> The classifier, CLIP, the Ivy judge "
                      "and SDS only report. Each one we optimized got gamed (the classifier was then right 93% of the time, "
                      "you 43%). Choose the final shadow by: physically possible first, then overlap.</li>"
                      "<li><b>Fix the safety bugs first</b> (next card).</li>"
                      "<li><b>Animations:</b> keep a few good poses per frame and choose the smoothest path through all frames, "
                      "instead of fixing each frame before the next. Not tested yet; today's frame-after-frame method stays as "
                      "the fallback.</li>"
                      "<li><b>Human tests with new people</b>, each glyph shown once and no answer shown. Your own reads no "
                      "longer count as fresh: the picking page showed you the right answer each time.</li></ol>",
            "next": "Cheapest first. (1) The floor fix (next card): make it the default and re-run the benchmark. (2) With 5 or "
                    "more new people (about 8 for a firm claim), test whether shadows from more tries are easier to read, and "
                    "whether Adam is still preferred at equal overlap. (3) Whole-animation planning. Full plan with every number "
                    "and source: <code>results/fastshadow/research/plan.md</code>.",
            "charts": [{"title": "More tries, keep the best", "fmt": "f3", "min": 0.68, "max": 0.76,
                        "ticks": [0.68, 0.70, 0.72, 0.74, 0.76],
                        "rows": [{"label": f"{k} {'try' if k == 1 else 'tries'}", "v": c, "color": "shape"}
                                 for k, c in zip((1, 2, 4, 8, 16), curve)],
                        "note": "Average overlap of the best of k separate CMA-ES solves. 10 digits, 16 solves each from "
                                "different random starts, in a longer solve setting; each bar averages every way of picking "
                                "k of the 16."}]}


def floor_item():
    v = load("research", "verify_claims.json")
    if not v:
        return None
    tp = v["table_penetration"][1]["numbers"]
    kf = v["keep_ignores_feasible"][1]["numbers"]
    rows = [{"label": "objective study, CMA-ES", "v": frac(tp["objective_study_cma_below_5mm"]), "color": "bad"},
            {"label": "objective study, Adam", "v": frac(tp["objective_study_adam_below_5mm"]), "color": "bad"},
            {"label": "arm-layout study, CMA-ES", "v": frac(tp["arrangement_cma_below_5mm"]), "color": "bad"},
            {"label": "benchmark, final picks", "v": frac(tp["big_budget_best_run_below_5mm"]), "color": "bad"}]
    f = load("research", "floor_fix.json")
    t = load("research", "floor_timing.json")
    res = ("One real problem. The solver's \"stay above the table\" check takes the lowest point of all arm parts, "
           "including the fixed base, and the base always reads 18 mm below the table top. So moving parts can sink about "
           "18 mm into the table for free; only deeper than that does the check push back. It also assumes no part reaches "
           "more than 35 mm below its centre, but the gripper reaches up to 84 mm, so the gripper goes deeper still. "
           f"Result: in {100 * tp['all_below_5mm_share']:.1f}% of all {tp['all_solves_n']:,} stored solves "
           f"({100 * frac(tp['big_budget_best_run_below_5mm']):.1f}% of the benchmark's final picks) the gripper or jaw goes "
           f"more than 5 mm into the table, up to {-tp['worst_penetration_mm']:.0f} mm, and every one is still marked "
           "\"collision free\". The shadow scores are computed correctly (there is no table in the shadow), but some were "
           f"reached by going through the table, CMA-ES about twice as often as Adam ({100 * frac(tp['objective_study_cma_below_5mm']):.1f}% "
           f"vs {100 * frac(tp['objective_study_adam_below_5mm']):.1f}%). On the real rig those poses hit the table. "
           "<br><br>Two smaller findings. Every \"keep the better one\" step compares overlap only and ignores collisions, "
           f"of an arm with itself or with another arm ({kf['benchmark_solves_colliding'].split('=')[1].strip()} of stored "
           f"benchmark solves collide, {kf['best_picks_colliding_total'].split(',')[0]} final picks). And the solver throws "
           "away a given starting pose and searches from random poses, keeping the old pose only as a fallback: for "
           "animations that is wasted time, not wrong numbers, but it also means two rows of the earlier ablation (\"aimed "
           "start\", \"backward sweep\") test something other than their names.")
    verdict, label, nxt = "pending", "still running", "Fix running."
    if f:
        b, a = f["before"], f["after"]
        ci = f["best_of_3_iou"]["change_glyph_ci"]
        res += ("<br><br><b>The fix.</b> With the new option <code>--floor mesh</code>, the check uses the real lowest point "
                f"of the parts that move. We re-solved {f['n_glyphs']} glyphs, 3 tries each. Before the fix "
                f"{100 * b['below_5mm']:.1f}% of these went more than 5 mm into the table (deepest {-b['lowest_mm']:.0f} mm); after, "
                f"{100 * a['below_5mm']:.1f}%, and every moving part stays at least {a['lowest_mm']:.1f} mm above it. Overlap of "
                f"the best of the 3 tries: {f['best_of_3_iou']['before']:.4f} before, {f['best_of_3_iou']['after']:.4f} after "
                f"(change {ci[0]:+.3f}; the true change is likely between {ci[1]:+.3f} and {ci[2]:+.3f}). So no cost we can see "
                "on 30 glyphs, though a small one is not ruled out.")
        if t:
            res += (f" Time per solve on an otherwise idle GPU ({t['n']} solves on 6 glyphs): {t['mujoco']:.2f} s before, "
                    f"{t['mesh']:.2f} s with the fix, about {100 * (t['mesh'] / t['mujoco'] - 1):.0f}% slower.")
        rows += [{"label": "same 30 glyphs x 3, before", "v": b["below_5mm"], "color": "bad", "op": 0.5},
                 {"label": "same 30 glyphs x 3, after", "v": a["below_5mm"], "color": "good"}]
        if a["below_5mm"] < 0.01 and ci[0] > -0.005:
            verdict, label = "yes", "found; fix ready, not yet default"
        else:
            verdict, label = "mixed", "fix has a cost"
        nxt = ("The fix is an option for now, on the CMA-ES path only (Adam's floor and the \"collision free\" flag still use "
               "the old check). In fleet-shadow-art (local branch <code>distort_target</code>, commit 5cbb9d7, needs your "
               "push) the same option exists, and the deploy-time guard that is meant to keep the real arms off the table, "
               "which had the same bug and could not lift any pose, now always uses the true lowest point (10 stored poses "
               "up to 3.9 cm into the table: all at least 1.0 cm above it after). Your decision: make the fix the default "
               "and re-run the benchmark, which changes the paper's baseline numbers slightly.")
    return {"title": "Checking the plan's claims: the arms go through the table", "verdict": verdict, "label": label,
            "what": "The plan said the solver has three bugs. Ten agents tried to prove the plan wrong: eight on the bug "
                    "claims (five on the floor check alone) and two on its other key numbers (more tries, and your A/B "
                    "choices, both on the first card). They recomputed from the code and from every stored solve. Then we "
                    "fixed the bug that mattered and re-solved.",
            "why": "A broken safety check means some \"solved\" shadows cannot be made with the real arms.",
            "expect": "Small effects, if any.",
            "result": res, "next": nxt,
            "charts": [{"title": "Solves with the gripper more than 5 mm into the table", "fmt": "pct", "min": 0, "max": 0.15,
                        "ticks": [0, 0.05, 0.10, 0.15], "rows": rows, "label_w": 190,
                        "note": "Lowest point of the moving arm parts, from the stored joint angles. Top four: stored solves "
                                "(1,772 CMA-ES and 992 Adam in the objective study, 800 in the arm-layout study, 562 benchmark "
                                "final picks). Bottom two: the 90-solve re-solve, before and after the fix."}]}


def sds_item():
    reps = {m: load("sds", m, "report.json") for m in ("sd15", "sdxl")}
    reps = {m: r for m, r in reps.items() if r}
    if not reps:
        return None
    names = {"sd15": "SD 1.5", "sdxl": "SDXL"}
    sub_names = {"letters_upper": "uppercase", "letters_lower": "lowercase", "digits": "digits"}
    charts, rows, hits = [], [], {}
    for m, r in reps.items():
        if "sanity" not in r:
            continue
        rows.append({"label": names[m], "head": True, "v": 0})
        hits[m] = [0, 0.0, 0]
        for sub, x in r["sanity"].items():
            rows.append({"label": f"{sub_names[sub]} ({x['n_images']} images)", "v": x["top1"], "ref": x["chance_top1"], "color": "sds"})
            hits[m][0] += round(x["top1"] * x["n_images"])
            hits[m][1] += x["chance_top1"] * x["n_images"]
            hits[m][2] += x["n_images"]
    if rows:
        charts.append({"title": "Can it tell clean letters apart?", "fmt": "pct", "min": 0, "max": 0.5,
                       "ticks": [0, 0.25, 0.5], "rows": rows, "label_w": 190,
                       "note": "Clean target glyphs, no robots. How often the correct description (e.g. \"the capital letter "
                               "A\") fits the image best, out of all 26 letters or 10 digits. Black tick = blind guessing "
                               "(1 in 26, 1 in 10). SD 1.5: 3 fonts per character; SDXL: 1 font, a lighter check."})
    ev = (reps.get("sd15") or {}).get("eval")
    x = load("research", "extra_numbers.json")
    sd = judge = base = None
    if ev:
        judge = ev["Ivy judge (held-out characters)"]
        sd = ev["sds_rel (vs neutral prompt)"]
        sdc = ev["sds_rel (vs neutral prompt), per-glyph centred"]
        base = load("ivy_reader", "report.json")
        nm = base["n"]["marks"]
        base = base["baselines"]
        rows = [{"label": "SDS score (SD 1.5)", "v": sd["mark_auc"], "color": "sds"},
                {"label": "plain overlap (IoU)", "v": base["iou"]["mark_auc"], "color": "base"},
                {"label": "Ivy judge", "v": judge["mark_auc"], "color": "judge"}]
        charts.append({"title": "Does it agree with your 'good' marks?", "fmt": "f2", "min": 0.3, "max": 0.9,
                       "ticks": [0.3, 0.5, 0.7, 0.9], "ref": 0.5, "ref_label": "coin flip", "rows": rows,
                       "note": f"AUC over the {nm:,} shadows you marked (62 characters): how often a shadow you marked good "
                               "scores higher than one you did not."})
    imgs = []
    ia = os.path.join(FS, "research", "control_letter_A.png")
    if os.path.exists(ia):
        imgs.append({"uri": png_uri(ia), "alt": "SD 1.5 drawing for the prompt capital letter A", "caption": 'Asked for <i>the capital letter "A", black on a white background</i>, '
                                                    'SD 1.5 drew this: a white "I" on grey.'})
    res = "No. "
    if "sd15" in hits:
        h = hits["sd15"]
        res += (f"Stable Diffusion 1.5 cannot tell clean letters apart: it picks the right letter about as often as a blind "
                f"guess ({h[0]} of {h[2]} images; guessing gives about {h[1]:.0f}). ")
    if "sdxl" in hits:
        h = hits["sdxl"]
        res += (f"SDXL, a newer model, got a lighter check and was also at chance ({h[0]} of {h[2]} images; guessing gives "
                f"about {h[1]:.0f}); it was not run on your shadows. ")
    if sd:
        res += (f"Take one shadow you marked good and one you did not: SD 1.5's score puts the good one higher only "
                f"{100 * sd['mark_auc']:.0f}% of the time (AUC {sd['mark_auc']:.2f}), a coin flip. Plain overlap does so "
                f"{100 * base['iou']['mark_auc']:.0f}% of the time and the Ivy judge {100 * judge['mark_auc']:.0f}%. Comparing only "
                f"shadows of the same character, SDS gets {sdc['mark_auc']:.2f}")
        res += f" and overlap {x['iou_within_glyph']['mark_auc']:.2f}. " if x else ". "
    res += ("A sanity test on pictures the model drew itself shows the measurement works: a dog photo fits \"dog\" best, "
            "and a drawn letter fits a letter description better than \"dog\", \"cat\" or an empty one. But the descriptions "
            "\"A\", \"O\" and \"X\" fit its drawn letters equally well, to three decimals. It cannot even draw the letters: "
            "asked for \"A\", it drew an \"I\" (picture). Telling letters apart is the whole job here.")
    return {"title": "SDS: can an image-generation model judge our shadows?", "verdict": "no",
            "what": "SDS (score distillation) borrows an image-generation model as a critic. We show it a shadow with a "
                    "description like <i>a black shadow of the digit \"3\" on a white wall</i>, add noise, and ask it to "
                    "remove the noise. If the description helps it clean the image up, the image fits the description. "
                    "The same signal can be turned into a push on the robot joints.",
            "why": "The objectives we optimize today only measure overlap with the target. People read shadows by meaning, "
                   "and the model has seen millions of letters. Before optimizing with it we check whether it agrees "
                   "with you, because every \"meaning\" score we optimized so far (classifier, CLIP) got gamed.",
            "expect": "To be usable it has to (a) recognise clean letters and (b) rank your good shadows above your bad ones "
                      "at least as well as plain overlap does. Our guess: maybe, though these models are known to be bad at text.",
            "result": res,
            "next": "Drop SDS with these models. It is only worth another look with a model that can write text (SD 3.5, "
                    "FLUX, DeepFloyd IF). To download most of them (SD 3.5, FLUX.1-dev, DeepFloyd IF) you must first accept "
                    "their licence on Hugging Face; FLUX.1-schnell is open but large for this laptop. The control's raw "
                    "numbers are in <code>results/fastshadow/research/sds_control_output.txt</code>.",
            "charts": charts, "images": imgs}


def sds_lora_item():
    r = load("sds", "lora_v2", "report.json")
    if not r:
        return None
    ev, g = r["eval"], r["gates"]
    q, io, jd = ev["SDS quality"], ev["IoU"], ev["Ivy judge (same folds)"]
    rd = ev["SDS reading margin"]["reads_auc"]
    cl = r["clean_top1"]
    c0 = r["folds"][0]["clean"]
    n_l = c0["letters_upper"]["n"] + c0["letters_lower"]["n"]
    ch_l = (c0["letters_upper"]["chance"] + c0["letters_lower"]["chance"]) / n_l
    ch_d = c0["digits"]["chance"] / c0["digits"]["n"]
    passed = sum(g.values())
    rows_auc = [{"label": "SDS fine-tuned", "v": q["marks_within"][0], "color": "sds"},
                {"label": "plain overlap (IoU)", "v": io["marks_within"][0], "color": "base"},
                {"label": "Ivy judge, same split", "v": jd["marks_within"][0], "color": "judge"}]
    rows_read = [{"label": "SDS fine-tuned", "v": rd, "color": "sds"},
                 {"label": "plain overlap (IoU)", "v": io["reads_auc"], "color": "base"},
                 {"label": "Ivy judge, same split", "v": jd["reads_auc"], "color": "judge"}]
    rows_clean = [{"label": "letters (156 images)", "v": cl["letters"], "ref": ch_l, "color": "sds"},
                  {"label": "digits (30 images)", "v": cl["digits"], "ref": ch_d, "color": "sds"}]
    return {"title": "SDS, done properly: fine-tuned on your shadows", "verdict": "yes" if r["go"] else "no",
            "label": "go" if r["go"] else f"stop ({passed} of {len(g)} checks passed)",
            "what": "The plan from the first SDS card, carried out. We taught Stable Diffusion two things with a small "
                    "add-on (LoRA): which character is which, from clean letters in 30 fonts, each character with its own "
                    "new word; and what a clear vs an unreadable robot-arm shadow looks like, from every shadow you marked. "
                    "Then we asked it, on characters whose shadows it never saw: is this shadow clear or unreadable, and "
                    "which character is it?",
            "why": "SDS can only steer the arms well if it can first judge shadows the way you do. A test on characters "
                   "it has not seen is the honest one: new targets will be new to it too.",
            "expect": "Some improvement over the untrained model (which was at chance); probably not enough to beat plain "
                      "overlap. Before running it, two reviewer agents tightened the test: hold out whole characters, as "
                      "for the Ivy judge; refit the judge on the same split; and fix five pass/fail checks in advance.",
            "result": (("It learned a little, not enough. " if cl["letters"] > 1.5 * ch_l and not r["go"] else "")
                       + f"Clean letters: right {100 * cl['letters']:.0f}% of the time (guessing: "
                      f"{100 * ch_l:.0f}%), digits {100 * cl['digits']:.0f}% (guessing {100 * ch_d:.0f}%); the untrained model "
                      f"was at chance. Judging your shadows, on characters it never trained on, it is "
                      + ("no better than a coin flip" if q["marks_within"][0] < 0.55 else
                         "better than chance but below plain overlap" if q["marks_within"][0] < io["marks_within"][0] else
                         "at least as good as plain overlap")
                      + f": comparing shadows of the same character, it agrees with your good marks at AUC "
                      f"{q['marks_within'][0]:.2f}, against {io['marks_within'][0]:.2f} for plain overlap and "
                      f"{jd['marks_within'][0]:.2f} for the Ivy judge on exactly the same split. For which shadows you "
                      f"read correctly: {rd:.2f}, against {io['reads_auc']:.2f} and {jd['reads_auc']:.2f}. Its own score is "
                      f"also unsteady: two halves of its noise draws agree at {r['reliability']:.2f} (needed 0.8). "
                      f"{passed} of the {len(g)} checks fixed in advance passed"
                      + (", so step 3, letting SDS steer the arms, was not run." if not r["go"] else ".")),
            "next": ("Leave SDS out. What it would have added (does this read as the right character?) the glyph classifier "
                     "inside the Ivy judge already does, cheaply and much better. Still open, if wanted: whether it helps on "
                     "characters it has seen (the 62 glyphs are always known in advance), where it would have to beat a "
                     "simple 'closest known-good shadow' baseline (about 0.70) to count.") if not r["go"] else
                    ("Step 3: let SDS steer the arms through the GPU renderer, mixed with overlap, on held-out characters, "
                     "with an overlap-only control and a blind check by you for gaming."),
            "charts": [{"title": "Agrees with your good marks (same character)", "fmt": "f2", "min": 0.3, "max": 0.9,
                        "ticks": [0.3, 0.5, 0.7, 0.9], "ref": 0.5, "ref_label": "coin flip", "rows": rows_auc,
                        "note": "Within-character AUC on characters held out from training (5 folds by character)."},
                       {"title": "Predicts which shadows you read", "fmt": "f2", "min": 0.3, "max": 0.9,
                        "ticks": [0.3, 0.5, 0.7, 0.9], "ref": 0.5, "ref_label": "coin flip", "rows": rows_read,
                        "note": "AUC over your guess-the-glyph answers, same held-out characters."},
                       {"title": "Clean letters read correctly", "fmt": "pct", "min": 0, "max": 1, "ticks": [0, 0.5, 1],
                        "rows": rows_clean, "note": "DejaVu target fonts, never used in training. Black tick = guessing."}]}


def robustness_item():
    r = load("robustness", "objective_study_dt2.json")
    if not r:
        return None
    sm = r["summary"]
    s1, s2 = sm["sigma_1deg"], sm["sigma_2deg"]
    n = sm["n_glyphs_multi_seed"]
    ps = r["per_solve"]
    d1 = np.array([o["nominal"] - o["mean_1"] for o in ps])
    d2 = np.array([o["nominal"] - o["mean_2"] for o in ps])
    q1, q2 = np.percentile(d1, [10, 90]), np.percentile(d2, [10, 90])
    n_ids = len({o["id"] for o in ps})
    return {"title": "Do small robot errors ruin the shadow?", "verdict": "yes", "label": "not a reason to change the pick",
            "what": f"Real arms never hit the exact angle: two calibrations of our arms differ by up to 1.5 degrees per joint. "
                    f"We took {sm['n_solves']} stored solves (dt2 objective, {n_ids} letters and digits, {n} of them with 3 tries "
                    "each). We nudged all 18 joints at random, 32 times, with a typical error of 1 degree and then 2 degrees, "
                    "and measured how much the shadow's overlap with the target drops.",
            "why": "If some solutions are fragile, the solver should prefer the sturdy ones, since the real robot will always "
                   "be a little off.",
            "expect": "Some shadows, the ones with thin strokes made by a single arm edge, should fall apart much faster than others.",
            "result": f"Small on average, with a moderate spread. At 1 degree the overlap drops by {s1['mean_iou_drop']:.3f} on "
                      f"average, and 1 nudge in 10 loses more than {s1['p10_iou_drop']:.3f}; most solves lose between "
                      f"{q1[0]:.3f} and {q1[1]:.3f}, the most fragile {d1.max():.3f}. At 2 degrees: {s2['mean_iou_drop']:.3f} on "
                      f"average (most between {q2[0]:.3f} and {q2[1]:.3f}, the most fragile {d2.max():.3f}). For choosing among a "
                      f"glyph's tries this hardly matters: the tries differ much more in overlap (typically "
                      f"{s1['within_glyph_nominal_spread_median']:.2f}) than in how much they lose. Picking by the 1-in-10 bad "
                      f"case instead of exact overlap changes the choice on only {s1['glyphs_where_robust_pick_differs']} of {n} "
                      f"glyphs at 1 degree, gaining {s1['robust_pick_p10_gain_mean']:+.4f} there while losing "
                      f"{s1['robust_pick_nominal_cost_mean']:.4f} of exact overlap; at 2 degrees it changes "
                      f"{s2['glyphs_where_robust_pick_differs']} of {n}, gaining {s2['robust_pick_p10_gain_mean']:+.4f}.",
            "next": "In simulation, random joint error is no reason to pick by sturdiness; keep sturdiness as a reported column, "
                    "not a selection rule. Calibration matters: 2 degrees of error costs about "
                    f"{s2['mean_iou_drop'] / s1['mean_iou_drop']:.1f} times as much as 1 degree. Still untested: errors in base "
                    "and light position, and whether the simulation matches the real rig (photos of real shadows).",
            "charts": [{"title": "Overlap lost to joint error", "fmt": "f3", "min": 0, "max": 0.08,
                        "ticks": [0, 0.02, 0.04, 0.06, 0.08], "label_w": 190,
                        "rows": [{"label": "1 degree, average", "v": s1["mean_iou_drop"], "color": "shape"},
                                 {"label": "1 degree, 1 nudge in 10 loses more", "v": s1["p10_iou_drop"], "color": "shape", "op": 0.5},
                                 {"label": "2 degrees, average", "v": s2["mean_iou_drop"], "color": "bad"},
                                 {"label": "2 degrees, 1 nudge in 10 loses more", "v": s2["p10_iou_drop"], "color": "bad", "op": 0.5}],
                        "note": f"Overlap lost vs the exact pose, {sm['n_solves']} solves, 32 random nudges each, drawn with our "
                                "fast GPU renderer."}]}


def seeds_item():
    s = load("research", "seeds_vs_objectives.json")
    if not s:
        return None
    P = s["pick"]
    a, b = P["12 tries of dt2"], P["13 objectives, 1 try each"]
    d = s["iou_diff_seeds_minus_menu"]
    kc = s["k_curve"]
    rows = []
    for k in ("1", "2", "4", "8", "12"):
        if k in kc:
            rows += [{"label": f"{k} {'try' if k == '1' else 'tries'} of dt2", "v": kc[k]["same_objective"], "color": "shape"},
                     {"label": f"{k} {'objective' if k == '1' else 'objectives'}", "v": kc[k]["objective_menu"], "color": "sds", "op": 0.6}]
    sol = [json.loads(line) for line in open(os.path.join(FS, "objective_study", "solves.jsonl"), encoding="utf-8")]
    dt2_s = float(np.mean([r["seconds"] for r in sol if r["objective"] == "dt2"]))
    x = load("research", "extra_numbers.json")
    same = abs(d[0]) < 0.005 or (d[1] < 0 < d[2])
    win = same or d[0] > 0
    res = (f"Retrying wins. On {s['n_glyphs']} letters and digits, the best of 12 dt2 tries has overlap {a['iou'][0]:.3f}, the best "
           f"of the 13 objectives {b['iou'][0]:.3f} (difference {d[0]:+.3f}, 95% range {d[1]:+.3f} to {d[2]:+.3f}; retrying is "
           f"better on {s['wins_seeds']} glyphs, the menu on {s['wins_menu']}). Shape of the picked shadow (tc_iou): "
           f"{a['tc_iou'][0]:.3f} vs {b['tc_iou'][0]:.3f}, higher is better. Share of the letter's holes (inside of A, O) covered "
           f"by shadow: {a['holes_filled'][0]:.3f} vs {b['holes_filled'][0]:.3f}, lower is better; about equal, and only 12 of "
           "the 30 glyphs have holes.")
    if x:
        res += (f" The menu is also slower: about {x['menu_seconds_per_glyph']:.0f} s per glyph for the 13 objectives against "
                f"about {12 * dt2_s:.0f} s for 12 dt2 tries ({dt2_s:.1f} s each), so at equal time the gap is larger.")
    return {"title": "Try again, or try a different objective?", "verdict": "yes" if win else "no",
            "label": "retry wins" if win else "menu wins",
            "what": "Compare two ways of spending about a dozen solves per glyph: run our default objective dt2 12 times from "
                    "different random starts, or run each of the study's 13 objectives once (all 16 except the 3 Ivy ones; "
                    "the menu even gets one extra solve). Either way, keep the solve with the best overlap.",
            "why": "If retrying one objective is as good as a menu of objectives, we can drop the collection of objectives "
                   "and keep one simple, safe one.",
            "expect": "Similar, maybe a slight edge for the menu, because different objectives land in more different places.",
            "result": res,
            "next": "Keep one objective (dt2) and spend extra time on more tries." if win else
                    "The menu wins: keep a few objectives in the pool.",
            "charts": [{"title": "Best overlap after k solves", "fmt": "f3", "min": 0.62, "max": 0.78, "label_w": 150,
                        "ticks": [0.62, 0.66, 0.70, 0.74, 0.78], "rows": rows,
                        "note": "Blue: best of k tries of dt2. Pink: best of k different objectives, 1 try each. Each bar "
                                "averages over random choices of which k."}]}


def soft_renderer_item():
    r = load("soft_shadow_check", "report.json")
    if not r:
        return None
    full, dec, nv = r["soft_shadow (full meshes)"], r["soft_shadow (400 faces/link)"], r["nvdiffrast (fastshadow)"]
    ratio = (dec["ms_soft_fwd_bwd"] / nv["ms_soft_fwd_bwd"], full["ms_soft_fwd_bwd"] / nv["ms_soft_fwd_bwd"])
    return {
        "title": "Is the soft shadow renderer in fleet-shadow-art right?",
        "verdict": "no", "label": "had bugs, replaced",
        "what": "fleet-shadow-art had its own shadow renderer for gradient solving (<code>diffrender/soft_shadow.py</code>). It "
                "projects every arm triangle onto the wall and gives each pixel a soft 0-to-1 \"how covered am I\" value. From "
                "that, an optimizer like Adam can work out which way to turn each joint, instead of trying random poses.",
        "why": "Souju reported that gradient solving with it gave poor results. Before blaming gradients, check the renderer: "
               "is the shadow the right shape, is the soft version a usable signal, and is it fast enough to optimize with.",
        "expect": "Shape correct (it projects the real meshes), but likely slow, because it tests every triangle against every pixel.",
        "result": f"The shape is right (overlap with MuJoCo {full['hard_iou_solved']:.3f} on 6 solved digit poses). Two problems. "
                  "(1) A bug: a few zero-area (flat) triangles make the whole empty wall read 0.5 instead of 0, so the soft "
                  "overlap the optimizer sees is far from the real one. The shape check missed it because it only asks \"above "
                  f"0.5?\". In our small gradient test the direction still mostly pointed the right way (agreement with the true "
                  f"change {full['grad_cos_vs_real_fd']:.2f} on full meshes and {dec['grad_cos_vs_real_fd']:.2f} simplified, 2 poses "
                  f"each, against {nv['grad_cos_vs_real_fd']:.2f} for our renderer on 8), so this bug alone may not explain the "
                  f"poor results. (2) It is far too slow: one gradient step takes {dec['ms_soft_fwd_bwd'] / 1000:.0f}-"
                  f"{full['ms_soft_fwd_bwd'] / 1000:.0f} s, against {nv['ms_soft_fwd_bwd']:.0f} ms for our GPU renderer "
                  f"(nvdiffrast), about {ratio[0]:,.0f} to {ratio[1]:,.0f} times slower, so in practice almost no steps get taken. "
                  "We replaced it in fleet-shadow-art with our renderer (<code>--renderer nvdiffrast</code>): a step along its "
                  f"gradient improved MuJoCo's overlap in {round(8 * nv['step_improves_real_iou'])} of 8 tests in this check, and "
                  "8 of 8 in fleet-shadow-art's own test.",
        "next": "Push the fleet-shadow-art branch <code>distort_target</code> and open a PR (needs your GitHub access). Even "
                "with this renderer, Adam still trails CMA-ES at equal time (first card), so CMA-ES stays the default.",
        "charts": [
            {"title": "Soft value on empty wall (should be 0)", "fmt": "f2", "min": 0, "max": 1, "ticks": [0, 0.5, 1],
             "rows": [{"label": "soft_shadow", "v": full["soft_outside"], "color": "bad"},
                      {"label": "nvdiffrast (ours)", "v": nv["soft_outside"], "color": "good"}],
             "note": "Average soft value outside the true shadow. At 0.5 the empty wall looks half covered."},
            {"title": "Time per gradient step", "fmt": "s", "min": 0, "max": 30, "ticks": [0, 10, 20, 30],
             "rows": [{"label": "soft_shadow, full meshes", "v": full["ms_soft_fwd_bwd"] / 1000, "color": "bad"},
                      {"label": "soft_shadow, simplified", "v": dec["ms_soft_fwd_bwd"] / 1000, "color": "bad"},
                      {"label": "nvdiffrast (ours)", "v": nv["ms_soft_fwd_bwd"] / 1000, "color": "good"}],
             "note": "One shadow plus its gradient, for one pose of 3 arms, 128 x 128 pixels. Adam needs thousands of these "
                     "per glyph."},
        ],
    }


def main():
    items = [x for x in (plan_item(), floor_item(), sds_item(), sds_lora_item(), robustness_item(), seeds_item(), soft_renderer_item()) if x]
    doc = {"intro": "Each card: what it is, why we did it, what we expected, what happened. Numbers come straight from the "
                    "result files. Words used below:<ul>" + "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in GLOSSARY) + "</ul>",
           "items": items}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT, len(doc["items"]), "items")


if __name__ == "__main__":
    main()
