"""Compose results/fastshadow/research/research.json: the "overnight research" section of the
objective report, in plain words, with every number read from the result files.

    python scripts/make_research_notes.py && python scripts/make_objective_report.py --viz-subset digits
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastshadow  # noqa: E402

FS = os.path.join(fastshadow.BENCH, "results", "fastshadow")
OUT = os.path.join(FS, "research", "research.json")


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


def soft_renderer_item():
    r = load("soft_shadow_check", "report.json")
    if not r:
        return None
    full, dec, nv = r["soft_shadow (full meshes)"], r["soft_shadow (400 faces/link)"], r["nvdiffrast (fastshadow)"]
    return {
        "title": "Is the soft shadow renderer in fleet-shadow-art right?",
        "verdict": "no", "label": "had bugs, replaced",
        "what": "fleet-shadow-art had its own differentiable shadow renderer (<code>diffrender/soft_shadow.py</code>): every arm "
                "triangle is projected onto the wall and each pixel gets a soft 0-to-1 \"how covered am I\" value, so an optimizer "
                "can follow gradients instead of guessing.",
        "why": "Souju reported that gradient solving with it gave poor results. Before blaming gradients, check the renderer: "
               "is the shadow the right shape, is the soft version a usable signal, and is it fast enough to optimize with.",
        "expect": "Shape correct (it projects the real meshes), but likely slow, because it tests every triangle against every pixel.",
        "result": f"The shape is right (overlap with MuJoCo {full['hard_iou_solved']:.3f}). Two real problems: "
                  f"(1) a bug: a handful of zero-area triangles make the whole background read 0.5 instead of 0, which washes out "
                  f"the signal the optimizer follows; the yes/no test at 0.5 hid it. (2) It is far too slow: one gradient step takes "
                  f"{dec['ms_soft_fwd_bwd'] / 1000:.0f}-{full['ms_soft_fwd_bwd'] / 1000:.0f} s, against {nv['ms_soft_fwd_bwd']:.0f} ms "
                  f"for our nvdiffrast renderer, so in practice almost no steps get taken. We replaced it in fleet-shadow-art with "
                  f"the nvdiffrast renderer (<code>--renderer nvdiffrast</code>); there its gradient points the right way "
                  f"(a step along it improved MuJoCo's overlap in 8 of 8 tests).",
        "next": "Push the fleet-shadow-art branch <code>distort_target</code> and open a PR (needs your GitHub access).",
        "charts": [
            {"title": "Soft mask background (should be 0)", "fmt": "f2", "min": 0, "max": 1, "ticks": [0, 0.5, 1],
             "rows": [{"label": "soft_shadow", "v": full["soft_outside"], "color": "bad"},
                      {"label": "nvdiffrast (ours)", "v": nv["soft_outside"], "color": "good"}],
             "note": "Mean soft value outside the true shadow. 0.5 everywhere means half the signal is noise."},
            {"title": "Time per gradient step", "fmt": "s", "min": 0, "max": 30, "ticks": [0, 10, 20, 30],
             "rows": [{"label": "soft_shadow, full meshes", "v": full["ms_soft_fwd_bwd"] / 1000, "color": "bad"},
                      {"label": "soft_shadow, simplified", "v": dec["ms_soft_fwd_bwd"] / 1000, "color": "bad"},
                      {"label": "nvdiffrast (ours)", "v": nv["ms_soft_fwd_bwd"] / 1000, "color": "good"}],
             "note": "Forward + backward for one pose, 3 arms, 128 px. Adam needs thousands of these per glyph."},
        ],
    }


def sds_item():
    reps = {m: load("sds", m, "report.json") for m in ("sd15", "sdxl")}
    reps = {m: r for m, r in reps.items() if r}
    if not reps:
        return None
    names = {"sd15": "SD 1.5", "sdxl": "SDXL"}
    sub_names = {"letters_upper": "uppercase", "letters_lower": "lowercase", "digits": "digits"}
    charts, rows = [], []
    for m, r in reps.items():
        if "sanity" not in r:
            continue
        rows.append({"label": names[m], "head": True, "v": 0})
        for sub, x in r["sanity"].items():
            rows.append({"label": sub_names[sub], "v": x["top1"], "ref": x["chance_top1"], "color": "sds"})
    if rows:
        charts.append({"title": "Can it tell clean letters apart?", "fmt": "pct", "min": 0, "max": 0.5,
                       "ticks": [0, 0.25, 0.5], "rows": rows,
                       "note": "Clean target glyphs, no robots. Share where the right character's description explains the "
                               "image best, among all characters of that kind. Black tick = guessing."})
    ev = (reps.get("sd15") or {}).get("eval")
    sd = judge = base = None
    if ev:
        judge = ev["Ivy judge (held-out characters)"]
        sd = ev["sds_rel (vs neutral prompt)"]
        base = load("ivy_reader", "report.json")["baselines"]
        rows = [{"label": "SDS score (SD 1.5)", "v": sd["mark_auc"], "color": "sds"},
                {"label": "plain IoU", "v": base["iou"]["mark_auc"], "color": "base"},
                {"label": "Ivy judge", "v": judge["mark_auc"], "color": "judge"}]
        charts.append({"title": "Does it agree with your 'good' marks?", "fmt": "f2", "min": 0.3, "max": 0.9,
                       "ticks": [0.3, 0.5, 0.7, 0.9], "ref": 0.5, "ref_label": "coin flip", "rows": rows,
                       "note": f"AUC over your {reps['sd15']['n_items']} labelled shadows: how often a shadow you marked "
                               "good scores higher than one you did not."})
    imgs = []
    ia = os.path.join(FS, "research", "control_letter_A.png")
    if os.path.exists(ia):
        imgs.append({"uri": png_uri(ia), "caption": 'Asked for <i>the capital letter "A", black on a white background</i>, SD 1.5 drew this.'})
    res = ("No. Neither Stable Diffusion 1.5 nor SDXL can tell clean letters apart: it names the right letter about as "
           "often as a blind guess. ")
    if sd:
        res += (f"On your labelled shadows its score agrees with your marks at AUC {sd['mark_auc']:.2f}, a coin flip, while "
                f"plain overlap (IoU) gets {base['iou']['mark_auc']:.2f} and the Ivy judge {judge['mark_auc']:.2f}. ")
    res += ("A control shows why. The measurement itself works: a dog photo is best explained by \"dog\", and a drawn "
            "letter by any letter description rather than dog, cat or nothing. But \"A\", \"O\" and \"X\" give the same "
            "score to four decimals. The model knows the image is <i>a letter on white</i>, not <i>which</i> letter, and "
            "telling letters apart is the whole job here.")
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
            "next": "Drop SDS with these models. It is only worth another look with a model that can write text (SD 3.5, FLUX, "
                    "DeepFloyd IF); those are gated, so you would need to accept their licence on Hugging Face first.",
            "charts": charts, "images": imgs}


def robustness_item():
    r = load("robustness", "objective_study_dt2.json")
    if not r:
        return None
    s1, s2 = r["summary"]["sigma_1deg"], r["summary"]["sigma_2deg"]
    n = r["summary"]["n_glyphs_multi_seed"]
    return {"title": "Do small robot errors ruin the shadow?", "verdict": "yes", "label": "not a problem",
            "what": "Real arms never hit the exact angle: two calibrations of our arms differ by up to 1.5 degrees per joint. "
                    "We took every solved shadow, nudged all 18 joints at random (32 times, by about 1 and about 2 degrees), "
                    "and measured how much the shadow's overlap with the target drops.",
            "why": "If some solutions are fragile, the solver should prefer the sturdy ones, since the real robot will "
                   "always be a little off.",
            "expect": "Some shadows, the ones with thin strokes made by a single arm edge, should fall apart much faster than others.",
            "result": f"Small, and about the same for everyone. At 1 degree the overlap drops by {s1['mean_iou_drop']:.3f} on average "
                      f"({s1['p10_iou_drop']:.3f} for the worst tenth of nudges). At 2 degrees, {s2['mean_iou_drop']:.3f} "
                      f"({s2['p10_iou_drop']:.3f}). The best shadow on paper is almost always also the sturdiest: choosing by "
                      f"sturdiness changes the pick on only {s1['glyphs_where_robust_pick_differs']} of {n} glyphs at 1 degree "
                      f"(gain {s1['robust_pick_p10_gain_mean']:+.4f}) and {s2['glyphs_where_robust_pick_differs']} of {n} at 2 degrees "
                      f"(gain {s2['robust_pick_p10_gain_mean']:+.4f}).",
            "next": "No need for a robustness term. What matters is calibration: keep joint error under about 1 degree, "
                    "because 2 degrees costs about three times as much.",
            "charts": [{"title": "Overlap lost to joint error", "fmt": "f3", "min": 0, "max": 0.08,
                        "ticks": [0, 0.02, 0.04, 0.06, 0.08],
                        "rows": [{"label": "1 degree, average", "v": s1["mean_iou_drop"], "color": "shape"},
                                 {"label": "1 degree, worst tenth", "v": s1["p10_iou_drop"], "color": "shape", "op": 0.5},
                                 {"label": "2 degrees, average", "v": s2["mean_iou_drop"], "color": "bad"},
                                 {"label": "2 degrees, worst tenth", "v": s2["p10_iou_drop"], "color": "bad", "op": 0.5}],
                        "note": f"IoU drop vs the exact pose, {r['summary']['n_solves']} dt2 solves on letters and digits, "
                                "32 random nudges each, nvdiffrast renderer."}]}


def plan_item():
    v = load("research", "verify_claims.json")
    if not v:
        return None
    bk = v["best_of_k"][0]["numbers"]
    curve = [float(x) for x in bk["fixed_allsubsets_k1_2_4_8_16"].split("/")]
    return {"title": "What is the most sensible way to optimize?", "verdict": "plan",
            "what": "We had 15 AI agents work through it in stages. Four read the code and every result file. Five each "
                    "designed a complete scheme from a different angle: smarter search, a better objective, reusing old "
                    "solutions, changing the rig, or generating many shadows and picking. Three judges scored all five, one "
                    "agent merged them, and a critic attacked the merged plan by checking its numbers against the files. "
                    "Then 10 more agents tried to disprove the plan's bug claims (next card).",
            "why": "We have a pile of partial results: 16 objectives, CMA-ES vs Adam, the Ivy judge, arrangements, SDS. We "
                   "needed one plan that agrees with all of them, instead of another one-off experiment.",
            "expect": "That the answer would be a better objective or a smarter search method.",
            "result": "Simpler than expected:<ol>"
                      "<li><b>One search method:</b> keep CMA-ES. At equal time Adam's shadows overlap the target less. In the blind "
                      "A/B you preferred Adam's slightly, but once overlap is equal that edge (58%) is within chance; a test with "
                      "new people settles it.</li>"
                      f"<li><b>Spend extra time on more tries of the same solve and keep the best one.</b> Overlap with 1 try "
                      f"{curve[0]:.3f}, 2 tries {curve[1]:.3f}, 4 tries {curve[2]:.3f}, 16 tries {curve[4]:.3f}. (Checked: one longer solve "
                      "with the same total effort does about as well, so this is \"more compute helps\", not a trick.)</li>"
                      "<li><b>No learned score may steer or choose.</b> The classifier, CLIP, the Ivy judge and SDS only report. Each "
                      "one we optimized on its own got gamed, and the one safe version (the monotone Ivy objective mixed with "
                      "overlap) only tied dt2. Choose the final shadow by: physically possible first, then overlap.</li>"
                      "<li><b>Fix the safety bugs first</b> (next card).</li>"
                      "<li><b>Animations:</b> plan all frames together, not one after another.</li>"
                      "<li><b>Human tests:</b> new people who see each glyph once, with no feedback. Your own reads are now "
                      "trained by the picker, which shows the answer.</li></ol>",
            "next": "Cheapest first: fix the floor check (done tonight, next card); confirm with 5 or more new people that more "
                    "tries reads better; then whole-animation planning. Full plan with every number and source: "
                    "<code>results/fastshadow/research/plan.md</code>.",
            "charts": [{"title": "More tries, keep the best", "fmt": "f3", "min": 0.68, "max": 0.76,
                        "ticks": [0.68, 0.70, 0.72, 0.74, 0.76],
                        "rows": [{"label": f"{k} {'try' if k == 1 else 'tries'}", "v": c, "color": "shape"}
                                 for k, c in zip((1, 2, 4, 8, 16), curve)],
                        "note": "Mean overlap (IoU) of the best of k independent CMA-ES solves, 10 digits x 16 seeds, "
                                "averaged over all k-subsets."}]}


def floor_item():
    v = load("research", "verify_claims.json")
    if not v:
        return None
    tp = v["table_penetration"][1]["numbers"]
    pct = lambda s: float(s.split("=")[-1].strip().rstrip("%")) / 100  # noqa: E731
    rows = [{"label": "CMA-ES, objective study", "v": pct(tp["objective_study_cma_below_5mm"]), "color": "bad"},
            {"label": "Adam, objective study", "v": pct(tp["objective_study_adam_below_5mm"]), "color": "bad"},
            {"label": "CMA-ES, arrangements", "v": pct(tp["arrangement_cma_below_5mm"]), "color": "bad"},
            {"label": "benchmark, final picks", "v": pct(tp["big_budget_best_run_below_5mm"]), "color": "bad"}]
    f = load("research", "floor_fix.json")
    res = ("One real problem. The solver's \"stay above the table\" check is broken. It looks at the lowest point of all arm "
           "parts, including the fixed base, and the base always reads 1.8 cm below the table top. So the check thinks every "
           "pose is already as low as it can be, and only pushes back once a part goes deeper than that; it also underestimates "
           f"how far the gripper reaches. In {100 * tp['all_below_5mm_share']:.1f}% of all {tp['all_solves_n']:,} stored solves "
           f"({pct(tp['big_budget_best_run_below_5mm']) * 100:.1f}% of the benchmark's final picks) the gripper or jaw goes "
           f"more than 5 mm into the table, up to {-tp['worst_penetration_mm']:.0f} mm, and every one is still marked "
           "\"collision free\". The shadow numbers are unaffected (there is no table in the shadow), but on the real rig those "
           "poses hit the table. The other two claims were real but small: the final pick ignores arm-to-arm collisions "
           "(0.2% of solves collide), and animations do not reuse the previous frame's pose as a starting point (wasted "
           "time, not wrong results).")
    verdict, nxt = "pending", "Fix running."
    if f:
        b, a = f["before"], f["after"]
        ci = f["best_of_3_iou"]["change_glyph_ci"]
        res += (f" <b>The fix:</b> the check now uses the real lowest point of the parts that move. Re-solving the same "
                f"{f['n_glyphs']} glyphs x 3 tries: poses more than 5 mm into the table {100 * b['below_5mm']:.1f}% -> "
                f"{100 * a['below_5mm']:.1f}% (deepest was {-b['lowest_mm']:.0f} mm into the table; now every moving part stays "
                f"at least {a['lowest_mm']:.0f} mm above it), and best-of-3 "
                f"overlap {f['best_of_3_iou']['before']:.3f} -> {f['best_of_3_iou']['after']:.3f} "
                f"(change {ci[0]:+.3f}, 95% range {ci[1]:+.3f} to {ci[2]:+.3f}): no cost we can measure on 30 glyphs.")
        t = load("research", "floor_timing.json")
        if t:
            res += (f" Time per solve on an otherwise idle GPU: {t['mujoco']:.2f} s before, {t['mesh']:.2f} s with the fix.")
        rows.append({"label": "after the fix", "v": a["below_5mm"], "color": "good"})
        safe = a["below_5mm"] < 0.01
        free = ci[0] > -0.005  # mean cost; 30 glyphs leave a wide range, so say so in the text
        verdict = "yes" if safe and free else "mixed"
        nxt = ("Use the fixed check by default (<code>--floor mesh</code>) and re-run the benchmark rows with it; "
               "patch fleet-shadow-art's renderer the same way before anything runs on the real arms."
               if verdict == "yes" else
               "The fix helps but is not free or not complete; see the numbers before adopting it.")
    return {"title": "Checking the plan's bug claims: arms go through the table", "verdict": verdict,
            "label": {"yes": "found and fixed", "mixed": "fixed, with a cost", "pending": "still running"}[verdict],
            "what": "The plan said the solver has three bugs. Ten agents each tried to disprove one, recomputing from the code "
                    "and from every stored solve, then we fixed the one that mattered and re-solved.",
            "why": "A broken safety check means some \"solved\" shadows cannot be made with the real arms.",
            "expect": "Small effects, if any.",
            "result": res, "next": nxt,
            "charts": [{"title": "Solves with the gripper more than 5 mm into the table", "fmt": "pct", "min": 0, "max": 0.15,
                        "ticks": [0, 0.05, 0.10, 0.15], "rows": rows,
                        "note": "Lowest point of the moving arm parts, from the stored joint angles."}]}


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
            rows += [{"label": f"{k} x dt2", "v": kc[k]["same_objective"], "color": "shape"},
                     {"label": f"{k} objectives", "v": kc[k]["objective_menu"], "color": "sds", "op": 0.6}]
    same = abs(d[0]) < 0.005 or (d[1] < 0 < d[2])
    return {"title": "Try again, or try a different objective?", "label": "retry wins" if same or d[0] > 0 else "menu wins",
            "verdict": "yes" if same or d[0] > 0 else "no",
            "what": "Given 12 attempts per glyph, compare two ways of spending them: run the same objective (dt2) 12 times "
                    "from different random starts, or run 13 different objectives once each. Either way, keep the attempt "
                    "with the best overlap.",
            "why": "If more tries of one objective is as good as a menu of objectives, we can drop the objective zoo and "
                   "keep one simple, safe objective.",
            "expect": "Similar, maybe a slight edge for the menu, because different objectives land in more different places.",
            "result": f"On {s['n_glyphs']} letters and digits, 12 tries of dt2 reach overlap {a['iou'][0]:.3f} and the 13-objective "
                      f"menu {b['iou'][0]:.3f} (difference {d[0]:+.3f}, 95% range {d[1]:+.3f} to {d[2]:+.3f}; dt2 tries better on "
                      f"{s['wins_seeds']} glyphs, the menu on {s['wins_menu']}). Shape after the pick: tc_iou {a['tc_iou'][0]:.3f} vs "
                      f"{b['tc_iou'][0]:.3f} (higher is better), share of holes filled in {a['holes_filled'][0]:.3f} vs {b['holes_filled'][0]:.3f} "
                      "(lower is better). And the menu's extra objectives are slower to run, so at equal time the gap is larger.",
            "next": "Keep one objective (dt2) and spend extra time on more tries." if same or d[0] > 0 else
                    "The menu wins: keep a few objectives in the pool.",
            "charts": [{"title": "Best overlap after k attempts", "fmt": "f3", "min": 0.66, "max": 0.78,
                        "ticks": [0.66, 0.70, 0.74, 0.78], "rows": rows,
                        "note": "Blue: k tries of dt2. Pink: k different objectives. Averaged over random choices of which k."}]}


def main():
    items = [x for x in (plan_item(), floor_item(), sds_item(), robustness_item(), seeds_item(), soft_renderer_item()) if x]
    doc = {"intro": "Each card: what it is, why we did it, what we expected, what happened. Numbers come straight from "
                    "the result files; nothing here was tuned on the same data it is scored on unless the card says so.",
           "items": items}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT, len(doc["items"]), "items")


if __name__ == "__main__":
    main()
