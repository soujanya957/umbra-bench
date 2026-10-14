"""Tables for results/fastshadow/compare_<tag>/per_target.jsonl (scripts/compare_grad.py).

Two questions, kept apart because they have different budgets:
  per solve   one CMA-ES solve vs one gradient solve: mean IoU over a target's
              solves, and seconds per solve on this machine
  best-of-N   the stored 10-solve best vs the best of the gradient solves run here

    python scripts/summarize_grad_compare.py --tag v1
"""

import argparse
import json
import os

import numpy as np

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    d = os.path.join(BENCH, "results", "fastshadow", f"compare_{a.tag}")
    rows = [json.loads(l) for l in open(os.path.join(d, "per_target.jsonl"), encoding="utf-8")]
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    has_pol = all("cma_polish" in r for r in rows)
    loc = [r for r in rows if "cma_local" in r]

    def m(key, sub):
        v = [r[key][sub] for r in rows if r.get(key) and r[key].get(sub) is not None]
        return float(np.mean(v)) if v else float("nan")

    def paired(k1, s1, k2, s2):
        x = np.array([r[k1][s1] for r in rows])
        y = np.array([r[k2][s2] for r in rows])
        dlt = y - x
        se = dlt.std(ddof=1) / np.sqrt(len(dlt)) if len(dlt) > 1 else float("nan")
        return dlt.mean(), se, float((dlt > 0.005).mean()), float((dlt < -0.005).mean())

    L = [f"# Gradient (nvdiffrast) vs CMA-ES — `{a.tag}`", ""]
    ga = meta["args"]
    L += [f"{len(rows)} targets from `{os.path.relpath(ga['results'], BENCH)}`; "
          f"gradient: {ga['n_restarts']} restarts × {ga['iters']} Adam steps from the best of "
          f"{ga['n_init']} random poses, lr {ga['lr']}, {ga['seeds']} solve(s) per target. "
          "All numbers MuJoCo-measured on the target the CMA-ES solver was shown.", ""]
    L += ["## Per solve", "",
          "| method | mean IoU per solve | tc_iou (best-loss solve) | collision-free | s / solve |",
          "|---|---|---|---|---|"]
    L.append(f"| CMA-ES (stored, {rows[0]['cma']['n']} solves) | {m('cma', 'mean_iou'):.4f} | "
             f"{m('cma', 'best_loss_tc_iou'):.4f} | {m('cma', 'collision_free_frac'):.3f} | "
             f"{m('cma', 'stored_seconds_per_solve'):.1f} (other machine) |")
    if loc:
        L.append(f"| CMA-ES (re-run here, {len(loc)} targets, 1 solve) | "
                 f"{np.mean([r['cma_local']['iou'] for r in loc]):.4f} | "
                 f"{np.mean([r['cma_local']['tc_iou'] for r in loc if r['cma_local']['tc_iou'] is not None]):.4f} | "
                 f"{np.mean([r['cma_local']['collision_free'] for r in loc]):.3f} | "
                 f"{np.mean([r['cma_local']['seconds'] for r in loc]):.1f} |")
    L.append(f"| gradient | {m('grad', 'mean_iou'):.4f} | {m('grad', 'best_loss_tc_iou'):.4f} | "
             f"{m('grad', 'collision_free_frac'):.3f} | {m('grad', 'seconds_per_solve'):.1f} |")
    if has_pol:
        L.append(f"| CMA-ES → Adam polish | {m('cma_polish', 'mean_iou'):.4f} | "
                 f"{m('cma_polish', 'best_loss_tc_iou'):.4f} | {m('cma_polish', 'collision_free_frac'):.3f} | "
                 f"+{m('cma_polish', 'seconds_all_runs') / rows[0]['cma']['n']:.2f} on top |")
    dm, se, w, l_ = paired("cma", "mean_iou", "grad", "mean_iou")
    L += ["", f"Paired, gradient minus CMA-ES mean-per-solve IoU: **{dm:+.4f} ± {se:.4f}** (SE); "
          f"gradient ahead on {100 * w:.0f}% of targets, behind on {100 * l_:.0f}% (±0.005 counted as a tie).", ""]
    if loc:
        x = np.array([r["cma_local"]["iou"] for r in loc])
        y = np.array([r["grad"]["runs"][0]["iou"] for r in loc])
        L += [f"Same machine, same targets, one solve each ({len(loc)} targets): CMA-ES {x.mean():.4f} in "
              f"{np.mean([r['cma_local']['seconds'] for r in loc]):.1f}s, gradient {y.mean():.4f} in "
              f"{np.mean([r['grad']['runs'][0]['seconds'] for r in loc]):.1f}s.", ""]
    L += ["## Best of N", "", "| method | best IoU | Δ vs CMA-ES best-of-10 |", "|---|---|---|",
          f"| CMA-ES best of {rows[0]['cma']['n']} | {m('cma', 'best_iou'):.4f} | — |"]
    dm, se, _, _ = paired("cma", "best_iou", "grad", "best_iou")
    L.append(f"| gradient best of {rows[0]['grad']['n']} | {m('grad', 'best_iou'):.4f} | {dm:+.4f} ± {se:.4f} |")
    if has_pol:
        dm, se, _, _ = paired("cma", "best_iou", "cma_polish", "best_iou")
        L.append(f"| CMA-ES → polish, best of {rows[0]['cma']['n']} | {m('cma_polish', 'best_iou'):.4f} | {dm:+.4f} ± {se:.4f} |")
    L += ["", "## By subset (mean IoU per solve)", "", "| subset | n | CMA-ES | gradient | Δ |", "|---|---|---|---|---|"]
    for sub in sorted({r["subset"] for r in rows}):
        rs = [r for r in rows if r["subset"] == sub]
        c = np.mean([r["cma"]["mean_iou"] for r in rs])
        g = np.mean([r["grad"]["mean_iou"] for r in rs])
        L.append(f"| {sub} | {len(rs)} | {c:.3f} | {g:.3f} | {g - c:+.3f} |")
    out = os.path.join(d, "SUMMARY.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
