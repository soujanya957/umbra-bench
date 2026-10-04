# Provenance - `aimed-start-glyph36`

Replicates `optimized/aimed-start-probe10` on more glyphs with `scripts/ablation_pool.py`. A cell is
one variant x glyph folder. `R` cells were copied from the reference study: solved on the
reference machine listed below and not re-solved here. `L<k>` cells were solved by pool
session k below, one single-thread process per cell. Per-run PNGs exist only where a cell
was solved. A variant dir's BUDGET.md is the reference's copy until a local solve of the
first PNG in the target tree (shard 0) rewrites it with this machine's budget.

Regenerated 2026-09-15 13:33:08.

## Reference variants

| variant | machine | fleet-shadow-art | generated | CLI rebuilt from | flags |
|---|---|---|---|---|---|
| full-tiny-n3 | dutchman | 7bdef82 | 2026-09-15 03:09:39 EDT | results.json x10, all agree | `--n-robots 3 --arm-gap 0.2 --size 128 --popsize 16 --phase1-iters 4 --phase2-iters 4 --final-iters 6 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --runs 5` |
| no_aimed-tiny-n3 | dutchman | 7bdef82 | 2026-09-15 03:22:27 EDT | results.json x10, all agree | `--n-robots 3 --arm-gap 0.2 --size 128 --popsize 16 --phase1-iters 4 --phase2-iters 4 --final-iters 6 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --cfg use_voronoi_warmstart=false --runs 5` |
| no_zones-tiny-n3 | dutchman | 7bdef82 | 2026-09-15 11:20:50 EDT | BUDGET.md + reference log | `--n-robots 3 --arm-gap 0.2 --size 128 --popsize 16 --phase1-iters 4 --phase2-iters 4 --final-iters 6 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --cfg use_hungarian=false --runs 5` |

## Pool sessions

| k | started | machine | platform | fleet-shadow-art | solver vs reference | procs | command |
|---|---|---|---|---|---|---|---|
| 1 | 2026-09-15 13:29:53 | ivy | win32 Windows-10-10.0.26200-SP0 | b86634c | vs 7bdef82: 1 file(s) differ, 0 .py, optimizer.py/renderer.py/loss.py identical | 18 | `C:\Users\hexia\miniconda3\envs\fleet-shadow\python.exe scripts\ablation_pool.py --reference optimized/aimed-start-probe10 --glyphs dvsb36 --out-sub aimed-start-glyph36 --no-seed-from-reference --specs full-tiny-n3 no_aimed-tiny-n3 no_zones-tiny-n3 --procs 18 --speed 1.05 --status-every 300` |

## Cells

`R` reference copy, `L<k>` solved in pool session k, `?` on disk but not recorded by the pool, `.` missing.

| glyph | full-tiny-n3 | no_aimed-tiny-n3 | no_zones-tiny-n3 |
|---|---|---|---|
| letters_upper/A_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/B_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/C_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/D_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/E_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/F_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/G_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/H_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/I_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/J_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/K_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/L_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/M_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/N_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/O_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/P_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/Q_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/R_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/S_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/T_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/U_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/V_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/W_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/X_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/Y_dejavusans-bold | L1 | L1 | L1 |
| letters_upper/Z_dejavusans-bold | L1 | L1 | L1 |
| digits/0_dejavusans-bold | L1 | L1 | L1 |
| digits/1_dejavusans-bold | L1 | L1 | L1 |
| digits/2_dejavusans-bold | L1 | L1 | L1 |
| digits/3_dejavusans-bold | L1 | L1 | L1 |
| digits/4_dejavusans-bold | L1 | L1 | L1 |
| digits/5_dejavusans-bold | L1 | L1 | L1 |
| digits/6_dejavusans-bold | L1 | L1 | L1 |
| digits/7_dejavusans-bold | L1 | L1 | L1 |
| digits/8_dejavusans-bold | L1 | L1 | L1 |
| digits/9_dejavusans-bold | L1 | L1 | L1 |

Totals: 0 reference, 108 local, 0 unrecorded, 0 missing.
