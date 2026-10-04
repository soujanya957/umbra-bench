# Provenance - `ablation-glyph36`

Replicates `optimized/ablation-probe10` on more glyphs with `scripts/ablation_pool.py`. A cell is
one variant x glyph folder. `R` cells were copied from the reference study: solved on the
reference machine listed below and not re-solved here. `L<k>` cells were solved by pool
session k below, one single-thread process per cell. Per-run PNGs exist only where a cell
was solved. A variant dir's BUDGET.md is the reference's copy until a local solve of the
first PNG in the target tree (shard 0) rewrites it with this machine's budget.

Regenerated 2026-09-15 13:29:51.

## Reference variants

| variant | machine | fleet-shadow-art | generated | CLI rebuilt from | flags |
|---|---|---|---|---|---|
| flat-n3 | dutchman | 7bdef82 | 2026-09-14 14:38:19 EDT | results.json x10, all agree | `--n-robots 3 --arm-gap 0.2 --size 128 --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --flat --runs 5` |
| fwd_joint-n3 | dutchman | 7bdef82 | 2026-09-14 14:52:11 EDT | results.json x10, all agree | `--n-robots 3 --arm-gap 0.2 --size 128 --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --cfg use_hungarian=false --cfg use_voronoi_warmstart=false --cfg phase2_iters=0 --cfg use_icp_restarts=false --cfg fd_iters=0 --runs 5` |
| plus_assign-n3 | dutchman | 7bdef82 | 2026-09-14 15:03:39 EDT | results.json x10, all agree | `--n-robots 3 --arm-gap 0.2 --size 128 --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --cfg use_voronoi_warmstart=false --cfg phase2_iters=0 --cfg use_icp_restarts=false --cfg fd_iters=0 --runs 5` |
| plus_voronoi-n3 | dutchman | 7bdef82 | 2026-09-14 15:15:11 EDT | results.json x10, all agree | `--n-robots 3 --arm-gap 0.2 --size 128 --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --cfg phase2_iters=0 --cfg use_icp_restarts=false --cfg fd_iters=0 --runs 5` |
| plus_backward-n3 | dutchman | 7bdef82 | 2026-09-14 15:26:43 EDT | results.json x10, all agree | `--n-robots 3 --arm-gap 0.2 --size 128 --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --cfg use_icp_restarts=false --cfg fd_iters=0 --runs 5` |
| plus_icp-n3 | dutchman | 7bdef82 | 2026-09-14 15:45:20 EDT | results.json x10, all agree | `--n-robots 3 --arm-gap 0.2 --size 128 --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --cfg fd_iters=0 --runs 5` |
| full-n3 | dutchman | 7bdef82 | 2026-09-14 16:03:54 EDT | results.json x10, all agree | `--n-robots 3 --arm-gap 0.2 --size 128 --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --runs 5` |
| full_no_assign-n3 | dutchman | 7bdef82 | 2026-09-14 17:07:40 EDT | results.json x10, all agree | `--n-robots 3 --arm-gap 0.2 --size 128 --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --cfg use_hungarian=false --runs 5` |
| full_no_voronoi-n3 | dutchman | 7bdef82 | 2026-09-14 16:24:16 EDT | results.json x10, all agree | `--n-robots 3 --arm-gap 0.2 --size 128 --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --cfg use_voronoi_warmstart=false --runs 5` |
| full_no_backward-n3 | dutchman | 7bdef82 | 2026-09-14 17:27:59 EDT | results.json x10, all agree | `--n-robots 3 --arm-gap 0.2 --size 128 --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --cfg phase2_iters=0 --runs 5` |
| full_no_icp-n3 | dutchman | 7bdef82 | 2026-09-14 17:41:10 EDT | results.json x10, all agree | `--n-robots 3 --arm-gap 0.2 --size 128 --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --cfg use_icp_restarts=false --runs 5` |
| full_no_fd-n3 | dutchman | 7bdef82 | 2026-09-14 18:01:26 EDT | results.json x10, all agree | `--n-robots 3 --arm-gap 0.2 --size 128 --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --cfg fd_iters=0 --runs 5` |
| flat-n5 | dutchman | 7bdef82 | 2026-09-14 18:20:07 EDT | results.json x10, all agree | `--n-robots 5 --arm-gap 0.2 --size 128 --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --flat --runs 5` |
| full-n5 | dutchman | 7bdef82 | 2026-09-14 18:48:45 EDT | results.json x10, all agree | `--n-robots 5 --arm-gap 0.2 --size 128 --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --runs 5` |
| full_long-n3 | dutchman | 7bdef82 | 2026-09-14 21:56:17 EDT | results.json x10, all agree | `--n-robots 3 --arm-gap 0.2 --size 128 --popsize 128 --phase1-iters 64 --phase2-iters 48 --final-iters 32 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --runs 3` |
| full_long_no_icp-n3 | dutchman | 7bdef82 | 2026-09-14 22:23:18 EDT | results.json x10, all agree | `--n-robots 3 --arm-gap 0.2 --size 128 --popsize 128 --phase1-iters 64 --phase2-iters 48 --final-iters 32 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --cfg use_icp_restarts=false --runs 3` |
| full_no_icp-n5 | dutchman | 7bdef82 | 2026-09-14 20:49:12 EDT | results.json x10, all agree | `--n-robots 5 --arm-gap 0.2 --size 128 --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --cfg use_icp_restarts=false --runs 5` |
| full_no_fd-n5 | dutchman | 7bdef82 | 2026-09-14 20:19:24 EDT | results.json x10, all agree | `--n-robots 5 --arm-gap 0.2 --size 128 --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --cfg fd_iters=0 --runs 5` |
| full_no_manifold-n5 | dutchman | 7bdef82 | 2026-09-14 19:36:17 EDT | results.json x10, all agree | `--n-robots 5 --arm-gap 0.2 --size 128 --popsize 48 --phase1-iters 16 --phase2-iters 16 --final-iters 30 --floor-penalty 40.0 --collision-penalty 400.0 --self-collision-penalty 200.0 --no-adaptive-final --cfg use_manifold_search=false --runs 5` |

## Pool sessions

| k | started | machine | platform | fleet-shadow-art | solver vs reference | procs | command |
|---|---|---|---|---|---|---|---|
| 1 | 2026-09-15 13:12:52 | ivy | win32 Windows-10-10.0.26200-SP0 | b86634c | vs 7bdef82: 1 file(s) differ, 0 .py, optimizer.py/renderer.py/loss.py identical | 18 | `C:\Users\hexia\miniconda3\envs\fleet-shadow\python.exe scripts\ablation_pool.py --reference optimized/ablation-probe10 --glyphs dvsb36 --out-sub ablation-glyph36 --no-seed-from-reference --procs 18 --speed 1.05 --status-every 300` |
| 2 | 2026-09-15 13:14:25 | ivy | win32 Windows-10-10.0.26200-SP0 | b86634c | vs 7bdef82: 1 file(s) differ, 0 .py, optimizer.py/renderer.py/loss.py identical | 18 | `C:\Users\hexia\miniconda3\envs\fleet-shadow\python.exe scripts\ablation_pool.py --reference optimized/ablation-probe10 --glyphs dvsb36 --out-sub ablation-glyph36 --no-seed-from-reference --specs full_long-n3 full_long_no_icp-n3 --procs 18 --speed 1.05 --status-every 300` |

## Cells

`R` reference copy, `L<k>` solved in pool session k, `?` on disk but not recorded by the pool, `.` missing.

| glyph | flat-n3 | fwd_joint-n3 | plus_assign-n3 | plus_voronoi-n3 | plus_backward-n3 | plus_icp-n3 | full-n3 | full_no_assign-n3 | full_no_voronoi-n3 | full_no_backward-n3 | full_no_icp-n3 | full_no_fd-n3 | flat-n5 | full-n5 | full_long-n3 | full_long_no_icp-n3 | full_no_icp-n5 | full_no_fd-n5 | full_no_manifold-n5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| letters_upper/A_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/B_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/C_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/D_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/E_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/F_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/G_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/H_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/I_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/J_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/K_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/L_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/M_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/N_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/O_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/P_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/Q_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/R_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/S_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/T_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/U_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/V_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/W_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/X_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/Y_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| letters_upper/Z_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| digits/0_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| digits/1_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| digits/2_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| digits/3_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| digits/4_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| digits/5_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| digits/6_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| digits/7_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| digits/8_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |
| digits/9_dejavusans-bold | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | L2 | L2 | ? | ? | ? |

Totals: 0 reference, 72 local, 612 unrecorded, 0 missing.

## Earlier PROVENANCE.md

Found here before this pool first ran and not written by it; its cells show as `?` above.

```text
- 2026-09-15 11:56:31 on `ivy` (win32), fleet-shadow-art @ `b86634c` (C:\Users\hexia\Documents\GitHub\fleet-shadow-art-main); every cell solved on this machine, none copied from ablation-probe10; 18 single-thread processes; launcher glyph36_launch.py tiers [['flat-n3', 'fwd_joint-n3', 'plus_assign-n3', 'plus_voronoi-n3', 'plus_backward-n3', 'plus_icp-n3', 'full-n3', 'full_no_assign-n3', 'full_no_voronoi-n3', 'full_no_backward-n3', 'full_no_icp-n3', 'full_no_fd-n3'], ['flat-n5', 'full-n5', 'full_no_icp-n5', 'full_no_fd-n5', 'full_no_manifold-n5']]
```
