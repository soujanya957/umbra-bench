#!/bin/sh
# Every solve in manifest.csv, in table order. Run from the
# umbra-bench repo root, then refresh Play's library.
#
# Mixed spacings on purpose: each choreography carries the
# bases its own solve used (pack.run_arm_gap), so a 0.15 clip
# and a 0.25 one are both correct -- they just describe
# different rigs. Set the stage to match before deploying one.
#
# No `set -e`: write_choreo refuses a clip whose keyframes
# demand more per-joint travel than the planner's LARGE_Q_JUMP,
# and that refusal is a result, not a reason to abandon the
# other 134. Failures are counted and listed at the end; add
# --force to a specific line only if you mean to ship it.
fail=0

# ICRA_scene_01_bird                         3 arms, gap 0.15?
python demo/export_library_clip.py --sequence ICRA_scene_01_bird || { echo "  FAILED: ICRA_scene_01_bird"; fail=$((fail+1)); }
# ICRA_scene_01_bird__optimizer_n5           5 arms, gap 0.25
python demo/export_library_clip.py --sequence ICRA_scene_01_bird --source optimizer_n5 || { echo "  FAILED: ICRA_scene_01_bird__optimizer_n5"; fail=$((fail+1)); }
# ICRA_scene_01_butterfly                    3 arms, gap 0.15?
python demo/export_library_clip.py --sequence ICRA_scene_01_butterfly || { echo "  FAILED: ICRA_scene_01_butterfly"; fail=$((fail+1)); }
# ICRA_scene_01_butterfly__optimizer_n5      5 arms, gap 0.25
python demo/export_library_clip.py --sequence ICRA_scene_01_butterfly --source optimizer_n5 || { echo "  FAILED: ICRA_scene_01_butterfly__optimizer_n5"; fail=$((fail+1)); }
# ICRA_scene_01_snake                        3 arms, gap 0.15?
python demo/export_library_clip.py --sequence ICRA_scene_01_snake || { echo "  FAILED: ICRA_scene_01_snake"; fail=$((fail+1)); }
# ICRA_scene_01_snake__optimizer_n5          5 arms, gap 0.25
python demo/export_library_clip.py --sequence ICRA_scene_01_snake --source optimizer_n5 || { echo "  FAILED: ICRA_scene_01_snake__optimizer_n5"; fail=$((fail+1)); }
# ICRA_scene_01_star                         3 arms, gap 0.15?
python demo/export_library_clip.py --sequence ICRA_scene_01_star || { echo "  FAILED: ICRA_scene_01_star"; fail=$((fail+1)); }
# ICRA_scene_03_a                            3 arms, gap 0.15?
python demo/export_library_clip.py --sequence ICRA_scene_03_a || { echo "  FAILED: ICRA_scene_03_a"; fail=$((fail+1)); }
# ICRA_scene_03_c                            3 arms, gap 0.15?
python demo/export_library_clip.py --sequence ICRA_scene_03_c || { echo "  FAILED: ICRA_scene_03_c"; fail=$((fail+1)); }
# ICRA_scene_03_i                            3 arms, gap 0.15?
python demo/export_library_clip.py --sequence ICRA_scene_03_i || { echo "  FAILED: ICRA_scene_03_i"; fail=$((fail+1)); }
# ICRA_scene_03_r                            3 arms, gap 0.15?
python demo/export_library_clip.py --sequence ICRA_scene_03_r || { echo "  FAILED: ICRA_scene_03_r"; fail=$((fail+1)); }
# bird                                       3 arms, gap 0.15?
python demo/export_library_clip.py --sequence bird || { echo "  FAILED: bird"; fail=$((fail+1)); }
# bird_scene_01_bird                         3 arms, gap 0.15?
python demo/export_library_clip.py --sequence bird_scene_01_bird || { echo "  FAILED: bird_scene_01_bird"; fail=$((fail+1)); }
# bird_scene_01_bird__optimizer_g25          3 arms, gap 0.25
python demo/export_library_clip.py --sequence bird_scene_01_bird --source optimizer_g25 || { echo "  FAILED: bird_scene_01_bird__optimizer_g25"; fail=$((fail+1)); }
# butterfly_scene_01_butterfly               3 arms, gap 0.25
python demo/export_library_clip.py --sequence butterfly_scene_01_butterfly || { echo "  FAILED: butterfly_scene_01_butterfly"; fail=$((fail+1)); }
# butterfly_scene_01_butterfly__optimizer_n5 5 arms, gap 0.25
python demo/export_library_clip.py --sequence butterfly_scene_01_butterfly --source optimizer_n5 || { echo "  FAILED: butterfly_scene_01_butterfly__optimizer_n5"; fail=$((fail+1)); }
# cheer_n5                                   5 arms, gap 0.15?
python demo/export_library_clip.py --sequence cheer_n5 || { echo "  FAILED: cheer_n5"; fail=$((fail+1)); }
# chicken_egg_scene_01_egg                   3 arms, gap 0.15?
python demo/export_library_clip.py --sequence chicken_egg_scene_01_egg || { echo "  FAILED: chicken_egg_scene_01_egg"; fail=$((fail+1)); }
# chicken_egg_scene_01_egg__optimizer_g25    3 arms, gap 0.25
python demo/export_library_clip.py --sequence chicken_egg_scene_01_egg --source optimizer_g25 || { echo "  FAILED: chicken_egg_scene_01_egg__optimizer_g25"; fail=$((fail+1)); }
# fabric_scene_01_fabric                     3 arms, gap 0.25
python demo/export_library_clip.py --sequence fabric_scene_01_fabric || { echo "  FAILED: fabric_scene_01_fabric"; fail=$((fail+1)); }
# fabric_scene_01_fabric__optimizer_n5       5 arms, gap 0.25
python demo/export_library_clip.py --sequence fabric_scene_01_fabric --source optimizer_n5 || { echo "  FAILED: fabric_scene_01_fabric__optimizer_n5"; fail=$((fail+1)); }
# family_ad_scene_01_I                       3 arms, gap 0.15?
python demo/export_library_clip.py --sequence family_ad_scene_01_I || { echo "  FAILED: family_ad_scene_01_I"; fail=$((fail+1)); }
# family_ad_scene_02_F                       3 arms, gap 0.15?
python demo/export_library_clip.py --sequence family_ad_scene_02_F || { echo "  FAILED: family_ad_scene_02_F"; fail=$((fail+1)); }
# family_ad_scene_03_F                       3 arms, gap 0.15?
python demo/export_library_clip.py --sequence family_ad_scene_03_F || { echo "  FAILED: family_ad_scene_03_F"; fail=$((fail+1)); }
# family_ad_scene_03_I                       3 arms, gap 0.15?
python demo/export_library_clip.py --sequence family_ad_scene_03_I || { echo "  FAILED: family_ad_scene_03_I"; fail=$((fail+1)); }
# family_ad_scene_04_M                       3 arms, gap 0.15?
python demo/export_library_clip.py --sequence family_ad_scene_04_M || { echo "  FAILED: family_ad_scene_04_M"; fail=$((fail+1)); }
# family_ad_scene_05_I                       3 arms, gap 0.15?
python demo/export_library_clip.py --sequence family_ad_scene_05_I || { echo "  FAILED: family_ad_scene_05_I"; fail=$((fail+1)); }
# family_ad_scene_05_M                       3 arms, gap 0.15?
python demo/export_library_clip.py --sequence family_ad_scene_05_M || { echo "  FAILED: family_ad_scene_05_M"; fail=$((fail+1)); }
# family_ad_scene_06_A                       3 arms, gap 0.15?
python demo/export_library_clip.py --sequence family_ad_scene_06_A || { echo "  FAILED: family_ad_scene_06_A"; fail=$((fail+1)); }
# family_ad_scene_06_F                       3 arms, gap 0.15?
python demo/export_library_clip.py --sequence family_ad_scene_06_F || { echo "  FAILED: family_ad_scene_06_F"; fail=$((fail+1)); }
# family_ad_scene_06_I                       3 arms, gap 0.15?
python demo/export_library_clip.py --sequence family_ad_scene_06_I || { echo "  FAILED: family_ad_scene_06_I"; fail=$((fail+1)); }
# family_ad_scene_06_L                       3 arms, gap 0.15?
python demo/export_library_clip.py --sequence family_ad_scene_06_L || { echo "  FAILED: family_ad_scene_06_L"; fail=$((fail+1)); }
# family_ad_scene_06_L_stab                  3 arms, gap 0.15?
python demo/export_library_clip.py --sequence family_ad_scene_06_L_stab || { echo "  FAILED: family_ad_scene_06_L_stab"; fail=$((fail+1)); }
# family_ad_scene_06_M                       3 arms, gap 0.15?
python demo/export_library_clip.py --sequence family_ad_scene_06_M || { echo "  FAILED: family_ad_scene_06_M"; fail=$((fail+1)); }
# family_ad_scene_06_Y                       3 arms, gap 0.15?
python demo/export_library_clip.py --sequence family_ad_scene_06_Y || { echo "  FAILED: family_ad_scene_06_Y"; fail=$((fail+1)); }
# family_ad_scene_06_Y_stab                  3 arms, gap 0.15?
python demo/export_library_clip.py --sequence family_ad_scene_06_Y_stab || { echo "  FAILED: family_ad_scene_06_Y_stab"; fail=$((fail+1)); }
# fish_scene_01_fish                         3 arms, gap 0.15?
python demo/export_library_clip.py --sequence fish_scene_01_fish || { echo "  FAILED: fish_scene_01_fish"; fail=$((fail+1)); }
# fish_scene_01_fish__optimizer_g25          3 arms, gap 0.25
python demo/export_library_clip.py --sequence fish_scene_01_fish --source optimizer_g25 || { echo "  FAILED: fish_scene_01_fish__optimizer_g25"; fail=$((fail+1)); }
# flower                                     3 arms, gap 0.15?
python demo/export_library_clip.py --sequence flower || { echo "  FAILED: flower"; fail=$((fail+1)); }
# flower__optimizer_seamfix                  3 arms, gap 0.15?
python demo/export_library_clip.py --sequence flower --source optimizer_seamfix || { echo "  FAILED: flower__optimizer_seamfix"; fail=$((fail+1)); }
# flower_blooming_scene_01_flower            3 arms, gap 0.15?
python demo/export_library_clip.py --sequence flower_blooming_scene_01_flower || { echo "  FAILED: flower_blooming_scene_01_flower"; fail=$((fail+1)); }
# flower_blooming_scene_01_flower__optimizer_g25 3 arms, gap 0.25
python demo/export_library_clip.py --sequence flower_blooming_scene_01_flower --source optimizer_g25 || { echo "  FAILED: flower_blooming_scene_01_flower__optimizer_g25"; fail=$((fail+1)); }
# heart_scene_01_heart                       3 arms, gap 0.25
python demo/export_library_clip.py --sequence heart_scene_01_heart || { echo "  FAILED: heart_scene_01_heart"; fail=$((fail+1)); }
# heart_scene_01_heart__optimizer_n5         5 arms, gap 0.25
python demo/export_library_clip.py --sequence heart_scene_01_heart --source optimizer_n5 || { echo "  FAILED: heart_scene_01_heart__optimizer_n5"; fail=$((fail+1)); }
# icra_2027_demo_scene_01_bird_i_2           3 arms, gap 0.15?
python demo/export_library_clip.py --sequence icra_2027_demo_scene_01_bird_i_2 || { echo "  FAILED: icra_2027_demo_scene_01_bird_i_2"; fail=$((fail+1)); }
# icra_2027_demo_scene_01_butterfly_R_2      3 arms, gap 0.15?
python demo/export_library_clip.py --sequence icra_2027_demo_scene_01_butterfly_R_2 || { echo "  FAILED: icra_2027_demo_scene_01_butterfly_R_2"; fail=$((fail+1)); }
# icra_2027_demo_scene_01_snake_c_0          3 arms, gap 0.15?
python demo/export_library_clip.py --sequence icra_2027_demo_scene_01_snake_c_0 || { echo "  FAILED: icra_2027_demo_scene_01_snake_c_0"; fail=$((fail+1)); }
# icra_2027_demo_scene_01_star_A_7           3 arms, gap 0.15?
python demo/export_library_clip.py --sequence icra_2027_demo_scene_01_star_A_7 || { echo "  FAILED: icra_2027_demo_scene_01_star_A_7"; fail=$((fail+1)); }
# me_scene_01_me                             3 arms, gap 0.25
python demo/export_library_clip.py --sequence me_scene_01_me || { echo "  FAILED: me_scene_01_me"; fail=$((fail+1)); }
# me_scene_01_me__optimizer_n5               5 arms, gap 0.25
python demo/export_library_clip.py --sequence me_scene_01_me --source optimizer_n5 || { echo "  FAILED: me_scene_01_me__optimizer_n5"; fail=$((fail+1)); }
# moose_scene_01_moose__optimizer_g25        3 arms, gap 0.25
python demo/export_library_clip.py --sequence moose_scene_01_moose --source optimizer_g25 || { echo "  FAILED: moose_scene_01_moose__optimizer_g25"; fail=$((fail+1)); }
# moose_scene_01_moose_stab                  3 arms, gap 0.15?
python demo/export_library_clip.py --sequence moose_scene_01_moose_stab || { echo "  FAILED: moose_scene_01_moose_stab"; fail=$((fail+1)); }
# mushroom_growing_scene_01_mushroom         3 arms, gap 0.25
python demo/export_library_clip.py --sequence mushroom_growing_scene_01_mushroom || { echo "  FAILED: mushroom_growing_scene_01_mushroom"; fail=$((fail+1)); }
# mushroom_growing_scene_01_mushroom__optimizer_n5 5 arms, gap 0.25
python demo/export_library_clip.py --sequence mushroom_growing_scene_01_mushroom --source optimizer_n5 || { echo "  FAILED: mushroom_growing_scene_01_mushroom__optimizer_n5"; fail=$((fail+1)); }
# pixar_scene_01_lamp                        3 arms, gap 0.15?
python demo/export_library_clip.py --sequence pixar_scene_01_lamp || { echo "  FAILED: pixar_scene_01_lamp"; fail=$((fail+1)); }
# pixar_scene_01_lamp__optimizer_n5          5 arms, gap 0.25
python demo/export_library_clip.py --sequence pixar_scene_01_lamp --source optimizer_n5 || { echo "  FAILED: pixar_scene_01_lamp__optimizer_n5"; fail=$((fail+1)); }
# pixar_scene_01_lamp_stab                   3 arms, gap 0.15?
python demo/export_library_clip.py --sequence pixar_scene_01_lamp_stab || { echo "  FAILED: pixar_scene_01_lamp_stab"; fail=$((fail+1)); }
# pixar_scene_02_R                           3 arms, gap 0.15?
python demo/export_library_clip.py --sequence pixar_scene_02_R || { echo "  FAILED: pixar_scene_02_R"; fail=$((fail+1)); }
# pixar_scene_02_lamp                        3 arms, gap 0.15?
python demo/export_library_clip.py --sequence pixar_scene_02_lamp || { echo "  FAILED: pixar_scene_02_lamp"; fail=$((fail+1)); }
# pixar_scene_02_lamp__optimizer_n5          5 arms, gap 0.25
python demo/export_library_clip.py --sequence pixar_scene_02_lamp --source optimizer_n5 || { echo "  FAILED: pixar_scene_02_lamp__optimizer_n5"; fail=$((fail+1)); }
# pixar_scene_03_I                           3 arms, gap 0.15?
python demo/export_library_clip.py --sequence pixar_scene_03_I || { echo "  FAILED: pixar_scene_03_I"; fail=$((fail+1)); }
# pixar_scene_03_I__optimizer_n5             5 arms, gap 0.25
python demo/export_library_clip.py --sequence pixar_scene_03_I --source optimizer_n5 || { echo "  FAILED: pixar_scene_03_I__optimizer_n5"; fail=$((fail+1)); }
# pixar_scene_03_lamp                        3 arms, gap 0.15?
python demo/export_library_clip.py --sequence pixar_scene_03_lamp || { echo "  FAILED: pixar_scene_03_lamp"; fail=$((fail+1)); }
# pixar_scene_03_lamp__optimizer_n5          5 arms, gap 0.25
python demo/export_library_clip.py --sequence pixar_scene_03_lamp --source optimizer_n5 || { echo "  FAILED: pixar_scene_03_lamp__optimizer_n5"; fail=$((fail+1)); }
# pixar_scene_03_lamp_stab                   3 arms, gap 0.15?
python demo/export_library_clip.py --sequence pixar_scene_03_lamp_stab || { echo "  FAILED: pixar_scene_03_lamp_stab"; fail=$((fail+1)); }
# pixar_scene_04_lamp                        3 arms, gap 0.15?
python demo/export_library_clip.py --sequence pixar_scene_04_lamp || { echo "  FAILED: pixar_scene_04_lamp"; fail=$((fail+1)); }
# pixar_scene_04_lamp__optimizer_n5          5 arms, gap 0.25
python demo/export_library_clip.py --sequence pixar_scene_04_lamp --source optimizer_n5 || { echo "  FAILED: pixar_scene_04_lamp__optimizer_n5"; fail=$((fail+1)); }
# plant                                      3 arms, gap 0.15?
python demo/export_library_clip.py --sequence plant || { echo "  FAILED: plant"; fail=$((fail+1)); }
# reeds_n3                                   3 arms, gap 0.15?
python demo/export_library_clip.py --sequence reeds_n3 || { echo "  FAILED: reeds_n3"; fail=$((fail+1)); }
# reeds_n5                                   5 arms, gap 0.15?
python demo/export_library_clip.py --sequence reeds_n5 || { echo "  FAILED: reeds_n5"; fail=$((fail+1)); }
# star_ds12                                  3 arms, gap 0.15?
python demo/export_library_clip.py --sequence star_ds12 || { echo "  FAILED: star_ds12"; fail=$((fail+1)); }
# star_ds12__optimizer_n5                    5 arms, gap 0.15?
python demo/export_library_clip.py --sequence star_ds12 --source optimizer_n5 || { echo "  FAILED: star_ds12__optimizer_n5"; fail=$((fail+1)); }
# star_spin                                  3 arms, gap 0.15?
python demo/export_library_clip.py --sequence star_spin || { echo "  FAILED: star_spin"; fail=$((fail+1)); }
# star_spinning_scene_01_star                3 arms, gap 0.15?
python demo/export_library_clip.py --sequence star_spinning_scene_01_star || { echo "  FAILED: star_spinning_scene_01_star"; fail=$((fail+1)); }
# star_spinning_scene_01_star__optimizer_g25 3 arms, gap 0.25
python demo/export_library_clip.py --sequence star_spinning_scene_01_star --source optimizer_g25 || { echo "  FAILED: star_spinning_scene_01_star__optimizer_g25"; fail=$((fail+1)); }
# star_spinning_scene_01_star__optimizer_n5  5 arms, gap 0.15?
python demo/export_library_clip.py --sequence star_spinning_scene_01_star --source optimizer_n5 || { echo "  FAILED: star_spinning_scene_01_star__optimizer_n5"; fail=$((fail+1)); }
# stick_wave                                 3 arms, gap 0.15?
python demo/export_library_clip.py --sequence stick_wave || { echo "  FAILED: stick_wave"; fail=$((fail+1)); }
# sun_scene_01_sun                           3 arms, gap 0.25
python demo/export_library_clip.py --sequence sun_scene_01_sun || { echo "  FAILED: sun_scene_01_sun"; fail=$((fail+1)); }
# sun_scene_01_sun__optimizer_n5             5 arms, gap 0.25
python demo/export_library_clip.py --sequence sun_scene_01_sun --source optimizer_n5 || { echo "  FAILED: sun_scene_01_sun__optimizer_n5"; fail=$((fail+1)); }
# swan_scene_01_swan                         3 arms, gap 0.25
python demo/export_library_clip.py --sequence swan_scene_01_swan || { echo "  FAILED: swan_scene_01_swan"; fail=$((fail+1)); }
# swan_scene_01_swan__optimizer_n5           5 arms, gap 0.25
python demo/export_library_clip.py --sequence swan_scene_01_swan --source optimizer_n5 || { echo "  FAILED: swan_scene_01_swan__optimizer_n5"; fail=$((fail+1)); }
# t_rex_scene_01_t-rex                       3 arms, gap 0.15?
python demo/export_library_clip.py --sequence t_rex_scene_01_t-rex || { echo "  FAILED: t_rex_scene_01_t-rex"; fail=$((fail+1)); }
# t_rex_scene_01_t-rex__optimizer_g25        3 arms, gap 0.25
python demo/export_library_clip.py --sequence t_rex_scene_01_t-rex --source optimizer_g25 || { echo "  FAILED: t_rex_scene_01_t-rex__optimizer_g25"; fail=$((fail+1)); }
# teacup_scene_01_teacup                     3 arms, gap 0.25
python demo/export_library_clip.py --sequence teacup_scene_01_teacup || { echo "  FAILED: teacup_scene_01_teacup"; fail=$((fail+1)); }
# teacup_scene_01_teacup__optimizer_n5       5 arms, gap 0.25
python demo/export_library_clip.py --sequence teacup_scene_01_teacup --source optimizer_n5 || { echo "  FAILED: teacup_scene_01_teacup__optimizer_n5"; fail=$((fail+1)); }
# teapot_scene_01_teapot                     3 arms, gap 0.25
python demo/export_library_clip.py --sequence teapot_scene_01_teapot || { echo "  FAILED: teapot_scene_01_teapot"; fail=$((fail+1)); }
# teapot_scene_01_teapot__optimizer_n5       5 arms, gap 0.25
python demo/export_library_clip.py --sequence teapot_scene_01_teapot --source optimizer_n5 || { echo "  FAILED: teapot_scene_01_teapot__optimizer_n5"; fail=$((fail+1)); }
# tree_ds12                                  3 arms, gap 0.15?
python demo/export_library_clip.py --sequence tree_ds12 || { echo "  FAILED: tree_ds12"; fail=$((fail+1)); }
# tree_ds12__optimizer_tw005                 3 arms, gap 0.15?
python demo/export_library_clip.py --sequence tree_ds12 --source optimizer_tw005 || { echo "  FAILED: tree_ds12__optimizer_tw005"; fail=$((fail+1)); }
# tree_scene_01_tree                         3 arms, gap 0.15?
python demo/export_library_clip.py --sequence tree_scene_01_tree || { echo "  FAILED: tree_scene_01_tree"; fail=$((fail+1)); }
# tree_scene_01_tree__optimizer_g25          3 arms, gap 0.25
python demo/export_library_clip.py --sequence tree_scene_01_tree --source optimizer_g25 || { echo "  FAILED: tree_scene_01_tree__optimizer_g25"; fail=$((fail+1)); }
# tree_scene_01_tree__optimizer_tw005        3 arms, gap 0.15?
python demo/export_library_clip.py --sequence tree_scene_01_tree --source optimizer_tw005 || { echo "  FAILED: tree_scene_01_tree__optimizer_tw005"; fail=$((fail+1)); }
# triangle                                   3 arms, gap 0.15?
python demo/export_library_clip.py --sequence triangle || { echo "  FAILED: triangle"; fail=$((fail+1)); }
# triangle_stab                              3 arms, gap 0.15?
python demo/export_library_clip.py --sequence triangle_stab || { echo "  FAILED: triangle_stab"; fail=$((fail+1)); }
# two_arm_wave_n5                            5 arms, gap 0.15?
python demo/export_library_clip.py --sequence two_arm_wave_n5 || { echo "  FAILED: two_arm_wave_n5"; fail=$((fail+1)); }
# umbrella_scene_01_umbrella                 3 arms, gap 0.25
python demo/export_library_clip.py --sequence umbrella_scene_01_umbrella || { echo "  FAILED: umbrella_scene_01_umbrella"; fail=$((fail+1)); }
# umbrella_scene_01_umbrella__optimizer_n5   5 arms, gap 0.25
python demo/export_library_clip.py --sequence umbrella_scene_01_umbrella --source optimizer_n5 || { echo "  FAILED: umbrella_scene_01_umbrella__optimizer_n5"; fail=$((fail+1)); }
# windmill_n3                                3 arms, gap 0.15?
python demo/export_library_clip.py --sequence windmill_n3 || { echo "  FAILED: windmill_n3"; fail=$((fail+1)); }
# windmill_n3__optimizer_seamfix             3 arms, gap 0.15?
python demo/export_library_clip.py --sequence windmill_n3 --source optimizer_seamfix || { echo "  FAILED: windmill_n3__optimizer_seamfix"; fail=$((fail+1)); }
# windmill_n5                                5 arms, gap 0.15?
python demo/export_library_clip.py --sequence windmill_n5 || { echo "  FAILED: windmill_n5"; fail=$((fail+1)); }
# windmill_n5__optimizer_seamfix             5 arms, gap 0.15?
python demo/export_library_clip.py --sequence windmill_n5 --source optimizer_seamfix || { echo "  FAILED: windmill_n5__optimizer_seamfix"; fail=$((fail+1)); }
# windmill_scene_01_windmill                 3 arms, gap 0.15?
python demo/export_library_clip.py --sequence windmill_scene_01_windmill || { echo "  FAILED: windmill_scene_01_windmill"; fail=$((fail+1)); }
# windmill_scene_01_windmill__optimizer_g25  3 arms, gap 0.25
python demo/export_library_clip.py --sequence windmill_scene_01_windmill --source optimizer_g25 || { echo "  FAILED: windmill_scene_01_windmill__optimizer_g25"; fail=$((fail+1)); }
# wiper                                      3 arms, gap 0.15?
python demo/export_library_clip.py --sequence wiper || { echo "  FAILED: wiper"; fail=$((fail+1)); }

echo "$fail export(s) refused; the rest are in fleet-shadow-art/choreographies/"
