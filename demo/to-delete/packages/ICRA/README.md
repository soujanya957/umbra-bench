# demo package: ICRA

Self-contained material for one show. Per element:

* `frames/` -- for a clip: 1-bit 1920x1080 canvas frames, dark = shadow,
  already at authored position/scale; align elements by `source_frame_ids`
  in `meta.json`, never by frame index. For a library element: the solver's
  rendered `silhouette.png` (128 px solver frame -- placement is yours) and
  the authored `target.png`.
* `joints.csv` -- one row per frame: `arm0_q0..arm{A}_q5`, **radians**,
  6 dof per arm, against the URDF named in `meta.json` and the stage layout
  recorded there. A library element has a single row (a held pose).
* `meta.json` -- fps, provenance, fit, IoU.

`video/` holds the already-composited scene mp4s if they were built.

## Robot deploy: `choreo/`

One `choreo/<element>.json` per element, in the exact "unified clip envelope"
Shadow_robot_ui consumes. To deploy:

1. copy `choreo/*.json` into `fleet-shadow-art/choreographies/`
2. start the UI (`Shadow_robot_ui/start.sh`) and open **Play** -- each element
   lists by filename; preview, arrange on the timeline, deploy.

The scene block carries the solve's own rig (light 1.0 m in front of arm 0,
arms 0.2 m apart in depth, screen 2.4 m behind, mapped into the UI frame), so
the preview's shadow geometry matches the solve. Joints are UNCALIBRATED
solver radians -- the arm driver applies each robot's visual offsets at send
time; do not pre-apply them. Serial ports are an operator setting in the
Robot console, never part of a clip. Robot ids are SR101 (nearest the light),
SR102, SR103; if your stage config names differ, Play remaps positionally and
says so in the status bar.

**Deploy caveat (unchanged from demo/README.md section C):** solver joints are
model-space. Physical playback still needs the lab's base placement, the
Play/render_server stage JSON, and the SR10x unit mapping -- the three
TODO-user facts. Everything knowable without the lab is in this folder.

Combine shadows by union (`np.minimum` on greyscale). Manual assembly is the
intended path.
