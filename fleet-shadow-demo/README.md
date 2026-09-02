# fleet-shadow-demo — frame prep pipeline

Two scripts that take the CCTV *FAMILY* spot from mp4 to per-letter cutouts
ready for shadow optimization.

Original source: 1920×1080, 25 fps, 90 s, 2250 frames. The working input is
your edited cut — the edit *is* the frame selection, so every frame it
contains gets extracted and segmented.

---

## Step 1 — scenes + frames  (`01_split_scenes.py`)

This is the script to use for the edited cut. `01_extract_frames.sh` is kept
for the un-split case.

```bash
python3 01_split_scenes.py FAMILY_trimmed.mp4 --sample 5
```

`--sample 5` keeps every 5th frame within each scene (25 fps source → 5 fps),
anchored at each scene's first frame. Frame ids stay **global and true to the
source**, so a sampled scene reads f0001, f0006, f0011… — the ids are not
renumbered, and the gaps are real.

It finds the white frames you used between clips, uses them as the cut points,
and leaves them **out** of the extracted frames.

Current result — 751 frames in, 5 separator runs of 12 frames each removed,
then sampled to 5 fps: **140 frames across 6 scenes**.

| scene | frame range | @5fps | glyphs |
|---|---|---|---|
| scene_01 | f0001–f0086 | 18 | `I` |
| scene_02 | f0102–f0207 | 22 | `F` bending into the father's back |
| scene_03 | f0224–f0354 | 27 | `I` + bent figure with cane |
| scene_04 | f0368–f0523 | 32 | `M` |
| scene_05 | f0537–f0667 | 27 | `M` + `I` under the umbrella |
| scene_06 | f0682–f0747 | 14 | `FAMILY` assembling |

```
scenes/scene_01/f0001.png …   frames, split by scene
review/scene_01.jpg …         ONE contact sheet per scene
frames_manifest.csv           frame_id, file, scene, timestamp, source_frame
```

Frame ids are **global** and skip the separators, so f0090–f0101 simply don't
exist. That keeps ids unique once step 2 regroups cutouts into `by_letter/`,
where per-scene numbering would collide.

Useful flags:

```bash
--scenes 3 4        # redo only these scenes
--no-extract        # rebuild sheets/manifest from existing frames
--mode scene        # ignore separators, cut on content change instead
--mode manual --cuts 90 212 356    # cut at these frames yourself
--fmt jpg           # ~8x smaller, much faster to write
```

### Two things worth knowing

**White separators are Y=235, not 255.** Video is encoded limited-range, so
"pure white" measures 235. The detector threshold is 225, which cleared your
separators with margin — content in this spot tops out at 197. If you ever
export full-range footage, raise `--white-yavg`.

**This folder is on your Desktop, which iCloud syncs.** An earlier run left
253 files named like `f0459 3.png` — iCloud conflict copies, created because
frames were being written and deleted faster than sync could keep up. They're
cleaned up now, and each scene is extracted in its own short ffmpeg call which
avoids the churn. But ~1.4 GB of PNGs in a synced folder will upload. If that's
unwanted, move the project somewhere outside iCloud, or use `--fmt jpg`.

---

## Step 1b — un-split video  (`01_extract_frames.sh`)

```bash
brew install ffmpeg          # if you don't have it
./01_extract_frames.sh
```

Drop the edited video in this folder (the script finds the only video here,
or pass `VIDEO=edited.mp4`) and it extracts **every frame** at the file's own
frame rate — no fps filter, so 1:1 with no resampling, no dropped frames and
no duplicates. Verified against the original: ffprobe reports 2250 frames and
the script writes exactly 2250.

```bash
SHEETS=0 ./01_extract_frames.sh        # skip contact sheets, just frames
FPS=5 ./01_extract_frames.sh           # subsample instead, if you ever want to
FMT=jpg ./01_extract_frames.sh         # smaller files
OUT=take2 ./01_extract_frames.sh       # second cut, separate folder + manifest
```

Produces:

| path | what |
|---|---|
| `frames/f0001.png …` | full-resolution frames, 1:1 with the edit |
| `frames_manifest.csv` | `frame_id, file, timestamp_sec, source_frame` |
| `review/sheet_001.jpg …` | 6×5 contact sheets, frame id burned into each tile |

The manifest is named after the output folder, so `OUT=take2` writes
`take2_manifest.csv` and won't clobber an earlier run.

**Export settings that matter:** render your cut at the *same* frame rate as
the source (25 fps). Exporting 25 fps material to 30 fps makes the encoder
duplicate frames, and you'd get near-identical neighbours in the frame set for
no benefit. A high-bitrate or lossless export also keeps compression mush off
the letter edges, which is what the segmentation keys on.

At 1080p PNG, budget roughly **2 MB per frame** — a 60-second cut at 25 fps is
about 3 GB. `FMT=jpg` cuts that by ~8× if disk gets tight.

---

## Step 2 — letter segmentation

Run **all scenes in one pass** — a separate run per scene would rewrite the
manifest each time, leaving only the last scene's rows.

```bash
python3 -m pip install opencv-python numpy

python3 02_segment_letters.py \
  --frames-dir scenes/scene_01 scenes/scene_02 scenes/scene_03 \
               scenes/scene_04 scenes/scene_05 scenes/scene_06 \
  --all --word "" \
  --scene-words scene_01=I scene_02=F scene_04=M scene_05=MI scene_06=FAMILY
```

Current result: **271 components from 140 frames** — 197 labelled `ok`,
49 `unlabelled` (scene_03, deliberately left positional), 25 flagged across
just 7 transition frames.

| scene | components | frames | note |
|---|---|---|---|
| scene_01 | 18 | 18 | exactly one `I` per frame |
| scene_02 | 27 | 22 | +5 on 4 dissolve frames |
| scene_03 | 49 | 27 | `I` + figure, positional names |
| scene_04 | 41 | 32 | +9 on 3 dissolve frames |
| scene_05 | 54 | 27 | exactly `M`+`I` per frame |
| scene_06 | 82 | 14 | 6 per frame once all letters are in |

The flagged frames are f0102, f0197–f0207, f0368–f0378 — cross-dissolves at
scene starts where the letters aren't formed yet. They're marked `mismatch` in
the manifest, so they're easy to drop. f0682 is flagged too but is correct:
only F, A, M, I have appeared at that point.

<details><summary>Old per-scene form (don't use — overwrites the manifest)</summary>

```bash
python3 02_segment_letters.py --frames-dir scenes/scene_01 --all
```
</details>

# a subset while you're tuning thresholds
python3 02_segment_letters.py --frames-dir scenes/scene_01 --frames f0001 f0045
```

Tune on a handful of frames first, check `overlay/`, then run `--all`.

Each scene here holds one or two glyphs rather than the whole word, so pass
`--word` per scene (or use `words.csv`). From the overview strip:

| scene | glyphs |
|---|---|
| scene_01 | `I` |
| scene_02 | `F` bending into the father's hunched back |
| scene_03 | `A` / `IA` in the city, father with a cane |
| scene_04 | `M` |
| scene_05 | `MI` under the red umbrella |
| scene_06 | `FAMILY` assembling |

```bash
python3 02_segment_letters.py --frames-dir scenes/scene_01 --all --word I
python3 02_segment_letters.py --frames-dir scenes/scene_06 --all --word FAMILY
```

### Output layout

```
letters/
├── by_frame/
│   └── f0092/
│       ├── f0092_F.png          RGBA cutout, tight crop, transparent bg
│       ├── f0092_F_mask.png     binary mask on the full 1920×1080 canvas
│       └── f0092_A.png …
├── by_letter/
│   ├── F/  f0092_F.png  f0117_F.png  f0400_F.png …
│   ├── A/  …
│   └── …                        ← this is your optimization input
├── overlay/
│   └── f0092.jpg                QC render: boxes + labels. Check these first.
└── letters_manifest.csv         frame, letter, bbox, area, thresholds used
```

`by_letter/` is the grouping for step 3: one folder per glyph, every frame's
instance of that glyph inside it. Send a whole folder (or a slice of one) to
the optimizer. `letters_manifest.csv` carries each cutout's `x,y,w,h`, which is
what puts the optimized result back on the 1920×1080 canvas.

### How it works, and why there's no GPU in the default path

The glyphs are high-saturation yellow/gold; the backgrounds are desaturated
warm beige. Measured on this footage, glyph pixels sit at S ≈ 195–255 and the
background median is S ≈ 130, so a saturation floor at **165** splits them with
margin on both sides. Two refinements were needed:

- **Warm-channel test** (`--warm-tol 25`) — keeps pixels where `R ≥ G − 25`.
  The park scenes have yellow-green trees that pass the hue window; foliage is
  green-dominant (G > R) and glyphs are not, so this drops all five trees while
  leaving the letters untouched. At tolerance 0 it starts eating holes in the
  glyphs — 25 is the measured sweet spot.
- **Auto-relax ladder** — the closing frames fade the word almost into the
  background and return nothing at S=165. The script walks
  S 165 → 130 → 110 → 95 and stops at the strictest rung that finds anything,
  so clean frames never get a looser threshold than they need. The rung
  actually used is recorded per row in the manifest.

The channel bug at top-right overlaps the letters' vertical band, so it's
removed as a corner box (`--exclude 0.85,0.0,1.0,0.16`) rather than a
horizontal crop. Verified over 60 random frames that letters never enter it.

On the clean studio frames this lands the glyphs pixel-tight — better than SAM
will give you, and it runs at video speed on a laptop.

### Labelling

Blobs are ordered left-to-right by **bounding-box left edge**, not centroid:
the F carries a long bar over the whole word, so its centroid lands mid-frame
while its reading order is first.

Frames don't all show the whole word — there are single-letter shots, and
letters that have morphed into figures. When the blob count doesn't match
`--word` (default `FAMILY`), nothing is dropped: blobs fall back to `c00, c01…`
and the frame is listed at the end of the run. Fix those in a `words.csv`:

```csv
frame_id,word
f0260,M
f0330,IA
f0361,MI
```

```bash
python3 02_segment_letters.py --all --words words.csv
```

Repeated letters in one frame get suffixed — `L`, `L2`.

### Optional SAM2 pass

Worth it only for the frames where letters overlap busy background or have
morphed into figures. The HSV blob boxes become SAM's box prompts.

```bash
pip install 'git+https://github.com/facebookresearch/sam2.git'

python3 02_segment_letters.py --frames-file hard_frames.txt --sam2 \
    --sam2-checkpoint checkpoints/sam2.1_hiera_large.pt \
    --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml
```

SAM's mask is accepted only if it still covers >60% of the colour blob and
hasn't grown past 3×, otherwise the HSV mask is kept — that guard stops SAM
latching onto the background. Refined rows are flagged `refined=1` in the
manifest and marked `*` in the overlay.

If you're on SAM3, check its current predictor API against the `Sam2Refiner`
class — the box-prompt call there is written against SAM2's
`SAM2ImagePredictor` and may need a small adapter.

### Tuning

Every threshold is a flag: `--s-lo`, `--v-lo`, `--warm-tol`, `--min-area`,
`--merge-area`, `--close-px`, `--open-px`, `--bottom-crop`, `--exclude`.
Run on 3–4 frames, look at `overlay/`, adjust, then commit to the full set.

The defaults were measured on the *original* encode. If your edit re-grades,
re-encodes hard, or changes resolution, the numbers may drift — run a few
frames and check `overlay/` before trusting `--all`. The `--exclude` logo box
is normalised, so it survives a resolution change, and is harmless if your cut
crops the channel bug out.

Three defaults were tuned against the trimmed cut specifically, and the
reasoning matters if you re-tune:

- **`--h-lo 18`** — the room set has a reddish-brown window frame and door
  outline that used to come through as extra components. Across 405 measured
  components, glyphs sit at hue 20–32 and those artifacts at 10–12, with
  nothing in between. This was the single biggest fix: it took scene_01 from
  2 components per frame down to exactly 1.
- **`--min-area 3200`** — kills the leftover specks, which top out near
  3.1k px. Do **not** raise it: in scene_06 the whole word is laid out small
  and its smallest glyph is only ~3.8k px, so area alone cannot separate
  glyphs from junk. Hue does that; area only cleans up.
- **`--max-bbox-frac 0.45`** — on the cross-dissolves between clips the whole
  washed-out set passes the colour test as one blob spanning 63–70% of the
  frame. The widest real glyph is the F whose bar runs over the whole word,
  at 37%, so 0.45 clears both.

Common ones:

- letters coming out fragmented → raise `--close-px`
- picking up background → raise `--s-lo`, or add an `--exclude` box
- accent marks landing as their own blob → raise `--merge-area`
- small decorations counted as letters → raise `--min-area`

---

## Step 3 — optimization (later)

`by_letter/<glyph>/` is a ready batch directory. `letters_manifest.csv` has
`frame_id`, `letter`, `x`, `y`, `w`, `h` per cutout — join it against
`frames_manifest.csv` on `frame_id` to recover the source timestamp, then place
optimized results back at `x,y` on a 1920×1080 canvas and re-encode at the
original 25 fps.
