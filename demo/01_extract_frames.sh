#!/usr/bin/env bash
# 01_extract_frames.sh — disassemble the demo video into a folder of frames.
#
#   ./01_extract_frames.sh                     # EVERY frame, native fps, PNG
#   FPS=5 ./01_extract_frames.sh               # subsample to 5 fps
#   FPS=2 FMT=jpg ./01_extract_frames.sh       # quick, small
#   SHEETS=0 ./01_extract_frames.sh            # skip the contact sheets
#   VIDEO="edited.mp4" ./01_extract_frames.sh
#
# Default is native: no fps filter at all, so every decoded frame comes out
# 1:1 with no resampling, no dropped frames and no duplicates. That's what
# you want when the edit itself is the selection.
#
# Outputs (next to this script):
#   frames/f0001.png ...        full-resolution frames
#   frames_manifest.csv         frame -> source frame index -> timestamp
#                               (named after $OUT, so OUT=t5 -> t5_manifest.csv)
#   review/sheet_01.jpg ...     numbered contact sheets (optional)
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

FPS="${FPS:-native}"
SHEETS="${SHEETS:-1}"
FMT="${FMT:-png}"
OUT="${OUT:-frames}"
TILE_COLS="${TILE_COLS:-6}"
TILE_ROWS="${TILE_ROWS:-5}"
MANIFEST="${OUT}_manifest.csv"   # must come after OUT's default is applied

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install it with:  brew install ffmpeg" >&2
  exit 1
fi

# Pick the video: $VIDEO if set, else the only mp4/mov/mkv in this folder.
if [[ -n "${VIDEO:-}" ]]; then
  SRC="$VIDEO"
else
  # NB: plain loop, not mapfile — macOS ships bash 3.2, which has no mapfile.
  CANDIDATES=()
  while IFS= read -r line; do
    CANDIDATES+=("$line")
  done < <(find . -maxdepth 1 -type f \
    \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.m4v' \) | sort)
  if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
    echo "No video found in $HERE. Set VIDEO=path/to/file.mp4" >&2; exit 1
  fi
  if [[ ${#CANDIDATES[@]} -gt 1 ]]; then
    echo "Multiple videos found; pick one with VIDEO=..." >&2
    printf '  %s\n' "${CANDIDATES[@]}" >&2; exit 1
  fi
  SRC="${CANDIDATES[0]}"
fi

echo "source : $SRC"

SRC_FPS_RAW="$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
               -of csv=p=0 "$SRC")"
SRC_FPS="$(awk -F/ '{ printf "%.6f", ($2 ? $1/$2 : $1) }' <<<"$SRC_FPS_RAW")"
DUR="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$SRC")"

NATIVE=0
case "$FPS" in
  native|all|0|"") NATIVE=1 ;;
esac

if [[ $NATIVE -eq 1 ]]; then
  echo "source fps: $SRC_FPS   duration: ${DUR}s   extracting: EVERY frame (1:1)"
else
  echo "source fps: $SRC_FPS   duration: ${DUR}s   subsampling to: ${FPS} fps"
fi

rm -rf "$OUT" review
mkdir -p "$OUT" review

# --- extract -------------------------------------------------------------
# Native mode uses no fps filter at all. -fps_mode passthrough keeps the
# decoded frames exactly as they are: no resampling, no dupes, no drops.
FILTER=()
[[ $NATIVE -eq 0 ]] && FILTER=(-vf "fps=${FPS}")

# -fps_mode replaced -vsync in ffmpeg 5; fall back for older builds.
if ffmpeg -hide_banner -h full 2>/dev/null | grep -q fps_mode; then
  SYNC=(-fps_mode passthrough)
else
  SYNC=(-vsync 0)
fi

QUALITY=()
[[ "$FMT" != "png" ]] && QUALITY=(-q:v 2)

ffmpeg -hide_banner -loglevel error -stats -i "$SRC" \
  "${FILTER[@]}" "${SYNC[@]}" "${QUALITY[@]}" "$OUT/f%04d.${FMT}"

COUNT="$(find "$OUT" -type f -name "f*.${FMT}" | wc -l | tr -d ' ')"
echo "extracted: $COUNT frames -> $OUT/"

if [[ $COUNT -gt 9999 ]]; then
  echo "note: past 9999 the ids grow to 5 digits (f10000), so plain"
  echo "      alphabetical sorting interleaves. Sort numerically if it matters."
fi

# --- manifest: map each frame back to the source timeline ----------------
# Native : output frame N (1-based) IS source frame N-1, t = (N-1)/SRC_FPS.
# Sampled: output frame N sits at t = (N-1)/FPS, i.e. source round(t*SRC_FPS).
{
  echo "frame_id,file,timestamp_sec,source_frame"
  i=1
  while [[ $i -le $COUNT ]]; do
    printf 'f%04d,%s/f%04d.%s,' "$i" "$OUT" "$i" "$FMT"
    awk -v n="$i" -v f="$FPS" -v s="$SRC_FPS" -v nat="$NATIVE" \
      'BEGIN {
         if (nat) { printf "%.3f,%d\n", (n-1)/s, n-1 }
         else     { t=(n-1)/f; printf "%.3f,%d\n", t, int(t*s + 0.5) }
       }'
    i=$((i+1))
  done
} > "$MANIFEST"
echo "manifest : $MANIFEST"

# --- contact sheets, numbered so you can call out frame ids --------------
if [[ "$SHEETS" == "0" ]]; then
  echo "contact sheets: skipped (SHEETS=0)"
  N_SHEETS=0
else
PER_SHEET=$((TILE_COLS * TILE_ROWS))
N_SHEETS=$(( (COUNT + PER_SHEET - 1) / PER_SHEET ))
echo "building $N_SHEETS contact sheet(s) of ${TILE_COLS}x${TILE_ROWS}..."

for ((s=0; s<N_SHEETS; s++)); do
  start=$((s * PER_SHEET + 1))
  n=$(printf '%03d' $((s+1)))
  ffmpeg -hide_banner -loglevel error \
    -start_number "$start" -i "$OUT/f%04d.${FMT}" \
    -frames:v 1 \
    -vf "drawtext=text='%{eif\:n+${start}\:d}':x=8:y=8:fontsize=42:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=6,scale=360:-1,tile=${TILE_COLS}x${TILE_ROWS}:padding=4:margin=4:color=0x111111" \
    -q:v 3 -y "review/sheet_${n}.jpg" 2>/dev/null || true
done
fi

echo
echo "done."
echo "  frames   : $OUT/   ($COUNT files)"
[[ "$N_SHEETS" != "0" ]] && echo "  sheets   : review/   ($N_SHEETS, tile number = frame id)"
echo "  manifest : $MANIFEST"
echo
echo "Next:  python3 02_segment_letters.py --all"
