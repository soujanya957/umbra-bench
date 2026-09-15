#!/usr/bin/env python3
"""User-study example figure for one shadow item.

Left panel  : how the question was asked (prompt + stimulus image + colored options).
Right panel : a single confidence violin whose dots are colored by the label each
              participant chose -- so "what people picked" and "how sure they were"
              live in one plot.

    python scripts/make_user_study_fig.py
        -> figures/fig6_user_study_<target>.pdf  and  .png

EVERYTHING you'd normally want to change is in the CONFIG block below, and each knob
is commented. Point TARGET at any item in the study workbook and the script pulls
that item's responses, options and correct answer automatically; drop in a different
IMG_REL and you have the figure for another target.
"""
import os, warnings
warnings.filterwarnings("ignore")            # silence openpyxl pivot-cache chatter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from openpyxl import load_workbook

# ============================== CONFIG ===============================
# ---- which study item to plot ----------------------------------------------
TARGET   = "Crab"          # correct label of the item, exactly as in the workbook.
                           # Change to plot another item, e.g. "Snake", "Whale", "Hand".
IMG_REL  = "Teleops/teleop_set2/tag_rectified/teleop_crab_01_rectified.png"
                           # stimulus image, path RELATIVE TO THE REPO ROOT.
QUESTION = "Q. Which target does this\nshadow most closely\nresemble?"
                           # prompt shown top-left; "\n" starts a new line.

# ---- data source ------------------------------------------------------------
XLSX_REL   = "results/google_form/shadow_recognition_study_responses.xlsx"
RESP_SHEET = "Form Responses 1"     # one participant per row; row 2 holds the answer key
OPT_SHEET  = "Full semantic table"  # has clean option1..option5 columns keyed by "GT label"
# Auto-detection finds the item's columns by matching TARGET against the answer-key
# row. If it ever guesses wrong, hard-set these (1-based Excel column numbers) and
# they win; leave as None to auto-detect.
CHOICE_COL = None                 # column with each participant's picked label
CONF_COL   = None                 # column with each participant's confidence rating
# Options are pulled from OPT_SHEET; set a list here to override (order = shown order).
OPTIONS_OVERRIDE = None           # e.g. ["Goat", "Cat", "Rabbit", "Moose", "Crab"]

# ---- colors: one per option -------------------------------------------------
# Options named in OPT_COLORS use those exact colors; every other option is given a
# leftover PALETTE color, guaranteed not to repeat one. These hexes are the repo's
# validated categorical palette (blue, orange, aqua, yellow, magenta...).
# Tip: give the options that actually got votes the first three (blue/orange/aqua) --
# they are the most colorblind-separable trio.
PALETTE    = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"]
OPT_COLORS = {
    "Crab":  "#2a78d6",   # blue
    "Cat":   "#eb6834",   # orange
    "Moose": "#1baf7a",   # aqua
}

# ---- confidence scale -------------------------------------------------------
CONF_MIN, CONF_MAX = 1, 5         # rating range shown on the x-axis
SCALE_LOW  = "Not confident"      # caption under the low end of the scale
SCALE_HIGH = "Very confident"     # caption under the high end

# ---- text on the figure -----------------------------------------------------
# TITLE_LEFT  = "How the question was asked"
TITLE_RIGHT = f"Reported confidence ({CONF_MIN}–{CONF_MAX})"
# IMG_CAPTION = "robot-cast shadow  ·  target: {target}"   # {target} auto-filled

# ---- layout & type sizes (tweak freely) -------------------------------------
# Sized to print at IEEE/ICRA's actual single-column width (3.5in max) -- these
# inches ARE the final print inches, so every fontsize below is a literal point
# size on the page, not something that gets rescaled later. Keep FIG_W <= 3.5.
FIG_W, FIG_H = 3.4, 2.7625        # figure size in inches (true print size);
                                   # aspect matches the requested 800x650 layout
                                   # while keeping width at the 3.5in column cap
DPI          = 600                # PNG resolution; figure is physically small, so
                                   # a high DPI keeps small text crisp (PDF is vector, unaffected)
VIOLIN_FILL  = "#efeee9"          # violin body color; keep neutral so dots pop
POINT_SIZE   = 30                 # dot area of each beeswarm point
DOT_STEP     = 0.15               # vertical gap between dots stacked at one rating
SHOW_MEAN    = True               # draw the mean vertical line + "mean X.X" label
FS_Q     = 11     # section headers (Q. prompt, violin panel title), bold
FS_EMPH  = 8.5    # emphasized inline stats: "mean X.X", "N = ..." , bold
FS_CHIP  = 8      # axis ticks -- the IEEE floor for any figure text (do not
                   # go smaller than this)
FS_TICK  = FS_CHIP  # axis tick numbers
FS_CAPTION = 8  # "Not confident"/"Very confident" -- at the IEEE 8pt floor;
                   # separate from FS_CHIP so it can be tuned independently
FS_LEG   = 9.5    # legend rows: option key + option/confidence list

# ---- output -----------------------------------------------------------------
OUT_REL  = "figures/fig6_user_study_{target}"   # {target} -> lowercased target name
FORMATS  = ["pdf", "png"]         # which files to write
# ============================ end CONFIG =============================

# ---- fixed ink / chrome colors (repo palette) -------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE, "svg.fonttype": "none",
})

# ---- resolve repo-relative paths --------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                    # scripts/ -> repo root
IMG_PATH  = os.path.join(REPO, IMG_REL)
XLSX_PATH = os.path.join(REPO, XLSX_REL)


def _norm(v):
    """Normalize a cell to a comparable string ('4.0' -> '4', trims, casefolds)."""
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip().casefold()


def load_item():
    """Return (pairs, options, correct) for TARGET, read from the workbook.

    pairs   : list of (chosen_label, confidence) -- one per participant
    options : the labels shown, in the order presented
    correct : the correct label (== TARGET, using the workbook's exact spelling)
    """
    wb = load_workbook(XLSX_PATH, data_only=True)
    ws = wb[RESP_SHEET]
    tgt = _norm(TARGET)

    # locate the choice/confidence columns ------------------------------------
    ch_col, cf_col, correct = CHOICE_COL, CONF_COL, TARGET
    if ch_col is None or cf_col is None:
        # Row 1 = questions, row 2 = answer key. Match the item by its answer.
        for c in range(1, ws.max_column + 1):
            head = str(ws.cell(1, c).value or "")
            if "which target" in head.casefold() and _norm(ws.cell(2, c).value) == tgt:
                ch_col, cf_col = c, c + 1
                correct = str(ws.cell(2, c).value).strip()
                break
    if ch_col is None:
        raise SystemExit(f"Could not find item '{TARGET}' in '{RESP_SHEET}'. "
                         f"Set CHOICE_COL / CONF_COL by hand in CONFIG.")

    # collect one (label, confidence) per participant (data starts at row 4) ---
    pairs = []
    for r in range(4, ws.max_row + 1):
        lab, conf = ws.cell(r, ch_col).value, ws.cell(r, cf_col).value
        if lab is None or conf is None:
            continue
        try:
            pairs.append((str(lab).strip(), float(conf)))
        except (TypeError, ValueError):
            pass

    # options shown, in order -------------------------------------------------
    options = OPTIONS_OVERRIDE
    if options is None:
        options = _read_options(wb, tgt)
    if not options:
        # last resort: whatever labels appeared, correct first
        seen = list(dict.fromkeys(p[0] for p in pairs))
        options = ([correct] if correct in seen else []) + [s for s in seen if s != correct]

    # match the option list's casing to the answer key (Crab, not crab), but leave
    # single letters / digits untouched; then adopt the answer's spelling as correct.
    options = _case_like(correct, options)
    for o in options:
        if _norm(o) == tgt:
            correct = o
            break
    return pairs, options, correct


def _read_options(wb, tgt):
    """Pull the option labels for TARGET, in shown order.

    Tries OPT_SHEET first, then a couple of known fallbacks. Handles two shapes:
    a sheet with option1..optionN columns (one label per column), or a sheet with a
    single 'A|B|C' options cell. Robust to header/data column drift by scanning the
    matched row for the pipe-delimited cell when needed.
    """
    for sh in [OPT_SHEET, "Full semantic table", "Clip response"]:
        if sh not in wb.sheetnames:
            continue
        ow = wb[sh]
        hdr = {str(ow.cell(1, c).value or "").strip().casefold(): c
               for c in range(1, ow.max_column + 1)}
        key_col = hdr.get("gt label") or hdr.get("answer")
        opt_cols = [c for h, c in sorted(hdr.items()) if h.startswith("option")]
        if not key_col:
            continue
        for r in range(2, ow.max_row + 1):
            if _norm(ow.cell(r, key_col).value) != tgt:
                continue
            if opt_cols:                                  # one label per column
                opts = [str(ow.cell(r, c).value).strip() for c in opt_cols
                        if ow.cell(r, c).value not in (None, "")]
            else:                                         # single 'A|B|C' cell
                opts = []
                for c in range(1, ow.max_column + 1):
                    v = ow.cell(r, c).value
                    if isinstance(v, str) and "|" in v:
                        opts = [x.strip() for x in v.split("|") if x.strip()]
                        break
            if opts:
                return opts
    return None


def _case_like(correct, opts):
    """Title-case multi-letter word options to match a Title-cased answer; letters
    and digits are left exactly as stored."""
    titleish = len(correct) > 1 and correct.isalpha() and correct == correct.title()
    if not titleish:
        return opts
    return [o.title() if isinstance(o, str) and o.isalpha() and len(o) > 1 else o
            for o in opts]


def assign_colors(options):
    """Map each option to a distinct color: OPT_COLORS wins, the rest draw from the
    PALETTE slots not already taken (so nothing collides)."""
    used = {OPT_COLORS[o] for o in options if o in OPT_COLORS}
    spare = [c for c in PALETTE if c not in used]
    colors, k = {}, 0
    for o in options:
        if o in OPT_COLORS:
            colors[o] = OPT_COLORS[o]
        else:
            colors[o] = spare[k % len(spare)] if spare else PALETTE[k % len(PALETTE)]
            k += 1
    return colors


def round_dot(ax, xy, r, **kwargs):
    """An Ellipse that renders as a true circle of x-radius `r` (in `ax`'s data
    coords) regardless of the axes box's aspect ratio -- a plain plt.Circle
    would come out stretched into an ellipse whenever the box isn't square."""
    box = ax.get_position()
    ry = r * (box.width * FIG_W) / (box.height * FIG_H)
    return Ellipse(xy, width=2 * r, height=2 * ry, **kwargs)


def flow_row(ax, items, colors_map, fontsize, dot_r=0.022, gap=0.03, x0=0.015, y=0.5,
             max_x=0.99):
    """Lay dot+label pairs left to right, each positioned from the PREVIOUS
    label's actual rendered width (measured via the real renderer) rather than
    assuming equal-width slots -- equal spacing overlaps as soon as the font
    size or a label's length changes, since text width isn't constant. If the
    row still doesn't fit at the requested size, shrink font+dots until it does."""
    renderer = ax.figure.canvas.get_renderer()

    def layout(fs, dr):
        x, artists = x0, []
        for lab in items:
            dot = round_dot(ax, (x + dr, y), dr, color=colors_map[lab], zorder=5)
            ax.add_patch(dot)
            t = ax.text(x + 2 * dr + 0.012, y, lab, va="center", ha="left",
                        fontsize=fs, color=INK)
            artists.append((dot, t))
            bbox = t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
            x1d = bbox.x1
            x = x1d + gap
        return x - gap, artists

    total, artists = layout(fontsize, dot_r)
    if total > max_x:                              # doesn't fit -- shrink and redo
        for dot, t in artists:
            dot.remove(); t.remove()
        layout(fontsize * max_x / total, dot_r * max_x / total)


def flow_grid(ax, rows, colors_map, fontsize, dot_r=0.022, col_gap=0.06, row_ys=(0.75, 0.25),
              x0=0.015, max_x=0.99):
    """Like flow_row, but for several rows that should align into columns --
    each column gets ONE width (the widest label anywhere in that column
    across all rows), so a dot in row 2 sits directly under the dot above it
    instead of each row flowing independently at its own pace."""
    renderer = ax.figure.canvas.get_renderer()
    n_cols = max(len(r) for r in rows)

    def col_widths(fs):
        widths = [0.0] * n_cols
        for row in rows:
            for c, lab in enumerate(row):
                t = ax.text(0, 0, lab, fontsize=fs)
                bbox = t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
                widths[c] = max(widths[c], bbox.x1 - bbox.x0)
                t.remove()
        return widths

    def layout(fs, dr):
        widths = col_widths(fs)
        col_x, x = [], x0
        for w in widths:
            col_x.append(x)
            x += 2 * dr + 0.012 + w + col_gap
        total = x - col_gap
        artists = []
        for row, y in zip(rows, row_ys):
            for c, lab in enumerate(row):
                cx = col_x[c]
                dot = round_dot(ax, (cx + dr, y), dr, color=colors_map[lab], zorder=5)
                ax.add_patch(dot)
                t = ax.text(cx + 2 * dr + 0.012, y, lab, va="center", ha="left",
                            fontsize=fs, color=INK)
                artists.append((dot, t))
        return total, artists

    total, artists = layout(fontsize, dot_r)
    if total > max_x:                              # doesn't fit -- shrink and redo
        for dot, t in artists:
            dot.remove(); t.remove()
        layout(fontsize * max_x / total, dot_r * max_x / total)


def main():
    pairs, options, correct = load_item()
    colors = assign_colors(options)
    # canonical map so a pick like "crab"/"Crab" both resolve to the option "Crab"
    canon = {o.casefold(): o for o in options}
    picks = [(canon.get(p[0].casefold(), p[0]), p[1]) for p in pairs]
    counts = {o: sum(1 for p in picks if p[0] == o) for o in options}
    opt_conf = {o: (np.mean([p[1] for p in picks if p[0] == o]) if counts[o] else None)
                for o in options}
    conf = np.array([p[1] for p in picks], float)
    N, cmean = len(picks), conf.mean()

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # ---- 5x5 grid, per the requested layout -------------------------------
    #   cols: left = 2/5, right = 3/5     rows: top = 2/5, bottom = 3/5
    #     top-left (2x2)    : crab image        top-right (2x3)   : question + option key
    #     bottom-left (3x2) : option/confidence  bottom-right (3x3): violin graph
    margin, gutter = 0.025, 0.035
    x_left0,  x_left1  = margin, 0.4 - gutter / 2
    x_right0, x_right1 = 0.4 + gutter / 2, 1 - margin * 0.3
    y_bot0, y_bot1 = margin, 0.6 - gutter / 2          # bottom row (y grows upward)
    y_top0, y_top1 = 0.6 + gutter / 2, 1 - margin      # top row

    # ------------------------------------------- top-left : the crab image ---
    # sized to the image's own aspect ratio (622x512) so the declared box is
    # fully used -- a mismatched box gets silently shrunk-to-fit by imshow
    img_h_in = (y_top1 - y_top0) * FIG_H                       # height-constrained
    img_w_in = img_h_in * (622 / 512)
    ax_img = fig.add_axes([x_left0, y_top0, img_w_in / FIG_W, img_h_in / FIG_H])
    if os.path.exists(IMG_PATH):
        ax_img.imshow(mpimg.imread(IMG_PATH))
    else:
        ax_img.text(0.5, 0.5, f"image not found:\n{IMG_REL}", ha="center", va="center",
                    fontsize=FS_CHIP, color=MUTED, transform=ax_img.transAxes)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for s in ax_img.spines.values():
        s.set_edgecolor(BASELINE); s.set_linewidth(0.8)

    # -------------------------------- top-right : question + option key -----
    fig.text(x_right0 + 0.01, y_top1, QUESTION, fontsize=FS_Q, fontweight="bold",
             color=INK, va="top", ha="left", linespacing=1.25)

    n_lines = QUESTION.count("\n") + 1
    q_h_frac = (FS_Q * 1.25 * n_lines / 72) / FIG_H    # actual question block height
    op_top = y_top1 - q_h_frac - 0.02
    op_h = 0.12                        # two rows now, so taller than the old
                                        # single-row 0.08
    op_right = 1 - margin * 0.02       # pushed almost flush to the true right
                                        # edge (bbox_inches="tight" crops any
                                        # excess anyway), for maximum headroom
    ax_op = fig.add_axes([x_right0, op_top - op_h, op_right - x_right0, op_h])
    ax_op.axis("off"); ax_op.set_xlim(0, 1); ax_op.set_ylim(0, 1)
    # split across 2 rows so each has room to sit at FS_LEG without shrinking --
    # 5 items squeezed into 1 row forced a shrink well below the IEEE 8pt floor
    opts5 = options[:5]
    half = -(-len(opts5) // 2)                     # ceil: row 1 gets the extra item
    # flow_grid aligns both rows into shared columns (dots stack vertically)
    # with real gaps between columns, rather than each row flowing on its own.
    # max_x well under 1.0 -- the fit-check is measured at the interactive
    # renderer's resolution, which doesn't perfectly match the final DPI=600
    # save, so the last dot can clip even when the check says "fits"; this
    # margin of safety is what actually prevents that, not the box width above
    flow_grid(ax_op, [opts5[:half], opts5[half:]], colors, FS_LEG, max_x=0.90)

    # ------------------------------- bottom-left : option/confidence list ----
    ax_lg = fig.add_axes([x_left0, y_bot0, x_left1 - x_left0, y_bot1 - y_bot0])
    ax_lg.axis("off"); ax_lg.set_xlim(0, 1); ax_lg.set_ylim(0, 1)
    renderer_lg = fig.canvas.get_renderer()
    ranked = sorted(options[:5], key=lambda o: counts[o], reverse=True)
    stats = [f"n = {counts[lab]}" + (f" ({opt_conf[lab]:.1f})" if counts[lab] else "")
             for lab in ranked]
    row_top, row_h = 0.93, (0.93 - 0.07) / (len(ranked) - 1)

    def measure(strings, fs):
        widths = []
        for s in strings:
            t = ax_lg.text(0, 0, s, fontsize=fs)
            bbox = t.get_window_extent(renderer=renderer_lg).transformed(ax_lg.transAxes.inverted())
            widths.append(bbox.x1 - bbox.x0)
            t.remove()
        return widths

    def try_layout(fs):
        label_w = measure(ranked, fs)
        stat_x = 0.13 + max(label_w) + 0.05
        stat_w = measure(stats, fs)
        return stat_x, stat_x + max(stat_w)

    fs_lg = FS_LEG
    stat_x, right_edge = try_layout(fs_lg)
    if right_edge > 0.99:                          # doesn't fit -- shrink to fit,
        fs_lg = max(FS_CHIP, FS_LEG * 0.99 / right_edge)  # but never below the
        stat_x, right_edge = try_layout(fs_lg)            # IEEE 8pt floor
        stat_x, _ = try_layout(fs_lg)

    for i, lab in enumerate(ranked):
        yy = row_top - i * row_h
        ax_lg.add_patch(round_dot(ax_lg, (0.05, yy), 0.028, color=colors[lab], zorder=5))
        ax_lg.text(0.13, yy, lab, fontsize=fs_lg, va="center", color=INK)
        ax_lg.text(stat_x, yy, stats[i], fontsize=fs_lg, va="center", ha="left",
                   color=(INK if counts[lab] else MUTED))

    # ------------------------------------------ bottom-right : violin plot ---
    # auto-shrunk to fit the column -- a longer TITLE_RIGHT string would
    # otherwise spill past x_right0 into the bottom-left legend, as it did
    # when this was a fixed fontsize (twice, with two different title strings)
    title_t = fig.text((x_right0 + x_right1) / 2, y_bot1, TITLE_RIGHT, fontsize=FS_Q,
                        fontweight="bold", color=INK, va="top", ha="center")
    title_bbox = title_t.get_window_extent(renderer=fig.canvas.get_renderer())
    title_bbox = title_bbox.transformed(fig.transFigure.inverted())
    title_w = title_bbox.x1 - title_bbox.x0
    title_avail = (x_right1 - x_right0) - 0.02
    if title_w > title_avail:
        title_t.set_fontsize(FS_Q * title_avail / title_w)
    fig.canvas.draw()                              # force a fresh layout pass so
                                                    # the fontsize change actually
                                                    # takes effect before savefig
    title_h_frac = (FS_Q * 1.25 / 72 + 0.03) / FIG_H
    ax = fig.add_axes([x_right0, y_bot0, x_right1 - x_right0,
                        y_bot1 - title_h_frac - 0.015 - y_bot0])
    vp = ax.violinplot(conf, positions=[0], vert=False, widths=1.15, showextrema=False)
    for b in vp["bodies"]:
        b.set_facecolor(VIOLIN_FILL); b.set_edgecolor(BASELINE)
        b.set_linewidth(1.0); b.set_alpha(1)

    # beeswarm: at each rating, stack dots vertically, correct label in the center
    for xv in range(CONF_MIN, CONF_MAX + 1):
        grp = sorted([p[0] for p in picks if round(p[1]) == xv],
                     key=lambda o: (o != correct, o))         # correct first -> center
        seq = [0]
        while len(seq) < len(grp):                            # 0,+1,-1,+2,-2,...
            m = len(seq) // 2 + 1; seq += [m, -m]
        for lab, off in zip(grp, seq[:len(grp)]):
            ax.scatter(xv, off * DOT_STEP, s=POINT_SIZE, color=colors.get(lab, MUTED),
                       edgecolor="white", linewidth=0.6,
                       zorder=(4 if lab == correct else 6))

    if SHOW_MEAN:
        ax.plot([cmean, cmean], [-0.9, 0.9], color=INK, lw=1.2, zorder=8)
        ax.text(cmean, 1.02, f"mean {cmean:.1f}", ha="center", va="bottom",
                fontsize=FS_EMPH, color=INK, fontweight="bold")

    ax.set_xlim(CONF_MIN - 0.5, CONF_MAX + 0.5); ax.set_ylim(-1.8, 1.35)
    ax.set_yticks([]); ax.set_xticks(range(CONF_MIN, CONF_MAX + 1))
    ax.set_xticklabels([str(i) for i in range(CONF_MIN, CONF_MAX + 1)],
                       fontsize=FS_TICK, color=INK)
    ax.tick_params(length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_position(("data", -0.95)); ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(0.8)
    # anchored at the xlim extremes (not the 1/5 ticks) to maximize the gap
    # between the two captions, so a bigger FS_CAPTION still doesn't collide
    ax.text(CONF_MIN - 0.5, -1.55, SCALE_LOW, fontsize=FS_CAPTION, color=MUTED,
             ha="left", va="center")
    ax.text(CONF_MAX + 0.5, -1.55, SCALE_HIGH, fontsize=FS_CAPTION, color=MUTED,
             ha="right", va="center")

    # ------------------------------------------------------------ save --------
    stem = os.path.join(REPO, OUT_REL.format(target=correct.lower().replace(" ", "_")))
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for ext in FORMATS:
        fig.savefig(f"{stem}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    print(f"wrote {stem}.{{{','.join(FORMATS)}}}")
    print(f"  N={N}  mean_conf={cmean:.2f}  counts={counts}")


if __name__ == "__main__":
    main()
