"""Render every figure of the manuscript and the supplement.

One function per figure. Each reads the tidy file that `18_figure_data.py`
wrote into `results/figure_data/`, so a number drawn inside a figure is the
same number that populates the corresponding table. Nothing here is drawn by
hand and nothing is edited afterwards.

Output goes to `Folder Figures/` at the repository root, three files per
figure: a vector PDF, which is the master, a 600 dpi PNG, which is what the
Word build embeds, and a 600 dpi TIFF for raster-only submission systems.

Style follows `09_Figures_Guide.md`: one sans-serif family, ink colours for
text, series colour for marks only, horizontal grid behind the data, left and
bottom spines only, no title inside the image. The type scale is one step
above the guide's floor, because a 180 mm figure placed in a 152 mm Word text
column loses 15 per cent of its size and the guide's 7 pt tick would land
below its own 6 pt limit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "results" / "figure_data"
OUT = ROOT / "Folder Figures"

# ---------------------------------------------------------------- palette --
INK = "#1A1A1A"          # labels and titles
INK2 = "#5C5C5A"         # annotations, spines, reference lines
GRID = "#E5E5E4"         # grid, histogram fill
NOSKILL = "#9A9A98"      # no-skill baselines, non-data ink
BLUE = "#0072B2"         # series 1, DS2, this study's primary
ORANGE = "#D55E00"       # series 2, highlight
GREEN = "#009E73"        # series 3
PURPLE = "#A34E86"       # series 4
PALE = "#F2F7FB"         # data-store fill, shaded regions

SEQ = LinearSegmentedColormap.from_list("seq", ["#FFFFFF", BLUE, "#00344F"])

MM = 1.0 / 25.4
W1 = 88 * MM             # single column
W2 = 180 * MM            # double column

# Type scale, points at final size.
TICK, AXTITLE, PANEL, LEGEND, ANNOT, PTITLE = 8.0, 9.0, 10.0, 8.0, 7.5, 8.5

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": TICK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.labelsize": AXTITLE,
    "axes.titlesize": PTITLE,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.labelsize": TICK,
    "ytick.labelsize": TICK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "legend.fontsize": LEGEND,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,      # embed as TrueType, never outline
    "ps.fonttype": 42,
})

DASH = {"solid": (0, ()), "dashed": (0, (5, 2)), "dashdot": (0, (6, 2, 1.4, 2)),
        "dotted": (0, (1.6, 1.8))}

# A white halo lets a direct label sit over its own curve and stay readable.
HALO = [pe.withStroke(linewidth=2.4, foreground="white")]


def halo(t):
    t.set_path_effects(HALO)
    t.set_gid("haloed")
    return t


def hgrid(ax, axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)


def panel_letter(ax, letter: str, dx: float = -0.02, dy: float = 1.06) -> None:
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=PANEL, fontweight="bold", color=INK)


SAVED: list[str] = []
PROBLEMS: list[str] = []


def _bboxes(fig):
    """Every visible text run and its rendered rectangle, in display pixels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        out.append((t, bb))
    return out


def _densify(xy, step=2.0):
    """Resample a rendered polyline so short segments still hit a text box."""
    out = []
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(2, int(np.hypot(*(b - a)) / step))
        out.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.vstack(out) if out else xy


def check_overlaps(fig, stem: str, tol: float = 0.05) -> None:
    """Flag text that collides with other text, and text that leaves its box.

    Collisions are reported when the shared area exceeds `tol` of the smaller
    run's area, which lets kerning-level touching pass and catches anything a
    reader would see as overlapping.
    """
    items = _bboxes(fig)
    for i in range(len(items)):
        ti, bi = items[i]
        for j in range(i + 1, len(items)):
            tj, bj = items[j]
            ox = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            oy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if ox <= 0 or oy <= 0:
                continue
            area = ox * oy
            smaller = min(bi.width * bi.height, bj.width * bj.height)
            if area > tol * smaller:
                PROBLEMS.append(
                    f"{stem}: text collision {area / smaller:.0%} between "
                    f"{ti.get_text()[:34]!r} and {tj.get_text()[:34]!r}")

    # Text drawn on top of a plotted line. Anything given a white halo is
    # exempt, because a halo is the fix rather than the defect.
    for ax in fig.axes:
        for ln in ax.lines:
            if not ln.get_visible() or ln.get_linestyle() == "None":
                continue
            raw = ln.get_xydata()
            if len(raw) < 2:
                continue
            xy = _densify(ln.get_transform().transform(raw))
            for t, bb in items:
                if (t.axes is not ax or t.get_text().startswith("(")
                        or t.get_gid() == "haloed"):
                    continue
                inside = ((xy[:, 0] > bb.x0 + 1) & (xy[:, 0] < bb.x1 - 1)
                          & (xy[:, 1] > bb.y0 + 1) & (xy[:, 1] < bb.y1 - 1))
                if inside.any():
                    d = ln.get_xydata()
                    PROBLEMS.append(
                        f"{stem}: {t.get_text()[:34]!r} sits on a plotted line "
                        f"[{d[0][0]:.4g},{d[0][1]:.4g} -> {d[-1][0]:.4g},{d[-1][1]:.4g}]")
                    break

    # Text that spills out of the patch it is meant to sit inside. A label is
    # owned by the SMALLEST box that contains its centre, because the boxes
    # nest: a sub-box label also sits inside the group box around it.
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        patches = [(p, p.get_window_extent(renderer=r)) for p in ax.patches
                   if isinstance(p, (FancyBboxPatch, Rectangle))
                   and p.get_width() > 0 and p.get_edgecolor()[3] > 0]
        for t, bt in items:
            if t.axes is not ax:
                continue
            cx, cy = (bt.x0 + bt.x1) / 2, (bt.y0 + bt.y1) / 2
            owners = [bp for _, bp in patches
                      if bp.x0 < cx < bp.x1 and bp.y0 < cy < bp.y1]
            if not owners:
                continue
            bp = min(owners, key=lambda b: b.width * b.height)
            # demand about a millimetre of clearance on all four sides, so no
            # label merely touches the rule it sits inside
            dx = max(bp.x0 - bt.x0, bt.x1 - bp.x1) + 4.0
            dy = max(bp.y0 - bt.y0, bt.y1 - bp.y1) + 4.0
            if dx > 0.5 or dy > 0.5:
                PROBLEMS.append(
                    f"{stem}: {t.get_text()[:34]!r} overflows its box by "
                    f"{dx:.1f} px across, {dy:.1f} px down")


def save(fig, stem: str) -> None:
    OUT.mkdir(exist_ok=True)
    check_overlaps(fig, stem)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    try:
        fig.savefig(OUT / f"{stem}.tif", dpi=600, bbox_inches="tight",
                    pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    except Exception as e:                      # PIL missing or no TIFF support
        print(f"    {stem}: TIFF skipped, {type(e).__name__}: {e}")
    plt.close(fig)
    SAVED.append(stem)
    print(f"  {stem}")


# ============================================================== figure 1 ===
# A block diagram drawn in millimetres. The figure is 180 by 140 mm and the
# axes span exactly that, so every coordinate below is a millimetre on paper.

def _box(ax, cx, cy, w, h, text, *, fill="white", ec=INK2, lw=0.8, fs=7.0,
         dashed=False, rounded=0.9, weight="normal", tcolor=INK):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rounded}",
                       linewidth=lw, edgecolor=ec, facecolor=fill,
                       linestyle=DASH["dashed"] if dashed else "solid", zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tcolor,
            linespacing=1.28, zorder=3, fontweight=weight)


def _arrow(ax, pts, *, lw=0.8, color=INK2, dashed=False, head=True):
    pts = np.asarray(pts, dtype=float)
    for a, b in zip(pts[:-1], pts[1:-1] if len(pts) > 2 else []):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=1,
                linestyle=DASH["dashed"] if dashed else "solid",
                solid_capstyle="round")
    a, b = pts[-2], pts[-1]
    if head:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>",
                                     mutation_scale=6.5, linewidth=lw,
                                     color=color, shrinkA=0, shrinkB=0,
                                     zorder=1,
                                     linestyle=DASH["dashed"] if dashed else "solid"))
    else:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=1,
                linestyle=DASH["dashed"] if dashed else "solid")


def fig1_pipeline():
    W, H = 180.0, 142.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ---- column 1, preprocessing chain -----------------------------------
    c1x, c1w = 16.0, 28.0
    chain = [
        (128, "MIT-BIH Arrhythmia Database\n48 records, lead MLII", True, 6.0),
        (114, "exclude paced records\n102, 104, 107, 217", False, 6.5),
        (100, "band-pass 0.5 to 40 Hz\nzero-phase Butterworth", False, 6.5),
        (86, "resample to 250 Hz", False, 6.5),
        (72, "R-peak-centred window\n−250 ms to +400 ms", False, 6.5),
        (58, "map to AAMI\nN, S, V, F, Q", False, 6.5),
    ]
    for cy, txt, store, fs in chain:
        _box(ax, c1x, cy, c1w, 9.0, txt, fill=PALE if store else "white",
             rounded=2.6 if store else 0.9, fs=fs)
    for (y0, *_), (y1, *_) in zip(chain[:-1], chain[1:]):
        _arrow(ax, [(c1x, y0 - 4.5), (c1x, y1 + 4.5)])

    # ---- column 2, partitioned stores and the descriptors ----------------
    c2x, c2w = 51.0, 30.0
    _box(ax, c2x, 116, c2w, 12, "DS1 training subset\n17 records\n39,464 beats",
         fill=PALE, rounded=2.6)
    _box(ax, c2x, 86, c2w, 15,
         "four rhythm descriptors\npre-RR, post-RR,\nRR / local mean,\npre-RR / post-RR",
         fill="white")
    _box(ax, c2x, 62, c2w, 12, "DS1 validation subset\n5 records\n11,326 beats",
         fill=PALE, rounded=2.6)

    # split of the mapped beat stream into the two DS1 subsets
    _arrow(ax, [(c1x + 14, 58), (33, 58), (33, 116), (36, 116)])
    _arrow(ax, [(33, 58), (33, 62), (36, 62)])
    ax.add_patch(Circle((33, 58), 0.55, color=INK2, zorder=4))
    # descriptors are computed for the same beats
    _arrow(ax, [(c2x, 110), (c2x, 93.5)])

    # ---- column 3, the model and the two fitted decision parameters ------
    c3x, c3w = 89.0, 34.0
    # teacher, dashed because it exists only during training
    _box(ax, c3x, 134, 42, 11,
         "teacher network: HuBERT-ECG\n30,537,350 parameters\ndistillation only",
         dashed=True, fs=6.6)
    _arrow(ax, [(c2x, 122), (c2x, 134), (68, 134)], dashed=True)
    _arrow(ax, [(c3x, 128.5), (c3x, 125)], dashed=True)

    # the model block
    ax.add_patch(FancyBboxPatch((c3x - c3w / 2, 56), c3w, 69,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                linewidth=1.0, edgecolor=INK, facecolor="white",
                                zorder=2))
    ax.text(c3x, 121.5, "student classifier, arm E", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=INK, zorder=3)
    for k, cy in enumerate((115, 105.5, 96, 86.5), start=1):
        _box(ax, c3x, cy, 30.5, 8.2, f"depthwise-separable\nblock {k}", fs=7.0)
    for y0, y1 in ((115, 105.5), (105.5, 96), (96, 86.5)):
        _arrow(ax, [(c3x, y0 - 4.1), (c3x, y1 + 4.1)], lw=0.7)
    _box(ax, c3x, 77.5, 30.5, 8.2, "global average pooling", fs=7.0)
    _arrow(ax, [(c3x, 82.4), (c3x, 81.6)], lw=0.7)
    ax.add_patch(Circle((c3x, 70), 2.4, facecolor="white", edgecolor=INK2,
                        lw=0.8, zorder=3))
    ax.text(c3x, 70, "+", ha="center", va="center", fontsize=8, color=INK,
            zorder=4)
    _arrow(ax, [(c3x, 73.4), (c3x, 72.4)], lw=0.7)
    _box(ax, c3x, 61.5, 30.5, 8.2, "linear layer, 5 classes", fs=7.0)
    _arrow(ax, [(c3x, 67.6), (c3x, 65.6)], lw=0.7)

    # descriptors bypass the convolutional stack and join at the merge node
    _arrow(ax, [(66, 86), (69, 86), (69, 70), (c3x - 2.4, 70)], color=ORANGE)
    ax.text(67.0, 74.0, "bypasses the\nconvolutional stack", ha="right",
            va="center", fontsize=6.6, color=ORANGE, linespacing=1.25)

    # training beats enter the top of the stack
    _arrow(ax, [(66, 116), (72, 116)])

    # the two parameters fitted on the validation subset
    _box(ax, c3x, 46, c3w, 8, "temperature scaling\n$T$ = 0.700", fs=6.8)
    _box(ax, c3x, 32, c3w, 8, "conformal threshold\n90 per cent target", fs=6.8)
    _arrow(ax, [(c3x, 56), (c3x, 50)])
    _arrow(ax, [(c3x, 42), (c3x, 36)])
    _arrow(ax, [(c2x, 56), (c2x, 46), (72, 46)])
    _arrow(ax, [(69, 46), (69, 32), (72, 32)])
    ax.add_patch(Circle((69, 46), 0.55, color=INK2, zorder=4))

    # ---- the partition boundary ------------------------------------------
    ax.plot([130, 130], [4, 141], color=INK, lw=2.0, zorder=5,
            solid_capstyle="butt")
    ax.text(128.4, 96, "partition boundary", rotation=90, ha="center",
            va="center", fontsize=8.0, fontweight="bold", color=INK, zorder=6)

    # ---- right of the boundary, inference only ---------------------------
    rx = 159.5
    for cy, txt in ((128, "DS2\n22 records, 49,478 beats"),
                    (114, "INCART\n75 records, 175,100 beats"),
                    (100, "SVDB\n78 records, 183,768 beats")):
        _box(ax, rx, cy, 37, 11, txt, fill=PALE, rounded=2.6, fs=6.8)
        ax.plot([141, 136], [cy, cy], color=INK2, lw=0.8, zorder=1)
        ax.add_patch(Circle((136, cy), 0.55, color=INK2, zorder=4))
    ax.plot([136, 136], [128, 100], color=INK2, lw=0.8, zorder=1)
    _arrow(ax, [(136, 100), (136, 84)])

    ax.add_patch(FancyBboxPatch((134, 48), 44, 36,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                linewidth=1.0, edgecolor=INK, facecolor="white",
                                zorder=2))
    ax.text(156, 78, "inference", ha="center", va="center", fontsize=7.6,
            fontweight="bold", color=INK, zorder=3)
    ax.text(156, 68, "signal-quality gate,\nfrozen weights,\nfrozen temperature,\n"
                     "frozen threshold.\nNo parameter refitted.",
            ha="center", va="center", fontsize=6.6, color=INK2, linespacing=1.3,
            zorder=3)
    _box(ax, 156, 33.5, 44, 11, "class decision\nN, S, V, F, Q\nor abstain", fs=7.0)
    _arrow(ax, [(156, 48), (156, 39)])

    # ---- exactly three arrows cross the boundary -------------------------
    crossings = [
        ((106, 76), [(106, 76), (134, 76)], (120, 78.6), "trained\nweights"),
        ((106, 46), [(106, 46), (110, 46), (110, 64), (134, 64)], (120, 66.6),
         "calibration\ntemperature"),
        ((106, 32), [(106, 32), (112, 32), (112, 54), (134, 54)], (122, 56.6),
         "conformal\nthreshold"),
    ]
    for _, path, (lx, ly), label in crossings:
        _arrow(ax, path, lw=1.0, color=INK)
        ax.text(lx, ly, label, ha="center", va="bottom", fontsize=7.0,
                color=INK, linespacing=1.25)

    ax.text(130 - 2.0, 12, "no beat crosses this rule", rotation=90, ha="center",
            va="center", fontsize=7.0, color=ORANGE, zorder=6)

    # a legend for the two box kinds, bottom left, no frame
    lx, ly = 6.0, 26.0
    _box(ax, lx + 8, ly, 16, 5.5, "process", fs=6.6)
    _box(ax, lx + 8, ly - 8, 16, 5.5, "data store", fill=PALE, rounded=2.0, fs=6.6)
    _box(ax, lx + 8, ly - 16, 16, 5.5, "training only", dashed=True, fs=6.6)
    ax.plot([lx + 2, lx + 14], [ly - 23, ly - 23], color=ORANGE, lw=0.8)
    ax.text(lx + 16, ly - 23, "descriptor path", ha="left", va="center",
            fontsize=6.6, color=INK2)

    save(fig, "Figure_1_pipeline")


# ============================================================== figure 2 ===

def fig2_confusion():
    df = pd.read_csv(FD / "fig2_confusion.csv")
    order = ["N", "S", "V", "F", "Q"]
    sets = ["DS2", "INCART", "SVDB"]

    fig = plt.figure(figsize=(W2, 68 * MM))
    axes = [fig.add_axes([0.055 + i * 0.295, 0.14, 0.245, 0.72]) for i in range(3)]
    cax = fig.add_axes([0.925, 0.14, 0.016, 0.72])

    for k, (ax, ds) in enumerate(zip(axes, sets)):
        sub = df[df.dataset == ds]
        frac = np.zeros((5, 5))
        cnt = np.zeros((5, 5), dtype=int)
        for _, r in sub.iterrows():
            i, j = order.index(r.true_class), order.index(r.pred_class)
            frac[i, j] = r.row_fraction
            cnt[i, j] = int(r["count"])

        ax.imshow(np.where(cnt > 0, frac, np.nan), cmap=SEQ, vmin=0, vmax=1,
                  aspect="equal", interpolation="nearest")
        for i in range(5):
            for j in range(5):
                if cnt[i, j] == 0:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                           facecolor="white", edgecolor=GRID,
                                           lw=0.4, zorder=2))
                    continue
                col = "white" if frac[i, j] > 0.55 else INK
                ax.text(j, i - 0.15, f"{frac[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.5, color=col, zorder=3)
                ax.text(j, i + 0.24, f"({cnt[i, j]:,})", ha="center", va="center",
                        fontsize=6.0, color=col, zorder=3)

        # the S row, N column cell is the argument of the caption
        si, ni = order.index("S"), order.index("N")
        ax.add_patch(Rectangle((ni - .5, si - .5), 1, 1, fill=False,
                               edgecolor=ORANGE, lw=1.2, zorder=4))

        ax.set_xticks(range(5), order)
        ax.set_yticks(range(5), order)
        ax.set_xlabel("predicted class")
        if k == 0:
            ax.set_ylabel("true class")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        panel_letter(ax, "abc"[k], dx=-0.16, dy=1.02)
        ax.text(0.5, 1.02, ds, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=PTITLE, color=INK)

    # One leader, routed down a column boundary and along a row gap so that it
    # crosses no cell text.
    a0 = axes[0]
    a0.plot([0.5, 0.5, 1.45], [1.5, 2.55, 2.55], color=ORANGE, lw=0.7, zorder=5)
    a0.text(1.55, 2.55, "S read as N", ha="left", va="center", fontsize=ANNOT,
            color=ORANGE, zorder=5)

    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax, ticks=[0, .25, .5, .75, 1])
    cb.set_label("row-normalised proportion", fontsize=AXTITLE)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK2)
    cb.ax.tick_params(length=2.4, width=0.6, labelsize=TICK)

    save(fig, "Figure_2_confusion_matrices")


# ============================================================== figure 3 ===

SUPPORT = {"N": 44046, "S": 1830, "V": 3207, "F": 388, "Q": 7}


def fig3_precision_recall():
    df = pd.read_csv(FD / "fig3_precision_recall.csv")
    order = ["N", "S", "V", "F", "Q"]

    fig, axes = plt.subplots(1, 5, figsize=(W2, 58 * MM), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.75, bottom=0.19, wspace=0.24)

    for k, (ax, cls) in enumerate(zip(axes, order)):
        s = df[df.cls == cls].sort_values("recall")
        ax.plot(s.recall, s.precision, color=BLUE, lw=1.2, solid_joinstyle="round")
        base = float(s.baseline.iloc[0])
        ax.axhline(base, color=NOSKILL, lw=0.8, linestyle=DASH["dashed"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        hgrid(ax)
        ax.set_title(f"{cls}, n = {SUPPORT[cls]:,}\n"
                     f"AP {float(s.average_precision.iloc[0]):.3f}, "
                     f"baseline {base:.3f}", pad=7, linespacing=1.35)
        panel_letter(ax, "abcde"[k], dx=-0.05, dy=1.30)
        if k == 0:
            ax.text(0.04, base - 0.035, "no-skill", ha="left", va="top",
                    fontsize=ANNOT, color=INK2)

        if cls == "S":
            # the operating point of section 7.7, as reported in Table 3
            ax.plot([0.115], [0.252], marker="o", ms=4.0, color=ORANGE,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5)
            halo(ax.annotate("operating point\nSe 11.5 %\nPPV 25.2 %",
                             xy=(0.115, 0.252), xytext=(0.30, 0.60),
                             fontsize=6.8, color=ORANGE, ha="left", va="bottom",
                             linespacing=1.3,
                             arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.7,
                                             shrinkA=0, shrinkB=3)))
        if cls == "Q":
            ax.text(0.5, 0.52, "7 beats in DS2;\nnot interpretable",
                    ha="center", va="center", fontsize=ANNOT, color=INK2,
                    linespacing=1.3)

    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("precision")
    save(fig, "Figure_3_precision_recall")


# ============================================================== figure 4 ===

def fig4_reliability():
    df = pd.read_csv(FD / "fig4_reliability.csv")
    fig = plt.figure(figsize=(W1, 88 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.10,
                          left=0.17, right=0.975, top=0.98, bottom=0.10)
    ax, axh = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    style = {"before": (ORANGE, "before temperature scaling"),
             "after": (BLUE, "after temperature scaling")}
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, linestyle=DASH["dashed"],
            zorder=1, label="perfectly calibrated")
    for stage, (col, lab) in style.items():
        s = df[(df.stage == stage) & (df.n > 0)]
        ax.plot(s.confidence, s.accuracy, color=col, lw=1.2, marker="o", ms=3.4,
                markeredgecolor="white", markeredgewidth=0.4, label=lab, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("observed frequency")
    ax.tick_params(labelbottom=False)
    hgrid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 1.0), handlelength=2.0,
              borderaxespad=0.0, labelspacing=0.35)
    ax.text(0.97, 0.30, "ECE before 0.1159\nECE after 0.0552\n"
                        "fitted temperature 0.700",
            ha="right", va="bottom", fontsize=ANNOT, color=INK2, linespacing=1.35)

    before = df[df.stage == "before"]
    ax.text(0.97, 0.06, "Both curves sit above the diagonal:\nobserved accuracy exceeds "
            "stated\nconfidence, so the classifier is\nunder-confident, not over-confident.",
            ha="right", va="bottom", fontsize=ANNOT, color=ORANGE, linespacing=1.35)

    w = float(before.bin_hi.iloc[0] - before.bin_lo.iloc[0])
    centres = (before.bin_lo + before.bin_hi) / 2
    axh.bar(centres, before.n.clip(lower=0.6), width=w * 0.86, color=GRID,
            edgecolor=NOSKILL, linewidth=0.4, zorder=2)
    axh.set_yscale("log")
    axh.set_xlim(0, 1)
    axh.set_ylim(0.6, 3e4)
    axh.set_xticks(np.arange(0, 1.01, 0.2))
    axh.set_yticks([1, 100, 10000])
    axh.set_xlabel("mean predicted probability")
    axh.set_ylabel("beats per bin,\nlog scale", fontsize=AXTITLE)
    hgrid(axh)
    save(fig, "Figure_4_reliability")


# ============================================================== figure 5 ===

def fig5_coverage():
    df = pd.read_csv(FD / "fig5_coverage.csv")
    style = {"DS2": (BLUE, "solid"), "INCART": (ORANGE, "dashed"),
             "SVDB": (GREEN, "dashdot")}

    fig = plt.figure(figsize=(W1, 76 * MM))
    ax = fig.add_axes([0.175, 0.135, 0.775, 0.845])

    marks, ends = {}, {}
    for ds, (col, dash) in style.items():
        s = df[df.dataset == ds].sort_values("observed_coverage")
        ax.plot(s.observed_coverage, s.macro_f1, color=col, lw=1.2,
                linestyle=DASH[dash], zorder=3)
        op = df[(df.dataset == ds) & (df.target_coverage.round(4) == 0.90)].iloc[0]
        marks[ds] = (float(op.observed_coverage), float(op.macro_f1))
        ax.plot(*marks[ds], marker="o", ms=4.2, color=col,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5)
        ends[ds] = (float(s.observed_coverage.iloc[0]), float(s.macro_f1.iloc[0]))

    ax.set_xlim(1.005, 0.495)                  # reversed: rightwards is more abstention
    ax.set_ylim(0.348, 0.430)
    ax.set_xticks(np.arange(0.5, 1.01, 0.1))
    ax.set_yticks(np.arange(0.35, 0.421, 0.02))
    ax.set_xlabel("retention, fraction of beats the classifier answers")
    ax.set_ylabel("macro-F1 on retained beats")
    hgrid(ax)

    xd, yd = marks["DS2"]
    ax.axvline(xd, color=INK2, lw=0.8, linestyle=DASH["dashed"], zorder=2)
    halo(ax.text(xd - 0.005, 0.386, "nominal target", rotation=90, ha="right",
                 va="bottom", fontsize=ANNOT, color=INK2))

    # direct labels just past the right-hand end of each curve
    for ds, (col, _) in style.items():
        x, y = ends[ds]
        halo(ax.text(x - 0.010, y, ds, ha="left", va="center", fontsize=ANNOT,
                     color=INK))

    xi, _ = marks["INCART"]
    xs, _ = marks["SVDB"]
    ax.text(0.945, 0.4275,
            "At the nominal 90 per cent conformal target the classifier answers\n"
            f"{xd * 100:.1f} per cent of DS2 beats, {xi * 100:.1f} per cent of INCART beats and "
            f"{xs * 100:.1f} per cent of\nSVDB beats. The larger external gap is "
            f"{(xd - xi) * 100:.1f} percentage points.",
            ha="left", va="top", fontsize=6.8, color=INK2, linespacing=1.4)

    handles = [Line2D([], [], color=c, lw=1.2, linestyle=DASH[d], label=k)
               for k, (c, d) in style.items()]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.985, 0.02),
              handlelength=2.4, labelspacing=0.35)
    save(fig, "Figure_5_coverage")


# ============================================================== figure 6 ===

def fig6_noise():
    df = pd.read_csv(FD / "fig6_noise.csv")
    clean = float(df.clean_macro_f1.iloc[0])
    mains = (df[df.noise_type.isin(["50 Hz mains", "60 Hz mains"])]
             .groupby("snr_db", as_index=False).macro_f1.mean())
    mains["noise_type"] = "mains interference"
    df = pd.concat([df[~df.noise_type.str.contains("mains")], mains], ignore_index=True)

    style = {"baseline wander": (BLUE, "solid"),
             "muscle artifact": (ORANGE, "dashed"),
             "electrode motion": (GREEN, "dashdot"),
             "mains interference": (PURPLE, "dotted")}
    snrs = [18, 12, 6, 0]
    xpos = {"clean": 0.0, 18: 1.35, 12: 2.35, 6: 3.35, 0: 4.35}

    fig = plt.figure(figsize=(120 * MM, 78 * MM))
    ax = fig.add_axes([0.115, 0.175, 0.605, 0.805])

    label_y = {}
    for name, (col, dash) in style.items():
        s = df[df.noise_type == name].set_index("snr_db").macro_f1
        ax.plot([xpos["clean"]], [clean], marker="o", ms=3.4, color=col,
                markeredgecolor="white", markeredgewidth=0.4, zorder=4)
        xs = [xpos[v] for v in snrs]
        ys = [float(s.loc[v]) for v in snrs]
        ax.plot(xs, ys, color=col, lw=1.2, linestyle=DASH[dash], marker="o",
                ms=3.4, markeredgecolor="white", markeredgewidth=0.4, zorder=4)
        label_y[name] = ys[-1]

    # direct labels at the right-hand end, pushed apart where they would collide
    for name, y in sorted(label_y.items(), key=lambda kv: kv[1]):
        halo(ax.text(xpos[0] + 0.16, y, name, ha="left", va="center",
                     fontsize=ANNOT, color=INK))

    ax.axhline(clean, color=INK2, lw=0.8, linestyle=DASH["dashed"], zorder=2)
    halo(ax.text(-0.32, clean - 0.0022, f"clean signal, {clean:.3f}", ha="left",
                 va="top", fontsize=ANNOT, color=INK2))

    # the axis is not continuous between the clean category and 18 dB
    ax.set_xlim(-0.35, 4.75)
    ax.set_ylim(0.302, 0.370)
    ax.set_xticks([xpos["clean"]] + [xpos[v] for v in snrs],
                  ["clean"] + [str(v) for v in snrs])
    ax.set_yticks(np.arange(0.31, 0.371, 0.01))
    for xb in (0.60, 0.75):
        ax.plot([xb - 0.06, xb + 0.06], [0.3015, 0.3025], color=INK2, lw=0.8,
                clip_on=False, zorder=6)
    ax.set_xlabel("signal-to-noise ratio, dB, conditions worsen to the right")
    ax.set_ylabel("macro-F1 on DS2")
    hgrid(ax)

    em6 = float(df[(df.noise_type == "electrode motion")
                   & (df.snr_db == 6)].macro_f1.iloc[0])
    halo(ax.annotate(f"{em6:.3f}", xy=(xpos[6], em6),
                     xytext=(xpos[6] - 0.60, em6 + 0.0045),
                     fontsize=ANNOT, color=GREEN, ha="right", va="center",
                     arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.7,
                                     shrinkA=2, shrinkB=3)))

    handles = [Line2D([], [], color=c, lw=1.2, linestyle=DASH[d], label=k)
               for k, (c, d) in style.items()]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0),
              handlelength=2.6, labelspacing=0.3)
    fig.text(0.115, 0.005,
             "Mains interference is the mean of the 50 and 60 Hz sweeps; at 0 dB, "
             "50 Hz alone gives 0.380 and 60 Hz alone 0.352.",
             ha="left", va="bottom", fontsize=6.4, color=INK2)
    save(fig, "Figure_6_noise")


# ============================================================== figure 7 ===

def fig7_frontier():
    df = pd.read_csv(FD / "fig7_frontier.csv")
    placeable = df[df.model_size_kb.notna()].copy()
    strip = df[df.model_size_kb.isna()].copy()

    NO_ENERGY = 55.0

    def bubble(e):                              # log area scale for energy
        if pd.isna(e):
            return NO_ENERGY
        lo, hi = np.log10(6e-5), np.log10(2.4e4)
        return 60.0 + 300.0 * (np.log10(e) - lo) / (hi - lo)

    # a task is comparable only if it is five-class AAMI beat classification
    five_aami = {"Farag": False, "Kim": True, "An": False, "Banjo": False,
                 "This work": True}
    labels = {"Farag": "Farag 2023\n3/4/5-class variants",
              "Kim": "Kim et al. 2024\n5-class AAMI",
              "An": "An et al. 2024\n4 rhythm classes",
              "Banjo": "Banjo and Ghoraani 2026\n4-class, Q omitted",
              "This work": "This work 2026\n5-class AAMI"}
    offsets = {"Farag": (0, 20), "An": (-14, -26), "Banjo": (14, -46),
               "Kim": (0, -26), "This work": (34, 24)}

    fig = plt.figure(figsize=(W2, 118 * MM))
    ax = fig.add_axes([0.070, 0.425, 0.585, 0.545])
    axs = fig.add_axes([0.070, 0.195, 0.585, 0.115])

    ax.add_patch(Rectangle((3.4, 95), 20 - 3.4, 5.4, facecolor=PALE,
                           edgecolor="none", zorder=0))
    ax.text(3.6, 95.3, "target region: model under 20 kB,\naccuracy above 95 per cent",
            ha="left", va="bottom", fontsize=ANNOT, color=INK2, linespacing=1.3)

    for _, r in placeable.iterrows():
        this = r.study == "This work"
        col = ORANGE if this else BLUE
        size = bubble(r.energy_uj) * (1.25 if this else 1.0)
        filled = bool(r.inter_patient)
        ax.scatter([r.model_size_kb], [r.accuracy_pct], s=size,
                   facecolor=col if filled else "none", edgecolor=col,
                   linewidth=0.9, zorder=4)
        if pd.isna(r.energy_uj):               # crossed centre, no energy reported
            ax.scatter([r.model_size_kb], [r.accuracy_pct], s=14, marker="x",
                       color="white" if filled else col, linewidth=0.9, zorder=5)
        if not five_aami[r.study]:
            ax.scatter([r.model_size_kb], [r.accuracy_pct], s=size * 3.0,
                       facecolor="none", edgecolor=INK2, linewidth=0.8,
                       linestyle=DASH["dotted"], zorder=3)
        dx, dy = offsets[r.study]
        ax.annotate(labels[r.study], xy=(r.model_size_kb, r.accuracy_pct),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.8, color=INK, ha="center", linespacing=1.25,
                    arrowprops=dict(arrowstyle="-", color=NOSKILL, lw=0.5,
                                    shrinkA=0, shrinkB=6))

    ax.set_xscale("log")
    ax.set_xlim(3.4, 120)
    ax.set_ylim(80, 100.4)
    ax.set_xticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
    ax.set_yticks(np.arange(80, 101, 5))
    ax.set_xlabel("model size, kB, log scale")
    ax.set_ylabel("reported accuracy, per cent\n(axis range 80 to 100, not from zero)")
    hgrid(ax, axis="both")
    ax.minorticks_off()

    # the strip: studies whose size is not denominated in bytes
    axs.set_xlim(0, 1)
    axs.set_ylim(0, 1)
    axs.set_yticks([])
    axs.set_xticks([])
    for s in axs.spines.values():
        s.set_visible(False)
    axs.axhline(1.0, color=INK2, lw=0.6)
    units = {"Mian": "Mian and Zafar 2024\nFPGA resource counts only, 5-class",
             "Diware": "Diware et al. 2025\n0.36 mm² die area, 11-class",
             "Mommen": "Mommen et al. 2026\n2,000 to 2,990 FPGA LUTs, 4-class"}
    for x, (_, r) in zip((0.16, 0.50, 0.84), strip.iterrows()):
        col = BLUE
        filled = bool(r.inter_patient)
        axs.scatter([x], [0.42], s=bubble(r.energy_uj),
                    facecolor=col if filled else "none", edgecolor=col,
                    linewidth=0.9, zorder=4)
        if pd.isna(r.energy_uj):
            axs.scatter([x], [0.42], s=14, marker="x",
                        color="white" if filled else col, linewidth=0.9, zorder=5)
        axs.text(x, 0.18, units[r.study], ha="center", va="top", fontsize=6.8,
                 color=INK, linespacing=1.25)
    axs.text(0.0, 0.95, "size not reported in bytes; not placeable on the axis above",
             ha="left", va="top", fontsize=ANNOT, color=INK2)

    fig.text(0.070, 0.035,
             "3 of the 7 anchor studies report no model size in bytes and 4 report no "
             "energy per inference.\nNo trend line is drawn: the points come from "
             "different protocols, datasets and class definitions, so no relationship\n"
             "between size and accuracy can be estimated from them.",
             ha="left", va="bottom", fontsize=6.8, color=INK2, linespacing=1.4)

    # legends, no frames
    prot = [Line2D([], [], marker="o", linestyle="", markerfacecolor=BLUE,
                   markeredgecolor=BLUE, ms=5, label="partitioned by patient"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                   markeredgecolor=BLUE, ms=5, label="not partitioned by patient,\nor not stated"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                   markeredgecolor=INK2, ms=8, markeredgewidth=0.8,
                   label="task is not five-class\nAAMI beat classification"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor=ORANGE,
                   markeredgecolor=ORANGE, ms=5, label="this study")]
    lg1 = ax.legend(handles=prot, loc="upper left", bbox_to_anchor=(1.05, 1.02),
                    title="evaluation protocol", handlelength=1.2,
                    labelspacing=0.85, borderaxespad=0.0)
    lg1.get_title().set_fontsize(AXTITLE)
    lg1.get_title().set_color(INK)
    ax.add_artist(lg1)

    ener = [Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                   markeredgecolor=BLUE, ms=np.sqrt(bubble(v)), label=lab)
            for v, lab in ((6e-5, "0.00006 µJ, FPGA, estimated"),
                           (111.34, "111.3 µJ, this work, computed"),
                           (2.331e4, "23,310 µJ, board level"))]
    ener.append(Line2D([], [], marker=r"$\otimes$", linestyle="", color=BLUE,
                       ms=np.sqrt(NO_ENERGY), label="no energy reported"))
    lg2 = ax.legend(handles=ener, loc="upper left", bbox_to_anchor=(1.05, 0.36),
                    title="energy per inference, log area scale", handlelength=1.4,
                    labelspacing=1.15, borderaxespad=0.0)
    lg2.get_title().set_fontsize(AXTITLE)
    lg2.get_title().set_color(INK)
    fig.text(0.675, 0.215,
             "The energy figures do not share a measurement\nboundary: a combinational "
             "logic network on\nprogrammable logic and a board-level figure that\n"
             "includes a host processor are not comparable.",
             ha="left", va="top", fontsize=6.6, color=INK2, linespacing=1.4)

    save(fig, "Figure_7_size_accuracy")


# ============================================================ figure S4.1 ===

def figS41_convergence():
    df = pd.read_csv(FD / "figS41_convergence.csv")
    groups = [("a", ["Random search", "CMA-ES", "TPE"]),
              ("b", ["WHOA", "GPC"])]
    cols = [BLUE, ORANGE, GREEN]
    dashes = ["solid", "dashed", "dashdot"]

    fig, axes = plt.subplots(1, 2, figsize=(W2, 78 * MM), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.205, wspace=0.07)

    for ax, (letter, names) in zip(axes, groups):
        for name, col, dash in zip(names, cols, dashes):
            s = df[df.optimiser == name].sort_values("evaluations")
            ax.plot(s.evaluations, s["mean"], color=col, lw=1.2,
                    linestyle=DASH[dash], zorder=3)
            ax.fill_between(s.evaluations, s.q25, s.q75, color=col, alpha=0.18,
                            linewidth=0, zorder=2)
        g = df[df.optimiser == "Gradient descent"].sort_values("evaluations")
        ax.plot(g.evaluations, g["mean"], color=INK2, lw=1.2, zorder=4)
        ax.fill_between(g.evaluations, g.q25, g.q75, color=INK2, alpha=0.18,
                        linewidth=0, zorder=2)

        ax.set_yscale("log")
        ax.set_xlim(0, 15000)
        ax.set_ylim(8e-3, 60)
        ax.set_xticks(range(0, 15001, 5000))
        ax.set_xlabel("objective evaluations")
        hgrid(ax)
        panel_letter(ax, letter, dx=-0.02, dy=1.02)

        # direct labels inside the right-hand end, nudged apart in log space
        ends = [(n, float(df[df.optimiser == n]["mean"].iloc[-1]), c)
                for n, c in list(zip(names, cols)) + [("Gradient descent", INK2)]]
        ends.sort(key=lambda t: t[1])
        ly = [np.log10(y) for _, y, _ in ends]
        for i in range(1, len(ly)):
            ly[i] = max(ly[i], ly[i - 1] + 0.24)
        for (n, _, c), y in zip(ends, ly):
            halo(ax.text(14700, 10 ** y, n, ha="right", va="center",
                         fontsize=ANNOT, color=c, zorder=6))
        handles = [Line2D([], [], color=c, lw=1.2, linestyle=DASH[d], label=n)
                   for n, c, d in zip(names, cols, dashes)]
        handles.append(Line2D([], [], color=INK2, lw=1.2,
                              label="gradient descent, reference"))
        ax.legend(handles=handles, loc="upper right", handlelength=2.4,
                  labelspacing=0.3)

    axes[0].set_ylabel("stage-one objective, log scale,\nlower is better")
    axes[1].set_xlim(0, 15000)
    fig.text(0.075, 0.012, "Bands are the interquartile range over 30 seeded runs, "
             "not a confidence interval. Gradient descent is repeated in both panels "
             "as the shared reference.", ha="left", va="bottom", fontsize=6.8,
             color=INK2)
    save(fig, "Figure_S4-1_optimiser_convergence")


# ============================================================ figure S4.2 ===

def figS42_roc():
    df = pd.read_csv(FD / "figS42_roc.csv")
    order = ["N", "S", "V", "F", "Q"]

    fig, axes = plt.subplots(1, 5, figsize=(W2, 56 * MM), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.79, bottom=0.19, wspace=0.24)

    for k, (ax, cls) in enumerate(zip(axes, order)):
        s = df[df.cls == cls].sort_values("fpr")
        ax.plot([0, 1], [0, 1], color=NOSKILL, lw=0.8, linestyle=DASH["dashed"],
                zorder=2)
        ax.plot(s.fpr, s.tpr, color=BLUE, lw=1.2, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        hgrid(ax)
        ax.set_title(f"{cls}, n = {SUPPORT[cls]:,}\n"
                     f"AUC {float(s.auc.iloc[0]):.3f}", pad=4, linespacing=1.35)
        panel_letter(ax, "abcde"[k], dx=-0.05, dy=1.26)
        if cls == "Q":
            ax.text(0.04, 0.90, "7 beats in DS2;\nnot interpretable", ha="left",
                    va="top", fontsize=ANNOT, color=INK2, linespacing=1.3)

    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    save(fig, "Figure_S4-2_roc")


# ============================================================ figure S4.3 ===

def figS43_external_confusion():
    df = pd.read_csv(FD / "fig2_confusion.csv")
    order = ["N", "S", "V", "F", "Q"]
    sets = ["INCART", "SVDB"]

    fig = plt.figure(figsize=(120 * MM, 66 * MM))
    axes = [fig.add_axes([0.085 + i * 0.415, 0.15, 0.345, 0.71]) for i in range(2)]
    cax = fig.add_axes([0.925, 0.15, 0.022, 0.71])

    for k, (ax, ds) in enumerate(zip(axes, sets)):
        sub = df[df.dataset == ds]
        frac = np.zeros((5, 5))
        cnt = np.zeros((5, 5), dtype=int)
        for _, r in sub.iterrows():
            i, j = order.index(r.true_class), order.index(r.pred_class)
            frac[i, j], cnt[i, j] = r.row_fraction, int(r["count"])
        ax.imshow(np.where(cnt > 0, frac, np.nan), cmap=SEQ, vmin=0, vmax=1,
                  aspect="equal", interpolation="nearest")
        for i in range(5):
            for j in range(5):
                if cnt[i, j] == 0:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                           facecolor="white", edgecolor=GRID,
                                           lw=0.4, zorder=2))
                    continue
                col = "white" if frac[i, j] > 0.55 else INK
                ax.text(j, i - 0.15, f"{frac[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.0, color=col, zorder=3)
                ax.text(j, i + 0.24, f"({cnt[i, j]:,})", ha="center", va="center",
                        fontsize=5.8, color=col, zorder=3)
        si, ni = order.index("S"), order.index("N")
        ax.add_patch(Rectangle((ni - .5, si - .5), 1, 1, fill=False,
                               edgecolor=ORANGE, lw=1.2, zorder=4))
        ax.set_xticks(range(5), order)
        ax.set_yticks(range(5), order)
        ax.set_xlabel("predicted class")
        if k == 0:
            ax.set_ylabel("true class")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        panel_letter(ax, "ab"[k], dx=-0.18, dy=1.02)
        ax.text(0.5, 1.02, ds, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=PTITLE, color=INK)

    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax, ticks=[0, .25, .5, .75, 1])
    cb.set_label("row-normalised proportion", fontsize=AXTITLE)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK2)
    cb.ax.tick_params(length=2.4, width=0.6, labelsize=TICK)
    save(fig, "Figure_S4-3_external_confusion")


# ============================================================ figure S4.4 ===

def figS44_slopegraph():
    df = pd.read_csv(FD / "figS44_slopegraph.csv")
    names = {"A": "A, best classical", "B": "B, fuzzy decoder",
             "B2": "B2, clinical rules", "C": "C, argmax decoder",
             "D": "D, separable CNN", "E": "E, student", "F": "F, TCN student"}

    fig = plt.figure(figsize=(132 * MM, 90 * MM))
    ax = fig.add_axes([0.305, 0.150, 0.385, 0.770])

    def nudge(vals, minsep):
        """Push label positions apart just enough that no two collide."""
        keys = sorted(vals, key=lambda k: vals[k])
        v = [vals[k] for k in keys]
        for i in range(1, len(v)):
            if v[i] - v[i - 1] < minsep:
                v[i] = v[i - 1] + minsep
        return dict(zip(keys, v))

    left = {r.arm: float(r.mixed) for _, r in df.iterrows()}
    right = {r.arm: float(r.inter) for _, r in df.iterrows()}
    lt, rt = nudge(left, 0.030), nudge(right, 0.030)

    for _, r in df.iterrows():
        changed = int(r.rank_change) != 0
        col = ORANGE if changed else INK2
        lw = 1.4 if changed else 1.0
        ax.plot([0, 1], [r.mixed, r.inter], color=col, lw=lw, zorder=3)
        ax.plot([0, 1], [r.mixed, r.inter], marker="o", ms=2.8, linestyle="",
                color=col, zorder=4)
        ax.text(-0.06, lt[r.arm], f"{names[r.arm]}  {r.mixed:.2f}", ha="right",
                va="center", fontsize=ANNOT, color=INK)
        ax.text(1.06, rt[r.arm], f"{r.inter:.2f}  {names[r.arm]}", ha="left",
                va="center", fontsize=ANNOT, color=INK)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.04, 0.80)
    ax.set_xticks([])
    # A slopegraph carries its values at both ends, so the numeric ticks would
    # only collide with the labels. The range is stated in the axis titles.
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_color(INK2)
    ax.text(0, 0.825, "patient-mixed partition", ha="center", va="bottom",
            fontsize=AXTITLE, color=INK)
    ax.text(1, 0.825, "inter-patient partition, DS2", ha="center", va="bottom",
            fontsize=AXTITLE, color=INK)
    changed = int((df.rank_change != 0).sum())
    gain = float((df.mixed - df.inter).mean())
    fig.text(0.5, 0.012,
             f"Both axes carry macro-F1 on one common scale, 0.04 to 0.80. "
             f"{changed} of 7 arms change rank;\nmean gain under the patient-mixed "
             f"partition {gain:.3f} macro-F1. Colour is reserved for a rank change, "
             f"so\nevery line is drawn in the same ink.",
             ha="center", va="bottom", fontsize=ANNOT, color=INK2,
             linespacing=1.35)
    save(fig, "Figure_S4-4_protocol_slopegraph")


# ------------------------------------------------------------------ main --

FIGURES = [fig1_pipeline, fig2_confusion, fig3_precision_recall, fig4_reliability,
           fig5_coverage, fig6_noise, fig7_frontier, figS41_convergence,
           figS42_roc, figS43_external_confusion, figS44_slopegraph]


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for fn in FIGURES:
        fn()
    if PROBLEMS:
        print(f"\n  {len(PROBLEMS)} layout problems:")
        for p in PROBLEMS:
            print(f"    {p}")
    else:
        print("\n  layout check clean: no text collision, no box overflow")
    print(f"FIGURES_COMPLETE {len(SAVED)} figures into {OUT.name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
