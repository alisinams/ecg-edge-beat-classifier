"""Update 09_Figures_Guide.md with the values and the structure the results imply.

Two kinds of edit. Numeric: a prompt that says "annotate with its value" now
carries the value, read from results/numbers.json, so the number drawn inside a
figure and the number printed in its table cannot drift apart. Structural:
Figure 7 changes shape, because only four of the seven anchor studies report a
model size in bytes and three of them cannot be placed on a kilobyte axis
without an assumption this paper would then have to own.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RESULTS, jload
from msctx import context, noise_at

GUIDE = ROOT / "09_Figures_Guide.md"
# The updater rewrites anchors that exist only in the original text, so running it
# twice would silently skip every replacement the first run consumed. It therefore
# always starts from a pristine copy of the guide as written before any results
# existed, which makes it idempotent and makes a rerun after new numbers correct.
TEMPLATE = Path(__file__).resolve().parent / "09_Figures_Guide.template.md"


def main() -> int:
    C = context()
    t2, t3, t5, t8 = C.t2, C.t3, C.t5, C.t8
    E, cal, abst, alarms = C.E, C.cal, C.abst, C.alarms
    prot, T1 = C.prot, C.T1
    text = (TEMPLATE if TEMPLATE.exists() else GUIDE).read_text(encoding="utf-8")

    n_size = sum(1 for k in T1 if not k.startswith("_")
                 and T1[k]["model_size_kb_for_fig7"] is not None)
    n_nosize = sum(1 for k in T1 if not k.startswith("_")
                   and T1[k]["model_size_kb_for_fig7"] is None)
    n_noenergy = sum(1 for k in T1 if not k.startswith("_") and T1[k]["energy"] == "nr")
    supports = {c: t2["DS2 test"][c] for c in C.CLS}
    sn = None
    fd = RESULTS / "figure_data" / "fig2_confusion.csv"
    if fd.exists():
        import pandas as pd
        d = pd.read_csv(fd)
        r = d[(d.dataset == "DS2") & (d.true_class == "S") & (d.pred_class == "N")]
        if len(r):
            sn = float(r.row_fraction.iloc[0])
    em6 = noise_at(C, "electrode motion", 6)
    clean = C.noise.get("clean_macro_f1", float("nan"))
    ds2c = abst.get("DS2", {}).get("coverage", float("nan"))
    incc = abst.get("INCART", {}).get("coverage", float("nan"))
    svdc = abst.get("SVDB", {}).get("coverage", float("nan"))
    gap = max(abs(ds2c - incc), abs(ds2c - svdc))
    budget = C.opt_setup.get("budget", 15000)

    reps = []

    # ---------------------------------------------------------------- header
    reps.append((
        "One entry per figure: what it is for, what data feeds it, how it is built, what must not appear in it, how to check it, and a complete generation prompt.",
        "One entry per figure: what it is for, what data feeds it, how it is built, what must not appear in it, how to check it, and a complete generation prompt. Every entry names the exact file in `results/figure_data/` that feeds it, and every numeric value quoted inside a prompt was written into that file by `code/18_figure_data.py` from the same predictions that produce the tables."))

    # ------------------------------------------------------------- figure 2
    reps.append((
        "**Data source.** The confusion matrices behind Table 3 for DS2, and behind Table 5 for INCART and SVDB.",
        f"**Data file.** `results/figure_data/fig2_confusion.csv`, columns dataset, true_class, pred_class, count, row_fraction. It carries the confusion matrices behind Table 3 for DS2 and behind Table 5 for INCART and SVDB, averaged over the five training seeds.\n\n**Value to check.** On DS2 the S-row, N-column cell holds {sn:.2f} of the S row" + (f", which is the single number the caption argues from." if sn is not None else ".")))

    # ------------------------------------------------------------- figure 3
    reps.append((
        'Panel titles are the class letter and its DS2 support, for example "S, n = [[RESULT: S beats in DS2]]".',
        "Panel titles are the class letter and its DS2 support: "
        + ", ".join(f'"{c}, n = {supports[c]:,}"' for c in C.CLS)
        + ". The Q panel is drawn but carries a note that its support is too small to interpret."))
    reps.append((
        "**Data source.** Per-class precision and recall at every threshold on DS2, from the same predictions that produce Table 3.",
        "**Data file.** `results/figure_data/fig3_precision_recall.csv`, columns cls, precision, recall, baseline, average_precision, thinned to at most 2,000 points per class."))
    reps.append((
        "One panel per class, in order N, S, V, F, Q, each panel titled with the class letter and its test-set support.",
        f"One panel per class, in order N, S, V, F, Q, titled "
        + ", ".join(f'"{c}, n = {supports[c]:,}"' for c in C.CLS) + "."))

    # ------------------------------------------------------------- figure 4
    reps.append((
        "**Data source.** Calibrated and uncalibrated predicted probabilities on DS2, 15 equal-width bins, from section 7.7.",
        f"**Data file.** `results/figure_data/fig4_reliability.csv`, columns stage, bin_lo, bin_hi, n, confidence, accuracy, for 15 equal-width bins on DS2 before and after temperature scaling.\n\n**Values to print.** Expected calibration error {cal.get('ece_before', float('nan')):.4f} before and {cal.get('ece_after', float('nan')):.4f} after, fitted temperature {cal.get('temperature', float('nan')):.3f}."))
    reps.append((
        "In the upper right print two lines of 7 pt text giving the expected calibration error before and after.",
        f"In the upper right print two lines of 7 pt text reading exactly \"ECE before {cal.get('ece_before', float('nan')):.4f}\" and \"ECE after {cal.get('ece_after', float('nan')):.4f}\"."))

    # ------------------------------------------------------------- figure 5
    reps.append((
        "**Data source.** The conformal abstention sweep of section 7.7, evaluated on DS2, INCART and SVDB.",
        f"**Data file.** `results/figure_data/fig5_coverage.csv`, columns dataset, target_coverage, observed_coverage, macro_f1, accuracy, n.\n\n**Values to check.** At the 90 per cent nominal target the observed coverage is {100*ds2c:.1f} per cent on DS2, {100*incc:.1f} per cent on INCART and {100*svdc:.1f} per cent on SVDB; the larger external gap is {100*gap:.1f} percentage points and it is the quantity the annotation must carry."))
    reps.append((
        "At that reference line, draw a short vertical connector between the DS2 curve and each external curve and annotate the larger gap once with its numeric size to two decimals.",
        f"At that reference line, draw a short vertical connector between the DS2 curve and each external curve and annotate the larger gap once, labelled \"{100*gap:.1f} percentage points\"."))

    # ------------------------------------------------------------- figure 6
    reps.append((
        "**Data source.** The noise stress sweep of section 7.7 at 18, 12, 6 and 0 dB, using the MIT-BIH Noise Stress Test Database plus synthetic mains interference.",
        f"**Data file.** `results/figure_data/fig6_noise.csv`, columns noise_type, snr_db, macro_f1, clean_macro_f1. Mains interference is drawn as one series; the 50 Hz and 60 Hz sweeps are separate rows in the file and are averaged for the figure, which the caption states.\n\n**Values to check.** Clean-signal macro-F1 {clean:.3f}; electrode motion at 6 dB {em6:.3f}, which is the annotated point."))
    reps.append((
        "Annotate the electrode-motion curve at 6 dB with its value to two decimals using a short leader line.",
        f"Annotate the electrode-motion curve at 6 dB with the value {em6:.3f} using a short leader line."))
    reps.append((
        'Draw a horizontal dashed reference line at the clean-signal macro-F1 in #5C5C5A, 0.8 pt, labelled "clean signal" at 7 pt.',
        f'Draw a horizontal dashed reference line at the clean-signal macro-F1 of {clean:.3f} in #5C5C5A, 0.8 pt, labelled "clean signal, {clean:.3f}" at 7 pt.'))

    # ------------------------------------------------------------- figure 7
    old7_start = text.find("# Figure 7. Accuracy against model size")
    old7_end = text.find("# Figure S4.1. Optimiser convergence")
    new7 = f"""# Figure 7. Accuracy against model size

**Where.** Section 7.8, Results. **Caption.** Already written.

**The question it answers.** Which region of the accuracy-against-size plane is occupied by studies that partitioned by patient and reported a deployment profile, and how empty is that region?

**Form.** A scatter plot with composite encoding, plus a separate strip beneath the axis for studies that report no size in bytes. Identity is carried by direct labels, not by hue, because each point is a different study and a hue per study would exceed any validated palette.

**Data file.** `results/figure_data/fig7_frontier.csv`, columns study, model_size_kb, size_reported_as, accuracy_pct, classes, aami, inter_patient, energy_uj, latency_ms, target, note.

**This entry changed after the anchor papers were read, and the reason is the finding.** Only {n_size} of the seven anchor studies report a model size in bytes at all. Mian and colleagues report FPGA resource counts, Diware and colleagues report die area in square millimetres, and Mommen and colleagues report lookup-table counts. Converting any of those into kilobytes requires an assumption about the target technology that this paper would then own and have to defend, so they are not converted. The {n_nosize} studies without a byte-denominated size are drawn in a strip beneath the horizontal axis, each with its reported resource unit printed next to it, and the figure says in its own annotation that they could not be placed on the axis. That is more honest than a scatter plot in which three of ten points are invented.

A second change follows from the same reading. The class definitions are not comparable across the anchor set: two studies drop the Q superclass and classify four, one uses an eleven-class severity grouping that explicitly sets the AAMI grouping aside, one classifies four rhythm classes on a different database rather than beats, and one reports a three-class headline whose five-class variant scores {T1['Farag2023']['macro_f1_5class_pct']:.1f} per cent average F1 on the same data. Every marker therefore carries its class count in its label, and the two points whose task is not five-class AAMI beat classification are drawn with a dotted outline and named in the annotation as measuring a different quantity.

**Axes.** Horizontal, model size in kilobytes, logarithmic, spanning roughly 10 to 100 kB, which is the range the byte-reporting studies actually occupy; do not pad the axis to three decades when the data occupy one. Vertical, reported accuracy in per cent. The vertical axis does not start at zero, and because this is a scatter plot and not a bar chart that is correct; state the range explicitly on the axis.

**Encoding.** Marker fill encodes protocol: filled markers for studies that partitioned by patient, hollow markers with a 0.8 pt outline for studies that did not or did not say. Marker area encodes reported energy per inference on a logarithmic area scale, spanning the {T1['Mommen2026']['energy'].split(',')[0]} of a logic-gate network on programmable logic to the {T1['Banjo2026']['energy'].split(';')[0]} of a board-level figure that includes a host processor; the legend must state that those two are not the same measurement boundary. Studies reporting no energy figure are drawn at a fixed small size with an open centre and a cross-hatch, and the legend states that {n_noenergy} of the points are of that kind. All markers share one colour, `#0072B2`, except this study's point, which is `#D55E00` and slightly larger. Each point carries a direct text label with the first author, the year and the class count at 6.5 pt.

**Annotations.** A light `#F2F7FB` shaded rectangle covering the upper-left region, small models with high inter-patient accuracy, labelled "target region" at 7 pt. The caption argues from the emptiness of that rectangle, so it must be drawn. One line of 7 pt text beneath the strip reading that {n_nosize} studies report no size in bytes and {n_noenergy} report no energy.

**Prohibited here.** No trend line, no regression, no fitted frontier curve. The points come from different protocols, different datasets and different class definitions, and a line through them would assert a relationship the data cannot support. No conversion of FPGA lookup-table counts or die area into kilobytes.

**Generation prompt.**

> Produce a single-panel scatter plot with a companion strip for a scientific paper, 180 mm wide, vector output. Main panel: horizontal axis "model size, kB" on a logarithmic scale spanning 10 to 100 with labelled ticks at 10, 20, 50 and 100; vertical axis "reported accuracy, per cent", range set to fit the data and stated explicitly, not starting at zero. Plot one marker per published study that reports a size in bytes, which is {n_size} studies, plus one marker for this study. Marker fill encodes evaluation protocol: solid fill for studies that partitioned by patient, hollow with a 0.8 pt outline for studies that did not or did not report it. Marker area encodes energy per inference on a logarithmic area scale; studies with no reported energy are drawn at a fixed small size with an open centre and a light cross-hatch. All markers are #0072B2 except the point for this study, which is #D55E00 and one step larger. Draw a dotted 0.8 pt outline around any marker whose classification task is not five-class AAMI beat classification. Label every point directly with first author, year and class count at 6.5 pt in ink colour, using thin leader lines where labels would collide. Beneath the horizontal axis, separated by a 2 mm gap and a thin rule, draw a one-dimensional strip containing {n_nosize} markers for the studies that report no size in bytes, each labelled with its author, year and its reported resource unit, for example "2000 to 2990 FPGA LUTs" or "0.36 mm2 die area"; the strip has no numeric axis and its own 7 pt label reading "size not reported in bytes; not placeable on the axis above". Shade the upper-left region of the main panel, small size and high accuracy, with a #F2F7FB rectangle behind the data, labelled "target region" at 7 pt. Add two legends without frames: one for protocol showing a solid and a hollow marker, one for energy showing three reference bubble sizes with their values and a 6.5 pt note that the energy figures use different measurement boundaries. Print one line of 7 pt text beneath the strip stating that {n_nosize} of the anchor studies report no size in bytes and {n_noenergy} report no energy. Do not draw any trend line, regression or frontier curve, and do not convert lookup-table counts or die area into kilobytes. Horizontal and vertical grid at 0.5 pt #E5E5E4 behind the data; left and bottom spines only. Sans-serif, tick labels 7 pt, axis titles 8 pt. No title inside the image, no shadow.

"""
    if old7_start > 0 and old7_end > old7_start:
        text = text[:old7_start] + new7 + text[old7_end:]

    # ------------------------------------------------------------- S4.1
    reps.append((
        'horizontal "objective evaluations" from 0 to 15000',
        f'horizontal "objective evaluations" from 0 to {budget}'))
    reps.append((
        "**Axes.** Horizontal, objective evaluations, 0 to 15000, linear.",
        f"**Data file.** `results/figure_data/figS41_convergence.csv`, columns optimiser, evaluations, mean, q25, q75.\n\n**Axes.** Horizontal, objective evaluations, 0 to {budget}, linear."))

    # ------------------------------------------------------------- S4.4
    reps.append((
        "**Form.** A slopegraph. Two vertical axes, one line per arm connecting its macro-F1 under each protocol.",
        f"**Data file.** `results/figure_data/figS44_slopegraph.csv`, columns arm, mixed, inter, rank_mixed, rank_inter, rank_change.\n\n**Form.** A slopegraph. Two vertical axes, one line per arm connecting its macro-F1 under each protocol. {prot['n_rank_changes']} of the seven arms change rank, so {prot['n_rank_changes']} lines are drawn in the contrasting colour."))
    reps.append((
        "Add one line of 7 pt text beneath the plot stating how many arms changed rank.",
        f"Add one line of 7 pt text beneath the plot reading \"{prot['n_rank_changes']} of 7 arms change rank; mean gain under the patient-mixed partition {prot['mean_gain']:.3f} macro-F1\"."))

    # ------------------------------------------------------------- S4.2
    reps.append((
        "**Encoding.** One curve per panel in `#0072B2`, the chance diagonal in `#9A9A98` dashed, area under the curve printed in the panel corner at 7 pt to two decimal places with its bootstrap interval.",
        "**Data file.** `results/figure_data/figS42_roc.csv`, columns cls, fpr, tpr, auc.\n\n**Encoding.** One curve per panel in `#0072B2`, the chance diagonal in `#9A9A98` dashed, area under the curve printed in the panel corner at 7 pt to two decimal places with its bootstrap interval."))

    # ------------------------------------------------------- build notes
    reps.append((
        "Every figure is regenerated by a command in supplement S5, so no figure is drawn by hand and none is edited after generation. Values printed inside a figure are read from the same result files that populate the tables, which is what keeps the two consistent under revision.",
        "Every figure is regenerated by a command in supplement S5, so no figure is drawn by hand and none is edited after generation. `code/18_figure_data.py` writes one tidy file per figure into `results/figure_data/`, and every value quoted in a prompt above was taken from those files, which are themselves written from the same prediction arrays that populate the tables. That is what keeps a number inside a figure identical to the same number inside its table under revision.\n\nOne consequence of reading the seven anchor papers is recorded here because it changed a figure and not only a table: three of the seven report no model size in bytes, two report no energy, and three do not partition by patient. Figure 7 was restructured around those absences rather than around the values, and its caption now argues from what the anchor set does not report."))

    for old, new in reps:
        if old not in text:
            print(f"  !! not found, skipped: {old[:70]}...")
            continue
        text = text.replace(old, new, 1)

    GUIDE.write_text(text, encoding="utf-8")
    print(f"FIGURES_GUIDE_UPDATED {len(text)} characters")
    left = text.count("[[RESULT")
    print(f"  remaining RESULT placeholders: {left}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
