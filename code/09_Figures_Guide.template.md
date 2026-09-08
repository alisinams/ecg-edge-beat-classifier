# Figure guide

One entry per figure: what it is for, what data feeds it, how it is built, what must not appear in it, how to check it, and a complete generation prompt. Figure numbers match `05_Manuscript.md` and `08_Manuscript.docx`, where each figure has a reserved slot and its final caption already written.

All seven main figures are cited in the text in numerical order. Supplementary figures carry S-numbers and are cited from `06_Supplementary.md`.

# Shared specification

Everything below applies to every figure unless an entry overrides it.

## Output

Vector PDF is the master format, with EPS exported alongside for journals that demand it, and a 600 dpi TIFF fallback for raster-only submission systems. Fonts are embedded, never outlined, so that the publisher can re-flow text. Nothing is rasterised inside a vector file except the heatmap cell fills in Figure 2, which are rendered as vector rectangles anyway.

Width is fixed at the target column, and figures are drawn at final size so that no scaling happens afterwards. Single column is 88 mm, double column is 180 mm. A figure that only reads at 180 mm is a double-column figure and should be declared as one, not shrunk.

## Typography

One sans-serif family throughout, Arial or Helvetica. Axis tick labels 7 pt, axis titles 8 pt, panel letters 9 pt bold, legend 7 pt, annotations 7 pt. No text below 6 pt at final size. Panel letters are lower-case in parentheses, (a), (b), (c), placed at the top left outside the axes.

Text carries ink colours and never a series colour: primary ink `#1A1A1A` for labels and titles, secondary ink `#5C5C5A` for annotations and captions inside the plot. A number printed next to a coloured mark stays in ink; the mark carries the identity.

## Line and mark weights

Data lines 1.2 pt. Axis spines 0.6 pt in `#5C5C5A`. Grid lines 0.5 pt in `#E5E5E4`, drawn behind the data, horizontal only unless the figure states otherwise. Markers at least 3 mm across at final size. Confidence bands are filled at 18 per cent opacity in the series colour with no outline. Only the left and bottom spines are drawn; top and right are removed.

## Colour

Palettes below were run through the colour-vision validator in `--pairs all` mode against a white surface. None was chosen by eye. Do not substitute colours without re-running that check.

| Role | Values | Validator result |
|---|---|---|
| Two series, for example before and after calibration | `#0072B2`, `#D55E00` | all checks pass, worst pair delta-E 21.9 under protanopia |
| Three series, for example DS2, INCART and SVDB | `#0072B2`, `#D55E00`, `#009E73` | all checks pass, worst pair delta-E 11.0 under deuteranopia |
| Four series, for example the noise types | `#0072B2`, `#D55E00`, `#009E73`, `#A34E86` | passes with the worst pair at delta-E 7.7 under protanopia, which sits in the floor band and is legal only with a second encoding, so this palette is always paired with distinct dash patterns and direct labels |
| Sequential, for magnitude | white to `#0072B2` to `#00344F`, one hue, light to dark | monotone in lightness |
| Reference and annotation | `#5C5C5A` for reference lines, `#9A9A98` for no-skill baselines | non-data ink |

Five AAMI classes are never encoded as five colours in one panel. Five hues cannot be separated under all-pairs colour-vision checking, and a palette that fails that check is not rescued by a legend. Per-class results are drawn as small multiples, one class per panel, one series per panel. This is the reason Figure 3 is a five-panel facet and not a five-line overlay.

Series colour follows the entity, so DS2 is `#0072B2` in every figure in which it appears, and a figure that drops a series never repaints the survivors.

## Prohibited in every figure

No dual y-axis. No 3D effects, gradients, drop shadows or bevels. No rainbow or jet colour map anywhere, including the confusion matrices. No hue at the midpoint of a diverging scale. No red-green pair carrying meaning without a second encoding. No chart junk, no background fill, no boxed frame around the plot. No value label on every point; labels are selective. No truncated y-axis on a bar chart. No pie chart. No figure title inside the image, since the caption carries it.

## Acceptance checks before a figure is considered done

Render it and look at it at final size on paper, not on screen at 200 per cent. Then confirm: every axis has a name and a unit; every series is identifiable without colour, through a direct label, a dash pattern or a facet; no label collides with another; nothing overflows the axes; the grey-scale print version is still readable; and every number drawn in the figure also appears in the corresponding table, with the same rounding.

# Figure 1. Processing pipeline

**Where.** Section 6.5, Methods. **Caption.** Already written in the manuscript.

**The question it answers.** Can a reader confirm, by looking, that no beat crosses the training and test boundary?

**Form.** A block diagram, not a chart. The data's job here is to show structure and a constraint, so no quantitative encoding applies.

**Data source.** None. This is drawn from the protocol in sections 6.1 to 6.9.

**Layout.** Left-to-right flow in two horizontal bands separated by a heavy vertical rule that runs the full height of the figure, labelled "partition boundary". Left band: MIT-BIH records, exclusion of the four paced records, band-pass 0.5 to 40 Hz, resample to 250 Hz, R-peak-centred windowing, AAMI mapping, DS1 split into training and validation subsets. Right band: DS2, INCART and SVDB, each entering only at inference. Below the boundary rule, three arrows cross it, each labelled with what it carries: trained weights, calibration temperature, conformal threshold. No arrow carrying beats crosses the rule, and that absence is the point of the figure.

Inside the model block, show the four depthwise-separable blocks, global average pooling, the join with the four rhythm descriptors, and the linear layer, so the reader can see that the descriptors bypass the convolutional stack.

**Encodings.** Process boxes in white with a 0.8 pt `#5C5C5A` outline. Data stores as cylinders or rounded boxes in `#F2F7FB` fill. The boundary rule 2 pt in `#1A1A1A`. Arrows 0.8 pt with small solid heads. Teacher network drawn as a dashed-outline box, since it exists only during training.

**Generation prompt.**

> Draw a publication-quality block diagram for a machine-learning pipeline, 180 mm wide, vector output, sans-serif labels at 7 to 8 pt. Layout is a left-to-right flow split by one heavy vertical rule labelled "partition boundary" that spans the full figure height. Left of the rule, in sequence: a data store "MIT-BIH Arrhythmia Database, 48 records"; a process box "exclude paced records 102, 104, 107, 217"; "band-pass 0.5 to 40 Hz, zero-phase Butterworth"; "resample to 250 Hz"; "R-peak-centred window, minus 250 ms to plus 400 ms"; "map to AAMI N, S, V, F, Q"; then a split into two stores, "DS1 training subset" and "DS1 validation subset". From the training store an arrow enters a model block containing four stacked sub-boxes labelled "depthwise-separable block" followed by "global average pooling", a small merge node where a separate input labelled "four rhythm descriptors: pre-RR, post-RR, RR ratio to local mean, RR ratio pre to post" joins, then "linear layer, 5 classes". A dashed-outline box above the model block labelled "teacher network, distillation only" connects to it with a dashed arrow. From the validation store, arrows feed two small boxes, "temperature scaling" and "conformal threshold". Right of the rule, three stores stacked vertically: "DS2, 22 records", "INCART", "SVDB", each with an arrow into a single "inference" box that outputs "class decision or abstain". Exactly three arrows cross the vertical rule, left to right, labelled "trained weights", "calibration temperature" and "conformal threshold". No other arrow crosses the rule. Style: white process boxes with 0.8 pt grey outlines, light blue #F2F7FB fills for data stores, the boundary rule 2 pt near-black, arrows 0.8 pt with small solid heads, no shadows, no gradients, no colour beyond the stated fills. Leave no title inside the image.

# Figure 2. Confusion matrices

**Where.** Section 7.2, Results. **Caption.** Already written.

**The question it answers.** Where does the classifier's error mass sit, and is the S-into-N confusion the dominant term on all three datasets?

**Form.** Three heatmaps as small multiples. Magnitude is the job, so a sequential single-hue ramp applies.

**Data source.** The confusion matrices behind Table 3 for DS2, and behind Table 5 for INCART and SVDB.

**Panels.** (a) DS2, (b) INCART, (c) SVDB, side by side at 180 mm, sharing one colour bar placed to the right of panel (c).

**Axes.** Rows are the true class, columns the predicted class, both in the fixed order N, S, V, F, Q. Row and column names on every panel, since a shared axis label across panels invites misreading.

**Encoding.** Row-normalised proportion drives the fill, on the sequential ramp white to `#0072B2` to `#00344F`. Each cell prints two lines of text: the proportion to two decimal places on the first line, the raw count in parentheses on the second, at 6.5 pt. Text switches from `#1A1A1A` to white when the cell fill exceeds 55 per cent of the ramp, so contrast is never lost. Cells with a count of zero are left unfilled with a light grey outline and no text.

**Annotations.** A thin outline in `#D55E00` around the S-row, N-column cell in all three panels, with a single leader label on panel (a) reading "S read as N". This is the only annotation; the caption carries the argument.

**Generation prompt.**

> Produce a three-panel confusion-matrix figure for a scientific paper, 180 mm wide, vector output. Panels (a) DS2, (b) INCART, (c) SVDB, arranged horizontally, sharing one vertical colour bar to the right of panel (c) labelled "row-normalised proportion" with ticks at 0, 0.25, 0.5, 0.75, 1. Each panel is a 5 by 5 grid; rows are true class and columns are predicted class, both ordered N, S, V, F, Q, with axis titles "true class" and "predicted class" on every panel. Fill colour encodes the row-normalised proportion on a single-hue sequential ramp running white to #0072B2 to #00344F; do not use a rainbow, viridis or jet map. In each cell print the proportion to two decimal places on the first line and the raw count in parentheses beneath it, at 6.5 pt, in #1A1A1A on light fills and white on fills above 55 per cent of the ramp. Cells with a zero count are left white with a 0.4 pt #E5E5E4 outline and no text. Draw a 1 pt #D55E00 outline around the cell at row S, column N in all three panels, and add one leader line on panel (a) labelled "S read as N" at 7 pt. Panel letters (a), (b), (c) in 9 pt bold at the top left outside each axes. Sans-serif throughout, tick labels 7 pt, axis titles 8 pt. No panel titles inside the image, no grid, no shadow, no frame.

# Figure 3. Precision-recall curves

**Where.** Section 7.2, Results. **Caption.** Already written.

**The question it answers.** For each class separately, how far above its own no-skill baseline does the classifier operate?

**Form.** Five small multiples, one class per panel, one curve per panel. This is the deliberate alternative to a five-colour overlay, which cannot pass all-pairs colour-vision checking.

**Data source.** Per-class precision and recall at every threshold on DS2, from the same predictions that produce Table 3.

**Panels.** Five panels in one row at 180 mm, or a 3-by-2 grid at 88 mm with the sixth cell used for the legend. Panel titles are the class letter and its DS2 support, for example "S, n = [[RESULT: S beats in DS2]]".

**Axes.** Recall on the horizontal axis, precision on the vertical, both 0 to 1 with ticks every 0.2, identical on every panel so the panels are comparable. Axis titles on the leftmost and bottom panels only.

**Encoding.** One curve per panel in `#0072B2` at 1.2 pt, with a bootstrap band at 18 per cent opacity. The class prevalence is drawn as a horizontal dashed line in `#9A9A98` at 0.8 pt, labelled "no-skill" once, on the first panel only. The operating point selected in section 7.7 is marked with a filled circle 3.5 mm across in `#D55E00` and labelled with its recall and precision on the S panel only.

**Prohibited here.** Do not add a receiver operating characteristic curve to the same axes, and do not print the area under the curve as the headline number, since the area between the curve and its own baseline is what the caption asks the reader to compare.

**Generation prompt.**

> Produce a five-panel precision-recall figure for a scientific paper, 180 mm wide, vector output, panels in a single row. One panel per class, in order N, S, V, F, Q, each panel titled with the class letter and its test-set support. Both axes run 0 to 1 with ticks every 0.2 and identical limits on every panel; horizontal axis "recall", vertical axis "precision", with axis titles only on the bottom-left panel. Each panel shows exactly one precision-recall curve in #0072B2 at 1.2 pt with a shaded 95 per cent bootstrap band in the same colour at 18 per cent opacity and no outline. In each panel draw a horizontal dashed line at that class's prevalence in #9A9A98 at 0.8 pt; label it "no-skill" at 7 pt on the first panel only. On the S panel only, mark the selected operating point with a filled #D55E00 circle 3.5 mm across and annotate it with its recall and precision to two decimals. Horizontal grid lines only, 0.5 pt #E5E5E4, behind the data; left and bottom spines only, 0.6 pt #5C5C5A. Sans-serif, tick labels 7 pt, axis titles 8 pt, panel titles 8 pt, panel letters (a) to (e) 9 pt bold outside the axes at the top left. No legend box is needed because each panel holds one series. No shadows, no fills behind the panels, no title inside the image.

# Figure 4. Reliability diagram

**Where.** Section 7.7, Results. **Caption.** Already written.

**The question it answers.** Is the classifier's confidence usable as a probability, and does a single temperature fix it without moving any decision?

**Form.** A line chart with a reference diagonal, plus a bin-count histogram beneath. Two panels stacked, sharing the horizontal axis.

**Data source.** Calibrated and uncalibrated predicted probabilities on DS2, 15 equal-width bins, from section 7.7.

**Panels.** Upper panel, the reliability curve, occupying about 75 per cent of the figure height. Lower panel, the count of beats per bin, sharing the x-axis. The histogram is not optional; without it the reader over-reads the tails.

**Axes.** Horizontal, mean predicted probability, 0 to 1. Upper vertical, observed frequency, 0 to 1. Lower vertical, beat count, logarithmic because the bin counts span orders of magnitude, with the log scale stated in the axis title.

**Encoding.** Uncalibrated curve `#D55E00`, calibrated curve `#0072B2`, both 1.2 pt with 3 mm circular markers at each bin centre. The perfectly calibrated diagonal in `#5C5C5A`, 0.8 pt, dashed. Histogram bars in `#E5E5E4` with a 0.4 pt `#9A9A98` outline and a 2 px gap between bars.

**Annotations.** Expected calibration error before and after, printed once in the upper left of the upper panel as two lines of 7 pt text, matching the values in section 7.7 exactly. A legend with two entries, upper right of the upper panel, no frame.

**Generation prompt.**

> Produce a two-panel reliability diagram for a scientific paper, 88 mm wide, vector output, panels stacked and sharing the horizontal axis, with the upper panel three times the height of the lower. Horizontal axis "mean predicted probability", 0 to 1, ticks every 0.2. Upper panel vertical axis "observed frequency", 0 to 1, ticks every 0.2; draw the perfectly calibrated diagonal from (0,0) to (1,1) as a dashed 0.8 pt #5C5C5A line; plot two curves over 15 equal-width bins, "before temperature scaling" in #D55E00 and "after temperature scaling" in #0072B2, each 1.2 pt with filled circular markers 3 mm across at bin centres. Place a legend with those two entries in the upper left area, no frame, 7 pt. In the upper right print two lines of 7 pt text giving the expected calibration error before and after. Lower panel vertical axis "beats per bin, log scale", drawn as a bar histogram of the same 15 bins in #E5E5E4 fill with 0.4 pt #9A9A98 outlines and a 2 px gap between bars. Horizontal grid only, 0.5 pt #E5E5E4, behind the data; left and bottom spines only. Sans-serif, tick labels 7 pt, axis titles 8 pt. No title inside the image, no shadow, no background fill.

# Figure 5. Accuracy against coverage under abstention

**Where.** Section 7.7, Results. **Caption.** Already written.

**The question it answers.** What does each abstention buy, and does the conformal coverage guarantee survive a change of recording chain?

**Form.** A line chart, three series, one per dataset. Three series pass all-pairs colour-vision checking, so colour is admissible here.

**Data source.** The conformal abstention sweep of section 7.7, evaluated on DS2, INCART and SVDB.

**Axes.** Horizontal, coverage, the fraction of beats retained, running 1.0 at the left to 0.5 at the right so that the reader moves rightwards into more abstention. Vertical, macro-F1 on retained beats. Both axes named with units, and no dual axis.

**Encoding.** DS2 `#0072B2` solid, INCART `#D55E00` dashed, SVDB `#009E73` dash-dot. The dash patterns are a second encoding and are not optional. Bootstrap bands at 18 per cent opacity. A vertical reference line in `#5C5C5A` at the nominal 90 per cent coverage target, labelled "nominal target". Each curve is directly labelled at its right-hand end, so the legend is a backup and not the only route to identity.

**Annotations.** At the nominal target, drop a short vertical tick from each external curve to the DS2 curve and label the gap once with its size. That gap is the quantity the caption discusses.

**Generation prompt.**

> Produce a single-panel line chart for a scientific paper, 88 mm wide, vector output. Horizontal axis "coverage, fraction of beats retained", reversed so it runs from 1.0 at the left to 0.5 at the right, ticks every 0.1. Vertical axis "macro-F1 on retained beats". Plot three curves: "DS2" in #0072B2 solid, "INCART" in #D55E00 dashed, "SVDB" in #009E73 dash-dot, each 1.2 pt, each with a shaded 95 per cent bootstrap band in its own colour at 18 per cent opacity and no outline. Label each curve directly at its right-hand end in 7 pt ink-coloured text, and also provide a three-entry legend without a frame in the lower left. Draw a vertical reference line at coverage 0.90 in #5C5C5A, 0.8 pt, dashed, labelled "nominal target" at 7 pt rotated vertically. At that reference line, draw a short vertical connector between the DS2 curve and each external curve and annotate the larger gap once with its numeric size to two decimals. Horizontal grid only, 0.5 pt #E5E5E4, behind the data; left and bottom spines only, 0.6 pt #5C5C5A. Sans-serif, tick labels 7 pt, axis titles 8 pt. No second y-axis, no title inside the image, no shadow.

# Figure 6. Noise degradation curves

**Where.** Section 7.7, Results. **Caption.** Already written.

**The question it answers.** Which noise type actually threatens a wearable deployment, and does the filter design handle the other three?

**Form.** A line chart, four series. Four series sit in the colour-vision floor band, so this figure carries dash patterns and direct labels as the mandatory second encoding.

**Data source.** The noise stress sweep of section 7.7 at 18, 12, 6 and 0 dB, using the MIT-BIH Noise Stress Test Database plus synthetic mains interference.

**Axes.** Horizontal, signal-to-noise ratio in dB, decreasing left to right so that conditions worsen rightwards, with an additional leftmost category for the clean signal. Vertical, macro-F1 on DS2.

**Encoding.** Baseline wander `#0072B2` solid, muscle artifact `#D55E00` dashed, electrode motion `#009E73` dash-dot, mains interference `#A34E86` dotted. Every curve directly labelled at its right-hand end. Markers at each measured level, 3 mm.

**Annotations.** A horizontal reference line at the clean-signal macro-F1 in `#5C5C5A`, dashed, labelled "clean signal". The electrode-motion curve at 6 dB is annotated with its value, because the caption and section 7.7 both quote that single number.

**Prohibited here.** Do not connect the clean-signal category to the 18 dB point with the same line style as the rest of the curve, since the horizontal axis is not continuous across that gap; use a visible break.

**Generation prompt.**

> Produce a single-panel line chart for a scientific paper, 88 mm wide, vector output. Horizontal axis "signal-to-noise ratio, dB", categorical with five positions in this order: "clean", 18, 12, 6, 0, so that conditions worsen from left to right; insert a visible axis break between "clean" and 18 and do not draw a connecting segment across it. Vertical axis "macro-F1 on DS2". Plot four curves, each 1.2 pt with 3 mm filled circular markers at every position: "baseline wander" in #0072B2 solid, "muscle artifact" in #D55E00 dashed, "electrode motion" in #009E73 dash-dot, "mains interference" in #A34E86 dotted. The dash patterns are required, not decorative. Label every curve directly at its right-hand end at 7 pt in ink colour, and also give a four-entry legend without a frame in the lower left. Draw a horizontal dashed reference line at the clean-signal macro-F1 in #5C5C5A, 0.8 pt, labelled "clean signal" at 7 pt. Annotate the electrode-motion curve at 6 dB with its value to two decimals using a short leader line. Horizontal grid only, 0.5 pt #E5E5E4, behind the data; left and bottom spines only. Sans-serif, tick labels 7 pt, axis titles 8 pt. No second y-axis, no title inside the image, no shadow, no fill behind the plot.

# Figure 7. Accuracy against model size

**Where.** Section 7.8, Results. **Caption.** Already written.

**The question it answers.** Which region of the accuracy-against-size plane is occupied by studies that partitioned by patient and reported a deployment profile, and how empty is that region?

**Form.** A scatter plot with composite encoding. Identity is carried by direct labels, not by hue, because each point is a different study and a hue per study would exceed any validated palette.

**Data source.** Table 1 of the manuscript, plus this study's Table 8 row. Every anchor value in Table 1 is tagged `[[VERIFY]]`, so the figure cannot be finalised until those cells are read from the primary texts.

**Axes.** Horizontal, model size in kilobytes, logarithmic, spanning roughly 1 to 10000 kB. Vertical, reported accuracy or macro-F1 in per cent. The vertical axis does not start at zero, and because this is a scatter plot and not a bar chart that is correct; state the range explicitly on the axis.

**Encoding.** Marker fill encodes protocol: filled markers for studies that partitioned by patient, hollow markers with a 0.8 pt outline for studies that did not or did not say. Marker area encodes reported energy per inference on a logarithmic area scale, with a size legend of three reference bubbles; studies reporting no energy figure are drawn at a fixed small size with an open centre and a cross-hatch, and the legend states how many such points there are. All markers share one colour, `#0072B2`, except this study's point, which is `#D55E00` and slightly larger. Each point carries a direct text label with the first author and year at 6.5 pt, placed with a leader line where crowding demands it.

**Annotations.** A light `#F2F7FB` shaded rectangle covering the upper-left region, small models with high inter-patient accuracy, labelled "target region" at 7 pt. The caption argues from the emptiness of that rectangle, so it must be drawn.

**Prohibited here.** No trend line, no regression, no fitted frontier curve. The points come from different protocols and different datasets, and a line through them would assert a relationship the data cannot support.

**Generation prompt.**

> Produce a single-panel scatter plot for a scientific paper, 180 mm wide, vector output. Horizontal axis "INT8 model size, kB", logarithmic, from 1 to 10000, with labelled decade ticks. Vertical axis "reported accuracy or macro-F1, per cent", range chosen to fit the data and stated explicitly, not starting at zero. Plot one marker per published study. Marker fill encodes evaluation protocol: solid fill for studies that partitioned by patient, hollow with a 0.8 pt outline for studies that did not or did not report it. Marker area encodes energy per inference on a logarithmic area scale; studies with no reported energy are drawn at a fixed small size with an open centre and a light cross-hatch. All markers are #0072B2 except the point for this study, which is #D55E00 and one step larger. Label every point directly with first author and year at 6.5 pt in ink colour, using thin leader lines where labels would collide. Add two legends without frames: one for protocol showing a solid and a hollow marker, one for energy showing three reference bubble sizes with their values, and state in the energy legend how many points report no energy. Shade the upper-left region, small size and high accuracy, with a #F2F7FB rectangle behind the data, labelled "target region" at 7 pt. Do not draw any trend line, regression or frontier curve. Horizontal and vertical grid at 0.5 pt #E5E5E4 behind the data; left and bottom spines only. Sans-serif, tick labels 7 pt, axis titles 8 pt. No title inside the image, no shadow.

# Figure S4.1. Optimiser convergence

**Where.** Supplement S4. **Question.** Does any metaheuristic reach a better objective than gradient descent at an equal evaluation budget?

**Form.** A line chart, six series, faceted into two panels to stay within a validated palette: panel (a) the three metaphor-free optimisers plus gradient descent, panel (b) WHOA and GPC against gradient descent, which is repeated in both panels as the shared reference.

**Axes.** Horizontal, objective evaluations, 0 to 15000, linear. Vertical, the stage-one objective value, lower being better.

**Encoding.** Gradient descent `#5C5C5A` solid in both panels as the reference. Within each panel the remaining series take `#0072B2`, `#D55E00` and `#009E73` with distinct dash patterns. Each curve is the mean over 30 seeded runs with an interquartile band at 18 per cent opacity, and the caption states that the band is the interquartile range and not a confidence interval.

**Generation prompt.**

> Produce a two-panel line chart for a supplementary figure, 180 mm wide, vector output. Both panels share axes: horizontal "objective evaluations" from 0 to 15000, vertical "stage-one objective, lower is better". Panel (a) plots random search, CMA-ES and tree-structured Parzen estimation; panel (b) plots WHOA and GPC. Gradient descent appears in both panels as a shared reference in #5C5C5A solid at 1.2 pt. Within each panel the other series use #0072B2, #D55E00 and #009E73 with solid, dashed and dash-dot patterns respectively, at 1.2 pt. Each curve is the mean over 30 seeded runs, with an interquartile band in its own colour at 18 per cent opacity and no outline. Direct-label each curve at its right-hand end at 7 pt and add a per-panel legend without a frame. Panel letters (a) and (b) 9 pt bold outside the axes at the top left. Horizontal grid only, 0.5 pt #E5E5E4; left and bottom spines only. Sans-serif, tick labels 7 pt, axis titles 8 pt. No title inside the image.

# Figure S4.2. Receiver operating characteristic curves

**Where.** Supplement S4, as the secondary curve to Figure 3.

**Form.** Five small multiples, one class per panel, matching Figure 3's layout exactly so the two can be compared panel by panel.

**Encoding.** One curve per panel in `#0072B2`, the chance diagonal in `#9A9A98` dashed, area under the curve printed in the panel corner at 7 pt to two decimal places with its bootstrap interval.

**Generation prompt.**

> Produce a five-panel receiver operating characteristic figure for a supplementary section, 180 mm wide, vector output, panels in one row, one per class in the order N, S, V, F, Q, with the same panel geometry as the precision-recall figure. Both axes 0 to 1 with ticks every 0.2; horizontal "false positive rate", vertical "true positive rate"; axis titles on the bottom-left panel only. One curve per panel in #0072B2 at 1.2 pt with a 95 per cent bootstrap band at 18 per cent opacity. Draw the chance diagonal from (0,0) to (1,1) in #9A9A98 dashed 0.8 pt. In the lower right of each panel print the area under the curve to two decimals with its 95 per cent interval at 7 pt. Panel titles are the class letter and its support. Horizontal grid only, 0.5 pt #E5E5E4; left and bottom spines only. Sans-serif, tick labels 7 pt, axis titles 8 pt. No legend, no title inside the image.

# Figure S4.3. External confusion matrices

**Where.** Supplement S4. Superseded if panels (b) and (c) of Figure 2 are retained in the main text; include only if the journal limits Figure 2 to a single panel.

**Generation prompt.** Identical to Figure 2 with panels (a) INCART and (b) SVDB, one shared colour bar, and the same sequential ramp, cell text rule, zero-cell rule and S-into-N outline.

# Figure S4.4. Protocol contrast slopegraph, optional

**Where.** Supplement S4, supporting Table 4. This figure is not required by the manuscript and is proposed because the protocol contrast is the study's headline claim and currently has no visual.

**The question it answers.** Does the ordering of the seven arms survive the change from a patient-mixed partition to inter-patient partitioning?

**Form.** A slopegraph. Two vertical axes, one line per arm connecting its macro-F1 under each protocol. This form is chosen because the data's job is to show a change in rank, which a slopegraph encodes directly and a grouped bar chart hides.

**Encoding.** One line per arm, all in `#5C5C5A` at 1 pt, except any arm whose rank changes, which is drawn in `#D55E00` at 1.4 pt. Colour therefore encodes the finding and not the identity, and identity is carried by the labels at both ends. Each end carries the arm name and its macro-F1 to two decimals.

**Generation prompt.**

> Produce a slopegraph for a supplementary figure, 88 mm wide, vector output. Two vertical axes side by side, the left labelled "patient-mixed partition" and the right labelled "inter-patient partition, DS2", both showing macro-F1 on the same scale with the range stated. Draw one straight line per model arm connecting its value on the left axis to its value on the right axis, for seven arms labelled A, B, B2, C, D, E and F. Draw every line in #5C5C5A at 1 pt with 2.5 mm filled end markers, except lines whose rank position changes between the two axes, which are drawn in #D55E00 at 1.4 pt. At each end print the arm name and its macro-F1 to two decimals at 7 pt, nudging labels vertically to avoid collision. No grid, no horizontal axis, spines only where the two vertical axes are drawn, 0.6 pt #5C5C5A. Add one line of 7 pt text beneath the plot stating how many arms changed rank. Sans-serif throughout. No title inside the image, no legend, no shadow.

# Build notes

Every figure is regenerated by a command in supplement S5, so no figure is drawn by hand and none is edited after generation. Values printed inside a figure are read from the same result files that populate the tables, which is what keeps the two consistent under revision.

The colour choices above were validated with the palette validator in all-pairs mode against a white surface. Re-run that check after any substitution. The single result worth restating: five categorical hues do not survive all-pairs colour-vision checking, which is why no figure in this set encodes the five AAMI classes by colour.

