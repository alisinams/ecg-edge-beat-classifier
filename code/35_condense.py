"""Final condensation pass on the BSPC manuscript source.

BSPC asks a full paper to run to about 5,000 words. This pass shortens the
remaining over-budget paragraphs, keeping every claim, every reported figure and
every citation that carries an argument; what goes is restatement, justification
the reader does not need twice, and method detail the supplement already holds.
Run it once against the hand-written source; `34_wordcount.py` reports the
result.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "Biomedical Signal Processing and Control" / "manuscript_source.txt"

EDITS: list[tuple[str, str]] = [

("Intra-patient evaluation inflates reported performance",
 "Intra-patient evaluation inflates reported performance across much of this "
 "literature, including our own earlier two-stage classifier, which reported 98.58 % "
 "accuracy, 98.13 % sensitivity and 96.47 % specificity over seven rhythm types "
 "after splitting 3.9-second segments at random into 85 % for training and 15 % for "
 "testing, a partition placing segments from one recording on both sides of the "
 "boundary {#12}. Hybrid convolutional and recurrent designs report high accuracy "
 "on MIT-BIH under mixed partitions, and those figures are not comparable with the "
 "ones reported here {#30}. A smaller group works under the inter-patient "
 "constraint and reports the lower numbers that follow {#27,#28,#29}; that group is "
 "the correct comparison set for Section 5.2, and studies training one architecture "
 "under both regimes attribute the difference to the split rather than the model "
 "{#26}."),

("Each record was band-pass filtered between 0.5 and 40 Hz",
 "Each record was band-pass filtered between 0.5 and 40 Hz with a zero-phase "
 "fourth-order Butterworth filter, using NeuroKit2 {#77}. The upper corner sits "
 "below both mains frequencies, so one filter covers MIT-BIH and SVDB at 60 Hz and "
 "INCART at 50 Hz with no notch stage; it is the monitoring-mode bandwidth of "
 "clinical practice and discards high-frequency QRS content {#19}, a loss we accept "
 "because the target device samples at a wearable rate. Records were resampled to "
 "250 Hz, so SVDB, recorded at 128 Hz, is upsampled, which interpolates and "
 "restores nothing, and every SVDB result carries that caveat. Beats were then "
 "extracted as windows centred on the supplied R-peak annotation, spanning 250 ms "
 "before to 400 ms after the peak; using annotated positions separates "
 "classification from detection, so every accuracy reported here excludes detector "
 "error. Four rhythm descriptors accompany each window, namely the preceding and "
 "following RR intervals, the ratio of the preceding interval to the average of the "
 "preceding ten, and the ratio of the preceding to the following interval, and "
 "these make supraventricular ectopy separable at all. The classes span four orders "
 "of magnitude in the DS1 training subset, 35,492 N beats against 2 Q beats, so we "
 "use class-weighted cross-entropy with weights capped at a searched multiple of "
 "the largest, rather than the synthetic oversampling that would put beats "
 "interpolated from a test-set neighbour into training {#91}."),

("where the <i>w</i> terms are the cubic",
 "where the <i>w</i> terms are the cubic, quadratic and linear coefficients, <i>a</i> "
 "is the sigmoid slope, <i>b</i> its threshold and <i>z</i> the score for class "
 "<i>c</i>. Grey Wolf Optimization with a multinomial logistic wrapper selected 13 "
 "of a 36-descriptor pool {#62}, so stage one holds 205 parameters, fitted by WHOA "
 "as in the source. Arm C replaces stage two by argmax over the class scores, its "
 "stage one identical to arm B weight for weight and read from the same fitted "
 "file, so any difference between the arms is attributable to the decoder alone. "
 "Arm B2 keeps the fuzzy system and rebuilds its rule base over named clinical "
 "quantities, namely RR-interval irregularity, the ratio of the current RR interval "
 "to the local mean, P-wave presence and QRS duration, with membership parameters "
 "fitted by Giza Pyramids Construction {#61}. Because the source solves a different "
 "problem with the same machinery, arm B retargets rather than replicates it: the "
 "source classifies seven rhythm types over 3.9-second segments at 128 Hz, its WHOA "
 "run was 50,000 evaluations against the 15,000 used here for parity across "
 "optimisers, and its fuzzy stage carries one input per class. No figure reported "
 "for arm B is a reproduction of a figure reported in the source."),

("For each superclass we report sensitivity",
 "For each superclass we report sensitivity, positive predictive value, specificity "
 "and F1, all one-versus-rest; aggregate performance is overall accuracy, "
 "macro-averaged F1, Cohen's kappa and the Matthews correlation coefficient "
 "{#78,#79}. Confidence intervals are 95 % bootstrap percentile intervals from "
 "2,000 resamples taken at the level of the record, because beats within a record "
 "are not independent {#75}, and within each resample the metric is averaged over "
 "all five training seeds. A single temperature fitted on the DS1 validation subset "
 "calibrates the softmax outputs, scored by the Brier score and the expected "
 "calibration error over 15 bins {#73,#74}. Abstention uses split conformal "
 "prediction, with the nonconformity score one minus the calibrated probability of "
 "the true class {#18}; conformal coverage and retention, the fraction the "
 "classifier answers, are different quantities and reported under those names, and "
 "because the exchangeability the guarantee assumes fails on the external sets, "
 "their figures are empirical. A signal-quality index gates each beat at the second "
 "percentile of its DS1 training distribution {#72}, and noise robustness adds "
 "three artifact types from the MIT-BIH Noise Stress Test Database to DS2 at 18, "
 "12, 6 and 0 dB with synthetic mains interference, mixed into the raw record "
 "before the filter {#71}. Reporting follows TRIPOD+AI and was checked against "
 "PROBAST+AI {#10,#11}; the databases are public through PhysioNet {#4,#5}."),

("Table 3 reports per-class results for the student classifier",
 "Table 3 reports per-class results for the student classifier, each with a 95 % "
 "bootstrap percentile interval from 2,000 record-level resamples averaged over "
 "five training seeds. Aggregate figures are overall accuracy 83.6 (76.7 to 90.0) "
 "%, macro-F1 0.338 (0.288 to 0.459), Cohen's kappa 0.399 (0.228 to 0.580) and "
 "Matthews correlation coefficient 0.417 (0.245 to 0.600); accuracy is quoted only "
 "because the literature quotes it, since with this class distribution it sits "
 "close to the N-class sensitivity. V is separable on morphology and holds up at "
 "83.0 (73.5 to 91.8) % sensitivity, while S is the difficult class at 11.5 (3.1 to "
 "28.0) % sensitivity and 25.2 (7.5 to 59.6) % positive predictive value. F sits "
 "between V and N by construction and inherits the errors of both, and Q is too "
 "rare to support a stable estimate."),

("The S-class figure needs one qualification.",
 "The S-class figure needs one qualification. Of the 1,830 S beats in DS2, 1,376, "
 "or 75 %, come from a single record, 232, spread across only 16 records in total, "
 "so that record sets the S column almost by itself. Six of the seven arms fail on "
 "it: arm A reaches 57 % S sensitivity on the remaining records and 0.9 % on record "
 "232. Arm B2, the clinical-quantity fuzzy system, does not, reaching 94 % on "
 "record 232 and 92 % across DS2, the highest of any arm by a wide margin, and "
 "paying for it with a macro-F1 of 0.254 (0.195 to 0.352). The descriptors a "
 "clinician would name are the ones carrying this record, and a macro-average hides "
 "the trade."),

("The arm ordering, given in the inter-patient column",
 "The arm ordering, given in the inter-patient column of Table 4, is not the one "
 "the design anticipated: the highest macro-F1 belongs to arm A, the best classical "
 "arm, at 0.408 (0.329 to 0.578), while the student ranks four of seven at 0.338 "
 "(0.288 to 0.459). Fig. 2 gives the confusion matrices, where the S row rather "
 "than the diagonal is informative: across all three databases the dominant "
 "off-diagonal mass sits at S misread as N, the error a timing-based class makes "
 "when the timing baseline belongs to somebody else. Fig. 3 gives the "
 "precision-recall curves, primary here because the class distribution is skewed by "
 "more than two orders of magnitude."),

("Table 4 scores every arm twice on the same beats",
 "Table 4 scores every arm twice on the same beats: once trained on DS1 and tested "
 "on DS2, and once refitted and tested under the patient-mixed partition. Every arm "
 "improved under the mixed partition, by 0.204 macro-F1 on average and from 0.058 "
 "for arm C to 0.340 for arm D, and no arm changed rank. That the ordering is "
 "identical under the two protocols is a result this study did not expect and "
 "reports against its own hypothesis. The non-uniformity is what survives, and it "
 "is enough: the largest gain is 5.9 times the smallest, and the arms gaining most "
 "are those with the capacity to memorise a recording, against arm C, whose first "
 "stage holds 205 parameters and cannot memorise anything {#7,#26}. A difference "
 "between two published patient-mixed numbers therefore cannot be read as a "
 "difference in behaviour on a new wearer unless the two models have similar "
 "capacity. Fig. 4 draws the two columns against each other."),

("On the first-stage objective, gradient descent reaches",
 "On the first-stage objective, gradient descent reaches a mean final value of "
 "0.01283 (SD 0.00016), against 0.14352 (0.04852) for WHOA, 0.20241 (0.06505) for "
 "GPC, 0.08382 (0.01303) for tree-structured Parzen estimation, 0.10978 (0.01987) "
 "for CMA-ES and 2.34190 (0.34002) for random search. Every comparison against "
 "gradient descent is significant at p < 0.0001 by Wilcoxon signed-rank with Holm "
 "correction, rank-biserial correlation +1.000, and the Friedman statistic is 143.9 "
 "on 5 degrees of freedom, p < 0.0001. WHOA leaves the objective 11 times higher on "
 "the same parameters, data and evaluation count, and its standard deviation across "
 "seeds is 296 times larger, so the arm B row of Table 4 measures what that "
 "optimiser does and not what the architecture can do. Refitting those parameters "
 "by gradient descent at the identical budget, decoded by the identical fuzzy "
 "stage, moves DS2 macro-F1 from 0.087 to 0.284, a change of +0.197 (+0.143 to "
 "+0.303). The optimiser table and convergence curves are supplementary."),

("Table 6 profiles the student classifier",
 "Table 6 profiles the student classifier for the STM32F446RE. The model fits the "
 "part with room to spare, 4.5 kB of weights against 512.0 kB of flash and 8.5 kB "
 "of peak static RAM against 128.0 kB, so the deployment claim does not depend on "
 "the accuracy of the latency model, and the gap between the mean and the 99th "
 "percentile latency comes from flash wait states and cache behaviour rather than "
 "from the model. Fig. 5 places this budget against the published anchors, obtained "
 "on different hardware and under different partitions, so the frontier it draws is "
 "indicative; the region it leaves almost empty is the point, since no study in "
 "this set, this one included, occupies the sub-20 kB, above-95 % corner with a "
 "five-class AAMI result obtained between patients and a complete deployment "
 "profile attached."),

("Before temperature scaling the classifier is under-confident",
 "Before temperature scaling the classifier is under-confident, with a mean "
 "confidence of 76.8 % against an accuracy of 88.4 % and an expected calibration "
 "error of 0.116; a single fitted temperature of 0.70 brings that error to 0.055 "
 "and the Brier score from 0.206 to 0.192. Against a nominal 90 %, conformal "
 "coverage is 89.4 % on DS2, 92.3 % on INCART and 89.5 % on SVDB, while retention "
 "is higher at 96.2 %, 95.4 % and 96.6 %; abstaining on 3.8 % of DS2 beats buys "
 "+0.006 macro-F1 on those that remain, so the rule is worth having because it "
 "bounds what the device claims to know, not because it improves the numbers. Under "
 "added noise at 0 dB, the hardest level tested, baseline wander costs 0.0016 "
 "macro-F1 and mains interference -0.0008, close enough to nothing that the filter "
 "corners can be said to have done their job, while muscle artifact costs 0.0097 "
 "and electrode motion 0.0454, 28 times the baseline-wander figure, because its "
 "spectrum overlaps the QRS complex and no linear filter separates the two. Alarm "
 "burden on DS2 at the abstention operating point is 3230 false ventricular and "
 "1066 false supraventricular alarms per 24 hours, against 3887 and 1114 without."),

("The result the design did not anticipate is the arm ordering.",
 "The result the design did not anticipate is the arm ordering. Arm A takes the "
 "highest inter-patient macro-F1 and the distilled student ranks four of seven, the "
 "distillation difference being -0.005 with an interval containing zero, so on this "
 "evidence the foundation-model teacher contributed nothing measurable and the "
 "paper does not claim otherwise. Under inter-patient partitioning on this "
 "database, gradient-boosted trees on hand-built descriptors are not a weak "
 "baseline to be cleared but the arm to beat; that arm also does not fit the "
 "device, serialising to 2015 kB against 512 kB of flash, which is why both numbers "
 "are reported rather than collapsed into one ranking. Either a 30-million-parameter "
 "model pretrained on twelve-lead corpora has little to transfer to a single-lead "
 "beat window, or the adaptation required to feed it one lead in twelve slots "
 "destroys most of what it had. Either way, a comparison of this kind belongs in "
 "any paper claiming a distillation benefit, because the benefit here is smaller "
 "than the gap between two ordinary arms."),

("Two further findings concern the reimplemented architecture",
 "Two further findings concern the reimplemented architecture rather than the "
 "proposed one. With stage one held identical, its fuzzy decoder differed from "
 "argmax by -0.000 macro-F1, the rule base being a one-hot mapping from class "
 "scores to classes, which is what argmax computes, so a second stage costing "
 "parameters, memory and an optimiser run did not earn them. Rebuilt over named "
 "clinical quantities the same formalism scores higher, by +0.167, so the "
 "interpretable formalism was not what cost accuracy, the features it was given "
 "were. Separately, refitting the same 205 parameters by gradient descent moves "
 "macro-F1 by +0.197, so most of what looked like a limitation of that architecture "
 "was a limitation of the metaheuristic used to fit it: an architecture evaluated "
 "only under the optimiser its authors chose has not been evaluated."),

("Against the inter-patient group the comparison does not flatter",
 "Against the inter-patient group the comparison does not flatter this work. Two of "
 "those studies report five-class results on the same partition of the same "
 "database and the classifier reported here sits below both {#27,#28}; the other "
 "two are not five-class studies, one dropping the Q superclass {#42} and one "
 "detecting premature ventricular contractions in two classes {#29}. Three "
 "differences account for part of the gap and none excuses it: dropping or merging "
 "classes raises a macro average that this paper computes over five including one "
 "with 7 test beats; every hyperparameter here is fitted on a validation subset "
 "carved from DS1, so no threshold is tuned on the test set; and the "
 "supraventricular result is dominated by a single record. The contribution of this "
 "work is the completeness of the reporting, not the height of the accuracy. "
 "Against the edge group several report smaller models, and Mommen and colleagues "
 "report an energy six orders of magnitude below the figure computed here, on "
 "programmable logic and not a microcontroller, which is a statement about the "
 "substrate {#42}; the same applies to our own earlier system, whose 0.32 ms per "
 "classification was obtained on a Raspberry Pi, with megabytes of memory rather "
 "than the kilobytes budgeted in Table 6 {#12}. What none of the seven anchors "
 "reports is inter-patient accuracy, model size, peak memory, latency percentiles, "
 "energy and external validation together {#36,#37,#38,#40,#41}."),

("A missed ventricular ectopic beat and a missed supraventricular",
 "A missed ventricular ectopic beat and a missed supraventricular ectopic beat "
 "carry different costs: sustained ventricular ectopy can precede an arrest, while "
 "isolated supraventricular ectopy is common and often benign, though its burden "
 "carries prognostic information for atrial fibrillation. The classifier's weakest "
 "class is S, which is tolerable for a device whose alarm is ventricular and "
 "unacceptable for one whose purpose is atrial fibrillation screening, and 3230 "
 "false ventricular alarms per day is far above what a consumer device could ship "
 "with. That the supraventricular class is the hard one is not a property of this "
 "classifier alone: the architecture reimplemented as arm B named a "
 "supraventricular rhythm, the class for which it held fewest examples, as its own "
 "worst, under a far more forgiving protocol {#12}. The intended use is therefore a "
 "screening aid that flags beats for review, not a diagnostic device and not an "
 "autonomous alarm. Software of this kind inside a device regulated under the "
 "Medical Devices Regulation falls under the high-risk classification of the "
 "Artificial Intelligence Act, which adds data governance, record-keeping, "
 "transparency, human oversight and post-market monitoring obligations on top of "
 "the existing conformity assessment {#80,#81}, with guidance on how the two "
 "interact issued as MDCG 2025-6 {#82}; Regulation (EU) 2026/1744 defers those "
 "obligations for artificial intelligence embedded in Annex I products, which "
 "include medical devices, to 2 August 2028 {#88}. Classifying on the device means "
 "the waveform need not leave it, and the channel by which a device reports its "
 "decisions is a second exposure: work on anomaly detection in Internet-of-Things "
 "networks has coupled federated aggregation to a blockchain-anchored record of "
 "model updates so that no participant surrenders raw data and every update stays "
 "auditable {#94}. Lifecycle obligations are set out for this domain in a recent "
 "statement {#86}."),

("Six limitations bound every claim above.",
 "Six limitations bound every claim above. The deployment budget is computed with "
 "no board metered, so the throughput assumption is the largest single source of "
 "uncertainty in Table 6. R-peak positions are taken from the supplied annotations, "
 "so no reported accuracy includes detector error. MIT-BIH carries the recording "
 "conditions of 1975 to 1979 and no race or ethnicity metadata, and with 22 records "
 "in DS2 the subgroup intervals by sex and age band all overlap. The Q superclass "
 "has 7 beats in DS2 and supports no conclusion. There is no ventricular "
 "fibrillation class, because the AAMI beat-level mapping does not define one. And "
 "no clinician has yet seen an output from this classifier inside a care pathway. "
 "Four experiments follow directly: meter the budget on an actual STM32F446, since "
 "the difference between computed and measured is itself the reportable result; "
 "aggregate beat-level decisions into rhythm-level events and report false events "
 "per 24 hours instead of false beats; personalise on device by adapting to the "
 "first hour of a wearer's own beats, with S-class sensitivity as the outcome and "
 "the added SRAM and energy as the cost; and test the federated path with clients "
 "defined by subject and a differential privacy budget stated explicitly "
 "{#51,#52,#55}, an arrangement already assembled outside the cardiac domain under "
 "exactly the constraints a wearable ECG deployment works under {#93}."),
]


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    for head, replacement in EDITS:
        start = text.index(head)
        end = text.index("\n@", start)
        text = text[:start] + replacement + text[end:]
        print(f"condensed: {head[:58]}")
    SOURCE.write_text(text, encoding="utf-8")
    for bad in ("—", "–"):
        assert bad not in text, f"dash character U+{ord(bad):04X} present"
    print(f"\nwrote {SOURCE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
