"""Manuscript section 6: Methods."""
from __future__ import annotations

from fmt import num, thousands


def build(C) -> str:
    hp, t2, t8, teacher = C.hp, C.t2, C.t8, C.teacher
    fs = hp.get("featsel", {})
    sel = fs.get("selected", [])
    n_rhythm = sum(1 for i in sel if i < 4)
    n_morph = len(sel) - n_rhythm
    tr_row, va_row, ds2_row = C.tr_row, C.va_row, C.ds2_row
    opt = C.opt_setup
    dev, asu = C.dev, C.assump
    trials = {a: hp.get(a, {}).get("n_trials", 0) for a in ("A", "B", "B2", "D", "E", "F")}
    teach_cov = 100.0 * float(hp.get("E", {}).get("teacher_coverage", 0.0) or 0.0)
    val_list = ", ".join(C.splits["ds1_val"])

    return f"""
# 6. Methods

## 6.1 Datasets and inter-patient partitioning

Development used the MIT-BIH Arrhythmia Database {{Moody, 2001 #3; Goldberger, 2000 #4}}: 48 half-hour two-lead recordings from 47 subjects, digitised at 360 Hz, collected at the Beth Israel Hospital arrhythmia laboratory between 1975 and 1979. Modified limb lead II was taken as the single channel, since it is the lead most nearly comparable to a chest-worn wearable placement.

The four paced records, 102, 104, 107 and 217, were excluded as required by ANSI/AAMI EC57 {{Association for the Advancement of Medical Instrumentation, 2012 #6}}. The remaining 44 records were split by the de Chazal partition {{deChazal, 2004 #7}}. DS1, used for training, comprises records 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223 and 230. DS2, used for testing, comprises records 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233 and 234.

Model selection needs a held-out set that is not DS2, so DS1 was split by record into a training and a validation subset under three constraints fixed in advance and never revisited. Every AAMI superclass present in DS1 must appear in both subsets, which forces one of records 101, 203 and 208 into the validation subset, because those three are the only DS1 records holding a Q beat. The validation subset must hold roughly a fifth of the DS1 beats. And both bundle-branch-block records of DS1, 118 and 124, must stay in the training subset: a right-bundle-branch-block beat carries the label N and a wide QRS complex, so a classifier that has never seen one calls it ventricular, and a validation subset that holds both of them measures their absence from training, not the model. The resulting validation subset is records {val_list}, {thousands(va_row['Total'])} beats against {thousands(tr_row['Total'])} for training. DS2 was untouched until every architectural choice, hyperparameter, calibration temperature and abstention threshold had been fixed on the DS1 validation subset. It was then evaluated once, and no DS2 result fed back into any modelling decision.

Beat labels were mapped to the five AAMI superclasses: normal (N), supraventricular ectopic (S), ventricular ectopic (V), fusion (F) and unclassifiable or paced (Q). The mapping from annotation symbol to superclass follows EC57 and is reproduced in supplement S3 {{Association for the Advancement of Medical Instrumentation, 2012 #6}}.

External validation used the St. Petersburg INCART 12-lead Arrhythmia Database, from which lead II was taken, and the MIT-BIH Supraventricular Arrhythmia Database (SVDB). Both carry beat-level annotations that map onto the same five superclasses without a change of labelling convention, and both were used for inference only, with no weight, threshold or calibration parameter refitted. Twelve-lead rhythm-level corpora such as PTB-XL attach one label to a whole recording, so a beat classifier evaluated on them would need a label mapping that confounds the measurement {{Wagner, 2020 #68}}.

A second partition of the same beats was built for the protocol contrast in section 7.3: all beats from all 44 records pooled and split at random in the DS1 to DS2 proportion, so that a subject appears on both sides. Every arm was refitted from scratch on that partition, with the same hyperparameters and the same five seeds, and it is used for that comparison alone.

## 6.2 Preprocessing

Each record was band-pass filtered between 0.5 and 40 Hz with a zero-phase fourth-order Butterworth filter. The lower corner removes baseline wander from respiration and electrode movement. The upper corner sits below both mains frequencies, which matters because MIT-BIH and SVDB were recorded in the United States at 60 Hz and INCART in Russia at 50 Hz, so one filter covers all three databases and no notch stage is needed. That 40 Hz corner is the monitoring-mode bandwidth of clinical practice, well below the 150 Hz diagnostic bandwidth, and it discards high-frequency QRS content carrying information about intraventricular conduction {{Kligfield, 2007 #19}}. We accept the loss because the target device samples at a wearable rate; supplement S3 repeats the experiment at a 100 Hz corner so the cost is visible.

All records were resampled to 250 Hz by polyphase filtering. MIT-BIH and INCART are downsampled, from 360 and 257 Hz. SVDB is recorded at 128 Hz and is upsampled, which interpolates and restores nothing, and every SVDB result in section 7.4 carries that caveat. Each record was then normalised by its own median and median absolute deviation, a statistic of the input signal that involves no label and no partition.

## 6.3 Segmentation and labelling

Beats were extracted as windows centred on the R-peak annotation supplied with each database, spanning 250 ms before to 400 ms after the peak, giving 163 samples at 250 Hz. Using annotated R-peak positions separates the classification question from the detection question, and it means every accuracy reported here excludes detector error. Section 8.5 states that assumption plainly.

Four rhythm descriptors accompany each window: the preceding RR interval, the following RR interval, the ratio of the preceding interval to the average of the preceding ten, and the ratio of the preceding to the following interval. These make supraventricular ectopy separable at all, since its morphology closely resembles a normal beat. Where fewer than ten preceding beats exist the average is taken over those available, and the first ten beats of every record are excluded from all partitions.

No window crosses a record boundary and no record contributes beats to more than one partition, so no window overlaps a partition boundary by construction. Windows within a record do overlap each other when consecutive beats fall closer than 650 ms apart, which is a property of the rhythm; because whole records are assigned to partitions, that overlap can never span the training and test boundary.

## 6.4 Class imbalance

The five superclasses are unbalanced, and in the DS1 training subset the imbalance spans four orders of magnitude: {thousands(tr_row['N'])} N beats against {tr_row['Q']} Q beats. We use class-weighted cross-entropy, with weights set to inverse class frequency in the DS1 training subset, capped at a stated multiple of the weight of the most frequent class, and normalised to sum to the number of classes. The cap is not cosmetic. An uncapped inverse-frequency weight would make the {tr_row['Q']} Q beats of the training subset count for more than its {thousands(tr_row['N'])} normal beats put together, and the network would spend its capacity on a class whose sensitivity cannot be estimated from the {ds2_row['Q']} Q beats of DS2. The cap was searched over 2 to 30 for the neural arms as a hyperparameter, on the DS1 validation subset, and the selected value per arm is in supplement S3. Weights are computed from the training subset alone, so nothing from the validation subset or from DS2 enters them.

We avoid synthetic minority oversampling in the main pipeline. Applied before partitioning it puts interpolated beats derived from a test-set neighbour into training, a leak of the same kind inter-patient partitioning exists to prevent {{Chawla, 2002 #91}}. Section 7.5 reports one arm oversampled correctly, after partitioning and inside the training subset only, as an ablation.

## 6.5 Model architectures

Seven arms were trained and evaluated under the identical protocol of sections 6.1 to 6.4 and 6.10.

Arm A holds classical baselines on the descriptors of section 6.3 plus 32 morphology samples taken uniformly from the window: logistic regression, linear and radial-basis-function support vector machines, a random forest, and gradient-boosted trees {{Chen, 2016 #23}}. The arm reports whichever of the five scores highest on the DS1 validation subset. The radial-basis-function fit is bounded in iterations and fitted on a stratified subsample, because at the top of the searched regularisation range an unbounded fit does not terminate in a usable time.

Arm B is a reimplementation of the published two-stage classifier {{Abbaszadeh, 2024 #12}}. Its first stage scores each class with a cubic polynomial over the selected features, squashed by a fitted sigmoid. For class c and feature vector x with D components,

$$ s_c(\\mathbf{{x}}) = \\sum_{{j=1}}^{{D}} \\left( w^{{(3)}}_{{c,j}} x_j^3 + w^{{(2)}}_{{c,j}} x_j^2 + w^{{(1)}}_{{c,j}} x_j \\right) \\tag{{1}} $$

$$ z_c(\\mathbf{{x}}) = \\frac{{1}}{{1 + \\exp\\left(-a_c \\left( s_c(\\mathbf{{x}}) - b_c \\right)\\right)}} \\tag{{2}} $$

where $w^{{(3)}}_{{c,j}}$, $w^{{(2)}}_{{c,j}}$ and $w^{{(1)}}_{{c,j}}$ are the cubic, quadratic and linear coefficients for class $c$ and feature $j$; $a_c$ is the sigmoid slope; $b_c$ is its threshold; and $z_c \\in (0,1)$ is the score for class $c$. Features were selected from the 36-descriptor pool by Grey Wolf Optimization with a multinomial logistic wrapper {{Mirjalili, 2014 #62}}, which chose {n_rhythm} of the four rhythm descriptors and {n_morph} of the 32 morphology samples; the selected indices are listed in supplement S3. With $D = 13$ selected features and five classes, stage one holds $5 \\times (3D + 2) = 205$ parameters, and it is fitted by WHOA as in the source architecture. Stage two is a Takagi-Sugeno fuzzy inference system taking the five scores as inputs, with two Gaussian membership functions per input whose widths are optimised, and a rule base in which rule $c$ fires when input $c$ is high and the rest are low.

Arm C is arm B with stage two deleted and replaced by

$$ \\hat{{y}}(\\mathbf{{x}}) = \\arg\\max_{{c}} z_c(\\mathbf{{x}}) \\tag{{3}} $$

Stage one is identical between B and C, weight for weight: the fitted parameter vector is written to one file per seed and both decoders read that same file, so any difference between the two arms is attributable to the decoder alone. This is the experiment that decides whether the fuzzy stage of the source architecture does anything.

Arm B2 keeps the fuzzy inference system and rebuilds its rule base over named clinical quantities: RR-interval irregularity, the ratio of the current RR interval to the local mean over the preceding ten beats, P-wave presence measured as energy in the 150 ms before QRS onset relative to the local baseline, and QRS duration measured as the width of the high-slope region around the R peak. Each input carries low, mid and high Gaussian membership functions whose centres and widths are fitted by Giza Pyramids Construction {{Harifi, 2021 #61}}. Rules take the form "if RR irregularity is high and P-wave presence is low then supraventricular ectopic", and the learned base is printed in supplement S6. Accuracy is not the contribution of this arm; whether its rules read as diagnostic criteria is.

Arm D is a compact one-dimensional convolutional network of depthwise-separable blocks with strided pooling, followed by global average pooling, concatenation with the four rhythm descriptors, and a linear layer {{Howard, 2017 #34}}. This is the workhorse comparator.

Arm E is the proposed student classifier. The teacher is HuBERT-ECG, a public ECG foundation model of {thousands(teacher.get('parameters', 0))} parameters pretrained by masked-unit prediction on large public corpora and released with open weights {{Coppola, 2024 #87}}. The teacher expects five seconds of twelve-lead signal at 100 Hz flattened into one sequence; this study has one lead, so the single available lead was written into every one of the twelve input slots and the whole network was fine-tuned on the DS1 training subset, which lets it adapt to that convention. Two changes were made to the pretrained backbone before fine-tuning. Transformer frames are pooled by a single learned attention query instead of averaged, because the label belongs to the beat at the centre of the five-second window and a flat average dilutes it. And the classification head also receives the four rhythm descriptors of section 6.3, the same four the student receives, so that the soft targets are not systematically worse than the student on the timing-driven S class. The teacher reached a DS1 validation selection score of {num(teacher.get('val_selection_f1', float('nan')))}, it supplied soft targets for {num(teach_cov, 1)} per cent of the DS1 training beats, and it never saw a DS2 beat.

The student shares arm D's topology at reduced width and is trained with the distillation objective

$$ \\mathcal{{L}} = \\alpha \\, \\mathcal{{L}}_{{\\mathrm{{CE}}}}\\big(y, \\sigma(\\mathbf{{u}}_s)\\big) + (1 - \\alpha) \\, T^2 \\, \\mathrm{{KL}}\\!\\left( \\sigma(\\mathbf{{u}}_t / T) \\, \\big\\| \\, \\sigma(\\mathbf{{u}}_s / T) \\right) \\tag{{4}} $$

where $\\mathbf{{u}}_s$ and $\\mathbf{{u}}_t$ are the student and teacher logit vectors, $\\sigma$ is the softmax function, $y$ is the class-weighted target, $T$ is the distillation temperature, $\\alpha \\in [0,1]$ balances the two terms, and $\\mathcal{{L}}_{{\\mathrm{{CE}}}}$ is class-weighted cross-entropy {{Hinton, 2015 #13}}. The $T^2$ factor keeps the gradient magnitudes of the two terms comparable as $T$ varies.

Arm F is a temporal-convolution student of matched parameter count with dilated kernels, included so that the distillation result is not confounded with the choice of student topology.

Figure 1 shows the pipeline from raw record to class decision, with the partition boundary drawn explicitly.

**Figure 1.** Processing pipeline from raw record to class decision, with the partition boundary drawn as a vertical wall across the diagram. Records enter at the left, are filtered to 0.5 to 40 Hz and resampled to 250 Hz, and are then assigned whole to one partition; every arrow crossing the wall carries a trained parameter, and none carries a beat. The four rhythm descriptors bypass the convolutional stack and join the classifier at the linear layer, which is why the S class depends on them. Downstream of the classifier the calibration temperature and the conformal threshold are fitted on the DS1 validation subset alone, so the abstention rule that operates on DS2 was never tuned on it. The teacher network sits on the training side of the wall only. The figure is included because most reported leakage in this field is invisible in prose and obvious in a diagram of this kind: a pipeline that cannot be drawn without an arrow crossing the wall is a pipeline that leaks.

## 6.6 Training protocol

Networks were trained in PyTorch {{Paszke, 2019 #76}} with the AdamW optimiser, batch size 256, cosine-decayed learning rate, for at most 200 epochs. Training stopped when the validation selection score failed to improve for 20 consecutive epochs, and the checkpoint with the best validation selection score was retained. Every arm was trained with five seeds, 0 through 4, and section 7 reports the mean across seeds with its bootstrap interval; the seed is fixed for data ordering, weight initialisation and augmentation.

Four label-preserving augmentations were applied to the training beats and to none of the evaluation beats: a log-uniform amplitude gain, a sub-beat time shift, additive white noise together with a slow sinusoidal baseline drift, and a small multiplicative jitter on the rhythm descriptors. Seventeen training records is a small number of subjects, and a network free to memorise their gain, their electrode contact and their exact sampling phase will do so; each transform removes one such shortcut without changing the class of a beat. The strength of each was searched as a hyperparameter and the selected values are in supplement S3.

Model selection on the DS1 validation subset used macro-F1 over the N, S and V superclasses. F and Q were excluded from the selection criterion, and from that criterion only, because the validation subset holds {va_row['F']} F beats and {va_row['Q']} Q beats, so their F1 is sampling noise and would decide model selection by coin flip. Both classes are reported in full on DS2, where they are not the selection signal. Every reported result in section 7 is a five-class macro average unless the text says otherwise.

Hyperparameters were searched with tree-structured Parzen estimation, scored on the DS1 validation selection score, with signal preprocessing performed in NeuroKit2 {{Bergstra, 2011 #65; Akiba, 2019 #66; Makowski, 2021 #77}}. The trial budget was {trials.get('D', 0)} for arm D, {trials.get('E', 0)} for arm E, {trials.get('B', 0)} for arm B, {trials.get('A', 0)} for arm A, {trials.get('B2', 0)} for arm B2 and {trials.get('F', 0)} for arm F. The three reductions are wall-clock decisions and are reported because they are part of the protocol: one arm A trial fits five classifiers, one arm B2 trial runs a 15,000-evaluation optimiser, and arm F is a dilated temporal-convolution network that does not downsample, so one of its trials costs more than an order of magnitude more arithmetic than one arm D trial. Neural trials were scored after a shortened 60-epoch schedule with a patience of 8; the winning configuration was then retrained to convergence with five seeds. Search spaces, selected values, training hardware and wall-clock cost per arm are in supplement S3.

## 6.7 Optimiser comparison and statistical testing

The objective of arm B's first stage, equations (1) and (2), is differentiable in every parameter, so a metaheuristic is not required to fit it. We therefore fit the same 205 parameters six ways under an identical budget of {thousands(opt.get('budget', 15000))} objective evaluations inside an identical box: WHOA {{Naruei, 2022 #60}}, GPC {{Harifi, 2021 #61}}, random search {{Bergstra, 2012 #63}}, CMA-ES {{Hansen, 2001 #64}}, tree-structured Parzen estimation {{Bergstra, 2011 #65}} and projected Adam gradient descent. The objective is class-weighted one-versus-rest binary cross-entropy on the five sigmoid scores, evaluated on a fixed stratified subsample of {thousands(opt.get('subsample_beats', 0))} DS1 training beats drawn once with a fixed seed, identical for every optimiser and every seed, so that 180 runs of that budget are affordable. Each optimiser was run 30 times with seeds 0 through 29. Population-based methods evaluate a whole population in one batched call, so the budget is counted in candidate evaluations and not in generations. One Adam step consumes one evaluation of the budget; a step also needs a backward pass, which the metaheuristics do not pay for, and the wall-clock column of Table 7 makes that asymmetry visible. We report the mean and standard deviation of the final objective, the best of 30, the convergence curve and wall-clock cost.

The comparison was fixed before the runs. Pairwise differences are tested with the Wilcoxon signed-rank test across the 30 paired runs; the six-way comparison uses the Friedman test with Nemenyi post-hoc analysis {{Demsar, 2006 #67}}. Effect sizes are reported as the rank-biserial correlation for pairwise tests and as the average rank difference for the omnibus test. The significance level is 0.05 with Holm correction across the pairwise family.

## 6.8 Quantisation and the deployment budget

Arm E was quantised to INT8 with quantisation-aware training: fake-quantisation nodes were inserted after the trained full-precision checkpoint and the network was fine-tuned for a further 40 epochs, with per-channel symmetric quantisation of weights and per-tensor asymmetric quantisation of activations {{Jacob, 2018 #15}}. For a real value $r$ mapped to an integer $q$,

$$ q = \\mathrm{{clip}}\\!\\left( \\mathrm{{round}}\\!\\left( \\frac{{r}}{{S}} \\right) + Z, \\; -128, \\; 127 \\right), \\qquad S = \\frac{{r_{{\\max}} - r_{{\\min}}}}{{255}} \\tag{{5}} $$

where $S$ is the scale, $Z$ is the zero point, and $r_{{\\max}}$ and $r_{{\\min}}$ are the calibration range observed on the DS1 training subset only.

No STM32F446 board was available for this study, and the consequence is stated here before any number is quoted: **the deployment profile in section 7.8 is computed, not metered**. Four of its rows are measurements of the model itself and are exact: the parameter count from the model definition, the multiply-accumulate count from the layer shapes, the INT8 and FP32 model size from the exported tensors, and the macro-F1 change from FP32 to INT8 on DS2. Three rows are computed from a stated model of the target device. Peak static RAM is the largest pair of consecutive activation tensors under a two-buffer arena, plus the CMSIS-NN im2col scratch and the input and output buffers. Latency is the multiply-accumulate count divided by an assumed CMSIS-NN throughput of {asu.get('cmsis_macs_per_cycle', 0.8)} INT8 multiply-accumulates per cycle on a Cortex-M4 with the DSP extension, plus a fixed {int(asu.get('per_layer_overhead_cycles', 900))} cycles of per-layer overhead, at the {dev.get('clock_mhz', 180)} MHz core clock. Energy per inference follows from that latency, the supply voltage and the datasheet active-mode current of {dev.get('active_current_ma', 38)} mA:

$$ E_{{\\mathrm{{inf}}}} = V_{{DD}} \\cdot I_{{\\mathrm{{active}}}} \\cdot t_{{\\mathrm{{inf}}}} \\tag{{6}} $$

Table 8 labels every row measured or computed, no computed quantity is described anywhere in this paper as a measurement, and the assumptions are printed inside the table so that a reader with a board can check them against hardware. Projected battery life follows from $E_{{\\mathrm{{inf}}}}$, a beat rate of {int(asu.get('beats_per_minute', 75))} per minute, one inference per beat and a {int(asu.get('cell_mah', 100))} mAh cell, and is a projection under those assumptions; no cell was discharged to check it. Section 8.6 lists metering this profile as the first piece of future work, because the difference between the computed and the metered figure is itself worth reporting.

## 6.9 Calibration and abstention

Softmax outputs of a trained network are not calibrated probabilities, so we fit a single temperature $T_{{\\mathrm{{cal}}}}$ by minimising negative log-likelihood on the DS1 validation subset, rescale the logits as $\\sigma(\\mathbf{{u}}/T_{{\\mathrm{{cal}}}})$, and report the reliability diagram, the Brier score and the expected calibration error over 15 equal-width bins {{Guo, 2017 #73; Pakdaman Naeini, 2015 #74}}.

Abstention uses split conformal prediction {{Angelopoulos, 2021 #18}}. The nonconformity score for a calibration beat with true class $y$ is $1 - \\hat{{p}}_y$, where $\\hat{{p}}_y$ is the calibrated probability of that class. For target coverage $1 - \\varepsilon$ the threshold $\\hat{{q}}$ is the $\\lceil (n+1)(1-\\varepsilon) \\rceil / n$ empirical quantile of the calibration scores, the prediction set contains every class whose calibrated probability exceeds $1 - \\hat{{q}}$, and the classifier abstains when that set has more than one member. Two quantities follow from that rule and they are not the same. Conformal coverage is the fraction of beats whose prediction set contains the true class, and it is what the guarantee bounds at $1 - \\varepsilon$. Retention is the fraction of beats whose set holds at most one class, so the classifier answers instead of abstaining. Section 7.7 reports both under those names. The guarantee assumes exchangeability between calibration and test data. It holds within DS2 and fails on INCART and SVDB, where equipment and population differ, so both quantities on the external sets are reported as empirical values with no guarantee attached.

One consequence of reporting a five-seed ensemble has to be stated here in the methods, not left for the reader to discover in the results. Temperature scaling is monotone and cannot change the argmax of a single model. The classifier reported in this paper is the mean of five seeds' probabilities and the temperature is applied to each seed before the mean is taken, so the ensemble decision is not a monotone function of the unscaled one and a small number of decisions do move. Section 7.7 reports how many. Table 3 is computed without temperature scaling and the abstention rule with it, and each is labelled at the point of use.

Signal-quality gating precedes classification. Each window receives a quality index combining flatline fraction, the ratio of in-band to out-of-band power, and agreement between the annotated R peak and a slope-based redetection inside the window; windows below a threshold fixed at the second percentile of the index on the DS1 training beats are rejected before the classifier sees them, and the rejection rate is reported for every dataset {{Clifford, 2012 #72}}.

## 6.10 Evaluation metrics and confidence intervals

For each AAMI superclass we report sensitivity, positive predictive value, specificity, F1 and false positive rate, all one-vs-rest. Aggregate performance is overall accuracy, macro-averaged F1, Cohen's kappa and the Matthews correlation coefficient {{Cohen, 1960 #78; Matthews, 1975 #79}}. Every aggregate figure states its averaging convention at the point of use, and no bare accuracy appears anywhere in this paper.

Confidence intervals are 95 per cent bootstrap percentile intervals from 2,000 resamples, taken at the level of the record, because beats within a record are not independent {{Efron, 1993 #75}}. Within each resample the metric is computed for all five training seeds and averaged, so one interval carries both the record sampling and the seed variation, and the point estimate is the same average on the unresampled data. That choice widens the intervals. Every metric here is a function of the confusion matrix, so a resample sums per-record confusion matrices instead of recomputing over resampled beats, which makes the interval exact at this resample count instead of approximate. Precision-recall curves are primary because the class distribution is skewed, and receiver operating characteristic curves appear in supplement S4.

Noise robustness was assessed by adding baseline wander, muscle artifact and electrode-motion artifact from the MIT-BIH Noise Stress Test Database to the DS2 records at signal-to-noise ratios of 18, 12, 6 and 0 dB, with synthetic 50 and 60 Hz interference at the same levels {{Moody, 1990 #71}}. Noise is mixed into the raw record at its native sampling rate, before the band-pass filter, and the whole chain of section 6.2 is then rerun on the contaminated record. Adding it to the extracted windows instead would place the contamination downstream of the filter that is supposed to remove it, and the figure would measure the classifier alone instead of the classifier behind its filter. Alarm burden is false alarms per 24 hours of continuous monitoring, computed over the full DS2 records at their native beat rate for the V and S classes.

## 6.11 Subgroup analysis

MIT-BIH records subject sex and age. We report sensitivity and positive predictive value for the V and S classes stratified by sex and by age band, with intervals computed as in section 6.10. With 22 records in DS2 these strata are small and the intervals wide, so we report the interval in place of the point estimate and draw no conclusion it does not support. Two DS2 records carry no age in their header, and they are reported as their own stratum, neither dropped nor imputed. Race and ethnicity metadata do not exist for this database, and section 8.5 records that as a limitation.

## 6.12 Reproducibility

Code, trained weights, record lists, the AAMI mapping table, the environment specification and the seeds are deposited in the public repository cited in section 10, and supplement S5 lists the command that regenerates each table and figure. The four databases are public through PhysioNet and are not redistributed with the code {{Goldberger, 2000 #4; Moody, 2005 #5; Tihonenko, 2007 #69; Greenwald, 1990 #70; Moody, 1990 #71}}.
"""
