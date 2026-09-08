"""Manuscript sections 1 to 5: title, abstract, keywords, introduction, related work."""
from __future__ import annotations

from fmt import f1, num, pct, signed, thousands, words


def build(C) -> str:
    T1, t8, ext, prot = C.T1, C.t8, C.ext, C.prot
    Emac, lat = C.Emac, C.lat
    dist = C.t6.get("distilled_vs_scratch", {})
    title_note = (
        "An earlier draft led with the distillation, and section 7.5 is the reason it no "
        "longer does: the distillation term is not distinguishable from zero on this "
        "problem, so a title that foregrounds it would promise a result the paper does "
        "not have."
        if dist.get("spans_zero") else
        "The distillation is retained in T-c because section 7.5 measures a benefit whose "
        "interval excludes zero.")
    _dm = ext["drop_mean"]
    ext_clause = (
        f"and holding that value on the external databases, changing by "
        f"{-ext['drop_incart']:+.3f} on INCART and {-ext['drop_svdb']:+.3f} on SVDB with no "
        "retraining."
        if abs(_dm) < 0.01 else
        f"falling by {num(_dm)} on the external databases.")
    _rt = (prot["max_gain"] / prot["min_gain"]) if prot["min_gain"] > 0 else float("nan")
    gain_ratio = f"{_rt:.1f}" if _rt == _rt else "several"
    rank_clause = ("no arm changed rank."
                   if prot["n_rank_changes"] == 0 else
                   f"{words(prot['n_rank_changes'])} arms changed rank.")
    dist_ci = (f", an interval of {dist.get('lo', float('nan')):+.3f} to "
               f"{dist.get('hi', float('nan')):+.3f} that includes zero"
               if dist.get("spans_zero") else "")
    order = sorted(C.ARMS, key=lambda a: -C.t4[a]["inter"][0])
    best_arm, e_rank = order[0], order.index("E") + 1
    n_inter = sum(1 for k in T1 if not k.startswith("_")
                  and str(T1[k]["inter_patient"]).lower().startswith("yes"))
    n_size = sum(1 for k in T1 if not k.startswith("_")
                 and T1[k]["model_size_kb_for_fig7"] is not None)
    n_energy = sum(1 for k in T1 if not k.startswith("_") and T1[k]["energy"] != "nr")
    size8 = num(t8.get("size_int8_kb", float("nan")), 1)
    sram = num(t8.get("sram_kb", float("nan")), 1)
    euj = num(t8.get("energy_uj", float("nan")), 1)
    med = f"{lat.get('median', float('nan')):.0f}"
    p99 = f"{lat.get('p99', float('nan')):.0f}"

    return f"""# 1. Title

Three candidates, each 20 words or fewer.

T-a. What inter-patient evaluation changes about lightweight ECG arrhythmia classification: seven arms, two external databases, one deployment budget

T-b. A single-lead INT8 arrhythmia classifier under 100 kB, evaluated between patients, validated externally and costed on named silicon

T-c. Knowledge distillation for microcontroller ECG arrhythmia classification with calibrated abstention, subgroup analysis and E3C-compliant reporting

Working title used throughout this draft: T-a. {title_note}

**Authors.** [[AUTHOR ACTION: full author list in final order, with affiliations and ORCID identifiers]]

**Corresponding author.** [[AUTHOR ACTION: name, postal address, email]]

**Running head.** Inter-patient evaluation of a microcontroller ECG beat classifier

**Word count.** This draft runs to roughly 17,800 words including tables, figure captions and declarations. Journals with a shorter limit are best served by moving sections 7.6 and 7.7 to the supplement, which are the two results sections that stand alone.

# 2. Abstract

**Background.** Reported accuracy for electrocardiogram (ECG) arrhythmia classification is dominated by the evaluation protocol: of 122 systematically reviewed studies, about 30 per cent partitioned between patients, about 31 per cent assessed embedded feasibility, and about 4 per cent did both with AAMI class mapping {{Silva, 2025 #8}}.

**Objective.** To build a single-lead beat classifier small enough for a microcontroller-class wearable, and to measure what it does for an unseen wearer.

**Methods.** A student classifier was distilled from a public ECG foundation model and quantised to INT8 with quantisation-aware training. Training used the de Chazal DS1 records of the MIT-BIH Arrhythmia Database with a patient-disjoint validation split; DS2 was opened once. Beats were mapped to the five AAMI superclasses and paced records excluded. External validation used St. Petersburg INCART and the MIT-BIH Supraventricular Arrhythmia Database without retraining. Six comparator arms ran under the same protocol, including a reimplemented polynomial-sigmoid and Takagi-Sugeno fuzzy classifier with its fuzzy decoder ablated to argmax, and every arm was refitted and scored again under a patient-mixed partition of the same beats. The deployment budget for an STM32F446 was computed from the layer shapes, an assumed CMSIS-NN throughput and the device datasheet; no board was metered.

**Results.** On DS2 the classifier reached a macro-F1 of {f1(Emac['f1'])} across the five AAMI superclasses, with per-class sensitivity, positive predictive value and record-level bootstrap intervals for every class, {ext_clause} It holds {thousands(t8.get('params', 0))} parameters, occupies {size8} kB in INT8 with {sram} kB of peak static RAM, and is computed to need {euj} microjoules per inference at a median {med} microseconds. Under the patient-mixed partition {words(prot['n_arms_improved'])} of the seven arms gained macro-F1, by {num(prot['mean_gain'])} on average, but the gain was {gain_ratio} times larger for the highest-capacity arm than for the lowest and {rank_clause} Distillation was worth {signed(dist.get('diff', float('nan')))} macro-F1 against the identical topology trained from scratch{dist_ci}, and the student ranked {words(e_rank)} of the seven arms; arm {best_arm} scored highest.

**Conclusions.** Accuracy measured with one wearer's beats on both sides of the split is not the accuracy that wearer would experience. This classifier is weaker by that measure and deployable by the other. It has no prospective validation and no ventricular fibrillation class, its deployment budget is computed from the layer shapes and the datasheet with no board metered, and it inherits the 1975 to 1979 recording conditions of MIT-BIH.

# 3. Keywords

Electrocardiography; Arrhythmias, Cardiac; Machine Learning; Wearable Electronic Devices; Signal Processing, Computer-Assisted; Calibration; Algorithms; Neural Networks, Computer

# 4. Introduction

Cardiovascular disease killed 19.2 million people in 2023, against 13.1 million in 1990, and accounted for 437 million disability-adjusted life years {{Stark, 2025 #1}}. Prevalent cases doubled over the same interval, from 311 million to 626 million, and roughly one death in three worldwide is now cardiovascular {{Stark, 2025 #1}}. Arrhythmia is a small part of that total mortality and a large part of its monitoring burden, because the events that matter are intermittent and a resting twelve-lead recording samples minutes out of a year.

Ambulatory monitoring closes some of that gap, and consumer wearables have pushed the recording population far beyond the cardiology clinic. The device that results is a single-lead, dry-electrode, battery-powered sensor worn during ordinary movement. Its computational budget is the part that constrains algorithm design: tens of kilobytes of flash for weights, tens of kilobytes of static RAM for activations, no floating-point throughput to spare, and an energy allowance measured against the days a user expects between charges. A classifier that runs on the phone or in the cloud instead pays for radio transmission, which on this class of device costs more energy per beat than the arithmetic does. The joint statement of the European Heart Rhythm Association, the Heart Rhythm Society and the European Society of Cardiology working group on e-cardiology set out the reporting that clinical adoption now requires, and found trial registration, participant details, data handling and training performance reported in under a fifth of the papers it assessed {{Svennberg, 2025 #2}}.

Against that budget the literature reports accuracy figures and little else. A systematic review of 122 highly cited ECG classification studies found that 37 of them, about 30 per cent, partitioned records so that no patient appeared on both sides of the split; 68 complied with the class definitions of ANSI/AAMI EC57 {{Association for the Advancement of Medical Instrumentation, 2012 #6}}; 38 assessed embedded feasibility in any form; and 5, about 4 per cent, satisfied all three at once {{Silva, 2025 #8}}. The review groups these as the E3C criteria and treats them as a floor that any study in this area should clear.

The partitioning question is the one that decides what a reported number means. De Chazal and colleagues showed two decades ago that a classifier trained and tested on beats from the same patient learns that patient's morphology, and proposed the DS1 and DS2 record split that keeps every subject on one side of the boundary {{deChazal, 2004 #7}}. The information the classifier exploits under a mixed partition, the shape of one individual's QRS complex, is precisely the information a device does not have when it is first strapped to somebody new. The inflation therefore lands exactly on the use case. Reported accuracies above 99 per cent are common in this subfield, and where the same architecture has been trained under both regimes the difference has been attributed to the split and not to the model. Our own earlier work sits inside that pattern and is reimplemented here as a comparator arm {{Silva, 2025 #8; Luz, 2016 #9; Bahrami, 2025 #26; Abbaszadeh, 2024 #12}}.

Two further consequences follow. First, if the protocol dominates, then rankings taken from intra-patient results cannot be trusted to survive the change to inter-patient evaluation, which makes most published comparison tables uninterpretable. That is a testable claim, and we test it. Second, without an embedded profile there is no way to know whether a reported model would fit the device it was designed for. A small group of studies now reports the quantities that would settle the question, and they are the anchors this work is measured against: a tiny matched-filter convolutional network for inter-patient classification at the edge {{Farag, 2023 #36}}, a cascaded quantised spiking network for real-time detection on edge hardware {{Banjo, 2026 #37}}, a one-dimensional convolutional network placed on a system-on-chip through HLS4ML {{Mian, 2024 #38}}, and logic-gate and lookup-table networks evaluated under inter-patient partitioning on programmable logic {{Mommen, 2026 #42}}. The full text of all seven anchor studies was read for this work, and their classes, protocols, sizes, latencies and energies are collected in Table 1 as reported by each primary source.

The compression target has also changed. Self-supervised pretraining on unlabelled waveform corpora has produced ECG foundation models with tens to hundreds of millions of parameters and open weights {{McKeen, 2024 #14; Coppola, 2024 #87; Wang, 2024 #46; Lin, 2026 #45}}. These are teachers, not deployment candidates. The open problem is moving what they know into a student small enough to run on the device, and knowledge distillation {{Hinton, 2015 #13}} with INT8 quantisation-aware training {{Jacob, 2018 #15}} is the established route.

This study builds the classifier and its evaluation together. We report the following contributions.

1. A student classifier of {thousands(t8.get('params', 0))} parameters and {size8} kB in INT8, distilled from a public ECG foundation model, trained on DS1 alone and evaluated once on DS2 under the five AAMI superclasses, with per-class sensitivity, positive predictive value and F1 reported with record-level bootstrap confidence intervals.
2. A paired evaluation of every arm under patient-mixed and inter-patient partitioning of the same beats, which tests two things: how much the protocol adds to each arm, and whether the ordering survives the change. The two answers are different, and section 7.3 reports both, including the one the design did not expect.
3. External validation with no retraining on St. Petersburg INCART and the MIT-BIH Supraventricular Arrhythmia Database, with the degradation reported as a result of the study.
4. Three tests of machinery that a published two-stage arrhythmia classifier never had tested: its Takagi-Sugeno fuzzy decoder against argmax under an identical first stage; the Wild Horse Optimizer with Giza Pyramids Construction against random search, CMA-ES, tree-structured Parzen estimation and gradient descent at a matched evaluation budget over 30 seeded runs; and the same architecture refitted by gradient descent, which separates what the architecture cannot do from what the optimiser did not do.
5. A deployment budget for an STM32F446 with every assumption named, reporting parameters, multiply-accumulate operations, INT8 size, peak static RAM, latency at the mean, median, 95th and 99th percentiles, energy per inference and projected battery life, with each quantity labelled measured or computed. No board was available for this study, and section 6.8 says so before any number is quoted.
6. A calibrated abstention rule reported as an accuracy-versus-coverage curve, together with false alarms per 24 hours of continuous monitoring and per-subgroup sensitivity and positive predictive value by sex and age band.

Section 5 organises prior work by theme. Section 6 gives the protocol in enough detail to reproduce it. Section 7 reports results, section 8 discusses what they mean for a wearable and for the regulatory pathway, and section 9 concludes.

# 5. Related work

## 5.1 Classical feature-based classification

Beat classification began with detection and measurement. The Pan and Tompkins detector fixed the standard preprocessing chain of band-pass filtering, differentiation, squaring and integration, and still supplies R-peak locations for much of this literature {{Pan, 1985 #20}}. On top of detection, two families of feature carry most of the discriminative load: morphology sampled around the fiducial point, and the interval structure of the rhythm, principally the preceding and following RR intervals and their ratio to a local average. De Chazal and colleagues combined both and reported per-class results under a partition that keeps patients separate {{deChazal, 2004 #7}}. Llamedo and Martínez pursued feature selection under the same constraint and reached comparable performance with a compact set {{Llamedo, 2011 #21}}. Mondéjar-Guerra and colleagues fused temporal and morphological descriptors through an ensemble of support vector machines {{Mondejar-Guerra, 2019 #22}}.

The performance band these methods define matters more than any individual result. Under inter-patient partitioning on MIT-BIH, supraventricular ectopic beats remain the hardest class for feature-based and deep methods alike, because the morphological difference from a normal beat is small and the diagnostic evidence sits in the rhythm and in P-wave behaviour, outside the QRS complex. Gradient-boosted trees {{Chen, 2016 #23}} and random forests on the same descriptors remain competitive baselines, and we include them as arm A.

## 5.2 Deep learning and the intra- versus inter-patient divide

Convolutional networks removed the feature-engineering step. Kiranyaz and colleagues trained a compact one-dimensional network per patient and reported real-time operation, with the explicit caveat that the model is patient-specific and requires a labelled segment from the individual being monitored {{Kiranyaz, 2016 #24}}. Hannun and colleagues trained a 34-layer network on 91,232 single-lead ambulatory recordings and evaluated it against a committee of cardiologists on held-out patients {{Hannun, 2019 #25}}.

Between those two designs sits the divide that shapes this field's reported numbers. When beats from one recording appear in both training and test partitions, a network can identify the recording instead of the rhythm. The gap this opens on identical data is larger than the gap between competing architectures, which makes the partition the dominant variable in any cross-paper comparison {{Silva, 2025 #8; Luz, 2016 #9}}. Bahrami and colleagues put that proposition directly to the test, training identical architectures under inter-patient, intra-patient and patient-specific regimes and comparing what each reports {{Bahrami, 2025 #26}}. Their design is the one we extend in section 7.3, across seven arms instead of one and with confidence intervals attached.

Intra-patient evaluation inflates reported performance across a large part of this literature, including our own earlier two-stage classifier {{Abbaszadeh, 2024 #12}}, which is reimplemented here as arm B and ablated as arm C. Recent hybrid convolutional and recurrent designs report high accuracy on MIT-BIH under mixed partitions {{Alamatsaz, 2024 #30; Mishra, 2024 #31; Dhara, 2024 #32}}, and those figures are not comparable with the ones reported here. A smaller group works under the inter-patient constraint and reports the lower numbers that follow from it: multiscale convolution with feature-channel and boundary attention {{Zhou, 2024 #27}}, a sequence-to-sequence architecture {{Midani, 2024 #28}}, and a network with attention over the beat window for premature ventricular contractions {{Chen, 2024 #29}}. This second group is the correct comparison set for the results in section 7.2.

## 5.3 Edge and TinyML deployment

Depthwise-separable convolution {{Howard, 2017 #34}} and post-training or quantisation-aware INT8 conversion {{Jacob, 2018 #15}} together reduce a floating-point network to something a Cortex-M can execute, and pruning removes further weight {{Han, 2016 #35}}. The runtime layer is equally settled: CMSIS-NN supplies kernels tuned for the instruction set {{Lai, 2018 #16}}, and TensorFlow Lite for Microcontrollers supplies an interpreter with a static arena and no dynamic allocation {{David, 2021 #17}}.

Seven studies define the anchor set for this work, chosen because each names its target silicon, and every one of them was read in full for Table 1. A matched-filter convolutional classifier was built for inter-patient classification at the edge and reports a 15 kB model with an average inference time below 1 ms on a Raspberry Pi, with generalisation checked on INCART, QT and PTB {{Farag, 2023 #36}}. A cascaded quantised spiking network was deployed on a PYNQ-Z2 and reports {T1['Banjo2026']['model_size']} at {T1['Banjo2026']['latency'].split(',')[0]} {{Banjo, 2026 #37}}, and a spiking architecture with adaptive multi-threshold encoding reached {T1['Diware2025']['energy']} in a 40 nm application-specific integrated circuit {{Diware, 2025 #39}}. A one-dimensional convolutional network was placed on a system-on-chip through HLS4ML, reporting power but no model size in bytes and no numeric latency outside a figure {{Mian, 2024 #38}}, a lightweight model was demonstrated on an nRF52840 wearable module at 25 kB and 172.3 ms {{Kim, 2024 #40}}, and a student network distilled for wearable monitoring was deployed on an STM32F429 at 25.34 KiB of read-only memory with an inference time of 19 ms {{An, 2024 #41}}. At the low-power extreme, logic-gate and lookup-table networks were evaluated under inter-patient partitioning on an Artix 7 device at an estimated 50 to 70 pJ per inference {{Mommen, 2026 #42}}. Wearable-oriented hybrid systems continue to appear reporting accuracy alone, sometimes with edge compatibility asserted in the title and unquantified in the text {{Hannan, 2024 #33; Alamatsaz, 2024 #30; Nugroho, 2026 #90}}.

Three observations survive that reading, and they are the reason Table 1 reports what each study measured alongside what it scored. Reported model sizes in this group span more than three orders of magnitude, so a size claim carries no information without a named target device and toolchain. Reported energies span nine orders of magnitude, from 50 picojoules for a combinational logic network on an FPGA to 23.31 millijoules for a board-level figure that includes the host processor, so an energy number without its measurement boundary is not comparable with another one. And the class definitions differ more than the architectures do: two of the seven use a four-class AAMI mapping with the Q superclass dropped, one uses an eleven-class severity grouping that explicitly sets the AAMI grouping aside, one classifies whole rhythms on a different database instead of beats, and the headline accuracy of another is a three-class result whose five-class variant scores {num(T1['Farag2023']['macro_f1_5class_pct'], 2)} per cent average F1 on the same data.

Table 1 collects what prior work reports. Of the studies listed, {n_inter} partition by patient, {n_size} state a model size in bytes, and {n_energy} state an energy figure.

**Table 1.** Reporting completeness in prior work. "Reported metric" names the metric set a study reports, not its value, because values obtained under different partitions are not comparable. "nr" means the study reports no value for that field, and the count of those cells is the argument for this study. Every cell for the seven edge-deployment studies was read from the primary text of that study; where a quantity appears only inside a figure the cell says so.

| Reference | Year | Classes | AAMI | Inter-patient | Datasets | External validation | Model size | Latency | Energy | Reported metric |
|---|---|---|---|---|---|---|---|---|---|---|
| de Chazal et al. {{deChazal, 2004 #7}} | 2004 | 5 | Yes | Yes | MIT-BIH | No | nr | nr | nr | Se, PPV, FPR, Acc |
| Llamedo and Martinez {{Llamedo, 2011 #21}} | 2011 | 5 | Yes | Yes | MIT-BIH, INCART | Yes | nr | nr | nr | Se, PPV, Acc |
| Kiranyaz et al. {{Kiranyaz, 2016 #24}} | 2016 | 5 | Yes | No, patient-specific | MIT-BIH | No | nr | reported | nr | Se, PPV, Acc |
| Mondejar-Guerra et al. {{Mondejar-Guerra, 2019 #22}} | 2019 | 5 | Yes | Yes | MIT-BIH | No | nr | nr | nr | Se, PPV, Acc |
| Hannun et al. {{Hannun, 2019 #25}} | 2019 | 12 | No | Yes | private ambulatory | No | nr | nr | nr | F1, AUC |
| Farag {{Farag, 2023 #36}} | 2023 | 3, 4 and 5, variants | Yes | Yes, de Chazal | MIT-BIH | Yes, INCART, QT, PTB | 15 kB | below 1 ms, Raspberry Pi | nr | Acc, Se, PPV, F1 |
| Zhou et al. {{Zhou, 2024 #27}} | 2024 | 5 | Yes | Yes | MIT-BIH | No | nr | nr | nr | Se, PPV, Acc |
| Midani et al. {{Midani, 2024 #28}} | 2024 | 5 | Yes | Yes | MIT-BIH | No | nr | nr | nr | Se, PPV, Acc |
| Chen et al. {{Chen, 2024 #29}} | 2024 | 2, PVC vs other | Partial | Yes | MIT-BIH | No | nr | nr | nr | Se, PPV, F1 |
| Bahrami et al. {{Bahrami, 2025 #26}} | 2025 | 5 | Yes | Yes and No, compared | MIT-BIH | No | nr | nr | nr | Acc per protocol |
| Mian and Zafar {{Mian, 2024 #38}} | 2024 | 5 | Yes, EC57:1998 cited | No, oversampling before splitting | MIT-BIH, 125 Hz preprocessed | No | nr, FPGA resource counts only | nr, inferences per second in a figure | nr, 1.655 W board power | Acc, Precision, Recall, F1 |
| Diware et al. {{Diware, 2025 #39}} | 2025 | 11, severity grouping | No, AAMI grouping set aside | No, 60/20/20 beat split | MIT-BIH | No | nr, 0.36 mm2 die area | 90 us mean, 680 us maximum | 0.059 uJ | Acc |
| Kim et al. {{Kim, 2024 #40}} | 2024 | 5 | Yes | Yes, subject-oriented | MIT-BIH | No | 25 kB quantised | 172.3 ms on nRF52840 | nr, not measured | Acc, Se, PPV, FPR |
| An et al. {{An, 2024 #41}} | 2024 | 4 rhythm classes | No | Record-level split, one record per subject | Chapman-Shaoxing | No | 25.34 KiB ROM, 11.65 KiB RAM | 19 ms on STM32F429 | nr | Acc |
| Banjo and Ghoraani {{Banjo, 2026 #37}} | 2026 | 4, Q omitted | Yes | No, patient-specific 80/20 | MIT-BIH | No | 0.064 MB INT8 | 11.54 ms on PYNQ-Z2 | 23.31 mJ, board level | Acc, macro-F1 |
| Mommen et al. {{Mommen, 2026 #42}} | 2026 | 4, Q omitted | Yes | Yes, de Chazal | MIT-BIH | No | nr, 2000 to 2990 FPGA LUTs | one 10 ns clock cycle | 50 to 70 pJ, estimated | Acc, jkappa |
| Alamatsaz et al. {{Alamatsaz, 2024 #30}} | 2024 | 9 | No | No | MIT-BIH | No | nr | nr | nr | Acc |
| Mishra and Tiwari {{Mishra, 2024 #31}} | 2024 | 2 | No | No | MIT-BIH | No | nr | nr | nr | Acc |
| Abbaszadeh and Bazargani {{Abbaszadeh, 2024 #12}} | 2024 | 7 rhythm types | No | No | MIT-BIH, LTAF | No | nr | 0.32 ms | nr | Acc, Se, Sp |
| This work | 2026 | 5 | Yes | Yes, and patient-mixed for contrast | MIT-BIH, INCART, SVDB | Yes, 2 databases | {size8} kB INT8 | {med} us median and {p99} us 99th percentile, computed | {euj} uJ, computed | Se, PPV, Sp, F1, FPR, macro-F1, kappa, MCC, all with CIs |

## 5.4 ECG foundation models and self-supervised pretraining

Pretraining on unlabelled waveforms changed what a small labelled dataset can support. ECG-FM released open weights trained by contrastive and masked objectives on large public corpora {{McKeen, 2024 #14}}. A foundation model built on more than ten million recordings followed {{Li, 2025 #43}}, alongside a transformer trained by self-supervision for assessment of cardiac and coronary function {{Moody, 2025 #44}} and a self-supervised model applied to disease prediction and genetic association {{Lin, 2026 #45}}. AnyECG learns embeddings through two-stage self-supervised pretraining across multiple datasets {{Wang, 2024 #46}}, HuBERT-ECG adapts masked-unit prediction from speech and is the model used as the teacher here {{Coppola, 2024 #87}}, and masked representation learning has been guided to capture spatio-temporal structure {{Na, 2024 #92}}. A systematic review catalogues the class {{Han, 2024 #47}}.

The parameter counts involved put every one of them beyond the target device by orders of magnitude, and a 2026 review argues that compression through quantisation, pruning and distillation is what stands between such models and deployment {{Khan, 2026 #48}}. State-space architectures offer a second route, with lower memory cost than attention at long sequence lengths {{Gu, 2023 #49}}. The extreme case has been demonstrated in an adjacent biosignal: a bidirectional Mamba model for electroencephalography compressed by quantisation-aware training and run on an ultra-low-power microcontroller {{Tegon, 2026 #50}}. That study is the closest published precedent for what section 6.8 attempts here, in a different modality.

## 5.5 Federated and personalised learning for wearable ECG

Federated averaging keeps waveforms on the device and shares gradients or weights instead {{McMahan, 2017 #51}}. Applied to arrhythmia monitoring, PFedAL-ECG combines a low-parameter feature extractor with personalised active learning aimed at resource-constrained devices and cross-user heterogeneity {{He, 2026 #52}}, and resource-aware schemes let local model complexity vary with the hardware while still aggregating {{Islam, 2025 #53}}. Privacy-preserving classification has been demonstrated across heterogeneous devices with communication cost reported {{Elmir, 2025 #54}}, and on-device learning has been combined with wearable ECG and smartphone photoplethysmography in an adaptive monitoring pipeline {{Kartali, 2026 #56}}. Sharing updates instead of data reduces exposure without eliminating it, which is the argument for differential privacy alongside it {{Bokhari, 2025 #55}}. We run no federated experiment here. Section 8.6 states what such an experiment would have to measure for this classifier.

## 5.6 Metaheuristic optimisation and its methodological critique

Metaphor-based metaheuristics are common in this application area and are under sustained methodological criticism. Sörensen argued that a large part of this literature re-describes existing operators in new vocabulary without adding mechanism {{Sörensen, 2015 #57}}. Component-based analyses have shown that several widely cited optimisers are equivalent to established algorithms once the metaphor is stripped away {{Camacho-Villalón, 2023 #58}}. A recent critical review of newly proposed metaheuristics finds the same pattern of competitive testing against other metaphor-based methods, benchmark-specific tuning, and justification by metaphor in place of mathematics {{Velasco, 2024 #59}}.

The two optimisers used in the classifier we reimplement are the Wild Horse Optimizer {{Naruei, 2022 #60}} and Giza Pyramids Construction {{Harifi, 2021 #61}}, with Grey Wolf Optimization for feature selection {{Mirjalili, 2014 #62}}. We adopt the abbreviation WHOA for the first, because WHO in a cardiovascular paper means the World Health Organization. The question these algorithms have never been asked in this application is whether they beat a metaphor-free control at the same evaluation budget. Random search {{Bergstra, 2012 #63}}, CMA-ES {{Hansen, 2001 #64}} and tree-structured Parzen estimation {{Bergstra, 2011 #65; Akiba, 2019 #66}} are the standard controls, and the objective in the reimplemented arm is differentiable, so gradient descent is available too. Section 6.7 fixes the comparison and the non-parametric tests {{Demsar, 2006 #67}} before the runs.
"""
