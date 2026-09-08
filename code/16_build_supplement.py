"""Assemble the supplementary document from results/numbers.json.

S1 and S2 reproduce the published TRIPOD+AI and EHRA checklists item by item,
with the item numbers and wording transcribed from the source forms, which are
stored alongside as results/tripod_ai_items.json and
results/ehra_checklist_items.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from armb import feature_names
from common import DS1, DS1_TRAIN, DS1_VAL, DS2, ROOT, RESULTS, jload
from fmt import f1, num, p_value, pct, signed, thousands
from msctx import context

OUT = ROOT / "06_Supplementary.md"

TRIPOD_ANSWER = {
    "1": ("Yes", "Section 1, title T-a"),
    "2": ("Yes", "Section 2"),
    "3a": ("Yes", "Section 4, paragraphs 1 to 3"),
    "3b": ("Yes", "Sections 4 and 8.4, intended use stated as a review aid"),
    "3c": ("Partial", "Section 7.9 reports sex and age strata; no health-inequality literature is cited for this cohort because MIT-BIH carries no socioeconomic metadata"),
    "4": ("Yes", "Section 4, contributions 1 to 6"),
    "5a": ("Yes", "Section 6.1"),
    "5b": ("Yes", "Section 6.1, recordings collected 1975 to 1979"),
    "6a": ("Yes", "Section 6.1, Beth Israel Hospital arrhythmia laboratory, single centre"),
    "6b": ("Yes", "Section 6.1, paced records excluded per EC57"),
    "6c": ("No", "Treatment data are not distributed with these databases"),
    "7": ("Yes", "Sections 6.2 and 6.3, and the signal-quality gate in section 6.9"),
    "8a": ("Yes", "Section 6.1, five AAMI superclasses, beat level, no time horizon"),
    "8b": ("Yes", "Database annotations by the original expert annotators"),
    "8c": ("Not applicable", "Labels were fixed before this study began"),
    "9a": ("Yes", "Sections 6.3 and 6.5, with Grey Wolf feature selection for arm B"),
    "9b": ("Yes", "Section 6.3"),
    "9c": ("Yes", "R-peak positions are database annotations, section 6.3 and section 8.5"),
    "10": ("Partial", "Section 7.1 gives the counts; no formal power calculation was performed and section 8.5 says so"),
    "11": ("Yes", "Section 6.3, beats with fewer than ten predecessors"),
    "12a": ("Yes", "Section 6.1, record-level partitioning with DS2 opened once"),
    "12b": ("Yes", "Section 6.2 normalisation and section 6.6 standardisation"),
    "12c": ("Yes", "Sections 6.5 to 6.7 and supplement S3"),
    "12d": ("Yes", "Section 6.10, record-level bootstrap with seed variation folded in"),
    "12e": ("Yes", "Section 6.10, pre-specified"),
    "12f": ("Yes", "Section 6.9, temperature scaling fitted on DS1 validation only"),
    "12g": ("Yes", "Section 6.12 and supplement S5"),
    "13": ("Yes", "Section 6.4, capped inverse-frequency weighting, with the cap justified"),
    "14": ("Yes", "Section 6.11 and section 7.9"),
    "15": ("Yes", "Sections 6.5 and 6.9, calibrated probabilities with a conformal abstention rule"),
    "16": ("Yes", "Sections 6.1 and 7.4, differences between MIT-BIH, INCART and SVDB stated"),
    "17": ("Yes", "Section 10, not required for de-identified public databases"),
    "18a": ("Yes", "Section 10"),
    "18b": ("Yes", "Section 10"),
    "18c": ("No", "No protocol was prepared; section 10 states this"),
    "18d": ("No", "Not registered; section 10 states this"),
    "18e": ("Yes", "Section 10, four PhysioNet DOIs"),
    "18f": ("Yes", "Section 10 and supplement S5"),
    "19": ("No", "None. Retrospective public databases only; stated in section 10"),
    "20a": ("Yes", "Section 7.1, Table 2"),
    "20b": ("Yes", "Section 7.1 and Table 9 for sex and age"),
    "20c": ("Yes", "Section 7.1, composition of INCART and SVDB against DS2"),
    "21": ("Yes", "Table 2 and Table 9"),
    "22": ("Yes", "Section 6.12, weights and code deposited"),
    "23a": ("Yes", "Section 7.2, Table 3, and Table 9 for subgroups, all with 95 per cent intervals"),
    "23b": ("Yes", "Supplement S4, per-record results on DS2"),
    "24": ("Yes", "Section 7.7, calibration is the only updating performed"),
    "25": ("Yes", "Sections 8.1 and 8.2"),
    "26": ("Yes", "Section 8.5"),
    "27a": ("Yes", "Section 6.9, signal-quality gate, with rejection rates in section 7.7"),
    "27b": ("Yes", "Section 8.3 and section 8.4, review aid with a clinician in the loop"),
    "27c": ("Yes", "Section 8.6"),
}

EHRA_ANSWER = {
    "i": ("Yes", "Section 1"),
    "1": ("Yes", "Section 8.4, screening aid that flags beats for clinician review"),
    "2": ("Partial", "Section 7.2 compares against classical baselines as arm A; no comparison against a clinical care pathway was possible"),
    "3": ("Yes", "Section 6.1"),
    "4": ("Yes", "Section 6.1, four public PhysioNet databases with DOIs"),
    "5": ("Yes", "Section 6.1 and Table 2"),
    "6": ("Yes", "Section 6.1, paced records excluded; 47 subjects with sex and age in section 7.9"),
    "7": ("Yes", "Section 6.1, beat annotations by the original database annotators mapped to AAMI superclasses"),
    "8": ("Yes", "Section 6.1, de Chazal DS2 opened once, plus a patient-mixed contrast partition"),
    "9": ("Partial", "Table 2 gives every count; no formal power calculation, stated in section 8.5"),
    "10": ("Yes", "Section 6.1, five AAMI superclasses"),
    "11": ("Yes", "Sections 6.2 and 6.3"),
    "12": ("Yes", "Sections 6.3 and 6.6, including the four training augmentations"),
    "13": ("Yes", "Section 6.4, capped inverse-frequency weighting, with an oversampling ablation in section 7.5"),
    "14": ("Yes", "Section 6.9, signal-quality gate; section 6.3, beats with fewer than ten predecessors"),
    "15": ("Yes", "Section 6.5, Grey Wolf selection for arm B; the neural arms use no hand-selected features"),
    "16": ("No", "Not a regulated device. Section 8.4 states the classification the software would fall under and the applicable dates"),
    "17": ("Yes", "Arm B2 prints a rule base in named clinical quantities, supplement S6"),
    "18": ("Yes", "Section 10, not required for de-identified public databases"),
    "19": ("Yes", "Section 7.9, sex and age strata with intervals; section 8.5 records the absent metadata"),
    "20": ("Yes", "Section 10 and supplement S5"),
    "21": ("Not applicable", "Not a trial. Section 10 states that no registration exists"),
    "22": ("Yes", "Section 7.1 and section 7.9"),
    "23": ("Yes", "Supplement S3, Table S3.4"),
    "24": ("Yes", "Section 7.2, Table 3"),
    "25": ("Yes", "Section 7.4, Table 5"),
    "26": ("Yes", "Section 6.10, per-class and macro metrics with record-level bootstrap intervals"),
    "27": ("Yes", "Section 7.2 error structure, Figure 2, and the per-record table S4.1"),
    "28": ("Yes", "Arm A holds five classical methods under the identical protocol"),
    "29": ("Yes", "Sections 7.3, 7.4 and 8.5"),
    "ii": ("Yes", "Section 9"),
}


def main() -> int:
    C = context()
    tri = jload(RESULTS / "tripod_ai_items.json")
    ehra = jload(RESULTS / "ehra_checklist_items.json")["items"]
    hp, perf, t7, t8, b2, bw = C.hp, C.perf, C.t7, C.t8, C.b2, C.bw
    fs = hp.get("featsel", {})
    names = feature_names()
    sel = fs.get("selected", [])
    env = jload(RESULTS / "environment.json") if (RESULTS / "environment.json").exists() else {}

    # ---- S1
    s1 = []
    for it in tri:
        ans, loc = TRIPOD_ANSWER.get(it["item"], ("Not addressed", ""))
        txt = it["text"].replace("|", ";")
        s1.append(f"| {it['section'].title()} | {it['item']} | {txt} | {ans} | {loc} |")
    s1 = "\n".join(s1)

    # ---- S2
    s2 = []
    for it in ehra:
        ans, loc = EHRA_ANSWER.get(it["n"], ("Not addressed", ""))
        s2.append(f"| {it['section']} | {it['n']} | {it['name']} | {it['text'].replace('|', ';')} "
                  f"| {ans} | {loc} |")
    s2 = "\n".join(s2)

    # ---- S3.3 search spaces and selected values
    def hpv(arm, key, nd=4):
        v = hp.get(arm, {}).get("best_params", {}).get(key)
        if v is None:
            return "not searched"
        return f"{v:.{nd}g}" if isinstance(v, float) else str(v)

    s33 = f"""| A | regularisation strength, C | log-uniform 1e-3 to 1e3 | {hpv('A','C')} |
| A | random forest trees | 100 to 1000 | {hpv('A','rf_trees')} |
| A | gradient-boosting depth | 3 to 10 | {hpv('A','gb_depth')} |
| A | gradient-boosting learning rate | log-uniform 0.01 to 0.3 | {hpv('A','gb_lr')} |
| B, C | WHOA population size | 20 to 200 | {hpv('B','whoa_pop')} |
| B, C | WHOA evaluation budget | fixed | 15,000 |
| B | fuzzy membership width, low | 0.05 to 1.0 | {hpv('B','sigma_low')} |
| B | fuzzy membership width, high | 0.05 to 1.0 | {hpv('B','sigma_high')} |
| B2 | GPC population size | 20 to 200 | {hpv('B2','gpc_pop')} |
| B2 | Gaussian membership width per input | 0.05 to 1.0, fitted | see Table S6.2 |
| D | width multiplier | 0.25 to 1.0 | {hpv('D','width')} |
| D | depth, blocks | 3 to 6 | {hpv('D','blocks')} |
| D | learning rate | log-uniform 1e-4 to 3e-3 | {hpv('D','lr')} |
| D | weight decay | log-uniform 1e-5 to 1e-2 | {hpv('D','weight_decay')} |
| D | class-weight cap | 2 to 30 | {hpv('D','weight_cap')} |
| D | augmentation: gain, noise, drift, shift | see note | {hpv('D','gain_hi')}, {hpv('D','noise')}, {hpv('D','drift')}, {hpv('D','shift')} |
| E | width multiplier | 0.25 to 1.0 | {hpv('E','width')} |
| E | depth, blocks | 3 to 6 | {hpv('E','blocks')} |
| E | learning rate | log-uniform 1e-4 to 3e-3 | {hpv('E','lr')} |
| E | weight decay | log-uniform 1e-5 to 1e-2 | {hpv('E','weight_decay')} |
| E | class-weight cap | 2 to 30 | {hpv('E','weight_cap')} |
| E | distillation temperature, T | 1 to 10 | {hpv('E','temperature')} |
| E | distillation weight, alpha | 0.1 to 0.9 | {hpv('E','alpha')} |
| E | QAT start epoch | 10 to 40 | {hpv('E','qat_start_epoch')}, carried in the search space but not used: the final design inserts the fake-quantisation nodes after the full-precision checkpoint has converged and fine-tunes for a further 40 epochs, so no value of this parameter changes the quantised model |
| E | augmentation: gain, noise, drift, shift | see note | {hpv('E','gain_hi')}, {hpv('E','noise')}, {hpv('E','drift')}, {hpv('E','shift')} |
| F | width multiplier | 0.25 to 1.0 | {hpv('F','width')} |
| F | depth, blocks | 3 to 6 | {hpv('F','blocks')} |
| F | learning rate | log-uniform 1e-4 to 3e-3 | {hpv('F','lr')} |
| F | weight decay | log-uniform 1e-5 to 1e-2 | {hpv('F','weight_decay')} |
| F | class-weight cap | 2 to 30 | {hpv('F','weight_cap')} |
| F | augmentation: gain, noise, drift, shift | see note | {hpv('F','gain_hi')}, {hpv('F','noise')}, {hpv('F','drift')}, {hpv('F','shift')} |"""

    # ---- S3.4 training performance
    s34 = []
    for a in C.ARMS:
        d = perf.get(a, {})
        ep = d.get("epochs_to_stop", [])
        eps = (f"{np.mean([e for e in ep if e]):.0f}" if any(ep) else "not applicable")
        s34.append(f"| {a} | {num(d.get('val_macro_f1_mean', float('nan')))} "
                   f"({num(d.get('val_macro_f1_sd', float('nan')))}) "
                   f"| {num(d.get('val_selection_f1_mean', float('nan')))} "
                   f"| {num(d.get('train_macro_f1_mean', float('nan')))} | {eps} "
                   f"| {num(d.get('wallclock_min', float('nan')), 1)} |")
    s34 = "\n".join(s34)

    # ---- S3.5 bandwidth
    s35 = []
    for r in bw.get("rows", []):
        s35.append(f"| {r['label']} | {num(r['macro_f1'])} ({num(r['macro_f1_sd'])}) "
                   f"| {100*r['se_v']:.1f} | {100*r['se_s']:.1f} |")
    s35 = "\n".join(s35) or "| not run | | | |"

    # ---- S4.1 per record
    def pct_or_blank(v):
        """A record holding no beat of that class gets an empty cell, not a zero."""
        return "" if v != v else f"{100 * v:.1f}"

    s41 = []
    for rec in sorted(C.s41):
        d = C.s41[rec]
        c = d["counts"]
        s41.append(f"| {rec} | {thousands(d['beats'])} | {c['N']} | {c['S']} | {c['V']} | "
                   f"{c['F']} | {c['Q']} | {num(d['macro_f1'])} | "
                   f"{pct_or_blank(d['se_v'])} | {pct_or_blank(d['se_s'])} |")
    s41 = "\n".join(s41)

    # ---- S6
    s61 = []
    for r in b2.get("rules", []):
        s61.append(f"| {r['rule']} | {r['antecedent']} | {r['consequent']} | "
                   f"{r['firing_strength']:.4f} | {r['weight']:.3f} | "
                   f"{'fires' if r['fires'] else 'never fires on DS2'} |")
    s61 = "\n".join(s61) or "| not run | | | | | |"
    s62 = []
    for nm, d in b2.get("memberships", {}).items():
        s62.append(f"| {nm} ({d['unit']}) | {d['low_mean']:.3f}, {d['low_width']:.3f} "
                   f"| {d['mid_mean']:.3f}, {d['mid_width']:.3f} "
                   f"| {d['high_mean']:.3f}, {d['high_width']:.3f} |")
    s62 = "\n".join(s62) or "| not run | | | |"

    sel_names = "; ".join(f"{i} ({names[i]})" for i in sel)

    DOC = f"""# Supplementary materials

Companion to the manuscript. Citations use the same EndNote temporary format as the main text and resolve against the same `references.ris` library.

# S0. Abbreviations and fixed terminology

**Table S0.1.** Abbreviations, each used for one concept only.

| Abbreviation | Expansion |
|---|---|
| AAMI | Association for the Advancement of Medical Instrumentation |
| CI | confidence interval |
| CMA-ES | covariance matrix adaptation evolution strategy |
| DS1, DS2 | the de Chazal training and test record sets of the MIT-BIH Arrhythmia Database |
| ECE | expected calibration error |
| ECG | electrocardiogram |
| E3C | embedded, clinical and comparative criteria |
| F | fusion beat superclass |
| FPR | false positive rate |
| GPC | Giza Pyramids Construction optimiser |
| GWO | Grey Wolf Optimization |
| INCART | St. Petersburg Institute of Cardiological Technics 12-lead Arrhythmia Database |
| INT8 | 8-bit integer quantised representation |
| MAC | multiply-accumulate operation |
| MCC | Matthews correlation coefficient |
| N | normal beat superclass |
| PPV | positive predictive value |
| Q | unclassifiable or paced beat superclass |
| QAT | quantisation-aware training |
| S | supraventricular ectopic beat superclass |
| Se | sensitivity |
| SQI | signal quality index |
| SRAM | static random-access memory |
| SVDB | MIT-BIH Supraventricular Arrhythmia Database |
| TPE | tree-structured Parzen estimator |
| V | ventricular ectopic beat superclass |
| WHOA | Wild Horse Optimizer |

WHO is used in the manuscript only for the World Health Organization. The Wild Horse Optimizer is WHOA throughout, including in tables and captions.

**Table S0.2.** Fixed object names. Each object carries the same name on every mention in the manuscript and in this supplement.

| Object | Name used | Never called |
|---|---|---|
| Model E, the proposed model | the student classifier | the model, the network, the framework |
| The model E was distilled from | the teacher network | the foundation model, the large model |
| Model D | the separable CNN | the baseline CNN, the convolutional model |
| Stage one of model B | the polynomial-sigmoid scorer | the SVM, the demarcation stage |
| Stage two of model B | the fuzzy decoder | the fuzzy system, the inference engine |
| The replacement in model C | the argmax decoder | the hard decision, the max rule |
| One annotated heartbeat | a beat | a segment, a window, a sample |

# S1. TRIPOD+AI checklist

Item numbers and wording are transcribed from the published TRIPOD+AI expanded checklist, version 7 February 2024 {{Collins, 2024 #10}}. Locations are manuscript sections, not page numbers, since pagination is set at typesetting. Items answered Partial, No or Not applicable are the honest state of this study and each carries the manuscript sentence that says so.

**Table S1.1.** Completed TRIPOD+AI checklist, all {len(tri)} items.

| Section | Item | Checklist item | Reported | Location |
|---|---|---|---|---|
{s1}

# S2. Electrophysiology artificial intelligence checklist

The checklist of the joint EHRA, HRS and ESC e-cardiology scientific statement carries 29 numbered items plus two lettered items, i) Title and ii) Conclusion; the wording below is transcribed from the published statement {{Svennberg, 2025 #2}}.

**Table S2.1.** Completed EHRA artificial intelligence checklist.

| Domain | Item | Name | Checklist item | Reported | Location |
|---|---|---|---|---|---|
{s2}

# S3. Hyperparameters, record lists and mappings

**Table S3.1.** Records of each partition. The DS1 validation records were chosen under the three constraints of section 6.1 and never revisited.

| Subset | Records | Beats |
|---|---|---|
| DS1 training | {', '.join(DS1_TRAIN)} | {thousands(C.tr_row['Total'])} |
| DS1 validation | {', '.join(DS1_VAL)} | {thousands(C.va_row['Total'])} |
| DS2 test | {', '.join(DS2)} | {thousands(C.ds2_row['Total'])} |

**Table S3.2.** Annotation symbol to AAMI superclass mapping, following EC57 {{Association for the Advancement of Medical Instrumentation, 2012 #6}}.

| Superclass | Annotation symbols |
|---|---|
| N, normal | N, L, R, e, j |
| S, supraventricular ectopic | A, a, J, S |
| V, ventricular ectopic | V, E |
| F, fusion | F |
| Q, unclassifiable or paced | /, f, Q |

**Feature pool and selection.** The pool holds 36 descriptors: the four rhythm descriptors of section 6.3 followed by 32 morphology samples taken uniformly across the 163-sample window. Grey Wolf Optimization with a multinomial logistic wrapper, 20 wolves over 30 iterations scored on a fixed stratified subsample of {thousands(fs.get('wrapper_beats', 0))} DS1 training beats, evaluated {fs.get('unique_subsets_evaluated', 0)} distinct subsets and selected indices {sel_names}. That subset is used by arms B and C and by the optimiser comparison of section 6.7.

**Table S3.3.** Search spaces and selected values. The selected column is filled from the completed search; the space column was fixed before any search ran. Augmentation strengths are, in order, the upper gain multiplier drawn log-uniformly on 1.0 to 1.6, the additive noise standard deviation on 0 to 0.15, the baseline drift amplitude on 0 to 0.3 and the time shift in samples on 0 to 16.

| Arm | Hyperparameter | Search space | Selected |
|---|---|---|---|
{s33}

**Table S3.4.** Training performance on the DS1 validation subset, reported because it is the quantity the electrophysiology checklist finds missing from most papers {{Svennberg, 2025 #2}}. Macro-F1 is over five classes; the selection score is macro-F1 over N, S and V, which is the quantity early stopping and the hyperparameter search maximised. Values are the mean over five seeds with the standard deviation in parentheses.

| Arm | Validation macro-F1 (SD) | Validation selection score | Training macro-F1 | Epochs to stop | Wall-clock, min |
|---|---|---|---|---|---|
{s34}

**Table S3.5.** Bandwidth ablation. The manuscript uses a 40 Hz upper corner; this table reports the cost of that choice against a 100 Hz corner, with the student topology and hyperparameters held fixed and scored on the DS1 validation subset. DS2 is not used here.

| Upper corner | Macro-F1 (SD) | Se, V % | Se, S % |
|---|---|---|---|
{s35}

Training hardware: {env.get('gpu', 'not recorded')}; {env.get('cpu', 'not recorded')}; {env.get('ram', 'not recorded')}. Software versions: {env.get('versions', 'not recorded')}. No STM32 toolchain was used, because the deployment budget of section 6.8 is computed rather than metered.

# S4. Per-record results and additional curves

**Table S4.1.** Per-record results on DS2 for the student classifier, averaged over five seeds. A record-level table is included because the record-level bootstrap of section 6.10 is only interpretable alongside it, and because one record can dominate a rare class. A blank sensitivity cell means the record holds no beat of that class.

| Record | Beats | N | S | V | F | Q | Macro-F1 | Se, V % | Se, S % |
|---|---|---|---|---|---|---|---|---|---|
{s41}

Figure S4.1. Optimiser convergence, objective against evaluation count, mean over 30 seeded runs with the interquartile band, for the six optimisers of Table 7. Source data: `results/optimiser_curves.npz`.

Figure S4.2. Receiver operating characteristic curves per superclass on DS2, reported as the secondary curve to the precision-recall curves of Figure 3. Source data: `results/ds2_predictions.csv`.

Figure S4.3. Confusion matrices for INCART and SVDB, included only if the journal limits Figure 2 of the manuscript to a single panel. Source data: `results/external_predictions.csv`.

Figure S4.4. Slopegraph of macro-F1 for all seven arms under the patient-mixed partition and under DS1 to DS2 partitioning, one line per arm, with arms whose rank changes drawn in a contrasting colour. This is the visual companion to Table 4, and the count of crossing lines is the finding. Source data: `results/numbers.json`, block `table4`.

Build specifications and generation prompts for every figure, main and supplementary, are in `09_Figures_Guide.md`.

# S5. Reproducibility

Repository structure, as deposited:

```
ecg-edge/
  code/
    common.py            partitions, AAMI mapping, filter corners, seeds
    01_download.py       fetches the four PhysioNet databases
    02_preprocess.py     filtering, resampling, segmentation, rhythm descriptors
    03_context.py        five-second context windows for the teacher
    04_teacher.py        teacher fine-tuning and soft-target export
    05_search.py         Grey Wolf feature selection and the TPE searches
    06_train_arms.py     five seeds per arm on DS1 and on the mixed partition
    07_optimisers.py     six optimisers, 30 seeds, matched budget
    08_predict.py        the file that opens DS2, and the external sets
    09_deploy.py         INT8 quantisation-aware training and the device budget
    10_tables.py         every reported metric, from the raw predictions
    11_ablations.py      Table 6 and the arm B2 rule base
    12_calibration.py    temperature, conformal abstention, noise, SQI, alarms
    13_optimiser_stats.py Wilcoxon, Friedman, Nemenyi
    14_bandwidth.py      the 40 Hz against 100 Hz ablation
    15_build_manuscript.py, 16_build_supplement.py
    models.py, train.py, evalutil.py, armb.py, optimisers.py, dataio.py
  results/               every table and figure input, as CSV and JSON
  cache/                 derived beat arrays and fitted arms, not redistributed
  environment.json
```

Seeds: 0 to 4 for model training, 0 to 29 for the optimiser comparison, 12345 for the optimiser objective subsample, 777 for the feature-selection subsample, 4242 for the arm B2 objective subsample. Seeding covers data ordering, weight initialisation, augmentation and optimiser initialisation.

Commands, one per artefact:

| Artefact | Command |
|---|---|
| Databases | `python code/01_download.py` |
| Beat arrays and Table 2 | `python code/02_preprocess.py` |
| Teacher soft targets | `python code/03_context.py && python code/04_teacher.py` |
| Hyperparameters, Tables S3.3 | `python code/05_search.py` |
| Trained arms, Table S3.4 | `python code/06_train_arms.py` |
| Table 7 and Figure S4.1 | `python code/07_optimisers.py && python code/13_optimiser_stats.py` |
| Tables 3, 4, 5, 9 and S4.1 | `python code/08_predict.py && python code/10_tables.py` |
| Table 8 | `python code/09_deploy.py` |
| Table 6 and supplement S6 | `python code/11_ablations.py` |
| Figures 4, 5, 6 and section 7.7 | `python code/12_calibration.py` |
| Table S3.5 | `python code/14_bandwidth.py` |
| Manuscript and supplement | `python code/15_build_manuscript.py && python code/16_build_supplement.py` |

Data access: the four databases are downloaded from PhysioNet by `01_download.py` and are not redistributed {{Goldberger, 2000 #4}}. Archived artefacts: trained weights in FP32 and INT8, the exact beat index lists for every partition, and the computed device budget, deposited at [[AUTHOR ACTION: Zenodo DOI, minted at submission]].

# S6. Learned rule base of arm B2

Arm B2 replaces the class-score inputs of the fuzzy decoder with four named clinical quantities, so that each rule can be read as a diagnostic criterion. Inputs are RR-interval irregularity, the ratio of the current RR interval to the local mean over the preceding ten beats, P-wave presence measured as energy in the 150 ms before QRS onset relative to the local baseline, and QRS duration measured as the width of the high-slope region around the R peak. Each input carries low, mid and high Gaussian membership functions whose centres and widths are fitted by GPC on the DS1 training subset {{Harifi, 2021 #61}}.

**Table S6.1.** The rule base as learned, printed in full. Firing strength is the mean over DS2 beats of the rule's normalised activation, and it is reported so that a rule that never fires can be identified as such.

| Rule | Antecedent | Consequent | Firing strength on DS2 | Fitted rule weight | Status |
|---|---|---|---|---|---|
{s61}

**Table S6.2.** Fitted membership function parameters, in the units of each input. Each cell is the centre followed by the width.

| Input | Low: mean, width | Mid: mean, width | High: mean, width |
|---|---|---|---|
{s62}

Two properties of this rule base decide whether the arm has done its job. Every rule should fire on some beats, since a rule with zero firing strength is a rule the classifier does not use, and {b2.get('n_silent', 0)} of the {b2.get('n_rules', 0)} rules meet that test by never firing on DS2. And each antecedent should correspond to a criterion a cardiologist would recognise: R3 and R5 encode wide QRS with abnormal timing, which is the textbook description of a ventricular ectopic beat, while R2 and R4 encode a premature beat with absent or altered P-wave activity, which is the description of a supraventricular ectopic beat. Where the fitted rule base departs from that reading, section 8.3 says so and leaves the rule as fitted.
"""
    OUT.write_text(DOC, encoding="utf-8")
    print(f"SUPPLEMENT_WRITTEN {len(DOC)} characters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
