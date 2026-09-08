# Pipeline

Everything in the manuscript is produced by the scripts in this directory, in
the order below. Nothing is edited by hand between stages, and no metric is
computed anywhere except in `evalutil.py`, applied to the raw prediction files.

## The rule that shapes the layout

DS2 is opened in exactly one file, `08_predict.py`. Every stage before it reads
DS1 only. Every stage after it reads the stored predictions and never the model.
That is what makes the claim in section 6.1 checkable rather than promised: to
find a leak you only have to read one file.

A second rule: the number in the abstract, the number in a table, the number in
a figure caption and the number in the conclusion are the same Python value.
`10_tables.py` writes `results/numbers.json`, and the four document builders read
from it. No builder recomputes anything.

## Order

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `01_download.py` | PhysioNet | `data/` |
| 2 | `02_preprocess.py` | `data/` | `cache/*.npz`, `results/partition_counts*.csv` |
| 3 | `03_context.py` | `data/`, DS1 only | `cache/ds1_context.npz` |
| 4 | `04_teacher.py` | DS1 contexts | `cache/teacher_logits.npz`, `cache/teacher_state.pt` |
| 4b | `04b_teacher_ds2.py` | DS2 records, teacher checkpoint | `cache/teacher_logits_ds2.npz`, used by the patient-mixed arm only |
| 5 | `05_search.py` | DS1 train and validation | `results/hyperparams.json` |
| 6 | `06_train_arms.py` | DS1, or the mixed partition | `cache/arms/*.pkl`, `results/train_performance_*.json` |
| 7 | `07_optimisers.py` | DS1 train subsample | `results/optimiser_runs.csv`, `results/optimiser_curves.npz` |
| 8 | `08_predict.py` | **DS2**, mixed, INCART, SVDB | `cache/preds_*.npz`, `results/*_predictions.csv` |
| 9 | `09_deploy.py` | arm E checkpoints, DS2 | `results/deployment.json` |
| 10 | `12_calibration.py` | arm E checkpoints, all sets | `results/calibration.json`, `abstention.json`, `noise_sweep.*`, `sqi_rejections.csv`, `alarms.json` |
| 11 | `11_ablations.py` | predictions, extra fits | `results/ablations.json`, `results/b2_rules.json` |
| 12 | `14_bandwidth.py` | DS1 at two filter corners | `results/bandwidth_ablation.json` |
| 13 | `13_optimiser_stats.py` | optimiser runs | `results/optimiser_stats.json` |
| 14 | `10_tables.py` | everything above | `results/numbers.json` |
| 15 | `18_figure_data.py` | predictions and numbers | `results/figure_data/*.csv` |
| 16 | `15_build_manuscript.py` | `numbers.json` | `05_Manuscript.md` |
| 17 | `16_build_supplement.py` | `numbers.json`, checklists | `06_Supplementary.md` |
| 18 | `19_build_persian_report.py` | `numbers.json` | `12_Persian_Report.md` |
| 19 | `20_update_figures_guide.py` | `numbers.json`, figure data | `09_Figures_Guide.md` |
| 20 | `17_build_docx.py` | the two Markdown files | `08_*.docx` |
| 21 | `21_check_citations.py` | manuscript, `references.ris` | pass or a list of mismatches |
| 22 | `22_style_audit.py` | manuscript, supplement | `results/style_audit.json` |

`run_rest.sh` runs steps 4b to 22 in that order and stops at the first failure.

## Modules

`common.py` holds the partitions, the AAMI mapping, the filter corners and the
seeds, so a record list is defined once. `dataio.py` decides what "the DS1
training subset" means. `models.py` defines the seven arms. `train.py` holds the
training loops and the augmentation. `armb.py` holds the arm B objective and the
Grey Wolf feature selection. `optimisers.py` holds the six optimisers behind one
budget interface. `evalutil.py` holds every metric and the record-level
bootstrap. `fmt.py` and `msctx.py` serve the document builders.

## Two things a reader should check first

The class weights are capped, and `dataio.class_weights` says why in its
docstring: the Q superclass holds two beats in the DS1 training subset.

The DS1 validation split is chosen under three stated constraints in
`common.py`, one of which keeps both bundle-branch-block records in training.
Without that constraint the validation subset measures the absence of those
records instead of the model, and the V class collapses.
