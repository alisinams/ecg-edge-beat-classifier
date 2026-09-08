# ECG arrhythmia classification on microcontroller-class wearables

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22662944.svg)](https://doi.org/10.5281/zenodo.22662944)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code, trained weights, partition manifests and per-beat predictions for the paper:

> **A Deep Learning ECG Arrhythmia Classifier for Microcontroller-Class Wearables: Inter-Patient Evaluation, External Validation and Embedded Deployment Analysis**
> Alisina Mousavi, Mahdi Bazargani
> Department of Computer Engineering, Za.C., Islamic Azad University, Zanjan, Iran
> Submitted to *Biomedical Signal Processing and Control*.

Everything reported in the manuscript is produced by the scripts in [`code/`](code/), in the order given in [`code/README.md`](code/README.md). No metric is computed anywhere except in `code/evalutil.py`, applied to the stored prediction files in [`results/predictions/`](results/predictions/).

## Data

This repository does **not** redistribute the raw waveforms. All four databases are public and are distributed through PhysioNet under the Open Data Commons Attribution License v1.0:

| Database | DOI |
| --- | --- |
| MIT-BIH Arrhythmia Database | [10.13026/C2F305](https://doi.org/10.13026/C2F305) |
| St Petersburg INCART 12-lead Arrhythmia Database | [10.13026/C2V88N](https://doi.org/10.13026/C2V88N) |
| MIT-BIH Supraventricular Arrhythmia Database | [10.13026/C2V30W](https://doi.org/10.13026/C2V30W) |
| MIT-BIH Noise Stress Test Database | [10.13026/C2HS3T](https://doi.org/10.13026/C2HS3T) |

`code/01_download.py` fetches all four into `data/`. What this repository does supply is everything needed to check the published numbers without re-downloading or re-training anything.

## Layout

```
code/         all pipeline scripts and modules, run order in code/README.md
models/       trained weights
  arms/         FP32 weights of the seven arms, five seeds each, under both
                the inter-patient (ds1_*) and the patient-mixed (mixed_*)
                partition, plus the four ablation fits
  E_int8_s*.pt  INT8 quantisation-aware-trained student, five seeds
  deploy_cms.npz  CMSIS-NN deployment tensors
partitions/   the exact record and beat lists behind every split
  record_lists.json    DS1, DS1_train, DS1_val, DS2, the excluded paced
                       records, the external test sets and the seeds
  mitdb_beats.csv.gz   every one of the 100,268 MIT-BIH beats with its
                       record, beat index, AAMI superclass and partition
  incart_beats.csv.gz  175,100 INCART beats
  svdb_beats.csv.gz    183,768 SVDB beats
results/      every aggregate the manuscript quotes
  numbers.json         the single source of every number in the paper
  environment.json     hardware and library versions of the reported run
  hyperparams.json     the selected hyperparameters
  figure_data/         the plotted values of every figure
  predictions/         per-beat posterior probabilities, gzipped CSV, for
                       DS2, the patient-mixed refit, INCART and SVDB
figures/      the five main figures and the six supplementary figures,
              vector PDF and TIFF
```

Each file under `results/predictions/` carries one row per beat per arm, with columns `record, beat_index, true_class, pred_class, p_N, p_S, p_V, p_F, p_Q, arm`. The working tree also holds an `external_predictions.csv`; it is omitted here because it is the exact concatenation of the INCART and SVDB files.

## Partitions

The inter-patient split is the de Chazal et al. 2004 record split of the MIT-BIH Arrhythmia Database, with the four paced records excluded as ANSI/AAMI EC57 requires. DS1 is carved into a training and a validation subset by record under three constraints that were fixed before any model was trained; `code/common.py` states them. The first ten beats of every record are dropped.

DS2 is opened in exactly one file, `code/08_predict.py`. Every stage before it reads DS1 only, and every stage after it reads the stored predictions and never the model.

## Reproducing

```
pip install -r requirements.txt
python code/01_download.py      # fetches the four PhysioNet databases into data/
bash  code/run_rest.sh          # steps 4b to 22
```

The reported run used Python 3.13.1 with PyTorch 2.11.0+cu128 on an NVIDIA GeForce RTX 3090; `results/environment.json` records the full environment. Seeds are 0 to 4 for the arms and 0 to 29 for the optimiser comparison, listed in `partitions/record_lists.json` and fixed in `code/common.py`.

To recompute the published metrics from the shipped predictions alone, without training, run `python code/10_tables.py`, which rewrites `results/numbers.json` from the prediction files.

## Licence

Code and documentation: MIT, see [LICENSE](LICENSE). The derived files under `models/`, `partitions/` and `results/` are released under CC BY 4.0. The underlying PhysioNet recordings remain under their own licence and are not redistributed here.

## Citation

Every release of this repository is archived on Zenodo. The concept DOI
[10.5281/zenodo.22662944](https://doi.org/10.5281/zenodo.22662944) always resolves to the latest
version; release `v1.0.0` is [10.5281/zenodo.22662945](https://doi.org/10.5281/zenodo.22662945).
Machine-readable metadata is in [CITATION.cff](CITATION.cff).
