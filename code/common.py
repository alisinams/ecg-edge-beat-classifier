"""Shared constants and helpers for the ECG edge study.

Everything that more than one script needs lives here so that a partition list,
a class order or a filter corner is defined exactly once.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ECG_DATA_ROOT", ROOT / "data"))
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _p in (CACHE, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- partitions
# ANSI/AAMI EC57 excludes the four paced records.
PACED = ["102", "104", "107", "217"]

# de Chazal et al. 2004 record split.
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
       "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
       "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
       "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
       "233", "234"]

# DS1 is carved into a training and a validation subset by record, under three
# constraints fixed before any model was trained and never revisited afterwards.
# Every AAMI superclass present in DS1 must appear in both subsets, which forces
# one of records 101, 203 and 208 into the validation subset because those are
# the only DS1 records holding a Q beat. The validation subset must hold roughly
# a fifth of the DS1 beats. And both bundle-branch-block records of DS1, 118 and
# 124, must stay in the training subset: a right-bundle-branch-block beat is
# labelled N and has a wide QRS, so a classifier that has never seen one calls
# it ventricular, and the validation subset then measures the absence of those
# two records rather than the model.
DS1_VAL = ["101", "114", "203", "220", "223"]
DS1_TRAIN = [r for r in DS1 if r not in DS1_VAL]

# ------------------------------------------------------------- AAMI mapping
CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

AAMI = {
    # N: normal and bundle-branch block, plus atrial and nodal escape
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # S: supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # V: ventricular ectopic
    "V": "V", "E": "V",
    # F: fusion of ventricular and normal
    "F": "F",
    # Q: unclassifiable or paced
    "/": "Q", "f": "Q", "Q": "Q",
}

# ------------------------------------------------------------ signal config
FS = 250                 # Hz, common resampling rate
BP_LOW = 0.5             # Hz, high-pass corner
BP_HIGH = 40.0           # Hz, low-pass corner (monitoring bandwidth)
PRE_MS = 250             # ms before the R peak
POST_MS = 400            # ms after the R peak
WIN = int(round(FS * (PRE_MS + POST_MS) / 1000.0)) + 1   # 163 samples
PRE_N = int(round(FS * PRE_MS / 1000.0))                 # 62
POST_N = int(round(FS * POST_MS / 1000.0))               # 100
SKIP_FIRST = 10          # first ten beats of every record are dropped

N_RHYTHM = 4             # rr_prev, rr_next, rr_prev/local10, rr_prev/rr_next

SEEDS = [0, 1, 2, 3, 4]
OPT_SEEDS = list(range(30))
ARMS = ["A", "B", "B2", "C", "D", "E", "F"]


def jdump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def jload(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
