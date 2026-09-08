"""Two corrections to the calibration and abstention record.

First, coverage. Split conformal prediction guarantees that the prediction SET
contains the true label with probability at least 1 - epsilon. That is not the
same quantity as the fraction of beats the classifier is willing to answer,
which is the fraction whose set holds exactly one class. An earlier version
reported the second and called it coverage. Both are computed here and named
separately: `conformal_coverage` is the quantity the guarantee bounds, and
`retention` is the fraction of beats not abstained on.

Second, monotonicity. Temperature scaling is monotone for one model, so it
cannot change that model's argmax. This study reports the mean of five seeds'
probabilities, and the mean of five temperature-scaled distributions is not a
monotone transform of the mean of the five unscaled ones, so a small number of
ensemble decisions do change. The count is measured here instead of assumed.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, RESULTS, SEEDS, jdump, jload
from dataio import external, mitdb_splits
from evalutil import conformal_threshold, softmax

EPS = 0.10
_STUDENTS = None


def students():
    global _STUDENTS
    if _STUDENTS is None:
        _STUDENTS = [pickle.loads((CACHE / "arms" / f"ds1_E_s{s}.pkl").read_bytes())["model"]
                     for s in SEEDS]
    return _STUDENTS


def mean_probs(split, T=1.0):
    L = np.stack([m.logits(split) for m in students()])
    return softmax((L / T).reshape(-1, 5)).reshape(L.shape).mean(axis=0)


def main() -> int:
    cal = jload(RESULTS / "calibration.json")
    ab = jload(RESULTS / "abstention.json")
    T = float(cal["temperature"])
    tr, va, ds2 = mitdb_splits()

    p_before = mean_probs(ds2, 1.0)
    p_after = mean_probs(ds2, T)
    changed = int((p_before.argmax(1) != p_after.argmax(1)).sum())
    cal["decisions_changed_by_scaling"] = changed
    cal["decisions_changed_fraction"] = float(changed / len(ds2.y))
    cal["monotonicity_note"] = (
        "Temperature scaling is monotone for a single model and cannot change its "
        "argmax. The reported classifier is the mean of five seeds' probabilities, "
        "and averaging is applied after scaling, so the ensemble argmax can move. "
        f"It moved on {changed} of {len(ds2.y)} DS2 beats."
    )
    jdump(cal, RESULTS / "calibration.json")
    print(f"  temperature scaling moved {changed} of {len(ds2.y)} DS2 decisions "
          f"({100*changed/len(ds2.y):.3f} per cent)", flush=True)

    p_cal = mean_probs(va, T)
    qhat = conformal_threshold(p_cal, va.y, EPS)
    for label, sp in (("DS2", ds2), ("INCART", external("incartdb")),
                      ("SVDB", external("svdb"))):
        p = mean_probs(sp, T)
        sets = p >= (1.0 - qhat)
        contains = sets[np.arange(len(sp.y)), sp.y]
        singleton = sets.sum(axis=1) <= 1
        d = ab.setdefault(label, {})
        d["conformal_coverage"] = float(contains.mean())
        d["retention"] = float(singleton.mean())
        d["mean_set_size"] = float(sets.sum(axis=1).mean())
        d["empty_set_rate"] = float((sets.sum(axis=1) == 0).mean())
        print(f"  {label}: conformal coverage {100*contains.mean():.1f} per cent against "
              f"a {100*(1-EPS):.0f} per cent target, retention {100*singleton.mean():.1f}, "
              f"mean set size {sets.sum(axis=1).mean():.2f}", flush=True)
    ab["definitions"] = {
        "conformal_coverage": "fraction of beats whose prediction set contains the true "
                              "class; this is the quantity the split conformal guarantee "
                              "bounds at 1 - epsilon",
        "retention": "fraction of beats whose prediction set holds at most one class, so "
                     "the classifier answers instead of abstaining",
        "coverage": "retained for backward compatibility; equal to retention",
    }
    jdump(ab, RESULTS / "abstention.json")
    print("CONFORMAL_FIX_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
