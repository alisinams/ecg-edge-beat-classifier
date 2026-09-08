"""Table S3.5: the cost of the 40 Hz upper corner, measured against 100 Hz.

The whole DS1 experiment is repeated on beats filtered to 0.5-100 Hz instead of
0.5-40 Hz, with the same architecture, the same hyperparameters and the same
five seeds, and scored on the DS1 validation subset. DS2 is not involved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DS1_TRAIN, DS1_VAL, RESULTS, SEEDS, jdump, jload
from dataio import Split, mitdb_splits
from evalutil import confusion, metrics_from_cm
from train import NeuralArm


def splits_from(npz_name):
    z = np.load(CACHE / npz_name, allow_pickle=True)
    rec = z["rec"].astype(str)
    def take(recs, name):
        m = np.isin(rec, recs)
        return Split(z["X"][m], z["y"][m], z["R"][m], rec[m], z["bidx"][m], name)
    return take(DS1_TRAIN, "train"), take(DS1_VAL, "val")


def run(tr, va, hp, label):
    cms = []
    for seed in SEEDS:
        m = NeuralArm("E", hp).fit(tr, va, seed=seed, epochs=200, patience=20)
        cms.append(confusion(va.y, m.logits(va).argmax(1)))
    mm = [metrics_from_cm(c) for c in cms]
    return dict(label=label,
                macro_f1=float(np.mean([x["macro_f1"] for x in mm])),
                macro_f1_sd=float(np.std([x["macro_f1"] for x in mm], ddof=1)),
                se_v=float(np.mean([x["se"][2] for x in mm])),
                se_s=float(np.mean([x["se"][1] for x in mm])))


def main() -> int:
    hp = jload(RESULTS / "hyperparams.json")
    bp = dict(hp["E"]["best_params"])
    bp.pop("qat_start_epoch", None)
    bp["gain_lo"] = 1.0 / bp.get("gain_hi", 1.33)
    bp["batch"] = 256

    tr40, va40, _ = mitdb_splits()
    r40 = run(tr40, va40, bp, "40 Hz, used in the manuscript")
    print(f"  40 Hz : macro-F1 {r40['macro_f1']:.4f}  Se V {r40['se_v']:.3f}  "
          f"Se S {r40['se_s']:.3f}", flush=True)

    tr100, va100 = splits_from("mitdb_bw100.npz")
    r100 = run(tr100, va100, bp, "100 Hz")
    print(f"  100 Hz: macro-F1 {r100['macro_f1']:.4f}  Se V {r100['se_v']:.3f}  "
          f"Se S {r100['se_s']:.3f}", flush=True)

    jdump(dict(rows=[r40, r100],
               delta_macro_f1=r100["macro_f1"] - r40["macro_f1"],
               note="Scored on the DS1 validation subset only; DS2 is not used here. "
                    "The student arm topology and hyperparameters are held fixed, so "
                    "the only change is the upper filter corner."),
          RESULTS / "bandwidth_ablation.json")
    print("BANDWIDTH_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
