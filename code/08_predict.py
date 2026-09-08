"""Stage 6 and 7: produce raw predictions. This is the file that opens DS2.

Nothing here computes a metric. It writes probabilities and predicted classes to
disk, once, for every arm and every seed, on DS2, on the patient-mixed test
partition and on the two external databases. Every table and figure in the
manuscript is then derived from these files by 10_tables.py, which guarantees
that the number in a table, the number in a figure and the number in the
abstract are the same number.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from armb import build_pool
from common import ARMS, CACHE, CLASSES, RESULTS, SEEDS
from dataio import external, mitdb_splits, mixed_partition
from evalutil import softmax
from models import PolySigmoidScorer, ts_fuzzy_decode
from train import arm_a_logits, arm_b2_logits

ARMDIR = CACHE / "arms"


def arm_logits(tag: str, arm: str, seed: int, split) -> np.ndarray:
    obj = pickle.loads((ARMDIR / f"{tag}_{arm}_s{seed}.pkl").read_bytes())
    if arm == "A":
        return arm_a_logits(obj["fit"], split)
    if arm == "B2":
        return arm_b2_logits(obj["fit"], split)
    if arm in ("B", "C"):
        Z = (build_pool(split, obj["mu"], obj["sd"])[0])[:, obj["sel"]]
        sc = PolySigmoidScorer(n_feat=len(obj["sel"]), n_class=5)
        z = sc.scores(Z.astype(np.float64), obj["theta"].astype(np.float64))
        if arm == "B":
            wf = ts_fuzzy_decode(z, obj["sigma_low"], obj["sigma_high"])
            return np.log(np.clip(wf, 1e-12, 1.0))
        return np.log(np.clip(z / np.maximum(z.sum(axis=1, keepdims=True), 1e-12), 1e-12, 1.0))
    return obj["model"].logits(split)


def run(tag: str, split, out_name: str, arms=None, extra_cols=None):
    arms = arms or ARMS
    store, rows = {}, []
    for arm in arms:
        per_seed = []
        for seed in SEEDS:
            lg = arm_logits(tag, arm, seed, split)
            per_seed.append(lg.astype(np.float32))
        L = np.stack(per_seed)                       # (n_seed, n_beat, 5)
        store[f"{arm}_logits"] = L
        P = softmax(L.reshape(-1, 5)).reshape(L.shape).mean(axis=0)
        pred = P.argmax(axis=1)
        d = pd.DataFrame({
            "record": split.rec, "beat_index": split.bidx,
            "true_class": [CLASSES[i] for i in split.y],
            "pred_class": [CLASSES[i] for i in pred],
            "p_N": P[:, 0], "p_S": P[:, 1], "p_V": P[:, 2],
            "p_F": P[:, 3], "p_Q": P[:, 4], "arm": arm})
        if extra_cols:
            for k, v in extra_cols.items():
                d[k] = v
        rows.append(d)
        acc = float((pred == split.y).mean())
        print(f"  {arm}: seed-mean accuracy {acc:.4f}", flush=True)

    np.savez_compressed(CACHE / f"preds_{out_name}.npz",
                        y=split.y, rec=split.rec, bidx=split.bidx, **store)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(RESULTS / f"{out_name}_predictions.csv", index=False)
    print(f"  wrote {out_name}_predictions.csv  ({len(df)} rows)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="ds2,mixed,external")
    a = ap.parse_args()
    what = a.what.split(",")

    if "ds2" in what:
        print("[DS2] opening the held-out partition, once", flush=True)
        _, _, ds2 = mitdb_splits()
        run("ds1", ds2, "ds2")

    if "mixed" in what:
        print("[mixed] patient-mixed partition of the same beats", flush=True)
        _, _, mte = mixed_partition(seed=0)
        run("mixed", mte, "mixed")

    if "external" in what:
        for db, label in (("incartdb", "INCART"), ("svdb", "SVDB")):
            print(f"[{label}]", flush=True)
            sp = external(db)
            run("ds1", sp, db, extra_cols={"dataset": label})
        parts = []
        for db, label in (("incartdb", "INCART"), ("svdb", "SVDB")):
            parts.append(pd.read_csv(RESULTS / f"{db}_predictions.csv"))
        pd.concat(parts, ignore_index=True).to_csv(
            RESULTS / "external_predictions.csv", index=False)
        print("  wrote external_predictions.csv", flush=True)

    print("PREDICT_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
