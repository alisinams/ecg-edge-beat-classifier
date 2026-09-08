"""Train the seven arms on DS1 with the selected hyperparameters and five seeds.

Arms B and C share stage one weight for weight: stage one is fitted once per
seed and both decoders read the same parameter vector, so Table 6's first row
isolates the decoder and nothing else.

Nothing in this file reads DS2. It writes fitted objects to cache/arms/ and the
DS1 validation performance to results/train_performance.json.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from armb import BOX_HI, BOX_LO, StageOneObjective, build_pool
from common import CACHE, RESULTS, SEEDS, jdump, jload
from dataio import class_weights, mitdb_splits
from evalutil import macro_f1, selection_f1
from models import ts_fuzzy_decode
from optimisers import Budget, whoa
from train import NeuralArm, arm_a_logits, fit_arm_a, fit_arm_b2

ARMDIR = CACHE / "arms"
ARMDIR.mkdir(parents=True, exist_ok=True)


def teacher_for(tr):
    """Soft targets by (record, annotation index).

    Two files are merged. The DS1 file is the one the inter-patient pipeline
    uses. The DS2 file exists only so that the patient-mixed refit of section
    7.3 has supervision for the beats it draws from DS2 records; it is produced
    from the same teacher checkpoint and it is never read when arm E is fitted
    on DS1.
    """
    table = {}
    for logits_name, ctx_name in (("teacher_logits.npz", "ds1_context.npz"),
                                  ("teacher_logits_ds2.npz", None)):
        p = CACHE / logits_name
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        # materialise once: indexing an NpzFile decompresses the whole array each time
        L = np.asarray(z["logits"])
        if ctx_name is not None:
            zc = np.load(CACHE / ctx_name, allow_pickle=True)
            recs, bidx = zc["rec"].astype(str), zc["bidx"]
        else:
            recs, bidx = z["rec"].astype(str), z["bidx"]
        for i, (r, b) in enumerate(zip(recs, bidx)):
            table[(str(r), int(b))] = L[i]
    out = np.zeros((len(tr.y), 5), dtype=np.float32)
    hit = 0
    for i, (r, b) in enumerate(zip(tr.rec, tr.bidx)):
        v = table.get((str(r), int(b)))
        if v is not None:
            out[i] = v
            hit += 1
    return out, hit / len(tr.y)


def fit_stage_one(tr, va, sel, mu, sd, pop, seed, budget=15000):
    Ztr = (build_pool(tr, mu, sd)[0])[:, sel]
    obj = StageOneObjective(Ztr, tr.y, class_weights(tr.y))
    b = Budget(obj.loss_numpy, budget)
    whoa(b, obj.dim, BOX_LO, BOX_HI, seed=seed, pop=pop)
    return b.best_x, b.best, obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B,B2,C,D,E,F")
    ap.add_argument("--tag", default="ds1")
    ap.add_argument("--mixed", action="store_true",
                    help="fit on the patient-mixed partition instead of DS1")
    a = ap.parse_args()

    if a.mixed:
        from dataio import mixed_partition
        tr, va, _ = mixed_partition(seed=0)
    else:
        tr, va, _ = mitdb_splits()
    print(f"{tr}  {va}", flush=True)

    hp = jload(RESULTS / "hyperparams.json")
    sel = np.array(hp["featsel"]["selected"])
    mu = np.array(hp["featsel"]["mu"], dtype=np.float32)
    sd = np.array(hp["featsel"]["sd"], dtype=np.float32)

    perf_path = RESULTS / f"train_performance_{a.tag}.json"
    perf = jload(perf_path) if perf_path.exists() else {}
    want = a.arms.split(",")

    teacher = None
    if "E" in want:
        teacher, cov = teacher_for(tr)
        print(f"  teacher targets cover {cov*100:.1f}% of training beats", flush=True)

    for arm in want:
        t0 = time.time()
        print(f"[{arm}]", flush=True)
        recs = []
        for seed in SEEDS:
            out = ARMDIR / f"{a.tag}_{arm}_s{seed}.pkl"
            if out.exists():
                recs.append(pickle.loads(out.read_bytes())["history"])
                continue
            if arm == "A":
                fit = fit_arm_a(tr, va, seed=seed, hp=hp["A"]["best_params"])
                pt = arm_a_logits(fit, tr).argmax(1)
                hist = dict(val_selection_f1=fit["val_selection_f1"],
                            val_macro_f1=fit["val_macro_f1"],
                            train_macro_f1=float(macro_f1(tr.y, pt)),
                            train_selection_f1=float(selection_f1(tr.y, pt)),
                            chosen=fit["name"], per_model=fit["per_model"])
                obj = dict(kind="A", fit=fit, history=hist)
            elif arm in ("B", "C"):
                pop = int(hp["B"]["best_params"]["whoa_pop"])
                sl = float(hp["B"]["best_params"]["sigma_low"])
                sh = float(hp["B"]["best_params"]["sigma_high"])
                shared = ARMDIR / f"{a.tag}_stage1_s{seed}.npz"
                if shared.exists():
                    theta = np.load(shared)["theta"]
                    fval = float(np.load(shared)["objective"])
                else:
                    theta, fval, _ = fit_stage_one(tr, va, sel, mu, sd, pop, seed)
                    np.savez(shared, theta=theta, objective=fval)
                Zva = (build_pool(va, mu, sd)[0])[:, sel]
                Ztr = (build_pool(tr, mu, sd)[0])[:, sel]
                from models import PolySigmoidScorer
                sc = PolySigmoidScorer(n_feat=len(sel), n_class=5)
                zva = sc.scores(Zva.astype(np.float64), theta.astype(np.float64))
                ztr = sc.scores(Ztr.astype(np.float64), theta.astype(np.float64))
                if arm == "B":
                    pv, pt = ts_fuzzy_decode(zva, sl, sh).argmax(1), ts_fuzzy_decode(ztr, sl, sh).argmax(1)
                else:
                    pv, pt = zva.argmax(1), ztr.argmax(1)
                hist = dict(val_selection_f1=float(selection_f1(va.y, pv)),
                            val_macro_f1=float(macro_f1(va.y, pv)),
                            train_macro_f1=float(macro_f1(tr.y, pt)),
                            train_selection_f1=float(selection_f1(tr.y, pt)),
                            stage_one_objective=fval, params=int(sc.n_params))
                obj = dict(kind=arm, theta=theta, sel=sel, mu=mu, sd=sd,
                           sigma_low=sl, sigma_high=sh, history=hist)
            elif arm == "B2":
                pop = int(hp["B2"]["best_params"]["gpc_pop"])
                fit = fit_arm_b2(tr, va, seed=seed, budget=15000, pop=pop)
                hist = dict(val_selection_f1=fit["val_selection_f1"],
                            val_macro_f1=fit["val_macro_f1"],
                            train_macro_f1=fit["train_macro_f1"],
                            train_selection_f1=fit["train_selection_f1"])
                obj = dict(kind="B2", fit=fit, history=hist)
            else:
                bp = dict(hp[arm]["best_params"])
                bp.pop("qat_start_epoch", None)
                bp["gain_lo"] = 1.0 / bp.get("gain_hi", 1.33)
                bp["batch"] = 256
                m = NeuralArm(arm, bp).fit(tr, va, seed=seed, epochs=200, patience=20,
                                           teacher_logits=teacher if arm == "E" else None)
                hist = m.history
                obj = dict(kind=arm, model=m, history=hist)
            hist["wallclock_min"] = (time.time() - t0) / 60.0 / max(1, SEEDS.index(seed) + 1)
            out.write_bytes(pickle.dumps(obj))
            recs.append(hist)
            print(f"  seed {seed}: val macro-F1 {hist['val_macro_f1']:.4f}", flush=True)

        vals = [r["val_macro_f1"] for r in recs]
        sels = [r["val_selection_f1"] for r in recs]
        trs = [r["train_macro_f1"] for r in recs]
        extra = {}
        if arm == "A":
            names = [r.get("chosen") for r in recs if r.get("chosen")]
            if names:
                extra["chosen"] = max(set(names), key=names.count)
                extra["chosen_per_seed"] = names
        perf[arm] = dict(**extra,
            val_macro_f1_mean=float(np.mean(vals)), val_macro_f1_sd=float(np.std(vals, ddof=1)),
            val_selection_f1_mean=float(np.mean(sels)),
            val_selection_f1_sd=float(np.std(sels, ddof=1)),
            train_macro_f1_mean=float(np.mean(trs)),
            train_selection_f1_mean=float(np.mean([r["train_selection_f1"] for r in recs])),
            epochs_to_stop=[int(r.get("epochs_to_stop", 0)) for r in recs],
            wallclock_min=float((time.time() - t0) / 60.0),
            per_seed=vals)
        jdump(perf, perf_path)
        print(f"  mean val macro-F1 {np.mean(vals):.4f} +- {np.std(vals, ddof=1):.4f} "
              f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    print("TRAIN_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
