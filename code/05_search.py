"""Stage 3 of the roadmap: hyperparameter search on the DS1 validation subset.

One hundred tree-structured Parzen estimator trials per arm, scored on DS1
validation selection-F1. Neural trials use a shortened schedule (60 epochs, patience
8); the configuration that wins is then retrained to convergence with five seeds
by 06_train_arms.py. DS2 is not touched anywhere in this file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna


def _tick(study, trial):
    """One line per finished trial, so a slow arm is visible while it runs."""
    print(f"    trial {trial.number:3d} value {trial.value if trial.value is not None else float('nan'):.4f} "
          f"best {study.best_value:.4f}", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
optuna.logging.set_verbosity(optuna.logging.WARNING)

from armb import BOX_HI, BOX_LO, D_SELECT, StageOneObjective, build_pool, gwo_select
from common import CACHE, RESULTS, jdump, jload
from dataio import class_weights, mitdb_splits
from evalutil import macro_f1, selection_f1
from models import ts_fuzzy_decode
from optimisers import Budget, whoa
from train import NeuralArm, fit_arm_a, fit_arm_b2

N_TRIALS = {"A": 40, "B": 100, "B2": 30, "D": 100, "E": 100, "F": 30}
SEARCH_EPOCHS, SEARCH_PATIENCE = 60, 8
# The Grey Wolf wrapper is scored on a fixed stratified subsample so that a
# 20-wolf, 30-iteration search finishes; the selected subset is then used on
# the whole training subset everywhere else.
GWO_BEATS = 8000


def stratified(y, n, seed):
    rng = np.random.default_rng(seed)
    cls, cnt = np.unique(y, return_counts=True)
    share = np.maximum(1, np.round(n * cnt / cnt.sum()).astype(int))
    return np.sort(np.concatenate([
        rng.choice(np.flatnonzero(y == c), size=min(k, int((y == c).sum())),
                   replace=False) for c, k in zip(cls, share)]))


def teacher_for(tr):
    """Align the teacher logits to the beat cache by (record, annotation index)."""
    z = np.load(CACHE / "teacher_logits.npz", allow_pickle=True)
    zc = np.load(CACHE / "ds1_context.npz", allow_pickle=True)
    key = {(r, int(b)): i for i, (r, b) in
           enumerate(zip(zc["rec"].astype(str), zc["bidx"]))}
    # materialise once: indexing an NpzFile decompresses the whole array each time
    L = np.asarray(z["logits"])
    out = np.zeros((len(tr.y), 5), dtype=np.float32)
    hit = 0
    for i, (r, b) in enumerate(zip(tr.rec, tr.bidx)):
        j = key.get((str(r), int(b)))
        if j is None:
            out[i] = 0.0                      # uniform soft target where no context fits
        else:
            out[i] = L[j]
            hit += 1
    return out, hit / len(tr.y)


def search_neural(arm, tr, va, teacher=None):
    def objective(trial):
        gh = trial.suggest_float("gain_hi", 1.0, 1.6)
        hp = dict(
            width=trial.suggest_float("width", 0.25, 1.0),
            blocks=trial.suggest_int("blocks", 3, 6),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            weight_cap=trial.suggest_float("weight_cap", 2.0, 30.0),
            noise=trial.suggest_float("noise", 0.0, 0.15),
            drift=trial.suggest_float("drift", 0.0, 0.3),
            shift=trial.suggest_int("shift", 0, 16),
            gain_hi=gh, gain_lo=1.0 / gh,
            batch=256,
        )
        if arm == "E":
            hp["temperature"] = trial.suggest_float("temperature", 1.0, 10.0)
            hp["alpha"] = trial.suggest_float("alpha", 0.1, 0.9)
            trial.suggest_int("qat_start_epoch", 10, 40)     # used in stage 5
        m = NeuralArm(arm, hp).fit(tr, va, seed=0, epochs=SEARCH_EPOCHS,
                                   patience=SEARCH_PATIENCE,
                                   teacher_logits=teacher)
        return m.history["val_selection_f1"]

    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(objective, n_trials=N_TRIALS[arm], show_progress_bar=False, callbacks=[_tick])
    return st


def search_a(tr, va):
    def objective(trial):
        hp = dict(C=trial.suggest_float("C", 1e-3, 1e3, log=True),
                  rf_trees=trial.suggest_int("rf_trees", 100, 1000),
                  gb_depth=trial.suggest_int("gb_depth", 3, 10),
                  gb_lr=trial.suggest_float("gb_lr", 0.01, 0.3, log=True))
        return fit_arm_a(tr, va, seed=0, hp=hp)["val_selection_f1"]

    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(objective, n_trials=N_TRIALS["A"], show_progress_bar=False, callbacks=[_tick])
    return st


def search_b(tr, va, sel, mu, sd):
    """Arm B stage one is fitted by WHOA, as in the source architecture.
    The searched quantities are the WHOA population size and the two membership
    widths of the fuzzy decoder; the evaluation budget is fixed at 15,000."""
    Ztr = ((build_pool(tr, mu, sd)[0])[:, sel])
    Zva = ((build_pool(va, mu, sd)[0])[:, sel])
    obj = StageOneObjective(Ztr, tr.y, class_weights(tr.y))

    def objective(trial):
        pop = trial.suggest_int("whoa_pop", 20, 200)
        sl = trial.suggest_float("sigma_low", 0.05, 1.0)
        sh = trial.suggest_float("sigma_high", 0.05, 1.0)
        b = Budget(obj.loss_numpy, 15000)
        whoa(b, obj.dim, BOX_LO, BOX_HI, seed=0, pop=pop)
        z = obj.scores(Zva, b.best_x)
        return selection_f1(va.y, ts_fuzzy_decode(z, sl, sh).argmax(1))

    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(objective, n_trials=N_TRIALS["B"], show_progress_bar=False, callbacks=[_tick])
    return st


def search_b2(tr, va):
    def objective(trial):
        pop = trial.suggest_int("gpc_pop", 20, 200)
        return fit_arm_b2(tr, va, seed=0, budget=15000, pop=pop)["val_selection_f1"]

    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(objective, n_trials=N_TRIALS["B2"], show_progress_bar=False, callbacks=[_tick])
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="D,E,F,B,B2,A")
    a = ap.parse_args()
    tr, va, _ = mitdb_splits()
    print(f"{tr}  {va}", flush=True)
    out_path = RESULTS / "hyperparams.json"
    hp_all = jload(out_path) if out_path.exists() else {}

    # feature selection is shared by B, C and the optimiser comparison
    if "featsel" not in hp_all:
        Ptr, mu, sd = build_pool(tr)
        Pva, _, _ = build_pool(va, mu, sd)
        t0 = time.time()
        sub = stratified(tr.y, GWO_BEATS, 777)
        sel, f1, ncand = gwo_select(Ptr[sub], tr.y[sub], Pva, va.y,
                                    d_select=D_SELECT, seed=0)
        hp_all["featsel"] = dict(selected=sel.tolist(), wrapper_val_macro_f1=f1,
                                 unique_subsets_evaluated=ncand,
                                 wrapper_beats=int(GWO_BEATS),
                                 mu=mu.tolist(), sd=sd.tolist(),
                                 minutes=(time.time() - t0) / 60.0)
        jdump(hp_all, out_path)
        print(f"  GWO selected {sel.tolist()} (wrapper macro-F1 {f1:.4f})", flush=True)
    sel = np.array(hp_all["featsel"]["selected"])
    mu = np.array(hp_all["featsel"]["mu"], dtype=np.float32)
    sd = np.array(hp_all["featsel"]["sd"], dtype=np.float32)

    teacher, cover = None, None
    if "E" in a.arms:
        teacher, cover = teacher_for(tr)
        print(f"  teacher soft targets cover {cover*100:.1f}% of DS1 training beats",
              flush=True)

    for arm in a.arms.split(","):
        if arm in hp_all and arm != "featsel":
            print(f"[{arm}] already searched, skipping", flush=True)
            continue
        t0 = time.time()
        print(f"[{arm}] TPE search", flush=True)
        if arm == "A":
            st = search_a(tr, va)
        elif arm == "B":
            st = search_b(tr, va, sel, mu, sd)
        elif arm == "B2":
            st = search_b2(tr, va)
        else:
            st = search_neural(arm, tr, va, teacher if arm == "E" else None)
        hp_all[arm] = dict(best_params=st.best_params, best_value=float(st.best_value),
                           n_trials=len(st.trials), minutes=(time.time() - t0) / 60.0)
        if arm == "E":
            hp_all[arm]["teacher_coverage"] = cover
        jdump(hp_all, out_path)
        print(f"  best {st.best_value:.4f} {st.best_params} "
              f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    print("SEARCH_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
