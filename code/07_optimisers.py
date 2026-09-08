"""Stage 4 of the roadmap: six optimisers, 30 seeded runs, one matched budget.

All six fit the same 205 parameters of arm B's stage one against the same
objective inside the same box, and each stops at exactly 15,000 objective
evaluations. Writes results/optimiser_runs.csv and results/optimiser_curves.npz.

The objective is evaluated on a fixed stratified subsample of the DS1 training
subset, identical for every optimiser and every seed, so that 180 runs of 15,000
evaluations are affordable. The subsample is drawn once with seed 12345 and
stored, so the comparison is exactly reproducible.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from armb import BOX_HI, BOX_LO, StageOneObjective, build_pool
from common import OPT_SEEDS, RESULTS, jdump, jload
from dataio import class_weights, mitdb_splits
from optimisers import METAPHOR, OPTIMISERS, run_one

BUDGET = 15000
SUBSAMPLE = 6000
SUB_SEED = 12345


def stratified_subsample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    share = np.maximum(1, np.round(n * counts / counts.sum()).astype(int))
    idx = []
    for c, k in zip(classes, share):
        pool = np.flatnonzero(y == c)
        idx.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
    return np.sort(np.concatenate(idx))


def main() -> int:
    hp = jload(RESULTS / "hyperparams.json")
    sel = np.array(hp["featsel"]["selected"])
    mu = np.array(hp["featsel"]["mu"], dtype=np.float32)
    sd = np.array(hp["featsel"]["sd"], dtype=np.float32)

    tr, _, _ = mitdb_splits()
    P, _, _ = build_pool(tr, mu, sd)
    keep = stratified_subsample(tr.y, SUBSAMPLE, SUB_SEED)
    obj = StageOneObjective(P[keep][:, sel], tr.y[keep], class_weights(tr.y))
    print(f"objective on {len(keep)} beats, {obj.dim} parameters, "
          f"box [{BOX_LO}, {BOX_HI}]", flush=True)

    rows, curves = [], {}
    for name in OPTIMISERS:
        t0 = time.time()
        for s in OPT_SEEDS:
            r = run_one(name, obj, obj.dim, BOX_LO, BOX_HI, s, BUDGET)
            rows.append(dict(optimiser=name, metaphor_based=METAPHOR[name], seed=s,
                             eval_count=r.evals, objective=r.best,
                             wallclock_s=r.wallclock_s))
            curves[f"{name}|{s}|e"] = np.asarray(r.curve_evals)
            curves[f"{name}|{s}|b"] = np.asarray(r.curve_best)
        d = pd.DataFrame([x for x in rows if x["optimiser"] == name])
        print(f"  {name:18s} mean {d.objective.mean():.6f} (SD {d.objective.std(ddof=1):.6f}) "
              f"best {d.objective.min():.6f}  {(time.time()-t0):.0f}s total", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "optimiser_runs.csv", index=False)
    np.savez_compressed(RESULTS / "optimiser_curves.npz", **curves)
    jdump({"budget": BUDGET, "subsample_beats": int(len(keep)),
           "subsample_seed": SUB_SEED, "dim": obj.dim,
           "box": [BOX_LO, BOX_HI], "seeds": OPT_SEEDS},
          RESULTS / "optimiser_setup.json")
    print("OPTIMISERS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
