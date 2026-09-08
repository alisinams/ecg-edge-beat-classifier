"""Table 7 statistics: Wilcoxon against gradient descent with Holm correction,
Friedman omnibus, Nemenyi critical difference.

The comparison was fixed before the runs, as section 6.7 states: the paired test
is Wilcoxon signed-rank across the 30 paired seeds, the omnibus is Friedman with
Nemenyi post-hoc, effect size is the rank-biserial correlation, alpha is 0.05
and the pairwise family carries a Holm correction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, jdump, jload
from optimisers import METAPHOR

REFERENCE = "Gradient descent"


def rank_biserial(x, y):
    """Paired rank-biserial correlation for the Wilcoxon signed-rank test."""
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    rp, rm = r[d > 0].sum(), r[d < 0].sum()
    return float((rp - rm) / r.sum())


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference for the Nemenyi test at alpha = 0.05."""
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
           8: 3.031, 9: 3.102, 10: 3.164}
    q = q05.get(k)
    if q is None:
        q = stats.norm.ppf(1 - alpha / (k * (k - 1))) * np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def main() -> int:
    df = pd.read_csv(RESULTS / "optimiser_runs.csv")
    names = list(df.optimiser.unique())
    piv = df.pivot(index="seed", columns="optimiser", values="objective")[names]
    n_seeds = len(piv)

    out = {"n_seeds": int(n_seeds), "budget": int(df.eval_count.max()),
           "reference": REFERENCE, "optimisers": {}}

    raw_p, keys = [], []
    for nm in names:
        if nm == REFERENCE:
            continue
        st = stats.wilcoxon(piv[nm].values, piv[REFERENCE].values,
                            zero_method="wilcox", alternative="two-sided")
        raw_p.append(float(st.pvalue))
        keys.append(nm)
    order = np.argsort(raw_p)
    holm = np.empty(len(raw_p))
    m = len(raw_p)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * raw_p[i])
        prev = max(prev, adj)
        holm[i] = prev

    for nm in names:
        w = df[df.optimiser == nm]
        rec = dict(metaphor_based=bool(METAPHOR[nm]),
                   mean=float(w.objective.mean()), sd=float(w.objective.std(ddof=1)),
                   best=float(w.objective.min()), median=float(w.objective.median()),
                   wallclock_s_mean=float(w.wallclock_s.mean()),
                   wallclock_s_sd=float(w.wallclock_s.std(ddof=1)))
        if nm == REFERENCE:
            rec.update(p_vs_reference=None, p_holm=None, effect_size=None)
        else:
            i = keys.index(nm)
            rec.update(p_vs_reference=raw_p[i], p_holm=float(holm[i]),
                       effect_size=rank_biserial(piv[nm].values, piv[REFERENCE].values))
        out["optimisers"][nm] = rec

    fr = stats.friedmanchisquare(*[piv[nm].values for nm in names])
    ranks = piv.rank(axis=1).mean(axis=0)
    out["friedman"] = dict(statistic=float(fr.statistic), df=len(names) - 1,
                           p=float(fr.pvalue),
                           mean_ranks={k: float(v) for k, v in ranks.items()},
                           nemenyi_cd=nemenyi_cd(len(names), n_seeds))
    sig = []
    rk = ranks.to_dict()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = abs(rk[a] - rk[b])
            sig.append(dict(a=a, b=b, rank_difference=float(d),
                            exceeds_cd=bool(d > out["friedman"]["nemenyi_cd"])))
    out["nemenyi_pairs"] = sig

    setup = jload(RESULTS / "optimiser_setup.json")
    out["setup"] = setup
    jdump(out, RESULTS / "optimiser_stats.json")

    for nm in names:
        r = out["optimisers"][nm]
        print(f"  {nm:18s} {r['mean']:.6f} ({r['sd']:.6f})  best {r['best']:.6f}  "
              f"{r['wallclock_s_mean']:.1f}s  "
              f"p={r['p_holm'] if r['p_holm'] is not None else 'ref'}", flush=True)
    print(f"  Friedman chi2={out['friedman']['statistic']:.1f} "
          f"df={out['friedman']['df']} p={out['friedman']['p']:.3e} "
          f"CD={out['friedman']['nemenyi_cd']:.3f}", flush=True)
    print("OPTIMISER_STATS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
