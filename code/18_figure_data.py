"""Export one tidy data file per figure.

The figure builder never touches a model or a metric function; it reads these
files. That is what guarantees that a number drawn in a figure is the number
printed in its table, since both come from results/numbers.json or from a file
written here out of the same prediction arrays.
"""
from __future__ import annotations

import sys
from pathlib import Path

import re

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, CLASSES, FIGURES, RESULTS, jdump, jload
from evalutil import confusion, metrics_from_cm, softmax

OUT = RESULTS / "figure_data"
OUT.mkdir(parents=True, exist_ok=True)


def mean_probs(L):
    return softmax(L.reshape(-1, 5)).reshape(L.shape).mean(axis=0)


def main() -> int:
    N = jload(RESULTS / "numbers.json")

    # ---- Figure 2: confusion matrices, row-normalised, counts kept
    rows = []
    for name, label in (("ds2", "DS2"), ("incartdb", "INCART"), ("svdb", "SVDB")):
        z = np.load(CACHE / f"preds_{name}.npz", allow_pickle=True)
        p = mean_probs(z["E_logits"]).argmax(1)
        cm = confusion(z["y"], p)
        for i, tc in enumerate(CLASSES):
            tot = cm[i].sum()
            for j, pc in enumerate(CLASSES):
                rows.append(dict(dataset=label, true_class=tc, pred_class=pc,
                                 count=int(cm[i, j]),
                                 row_fraction=float(cm[i, j] / tot) if tot else np.nan))
    pd.DataFrame(rows).to_csv(OUT / "fig2_confusion.csv", index=False)

    # ---- Figure 3 and S4.2: precision-recall and ROC on DS2
    z = np.load(CACHE / "preds_ds2.npz", allow_pickle=True)
    P = mean_probs(z["E_logits"])
    y = z["y"]
    from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
    pr, roc = [], []
    for k, c in enumerate(CLASSES):
        yk = (y == k).astype(int)
        if yk.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(yk, P[:, k])
        step = max(1, len(prec) // 2000)
        for a, b in zip(prec[::step], rec[::step]):
            pr.append(dict(cls=c, precision=float(a), recall=float(b),
                           baseline=float(yk.mean()),
                           average_precision=float(average_precision_score(yk, P[:, k]))))
        fpr, tpr, _ = roc_curve(yk, P[:, k])
        step = max(1, len(fpr) // 2000)
        for a, b in zip(fpr[::step], tpr[::step]):
            roc.append(dict(cls=c, fpr=float(a), tpr=float(b),
                            auc=float(roc_auc_score(yk, P[:, k]))))
    pd.DataFrame(pr).to_csv(OUT / "fig3_precision_recall.csv", index=False)
    pd.DataFrame(roc).to_csv(OUT / "figS42_roc.csv", index=False)

    # ---- Figure 4: reliability bins
    cal = jload(RESULTS / "calibration.json")
    rows = []
    for tag in ("before", "after"):
        for b in cal[f"bins_{tag}"]:
            rows.append(dict(stage=tag, bin_lo=b["lo"], bin_hi=b["hi"], n=b["n"],
                             confidence=b["conf"], accuracy=b["acc"]))
    pd.DataFrame(rows).to_csv(OUT / "fig4_reliability.csv", index=False)

    # ---- Figure 5: accuracy against coverage
    ab = jload(RESULTS / "abstention.json")
    rows = []
    for ds in ("DS2", "INCART", "SVDB"):
        for r in ab.get(ds, {}).get("curve", []):
            rows.append(dict(dataset=ds, **r))
    pd.DataFrame(rows).to_csv(OUT / "fig5_coverage.csv", index=False)

    # ---- Figure 6: noise sweep
    ns = jload(RESULTS / "noise_sweep.json")
    df = pd.DataFrame(ns["rows"])
    df["clean_macro_f1"] = ns["clean_macro_f1"]
    df.to_csv(OUT / "fig6_noise.csv", index=False)

    # ---- Figure 7: accuracy against size, anchors plus this work
    T1 = jload(RESULTS / "table1_verified.json")
    t8 = jload(RESULTS / "deployment.json")
    rows = []
    for k, v in T1.items():
        if k.startswith("_"):
            continue
        acc = v.get("accuracy_pct")
        if acc is None:
            acc = v.get("accuracy_headline_pct")
        m = re.search(r"(\d{4})$", k)
        rows.append(dict(
            study=re.sub(r"(\d{4})$", "", k), year=int(m.group(1)) if m else None,
            model_size_kb=v.get("model_size_kb_for_fig7"),
            size_reported_as=v.get("model_size"),
            accuracy_pct=acc,
            classes=v.get("classes_short", v.get("classes")),
            aami=v.get("aami"),
            inter_patient=str(v.get("inter_patient", "")).lower().startswith("yes"),
            energy_uj=v.get("energy_uj_for_fig7"),
            latency_ms=v.get("latency_ms_for_fig7"),
            target=v.get("target_hardware"),
            note=("size reported only in non-byte units"
                  if v.get("model_size_kb_for_fig7") is None else "")))
    e = N["table3"]["E"]
    rows.append(dict(
        study="This work", year=2026, model_size_kb=t8["size_int8_kb"],
        size_reported_as=f"{t8['size_int8_kb']:.1f} kB INT8",
        accuracy_pct=100 * N["ds2_aggregate"]["E"]["accuracy"][0],
        classes="5", aami="Yes", inter_patient=True,
        energy_uj=t8["energy_uj"], latency_ms=t8["latency_us"]["median"] / 1000.0,
        target=t8["device"]["part"], note="energy and latency computed, not metered"))
    pd.DataFrame(rows).to_csv(OUT / "fig7_frontier.csv", index=False)

    # ---- Figure S4.1: optimiser convergence
    cz = np.load(RESULTS / "optimiser_curves.npz")
    keys = sorted({k.rsplit("|", 2)[0] for k in cz.files})
    rows = []
    for nm in keys:
        seeds = sorted({int(k.split("|")[1]) for k in cz.files if k.startswith(nm + "|")})
        grid = None
        vals = []
        for s in seeds:
            e, b = cz[f"{nm}|{s}|e"], cz[f"{nm}|{s}|b"]
            if grid is None:
                grid = e
            vals.append(np.interp(grid, e, b))
        Vv = np.stack(vals)
        for i, g in enumerate(grid):
            rows.append(dict(optimiser=nm, evaluations=int(g),
                             mean=float(Vv[:, i].mean()),
                             q25=float(np.percentile(Vv[:, i], 25)),
                             q75=float(np.percentile(Vv[:, i], 75))))
    pd.DataFrame(rows).to_csv(OUT / "figS41_convergence.csv", index=False)

    # ---- Figure S4.4: protocol slopegraph
    rows = []
    for a, d in N["table4"].items():
        rows.append(dict(arm=a, mixed=d["mixed"][0], inter=d["inter"][0],
                         rank_mixed=d["rank_mixed"], rank_inter=d["rank_inter"],
                         rank_change=d["rank_change"]))
    pd.DataFrame(rows).to_csv(OUT / "figS44_slopegraph.csv", index=False)

    print("FIGURE_DATA_COMPLETE", *(p.name for p in sorted(OUT.glob("*.csv"))), sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
