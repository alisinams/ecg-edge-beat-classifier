"""Derive every reported number from the raw prediction files.

This is the only file that turns predictions into metrics. If a number appears
in the abstract, in a table, in a figure caption and in the conclusion, it was
computed here once and written to results/numbers.json, and the manuscript
builder reads it from there. That is the mechanism that keeps the four copies of
a number identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARMS, CACHE, CLASSES, DS1_TRAIN, DS1_VAL, DS2, RESULTS, SEEDS, jdump, jload
from dataio import mitdb_meta, mitdb_splits
from evalutil import boot_cm, confusion, metrics_from_cm

ARM_LABEL = {
    "A": "A, best classical",
    "B": "B, scorer with fuzzy decoder",
    "B2": "B2, clinical rule base",
    "C": "C, scorer with argmax decoder",
    "D": "D, separable CNN",
    "E": "E, student classifier",
    "F": "F, temporal-convolution student",
}


def load_preds(name: str):
    z = np.load(CACHE / f"preds_{name}.npz", allow_pickle=True)
    return z


def cms_for(z, arm: str, mask=None):
    """Per-record confusion matrices, one stack per training seed."""
    y, rec = z["y"], z["rec"].astype(str)
    L = z[f"{arm}_logits"]
    if mask is not None:
        y, rec, L = y[mask], rec[mask], L[:, mask]
    recs = np.unique(rec)
    out = []
    for s in range(L.shape[0]):
        p = L[s].argmax(axis=1)
        cs = np.zeros((len(recs), 5, 5))
        for i, r in enumerate(recs):
            m = rec == r
            cs[i] = confusion(y[m], p[m])
        out.append(cs)
    return out, recs


def ci(cms, stat, seed=0):
    return boot_cm(cms, stat, n_boot=2000, seed=seed)


def fmt_pct(t):
    p, lo, hi = t
    return f"{100*p:.1f} ({100*lo:.1f} to {100*hi:.1f})"


def fmt_f1(t):
    p, lo, hi = t
    return f"{p:.3f} ({lo:.3f} to {hi:.3f})"


def main() -> int:
    N: dict = {}

    # ---------------------------------------------------------- Table 2
    pc = pd.read_csv(RESULTS / "partition_counts.csv")
    piv = pc.pivot_table(index="partition", columns="class", values="count",
                         aggfunc="sum").reindex(columns=CLASSES).fillna(0).astype(int)
    piv["Total"] = piv.sum(axis=1)
    piv["Records"] = pc.groupby("partition")["record"].nunique()
    N["table2"] = {k: {c: int(v) for c, v in row.items()}
                   for k, row in piv.iterrows()}
    N["splits"] = dict(ds1_train=DS1_TRAIN, ds1_val=DS1_VAL, ds2=DS2)

    # ---------------------------------------------------------- Table 3
    z2 = load_preds("ds2")
    t3, agg = {}, {}
    for arm in ARMS:
        cms, recs = cms_for(z2, arm)
        row = {}
        for k, cname in enumerate(CLASSES):
            row[cname] = dict(
                se=ci(cms, lambda m, k=k: m["se"][k]),
                ppv=ci(cms, lambda m, k=k: m["ppv"][k]),
                sp=ci(cms, lambda m, k=k: m["sp"][k]),
                f1=ci(cms, lambda m, k=k: m["f1"][k]),
                fpr=ci(cms, lambda m, k=k: m["fpr"][k]),
                support=int(sum(c.sum(axis=0)[k, :].sum() for c in [cms[0]])),
            )
        row["macro"] = dict(
            se=ci(cms, lambda m: m["macro_se"]),
            ppv=ci(cms, lambda m: m["macro_ppv"]),
            sp=ci(cms, lambda m: m["macro_sp"]),
            f1=ci(cms, lambda m: m["macro_f1"]),
            fpr=ci(cms, lambda m: m["macro_fpr"]),
        )
        row["aggregate"] = dict(
            accuracy=ci(cms, lambda m: m["accuracy"]),
            kappa=ci(cms, lambda m: m["kappa"]),
            mcc=ci(cms, lambda m: m["mcc"]),
            macro_f1=row["macro"]["f1"],
        )
        t3[arm] = row
        agg[arm] = row["aggregate"]
        print(f"  DS2 {arm}: macro-F1 {fmt_f1(row['macro']['f1'])}", flush=True)
    N["table3"] = t3
    N["ds2_aggregate"] = agg

    # per-record table S4.1
    y2, rec2 = z2["y"], z2["rec"].astype(str)
    LE = z2["E_logits"]
    s41 = {}
    for r in sorted(set(rec2)):
        m = rec2 == r
        cm = np.mean([confusion(y2[m], LE[s][m].argmax(1)) for s in range(LE.shape[0])], axis=0)
        mm = metrics_from_cm(cm)
        s41[r] = dict(beats=int(m.sum()),
                      counts={c: int((y2[m] == i).sum()) for i, c in enumerate(CLASSES)},
                      macro_f1=float(mm["macro_f1"]),
                      se_v=float(mm["se"][2]), se_s=float(mm["se"][1]))
    N["table_s41"] = s41

    # ------------------------------------------- the S class, by record
    # One DS2 record holds most of the S beats, and every arm fails on it, so the
    # aggregate S sensitivity is a statement about one subject. Section 7.2 says
    # so and this block supplies the numbers.
    s_by_rec = {r: int(((y2 == 1) & (rec2 == r)).sum()) for r in sorted(set(rec2))}
    dom = max(s_by_rec, key=s_by_rec.get)
    n_s = int((y2 == 1).sum())
    dom_detail = {}
    for arm in ARMS:
        L = z2[f"{arm}_logits"]
        p = np.stack([L[s].argmax(1) for s in range(L.shape[0])])
        se_all, se_ex, se_only = [], [], []
        for s in range(p.shape[0]):
            keep = rec2 != dom
            se_all.append(metrics_from_cm(confusion(y2, p[s]))["se"][1])
            se_ex.append(metrics_from_cm(confusion(y2[keep], p[s][keep]))["se"][1])
            se_only.append(metrics_from_cm(confusion(y2[~keep], p[s][~keep]))["se"][1])
        dom_detail[arm] = dict(se_s_all=float(np.nanmean(se_all)),
                               se_s_excluding=float(np.nanmean(se_ex)),
                               se_s_only=float(np.nanmean(se_only)))
    N["s_class_by_record"] = dict(
        dominant_record=dom, dominant_s_beats=s_by_rec[dom], total_s_beats=n_s,
        dominant_share=float(s_by_rec[dom] / n_s), per_arm=dom_detail,
        n_records_with_s=int(sum(1 for v in s_by_rec.values() if v > 0)))
    print(f"  record {dom} holds {s_by_rec[dom]} of {n_s} DS2 S beats "
          f"({100*s_by_rec[dom]/n_s:.0f} per cent)", flush=True)

    # ---------------------------------------------------------- Table 4
    zm = load_preds("mixed")
    t4 = {}
    for arm in ARMS:
        cmm, _ = cms_for(zm, arm)
        cmi, _ = cms_for(z2, arm)
        t4[arm] = dict(mixed=ci(cmm, lambda m: m["macro_f1"]),
                       inter=ci(cmi, lambda m: m["macro_f1"]))
    order_m = sorted(ARMS, key=lambda a: -t4[a]["mixed"][0])
    order_i = sorted(ARMS, key=lambda a: -t4[a]["inter"][0])
    for a in ARMS:
        t4[a]["rank_mixed"] = order_m.index(a) + 1
        t4[a]["rank_inter"] = order_i.index(a) + 1
        t4[a]["rank_change"] = t4[a]["rank_inter"] - t4[a]["rank_mixed"]
        t4[a]["gain"] = t4[a]["mixed"][0] - t4[a]["inter"][0]
    N["table4"] = t4
    N["protocol"] = dict(
        mean_gain=float(np.mean([t4[a]["gain"] for a in ARMS])),
        n_rank_changes=int(sum(1 for a in ARMS if t4[a]["rank_change"] != 0)),
        n_arms_improved=int(sum(1 for a in ARMS if t4[a]["gain"] > 0)),
        max_gain=float(max(t4[a]["gain"] for a in ARMS)),
        min_gain=float(min(t4[a]["gain"] for a in ARMS)),
        max_gain_arm=max(ARMS, key=lambda a: t4[a]["gain"]),
        min_gain_arm=min(ARMS, key=lambda a: t4[a]["gain"]),
    )

    # ---------------------------------------------------------- Table 5
    t5 = {}
    cms2, _ = cms_for(z2, "E")
    t5["DS2"] = dict(accuracy=ci(cms2, lambda m: m["accuracy"]),
                     macro_f1=ci(cms2, lambda m: m["macro_f1"]),
                     se_s=ci(cms2, lambda m: m["se"][1]),
                     se_v=ci(cms2, lambda m: m["se"][2]))
    for db, label in (("incartdb", "INCART"), ("svdb", "SVDB")):
        ze = load_preds(db)
        cmse, _ = cms_for(ze, "E")
        t5[label] = dict(accuracy=ci(cmse, lambda m: m["accuracy"]),
                         macro_f1=ci(cmse, lambda m: m["macro_f1"]),
                         se_s=ci(cmse, lambda m: m["se"][1]),
                         se_v=ci(cmse, lambda m: m["se"][2]))
    N["table5"] = t5
    ext = [t5["INCART"]["macro_f1"][0], t5["SVDB"]["macro_f1"][0]]
    N["external"] = dict(
        drop_incart=t5["DS2"]["macro_f1"][0] - t5["INCART"]["macro_f1"][0],
        drop_svdb=t5["DS2"]["macro_f1"][0] - t5["SVDB"]["macro_f1"][0],
        drop_mean=t5["DS2"]["macro_f1"][0] - float(np.mean(ext)),
        retained_fraction=float(np.mean(ext)) / t5["DS2"]["macro_f1"][0])

    # ---------------------------------------------------------- Table 6
    abl = jload(RESULTS / "ablations.json") if (RESULTS / "ablations.json").exists() else {}
    N["table6"] = abl

    # ---------------------------------------------------------- Table 7
    if (RESULTS / "optimiser_stats.json").exists():
        N["table7"] = jload(RESULTS / "optimiser_stats.json")

    # ---------------------------------------------------------- Table 8
    if (RESULTS / "deployment.json").exists():
        N["table8"] = jload(RESULTS / "deployment.json")

    # Arm A's own size, measured the same way, so the accuracy comparison in
    # section 8.1 can be read against the memory budget instead of asserted.
    try:
        import pickle
        fit = pickle.loads((CACHE / "arms" / "ds1_A_s0.pkl").read_bytes())["fit"]
        m = fit["model"]
        if hasattr(m, "get_booster"):
            raw = m.get_booster().save_raw("ubj")
            size_kb = len(raw) / 1024.0
            how = "XGBoost raw model buffer"
        else:
            size_kb = len(pickle.dumps(m)) / 1024.0
            how = "pickled estimator"
        N["arm_a_size"] = dict(name=fit["name"], size_kb=float(size_kb), measured_as=how)
        print(f"  arm A ({fit['name']}) serialises to {size_kb:.0f} kB via {how}", flush=True)
    except Exception as exc:  # noqa: BLE001
        N["arm_a_size"] = dict(error=str(exc))

    # ---------------------------------------------------------- Table 9
    meta = mitdb_meta()
    strata = {
        "Female": [r for r in DS2 if meta.get(r, ("U", np.nan))[0] == "F"],
        "Male": [r for r in DS2 if meta.get(r, ("U", np.nan))[0] == "M"],
        "Age below 60": [r for r in DS2 if np.isfinite(meta.get(r, ("U", np.nan))[1])
                         and meta[r][1] < 60],
        "Age 60 and above": [r for r in DS2 if np.isfinite(meta.get(r, ("U", np.nan))[1])
                             and meta[r][1] >= 60],
        "Age not recorded": [r for r in DS2 if not np.isfinite(meta.get(r, ("U", np.nan))[1])],
    }
    t9 = {}
    for label, recs in strata.items():
        if not recs:
            continue
        m = np.isin(rec2, recs)
        cms, _ = cms_for(z2, "E", mask=m)
        t9[label] = dict(
            records=len(recs),
            v_beats=int((y2[m] == 2).sum()), s_beats=int((y2[m] == 1).sum()),
            se_v=ci(cms, lambda mm: mm["se"][2]), ppv_v=ci(cms, lambda mm: mm["ppv"][2]),
            se_s=ci(cms, lambda mm: mm["se"][1]), ppv_s=ci(cms, lambda mm: mm["ppv"][1]))
    N["table9"] = t9
    N["subgroup_meta"] = {r: dict(sex=meta.get(r, ("U", np.nan))[0],
                                  age=(None if not np.isfinite(meta.get(r, ("U", np.nan))[1])
                                       else float(meta[r][1]))) for r in DS2}

    # -------------------------------------------------- calibration etc
    for f in ("calibration.json", "abstention.json", "noise_sweep.json",
              "sqi.json", "alarms.json", "b2_rules.json", "train_performance_ds1.json",
              "hyperparams.json", "teacher.json", "bandwidth_ablation.json"):
        p = RESULTS / f
        if p.exists():
            N[f.replace(".json", "")] = jload(p)

    if (RESULTS / "table1_verified.json").exists():
        N["table1_verified"] = jload(RESULTS / "table1_verified.json")

    jdump(N, RESULTS / "numbers.json")
    print(f"NUMBERS_COMPLETE  {len(N)} top-level blocks", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
