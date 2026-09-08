"""Stage 10 and 11: the six ablations of Table 6, and the arm B2 rule base.

Each ablation changes one component and holds everything else fixed. Rows one
and two reuse arms already trained; rows three to six train the extra arms the
comparison needs, with the same protocol, the same seeds and the same
hyperparameters as the arm they are being compared against.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from armb import BOX_HI, BOX_LO, StageOneObjective, build_pool
from common import CACHE, CLASSES, RESULTS, SEEDS, jdump, jload
from dataio import class_weights, mitdb_splits
from optimisers import Budget, gradient_descent
from evalutil import boot_cm, confusion, metrics_from_cm
from models import (CLINICAL_RULES, CLINICAL_INPUTS, ClinicalFuzzy, PolySigmoidScorer,
                    clinical_features, ts_fuzzy_decode)
from train import NeuralArm

ARMDIR = CACHE / "arms"


def cms_from_logits(L, y, rec):
    recs = np.unique(rec)
    out = []
    for s in range(L.shape[0]):
        p = L[s].argmax(axis=1)
        cs = np.zeros((len(recs), 5, 5))
        for i, r in enumerate(recs):
            m = rec == r
            cs[i] = confusion(y[m], p[m])
        out.append(cs)
    return out


def diff_ci(cmsA, cmsB, n_boot=2000, seed=0):
    """Paired bootstrap of the macro-F1 difference, resampling records."""
    rng = np.random.default_rng(seed)
    n_rec = cmsA[0].shape[0]
    pa = np.mean([metrics_from_cm(c.sum(axis=0))["macro_f1"] for c in cmsA])
    pb = np.mean([metrics_from_cm(c.sum(axis=0))["macro_f1"] for c in cmsB])
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_rec, size=n_rec)
        a = np.mean([metrics_from_cm(c[pick].sum(axis=0))["macro_f1"] for c in cmsA])
        bb = np.mean([metrics_from_cm(c[pick].sum(axis=0))["macro_f1"] for c in cmsB])
        vals[b] = a - bb
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return dict(a=float(pa), b=float(pb), diff=float(pa - pb),
                lo=float(lo), hi=float(hi),
                spans_zero=bool(lo <= 0.0 <= hi))


def oversample_after_partition(tr, seed):
    """SMOTE applied only inside the training partition, never across it.

    Every minority class is interpolated up to the majority count, which is the
    textbook application and the one the ablation is meant to test. The number
    of neighbours is capped by the smallest class, because the Q superclass
    holds two beats in the DS1 training subset and a five-neighbour search is
    not defined there. The degeneracy that produces is part of the finding.
    """
    from imblearn.over_sampling import SMOTE
    from dataio import Split
    Xf = np.concatenate([tr.X, tr.R], axis=1)
    counts = np.bincount(tr.y, minlength=5)
    k = max(1, int(min(counts[counts > 0].min() - 1, 5)))
    sm = SMOTE(random_state=seed, k_neighbors=k, sampling_strategy="auto")
    Xr, yr = sm.fit_resample(Xf, tr.y)
    n = tr.X.shape[1]
    return Split(Xr[:, :n].astype(np.float32), yr.astype(np.int64),
                 Xr[:, n:].astype(np.float32),
                 np.array(["synthetic"] * len(yr)), np.arange(len(yr)), "oversampled")


def main() -> int:
    tr, va, ds2 = mitdb_splits()
    hp = jload(RESULTS / "hyperparams.json")
    z2 = np.load(CACHE / "preds_ds2.npz", allow_pickle=True)
    y2, rec2 = z2["y"], z2["rec"].astype(str)

    def cms(arm):
        return cms_from_logits(z2[f"{arm}_logits"], y2, rec2)

    res = {}

    # 1. fuzzy decoder against argmax, identical stage one
    res["fuzzy_vs_argmax"] = diff_ci(cms("B"), cms("C"))
    print(f"  fuzzy vs argmax: {res['fuzzy_vs_argmax']}", flush=True)

    # 2. clinical rule base against class-score rule base
    res["b2_vs_b"] = diff_ci(cms("B2"), cms("B"))

    # 3. distilled student against the same topology trained from scratch
    bpE = dict(hp["E"]["best_params"])
    bpE.pop("qat_start_epoch", None)
    bpE["gain_lo"] = 1.0 / bpE.get("gain_hi", 1.33)
    bpE["batch"] = 256
    scratch = []
    for seed in SEEDS:
        out = ARMDIR / f"ablation_scratch_s{seed}.pkl"
        if out.exists():
            m = pickle.loads(out.read_bytes())
        else:
            m = NeuralArm("E", bpE).fit(tr, va, seed=seed, epochs=200, patience=20,
                                        teacher_logits=None)
            out.write_bytes(pickle.dumps(m))
        scratch.append(m.logits(ds2))
    res["distilled_vs_scratch"] = diff_ci(cms("E"), cms_from_logits(np.stack(scratch), y2, rec2))
    print(f"  distilled vs scratch: {res['distilled_vs_scratch']}", flush=True)

    # 4. INT8 against FP32
    if (CACHE / "deploy_cms.npz").exists():
        dz = np.load(CACHE / "deploy_cms.npz")
        f32 = float(np.mean([metrics_from_cm(c)["macro_f1"] for c in dz["fp32"]]))
        i8 = float(np.mean([metrics_from_cm(c)["macro_f1"] for c in dz["int8"]]))
        res["int8_vs_fp32"] = dict(a=i8, b=f32, diff=i8 - f32, lo=None, hi=None,
                                   spans_zero=None,
                                   note="whole-set confusion matrices, no record-level "
                                        "bootstrap, because the quantised model is a "
                                        "deterministic transform of the FP32 one")

    # 5. rhythm descriptors included against morphology only
    morph = []
    for seed in SEEDS:
        out = ARMDIR / f"ablation_morphonly_s{seed}.pkl"
        if out.exists():
            m = pickle.loads(out.read_bytes())
        else:
            m = NeuralArm("E", bpE).fit(tr, va, seed=seed, epochs=200, patience=20,
                                        teacher_logits=None, use_rhythm=False)
            out.write_bytes(pickle.dumps(m))
        morph.append(m.logits(ds2, use_rhythm=False))
    cms_morph = cms_from_logits(np.stack(morph), y2, rec2)
    res["with_rr_vs_without"] = diff_ci(cms("E"), cms_morph)
    # the S class is the point of this row
    se_s_with = float(np.mean([metrics_from_cm(c.sum(axis=0))["se"][1] for c in cms("E")]))
    se_s_without = float(np.mean([metrics_from_cm(c.sum(axis=0))["se"][1] for c in cms_morph]))
    res["with_rr_vs_without"]["se_s_with"] = se_s_with
    res["with_rr_vs_without"]["se_s_without"] = se_s_without
    print(f"  with RR vs without: {res['with_rr_vs_without']['diff']:+.3f}, "
          f"S sensitivity {se_s_with:.3f} against {se_s_without:.3f}", flush=True)

    # 6. class-weighted loss against oversampling applied after partitioning
    over = []
    for seed in SEEDS:
        out = ARMDIR / f"ablation_smote_s{seed}.pkl"
        if out.exists():
            m = pickle.loads(out.read_bytes())
        else:
            hp_o = dict(bpE); hp_o["weight_cap"] = 1.0     # no class weighting
            trn = oversample_after_partition(tr, seed)
            m = NeuralArm("E", hp_o).fit(trn, va, seed=seed, epochs=200, patience=20)
            out.write_bytes(pickle.dumps(m))
        over.append(m.logits(ds2))
    res["weighted_vs_oversampled"] = diff_ci(cms("E"), cms_from_logits(np.stack(over), y2, rec2))
    print(f"  weighted vs oversampled: {res['weighted_vs_oversampled']['diff']:+.3f}", flush=True)

    # 7. arm B's own stage one, fitted by gradient descent instead of WHOA.
    #    This row exists to be fair to the source architecture: it separates
    #    "the architecture is weak" from "the optimiser it was fitted with is
    #    weak", and Table 7 says the two are not the same question.
    sel = np.array(hp["featsel"]["selected"])
    mu = np.array(hp["featsel"]["mu"], dtype=np.float32)
    sdv = np.array(hp["featsel"]["sd"], dtype=np.float32)
    sl = float(hp["B"]["best_params"]["sigma_low"])
    sh = float(hp["B"]["best_params"]["sigma_high"])
    Ztr = (build_pool(tr, mu, sdv)[0])[:, sel]
    Zds2 = (build_pool(ds2, mu, sdv)[0])[:, sel]
    scr = PolySigmoidScorer(n_feat=len(sel), n_class=5)
    gd_logits, gd_obj = [], []
    for seed in SEEDS:
        f = ARMDIR / f"ablation_bgd_s{seed}.npz"
        if f.exists():
            z = np.load(f); theta, val = z["theta"], float(z["objective"])
        else:
            obj = StageOneObjective(Ztr, tr.y, class_weights(tr.y))
            b = Budget(obj.loss_numpy, 15000)
            gradient_descent(b, obj.dim, BOX_LO, BOX_HI, seed, torch_obj=obj)
            theta, val = b.best_x, b.best
            np.savez(f, theta=theta, objective=val)
        gd_obj.append(val)
        zz = scr.scores(Zds2.astype(np.float64), theta.astype(np.float64))
        gd_logits.append(np.log(np.clip(ts_fuzzy_decode(zz, sl, sh), 1e-12, 1.0)))
    res["b_gradient_vs_b_whoa"] = diff_ci(
        cms_from_logits(np.stack(gd_logits), y2, rec2), cms("B"))
    res["b_gradient_vs_b_whoa"]["gd_objective_mean"] = float(np.mean(gd_obj))
    res["b_gradient_vs_b_whoa"]["gd_objective_sd"] = float(np.std(gd_obj, ddof=1))
    print(f"  arm B fitted by gradient descent against WHOA: "
          f"{res['b_gradient_vs_b_whoa']['diff']:+.3f}", flush=True)

    jdump(res, RESULTS / "ablations.json")

    # ------------------------------------------------------ arm B2 rules
    fits = [pickle.loads((ARMDIR / f"ds1_B2_s{s}.pkl").read_bytes())["fit"] for s in SEEDS]
    fit = fits[0]
    fz: ClinicalFuzzy = fit["fuzzy"]
    Cds2 = (clinical_features(ds2.X, ds2.R) - fit["mu"]) / fit["sd"]
    W = fz.firing(Cds2, fit["theta"])
    Wn = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    strengths = Wn.mean(axis=0)
    p = fit["theta"][: fz.n_inputs * 5].reshape(fz.n_inputs, 5)
    mf = {}
    for j, nm in enumerate(CLINICAL_INPUTS):
        m_lo, m_hi, s_lo, s_mid, s_hi = p[j]
        mf[nm] = dict(
            low_mean=float(m_lo * fit["sd"][j] + fit["mu"][j]),
            low_width=float(ClinicalFuzzy._width(s_lo) * fit["sd"][j]),
            mid_mean=float(0.5 * (m_lo + m_hi) * fit["sd"][j] + fit["mu"][j]),
            mid_width=float(ClinicalFuzzy._width(s_mid) * fit["sd"][j]),
            high_mean=float(m_hi * fit["sd"][j] + fit["mu"][j]),
            high_width=float(ClinicalFuzzy._width(s_hi) * fit["sd"][j]),
            unit=("dimensionless" if j < 3 else "ms"))
    rules = []
    for r, (ante, cls) in enumerate(CLINICAL_RULES):
        rules.append(dict(
            rule=f"R{r+1}",
            antecedent=" and ".join(f"{CLINICAL_INPUTS[j]} is {lvl}" for j, lvl in ante.items()),
            consequent=CLASSES[cls],
            firing_strength=float(strengths[r]),
            weight=float(abs(fit["theta"][fz.n_inputs * 5 + r])),
            fires=bool(strengths[r] > 1e-4)))
    jdump(dict(rules=rules, memberships=mf,
               val_macro_f1=[f["val_macro_f1"] for f in fits],
               n_rules=len(rules),
               n_silent=int(sum(1 for r in rules if not r["fires"]))),
          RESULTS / "b2_rules.json")
    print("ABLATIONS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
