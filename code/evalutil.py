"""Metrics, record-level bootstrap, calibration, conformal abstention.

Every number in the manuscript that is a metric comes from this module, applied
to a raw prediction file. Nothing computes a metric anywhere else.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

CLASSES = ["N", "S", "V", "F", "Q"]


def per_class(y: np.ndarray, p: np.ndarray, n_class: int = 5) -> dict:
    out = {}
    for c in range(n_class):
        tp = int(((y == c) & (p == c)).sum())
        fn = int(((y == c) & (p != c)).sum())
        fp = int(((y != c) & (p == c)).sum())
        tn = int(((y != c) & (p != c)).sum())
        se = tp / (tp + fn) if (tp + fn) else np.nan
        ppv = tp / (tp + fp) if (tp + fp) else np.nan
        sp = tn / (tn + fp) if (tn + fp) else np.nan
        fpr = fp / (fp + tn) if (fp + tn) else np.nan
        f1 = (2 * se * ppv / (se + ppv)) if (se and ppv and np.isfinite(se) and
                                            np.isfinite(ppv) and (se + ppv) > 0) else (
            0.0 if (tp + fn + fp) else np.nan)
        out[CLASSES[c]] = dict(tp=tp, fn=fn, fp=fp, tn=tn, se=se, ppv=ppv,
                               sp=sp, fpr=fpr, f1=f1, support=tp + fn)
    return out


def macro_f1(y: np.ndarray, p: np.ndarray, n_class: int = 5,
             present_only: bool = True, classes: list[int] | None = None) -> float:
    pc = per_class(y, p, n_class)
    names = CLASSES[:n_class] if classes is None else [CLASSES[i] for i in classes]
    vals = []
    for c in names:
        d = pc[c]
        if present_only and d["support"] == 0:
            continue
        vals.append(0.0 if not np.isfinite(d["f1"]) else d["f1"])
    return float(np.mean(vals)) if vals else np.nan


# Classes used for model selection on the DS1 validation subset. F and Q are
# excluded there and only there: that subset holds 17 F beats and 4 Q beats, so
# their F1 is sampling noise and would decide model selection by coin flip. Both
# classes are reported in full on DS2, where they are not the selection signal.
SELECTION_CLASSES = [0, 1, 2]


def selection_f1(y: np.ndarray, p: np.ndarray, n_class: int = 5) -> float:
    return macro_f1(y, p, n_class, classes=SELECTION_CLASSES)


def aggregate(y: np.ndarray, p: np.ndarray, n_class: int = 5) -> dict:
    return dict(
        accuracy=float((y == p).mean()),
        macro_f1=macro_f1(y, p, n_class),
        kappa=float(cohen_kappa_score(y, p, labels=list(range(n_class)))),
        mcc=float(matthews_corrcoef(y, p)),
    )


def macro_avg(y: np.ndarray, p: np.ndarray, key: str, n_class: int = 5) -> float:
    pc = per_class(y, p, n_class)
    vals = [pc[c][key] for c in CLASSES[:n_class] if pc[c]["support"] > 0]
    vals = [0.0 if not np.isfinite(v) else v for v in vals]
    return float(np.mean(vals)) if vals else np.nan


def bootstrap_records(y: np.ndarray, p: np.ndarray, rec: np.ndarray, fn,
                      n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """95 per cent percentile interval, resampling whole records.

    fn takes (y, p) and returns a scalar. Records are the resampling unit
    because beats inside a record are not independent.
    """
    rng = np.random.default_rng(seed)
    recs = np.unique(rec)
    idx_by_rec = {r: np.flatnonzero(rec == r) for r in recs}
    point = fn(y, p)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(recs, size=len(recs), replace=True)
        idx = np.concatenate([idx_by_rec[r] for r in pick])
        vals[b] = fn(y[idx], p[idx])
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# ------------------------------------------------------------- calibration
def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """One scalar temperature by minimising negative log-likelihood."""
    import torch
    lg = torch.tensor(logits, dtype=torch.float64)
    yy = torch.tensor(y, dtype=torch.long)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / torch.exp(logT), yy)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(logT).item())


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> tuple[float, list]:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    e, rows = 0.0, []
    n = len(y)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            rows.append(dict(lo=lo, hi=hi, n=0, conf=np.nan, acc=np.nan))
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        e += (m.sum() / n) * abs(a - c)
        rows.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()), conf=c, acc=a))
    return float(e), rows


def brier(probs: np.ndarray, y: np.ndarray, n_class: int = 5) -> float:
    oh = np.zeros_like(probs)
    oh[np.arange(len(y)), y] = 1.0
    return float(((probs - oh) ** 2).sum(axis=1).mean())


# --------------------------------------------------------------- conformal
def conformal_threshold(probs_cal: np.ndarray, y_cal: np.ndarray, eps: float) -> float:
    """Split-conformal quantile of the nonconformity score 1 - p_true."""
    s = 1.0 - probs_cal[np.arange(len(y_cal)), y_cal]
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - eps)))
    k = min(max(k, 1), n)
    return float(np.sort(s)[k - 1])


def conformal_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean membership matrix: class k is in the set if p_k >= 1 - qhat."""
    return probs >= (1.0 - qhat)


# ------------------------------------------------- confusion-matrix layer
# Every metric in the manuscript is a function of the confusion matrix, so the
# record-level bootstrap sums per-record confusion matrices instead of
# recomputing metrics over resampled beats. Two thousand resamples across five
# seeds then costs milliseconds and is exact, not approximate.

def confusion(y: np.ndarray, p: np.ndarray, n_class: int = 5) -> np.ndarray:
    return np.bincount(y * n_class + p, minlength=n_class * n_class
                       ).reshape(n_class, n_class).astype(np.float64)


def per_record_cms(y, p, rec, n_class: int = 5):
    recs = np.unique(rec)
    out = np.zeros((len(recs), n_class, n_class))
    for i, r in enumerate(recs):
        m = rec == r
        out[i] = confusion(y[m], p[m], n_class)
    return recs, out


def metrics_from_cm(cm: np.ndarray) -> dict:
    n = cm.sum()
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = n - tp - fn - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        ppv = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        sp = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
        fpr = np.where(fp + tn > 0, fp / (fp + tn), np.nan)
        f1 = np.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    support = tp + fn
    present = support > 0
    acc = tp.sum() / n if n else np.nan
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n * n) if n else np.nan
    kappa = (acc - pe) / (1 - pe) if (n and pe < 1) else np.nan
    c, s = tp.sum(), n
    pk, tk = cm.sum(axis=0), cm.sum(axis=1)
    num = c * s - (pk * tk).sum()
    den = np.sqrt(max(s * s - (pk * pk).sum(), 0)) * np.sqrt(max(s * s - (tk * tk).sum(), 0))
    mcc = num / den if den > 0 else np.nan
    sel = np.array(SELECTION_CLASSES)
    return dict(se=se, ppv=ppv, sp=sp, fpr=fpr, f1=f1, support=support,
                accuracy=acc, kappa=kappa, mcc=mcc,
                macro_f1=float(np.nanmean(np.where(present, np.nan_to_num(f1), np.nan))),
                macro_se=float(np.nanmean(np.where(present, np.nan_to_num(se), np.nan))),
                macro_ppv=float(np.nanmean(np.where(present, np.nan_to_num(ppv), np.nan))),
                macro_sp=float(np.nanmean(np.where(present, np.nan_to_num(sp), np.nan))),
                macro_fpr=float(np.nanmean(np.where(present, np.nan_to_num(fpr), np.nan))),
                selection_f1=float(np.nanmean(
                    np.where(present[sel], np.nan_to_num(f1[sel]), np.nan))))


def boot_cm(cms_by_seed: list[np.ndarray], stat, n_boot: int = 2000, seed: int = 0):
    """Record-level bootstrap over per-record confusion matrices.

    cms_by_seed is a list, one entry per training seed, each of shape
    (n_records, n_class, n_class). Each replicate resamples records with
    replacement, sums their matrices, evaluates `stat` for every seed and
    averages, so the interval carries record sampling and seed variation at
    once. The point estimate is the same average on the unresampled data.
    """
    rng = np.random.default_rng(seed)
    n_rec = cms_by_seed[0].shape[0]
    point = float(np.mean([stat(metrics_from_cm(c.sum(axis=0))) for c in cms_by_seed]))
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_rec, size=n_rec)
        vals[b] = np.mean([stat(metrics_from_cm(c[pick].sum(axis=0))) for c in cms_by_seed])
    vals = vals[np.isfinite(vals)]
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan))
    return point, float(lo), float(hi)
