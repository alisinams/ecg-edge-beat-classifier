"""Stage 8: calibration, conformal abstention, noise sweep, signal-quality gate,
alarm burden.

The calibration temperature and the conformal threshold are fitted on the DS1
validation subset and applied unchanged to DS2 and to both external databases,
so nothing downstream of the classifier was tuned on the data it is scored on.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BP_HIGH, BP_LOW, CACHE, CLASSES, DATA, FS, POST_N, PRE_N,
                    RESULTS, SEEDS, WIN, jdump)
from dataio import external, mitdb_splits
from evalutil import (brier, conformal_threshold, confusion, ece, fit_temperature,
                      metrics_from_cm, softmax)

ARMDIR = CACHE / "arms"
EPS = 0.10                    # conformal target coverage 90 per cent
COVERAGES = np.round(np.arange(0.50, 1.001, 0.02), 3)


_STUDENTS = None


def students():
    """The five trained student arms, loaded once. The noise sweep calls this
    sixty times and unpickling a model per call would dominate its runtime."""
    global _STUDENTS
    if _STUDENTS is None:
        _STUDENTS = [pickle.loads((ARMDIR / f"ds1_E_s{s}.pkl").read_bytes())["model"]
                     for s in SEEDS]
    return _STUDENTS


def student_logits(split):
    return np.stack([m.logits(split) for m in students()])


def mean_probs(L, T=1.0):
    return softmax((L / T).reshape(-1, 5)).reshape(L.shape).mean(axis=0)


# ------------------------------------------------------------------ SQI
def sqi(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Three-part signal quality index in [0, 1].

    flatline fraction, in-band to out-of-band power ratio, and agreement
    between the annotated R peak and a simple slope-based redetection inside
    the window. The three parts are multiplied, so any one of them can reject.
    """
    d = np.abs(np.diff(X, axis=1))
    flat = 1.0 - (d < (1e-3 * (np.abs(X).max(axis=1, keepdims=True) + 1e-9))).mean(axis=1)

    f, P = sps.welch(X, fs=FS, nperseg=min(128, X.shape[1]), axis=1)
    inb = P[:, (f >= 1.0) & (f <= 30.0)].sum(axis=1)
    out = P[:, (f > 30.0)].sum(axis=1) + 1e-12
    ratio = inb / (inb + out)

    env = np.abs(sps.savgol_filter(X, 9, 2, deriv=1, axis=1))
    peak = env.argmax(axis=1)
    agree = np.exp(-0.5 * ((peak - PRE_N) / 12.0) ** 2)
    return np.clip(flat * ratio * agree, 0.0, 1.0)


# ---------------------------------------------------------------- noise
def load_noise():
    import wfdb
    out = {}
    for rec, label in (("bw", "baseline wander"), ("ma", "muscle artifact"),
                       ("em", "electrode motion")):
        r = wfdb.rdrecord(str(DATA / "nstdb" / rec))
        x = np.nan_to_num(np.asarray(r.p_signal[:, 0], dtype=np.float64))
        from math import gcd
        g = gcd(FS, int(r.fs))
        x = sps.resample_poly(x, FS // g, int(r.fs) // g)
        out[label] = (x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))) + 1e-9)
    return out


def add_noise(X, noise, snr_db, rng):
    sig_p = (X ** 2).mean(axis=1, keepdims=True)
    n = len(noise)
    starts = rng.integers(0, max(1, n - X.shape[1]), size=len(X))
    seg = np.stack([noise[s:s + X.shape[1]] for s in starts])
    if seg.shape[1] < X.shape[1]:
        seg = np.pad(seg, ((0, 0), (0, X.shape[1] - seg.shape[1])))
    noise_p = (seg ** 2).mean(axis=1, keepdims=True) + 1e-12
    scale = np.sqrt(sig_p / (noise_p * (10 ** (snr_db / 10.0))))
    return (X + scale * seg).astype(np.float32)


def add_mains(X, hz, snr_db, rng):
    t = np.arange(X.shape[1]) / FS
    ph = rng.uniform(0, 2 * np.pi, size=(len(X), 1))
    seg = np.sin(2 * np.pi * hz * t[None, :] + ph)
    sig_p = (X ** 2).mean(axis=1, keepdims=True)
    scale = np.sqrt(sig_p / (0.5 * (10 ** (snr_db / 10.0))))
    return (X + scale * seg).astype(np.float32)


def main() -> int:
    tr, va, ds2 = mitdb_splits()
    Lva, Lds2 = student_logits(va), student_logits(ds2)

    # -------------------------------------------------------- calibration
    lv = Lva.mean(axis=0)
    T = fit_temperature(lv, va.y)
    p_before = mean_probs(Lds2, 1.0)
    p_after = mean_probs(Lds2, T)
    e0, rows0 = ece(p_before, ds2.y)
    e1, rows1 = ece(p_after, ds2.y)
    conf0 = float(p_before.max(axis=1).mean())
    acc0 = float((p_before.argmax(1) == ds2.y).mean())
    cal = dict(temperature=float(T),
               direction=("over-confident" if conf0 > acc0 else "under-confident"),
               mean_confidence_before=conf0, accuracy=acc0,
               ece_before=e0, ece_after=e1,
               brier_before=brier(p_before, ds2.y), brier_after=brier(p_after, ds2.y),
               bins_before=rows0, bins_after=rows1,
               argmax_unchanged=bool((p_before.argmax(1) == p_after.argmax(1)).all()))
    jdump(cal, RESULTS / "calibration.json")
    print(f"  temperature {T:.3f}  ECE {e0:.4f} -> {e1:.4f}  "
          f"Brier {cal['brier_before']:.4f} -> {cal['brier_after']:.4f}", flush=True)

    # --------------------------------------------------------- conformal
    p_cal = mean_probs(Lva, T)
    qhat = conformal_threshold(p_cal, va.y, EPS)

    def curve(probs, y, rec):
        rows = []
        for cov in COVERAGES:
            q = conformal_threshold(p_cal, va.y, 1.0 - cov)
            keep = (probs >= (1.0 - q)).sum(axis=1) <= 1
            if keep.sum() < 10:
                continue
            cm = confusion(y[keep], probs[keep].argmax(1))
            rows.append(dict(target_coverage=float(cov),
                             observed_coverage=float(keep.mean()),
                             macro_f1=float(metrics_from_cm(cm)["macro_f1"]),
                             accuracy=float(metrics_from_cm(cm)["accuracy"]),
                             n=int(keep.sum())))
        return rows

    absten = {"threshold": float(qhat), "target": 1 - EPS}
    sets_ds2 = mean_probs(Lds2, T) >= (1.0 - qhat)
    keep_ds2 = sets_ds2.sum(axis=1) <= 1
    cm_ds2 = confusion(ds2.y[keep_ds2], p_after[keep_ds2].argmax(1))
    absten["DS2"] = dict(coverage=float(keep_ds2.mean()),
                         abstention_rate=float(1 - keep_ds2.mean()),
                         macro_f1_retained=float(metrics_from_cm(cm_ds2)["macro_f1"]),
                         macro_f1_all=float(metrics_from_cm(
                             confusion(ds2.y, p_after.argmax(1)))["macro_f1"]),
                         curve=curve(p_after, ds2.y, ds2.rec))

    ext_probs = {}
    for db, label in (("incartdb", "INCART"), ("svdb", "SVDB")):
        sp = external(db)
        Le = student_logits(sp)
        pe = mean_probs(Le, T)
        ext_probs[label] = (pe, sp)
        ke = (pe >= (1.0 - qhat)).sum(axis=1) <= 1
        cme = confusion(sp.y[ke], pe[ke].argmax(1))
        absten[label] = dict(coverage=float(ke.mean()),
                             abstention_rate=float(1 - ke.mean()),
                             macro_f1_retained=float(metrics_from_cm(cme)["macro_f1"]),
                             macro_f1_all=float(metrics_from_cm(
                                 confusion(sp.y, pe.argmax(1)))["macro_f1"]),
                             curve=curve(pe, sp.y, sp.rec))
        print(f"  {label}: observed coverage {ke.mean()*100:.1f}% against a "
              f"{100*(1-EPS):.0f}% target", flush=True)
    jdump(absten, RESULTS / "abstention.json")

    # --------------------------------------------------------------- SQI
    q_tr = sqi(tr.X, tr.R)
    thr = float(np.percentile(q_tr, 2.0))       # fixed on DS1 training beats only
    rows = []
    for name, sp in (("DS2", ds2), ("INCART", external("incartdb")),
                     ("SVDB", external("svdb"))):
        q = sqi(sp.X, sp.R)
        rej = int((q < thr).sum())
        rows.append(dict(dataset=name, rejected=rej, total=int(len(q)),
                         rate=float(rej / len(q))))
        print(f"  SQI {name}: {rej}/{len(q)} rejected ({100*rej/len(q):.2f}%)", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "sqi_rejections.csv", index=False)
    jdump(dict(threshold=thr, rows=rows,
               threshold_rule="second percentile of the index on the DS1 training beats"),
          RESULTS / "sqi.json")

    # ------------------------------------------------------- noise sweep
    noise = load_noise()
    rng = np.random.default_rng(0)
    sweep = []
    base_cm = confusion(ds2.y, p_after.argmax(1))
    base_f1 = float(metrics_from_cm(base_cm)["macro_f1"])
    from dataio import Split
    for label, wav in noise.items():
        for snr in (18, 12, 6, 0):
            Xn = add_noise(ds2.X, wav, snr, rng)
            sp = Split(Xn, ds2.y, ds2.R, ds2.rec, ds2.bidx, "noisy")
            pn = mean_probs(student_logits(sp), T)
            f1 = float(metrics_from_cm(confusion(ds2.y, pn.argmax(1)))["macro_f1"])
            sweep.append(dict(noise_type=label, snr_db=snr, macro_f1=f1))
            print(f"  {label} at {snr} dB: macro-F1 {f1:.4f}", flush=True)
    for hz in (50, 60):
        for snr in (18, 12, 6, 0):
            Xn = add_mains(ds2.X, hz, snr, rng)
            sp = Split(Xn, ds2.y, ds2.R, ds2.rec, ds2.bidx, "mains")
            pn = mean_probs(student_logits(sp), T)
            f1 = float(metrics_from_cm(confusion(ds2.y, pn.argmax(1)))["macro_f1"])
            sweep.append(dict(noise_type=f"{hz} Hz mains", snr_db=snr, macro_f1=f1))
    df = pd.DataFrame(sweep)
    df.to_csv(RESULTS / "noise_sweep.csv", index=False)
    jdump(dict(clean_macro_f1=base_f1, rows=sweep), RESULTS / "noise_sweep.json")

    # ------------------------------------------------------ alarm burden
    # DS2 holds 22 records of about 30 minutes each; false alarms are scaled to
    # 24 hours at each record's own beat rate, at the operating point of the
    # abstention rule.
    hours = 0.0
    import wfdb
    for r in sorted(set(ds2.rec)):
        h = wfdb.rdheader(str(DATA / "mitdb" / r))
        hours += h.sig_len / h.fs / 3600.0
    pred = p_after.argmax(1)
    kept = keep_ds2
    fp_v = int(((pred == 2) & (ds2.y != 2) & kept).sum())
    fp_s = int(((pred == 1) & (ds2.y != 1) & kept).sum())
    alarms = dict(monitored_hours=float(hours),
                  false_v=fp_v, false_s=fp_s,
                  false_v_per_24h=float(fp_v / hours * 24.0),
                  false_s_per_24h=float(fp_s / hours * 24.0),
                  coverage=float(kept.mean()),
                  false_v_per_24h_no_abstention=float(
                      ((pred == 2) & (ds2.y != 2)).sum() / hours * 24.0),
                  false_s_per_24h_no_abstention=float(
                      ((pred == 1) & (ds2.y != 1)).sum() / hours * 24.0))
    jdump(alarms, RESULTS / "alarms.json")
    print(f"  alarms per 24 h: V {alarms['false_v_per_24h']:.0f}, "
          f"S {alarms['false_s_per_24h']:.0f} over {hours:.1f} monitored hours", flush=True)

    np.savez_compressed(CACHE / "calibrated.npz", T=T, qhat=qhat,
                        p_ds2=p_after, y_ds2=ds2.y, rec_ds2=ds2.rec)
    print("CALIBRATION_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
