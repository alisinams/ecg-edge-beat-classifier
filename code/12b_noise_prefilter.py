"""Noise stress test, added to the raw record before filtering.

This replaces an earlier version that added noise to the extracted beat windows.
Adding it there is wrong for the question the figure asks. The manuscript claims
that the 0.5 Hz corner removes baseline wander and the 40 Hz corner removes mains
interference, and neither claim is testable if the noise is injected downstream
of the filter that is supposed to remove it. Here the noise is mixed into the
raw signal at its native sampling rate, and the whole chain of section 6.2 then
runs on the contaminated record: band-pass, resample, normalise, segment.

Signal power is the variance of the raw record. Noise power is the variance of
the noise segment. The scale factor sets the ratio to the target signal-to-noise
ratio, which is the convention of the Noise Stress Test Database.
"""
from __future__ import annotations

import pickle
import sys
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (AAMI, BP_HIGH, BP_LOW, CACHE, CLASS_IDX, DATA, DS2, FS,
                    POST_N, PRE_N, RESULTS, SEEDS, SKIP_FIRST, WIN, jdump, jload)
from dataio import Split
from evalutil import confusion, metrics_from_cm, softmax

import importlib.util
_p = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("prep", str(_p / "02_preprocess.py"))
_prep = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_prep)

ARMDIR = CACHE / "arms"
SNRS = (18, 12, 6, 0)
_STUDENTS = None


def students():
    global _STUDENTS
    if _STUDENTS is None:
        _STUDENTS = [pickle.loads((ARMDIR / f"ds1_E_s{s}.pkl").read_bytes())["model"]
                     for s in SEEDS]
    return _STUDENTS


def mean_probs(split, T):
    L = np.stack([m.logits(split) for m in students()])
    return softmax((L / T).reshape(-1, 5)).reshape(L.shape).mean(axis=0)


def load_noise_raw():
    """The three noise records, kept at their native 360 Hz."""
    out = {}
    for rec, label in (("bw", "baseline wander"), ("ma", "muscle artifact"),
                       ("em", "electrode motion")):
        r = wfdb.rdrecord(str(DATA / "nstdb" / rec))
        x = np.nan_to_num(np.asarray(r.p_signal[:, 0], dtype=np.float64))
        out[label] = (x, float(r.fs))
    return out


def resample_to(x, fs_in, fs_out):
    if abs(fs_in - fs_out) < 1e-9:
        return x
    a, b = int(round(fs_out)), int(round(fs_in))
    g = gcd(a, b)
    return sps.resample_poly(x, a // g, b // g)


def contaminate_and_segment(noise_spec, snr_db, seed):
    """Run the full chain of section 6.2 on every DS2 record with noise added."""
    rng = np.random.default_rng(seed)
    X, y, R, recs, bidx = [], [], [], [], []
    for rec in DS2:
        path = str(DATA / "mitdb" / rec)
        r = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, "atr")
        ch = _prep.pick_lead(list(r.sig_name), "mitdb")
        x = np.nan_to_num(np.asarray(r.p_signal[:, ch], dtype=np.float64))
        fs_in = float(r.fs)

        if noise_spec is not None:
            kind, payload = noise_spec
            if kind == "record":
                nz, nfs = payload
                nz = resample_to(nz, nfs, fs_in)
                if len(nz) < len(x):
                    nz = np.tile(nz, int(np.ceil(len(x) / len(nz))))
                start = int(rng.integers(0, max(1, len(nz) - len(x))))
                seg = nz[start:start + len(x)]
            else:                                    # synthetic mains
                hz = payload
                t = np.arange(len(x)) / fs_in
                seg = np.sin(2 * np.pi * hz * t + rng.uniform(0, 2 * np.pi))
            ps, pn = float(np.var(x)), float(np.var(seg)) + 1e-12
            x = x + seg * np.sqrt(ps / (pn * 10 ** (snr_db / 10.0)))

        x = _prep.bandpass(x, fs_in, BP_LOW, BP_HIGH)
        x = resample_to(x, fs_in, FS)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        x = (x - med) / (1.4826 * mad + 1e-9)

        scale = FS / fs_in
        keep = [i for i, s in enumerate(ann.symbol) if s in AAMI]
        rp = np.round(np.asarray(ann.sample)[keep] * scale).astype(int)
        syms = [ann.symbol[i] for i in keep]
        feats = _prep.rhythm_features(rp, FS)
        for i in range(len(rp)):
            if i < SKIP_FIRST:
                continue
            a, b = rp[i] - PRE_N, rp[i] + POST_N + 1
            if a < 0 or b > len(x):
                continue
            w = x[a:b]
            if len(w) != WIN:
                continue
            X.append(w.astype(np.float32)); y.append(CLASS_IDX[AAMI[syms[i]]])
            R.append(feats[i]); recs.append(rec); bidx.append(i)
    return Split(np.stack(X), np.asarray(y, dtype=np.int64), np.stack(R),
                 np.asarray(recs), np.asarray(bidx, dtype=np.int64), "noisy")


def main() -> int:
    cal = jload(RESULTS / "calibration.json")
    T = float(cal["temperature"])
    noise = load_noise_raw()

    clean = contaminate_and_segment(None, 0, 0)
    p = mean_probs(clean, T)
    base = float(metrics_from_cm(confusion(clean.y, p.argmax(1)))["macro_f1"])
    print(f"  clean signal, rebuilt through the same chain: macro-F1 {base:.4f}", flush=True)

    rows = []
    for label, (wav, nfs) in noise.items():
        for snr in SNRS:
            sp = contaminate_and_segment(("record", (wav, nfs)), snr, hash(label) % 1000 + snr)
            f1 = float(metrics_from_cm(confusion(
                sp.y, mean_probs(sp, T).argmax(1)))["macro_f1"])
            rows.append(dict(noise_type=label, snr_db=snr, macro_f1=f1))
            print(f"  {label} at {snr} dB: macro-F1 {f1:.4f}", flush=True)
    for hz in (50, 60):
        for snr in SNRS:
            sp = contaminate_and_segment(("mains", hz), snr, hz + snr)
            f1 = float(metrics_from_cm(confusion(
                sp.y, mean_probs(sp, T).argmax(1)))["macro_f1"])
            rows.append(dict(noise_type=f"{hz} Hz mains", snr_db=snr, macro_f1=f1))
            print(f"  {hz} Hz mains at {snr} dB: macro-F1 {f1:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(RESULTS / "noise_sweep.csv", index=False)
    jdump(dict(clean_macro_f1=base, rows=rows,
               injection="added to the raw record before the band-pass filter, then "
                         "the whole chain of section 6.2 was rerun on the contaminated "
                         "record"),
          RESULTS / "noise_sweep.json")
    print("NOISE_PREFILTER_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
