"""Stage 1 of the roadmap: build beat arrays from the four PhysioNet databases.

Chain, in order:
  exclude the four paced records (MIT-BIH only, as required by EC57)
  band-pass 0.5-40 Hz, zero-phase fourth-order Butterworth
  resample to 250 Hz by polyphase filtering
  normalise by the record's own median and median absolute deviation
  window each annotated R peak from 250 ms before to 400 ms after
  compute four rhythm descriptors
  map annotation symbols to the five AAMI superclasses
  drop the first ten beats of every record

Outputs one .npz per database in cache/ plus results/partition_counts.csv.

A second pass writes cache/mitdb_bw100.npz, the same MIT-BIH beats with a
100 Hz upper corner, for the bandwidth ablation of Table S3.5.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (AAMI, BP_HIGH, BP_LOW, CACHE, CLASS_IDX, DATA, DS1, DS1_TRAIN,
                    DS1_VAL, DS2, FS, PACED, POST_N, PRE_N, RESULTS, SKIP_FIRST, WIN)

LEAD_PREF = {
    "mitdb": ["MLII", "II", "V5", "V1"],
    "svdb": ["ECG1", "II", "MLII"],
    "incartdb": ["II"],
}


def pick_lead(sig_names: list[str], db: str) -> int:
    for want in LEAD_PREF.get(db, ["II"]):
        for i, nm in enumerate(sig_names):
            if nm.strip().upper() == want.upper():
                return i
    return 0


def bandpass(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    ny = fs / 2.0
    hi = min(high / ny, 0.99)
    sos = sps.butter(4, [low / ny, hi], btype="bandpass", output="sos")
    return sps.sosfiltfilt(sos, x)


def resample_to(x: np.ndarray, fs_in: float, fs_out: int) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-9:
        return x
    from math import gcd
    a, b = int(round(fs_out)), int(round(fs_in))
    g = gcd(a, b)
    return sps.resample_poly(x, a // g, b // g)


def rhythm_features(rp: np.ndarray, fs: int) -> np.ndarray:
    """Four descriptors per beat, in seconds and dimensionless ratios."""
    t = rp / float(fs)
    n = len(t)
    rr_prev = np.full(n, np.nan)
    rr_next = np.full(n, np.nan)
    rr_prev[1:] = np.diff(t)
    rr_next[:-1] = np.diff(t)
    rr_prev[0] = rr_prev[1] if n > 1 else 0.8
    rr_next[-1] = rr_next[-2] if n > 1 else 0.8

    local10 = np.empty(n)
    for i in range(n):
        lo = max(0, i - 10)
        seg = rr_prev[lo:i]
        seg = seg[np.isfinite(seg)]
        local10[i] = seg.mean() if seg.size else rr_prev[i]
    ratio_local = rr_prev / np.maximum(local10, 1e-6)
    ratio_next = rr_prev / np.maximum(rr_next, 1e-6)
    return np.stack([rr_prev, rr_next, ratio_local, ratio_next], axis=1).astype(np.float32)


def process_record(db: str, rec: str, bp_high: float = BP_HIGH):
    path = str(DATA / db / rec)
    r = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    ch = pick_lead(list(r.sig_name), db)
    x = np.asarray(r.p_signal[:, ch], dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0)
    fs_in = float(r.fs)

    x = bandpass(x, fs_in, BP_LOW, bp_high)
    x = resample_to(x, fs_in, FS)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    x = (x - med) / (1.4826 * mad + 1e-9)

    scale = FS / fs_in
    keep = np.array([i for i, s in enumerate(ann.symbol) if s in AAMI], dtype=int)
    if keep.size == 0:
        return None
    rp = np.round(np.asarray(ann.sample)[keep] * scale).astype(int)
    syms = [ann.symbol[i] for i in keep]
    order = np.argsort(rp)
    rp, syms = rp[order], [syms[i] for i in order]

    feats = rhythm_features(rp, FS)

    beats, labels, rfeat, idx = [], [], [], []
    for i in range(len(rp)):
        if i < SKIP_FIRST:
            continue
        a, b = rp[i] - PRE_N, rp[i] + POST_N + 1
        if a < 0 or b > len(x):
            continue
        w = x[a:b]
        if len(w) != WIN:
            continue
        beats.append(w.astype(np.float32))
        labels.append(CLASS_IDX[AAMI[syms[i]]])
        rfeat.append(feats[i])
        idx.append(i)

    if not beats:
        return None
    return (np.stack(beats), np.asarray(labels, dtype=np.int64),
            np.stack(rfeat), np.asarray(idx, dtype=np.int64), r)


def header_meta(db: str, rec: str) -> tuple[str, float]:
    """Subject sex and age from the WFDB header comments (MIT-BIH format)."""
    try:
        h = wfdb.rdheader(str(DATA / db / rec))
        for c in (h.comments or []):
            parts = c.split()
            if len(parts) < 2:
                continue
            try:
                age = float(parts[0])
            except ValueError:
                continue
            sex = parts[1].upper()[0]
            if sex in ("M", "F"):
                # MIT-BIH writes -1 where the age is not recorded
                return sex, (float("nan") if age < 0 else age)
    except Exception:  # noqa: BLE001
        pass
    return "U", float("nan")


def build_db(db: str, records: list[str], bp_high: float = BP_HIGH):
    X, y, R, rec_id, bidx = [], [], [], [], []
    meta = {}
    for rec in records:
        out = process_record(db, rec, bp_high)
        if out is None:
            print(f"    {db}/{rec}: no usable beats", flush=True)
            continue
        b, lab, rf, ix, _ = out
        X.append(b); y.append(lab); R.append(rf)
        rec_id.append(np.array([rec] * len(lab)))
        bidx.append(ix)
        meta[rec] = header_meta(db, rec)
        print(f"    {db}/{rec}: {len(lab):6d} beats", flush=True)
    return (np.concatenate(X), np.concatenate(y), np.concatenate(R),
            np.concatenate(rec_id), np.concatenate(bidx), meta)


def list_records(db: str) -> list[str]:
    recs = sorted(p.stem for p in (DATA / db).glob("*.hea"))
    if db == "mitdb":
        recs = [r for r in recs if r not in PACED]
    if db == "nstdb":
        recs = [r for r in recs if r in ("bw", "em", "ma")]
    return recs


def counts_frame(db_label: str, y, rid, part_of=None) -> list[dict]:
    rows = []
    for rec in sorted(set(rid.tolist())):
        m = rid == rec
        part = part_of[rec] if part_of else db_label
        for ci, cname in enumerate(["N", "S", "V", "F", "Q"]):
            rows.append({"partition": part, "record": rec, "class": cname,
                         "count": int(((y == ci) & m).sum())})
    return rows


def run_mitdb() -> list[dict]:
    print("[mitdb] 0.5-40 Hz", flush=True)
    recs = [r for r in list_records("mitdb") if r in DS1 + DS2]
    X, y, R, rid, bix, meta = build_db("mitdb", recs)
    np.savez_compressed(CACHE / "mitdb.npz", X=X, y=y, R=R, rec=rid, bidx=bix,
                        meta_rec=np.array(list(meta.keys())),
                        meta_sex=np.array([meta[k][0] for k in meta]),
                        meta_age=np.array([meta[k][1] for k in meta], dtype=np.float64))
    part_of = {}
    for r in DS1_TRAIN:
        part_of[r] = "DS1 training subset"
    for r in DS1_VAL:
        part_of[r] = "DS1 validation subset"
    for r in DS2:
        part_of[r] = "DS2 test"
    rows = counts_frame("mitdb", y, rid, part_of)

    print("[mitdb] 0.5-100 Hz for the bandwidth ablation", flush=True)
    X2, y2, R2, rid2, bix2, _ = build_db("mitdb", DS1, bp_high=100.0)
    np.savez_compressed(CACHE / "mitdb_bw100.npz", X=X2, y=y2, R=R2, rec=rid2, bidx=bix2)
    return rows


def run_external(db: str) -> list[dict]:
    print(f"[{db}]", flush=True)
    recs = list_records(db)
    X, y, R, rid, bix, meta = build_db(db, recs)
    np.savez_compressed(CACHE / f"{db}.npz", X=X, y=y, R=R, rec=rid, bidx=bix)
    return counts_frame(db, y, rid)


def main(argv: list[str]) -> int:
    t0 = time.time()
    want = argv[1:] or ["mitdb", "incartdb", "svdb"]
    for db in want:
        rows = run_mitdb() if db == "mitdb" else run_external(db)
        out = RESULTS / f"partition_counts_{db}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"    wrote {out.name}", flush=True)

    frames = [pd.read_csv(RESULTS / f"partition_counts_{d}.csv")
              for d in ("mitdb", "incartdb", "svdb")
              if (RESULTS / f"partition_counts_{d}.csv").exists()]
    if frames:
        pd.concat(frames).to_csv(RESULTS / "partition_counts.csv", index=False)
    print(f"PREPROCESS_COMPLETE in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
