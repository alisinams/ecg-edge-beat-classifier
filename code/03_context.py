"""Five-second context windows at 100 Hz for the teacher network.

The teacher is a public ECG foundation model that expects five seconds of
signal, not a single beat window, so a second pass over the DS1 records stores a
five-second context centred on every DS1 beat. DS2 is never touched here: the
teacher exists only to supply soft targets on the DS1 training subset, and the
student is what gets evaluated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AAMI, BP_HIGH, BP_LOW, CACHE, CLASS_IDX, DATA, DS1, SKIP_FIRST

FS_CTX = 100          # Hz, the rate the teacher was pretrained at
CTX_SEC = 5
CTX_N = FS_CTX * CTX_SEC          # 500 samples per lead
N_LEADS = 12                      # the teacher's input convention

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "prep", str(Path(__file__).resolve().parent / "02_preprocess.py"))
_prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prep)


def main() -> int:
    ctxs, labels, recs, bidx = [], [], [], []
    for rec in DS1:
        path = str(DATA / "mitdb" / rec)
        r = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, "atr")
        ch = _prep.pick_lead(list(r.sig_name), "mitdb")
        x = np.nan_to_num(np.asarray(r.p_signal[:, ch], dtype=np.float64))
        x = _prep.bandpass(x, float(r.fs), BP_LOW, BP_HIGH)
        x = _prep.resample_to(x, float(r.fs), FS_CTX)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        x = (x - med) / (1.4826 * mad + 1e-9)

        scale = FS_CTX / float(r.fs)
        keep = [i for i, s in enumerate(ann.symbol) if s in AAMI]
        rp = np.round(np.asarray(ann.sample)[keep] * scale).astype(int)
        syms = [ann.symbol[i] for i in keep]
        half = CTX_N // 2
        n_kept = 0
        for i in range(len(rp)):
            if i < SKIP_FIRST:
                continue
            a, b = rp[i] - half, rp[i] + half
            if a < 0 or b > len(x):
                continue
            ctxs.append(x[a:b].astype(np.float32))
            labels.append(CLASS_IDX[AAMI[syms[i]]])
            recs.append(rec)
            bidx.append(i)          # same annotation index the beat cache uses
            n_kept += 1
        print(f"  {rec}: {n_kept} contexts", flush=True)

    C = np.stack(ctxs)
    np.savez_compressed(CACHE / "ds1_context.npz", C=C,
                        y=np.asarray(labels, dtype=np.int64),
                        rec=np.asarray(recs),
                        bidx=np.asarray(bidx, dtype=np.int64))
    print("CONTEXT_COMPLETE", C.shape, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
