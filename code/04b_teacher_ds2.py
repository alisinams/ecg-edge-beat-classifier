"""Teacher soft targets for the DS2 records, used by the patient-mixed arm only.

The protocol contrast of section 7.3 refits every arm on a partition in which a
subject appears on both sides. Arm E is a distilled student, so refitting it
there needs soft targets for the DS2 beats as well. Those targets come from the
same teacher checkpoint, fine-tuned on the DS1 training records and never
refitted here, and they are written to a separate file that the inter-patient
pipeline never reads. Nothing in this file touches the DS1-to-DS2 result: it
exists so that the patient-mixed arm is a fair copy of the inter-patient arm
rather than a version of it with half its supervision removed.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import wfdb

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AAMI, BP_HIGH, BP_LOW, CACHE, CLASS_IDX, DATA, DS2, SKIP_FIRST

import importlib.util
_p = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("prep", str(_p / "02_preprocess.py"))
_prep = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_prep)
_spec2 = importlib.util.spec_from_file_location("teach", str(_p / "04_teacher.py"))
_teach = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(_teach)

FS_CTX, CTX_N = 100, 500
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build_contexts(records):
    ctxs, labels, recs, bidx = [], [], [], []
    for rec in records:
        path = str(DATA / "mitdb" / rec)
        r = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, "atr")
        ch = _prep.pick_lead(list(r.sig_name), "mitdb")
        x = np.nan_to_num(np.asarray(r.p_signal[:, ch], dtype=np.float64))
        x = _prep.bandpass(x, float(r.fs), BP_LOW, BP_HIGH)
        x = _prep.resample_to(x, float(r.fs), FS_CTX)
        med = np.median(x); mad = np.median(np.abs(x - med))
        x = (x - med) / (1.4826 * mad + 1e-9)
        scale = FS_CTX / float(r.fs)
        keep = [i for i, s in enumerate(ann.symbol) if s in AAMI]
        rp = np.round(np.asarray(ann.sample)[keep] * scale).astype(int)
        syms = [ann.symbol[i] for i in keep]
        half = CTX_N // 2
        for i in range(len(rp)):
            if i < SKIP_FIRST:
                continue
            a, b = rp[i] - half, rp[i] + half
            if a < 0 or b > len(x):
                continue
            ctxs.append(x[a:b].astype(np.float32))
            labels.append(CLASS_IDX[AAMI[syms[i]]])
            recs.append(rec); bidx.append(i)
        print(f"  {rec}: {len(recs)} cumulative", flush=True)
    return (np.stack(ctxs), np.asarray(labels, dtype=np.int64),
            np.asarray(recs), np.asarray(bidx, dtype=np.int64))


def main() -> int:
    out = CACHE / "teacher_logits_ds2.npz"
    if out.exists():
        print("teacher DS2 logits already present", flush=True)
        return 0
    C, y, rec, bidx = build_contexts(DS2)
    R, miss = _teach.align_rhythm(rec, bidx)
    # the same standardisation the teacher saw, taken from the DS1 training beats
    z = np.load(CACHE / "ds1_context.npz", allow_pickle=True)
    from common import DS1_TRAIN
    Rtr, _ = _teach.align_rhythm(z["rec"].astype(str), z["bidx"])
    m = np.isin(z["rec"].astype(str), DS1_TRAIN)
    mu, sd = Rtr[m].mean(axis=0), Rtr[m].std(axis=0) + 1e-6
    R = ((R - mu) / sd).astype(np.float32)

    model = _teach.Teacher().to(DEV)
    model.load_state_dict(torch.load(CACHE / "teacher_state.pt", map_location=DEV))
    model.eval()
    logits = np.zeros((len(y), 5), dtype=np.float32)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
        for i in range(0, len(y), 256):
            sl = slice(i, min(i + 256, len(y)))
            logits[sl] = model(torch.from_numpy(C[sl]).to(DEV),
                               torch.from_numpy(R[sl]).to(DEV)).float().cpu().numpy()
    np.savez_compressed(out, logits=logits, y=y, rec=rec, bidx=bidx)
    print(f"TEACHER_DS2_COMPLETE {logits.shape}, rhythm missing for {miss}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
