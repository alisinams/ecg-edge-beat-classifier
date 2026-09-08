"""Fine-tune the teacher network and export its soft targets.

The teacher is HuBERT-ECG small, a public ECG foundation model pretrained by
masked-unit prediction on large public corpora and released with open weights.
It expects five seconds of twelve-lead signal at 100 Hz flattened into a single
6000-sample sequence. This study has one lead, so the single available lead is
written into every one of the twelve input slots and the whole network is
fine-tuned on the DS1 training subset, which lets it adapt to that convention.

Two additions to the pretrained backbone. Frames are pooled by a single learned
attention query rather than averaged, because the label belongs to the beat at
the centre of the window and a flat average over five seconds dilutes it. And
the classification head also receives the four rhythm descriptors of section
6.3, the same four the student receives, so that the soft targets the student
learns from are not systematically worse than the student on the timing-driven
S class.

The teacher is never evaluated on DS2 and never sees a DS2 beat. Its only role
is to supply logits on DS1 training beats for arm E to distil from.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DS1_TRAIN, DS1_VAL, RESULTS, jdump
from dataio import class_weights, mitdb_splits
from evalutil import macro_f1, per_class, selection_f1

MODEL_ID = "Edoardo-Coppola/hubert-ecg-small"
N_LEADS = 12
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class AttentionPool(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.q = nn.Linear(h, 1)

    def forward(self, x):                      # (B, T, H)
        a = torch.softmax(self.q(x).squeeze(-1), dim=1)
        return torch.einsum("bt,bth->bh", a, x)


class Teacher(nn.Module):
    def __init__(self, n_class: int = 5, n_rhythm: int = 4):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
        h = self.backbone.config.hidden_size
        self.pool = AttentionPool(h)
        self.norm = nn.LayerNorm(h)
        self.head = nn.Sequential(
            nn.Linear(h + n_rhythm, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, n_class))

    def forward(self, x, r):                   # x: (B, 500) at 100 Hz, r: (B, 4)
        b = x.shape[0]
        rep = x.unsqueeze(1).expand(b, N_LEADS, x.shape[1]).reshape(b, -1)
        h = self.norm(self.pool(self.backbone(rep).last_hidden_state))
        return self.head(torch.cat([h, r], dim=1))


def align_rhythm(ctx_rec, ctx_bidx):
    """Rhythm descriptors for every context window, from the beat cache."""
    z = np.load(CACHE / "mitdb.npz", allow_pickle=True)
    key = {(r, int(b)): i for i, (r, b) in
           enumerate(zip(z["rec"].astype(str), z["bidx"]))}
    R = np.zeros((len(ctx_rec), z["R"].shape[1]), dtype=np.float32)
    miss = 0
    for i, (r, b) in enumerate(zip(ctx_rec, ctx_bidx)):
        j = key.get((str(r), int(b)))
        if j is None:
            miss += 1
        else:
            R[i] = z["R"][j]
    return R, miss


def batches(n, bs, shuffle, rng=None):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def main() -> int:
    t0 = time.time()
    z = np.load(CACHE / "ds1_context.npz", allow_pickle=True)
    C, y, rec, bidx = z["C"], z["y"], z["rec"].astype(str), z["bidx"]
    R, miss = align_rhythm(rec, bidx)
    tr = np.isin(rec, DS1_TRAIN)
    va = np.isin(rec, DS1_VAL)
    mu, sd = R[tr].mean(axis=0), R[tr].std(axis=0) + 1e-6
    R = ((R - mu) / sd).astype(np.float32)
    print(f"teacher contexts: train {tr.sum()}, val {va.sum()}, "
          f"rhythm descriptors missing for {miss}", flush=True)

    w = torch.tensor(class_weights(y[tr]), dtype=torch.float32, device=DEV)
    print("  class weights:", np.round(w.cpu().numpy(), 4).tolist(), flush=True)
    model = Teacher().to(DEV)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 3e-5},
        {"params": list(model.pool.parameters()) + list(model.norm.parameters())
         + list(model.head.parameters()), "lr": 3e-4}], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))

    Xtr, Rtr, ytr = torch.from_numpy(C[tr]), torch.from_numpy(R[tr]), torch.from_numpy(y[tr])
    Xva, Rva, yva = torch.from_numpy(C[va]), torch.from_numpy(R[va]), y[va]
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    best, best_state, patience = -1.0, None, 0
    EPOCHS, BS = 12, 96
    steps = EPOCHS * int(np.ceil(tr.sum() / BS))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[5e-5, 5e-4], total_steps=steps, pct_start=0.15)

    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for bi in batches(len(ytr), BS, True, rng):
            xb, rb = Xtr[bi].to(DEV), Rtr[bi].to(DEV)
            yb = ytr[bi].to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
                loss = F.cross_entropy(model(xb, rb), yb, weight=w)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss) * len(bi)

        model.eval()
        preds = []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
            for bi in batches(len(yva), 256, False):
                preds.append(model(Xva[bi].to(DEV), Rva[bi].to(DEV)).float().argmax(1).cpu().numpy())
        pv = np.concatenate(preds)
        sf1, f1 = selection_f1(yva, pv), macro_f1(yva, pv)
        pc = per_class(yva, pv)
        detail = " ".join(f"{c}:{pc[c]['f1']:.2f}" for c in "NSVFQ")
        print(f"  epoch {ep}: loss {tot/len(ytr):.4f}  sel-F1 {sf1:.4f}  "
              f"macro-F1 {f1:.4f}  {detail}", flush=True)
        if sf1 > best:
            best, patience = sf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 3:
                break

    model.load_state_dict(best_state)
    model.eval()

    logits = np.zeros((len(y), 5), dtype=np.float32)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
        for i in range(0, len(y), 256):
            sl = slice(i, min(i + 256, len(y)))
            logits[sl] = model(torch.from_numpy(C[sl]).to(DEV),
                               torch.from_numpy(R[sl]).to(DEV)).float().cpu().numpy()

    np.savez_compressed(CACHE / "teacher_logits.npz", logits=logits, y=y, rec=rec, bidx=bidx)
    torch.save(best_state, CACHE / "teacher_state.pt")
    n_par = sum(p.numel() for p in model.parameters())
    jdump({"model_id": MODEL_ID, "parameters": int(n_par),
           "val_selection_f1": float(best), "epochs_run": ep + 1,
           "wallclock_min": (time.time() - t0) / 60.0,
           "input": "5 s at 100 Hz, single lead replicated into 12 slots, "
                    "attention pooling, four rhythm descriptors at the head"},
          RESULTS / "teacher.json")
    print(f"TEACHER_COMPLETE best val selection-F1 {best:.4f}, "
          f"{n_par/1e6:.1f}M params, {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
