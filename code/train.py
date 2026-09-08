"""Training loops for the neural arms, and the fitting routines for A, B, B2, C.

Every arm exposes the same interface: fit on a training Split, select on a
validation Split, then produce logits for any Split. Nothing here computes a
reported metric; that is evalutil's job, applied later to the raw predictions.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from dataio import class_weights
from evalutil import macro_f1, selection_f1
from models import (ClinicalFuzzy, CLINICAL_RULES, build_neural, classical_features,
                    clinical_features, count_params, ts_fuzzy_decode)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------- augmentation
def augment(x: torch.Tensor, r: torch.Tensor, g: torch.Generator, cfg: dict):
    """Label-preserving perturbations applied on the fly during training.

    Seventeen training records is a small number of subjects, and a network that
    is free to memorise their amplitude, their electrode gain and their exact
    sampling phase will do so. Each transform below removes one such shortcut.
    None of them changes the class of a beat: an amplitude change or a one-sample
    shift does not turn a ventricular beat into a normal one.
    """
    b, _, n = x.shape
    dev = x.device
    # multiplicative gain, log-uniform, models electrode contact and preamp gain
    lo, hi = cfg.get("gain", (0.75, 1.33))
    g_amp = torch.exp(torch.rand(b, 1, 1, generator=g, device=dev)
                      * (np.log(hi) - np.log(lo)) + np.log(lo))
    x = x * g_amp
    # sub-beat time shift, models R-peak detector jitter
    s = int(cfg.get("shift", 10))
    if s > 0:
        k = torch.randint(-s, s + 1, (b,), generator=g, device=dev)
        idx = (torch.arange(n, device=dev)[None, :] + k[:, None]).clamp(0, n - 1)
        x = torch.gather(x.squeeze(1), 1, idx).unsqueeze(1)
    # additive white noise and a slow baseline drift the 0.5 Hz corner leaves
    sd = float(cfg.get("noise", 0.08))
    x = x + torch.randn(x.shape, generator=g, device=dev) * (
        torch.rand(b, 1, 1, generator=g, device=dev) * sd)
    amp = torch.rand(b, 1, 1, generator=g, device=dev) * float(cfg.get("drift", 0.15))
    ph = torch.rand(b, 1, 1, generator=g, device=dev) * 2 * np.pi
    t = torch.linspace(0, 1, n, device=dev)[None, None, :]
    x = x + amp * torch.sin(2 * np.pi * t + ph)
    # small heart-rate scaling on the rhythm descriptors
    jr = float(cfg.get("rr_jitter", 0.04))
    r = r * (1.0 + torch.randn(r.shape, generator=g, device=dev) * jr)
    return x, r


# ------------------------------------------------------------------ neural
def standardise(train_R, R):
    mu, sd = train_R.mean(axis=0), train_R.std(axis=0) + 1e-6
    return ((R - mu) / sd).astype(np.float32), (mu, sd)


class NeuralArm:
    def __init__(self, arm: str, hp: dict, n_class: int = 5):
        self.arm, self.hp, self.n_class = arm, hp, n_class
        self.model = None
        self.rstats = None
        self.history = {}

    def _tensors(self, split, rstats=None):
        R = split.R if rstats is None else (split.R - rstats[0]) / rstats[1]
        x = torch.tensor(split.X[:, None, :], dtype=torch.float32)
        r = torch.tensor(np.asarray(R, dtype=np.float32))
        y = torch.tensor(split.y, dtype=torch.long)
        return x, r, y

    def fit(self, tr, va, seed: int = 0, epochs: int = 200, patience: int = 20,
            teacher_logits: np.ndarray | None = None, verbose: bool = False,
            use_rhythm: bool = True):
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        _, self.rstats = standardise(tr.R, tr.R)
        xtr, rtr, ytr = self._tensors(tr, self.rstats)
        xva, rva, yva = self._tensors(va, self.rstats)
        if not use_rhythm:
            rtr = torch.zeros_like(rtr)
            rva = torch.zeros_like(rva)

        w = torch.tensor(class_weights(tr.y, self.n_class, self.hp.get("weight_cap", 10.0)),
                         dtype=torch.float32, device=DEV)
        model = build_neural(self.arm, self.hp, self.n_class).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=self.hp.get("lr", 1e-3),
                                weight_decay=self.hp.get("weight_decay", 1e-3))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        bs = self.hp.get("batch", 256)
        rng = np.random.default_rng(seed)
        gen = torch.Generator(device=DEV); gen.manual_seed(seed)
        aug = dict(gain=(self.hp.get("gain_lo", 0.75), self.hp.get("gain_hi", 1.33)),
                   shift=self.hp.get("shift", 10), noise=self.hp.get("noise", 0.08),
                   drift=self.hp.get("drift", 0.15),
                   rr_jitter=self.hp.get("rr_jitter", 0.04))
        do_aug = bool(self.hp.get("augment", True))

        TL = None
        if teacher_logits is not None:
            TL = torch.tensor(teacher_logits, dtype=torch.float32)
        T = float(self.hp.get("temperature", 4.0))
        alpha = float(self.hp.get("alpha", 0.5))

        xtr_d, rtr_d, ytr_d = xtr.to(DEV), rtr.to(DEV), ytr.to(DEV)
        TL_d = TL.to(DEV) if TL is not None else None
        xva_d, rva_d = xva.to(DEV), rva.to(DEV)

        best, best_state, bad, best_ep, best_full = -1.0, None, 0, 0, float("nan")
        n = len(ytr)
        for ep in range(epochs):
            model.train()
            for bi in np.array_split(rng.permutation(n), max(1, n // bs)):
                bi_t = torch.from_numpy(bi).to(DEV)
                xb, rb = xtr_d[bi_t], rtr_d[bi_t]
                if do_aug:
                    xb, rb = augment(xb, rb, gen, aug)
                opt.zero_grad(set_to_none=True)
                out = model(xb, rb)
                loss = F.cross_entropy(out, ytr_d[bi_t], weight=w)
                if TL_d is not None:
                    kl = F.kl_div(F.log_softmax(out / T, dim=1),
                                  F.log_softmax(TL_d[bi_t] / T, dim=1),
                                  reduction="batchmean", log_target=True)
                    loss = alpha * loss + (1.0 - alpha) * (T ** 2) * kl
                loss.backward()
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                pv = model(xva_d, rva_d).argmax(1).cpu().numpy()
            f1 = selection_f1(va.y, pv, self.n_class)
            if f1 > best + 1e-6:
                best, bad, best_ep = f1, 0, ep
                best_full = macro_f1(va.y, pv, self.n_class)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
            if verbose and ep % 10 == 0:
                print(f"    ep {ep:3d} val selection-F1 {f1:.4f} (best {best:.4f})", flush=True)

        model.load_state_dict(best_state)
        self.model = model.eval()
        with torch.no_grad():
            ptr = model(xtr_d, rtr_d).argmax(1).cpu().numpy()
        self.history = dict(val_selection_f1=float(best),
                            val_macro_f1=float(best_full),
                            train_macro_f1=float(macro_f1(tr.y, ptr, self.n_class)),
                            train_selection_f1=float(selection_f1(tr.y, ptr, self.n_class)),
                            epochs_to_stop=int(best_ep + 1),
                            epochs_run=int(ep + 1),
                            wallclock_min=(time.time() - t0) / 60.0,
                            params=count_params(model))
        return self

    def logits(self, split, use_rhythm: bool = True) -> np.ndarray:
        x, r, _ = self._tensors(split, self.rstats)
        if not use_rhythm:
            r = torch.zeros_like(r)
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 4096):
                out.append(self.model(x[i:i + 4096].to(DEV),
                                      r[i:i + 4096].to(DEV)).cpu().numpy())
        return np.concatenate(out)


# --------------------------------------------------------------- arm A
def fit_arm_a(tr, va, seed: int = 0, hp: dict | None = None):
    """Five classical baselines; the best on validation macro-F1 is arm A."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC, SVC
    from xgboost import XGBClassifier

    hp = hp or {}
    Ftr = classical_features(tr.X, tr.R)
    Fva = classical_features(va.X, va.R)
    sc = StandardScaler().fit(Ftr)
    Ztr, Zva = sc.transform(Ftr), sc.transform(Fva)

    cw = class_weights(tr.y)
    sw = cw[tr.y]
    C = hp.get("C", 1.0)
    cands = {
        "logistic regression": LogisticRegression(C=C, max_iter=800, class_weight="balanced"),
        "linear SVM": LinearSVC(C=C, class_weight="balanced", max_iter=2000, dual="auto", tol=1e-3),
        # the RBF SVM is bounded in iterations: at the top of the C search
        # range an unbounded fit does not terminate in a usable time, and a
        # capped fit is still a fair representative of the family
        "RBF SVM": SVC(C=C, gamma="scale", class_weight="balanced",
                       probability=False, cache_size=2000, max_iter=60_000),
        "random forest": RandomForestClassifier(
            n_estimators=hp.get("rf_trees", 400), class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed, max_depth=hp.get("rf_depth", None),
            min_samples_leaf=2),
        "gradient-boosted trees": XGBClassifier(
            n_estimators=400, max_depth=hp.get("gb_depth", 6),
            learning_rate=hp.get("gb_lr", 0.1), subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", device="cuda" if DEV == "cuda" else "cpu",
            random_state=seed, num_class=5, objective="multi:softprob"),
    }
    # the RBF SVM does not scale to 39k x 36 with probability estimation, so it
    # is fitted on a stratified subsample and scored on the full validation set
    results, full = {}, {}
    fitted = {}
    for name, clf in cands.items():
        Zt, yt, swt = Ztr, tr.y, sw
        if name == "RBF SVM":
            rng = np.random.default_rng(seed)
            keep = np.concatenate([rng.choice(np.flatnonzero(tr.y == c),
                                              size=min(800, int((tr.y == c).sum())),
                                              replace=False)
                                   for c in range(5) if (tr.y == c).sum() > 0])
            Zt, yt, swt = Ztr[keep], tr.y[keep], sw[keep]
        try:
            if name == "gradient-boosted trees":
                clf.fit(Zt, yt, sample_weight=swt)
            else:
                clf.fit(Zt, yt)
            pv = clf.predict(Zva)
            f1 = selection_f1(va.y, pv)
            results[name] = float(f1)
            full[name] = float(macro_f1(va.y, pv))
            fitted[name] = clf
        except Exception as exc:  # noqa: BLE001
            results[name] = float("nan")
            print(f"    arm A: {name} failed: {exc}", flush=True)
    best_name = max(results, key=lambda k: (results[k] if np.isfinite(results[k]) else -1))
    return dict(scaler=sc, model=fitted[best_name], name=best_name,
                per_model=results, per_model_macro_f1=full,
                val_selection_f1=results[best_name], val_macro_f1=full[best_name])


def arm_a_logits(fit, split) -> np.ndarray:
    Z = fit["scaler"].transform(classical_features(split.X, split.R))
    m = fit["model"]
    if hasattr(m, "predict_proba"):
        p = m.predict_proba(Z)
        return np.log(np.clip(p, 1e-12, 1.0))
    d = m.decision_function(Z)
    return d if d.ndim == 2 else np.stack([-d, d], axis=1)


# --------------------------------------------------------------- arm B2
B2_OBJECTIVE_BEATS = 12000


def fit_arm_b2(tr, va, seed: int = 0, budget: int = 15000, pop: int = 50):
    """Arm B2: the clinical-quantity fuzzy system, membership widths fitted by GPC.

    The GPC objective is evaluated on a fixed stratified subsample of the
    training beats, drawn once with a fixed seed, so that a 15,000-evaluation
    budget is affordable. The fitted system is then scored on the whole
    training and validation subsets.
    """
    from optimisers import Budget, gpc

    Ctr = clinical_features(tr.X, tr.R)
    Cva = clinical_features(va.X, va.R)
    mu, sd = Ctr.mean(axis=0), Ctr.std(axis=0) + 1e-6
    Ztr, Zva = (Ctr - mu) / sd, (Cva - mu) / sd
    fz = ClinicalFuzzy()
    w = class_weights(tr.y)

    rng = np.random.default_rng(4242)
    cls, cnt = np.unique(tr.y, return_counts=True)
    share = np.maximum(1, np.round(B2_OBJECTIVE_BEATS * cnt / cnt.sum()).astype(int))
    keep = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(tr.y == c), size=min(k, int((tr.y == c).sum())),
                   replace=False) for c, k in zip(cls, share)]))
    Zs, ys = Ztr[keep], tr.y[keep]
    wbs = w[ys]
    rows = np.arange(len(ys))

    def loss_batch(TH):
        out = np.empty(len(TH))
        for i, th in enumerate(TH):
            p = fz.predict_proba(Zs, th)
            ll = -np.log(np.clip(p[rows, ys], 1e-9, 1.0))
            out[i] = float((ll * wbs).mean())
        return out

    b = Budget(loss_batch, budget)
    gpc(b, fz.n_params, -3.0, 3.0, seed, pop=pop)
    fz.theta = b.best_x
    pv = fz.predict_proba(Zva, b.best_x).argmax(1)
    pt = fz.predict_proba(Ztr, b.best_x).argmax(1)
    return dict(fuzzy=fz, theta=b.best_x, mu=mu, sd=sd,
                val_selection_f1=float(selection_f1(va.y, pv)),
                val_macro_f1=float(macro_f1(va.y, pv)),
                train_macro_f1=float(macro_f1(tr.y, pt)),
                train_selection_f1=float(selection_f1(tr.y, pt)),
                evals=b.used)


def arm_b2_logits(fit, split) -> np.ndarray:
    Z = (clinical_features(split.X, split.R) - fit["mu"]) / fit["sd"]
    p = fit["fuzzy"].predict_proba(Z, fit["theta"])
    return np.log(np.clip(p, 1e-12, 1.0))


# ------------------------------------------------------------ arms B and C
def arm_b_logits(theta, sel, mu, sd, split, sigma_low=0.25, sigma_high=0.25):
    """Fuzzy-decoder scores. Arm C is the same z with argmax instead."""
    Z = ((classical_features(split.X, split.R) - mu) / sd)[:, sel]
    from models import PolySigmoidScorer
    sc = PolySigmoidScorer(n_feat=len(sel), n_class=5)
    z = sc.scores(Z.astype(np.float64), np.asarray(theta, dtype=np.float64))
    wf = ts_fuzzy_decode(z, sigma_low, sigma_high)
    return np.log(np.clip(wf, 1e-12, 1.0)), np.log(np.clip(z / z.sum(axis=1, keepdims=True), 1e-12, 1.0))
