"""Arm B stage one: feature selection, the objective, and the fitted scorer.

The same objective object serves the optimiser comparison of section 6.7 and the
fit that arms B, B2 and C are built on, so Table 7 and Table 3 are measuring the
same function.
"""
from __future__ import annotations

import numpy as np
import torch

from models import PolySigmoidScorer, classical_features

BOX_LO, BOX_HI = -5.0, 5.0
D_SELECT = 13


class StageOneObjective:
    """Class-weighted one-vs-rest binary cross-entropy on the five sigmoid scores.

    Differentiable in every parameter, which is why section 6.7 can put a
    metaheuristic against gradient descent at all. A numpy path evaluates a whole
    population in one call; a torch path supplies the analytic gradient.
    """

    def __init__(self, Z: np.ndarray, y: np.ndarray, w: np.ndarray,
                 n_class: int = 5, device: str | None = None):
        self.scorer = PolySigmoidScorer(n_feat=Z.shape[1], n_class=n_class)
        self.dim = self.scorer.n_params
        self.C, self.D = n_class, Z.shape[1]
        self.Z = Z.astype(np.float32)
        self.Z2 = (self.Z ** 2)
        self.Z3 = (self.Z ** 3)
        self.Y = np.zeros((len(y), n_class), dtype=np.float32)
        self.Y[np.arange(len(y)), y] = 1.0
        self.wb = w[y].astype(np.float32)           # per-beat weight
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tZ = torch.tensor(self.Z, device=self.device)
        self.tZ2 = torch.tensor(self.Z2, device=self.device)
        self.tZ3 = torch.tensor(self.Z3, device=self.device)
        self.tY = torch.tensor(self.Y, device=self.device)
        self.twb = torch.tensor(self.wb, device=self.device)

    # ---- shared maths -------------------------------------------------
    def _split(self, TH):
        C, D = self.C, self.D
        k = C * D
        return (TH[:, 0:k].reshape(-1, C, D),
                TH[:, k:2 * k].reshape(-1, C, D),
                TH[:, 2 * k:3 * k].reshape(-1, C, D),
                TH[:, 3 * k:3 * k + C],
                TH[:, 3 * k + C:3 * k + 2 * C])

    def loss_torch(self, TH: torch.Tensor) -> torch.Tensor:
        w3, w2, w1, a, b = self._split(TH)
        s = (torch.einsum("nd,pcd->pnc", self.tZ3, w3)
             + torch.einsum("nd,pcd->pnc", self.tZ2, w2)
             + torch.einsum("nd,pcd->pnc", self.tZ, w1))
        aa = 0.1 + 4.9 * torch.sigmoid(a)
        logit = aa[:, None, :] * (s - b[:, None, :])
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, self.tY[None].expand_as(logit), reduction="none")
        return (bce.mean(dim=2) * self.twb[None, :]).mean(dim=1)

    def loss_numpy(self, TH: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(np.atleast_2d(TH), dtype=torch.float32, device=self.device)
            return self.loss_torch(t).cpu().numpy().astype(np.float64)

    # ---- decoding -----------------------------------------------------
    def scores(self, Z: np.ndarray, theta: np.ndarray) -> np.ndarray:
        sc = PolySigmoidScorer(n_feat=Z.shape[1], n_class=self.C)
        return sc.scores(Z.astype(np.float64), np.asarray(theta, dtype=np.float64))


def gwo_select(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
               d_select: int = D_SELECT, pop: int = 20, iters: int = 30,
               seed: int = 0):
    """Grey Wolf Optimization for feature selection.

    Each wolf is a continuous vector over the feature pool; the d_select
    largest components name the selected features. The wrapper score is the
    validation macro-F1 of a multinomial logistic regression on those features,
    which is cheap enough to run inside the search and does not presume the
    architecture that will later use the features.
    """
    from sklearn.linear_model import LogisticRegression
    from evalutil import macro_f1

    rng = np.random.default_rng(seed)
    n_feat = Xtr.shape[1]
    W = rng.uniform(0, 1, size=(pop, n_feat))
    cache: dict[tuple, float] = {}

    def score(v):
        sel = tuple(sorted(np.argsort(-v)[:d_select].tolist()))
        if sel in cache:
            return cache[sel], sel
        clf = LogisticRegression(max_iter=400, class_weight="balanced", n_jobs=-1)
        clf.fit(Xtr[:, list(sel)], ytr)
        f = macro_f1(yva, clf.predict(Xva[:, list(sel)]))
        cache[sel] = f
        return f, sel

    fits = np.array([score(w)[0] for w in W])
    order = np.argsort(-fits)
    alpha, beta, delta = W[order[0]].copy(), W[order[1]].copy(), W[order[2]].copy()
    fa = fits[order[0]]

    for it in range(iters):
        a = 2.0 - 2.0 * it / max(1, iters - 1)
        for i in range(pop):
            new = np.zeros(n_feat)
            for leader in (alpha, beta, delta):
                A = 2 * a * rng.random(n_feat) - a
                Cc = 2 * rng.random(n_feat)
                Dl = np.abs(Cc * leader - W[i])
                new += leader - A * Dl
            W[i] = np.clip(new / 3.0, 0, 1)
        fits = np.array([score(w)[0] for w in W])
        order = np.argsort(-fits)
        if fits[order[0]] > fa:
            fa = fits[order[0]]
            alpha = W[order[0]].copy()
        beta, delta = W[order[1]].copy(), W[order[2]].copy()

    sel = sorted(np.argsort(-alpha)[:d_select].tolist())
    return np.array(sel), float(fa), len(cache)


def feature_names() -> list[str]:
    names = ["RR previous (s)", "RR next (s)", "RR previous / local mean of 10",
             "RR previous / RR next"]
    names += [f"morphology sample {i + 1} of 32" for i in range(32)]
    return names


def build_pool(split, mu=None, sd=None):
    """Full feature pool, standardised with DS1 training statistics."""
    F = classical_features(split.X, split.R)
    if mu is None:
        mu, sd = F.mean(axis=0), F.std(axis=0) + 1e-6
    return ((F - mu) / sd).astype(np.float32), mu, sd
