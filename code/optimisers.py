"""Six optimisers on one objective at one budget.

Every optimiser fits the same 205 parameters of arm B's stage one, inside the
same box, against the same objective, and stops at the same number of objective
evaluations. Population-based methods evaluate a whole population in one batched
call, so the budget is counted in candidate evaluations, not in generations.

WHOA  Wild Horse Optimizer, Naruei and Keynia 2022
GPC   Giza Pyramids Construction, Harifi et al. 2021
RS    uniform random search, Bergstra and Bengio 2012
CMAES covariance matrix adaptation, Hansen and Ostermeier 2001, via the cma package
TPE   tree-structured Parzen estimator, Bergstra et al. 2011, univariate form
GD    projected Adam on the analytic gradient
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RunResult:
    optimiser: str
    seed: int
    best: float
    evals: int
    wallclock_s: float
    curve_evals: list = field(default_factory=list)
    curve_best: list = field(default_factory=list)


class Budget:
    """Counts evaluations and records the best-so-far curve."""

    def __init__(self, fn_batch, total: int, record_every: int = 250):
        self.fn = fn_batch
        self.total = total
        self.used = 0
        self.best = np.inf
        self.best_x = None
        self.record_every = record_every
        self.curve_e, self.curve_b = [], []
        self._next_mark = 0

    def __call__(self, TH: np.ndarray) -> np.ndarray:
        TH = np.atleast_2d(TH)
        room = self.total - self.used
        if room <= 0:
            return np.full(len(TH), np.inf)
        if len(TH) > room:
            TH = TH[:room]
        vals = self.fn(TH)
        self.used += len(TH)
        i = int(np.argmin(vals))
        if vals[i] < self.best:
            self.best = float(vals[i])
            self.best_x = TH[i].copy()
        while self.used >= self._next_mark:
            self.curve_e.append(self._next_mark)
            self.curve_b.append(self.best)
            self._next_mark += self.record_every
        return vals

    @property
    def exhausted(self) -> bool:
        return self.used >= self.total


def _clip(X, lo, hi):
    return np.clip(X, lo, hi)


# ------------------------------------------------------------------- WHOA
def whoa(budget: Budget, dim: int, lo: float, hi: float, seed: int,
         pop: int = 50, ps: float = 0.2, pc: float = 0.13) -> None:
    """Wild Horse Optimizer.

    Population split into groups, each with a stallion; foals graze around their
    stallion, a fraction cross over between groups, stallions move towards the
    water hole, and a stallion is replaced when a group member beats it.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(lo, hi, size=(pop, dim))
    fit = budget(X)
    n_groups = max(2, int(round(pop * ps)))
    order = np.argsort(fit)
    stal = list(order[:n_groups])
    members = np.array_split(np.array(order[n_groups:]), n_groups)
    it, max_it = 0, 10 ** 9

    while not budget.exhausted:
        it += 1
        tdr = 1.0 - it * (1.0 / max(1, budget.total // pop))
        tdr = max(tdr, 0.0)
        gbest = budget.best_x.copy()
        newX = X.copy()
        for g in range(n_groups):
            s = stal[g]
            for m in members[g]:
                if rng.random() > pc:
                    z = rng.random(dim) < tdr
                    r = rng.random()
                    rr = np.where(z, 2.0 * r - 1.0, r)
                    newX[m] = 2.0 * rr * np.cos(2.0 * np.pi * rr) * (X[s] - X[m]) + X[s]
                else:  # crossover between two groups
                    gg = rng.integers(0, n_groups)
                    a, b = rng.choice(members[g]), rng.choice(members[gg])
                    cut = rng.integers(1, dim)
                    newX[m] = np.concatenate([X[a][:cut], X[b][cut:]])
            # stallion moves towards the water hole
            r3 = rng.random()
            if r3 > 0.5:
                newX[s] = 2.0 * rng.random(dim) * np.cos(2 * np.pi * rng.random()) \
                    * (gbest - X[s]) + gbest
            else:
                newX[s] = 2.0 * rng.random(dim) * np.cos(2 * np.pi * rng.random()) \
                    * (gbest - X[s]) - gbest
        newX = _clip(newX, lo, hi)
        newfit = budget(newX)
        if len(newfit) < len(newX):
            break
        better = newfit < fit
        X[better], fit[better] = newX[better], newfit[better]
        order = np.argsort(fit)
        stal = list(order[:n_groups])
        members = np.array_split(np.array(order[n_groups:]), n_groups)


# -------------------------------------------------------------------- GPC
def gpc(budget: Budget, dim: int, lo: float, hi: float, seed: int,
        pop: int = 50, g: float = 9.8, theta: float = np.pi / 4,
        mu: float = 1.0, sub: float = 0.5) -> None:
    """Giza Pyramids Construction.

    Each worker pushes a stone block up a ramp; the displacement of the block
    and the worker's own movement give the position update, and a substitution
    step exchanges components with the best worker.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(lo, hi, size=(pop, dim))
    fit = budget(X)
    v0 = 1.0
    while not budget.exhausted:
        best = budget.best_x.copy()
        d = (v0 ** 2) / (2.0 * g * (np.sin(theta) + mu * np.cos(theta)))
        x_wander = (2.0 * d) / (v0 * (np.cos(theta) + 1e-9))
        newX = X + rng.random((pop, dim)) * (d + x_wander) * (best - X) \
            + rng.normal(0, 0.1, size=(pop, dim))
        mask = rng.random((pop, dim)) < sub
        newX = np.where(mask, newX, X)
        newX = _clip(newX, lo, hi)
        newfit = budget(newX)
        if len(newfit) < len(newX):
            break
        better = newfit < fit
        X[better], fit[better] = newX[better], newfit[better]
        v0 = max(0.1, v0 * 0.999)


# --------------------------------------------------------- random search
def random_search(budget: Budget, dim: int, lo: float, hi: float, seed: int,
                  chunk: int = 200) -> None:
    rng = np.random.default_rng(seed)
    while not budget.exhausted:
        budget(rng.uniform(lo, hi, size=(chunk, dim)))


# ----------------------------------------------------------------- CMA-ES
def cmaes(budget: Budget, dim: int, lo: float, hi: float, seed: int,
          sigma0: float = 2.0) -> None:
    import cma
    es = cma.CMAEvolutionStrategy(
        np.zeros(dim), sigma0,
        {"bounds": [lo, hi], "seed": seed + 1, "verbose": -9, "popsize": 50})
    while not budget.exhausted and not es.stop():
        A = np.asarray(es.ask())
        v = budget(A)
        if len(v) < len(A):
            break
        es.tell(list(A), list(v))


# -------------------------------------------------------------------- TPE
def tpe(budget: Budget, dim: int, lo: float, hi: float, seed: int,
        n_startup: int = 200, gamma: float = 0.15, n_cand: int = 24,
        batch: int = 25, n_obs_max: int = 600) -> None:
    """Univariate tree-structured Parzen estimator.

    Observations are split at the gamma quantile into a good set l and a bad set
    g. For each dimension a Parzen mixture is built over the observations with
    the standard adaptive bandwidth, candidates are drawn from l, and the
    candidate maximising l(x)/g(x) is proposed. Candidates are proposed in
    batches so that the objective can be evaluated in one vectorised call, and
    the whole proposal step runs on the same device as the objective, over all
    dimensions at once. The most recent n_obs_max observations feed the density
    estimates, which bounds the per-proposal cost at a large budget; this is the
    same practical bound established implementations apply.
    """
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xo = rng.uniform(lo, hi, size=(n_startup, dim))
    Yo = budget(Xo)
    Xt = torch.tensor(Xo, dtype=torch.float32, device=dev)
    Yt = torch.tensor(Yo, dtype=torch.float32, device=dev)
    span = hi - lo
    LOG2PI = float(np.log(2 * np.pi))

    def parzen(V):                          # V: (dim, M) sorted along M
        Vs, _ = torch.sort(V, dim=1)
        M = Vs.shape[1]
        if M == 1:
            return Vs, torch.full_like(Vs, span * 0.5)
        d = Vs[:, 1:] - Vs[:, :-1]
        sig = torch.empty_like(Vs)
        sig[:, 1:-1] = torch.maximum(d[:, :-1], d[:, 1:])
        sig[:, 0] = d[:, 0]
        sig[:, -1] = d[:, -1]
        return Vs, sig.clamp(span / min(100.0, M + 1.0), span)

    def logmix(cand, mu, sig, chunk=32):    # cand (dim,K,C); mu/sig (dim,M)
        out = torch.empty(cand.shape[:3], device=dev)
        for a in range(0, cand.shape[0], chunk):
            b = min(a + chunk, cand.shape[0])
            z = (cand[a:b, :, :, None] - mu[a:b, None, None, :]) / sig[a:b, None, None, :]
            lp = -0.5 * z * z - torch.log(sig[a:b, None, None, :]) - 0.5 * LOG2PI
            out[a:b] = torch.logsumexp(lp, dim=-1) - float(np.log(mu.shape[1]))
        return out

    while not budget.exhausted:
        Xa, Ya = Xt[-n_obs_max:], Yt[-n_obs_max:]
        k = max(10, int(np.ceil(gamma * Xa.shape[0])))
        idx = torch.argsort(Ya)
        good = Xa[idx[:k]].T.contiguous()          # (dim, k)
        bad = Xa[idx[k:]].T.contiguous()
        if bad.shape[1] < 5:
            bad = Xa.T.contiguous()

        mg, sg = parzen(good)
        mb, sb = parzen(bad)
        pick = torch.randint(0, mg.shape[1], (dim, batch, n_cand), generator=gen, device=dev)
        base = torch.gather(mg[:, None, :].expand(dim, batch, mg.shape[1]), 2, pick)
        width = torch.gather(sg[:, None, :].expand(dim, batch, sg.shape[1]), 2, pick)
        cand = (base + width * torch.randn((dim, batch, n_cand), generator=gen,
                                           device=dev)).clamp(lo, hi)
        score = logmix(cand, mg, sg) - logmix(cand, mb, sb)
        best = score.argmax(dim=2, keepdim=True)
        prop = torch.gather(cand, 2, best).squeeze(2).T.contiguous()   # (batch, dim)

        arr = prop.cpu().numpy().astype(np.float64)
        v = budget(arr)
        if len(v) < len(arr):
            break
        Xt = torch.cat([Xt, prop], dim=0)
        Yt = torch.cat([Yt, torch.tensor(v, dtype=torch.float32, device=dev)], dim=0)


# --------------------------------------------------------- gradient descent
def gradient_descent(budget: Budget, dim: int, lo: float, hi: float, seed: int,
                     torch_obj=None, lr: float = 0.05) -> None:
    """Projected Adam on the analytic gradient of the same objective.

    One Adam step consumes one evaluation of the budget. A step also needs a
    backward pass, which the metaheuristics do not pay for; section 6.7 states
    this, and the wall-clock column of Table 7 makes the real cost visible.
    """
    import torch
    rng = np.random.default_rng(seed)
    th = torch.tensor(rng.uniform(lo, hi, size=dim), dtype=torch.float32,
                      device=torch_obj.device, requires_grad=True)
    opt = torch.optim.Adam([th], lr=lr)
    while not budget.exhausted:
        opt.zero_grad(set_to_none=True)
        loss = torch_obj.loss_torch(th.unsqueeze(0))[0]
        loss.backward()
        opt.step()
        with torch.no_grad():
            th.clamp_(lo, hi)
        budget(th.detach().cpu().numpy()[None, :])


OPTIMISERS = {
    "WHOA": whoa,
    "GPC": gpc,
    "Random search": random_search,
    "CMA-ES": cmaes,
    "TPE": tpe,
    "Gradient descent": gradient_descent,
}
METAPHOR = {"WHOA": True, "GPC": True, "Random search": False,
            "CMA-ES": False, "TPE": False, "Gradient descent": False}


def run_one(name: str, obj, dim: int, lo: float, hi: float, seed: int,
            total: int) -> RunResult:
    b = Budget(obj.loss_numpy, total)
    t0 = time.perf_counter()
    kw = {"torch_obj": obj} if name == "Gradient descent" else {}
    OPTIMISERS[name](b, dim, lo, hi, seed, **kw)
    wall = time.perf_counter() - t0
    return RunResult(name, seed, b.best, b.used, wall, b.curve_e, b.curve_b)
