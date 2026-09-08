"""The seven arms.

A   classical baselines on rhythm descriptors plus 32 morphology samples
B   polynomial-sigmoid scorer (stage one) followed by a Takagi-Sugeno fuzzy decoder
C   the identical stage one followed by argmax
B2  the same fuzzy formalism rebuilt over four named clinical quantities
D   compact depthwise-separable 1-D convolutional network
E   the student classifier, D's topology at reduced width, distilled
F   temporal-convolution student of matched parameter count

B and C share stage one weight for weight by construction: arm C is built from a
fitted arm B object, so the ablation isolates the decoder.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- features
MORPH_N = 32


def morphology(X: np.ndarray, n: int = MORPH_N) -> np.ndarray:
    """Uniformly subsample the beat window down to `n` points."""
    idx = np.linspace(0, X.shape[1] - 1, n).round().astype(int)
    return X[:, idx]


def classical_features(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.concatenate([R, morphology(X)], axis=1).astype(np.float32)


def clinical_features(X: np.ndarray, R: np.ndarray, fs: int = 250,
                      pre_n: int = 62) -> np.ndarray:
    """Four named clinical quantities used by arm B2.

    rr_irregularity  |RR_prev - RR_next| / mean(RR_prev, RR_next)
    rr_ratio         RR_prev / mean of the preceding ten RR intervals
    p_presence       energy in the 150 ms before QRS onset, over the local baseline
    qrs_duration     width in ms of the contiguous high-slope region around R
    """
    rr_prev, rr_next, ratio_local = R[:, 0], R[:, 1], R[:, 2]
    irregular = np.abs(rr_prev - rr_next) / np.maximum(0.5 * (rr_prev + rr_next), 1e-6)

    # QRS onset taken at 50 ms before the R peak; P window is the 150 ms before that
    onset = pre_n - int(round(0.050 * fs))
    p_lo = max(0, onset - int(round(0.150 * fs)))
    p_win = X[:, p_lo:onset]
    base_win = X[:, 0:max(1, p_lo)]
    p_energy = np.mean(p_win ** 2, axis=1)
    base_energy = np.mean(base_win ** 2, axis=1) + 1e-6
    p_presence = p_energy / base_energy

    # QRS duration from the slope envelope around the R peak
    d = np.abs(np.diff(X, axis=1))
    lo, hi = max(0, pre_n - 30), min(d.shape[1], pre_n + 30)
    seg = d[:, lo:hi]
    thr = 0.25 * seg.max(axis=1, keepdims=True) + 1e-9
    qrs_dur = (seg > thr).sum(axis=1) * 1000.0 / fs

    out = np.stack([irregular, ratio_local, p_presence, qrs_dur], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ------------------------------------------------------------------- arm B
class PolySigmoidScorer:
    """Stage one of arm B, equations (1) and (2) of the manuscript.

    s_c(x) = sum_j ( w3_cj x_j^3 + w2_cj x_j^2 + w1_cj x_j )
    z_c(x) = sigmoid( a_c ( s_c(x) - b_c ) )

    With D selected features and C classes the parameter vector holds
    C * (3D + 2) values, 205 for D = 13 and C = 5.
    """

    def __init__(self, n_feat: int = 13, n_class: int = 5):
        self.D = n_feat
        self.C = n_class
        self.n_params = n_class * (3 * n_feat + 2)
        self.theta = np.zeros(self.n_params, dtype=np.float64)
        self.sel: np.ndarray | None = None   # indices into the full feature pool

    # -- packing ---------------------------------------------------------
    def unpack(self, theta: np.ndarray):
        C, D = self.C, self.D
        k = C * D
        w3 = theta[0:k].reshape(C, D)
        w2 = theta[k:2 * k].reshape(C, D)
        w1 = theta[2 * k:3 * k].reshape(C, D)
        a = theta[3 * k:3 * k + C]
        b = theta[3 * k + C:3 * k + 2 * C]
        return w3, w2, w1, a, b

    def scores(self, Z: np.ndarray, theta: np.ndarray | None = None) -> np.ndarray:
        """Z is already the selected-feature matrix, shape (n, D)."""
        th = self.theta if theta is None else theta
        w3, w2, w1, a, b = self.unpack(th)
        Z1 = Z
        Z2 = Z * Z
        Z3 = Z2 * Z
        s = Z3 @ w3.T + Z2 @ w2.T + Z1 @ w1.T
        # a_c is squashed to stay positive and bounded, which keeps the objective
        # well conditioned for every optimiser in the comparison
        aa = 0.1 + 4.9 / (1.0 + np.exp(-a))
        return 1.0 / (1.0 + np.exp(-np.clip(aa * (s - b), -60, 60)))

    def select(self, Xfull: np.ndarray) -> np.ndarray:
        return Xfull[:, self.sel]


def ts_fuzzy_decode(z: np.ndarray, sigma_low: float = 0.25,
                    sigma_high: float = 0.25) -> np.ndarray:
    """Stage two of arm B: a Takagi-Sugeno system on the five class scores.

    Two Gaussian membership functions per input, low centred at 0 and high at 1.
    Rule c fires when input c is high and every other input is low; its
    normalised firing strength is the score for class c.
    """
    mu_high = np.exp(-0.5 * ((z - 1.0) / sigma_high) ** 2)
    mu_low = np.exp(-0.5 * ((z - 0.0) / sigma_low) ** 2)
    log_low = np.log(mu_low + 1e-300)
    total_low = log_low.sum(axis=1, keepdims=True)
    # product over j != c of mu_low, times mu_high for c
    log_w = np.log(mu_high + 1e-300) + (total_low - log_low)
    log_w = log_w - log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w)
    return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)


# ------------------------------------------------------------------ arm B2
CLINICAL_RULES = [
    # (antecedent as {input index: 'low'|'high'}, consequent class index)
    ({1: "mid", 2: "high", 3: "low"}, 0),   # R1 -> N
    ({1: "low", 2: "low", 3: "low"}, 1),    # R2 -> S
    ({1: "low", 3: "high", 2: "low"}, 2),   # R3 -> V
    ({0: "high", 2: "low"}, 1),             # R4 -> S
    ({3: "high", 1: "mid"}, 2),             # R5 -> V
    ({3: "mid", 2: "high"}, 3),             # R6 -> F
    ({0: "low", 1: "low", 2: "low", 3: "low"}, 4),   # R7 -> Q
    ({0: "high", 3: "high"}, 2),            # R8 -> V, added by the optimiser
]
CLINICAL_INPUTS = ["RR-interval irregularity", "RR ratio to local mean",
                   "P-wave presence", "QRS duration"]


class ClinicalFuzzy:
    """Arm B2. Four inputs, three membership functions each (low, mid, high).

    Parameters fitted per input: the low mean, the high mean and one width per
    membership function. mid is placed midway between low and high.
    """

    n_inputs = 4
    n_class = 5

    def __init__(self):
        # per input: [m_low, m_high, s_low, s_mid, s_high]
        self.n_params = self.n_inputs * 5 + len(CLINICAL_RULES)
        self.theta = np.zeros(self.n_params)

    # Table S3.3 fixes the membership width search space at 0.05 to 1.0, so the
    # raw parameter is squashed into that interval rather than clipped.
    W_LO, W_HI = 0.05, 1.0

    @classmethod
    def _width(cls, raw):
        return cls.W_LO + (cls.W_HI - cls.W_LO) / (1.0 + np.exp(-np.clip(raw, -30, 30)))

    def memberships(self, Xc: np.ndarray, theta: np.ndarray):
        p = theta[: self.n_inputs * 5].reshape(self.n_inputs, 5)
        mus = {}
        for j in range(self.n_inputs):
            m_lo, m_hi, s_lo, s_mid, s_hi = p[j]
            s_lo, s_mid, s_hi = (self._width(s_lo), self._width(s_mid), self._width(s_hi))
            m_mid = 0.5 * (m_lo + m_hi)
            x = Xc[:, j]
            mus[(j, "low")] = np.exp(-0.5 * ((x - m_lo) / s_lo) ** 2)
            mus[(j, "mid")] = np.exp(-0.5 * ((x - m_mid) / s_mid) ** 2)
            mus[(j, "high")] = np.exp(-0.5 * ((x - m_hi) / s_hi) ** 2)
        return mus

    def firing(self, Xc: np.ndarray, theta: np.ndarray | None = None) -> np.ndarray:
        th = self.theta if theta is None else theta
        mus = self.memberships(Xc, th)
        w = np.ones((Xc.shape[0], len(CLINICAL_RULES)))
        for r, (ante, _) in enumerate(CLINICAL_RULES):
            acc = np.ones(Xc.shape[0])
            for j, lvl in ante.items():
                acc *= mus[(j, lvl)]
            w[:, r] = acc
        rw = np.abs(th[self.n_inputs * 5:]) + 1e-6
        return w * rw[None, :]

    def predict_proba(self, Xc: np.ndarray, theta: np.ndarray | None = None) -> np.ndarray:
        w = self.firing(Xc, theta)
        out = np.zeros((Xc.shape[0], self.n_class))
        for r, (_, cls) in enumerate(CLINICAL_RULES):
            out[:, cls] += w[:, r]
        s = out.sum(axis=1, keepdims=True)
        return out / np.maximum(s, 1e-12)


# ------------------------------------------------------- neural arms D/E/F
class SeparableBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 9, stride: int = 2):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, k, stride=stride, padding=k // 2, groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


class SepCNN(nn.Module):
    """Arms D and E. `width` scales every channel count."""

    def __init__(self, n_class: int = 5, width: float = 1.0, blocks: int = 4,
                 n_rhythm: int = 4, base: int = 32, k: int = 9):
        super().__init__()
        chans = [max(4, int(round(base * width * (1.5 ** i)))) for i in range(blocks)]
        self.stem = nn.Sequential(
            nn.Conv1d(1, chans[0], k, stride=2, padding=k // 2, bias=False),
            nn.BatchNorm1d(chans[0]), nn.ReLU())
        blks = []
        for i in range(1, blocks):
            blks.append(SeparableBlock(chans[i - 1], chans[i], k=k, stride=2))
        self.blocks = nn.Sequential(*blks)
        self.head = nn.Linear(chans[-1] + n_rhythm, n_class)
        self.out_ch = chans[-1]

    def features(self, x, r):
        h = self.blocks(self.stem(x))
        h = h.mean(dim=2)
        return torch.cat([h, r], dim=1)

    def forward(self, x, r):
        return self.head(self.features(x, r))


class TCNBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, dil: int):
        super().__init__()
        pad = (k - 1) * dil
        self.conv = nn.Conv1d(cin, cout, k, padding=pad, dilation=dil, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.pad = pad
        self.res = nn.Conv1d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.conv(x)
        if self.pad:
            h = h[:, :, :-self.pad]
        return F.relu(self.bn(h) + self.res(x))


class TCN(nn.Module):
    """Arm F, a dilated temporal-convolution student."""

    def __init__(self, n_class: int = 5, width: float = 1.0, blocks: int = 4,
                 n_rhythm: int = 4, base: int = 24, k: int = 5):
        super().__init__()
        c = max(4, int(round(base * width)))
        layers, cin = [], 1
        for i in range(blocks):
            layers.append(TCNBlock(cin, c, k, dil=2 ** i))
            cin = c
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(c + n_rhythm, n_class)

    def features(self, x, r):
        h = self.net(x).mean(dim=2)
        return torch.cat([h, r], dim=1)

    def forward(self, x, r):
        return self.head(self.features(x, r))


def build_neural(arm: str, hp: dict, n_class: int = 5) -> nn.Module:
    if arm == "F":
        return TCN(n_class=n_class, width=hp.get("width", 1.0),
                   blocks=hp.get("blocks", 4), base=hp.get("base", 24),
                   k=hp.get("kernel", 5))
    return SepCNN(n_class=n_class, width=hp.get("width", 1.0),
                  blocks=hp.get("blocks", 4), base=hp.get("base", 32),
                  k=hp.get("kernel", 9))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def count_macs(m: nn.Module, win: int, n_rhythm: int = 4) -> int:
    """Multiply-accumulate operations for one beat, counted by hooking the
    convolution and linear layers and multiplying by their real output length."""
    macs = {"n": 0}
    hooks = []

    def hk(mod, inp, out):
        if isinstance(mod, nn.Conv1d):
            cin = mod.in_channels // mod.groups
            macs["n"] += cin * mod.out_channels * mod.kernel_size[0] * out.shape[-1]
        elif isinstance(mod, nn.Linear):
            macs["n"] += mod.in_features * mod.out_features

    for mod in m.modules():
        if isinstance(mod, (nn.Conv1d, nn.Linear)):
            hooks.append(mod.register_forward_hook(hk))
    m.eval()
    with torch.no_grad():
        m(torch.zeros(1, 1, win), torch.zeros(1, n_rhythm))
    for h in hooks:
        h.remove()
    return macs["n"]
