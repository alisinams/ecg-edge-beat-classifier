"""Loading, partitioning and class weighting.

One place decides what "the DS1 training subset" means, so every arm sees the
same beats in the same order.
"""
from __future__ import annotations

import numpy as np

from common import CACHE, CLASSES, DS1, DS1_TRAIN, DS1_VAL, DS2


class Split:
    __slots__ = ("X", "y", "R", "rec", "bidx", "name")

    def __init__(self, X, y, R, rec, bidx, name):
        self.X, self.y, self.R, self.rec, self.bidx, self.name = X, y, R, rec, bidx, name

    def __len__(self):
        return len(self.y)

    def __repr__(self):
        return f"<Split {self.name} n={len(self.y)} recs={len(np.unique(self.rec))}>"


def _load(npz_name: str):
    z = np.load(CACHE / npz_name, allow_pickle=True)
    return z


def subset(z, records, name) -> Split:
    rec = z["rec"].astype(str)
    m = np.isin(rec, records)
    return Split(z["X"][m], z["y"][m], z["R"][m], rec[m], z["bidx"][m], name)


def mitdb_splits(npz_name: str = "mitdb.npz"):
    z = _load(npz_name)
    return (subset(z, DS1_TRAIN, "DS1 train"),
            subset(z, DS1_VAL, "DS1 val"),
            subset(z, DS2, "DS2"))


def mitdb_meta():
    z = _load("mitdb.npz")
    return {r: (s, a) for r, s, a in
            zip(z["meta_rec"].astype(str), z["meta_sex"].astype(str), z["meta_age"])}


def external(db: str) -> Split:
    z = _load(f"{db}.npz")
    rec = z["rec"].astype(str)
    return Split(z["X"], z["y"], z["R"], rec, z["bidx"], db)


def mixed_partition(seed: int = 0):
    """The patient-mixed contrast of section 6.1.

    All beats from all 44 records pooled and split at random in the DS1 to DS2
    proportion, so a subject appears on both sides. Used for Table 4 only.
    """
    z = _load("mitdb.npz")
    rec = z["rec"].astype(str)
    n = len(z["y"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train_all = int(round(n * (50790 / 100268)))     # the DS1 share of all beats
    n_val = int(round(n_train_all * (11528 / 50790)))  # the same validation share
    tr = perm[: n_train_all - n_val]
    va = perm[n_train_all - n_val: n_train_all]
    te = perm[n_train_all:]

    def take(idx, name):
        return Split(z["X"][idx], z["y"][idx], z["R"][idx], rec[idx], z["bidx"][idx], name)

    return take(tr, "mixed train"), take(va, "mixed val"), take(te, "mixed test")


WEIGHT_CAP = 50.0


def class_weights(y: np.ndarray, n_class: int = 5, cap: float = WEIGHT_CAP) -> np.ndarray:
    """Inverse class frequency, capped, normalised to sum to the number of classes.

    Computed from the training subset alone, so nothing from validation or DS2
    enters them. Classes with no beats get weight zero.

    The cap is at `cap` times the weight of the most frequent class and it is
    not cosmetic: the Q superclass holds four beats in the DS1 training subset,
    so an uncapped inverse-frequency weight would make those four beats count
    for more than the thirty-five thousand normal beats put together, and the
    network would spend its capacity on a class that cannot be estimated.
    """
    cnt = np.array([(y == c).sum() for c in range(n_class)], dtype=np.float64)
    present = cnt > 0
    w = np.zeros(n_class, dtype=np.float64)
    if not present.any():
        return w
    n_max = cnt[present].max()
    w[present] = np.minimum(n_max / cnt[present], cap)
    w[present] = w[present] / w[present].sum() * present.sum()
    return w
