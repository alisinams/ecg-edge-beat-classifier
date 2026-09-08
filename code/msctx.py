"""Context shared by every manuscript section builder.

One dictionary, loaded once from results/numbers.json. Every section reads its
numbers from here, so a quantity that appears in the abstract, a table, a figure
caption and the conclusion is literally the same Python value in all four.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

from common import RESULTS, jload

CLS = ["N", "S", "V", "F", "Q"]
ARMS = ["A", "B", "B2", "C", "D", "E", "F"]
ARM_LABEL = {
    "A": "A, best classical", "B": "B, scorer with fuzzy decoder",
    "B2": "B2, clinical rule base", "C": "C, scorer with argmax decoder",
    "D": "D, separable CNN", "E": "E, student classifier",
    "F": "F, temporal-convolution student"}
CLASS_NAME = {"N": "normal", "S": "supraventricular ectopic",
              "V": "ventricular ectopic", "F": "fusion",
              "Q": "unclassifiable or paced"}


def context() -> SimpleNamespace:
    N = jload(RESULTS / "numbers.json")
    C = SimpleNamespace(
        N=N,
        T1=N.get("table1_verified", {}),
        t2=N["table2"], t3=N["table3"], t4=N["table4"], t5=N["table5"],
        t6=N.get("table6", {}), t7=N.get("table7", {}), t8=N.get("table8", {}),
        t9=N.get("table9", {}), s41=N.get("table_s41", {}),
        prot=N["protocol"], ext=N["external"],
        cal=N.get("calibration", {}), abst=N.get("abstention", {}),
        noise=N.get("noise_sweep", {}), sqi=N.get("sqi", {}),
        alarms=N.get("alarms", {}), b2=N.get("b2_rules", {}),
        hp=N.get("hyperparams", {}), perf=N.get("train_performance_ds1", {}),
        teacher=N.get("teacher", {}), bw=N.get("bandwidth_ablation", {}),
        splits=N["splits"],
        CLS=CLS, ARMS=ARMS, ARM_LABEL=ARM_LABEL, CLASS_NAME=CLASS_NAME,
    )
    C.E = C.t3["E"]
    C.Emac = C.E["macro"]
    C.Eagg = C.E["aggregate"]
    C.lat = C.t8.get("latency_us", {})
    C.dev = C.t8.get("device", {})
    C.assump = C.t8.get("assumptions", {})
    C.opt_setup = C.t7.get("setup", {})
    C.tr_row = C.t2["DS1 training subset"]
    C.va_row = C.t2["DS1 validation subset"]
    C.ds2_row = C.t2["DS2 test"]
    return C


def noise_at(C, kind, snr):
    for r in C.noise.get("rows", []):
        if r["noise_type"] == kind and r["snr_db"] == snr:
            return r["macro_f1"]
    return float("nan")


def sqi_rate(C, ds):
    for r in C.sqi.get("rows", []):
        if r["dataset"] == ds:
            return r["rate"]
    return float("nan")


def best_arm(C, key="macro"):
    return max(C.ARMS, key=lambda a: C.t3[a][key]["f1"][0])


def rank_of(C, arm):
    order = sorted(C.ARMS, key=lambda a: -C.t3[a]["macro"]["f1"][0])
    return order.index(arm) + 1
