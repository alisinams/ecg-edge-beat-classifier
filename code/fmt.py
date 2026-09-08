"""Formatting helpers shared by the manuscript and supplement builders.

One rounding rule per quantity, applied everywhere, so a number cannot appear
with two different roundings in two places.
"""
from __future__ import annotations

import math


def pct(t, nd=1):
    """A (point, lo, hi) triple as 'xx.x (yy.y to zz.z)' in per cent."""
    p, lo, hi = t
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "not estimable"
    return f"{100*p:.{nd}f} ({100*lo:.{nd}f} to {100*hi:.{nd}f})"


def pct1(v, nd=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "not estimable"
    return f"{100*v:.{nd}f}"


def f1(t, nd=3):
    p, lo, hi = t
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "not estimable"
    return f"{p:.{nd}f} ({lo:.{nd}f} to {hi:.{nd}f})"


def num(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "not estimable"
    return f"{v:.{nd}f}"


def signed(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "not estimable"
    return f"{v:+.{nd}f}"


def thousands(n):
    return f"{int(round(n)):,}"


def p_value(p):
    if p is None:
        return "reference"
    if p < 1e-4:
        return "< 0.0001"
    if p < 0.001:
        return f"{p:.5f}"
    return f"{p:.4f}"


def words(n: int) -> str:
    w = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve"]
    return w[n] if 0 <= n < len(w) else str(n)
