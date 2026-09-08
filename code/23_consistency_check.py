"""Cross-check that a quantity carries the same value everywhere it appears.

The builders read every number from results/numbers.json, so agreement should be
structural. This script verifies it anyway, by finding each quantity in the
built text and comparing the strings, because a formatting slip in one place is
exactly the failure the single-source design is meant to prevent and the only
way to know it worked is to look at the output.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RESULTS, jload

MS = ROOT / "05_Manuscript.md"
SUP = ROOT / "06_Supplementary.md"
GUIDE = ROOT / "09_Figures_Guide.md"
FA = ROOT / "12_Persian_Report.md"


def occurrences(text: str, needle: str) -> int:
    return text.count(needle)


def main() -> int:
    N = jload(RESULTS / "numbers.json")
    ms = MS.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8") if GUIDE.exists() else ""
    fa = FA.read_text(encoding="utf-8") if FA.exists() else ""

    E = N["table3"]["E"]
    t8 = N.get("table8", {})
    checks, failures = [], 0

    def check(name, needle, where, minimum=2):
        nonlocal failures
        n = occurrences(where, needle)
        ok = n >= minimum
        checks.append((name, needle, n, minimum, ok))
        if not ok:
            failures += 1

    # the DS2 macro-F1 of the student classifier: abstract, Table 3, section 9
    p, lo, hi = E["macro"]["f1"]
    check("DS2 macro-F1 of the student classifier",
          f"{p:.3f} ({lo:.3f} to {hi:.3f})", ms, minimum=3)

    # INT8 size: abstract, Table 1, Table 8, section 8.1, section 9
    if t8:
        check("INT8 model size", f"{t8['size_int8_kb']:.1f} kB", ms, minimum=4)
        check("peak SRAM", f"{t8['sram_kb']:.1f} kB", ms, minimum=3)
        check("parameter count", f"{int(round(t8['params'])):,}", ms, minimum=3)

    # the protocol contrast
    check("mean macro-F1 gain under patient mixing",
          f"{N['protocol']['mean_gain']:.3f}", ms, minimum=3)

    # the external drop
    check("mean external macro-F1 drop",
          f"{N['external']['drop_mean']:.3f}", ms, minimum=2)

    # the figure guide must quote the same electrode-motion value as the text
    ns = N.get("noise_sweep", {})
    em6 = next((r["macro_f1"] for r in ns.get("rows", [])
                if r["noise_type"] == "electrode motion" and r["snr_db"] == 6), None)
    if em6 is not None and guide:
        both = f"{em6:.3f}"
        n_ms, n_g = ms.count(both), guide.count(both)
        ok = n_ms >= 1 and n_g >= 1
        checks.append(("electrode motion at 6 dB, text and figure guide",
                       both, f"{n_ms} in manuscript, {n_g} in guide", "1 each", ok))
        if not ok:
            failures += 1

    # the Persian report must carry the same headline macro-F1
    if fa:
        n_fa = fa.count(f"{p:.3f}")
        ok = n_fa >= 1
        checks.append(("DS2 macro-F1 in the Persian report", f"{p:.3f}",
                       n_fa, 1, ok))
        if not ok:
            failures += 1

    for name, needle, n, need, ok in checks:
        mark = "ok  " if ok else "FAIL"
        print(f"  {mark} {name}: '{needle}' appears {n} times, need {need}")

    print("CONSISTENCY_OK" if failures == 0 else f"CONSISTENCY_FAILURES {failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
