"""Apply the fixes the reference and text audits found.

Three cited preprints have since appeared as peer-reviewed articles, and the
journal's guide requires the formal publication to be cited instead. Their
entries are rewritten in `13_Reference_List.md`. The manuscript source is then
corrected for an uncited figure, two inconsistent numeric formats, two
abbreviations used before they are expanded, and eight phrases that read as
conversational rather than expository.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFS = ROOT / "13_Reference_List.md"
SRC = ROOT / "Biomedical Signal Processing and Control" / "manuscript_source.txt"

# ---------------------------------------------------------------- references
# Each verified at Crossref by DOI on the date of this run.
REF_REPLACEMENTS = [
    # arXiv 2503.07276 -> Research on Biomedical Engineering 42(3), 2026
    ("[#8]",
     "Silva G.A.L., Silva P.H.L., Moreira G.J.P., Freitas V.L.S., Gertrudes J.C., "
     "Luz E.J.S. (2026) A systematic review of ECG arrhythmia classification: "
     "embedded feasibility, adherence to standards, and fair evaluation. "
     "Research on Biomedical Engineering, 42(3). "
     "https://doi.org/10.1007/s42600-026-00485-z  [#8]"),
    # arXiv 2408.05178 -> JAMIA Open 8(5), 2025
    ("[#14]",
     "McKeen K., Masood S., Toma A., Rubin B., Wang B. (2025) ECG-FM: an open "
     "electrocardiogram foundation model. JAMIA Open, 8(5). "
     "https://doi.org/10.1093/jamiaopen/ooaf122  [#14]"),
    # arXiv 2107.07511 -> Foundations and Trends in Machine Learning 16(4), 2023
    ("[#18]",
     "Angelopoulos A.N., Bates S. (2023) Conformal Prediction: A Gentle "
     "Introduction. Foundations and Trends in Machine Learning, 16(4), 494-591. "
     "https://doi.org/10.1561/2200000101  [#18]"),
]

# ------------------------------------------------------------------ the text
TEXT_FIXES = [
    # -- Fig. 1 was captioned but never cited --------------------------
    ("Beats were then extracted as windows centred on the supplied R-peak "
     "annotation",
     "Fig. 1 shows the whole processing chain. Beats were then extracted as "
     "windows centred on the supplied R-peak annotation"),

    # -- abbreviations expanded at first use ---------------------------
    ("changing by +0.005 on INCART and -0.005 on SVDB.",
     "changing by +0.005 on INCART and -0.005 on the supraventricular database."),
    ("uses the Wild Horse Optimizer {#60} and Giza Pyramids Construction {#61}",
     "uses the Wild Horse Optimizer {#60} and Giza Pyramids Construction (GPC) "
     "{#61}"),

    # -- one precision for the energy, one unit for the latency --------
    ("is computed to need 111.3 microjoules per inference at a median 886 "
     "microseconds on an STM32F446",
     "is computed to need 111.3 microjoules per inference at a median latency of "
     "0.886 ms on an STM32F446"),
    ("at a computed 111.34 microjoules per inference on named silicon",
     "at a computed 111.3 microjoules per inference on named silicon"),
    ("a beat is computed to classify in 0.886 ms at 111.34 microjoules on an "
     "STM32F446",
     "a beat is computed to classify in 0.886 ms at 111.3 microjoules on an "
     "STM32F446"),
    ("Energy per inference|111.34 uJ|computed from Eq. (4)",
     "Energy per inference|111.3 uJ|computed from Eq. (4)"),

    # -- register ------------------------------------------------------
    ("is precisely what a device does not have when first strapped to somebody "
     "new, so the inflation lands exactly on the use case",
     "is precisely what a device does not have when first worn by a new user, so "
     "the inflation falls on the intended use case"),
    ("the timing baseline belongs to somebody else",
     "the timing baseline belongs to another subject"),
    ("gradient-boosted trees on hand-built descriptors are not a weak baseline "
     "to be cleared but the arm to beat; that arm also does not fit the device",
     "gradient-boosted trees on hand-built descriptors are not a weak baseline "
     "but the strongest comparator in this set; that arm also does not fit the "
     "device"),
    ("The protocol contrast came out half the way the design expected.",
     "The protocol contrast matched the design's expectation only in part."),
    ("the frontier it draws is indicative; the region it leaves almost empty is "
     "the point, since no study in this set",
     "the frontier it draws is indicative. The sparsity of the region it leaves "
     "is the relevant observation, since no study in this set"),
    ("a second stage costing parameters, memory and an optimiser run did not "
     "earn them",
     "a second stage costing parameters, memory and an optimiser run is not "
     "justified by these results"),
    ("so the rule is worth having because it bounds what the device claims to "
     "know, not because it improves the numbers",
     "so the rule is retained because it bounds what the device claims to know "
     "rather than because it improves the reported metrics"),
    ("Against the inter-patient group the comparison does not flatter this work.",
     "Against the inter-patient group the comparison is unfavourable to this "
     "work."),
    ("far above what a consumer device could ship with. That the supraventricular",
     "far above what an unattended consumer device could sustain. That the "
     "supraventricular"),
]


def main() -> int:
    text = REFS.read_text(encoding="utf-8")
    for key, entry in REF_REPLACEMENTS:
        pattern = re.compile(r"^.*" + re.escape(key) + r"\s*$", re.M)
        hits = pattern.findall(text)
        assert len(hits) == 1, f"{key} matched {len(hits)} reference lines"
        text = pattern.sub(lambda _: entry, text)
        print(f"reference {key} replaced with the published version")
    REFS.write_text(text, encoding="utf-8")  # idempotent: rerun rewrites the same lines

    src = SRC.read_text(encoding="utf-8")
    for old, new in TEXT_FIXES:
        assert src.count(old) == 1, f"matched {src.count(old)}: {old[:60]!r}"
        src = src.replace(old, new)
        print(f"text: {old[:62]}")
    SRC.write_text(src, encoding="utf-8")
    for bad in ("—", "–"):
        assert bad not in src, "dash character present"
    print(f"\n{len(REF_REPLACEMENTS)} references and {len(TEXT_FIXES)} passages updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
