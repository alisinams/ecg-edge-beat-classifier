"""Audit the BSPC manuscript text against what a reviewer or editor would query.

Checks, in order of how much damage each would do if it reached a referee:

  placeholders   unfilled [[AUTHOR ACTION]] markers left in the body
  crossrefs      every "Section n.m" points at a section that exists
  labels         every table and figure is both defined and cited
  numbers        figures quoted in more than one place agree with each other
  abbreviations  each abbreviation is expanded at first use
  register       informal or rhetorical phrasing that does not belong in an
                 indexed journal article
  self-citation  how much of the reference list is the authors' own work

Writes `text_audit.md` next to the manuscript.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

build = import_module("33_build_bspc")

OUT = build.OUT_DIR / "text_audit.md"

# Phrasing that reads as conversational or rhetorical rather than expository.
REGISTER = {
    r"\bstrapped to somebody new\b": "colloquial; consider \"first worn by a new user\"",
    r"\bdoes not flatter\b": "rhetorical",
    r"\bthe arm to beat\b": "colloquial",
    r"\bhonest state\b": "editorialising",
    r"\bwe would rather\b": "expresses preference rather than result",
    r"\bhalf the way the design expected\b": "conversational",
    r"\bis the point\b": "rhetorical",
    r"\bsays nothing about\b": "overstated; prefer \"is not informative about\"",
    r"\bstupid|silly|obvious(ly)?\b": "judgemental or filler",
    r"\bhuge|massive|tiny|dramatic(ally)?\b": "imprecise intensifier",
    r"\bof course\b": "filler",
    r"\bclearly\b": "asserts rather than shows",
    r"\bnovel\b": "claim of novelty is for the reader to draw",
    r"\bstate.of.the.art\b": "unqualified superlative",
    r"\bcannot be said\b": "hedged to the point of vagueness",
    r"\bdid not earn them\b": "colloquial",
    r"\bworth having\b": "colloquial",
    r"\bmust be worn\b|\bwhether a monitor is worn\b": "check phrasing",
    r"\bsomebody else\b": "colloquial; prefer \"another subject\"",
    r"\bat home\b": "colloquial in this context",
}

ABBREVIATIONS = {
    "AAMI": "Association for the Advancement of Medical Instrumentation",
    "ECG": "electrocardiogram",
    "INT8": None,          # conventional in the embedded literature
    "RAM": None,
    "SVDB": "MIT-BIH Supraventricular Arrhythmia Database",
    "INCART": None,
    "WHOA": "Wild Horse Optimizer",
    "GPC": "Giza Pyramids Construction",
    "CMA-ES": None,
    "MDCG": None,
    "SD": None,
    "FPGA": None,
    "ASIC": None,
    "ROM": None,
    "LUT": None,
    "DS1": None,
    "DS2": None,
}

OWN_WORK = ("Abbaszadeh", "Bazargani")


def flatten(blocks):
    """Return (section label, kind, text) for every text-bearing block."""
    section = None
    out = []
    for tag, payload in blocks:
        if tag in ("H1", "HU"):
            section = payload
        if tag == "H2":
            section = payload
        if tag == "TABLE":
            for row in payload:
                out.append((section, "table", " | ".join(row)))
        elif isinstance(payload, str):
            out.append((section, tag, payload))
    return out


def main() -> int:
    raw = build.SOURCE.read_text(encoding="utf-8")
    blocks = build.number_headings(build.parse_source(raw))
    items = flatten(blocks)
    body = "\n".join(t for _, kind, t in items if kind in ("P", "TABLECAP",
                                                           "FIGCAP", "table"))
    prose = "\n".join(t for _, kind, t in items if kind == "P")

    findings: dict[str, list[str]] = {}

    def add(key, msg):
        findings.setdefault(key, []).append(msg)

    # ---- section numbering ------------------------------------------
    numbered = {}
    for tag, payload in blocks:
        if tag in ("H1", "H2"):
            m = re.match(r"^(\d+(?:\.\d+)?)\s+(.*)$", payload)
            if m:
                numbered[m.group(1)] = m.group(2)
    for m in re.finditer(r"[Ss]ections?\s+(\d+(?:\.\d+)?)", body):
        ref = m.group(1)
        if ref not in numbered:
            ctx = body[max(0, m.start() - 70):m.end() + 40].replace("\n", " ")
            add("crossrefs", f"`Section {ref}` does not exist. Context: ...{ctx}...")

    # ---- placeholders ------------------------------------------------
    for m in re.finditer(r"\[\[AUTHOR ACTION[^\]]*\]\]", body):
        sec = next((s for s, _, t in items if m.group(0) in t), "?")
        add("placeholders", f"in **{sec}**: {m.group(0)[:110]}")

    # ---- tables and figures ------------------------------------------
    defined_tables = {int(re.match(r"Table (\d+)", p).group(1))
                      for tag, p in blocks if tag == "TABLECAP"
                      and re.match(r"Table (\d+)", p)}
    defined_figs = {int(re.match(r"Fig\. (\d+)", p).group(1))
                    for tag, p in blocks if tag == "FIGCAP"
                    and re.match(r"Fig\. (\d+)", p)}
    cited_tables = {int(n) for n in re.findall(r"Table (\d+)", prose)}
    cited_figs = {int(n) for n in re.findall(r"Fig\. (\d+)", prose)}
    for n in sorted(defined_tables - cited_tables):
        add("labels", f"Table {n} is captioned but never cited in the text")
    for n in sorted(cited_tables - defined_tables):
        add("labels", f"Table {n} is cited but has no caption")
    for n in sorted(defined_figs - cited_figs):
        add("labels", f"Fig. {n} is captioned but never cited in the text")
    for n in sorted(cited_figs - defined_figs):
        add("labels", f"Fig. {n} is cited but has no caption")
    if defined_tables and sorted(defined_tables) != list(
            range(1, max(defined_tables) + 1)):
        add("labels", f"table numbering is not contiguous: {sorted(defined_tables)}")
    if defined_figs and sorted(defined_figs) != list(range(1, max(defined_figs) + 1)):
        add("labels", f"figure numbering is not contiguous: {sorted(defined_figs)}")
    # cited in order of first appearance?
    first_seen = []
    for n in re.findall(r"Fig\. (\d+)", prose):
        if int(n) not in first_seen:
            first_seen.append(int(n))
    if first_seen != sorted(first_seen):
        add("labels", f"figures are first cited out of order: {first_seen}")

    # ---- numeric consistency -----------------------------------------
    KEY = {
        "student DS2 macro-F1": r"0\.338 \(0\.288 to 0\.459\)",
        "parameter count": r"4,516",
        "INT8 size": r"4\.5 kB",
        "peak SRAM": r"8\.5 kB",
        "mean protocol gain": r"0\.204",
        "gain ratio": r"5\.9",
        "arm A macro-F1": r"0\.408 \(0\.329 to 0\.578\)",
        "arm B macro-F1": r"0\.087",
        "distillation delta": r"-0\.005",
    }
    counts = {k: len(re.findall(v, body)) for k, v in KEY.items()}
    for k, v in counts.items():
        if v == 0:
            add("numbers", f"key figure missing from the text: {k}")
    # energy quoted two ways
    e3 = len(re.findall(r"111\.3 ", body))
    e34 = len(re.findall(r"111\.34", body))
    if e3 and e34:
        add("numbers", f"energy per inference appears as both 111.3 uJ ({e3}x) "
                       f"and 111.34 uJ ({e34}x); use one precision throughout")
    # latency
    if re.search(r"886 microseconds", body) and re.search(r"0\.886 ms", body):
        add("numbers", "latency appears as both 886 microseconds and 0.886 ms; "
                       "harmonise the unit")

    # ---- abbreviations -----------------------------------------------
    for abbr, expansion in ABBREVIATIONS.items():
        if expansion is None:
            continue
        if abbr not in body:
            continue
        first = body.index(abbr)
        window = body[max(0, first - 200):first + 200]
        if expansion.lower() not in window.lower():
            add("abbreviations",
                f"`{abbr}` is used before it is expanded as \"{expansion}\"")

    # ---- register -----------------------------------------------------
    for pattern, why in REGISTER.items():
        for m in re.finditer(pattern, prose, re.I):
            ctx = prose[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")
            add("register", f"\"{m.group(0)}\" ({why}): ...{ctx}...")

    # ---- self-citation -------------------------------------------------
    refs = build.load_references()
    order = build.collect_citation_order(blocks)
    own = [(n, refs[k]) for k, n in order.items()
           if any(a in refs[k].split("(")[0] for a in OWN_WORK)]
    share = 100.0 * len(own) / len(order)
    add("selfcite", f"{len(own)} of {len(order)} references "
                    f"({share:.1f} %) are the authors' own work")
    for n, e in sorted(own):
        add("selfcite", f"  [{n}] {e[:100]}")

    # ---- report --------------------------------------------------------
    L = ["# Manuscript text audit", "",
         "Checks applied to `manuscript_source.txt`, from which "
         "`02_Manuscript.docx` is built.", ""]
    order_of_sections = [
        ("placeholders", "Unfilled placeholders in the manuscript body",
         "These would reach a referee verbatim. Every one must be filled or the "
         "sentence removed."),
        ("crossrefs", "Broken section cross-references",
         "A reference to a section number that does not exist."),
        ("labels", "Table and figure labelling",
         "Each table and figure must be defined once, cited at least once, and "
         "numbered in order of first citation."),
        ("numbers", "Numerical consistency",
         "Values that appear more than once must agree, and units must be used "
         "consistently."),
        ("abbreviations", "Abbreviations",
         "Each non-standard abbreviation must be expanded at first use."),
        ("register", "Register and phrasing",
         "Wording that reads as conversational, rhetorical or judgemental rather "
         "than expository. None of these is an error of fact; they are the "
         "sentences a copy editor is most likely to query."),
        ("selfcite", "Self-citation",
         "Reported so the proportion is a deliberate choice rather than an "
         "accident."),
    ]
    total = 0
    for key, title, note in order_of_sections:
        got = findings.get(key, [])
        if key != "selfcite":
            total += len(got)
        L += [f"## {title}", "", note, ""]
        if not got:
            L += ["No findings.", ""]
        else:
            L += [f"- {g}" for g in got] + [""]
    L.insert(4, f"**{total} findings** outside the self-citation count.\n")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")

    for key, title, _ in order_of_sections:
        print(f"  {len(findings.get(key, [])):>3}  {title}")
    print(f"\nwrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
