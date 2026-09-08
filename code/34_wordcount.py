"""Word count for the BSPC manuscript source, split narrative from tables.

BSPC asks that a full paper be "normally about 5,000 words", so this reports
the count the way an editor would read it: main text from Introduction to
Conclusion, with the table contents separated out.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

build = import_module("33_build_bspc")

CORE = ["Introduction", "Related work", "Methods",
        "Results", "Discussion", "Conclusion"]


def main() -> int:
    blocks = build.number_headings(
        build.parse_source(build.SOURCE.read_text(encoding="utf-8")))
    section = None
    narrative: dict[str, int] = {}
    tables: dict[str, int] = {}
    for tag, payload in blocks:
        if tag in ("H1", "HU"):
            section = re.sub(r"^\d+\.\s*", "", payload)
        if tag == "TABLE":
            words = sum(len(c.split()) for row in payload for c in row)
            tables[section] = tables.get(section, 0) + words
            continue
        text = re.sub(r"\{#[^}]*\}", "", re.sub(r"</?i>", "", str(payload)))
        words = len(text.split())
        if tag == "P":
            narrative[section] = narrative.get(section, 0) + words
        elif tag == "TABLECAP":
            tables[section] = tables.get(section, 0) + words

    for k in CORE:
        print(f"  {narrative.get(k, 0):5d} narrative + {tables.get(k, 0):4d} tables   {k}")
    n = sum(narrative.get(k, 0) for k in CORE)
    t = sum(tables.get(k, 0) for k in CORE)
    print(f"\n  narrative, Introduction to Conclusion : {n}")
    print(f"  tables and their captions             : {t}")
    print(f"  main text total                       : {n + t}")
    print(f"  abstract                              : {narrative.get('Abstract', 0)}"
          f"   (BSPC limit 250)")
    back = sum(v for k, v in narrative.items() if k not in CORE and k != "Abstract")
    print(f"  declarations and back matter          : {back}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
