"""Assemble the final manuscript from results/numbers.json.

Every quantity in the text, in a table, in a caption and in the abstract is read
from the same dictionary entry, so the copies of a number are the same number by
construction. Nothing here recomputes a metric.

Citations stay in the EndNote temporary form {Author, Year #RecordNumber} so
that the accompanying references.ris library can be attached in Word and the
bibliography formatted in the journal's style.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT
from msctx import context

import ms_p1
import ms_p2
import ms_p3

OUT = ROOT / "05_Manuscript.md"


def main() -> int:
    C = context()
    doc = ms_p1.build(C) + ms_p2.build(C) + ms_p3.build(C)
    doc = re.sub(r"\n{3,}", "\n\n", doc)
    OUT.write_text(doc, encoding="utf-8")

    leftovers = re.findall(r"\[\[(?!AUTHOR ACTION)[^\]]*\]\]", doc)
    words = len(re.findall(r"\b[\w'-]+\b", doc))
    author_actions = doc.count("[[AUTHOR ACTION")
    print(f"MANUSCRIPT_WRITTEN  {len(doc)} characters, about {words} words")
    print(f"  unresolved RESULT or VERIFY placeholders: {len(leftovers)}")
    if leftovers:
        for x in leftovers[:20]:
            print("   ", x)
    print(f"  author-action placeholders left on purpose: {author_actions}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
