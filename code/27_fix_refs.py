"""Repair two transport defects in `references.ris`, in place, with a backup.

Neither defect is a judgement call and both would reach the printed reference
list of a submitted paper.

First, the string "GBD 2023 Cardiovascular Diseases Collaborators" appears as
an author on seven records. It belongs on exactly one of them, the Global
Burden of Disease study itself. On the other six it is a copy-paste artefact
that displaced the real author tail: each of those six carries three named
authors and then the collaborator group, while the library elsewhere holds
records with up to twelve authors, so the list is not being truncated by
policy. The line is removed from the six, and a note is written into the record
saying the author list is short, so that the formatter can print "et al."
instead of silently implying the paper has three authors.

Second, three venue fields carry HTML entities, "&amp;" and the doubly escaped
"&amp;amp;", from the Crossref response. The registry value is "&".

The original file is copied to `references.original.ris` before anything is
written, and the script is safe to run twice.
"""
from __future__ import annotations

import html
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RIS = ROOT / "references.ris"
BACKUP = ROOT / "references.original.ris"

STRAY = "GBD 2023 Cardiovascular Diseases Collaborators"
KEEP_ON = {1}                       # the GBD study itself, where it belongs
TRUNC_NOTE = ("Author list in this record is truncated to the first three; "
              "the published article carries more. Re-fetch by DOI before "
              "submission if the full list is required.")


def unescape(s: str) -> str:
    prev = None
    while prev != s:
        prev, s = s, html.unescape(s)
    return s


def main() -> int:
    raw = RIS.read_text(encoding="utf-8-sig")
    if not BACKUP.exists():
        shutil.copy2(RIS, BACKUP)
        print(f"  backed up references.ris -> {BACKUP.name}")

    chunks = re.split(r"(?m)^(ER  -\s*)$", raw)
    out, fixed_authors, fixed_entities = [], [], 0

    for chunk in chunks:
        if not chunk.strip() or chunk.startswith("ER  -"):
            out.append(chunk)
            continue

        rid = re.search(r"(?m)^ID  - (\d+)", chunk)
        num = int(rid.group(1)) if rid else None

        if STRAY in chunk and num not in KEEP_ON:
            chunk = re.sub(rf"(?m)^AU  - {re.escape(STRAY)}\s*\n", "", chunk)
            if TRUNC_NOTE not in chunk:
                chunk = re.sub(r"(?m)^(ID  - \d+\s*)$",
                               f"N1  - {TRUNC_NOTE}\n\\1", chunk)
            fixed_authors.append(num)

        def clean(m):
            nonlocal fixed_entities
            new = unescape(m.group(2))
            if new != m.group(2):
                fixed_entities += 1
            return m.group(1) + new

        chunk = re.sub(r"(?m)^((?:T2|JO|TI)  - )(.*)$", clean, chunk)
        out.append(chunk)

    RIS.write_text("".join(out), encoding="utf-8")
    print(f"  stray collaborator author removed from records: "
          f"{sorted(n for n in fixed_authors if n)}")
    print(f"  kept on record(s) where it belongs: {sorted(KEEP_ON)}")
    print(f"  HTML entities unescaped in {fixed_entities} fields")

    check = RIS.read_text(encoding="utf-8")
    print(f"  remaining occurrences of the stray string: "
          f"{check.count(STRAY)} (expected 1)")
    print(f"  remaining HTML entities: "
          f"{len(re.findall(r'&[a-z]+;|&#\\d+;', check))} (expected 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
