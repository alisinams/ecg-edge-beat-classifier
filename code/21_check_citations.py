"""Validate every in-text citation marker against references.ris.

EndNote matches a temporary citation on the record number, and then shows the
author and year it found. If the author or year in the manuscript disagrees with
the record, Word will silently format the record and leave the reader with a
mismatch that no spell-checker catches. This script catches it instead.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT

RIS = ROOT / "references.ris"
DOCS = [ROOT / "05_Manuscript.md", ROOT / "06_Supplementary.md"]

MARKER = re.compile(r"\{([^{}]*#\d+[^{}]*)\}")
ONE = re.compile(r"^\s*(?P<author>.+?),\s*(?P<year>\d{4}[a-z]?)\s*#(?P<id>\d+)\s*$")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def load_ris():
    text = RIS.read_text(encoding="utf-8-sig")
    recs = {}
    for block in text.split("ER  -"):
        m = re.search(r"^ID  - (\d+)\s*$", block, re.M)
        if not m:
            continue
        rid = int(m.group(1))
        au = re.findall(r"^AU  - (.+)$", block, re.M)
        py = re.search(r"^PY  - (\d{4})", block, re.M)
        ti = re.search(r"^TI  - (.+)$", block, re.M)
        recs[rid] = dict(authors=au, year=py.group(1) if py else None,
                         title=ti.group(1).strip() if ti else "")
    return recs


def surname(a: str) -> str:
    return strip_accents(a.split(",")[0].strip()).lower()


def main() -> int:
    recs = load_ris()
    problems, used = [], set()
    for doc in DOCS:
        if not doc.exists():
            print(f"  skip {doc.name}, not built")
            continue
        text = doc.read_text(encoding="utf-8")
        for m in MARKER.finditer(text):
            for part in m.group(1).split(";"):
                mm = ONE.match(part)
                if not mm:
                    problems.append((doc.name, part.strip(), "unparseable marker"))
                    continue
                rid = int(mm.group("id"))
                used.add(rid)
                rec = recs.get(rid)
                if rec is None:
                    problems.append((doc.name, part.strip(), f"record {rid} not in library"))
                    continue
                if rec["year"] and mm.group("year")[:4] != rec["year"]:
                    problems.append((doc.name, part.strip(),
                                     f"year is {rec['year']} in the library"))
                cited = strip_accents(mm.group("author")).lower()
                sn = [surname(a) for a in rec["authors"]]
                first = sn[0] if sn else ""
                ok = (first and (first in cited or cited.split()[0] in first))
                if not ok and rec["authors"]:
                    ok = any(s in cited for s in sn)
                if not ok:
                    problems.append((doc.name, part.strip(),
                                     f"first author is {rec['authors'][0] if rec['authors'] else '?'}"))

    unused = sorted(set(recs) - used)
    print(f"records in library: {len(recs)}")
    print(f"records cited:      {len(used)}")
    if unused:
        print(f"records never cited ({len(unused)}): {unused}")
    if problems:
        print(f"PROBLEMS: {len(problems)}")
        for d, p, why in problems:
            print(f"  {d}: {{{p}}} -> {why}")
        return 1
    print("CITATIONS_OK: every marker resolves and its author and year match the library")
    return 0


if __name__ == "__main__":
    sys.exit(main())
