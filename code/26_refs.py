"""Parse `references.ris`, resolve every in-text citation, format a bibliography.

Three jobs.

First, read the RIS library into records keyed by the EndNote record number,
which is the `#N` inside every temporary citation `{Author, Year #N}`.

Second, walk the manuscript and resolve every marker: a marker that names no
record in the library, or whose author or year disagrees with the record, is a
defect and is reported rather than quietly formatted.

Third, emit an author-date reference list, alphabetical by first author, with
the record number carried in each entry so that `{Farag, 2023 #36}` maps to its
line without ambiguity and the EndNote workflow still works if the library is
re-attached in Word.
"""
from __future__ import annotations

import re
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RIS = ROOT / "references.ris"

# {Author, Year #N}, possibly several separated by semicolons inside one brace.
CITE_BLOCK = re.compile(r"\{([^{}]*#\d+[^{}]*)\}")
ONE_CITE = re.compile(r"\s*(.+?),\s*(\d{4}[a-z]?)\s*#(\d+)\s*$")

TYPE_NAME = {"JOUR": "journal article", "CONF": "conference paper",
             "CPAPER": "conference paper", "BOOK": "book", "CHAP": "chapter",
             "RPRT": "report", "GEN": "generic", "ELEC": "web page",
             "STAND": "standard", "LEGAL": "legal rule", "JFULL": "journal"}


def parse_ris(path: Path) -> "OrderedDict[int, dict]":
    records: OrderedDict[int, dict] = OrderedDict()
    cur: dict = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.rstrip()
        m = re.match(r"^([A-Z][A-Z0-9])  - ?(.*)$", line)
        if not m:
            if line.strip() and cur:                      # continuation line
                for k in ("TI", "AB", "T2", "JO"):
                    if cur.get(k):
                        cur[k][-1] += " " + line.strip()
                        break
            continue
        tag, val = m.group(1), m.group(2).strip()
        if tag == "TY":
            cur = {"TY": val}
            continue
        if tag == "ER":
            n = cur.get("ID", [None])
            try:
                num = int(n[0])
            except (TypeError, ValueError):
                num = None
            if num is not None:
                records[num] = cur
            cur = {}
            continue
        cur.setdefault(tag, []).append(val)
    return records


def authors(rec: dict) -> list[str]:
    return rec.get("AU") or rec.get("A1") or rec.get("A2") or []


def surname(name: str) -> str:
    return name.split(",")[0].strip()


def year(rec: dict) -> str:
    for tag in ("PY", "Y1", "DA"):
        for v in rec.get(tag, []):
            m = re.search(r"(\d{4})", v)
            if m:
                return m.group(1)
    return ""


def title(rec: dict) -> str:
    for tag in ("TI", "T1"):
        if rec.get(tag):
            return rec[tag][0].rstrip(".")
    return ""


def venue(rec: dict) -> str:
    for tag in ("JO", "JF", "T2", "J2", "PB"):
        if rec.get(tag):
            return rec[tag][0].rstrip(".")
    return ""


def first(rec: dict, tag: str) -> str:
    return rec.get(tag, [""])[0]


TRUNCATED = "Author list in this record is truncated"


def author_string(rec: dict) -> str:
    """Surname, initials — with 'et al.' after six, as most journals set it.

    A record whose note says its author list was cut short by the library also
    gets 'et al.', so the entry never implies a three-author paper that has
    twenty.
    """
    names = authors(rec)
    if not names:
        return first(rec, "PB") or "Anon."
    out = []
    for n in names[:6]:
        if "," in n:
            last, rest = n.split(",", 1)
            inits = "".join(p[0] + "." for p in re.split(r"[\s.]+", rest) if p)
            out.append(f"{last.strip()} {inits}")
        else:
            out.append(n.strip())
    short = any(TRUNCATED in note for note in rec.get("N1", []))
    if len(names) > 6 or short:
        out.append("et al.")
    return ", ".join(out)


def format_entry(num: int, rec: dict) -> str:
    bits = [author_string(rec)]
    y = year(rec)
    bits.append(f"({y})" if y else "(n.d.)")
    t = title(rec)
    if t:
        bits.append(t + ".")
    v = venue(rec)
    if v:
        vol, iss, sp, ep = (first(rec, k) for k in ("VL", "IS", "SP", "EP"))
        loc = v
        if vol:
            loc += f", {vol}"
            if iss:
                loc += f"({iss})"
        if sp:
            loc += f", {sp}" + (f"-{ep}" if ep else "")
        bits.append(loc + ".")
    doi = first(rec, "DO")
    if doi:
        bits.append("https://doi.org/" + doi.replace("https://doi.org/", ""))
    elif first(rec, "UR"):
        bits.append(first(rec, "UR"))
    elif first(rec, "SN"):
        bits.append("ISBN " + first(rec, "SN"))
    return " ".join(bits).replace("..", ".")


def citations_in(text: str) -> list[tuple[str, str, int, int]]:
    """Every marker as (author, year, record number, character offset)."""
    out = []
    for blk in CITE_BLOCK.finditer(text):
        for part in blk.group(1).split(";"):
            m = ONE_CITE.match(part)
            if m:
                out.append((m.group(1).strip(), m.group(2), int(m.group(3)),
                            blk.start()))
    return out


def audit(md: Path, records) -> tuple[list[str], list[int]]:
    text = md.read_text(encoding="utf-8")
    problems, used = [], []
    for auth, yr, num, off in citations_in(text):
        if num not in records:
            problems.append(f"#{num} ({auth}, {yr}) is cited but is not in the library")
            continue
        used.append(num)
        rec = records[num]
        ry = year(rec)
        names = authors(rec)
        sn = surname(names[0]) if names else first(rec, "PB")
        key = auth.split(" and ")[0].split(" et al")[0].strip()
        if ry and yr[:4] != ry:
            problems.append(f"#{num}: marker says {yr}, library says {ry} "
                            f"({title(rec)[:60]})")
        if sn and key and key.split()[-1].lower() not in sn.lower() \
                and sn.lower() not in key.lower():
            problems.append(f"#{num}: marker names '{auth}', library first "
                            f"author is '{sn}'")
    return problems, used


def sort_key(rec: dict) -> tuple:
    names = authors(rec)
    lead = surname(names[0]) if names else (first(rec, "PB") or "zzz")
    strip = str.maketrans("áàâäãåéèêëíìîïóòôöõúùûüñçÁÀÂÄÃÅÉÈÊËÍÌÎÏÓÒÔÖÕÚÙÛÜÑÇ",
                          "aaaaaaeeeeiiiiooooouuuuncAAAAAAEEEEIIIIOOOOOUUUUNC")
    return (lead.translate(strip).lower(), year(rec), title(rec).lower())


def bibliography(records) -> list[str]:
    """Author-date entries, alphabetical, each carrying its record number."""
    ordered = sorted(records.items(), key=lambda kv: sort_key(kv[1]))
    return [f"{format_entry(n, r)}  [#{n}]" for n, r in ordered]


def write_bibliography(records, out: Path) -> None:
    lines = ["# Reference list", "",
             "Ninety-two records, alphabetical by first author. The bracketed "
             "number at the end of each entry is the EndNote record number, so "
             "an in-text marker of the form {Author, Year #N} maps to its entry "
             "without ambiguity. Attaching `references.ris` in Word and "
             "formatting the bibliography replaces this list with the same "
             "records in the journal's own style.", ""]
    lines += bibliography(records)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out.name}: {len(records)} entries")


def main() -> int:
    records = parse_ris(RIS)
    print(f"  library: {len(records)} records, numbers "
          f"{min(records)} to {max(records)}")

    if "--write" in sys.argv:
        write_bibliography(records, ROOT / "13_Reference_List.md")

    for md in (ROOT / "05_Manuscript.md", ROOT / "06_Supplementary.md"):
        if not md.exists():
            continue
        problems, used = audit(md, records)
        uniq = sorted(set(used))
        print(f"\n  {md.name}: {len(used)} markers, {len(uniq)} distinct records")
        if problems:
            print(f"  {len(problems)} problems:")
            for p in sorted(set(problems)):
                print(f"    {p}")
        else:
            print("  every marker resolves, author and year agree")

    text = (ROOT / "05_Manuscript.md").read_text(encoding="utf-8")
    cited = {n for _, _, n, _ in citations_in(text)}
    unused = [n for n in records if n not in cited]
    if unused:
        print(f"\n  {len(unused)} library records never cited in the "
              f"manuscript: {sorted(unused)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
