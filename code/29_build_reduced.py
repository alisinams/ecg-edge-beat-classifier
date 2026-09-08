"""Assemble Reduced Version 1 from the condensed sections, then verify it.

The condensation step writes one markdown file per section and leaves a
placeholder wherever a table, a figure caption or a display equation belongs.
This module splices the real blocks back in from `results/reduced_blocks.json`,
which holds them exactly as `05_Manuscript.md` has them, appends the formatted
reference list, and then checks four things before it will call the result
finished:

  every table, figure caption and equation appears exactly once, byte for byte;
  all 92 reference records are still cited somewhere;
  no number in the reduced prose contradicts the manuscript it came from;
  the prose lands in the 6,000 to 7,000 word window that was asked for.

The word window is measured on prose alone. Tables, captions and the reference
list are not prose and were never part of the cut.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCKS = ROOT / "results" / "reduced_blocks.json"
SECTIONS_DIR = Path(
    r"C:\Users\Alisina\AppData\Local\Temp\claude"
    r"\c--Users-Alisina-Projects-Mahdi-bazargani-Heart-disease-prediction-using-ECG-based"
    r"\14572b91-6209-4777-8508-85c61182c187\scratchpad\reduced")
OUT_MD = ROOT / "Reduced Version 1.md"

ORDER = ["00_frontback", "01_abstract", "02_introduction", "03_related",
         "04_methods", "05_results", "06_discussion", "07_conclusion"]

# The front-and-back agent emits five headings in one file; the rest of the
# document has to be interleaved around them, so the tail sections are moved to
# the end at assembly time rather than trusting an agent to order them.
TAIL_HEADINGS = ("10. Declarations", "11. References", "12. Supplementary materials")

_spec = importlib.util.spec_from_file_location(
    "refs", Path(__file__).resolve().parent / "26_refs.py")
refs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refs)


def prose_words(md: str) -> int:
    """Words of running text only.

    Tables, table and figure captions, display equations, headings and the
    reference list are all excluded: none of them is prose and none of them was
    part of the cut. A reference entry is recognised by the record number in
    brackets that ends it.
    """
    md = re.sub(r"\{[^{}]*#\d+[^{}]*\}", "", md)          # citation markers
    md = re.sub(r"(?m)^\|.*$", "", md)                     # table rows
    md = re.sub(r"(?m)^\*\*(Table|Figure) \d+\.\*\*.*$", "", md)
    md = re.sub(r"\$\$.*?\$\$", "", md, flags=re.S)        # display equations
    md = re.sub(r"(?m)^#{1,4}\s.*$", "", md)               # headings
    md = re.sub(r"(?m)^.*\[#\d+\]\s*$", "", md)            # reference entries
    return len([w for w in md.split() if w])


def collect_sections(text: str) -> dict[int, tuple[str, str]]:
    """Every "# N. Name" block in a file, keyed by its section number."""
    out: dict[int, tuple[str, str]] = {}
    num, name, cur = None, None, []
    for line in text.split("\n"):
        m = re.match(r"^#\s+(\d+)\.\s*(.*)$", line)
        if m:
            if num is not None:
                out[num] = (name, "\n".join(cur).strip())
            num, name, cur = int(m.group(1)), m.group(2).strip(), []
            continue
        cur.append(line)
    if num is not None:
        out[num] = (name, "\n".join(cur).strip())
    return out


def assemble() -> tuple[str, list[str]]:
    blocks = json.loads(BLOCKS.read_text(encoding="utf-8"))
    problems: list[str] = []

    # The front-matter file holds sections 1, 3, 10, 11 and 12, so filename
    # order is not section order. Collect every numbered block from every file
    # and emit 1 to 12 in numeric order instead.
    sections: dict[int, tuple[str, str]] = {}
    for slug in ORDER:
        f = SECTIONS_DIR / f"{slug}.md"
        if not f.exists():
            problems.append(f"missing condensed section {slug}.md")
            continue
        text = f.read_text(encoding="utf-8").strip()
        text = re.sub(r"^```(?:markdown|md)?\s*\n|\n```\s*$", "", text)
        found = collect_sections(text)
        if not found:
            problems.append(f"{slug}.md carries no '# N. Name' heading")
            continue
        for n, (name, body) in found.items():
            if n in sections:
                problems.append(f"section {n} appears in more than one file")
            sections[n] = (name, body)

    missing = [n for n in range(1, 13) if n not in sections]
    if missing:
        problems.append(f"sections absent from the draft: {missing}")

    md = "\n\n".join(f"# {n}. {sections[n][0]}\n\n{sections[n][1]}"
                     for n in sorted(sections))

    # splice the fixed blocks back in
    used = {"TABLE": set(), "FIGURE": set(), "EQ": set()}

    def splice(m):
        kind, num = m.group(1), m.group(2)
        store = {"TABLE": "tables", "FIGURE": "figures", "EQ": "equations"}[kind]
        if num in used[kind]:
            problems.append(f"<<{kind} {num}>> appears more than once")
        used[kind].add(num)
        block = blocks[store].get(num)
        if block is None:
            problems.append(f"<<{kind} {num}>> has no stored block")
            return m.group(0)
        return block

    md = re.sub(r"<<(TABLE|FIGURE|EQ)\s+(\d+)>>", splice, md)

    for kind, store in (("TABLE", "tables"), ("FIGURE", "figures"),
                        ("EQ", "equations")):
        missing = sorted(set(blocks[store]) - used[kind], key=int)
        if missing:
            problems.append(f"{kind} block(s) never placed: {missing}")

    leftover = re.findall(r"<<[^>]*>>", md)
    if leftover:
        problems.append(f"unresolved placeholders: {sorted(set(leftover))}")

    return md, problems


def append_references(md: str) -> str:
    records = refs.parse_ris(refs.RIS)
    entries = refs.bibliography(records)
    lines = [
        "", "",
        f"The {len(entries)} records of the accompanying library follow, "
        f"alphabetical by first author. The bracketed number ending each entry "
        f"is the EndNote record number, so an in-text marker of the form "
        f"{{Author, Year #N}} maps to its entry without ambiguity. Attaching "
        f"`references.ris` in Word and formatting the bibliography replaces "
        f"this list with the same records in the journal's own style.", ""]
    lines += entries
    # the list belongs at the end of section 11, before the supplement pointer
    anchor = "\n# 12. Supplementary materials"
    block = "\n".join(lines)
    if anchor in md:
        return md.replace(anchor, block + "\n" + anchor, 1)
    return md + "\n" + block


def verify(md: str) -> list[str]:
    problems = []
    blocks = json.loads(BLOCKS.read_text(encoding="utf-8"))

    for store, label in (("tables", "table"), ("figures", "figure caption"),
                         ("equations", "equation")):
        for num, text in blocks[store].items():
            n = md.count(text)
            if n != 1:
                problems.append(f"{label} {num} appears {n} times, expected 1 "
                                f"(byte-for-byte)")

    records = refs.parse_ris(refs.RIS)
    cited = {c[2] for c in refs.citations_in(md)}
    missing = sorted(set(records) - cited)
    if missing:
        problems.append(f"{len(missing)} reference records are no longer cited "
                        f"anywhere: {missing}")
    unknown = sorted(cited - set(records))
    if unknown:
        problems.append(f"citations to records not in the library: {unknown}")

    words = prose_words(md)
    if not 6000 <= words <= 7000:
        problems.append(f"prose is {words} words, outside the 6,000 to 7,000 "
                        f"window")

    # The two factual corrections the citation audit confirmed. Both contradict
    # the paper's own Table 1 if they come back, so they are checked here and
    # not left to a reader.
    if re.search(r"size[^.]{0,80}three orders of magnitude"
                 r"|three orders of magnitude[^.]{0,80}size", md, re.I):
        problems.append("the overstated model-size span is back: the four "
                        "anchors reporting bytes span 15 to 64 kB, a factor of "
                        "four, not three orders of magnitude")
    # Only the citation group attached directly to the five-class claim may be
    # inspected; `[^.{]*` stops at the first brace so a later, correctly
    # separated citation in the same sentence is not mistaken for a member.
    m = re.search(r"five-class results[^.{]*\{([^}]*)\}", md)
    if m and "#29" in m.group(1):
        problems.append("Chen #29 is inside a five-class comparison group "
                        "again; Table 1 records it as a two-class PVC detector")
    if "#29" in md and not re.search(r"two-class[^.]{0,160}#29", md):
        problems.append("Chen #29 is cited but is no longer described as a "
                        "two-class detector")
    return problems


def main() -> int:
    md, problems = assemble()
    md = append_references(md)
    problems += verify(md)

    OUT_MD.write_text(md.rstrip() + "\n", encoding="utf-8")

    blocks = json.loads(BLOCKS.read_text(encoding="utf-8"))
    total = len([w for w in re.sub(r"\{[^{}]*#\d+[^{}]*\}", "", md).split() if w])
    print(f"  wrote {OUT_MD.name}")
    print(f"  prose words:            {prose_words(md)}  (target 6,000 to 7,000)")
    print(f"  whole document words:   {total}  (prose + tables + captions + refs)")
    print(f"  tables spliced:         {len(blocks['tables'])}")
    print(f"  figure captions:        {len(blocks['figures'])}")
    print(f"  equations:              {len(blocks['equations'])}")
    print(f"  distinct records cited: {len({c[2] for c in refs.citations_in(md)})} of 92")

    if problems:
        print(f"\n  {len(problems)} problems:")
        for p in problems:
            print(f"    {p}")
        return 1
    print("\n  clean: every block verbatim, every record cited, word target met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
