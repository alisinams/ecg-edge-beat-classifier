"""Lift the blocks of the manuscript that must survive a cut untouched.

The reduced version keeps every table and every figure caption exactly as the
full manuscript has them. The safe way to guarantee that is to never let a
writing step retype them: this module pulls each block out of
`05_Manuscript.md` verbatim, keyed by its number, and the assembly step splices
the same strings back in. Anything that differs afterwards is a bug, and
`verify()` says so.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "05_Manuscript.md"
OUT = ROOT / "results" / "reduced_blocks.json"


def extract() -> dict:
    lines = SRC.read_text(encoding="utf-8").split("\n")
    tables: dict[str, str] = {}
    figures: dict[str, str] = {}
    equations: dict[str, str] = {}

    i = 0
    while i < len(lines):
        s = lines[i]

        m = re.match(r"^\*\*Table (\d+)\.\*\*", s.strip())
        if m:
            # Take the raw slice from the caption to the last pipe row, blank
            # lines included, so the stored block is the source byte for byte.
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("|"):
                j += 1
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            tables[m.group(1)] = "\n".join(lines[i:j]).rstrip()
            i = j
            continue

        m = re.match(r"^\*\*Figure (\d+)\.\*\*", s.strip())
        if m:
            figures[m.group(1)] = s.rstrip()
            i += 1
            continue

        if s.strip().startswith("$$"):
            block = [s]
            if not s.strip().endswith("$$") or s.strip() == "$$":
                j = i + 1
                while j < len(lines) and not lines[j].strip().endswith("$$"):
                    block.append(lines[j])
                    j += 1
                if j < len(lines):
                    block.append(lines[j])
                i = j
            body = "\n".join(block)
            tag = re.search(r"\\tag\{(\d+)\}", body)
            equations[tag.group(1) if tag else str(len(equations) + 1)] = body
        i += 1

    return {"tables": tables, "figures": figures, "equations": equations}


def verify(blocks: dict) -> list[str]:
    src = SRC.read_text(encoding="utf-8")
    bad = []
    for kind in ("tables", "figures"):
        for num, text in blocks[kind].items():
            if text not in src:
                bad.append(f"{kind[:-1]} {num} does not match the manuscript "
                           f"byte for byte")
    return bad


def main() -> int:
    blocks = extract()
    bad = verify(blocks)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(blocks, ensure_ascii=False, indent=1),
                   encoding="utf-8")

    def wc(s: str) -> int:
        s = re.sub(r"\{[^{}]*#\d+[^{}]*\}", "", s)
        return len([w for w in s.split() if w])

    tw = sum(wc(v) for v in blocks["tables"].values())
    fw = sum(wc(v) for v in blocks["figures"].values())
    print(f"  tables:    {len(blocks['tables'])} "
          f"({', '.join(sorted(blocks['tables'], key=int))}), {tw} words")
    print(f"  figures:   {len(blocks['figures'])} captions "
          f"({', '.join(sorted(blocks['figures'], key=int))}), {fw} words")
    print(f"  equations: {len(blocks['equations'])} display blocks "
          f"({', '.join(sorted(blocks['equations'], key=int))})")
    print(f"  {'PROBLEMS: ' + '; '.join(bad) if bad else 'every block matches the manuscript byte for byte'}")
    print(f"  wrote {OUT.relative_to(ROOT)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
