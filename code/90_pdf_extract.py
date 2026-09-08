"""Extract text from the seven edge-deployment reference PDFs so their reported
numbers can be read from the primary text (roadmap section 4.1)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "7 missing reference papers"
OUT = ROOT / "results" / "pdf_text"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(SRC.glob("*.pdf")):
        doc = fitz.open(pdf)
        pages = [doc[i].get_text("text") for i in range(doc.page_count)]
        text = "\n\n===PAGE BREAK===\n\n".join(pages)
        # first 400 chars give the title, which lets us name the file
        head = re.sub(r"\s+", " ", pages[0][:400]).strip()
        (OUT / f"{pdf.stem}.txt").write_text(text, encoding="utf-8")
        safe = head[:300].encode("ascii", "replace").decode("ascii")
        print(f"{pdf.name}  pages={doc.page_count}  chars={len(text)}")
        print(f"   HEAD: {safe}")
        print()
        doc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
