"""Render the manuscript and the supplement as Word documents.

Markdown headings, paragraphs, pipe tables and bold spans are converted. The
EndNote temporary citations {Author, Year #N} are left exactly as they are, so
that attaching references.ris in Word and pressing Format Bibliography produces
the reference list in the journal's style.

Mathematics is converted into native Word equations rather than left as LaTeX
source. The chain is LaTeX -> MathML, with latex2mathml, then MathML -> OMML
through the MML2OMML.xsl stylesheet that ships with Office. The result is a real
equation object that Word renders and that an editor can click into, not a
picture and not a line of markup. Display equations keep their numbers, set
flush right against a tab stop the way journals set them, and $...$ spans inside
running text and table cells become inline equations.

If Office is not installed, or either library is missing, every equation falls
back to its LaTeX source in a monospace run and the build still completes; the
count of fallbacks is printed so that a silent degradation is visible.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.shared import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT

JOBS = [(ROOT / "05_Manuscript.md", ROOT / "08_Manuscript.docx"),
        (ROOT / "06_Supplementary.md", ROOT / "08_Supplementary.docx")]

# XML forbids most C0 control characters and python-docx raises instead of
# dropping them, so one stray character in a hundred thousand would fail the
# whole build. Every run is scrubbed on the way in, and the count is printed so
# that a corruption upstream is visible rather than silently swallowed.
CONTROL_CHARS = "".join(chr(c) for c in list(range(0x00, 0x09))
                        + [0x0b, 0x0c] + list(range(0x0e, 0x20)))
CONTROL_RE = re.compile("[" + re.escape(CONTROL_CHARS) + "﻿￾]")

# Where Office keeps the MathML to OMML transform. The 32-bit install path and
# the older suite layouts are tried in turn.
XSL_CANDIDATES = [
    r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.xsl",
    r"C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.xsl",
    r"C:\Program Files\Microsoft Office\Office16\MML2OMML.xsl",
    r"C:\Program Files (x86)\Microsoft Office\Office16\MML2OMML.xsl",
    r"C:\Program Files\Microsoft Office\Office15\MML2OMML.xsl",
]

_TRANSFORM = None
_CONVERT = None
MATH_READY = False
FALLBACKS = 0
CONVERTED = 0


def _init_math() -> bool:
    """Load the converter and the stylesheet once; report whether math works."""
    global _TRANSFORM, _CONVERT, MATH_READY
    if MATH_READY:
        return True
    try:
        import latex2mathml.converter as conv
        from lxml import etree
    except ImportError as e:
        print(f"  math disabled, missing library: {e}")
        return False
    xsl = next((p for p in XSL_CANDIDATES if Path(p).exists()), None)
    if xsl is None:
        print("  math disabled, MML2OMML.xsl not found in any known Office path")
        return False
    _TRANSFORM = etree.XSLT(etree.parse(xsl))
    _CONVERT = conv.convert
    MATH_READY = True
    print(f"  math enabled via {Path(xsl).parent.name}/MML2OMML.xsl")
    return True


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _style_math_runs(root) -> None:
    """Give every math run the Cambria Math font.

    MML2OMML.xsl emits the equation structure but no run properties, so Word
    draws the math in whatever font the paragraph uses. Calibri has no glyph for
    the summation sign or for the mathematical bold alphabet that \\mathbf
    produces, and the equation comes out as a row of boxes. Word itself always
    writes w:rFonts on a math run, and this puts it back.

    In the m:r schema the order is m:rPr, then w:rPr, then the run content, so
    w:rPr is inserted after any m:rPr and before m:t.
    """
    from lxml import etree
    for r in root.iter("{%s}r" % M_NS):
        if r.find("{%s}rPr" % W_NS) is not None:
            continue
        rpr = etree.SubElement(r, "{%s}rPr" % W_NS)
        fonts = etree.SubElement(rpr, "{%s}rFonts" % W_NS)
        for attr in ("ascii", "hAnsi", "cs"):
            fonts.set("{%s}%s" % (W_NS, attr), "Cambria Math")
        # Place it after m:rPr if there is one, otherwise first.
        mrpr = r.find("{%s}rPr" % M_NS)
        r.remove(rpr)
        r.insert(1 if mrpr is not None else 0, rpr)


def omml(tex: str):
    """LaTeX string to an OMML element, or None if it cannot be converted."""
    if not MATH_READY:
        return None
    from lxml import etree
    try:
        ml = _CONVERT(tex.strip())
        node = _TRANSFORM(etree.fromstring(ml.encode("utf-8"))).getroot()
        _style_math_runs(node)
        return node
    except Exception as e:  # a single bad equation must not fail the document
        print(f"    equation fell back to source: {type(e).__name__}: {tex[:60]}")
        return None


def add_math(par, tex: str) -> None:
    """Append one equation to a paragraph, or its source if conversion failed."""
    global FALLBACKS, CONVERTED
    node = omml(tex)
    if node is None:
        FALLBACKS += 1
        r = par.add_run(tex.strip())
        r.font.name = "Cambria Math"
        r.italic = True
        return
    CONVERTED += 1
    par._p.append(node)


def scrub(s: str) -> str:
    return CONTROL_RE.sub("", s)


# Bold spans and inline math, split in one pass so that neither swallows the
# other. Math is matched first because a $...$ span may contain asterisks.
SPLIT_RE = re.compile(r"(\$[^$]+\$|\*\*[^*]+\*\*)")


def add_runs(par, text: str):
    """Bold spans marked **like this**, inline math as $...$, rest plain."""
    for chunk in SPLIT_RE.split(text):
        if not chunk:
            continue
        if chunk.startswith("$") and chunk.endswith("$") and len(chunk) > 2:
            add_math(par, chunk[1:-1])
        elif chunk.startswith("**") and chunk.endswith("**"):
            r = par.add_run(chunk[2:-2])
            r.bold = True
        else:
            par.add_run(chunk)


TAG_RE = re.compile(r"\\tag\{([^}]*)\}")


def add_display_math(doc, body: str) -> None:
    """A numbered display equation: centred, with its number flush right."""
    tag = TAG_RE.search(body)
    number = tag.group(1) if tag else ""
    tex = TAG_RE.sub("", body).strip()

    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.tab_stops.add_tab_stop(Inches(3.25), WD_TAB_ALIGNMENT.CENTER)
    pf.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    p.add_run("\t")
    add_math(p, tex)
    if number:
        p.add_run("\t(" + number + ")")


def is_table_line(s: str) -> bool:
    return s.startswith("|") and s.rstrip().endswith("|")


def is_sep_line(s: str) -> bool:
    return bool(re.fullmatch(r"\|[\s:|-]+\|", s.strip()))


def convert(md_path: Path, docx_path: Path) -> None:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    raw = md_path.read_text(encoding="utf-8")
    cleaned = scrub(raw)
    if cleaned != raw:
        print(f"    scrubbed {len(raw) - len(cleaned)} control characters "
              f"from {md_path.name}")
    lines = cleaned.split("\n")

    i = 0
    while i < len(lines):
        s = lines[i].rstrip()

        if not s.strip():
            i += 1
            continue

        if s.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            p = doc.add_paragraph()
            r = p.add_run("\n".join(block))
            r.font.name = "Consolas"
            r.font.size = Pt(9)
            continue

        # Display equation. Either "$$ ... $$" on one line, or a $$ fence.
        if s.startswith("$$"):
            inner = s[2:]
            if inner.rstrip().endswith("$$"):
                add_display_math(doc, inner.rstrip()[:-2])
                i += 1
            else:
                body = [inner]
                i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    body.append(lines[i])
                    i += 1
                if i < len(lines):
                    body.append(lines[i].rstrip().rstrip("$"))
                    i += 1
                add_display_math(doc, "\n".join(body))
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            doc.add_heading(m.group(2).strip(), level=min(len(m.group(1)), 4))
            i += 1
            continue

        if is_table_line(s):
            rows = []
            while i < len(lines) and is_table_line(lines[i].rstrip()):
                if not is_sep_line(lines[i]):
                    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                    rows.append(cells)
                i += 1
            if rows:
                ncol = max(len(r) for r in rows)
                tbl = doc.add_table(rows=0, cols=ncol)
                tbl.style = "Table Grid"
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                for ri, r in enumerate(rows):
                    cells = tbl.add_row().cells
                    for ci in range(ncol):
                        txt = r[ci] if ci < len(r) else ""
                        par = cells[ci].paragraphs[0]
                        add_runs(par, txt)
                        for run in par.runs:
                            run.font.size = Pt(8)
                            if ri == 0:
                                run.bold = True
                doc.add_paragraph()
            continue

        m = re.match(r"^(\d+)\.\s+(.*)$", s)
        if m:
            p = doc.add_paragraph(style="List Number")
            add_runs(p, m.group(2))
            i += 1
            continue

        p = doc.add_paragraph()
        add_runs(p, s)
        i += 1

    doc.save(docx_path)


def main() -> int:
    _init_math()
    for md, dx in JOBS:
        if not md.exists():
            print(f"  skip {md.name}, not built yet")
            continue
        convert(md, dx)
        print(f"  {md.name} -> {dx.name} ({dx.stat().st_size / 1024:.0f} kB)")
    print(f"  equations converted: {CONVERTED}, fell back to source: {FALLBACKS}")
    if FALLBACKS:
        print("  WARNING: some equations are LaTeX source, not Word equations")
    print("DOCX_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
