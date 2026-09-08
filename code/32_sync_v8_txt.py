"""Apply the group-2 edits of 31_build_v8.py to the plain-text draft v8.txt.

v8.txt already carries the group-1 edits. It is kept in sync by hand rather than
regenerated from the docx, because it renders the inline equations that a docx
text extraction drops.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

build = import_module("31_build_v8")

ROOT = Path(__file__).resolve().parent.parent
TXT = ROOT / "v8.txt"

# group 1 is already in v8.txt; skip those three.
GROUP_2 = build.EDITS[3:]  # index 3 is the word-count line, then the ref #12 edits

text = TXT.read_text(encoding="utf-8")

for _anchor, old, new in GROUP_2:
    if new in text:
        print(f"already : {old[:56]}")
        continue
    assert text.count(old) == 1, f"{text.count(old)} matches for {old[:60]!r}"
    text = text.replace(old, new)
    print(f"edited  : {old[:56]}")

# the arm B insert follows the paragraph ending in "...the rest are low."
anchor, insert = build.INSERTS[0]
tail = "and a rule base in which rule c fires when input c is high and the rest are low."
if insert in text:
    print(f"already : insert after {anchor[:46]}")
else:
    assert text.count(tail) == 1, f"{text.count(tail)} matches for the arm B tail"
    text = text.replace(tail, tail + "\n" + insert)
    print(f"inserted: after {anchor[:46]}")

TXT.write_text(text, encoding="utf-8")
print(f"\nwrote {TXT.name}")
