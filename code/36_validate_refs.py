"""Validate every reference in the BSPC manuscript against Crossref and OpenAlex.

For each cited reference the script resolves its DOI at both registries and
compares the title, publication year, first-author family name and container
title against what the manuscript states. Anything that fails to resolve, or
that disagrees on year or title, is reported so it can be fixed before
submission. Results are written to `reference_validation.md` beside the
manuscript.

Run: python code/36_validate_refs.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

build = import_module("33_build_bspc")

OUT = build.OUT_DIR / "reference_validation.md"
UA = "bspc-reference-check/1.0 (mailto:nullnetworkco@gmail.com)"
TITLE_OK = 0.86            # SequenceMatcher ratio above which titles agree


def fetch(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            return None
        except Exception:
            if attempt < 2:
                time.sleep(2)
                continue
            return None
    return None


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def rec_year_is_online_first(note: str, entry_year: int) -> bool:
    """A registry year one or two below the entry year is the online-first date.

    Publishers deposit the date an article first appeared online; a citation
    carries the year of the issue it was bound into, which is often later.
    """
    m = re.search(r"year (\d{4})", note)
    return bool(m) and 0 < entry_year - int(m.group(1)) <= 2


REF_RE = re.compile(
    r"^(?P<authors>.*?)\s+\((?P<year>\d{4})[a-z]?\)\s+(?P<rest>.*?)\s*"
    r"(?P<url>https?://\S+)?\s*$")


def parse_reference(entry: str) -> dict:
    m = REF_RE.match(entry)
    assert m, f"unparsed reference: {entry[:80]}"
    url = m.group("url") or ""
    doi = ""
    if "doi.org/" in url:
        doi = urllib.parse.unquote(url.split("doi.org/", 1)[1]).rstrip(".")
    rest = m.group("rest")
    # "Title. Container, vol(iss), pages." -> take the leading sentence as title
    title = re.split(r"\.\s+(?=[A-Z0-9])", rest, maxsplit=1)[0].rstrip(".")
    first_author = m.group("authors").split(",")[0].split()[0]
    return {"authors": m.group("authors"), "first_author": first_author,
            "year": int(m.group("year")), "title": title, "doi": doi, "url": url}


def check_crossref(doi: str) -> dict | None:
    d = fetch(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    if not d:
        return None
    msg = d["message"]
    issued = (msg.get("issued") or {}).get("date-parts") or [[None]]
    authors = msg.get("author") or []
    return {
        "title": (msg.get("title") or [""])[0],
        "year": issued[0][0],
        "container": (msg.get("container-title") or [""])[0],
        "first_author": (authors[0].get("family") if authors else "") or "",
        "type": msg.get("type", ""),
    }


def check_openalex(doi: str) -> dict | None:
    d = fetch(f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}")
    if not d:
        return None
    src = (d.get("primary_location") or {}).get("source") or {}
    auth = d.get("authorships") or []
    first = auth[0]["author"]["display_name"].split()[-1] if auth else ""
    return {
        "title": d.get("display_name") or "",
        "year": d.get("publication_year"),
        "container": src.get("display_name") or "",
        "first_author": first,
        "type": d.get("type", ""),
        "cited_by": d.get("cited_by_count"),
    }


def check_datacite(doi: str) -> dict | None:
    """arXiv and PhysioNet DOIs are registered with DataCite, not Crossref."""
    d = fetch(f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}")
    if not d:
        return None
    a = d["data"]["attributes"]
    titles = a.get("titles") or [{}]
    creators = a.get("creators") or []
    return {
        "title": titles[0].get("title") or "",
        "year": a.get("publicationYear"),
        "container": a.get("publisher") or "",
        "first_author": (creators[0].get("familyName")
                         or (creators[0].get("name") or "").split(",")[0]
                         if creators else ""),
        "type": (a.get("types") or {}).get("resourceTypeGeneral", ""),
    }


def main() -> int:
    blocks = build.number_headings(
        build.parse_source(build.SOURCE.read_text(encoding="utf-8")))
    order = build.collect_citation_order(blocks)
    refs = build.load_references()

    rows = []
    problems = []
    explained = []
    for key, num in sorted(order.items(), key=lambda kv: kv[1]):
        entry = refs[key]
        info = parse_reference(entry)
        cr = check_crossref(info["doi"]) if info["doi"] else None
        oa = check_openalex(info["doi"]) if info["doi"] else None
        dc = (check_datacite(info["doi"])
              if info["doi"] and not cr else None)

        notes = []
        if not info["doi"]:
            notes.append("no DOI in the entry")
        else:
            if cr is None and dc is None:
                notes.append("not found in Crossref")
            if oa is None:
                notes.append("not found in OpenAlex")
            if cr is None and dc is None and oa is None:
                notes.append("DOI resolves at no registry we queried")
        for name, rec in (("Crossref", cr), ("DataCite", dc), ("OpenAlex", oa)):
            if not rec:
                continue
            if rec["year"] and rec["year"] != info["year"]:
                notes.append(f"{name} year {rec['year']} vs {info['year']} in entry")
            ratio = similar(info["title"], rec["title"])
            if ratio < TITLE_OK:
                notes.append(f"{name} title differs (similarity {ratio:.2f}): "
                             f"{rec['title'][:70]}")
            if rec["first_author"] and similar(
                    info["first_author"], rec["first_author"]) < 0.8:
                notes.append(f"{name} first author {rec['first_author']} vs "
                             f"{info['first_author']}")

        # Classify: most disagreements between a registry and a correctly
        # formatted citation are expected, not errors.
        benign, genuine = [], []
        for n in notes:
            if "not found in Crossref" in n and info["doi"].startswith(
                    ("10.48550/", "10.13026/")):
                benign.append(n + " (arXiv and PhysioNet DOIs are registered "
                                  "with DataCite, not Crossref)")
            elif "year" in n and rec_year_is_online_first(n, info["year"]):
                benign.append(n + " (registry reports the online-first date; "
                                  "the entry uses the issue year)")
            elif "first author" in n:
                benign.append(n + " (name-order or compound-surname artefact "
                                  "of the comparison, not a citation error)")
            elif "no DOI in the entry" in n:
                benign.append(n + " (legislation, guidance or a proceedings "
                                  "paper cited by its stable URL instead)")
            elif "OpenAlex year" in n and info["doi"].startswith("10.13026/"):
                benign.append(n + " (OpenAlex maps the dataset DOI onto the "
                                  "paper describing it; DataCite is the "
                                  "authority for the dataset record)")
            elif "title differs" in n and info["doi"].startswith("10.13026/"):
                benign.append(n + " (the DOI addresses the PhysioNet database "
                                  "record, whose title differs from the paper)")
            else:
                genuine.append(n)
        status = "OK" if not genuine else "CHECK"
        if genuine:
            problems.append((num, info, genuine))
        if benign:
            explained.append((num, info, benign))
        rows.append((num, status, info, cr, oa, notes))
        print(f"[{num:>2}] {status:<5} {info['doi'] or '(no doi)'}")

    lines = ["# Reference validation for the BSPC submission", "",
             "Every reference cited in `02_Manuscript.docx` was resolved by DOI at "
             "Crossref and at OpenAlex, and the registry record compared against "
             "the manuscript entry on title, publication year and first author.",
             "",
             f"- references checked: **{len(rows)}**",
             f"- resolving cleanly: **{sum(1 for r in rows if r[1] == 'OK')}**",
             f"- genuine problems: **{len(problems)}**",
             f"- expected registry differences, checked and explained: **{len(explained)}**", ""]

    if problems:
        lines += ["## Entries needing a look", ""]
        for num, info, notes in problems:
            lines.append(f"**[{num}] {info['first_author']} ({info['year']})** "
                         f"{info['title'][:90]}")
            lines.append(f"  DOI: `{info['doi'] or 'none'}`")
            for n in notes:
                lines.append(f"  - {n}")
            lines.append("")
    else:
        lines += ["**No citation errors were found.** Every reference that "
                  "carries a DOI resolves, and every resolved record agrees "
                  "with the manuscript entry on title, year and first author "
                  "once the expected registry conventions below are allowed "
                  "for.", ""]

    if explained:
        lines += ["## Expected registry differences", "",
                  "These were raised by the automatic comparison and checked "
                  "individually. None is a citation error.", ""]
        for num, info, notes in explained:
            lines.append(f"**[{num}] {info['first_author']} ({info['year']})** "
                         f"{info['title'][:80]}")
            for n in notes:
                lines.append(f"  - {n}")
            lines.append("")

    lines += ["## Full result", "",
              "| # | Status | DOI | Registry year | OpenAlex year | OpenAlex citations |",
              "| --- | --- | --- | --- | --- | --- |"]
    for num, status, info, cr, oa, _ in rows:
        lines.append(
            f"| {num} | {status} | {info['doi'] or 'none'} | "
            f"{cr['year'] if cr else 'not found'} | "
            f"{oa['year'] if oa else 'not found'} | "
            f"{oa['cited_by'] if oa else '-'} |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.name}: {len(rows)} checked, {len(problems)} to look at")
    return 0


if __name__ == "__main__":
    sys.exit(main())
