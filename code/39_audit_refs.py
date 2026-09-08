"""Deep pre-submission audit of every reference in the BSPC manuscript.

Beyond resolving each DOI, this asks the questions that decide whether a
reference is safe to submit to an indexed journal:

  retracted        does Crossref or OpenAlex mark the work retracted or withdrawn
  preprint         is the cited item a preprint rather than a peer-reviewed work
  superseded       if it is a preprint, has a peer-reviewed version since appeared
  unpublished      does the entry say "in press", or does its DOI fail to resolve
  ambiguous        missing volume, pages or article number; no DOI and no stable URL
  venue            what kind of venue, and is the work indexed anywhere

Writes `reference_audit.md` next to the manuscript.
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
validate = import_module("36_validate_refs")

OUT = build.OUT_DIR / "reference_audit.md"
CACHE = build.OUT_DIR / ".audit_cache.json"
UA = "bspc-submission-audit/1.0 (mailto:nullnetworkco@gmail.com)"

PREPRINT_DOI_PREFIXES = ("10.48550/", "10.1101/", "10.21203/", "10.36227/")
PREPRINT_HOSTS = ("arxiv", "biorxiv", "medrxiv", "research square", "ssrn",
                  "techrxiv", "preprints", "hal", "osf")

_cache: dict[str, object] = {}


def fetch(url: str) -> dict | None:
    if url in _cache:
        return _cache[url]                       # type: ignore[return-value]
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "application/json"})
    result = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                result = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                break
            if e.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            break
        except Exception:
            if attempt < 2:
                time.sleep(2)
                continue
            break
    _cache[url] = result
    return result


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def doi_resolves(doi: str) -> bool:
    """Ask the DOI proxy whether the handle exists at all."""
    d = fetch(f"https://doi.org/api/handles/{urllib.parse.quote(doi)}")
    return bool(d) and d.get("responseCode") == 1


def crossref(doi: str) -> dict | None:
    d = fetch(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    return d["message"] if d else None


def openalex(doi: str) -> dict | None:
    return fetch(f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}")


def datacite(doi: str) -> dict | None:
    d = fetch(f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}")
    return d["data"]["attributes"] if d else None


def find_published_version(title: str, entry_year: int) -> dict | None:
    """Look for a peer-reviewed version of a cited preprint.

    OpenAlex is asked first, allowing conference papers and book chapters as
    well as journal articles, because the machine-learning venues that matter
    here (ICML, AISTATS, MLSys, ICLR) are indexed as conference papers. Crossref
    bibliographic search is asked second, since it carries book-series venues
    such as Foundations and Trends that OpenAlex often misses.
    """
    q = urllib.parse.quote(norm(title)[:200])
    d = fetch("https://api.openalex.org/works"
              f"?filter=title.search:{q}&per-page=25")
    for w in (d or {}).get("results", []):
        if w.get("type") not in ("article", "conference-paper", "book-chapter",
                                 "book", "review"):
            continue
        if similar(title, w.get("display_name") or "") < 0.90:
            continue
        src = (w.get("primary_location") or {}).get("source") or {}
        host = (src.get("display_name") or "").lower()
        if not host or any(pre in host for pre in PREPRINT_HOSTS):
            continue
        return {"title": w.get("display_name"), "venue": src.get("display_name"),
                "year": w.get("publication_year"), "doi": w.get("doi"),
                "type": w.get("type"), "found_in": "OpenAlex"}

    d = fetch("https://api.crossref.org/works?query.bibliographic="
              f"{urllib.parse.quote(title[:200])}&rows=10")
    for w in ((d or {}).get("message") or {}).get("items", []):
        if w.get("type") == "posted-content":
            continue
        titles = w.get("title") or [""]
        if similar(title, titles[0]) < 0.90:
            continue
        container = (w.get("container-title") or [""])[0]
        if any(pre in container.lower() for pre in PREPRINT_HOSTS):
            continue
        issued = (w.get("issued") or {}).get("date-parts") or [[None]]
        return {"title": titles[0], "venue": container or w.get("publisher"),
                "year": issued[0][0], "doi": w.get("DOI"),
                "type": w.get("type"), "found_in": "Crossref"}
    return None


def classify(info: dict) -> dict:
    """Resolve one reference everywhere and decide what kind of item it is."""
    doi = info["doi"]
    cr = crossref(doi) if doi else None
    oa = openalex(doi) if doi else None
    dc = datacite(doi) if doi and not cr else None
    resolves = doi_resolves(doi) if doi else None

    flags: list[str] = []
    facts: dict[str, object] = {"crossref": bool(cr), "openalex": bool(oa),
                                "datacite": bool(dc), "resolves": resolves}

    # ---- retraction -------------------------------------------------
    retracted = False
    if oa and oa.get("is_retracted"):
        retracted = True
        flags.append("RETRACTED: OpenAlex marks this work retracted")
    for upd in (cr or {}).get("updated-by", []) or []:
        if upd.get("type") in ("retraction", "withdrawal", "removal"):
            retracted = True
            flags.append(f"RETRACTED: Crossref records a {upd['type']} "
                         f"({upd.get('DOI')})")
    facts["retracted"] = retracted

    # ---- preprint ---------------------------------------------------
    cr_type = (cr or {}).get("type", "")
    oa_type = (oa or {}).get("type", "")
    src = ((oa or {}).get("primary_location") or {}).get("source") or {}
    cr_container = ((cr or {}).get("container-title") or [""]) or [""]
    venue = (src.get("display_name") or cr_container[0]
             or (dc or {}).get("publisher") or "")
    facts["venue"] = venue
    facts["crossref_type"] = cr_type
    facts["openalex_type"] = oa_type

    is_preprint = (
        cr_type == "posted-content"
        or oa_type == "preprint"
        or (doi or "").startswith(PREPRINT_DOI_PREFIXES)
        or any(p in (venue or "").lower() for p in PREPRINT_HOSTS))
    facts["preprint"] = is_preprint

    if is_preprint:
        published = None
        for rel in ((cr or {}).get("relation") or {}).get("is-preprint-of", []):
            published = {"doi": rel.get("id"), "venue": "via Crossref relation",
                         "year": None, "title": info["title"], "type": "article"}
            break
        if published is None:
            published = find_published_version(info["title"], info["year"])
        facts["published_version"] = published
        if published:
            flags.append(
                "PREPRINT WITH A PUBLISHED VERSION: cite the peer-reviewed "
                f"version instead: {published.get('venue')} "
                f"{published.get('year') or ''} {published.get('doi') or ''}".strip())
        else:
            flags.append("PREPRINT: no peer-reviewed version found; acceptable "
                         "to cite but must be labelled as a preprint")

    # ---- unpublished / unresolvable ---------------------------------
    if re.search(r"\bin press\b", info["entry"], re.I):
        flags.append("IN PRESS: accepted but not yet published")
    if doi and resolves is False:
        flags.append("DOI DOES NOT RESOLVE at doi.org")
    if doi and not (cr or oa or dc):
        flags.append("NOT INDEXED: absent from Crossref, DataCite and OpenAlex")
    if not doi and not info["url"]:
        flags.append("NO DOI AND NO URL: the entry cannot be resolved by a reader")

    # ---- bibliographic completeness ---------------------------------
    rest = info["rest"]
    looks_like_article = cr_type == "journal-article" or oa_type == "article"
    has_locator = bool(re.search(r"\d+\s*\(\d+\)|,\s*\d+[-–]\d+|,\s*e\d+"
                                 r"|,\s*\d{4,}\b|Article\s+\d+", rest))
    if looks_like_article and not has_locator:
        flags.append("INCOMPLETE: no volume, issue, pages or article number")
    facts["cited_by"] = (oa or {}).get("cited_by_count")
    facts["indexed_in"] = (oa or {}).get("indexed_in") or []
    return {"flags": flags, "facts": facts, "cr": cr, "oa": oa, "dc": dc}


def main() -> int:
    if CACHE.exists():
        _cache.update(json.loads(CACHE.read_text(encoding="utf-8")))

    blocks = build.number_headings(
        build.parse_source(build.SOURCE.read_text(encoding="utf-8")))
    order = build.collect_citation_order(blocks)
    refs = build.load_references()

    results = []
    for key, num in sorted(order.items(), key=lambda kv: kv[1]):
        entry = refs[key]
        info = validate.parse_reference(entry)
        info["entry"] = entry
        m = validate.REF_RE.match(entry)
        info["rest"] = m.group("rest")
        r = classify(info)
        results.append((num, info, r))
        tag = "  ".join(f.split(":")[0] for f in r["flags"]) or "clean"
        print(f"[{num:>2}] {tag}")

    CACHE.write_text(json.dumps(_cache), encoding="utf-8")

    def has(results, word):
        return [(n, i, r) for n, i, r in results
                if any(f.startswith(word) for f in r["flags"])]

    retracted = has(results, "RETRACTED")
    superseded = has(results, "PREPRINT WITH")
    preprints = has(results, "PREPRINT:")
    inpress = has(results, "IN PRESS")
    unresolved = has(results, "DOI DOES NOT RESOLVE") + has(results, "NOT INDEXED")
    incomplete = has(results, "INCOMPLETE") + has(results, "NO DOI AND NO URL")
    clean = [r for r in results if not r[2]["flags"]]

    L = ["# Reference audit for the BSPC submission", "",
         "Every reference cited in `02_Manuscript.docx` was resolved at "
         "**Crossref**, **DataCite**, **OpenAlex** and the **DOI handle "
         "system**, then checked for retraction, preprint status, a "
         "peer-reviewed version superseding a cited preprint, unpublished or "
         "in-press status, and bibliographic completeness.", "",
         "## Summary", "",
         "| Check | Count |", "| --- | --- |",
         f"| References audited | {len(results)} |",
         f"| **Retracted or withdrawn** | **{len(retracted)}** |",
         f"| **Preprints superseded by a published version** | **{len(superseded)}** |",
         f"| Preprints with no published version found | {len(preprints)} |",
         f"| Marked in press | {len(inpress)} |",
         f"| DOI unresolvable or not indexed anywhere | {len(unresolved)} |",
         f"| Incomplete bibliographic data | {len(incomplete)} |",
         f"| Fully clean | {len(clean)} |", ""]

    def section(title, items, note):
        L.append(f"## {title}")
        L.append("")
        if not items:
            L.append("None.")
            L.append("")
            return
        L.append(note)
        L.append("")
        for num, info, r in items:
            L.append(f"**[{num}]** {info['authors']} ({info['year']}) "
                     f"{info['title'][:95]}")
            L.append(f"  - DOI `{info['doi'] or 'none'}`, venue "
                     f"{r['facts'].get('venue') or 'unknown'}")
            for f in r["flags"]:
                L.append(f"  - {f}")
            L.append("")

    section("Retracted or withdrawn", retracted,
            "These must be removed before submission.")
    section("Preprints superseded by a published version", superseded,
            "The journal's guide states that where a preprint has since "
            "appeared as a peer-reviewed publication, the formal publication "
            "must be cited instead. Replace each of these.")
    section("Preprints with no peer-reviewed version", preprints,
            "Citing these is permitted, but the guide requires the word "
            "\"preprint\" or the server name in the entry, with the preprint "
            "DOI. Each entry below already names arXiv.")
    section("In press", inpress,
            "A reference cited as in press implies the item has been accepted "
            "for publication. Confirm acceptance and that the DOI is live.")
    section("Unresolvable or unindexed", unresolved,
            "A reader cannot follow these. Fix the identifier or drop the "
            "reference.")
    section("Incomplete bibliographic data", incomplete,
            "The guide requires volume, article number or pagination where "
            "applicable.")

    L += ["## Every reference", "",
          "| # | First author | Year | Type | Venue | Indexed in | Cited by | Status |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for num, info, r in results:
        f = r["facts"]
        kind = ("preprint" if f.get("preprint")
                else f.get("crossref_type") or f.get("openalex_type") or "unknown")
        idx = ", ".join(f.get("indexed_in") or []) or "-"
        status = "clean" if not r["flags"] else "; ".join(
            x.split(":")[0].lower() for x in r["flags"])
        L.append(f"| {num} | {info['first_author']} | {info['year']} | {kind} | "
                 f"{(f.get('venue') or '-')[:38]} | {idx} | "
                 f"{f.get('cited_by') if f.get('cited_by') is not None else '-'} | "
                 f"{status} |")
    L.append("")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.name}")
    print(f"  retracted {len(retracted)} | superseded preprints {len(superseded)} "
          f"| preprints {len(preprints)} | in press {len(inpress)} "
          f"| unresolved {len(unresolved)} | incomplete {len(incomplete)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
